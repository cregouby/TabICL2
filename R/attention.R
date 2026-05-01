#' @importFrom torch nn_module nn_linear nn_dropout nn_init_zeros_
#' @importFrom torch torch_tensor torch_arange torch_randn torch_zeros
#' @importFrom torch torch_cat torch_stack torch_transpose
#' @importFrom torch torch_reshape
#' @importFrom torch torch_min torch_max torch_abs torch_equal
#' @importFrom torch nnf_linear torch_scaled_dot_product_attention
#' @importFrom torch torch_float torch_bfloat16 torch_float16 torch_long torch_int32
NULL


#' Global flag for Flash Attention 3 usage
#'
#' @keywords internal
.flash_attn3_enabled <- TRUE


#' Temporarily enable or disable Flash Attention 3
#'
#' This function provides a simple save/restore pattern to control whether
#' Flash Attention 3 is used during attention computation.
#'
#' @param enabled Logical. Whether to enable Flash Attention 3.
#' @return A function that, when called, restores the previous setting.
#'
#' @examples
#' \dontrun{
#' # Temporarily disable Flash Attention 3
#' restore <- flash_attn3_toggle(FALSE)
#' # ... run inference ...
#' restore()  # Restore previous setting
#'
#' # Or use with on.exit() for automatic restoration
#' old <- flash_attn3_toggle(FALSE)
#' on.exit(old(), add = TRUE)
#' # ... code that uses attention ...
#' }
#'
#' @export
flash_attn3_toggle <- function(enabled) {
  old <- .flash_attn3_enabled
  .flash_attn3_enabled <<- enabled

  # Return a function to restore the old value
  function() {
    .flash_attn3_enabled <<- old
    invisible(old)
  }
}


#' Check if Flash Attention 3 is available and enabled
#'
#' @return Logical. `TRUE` if Flash Attention 3 can be used.
#' @keywords internal
.is_flash_attn3_available <- function() {
  # Check if flash-attn package is installed and loaded
  # In R, we check for the presence of the underlying C++ binding
  # This is a placeholder; actual implementation depends on R binding availability
  has_pkg <- requireNamespace("flashattn", quietly = TRUE)
  has_pkg && .flash_attn3_enabled
}


#' Apply scaled dot-product attention with flattened batch dimensions
#'
#' This function handles arbitrary batch dimensions by flattening them before
#' applying PyTorch's `scaled_dot_product_attention` and then reshaping the
#' output back to the original shape. This flattening is necessary to properly
#' trigger Flash Attention.
#'
#' @param q Query tensor of shape `(.., nh, tgt_len, hs)` where:
#'   - `..` represents arbitrary batch dimensions
#'   - `nh` is the number of attention heads
#'   - `tgt_len` is the target sequence length
#'   - `hs` is the head size (embedding dimension per head)
#' @param k Key tensor of shape `(.., nh, src_len, hs)` with matching batch dimensions.
#' @param v Value tensor of shape `(.., nh, src_len, hs)` with matching batch dimensions.
#' @param attn_mask Optional attention mask of shape `(.., nh, tgt_len, src_len)`.
#' @param dropout_p Numeric. Dropout probability applied to attention weights
#'   (default: `0.0`).
#' @param ssmax_layer Optional `nn_module`. If provided, applies scalable softmax
#'   (SSMax) scaling to queries before attention computation.
#'
#' @return Attention output tensor of shape `(.., nh, tgt_len, hs)` preserving
#'   the original batch dimensions of the input.
#'
#' @keywords internal
sdpa_with_flattened_batch <- function(q, k, v, attn_mask = NULL,
                                       dropout_p = 0, ssmax_layer = NULL) {
  # Store original shape for restoration
  q_shape <- q$shape

  # Flatten batch dimensions: (.., nh, tgt_len, hs) -> (-1, nh, tgt_len, hs)
  # R: compute product of all but last 3 dimensions
  batch_prod <- if (q$ndim > 3L) prod(q_shape[seq_len(q$ndim - 3L)]) else 1L

  q <- q$view(c(batch_prod, q_shape[(q$ndim-2L):q$ndim]))
  k <- k$view(c(batch_prod, k$shape[(k$ndim-2L):k$ndim]))
  v <- v$view(c(batch_prod, v$shape[(v$ndim-2L):v$ndim]))

  if (!is.null(attn_mask)) {
    attn_mask <- attn_mask$view(c(batch_prod, attn_mask$shape[(attn_mask$ndim-3L):attn_mask$ndim]))
  }

  # Apply SSMax scaling if provided
  if (!is.null(ssmax_layer)) {
    src_len <- k$size(-2L)
    q <- ssmax_layer(q, src_len)
  }

  # FlashAttention 3 path (conditional on availability and settings)
  if (.is_flash_attn3_available() && q$is_cuda && is.null(attn_mask) && dropout_p == 0.0) {
    # FlashAttention only supports fp16, bf16, and fp8_e4m3
    # Convert to bf16 if needed, then convert back to original dtype
    orig_dtype <- q$dtype
    if (!orig_dtype %in% list(torch_float16(), torch_bfloat16())) {
      fa_dtype <- torch_float16()
    } else {
      fa_dtype <- orig_dtype
    }

    # Extract dimensions after flattening
    flat_bs <- q$size(1L)
    nheads <- q$size(2L)
    seqlen_q <- q$size(3L)
    headdim <- q$size(4L)
    seqlen_k <- k$size(3L)

    # Reshape for FlashAttention varlen format: (flat_bs*seqlen, nheads, headdim)
    # Transpose: (flat_bs, nheads, seqlen, headdim) -> (flat_bs, seqlen, nheads, headdim)
    q_fa <- q$transpose(2L, 3L)$reshape(c(flat_bs * seqlen_q, nheads, headdim))$contiguous()$to(dtype = fa_dtype)
    k_fa <- k$transpose(2L, 3L)$reshape(c(flat_bs * seqlen_k, nheads, headdim))$contiguous()$to(dtype = fa_dtype)
    v_fa <- v$transpose(2L, 3L)$reshape(c(flat_bs * seqlen_k, nheads, headdim))$contiguous()$to(dtype = fa_dtype)

    # Cumulative sequence lengths for varlen attention
    # R: arange with step, then cumsum-style indexing
    cu_seqlens_q <- torch_arange(0L, flat_bs * seqlen_q, seqlen_q,
                                  dtype = torch_int32(), device = q$device)
    cu_seqlens_k <- torch_arange(0L, flat_bs * seqlen_k, seqlen_k,
                                  dtype = torch_int32(), device = q$device)

    # Call FlashAttention 3 (placeholder: actual binding depends on R package)
    # This assumes a hypothetical R binding `flash_attn_varlen_func`
    if (exists("flash_attn_varlen_func", mode = "function")) {
      out <- flash_attn_varlen_func(
        q_fa, k_fa, v_fa,
        cu_seqlens_q, cu_seqlens_k,
        seqlen_q, seqlen_k
      )
    } else {
      # Fallback: warn and use standard SDPA
      cli_warn("FlashAttention 3 requested but not available; falling back to standard SDPA")
      out <- torch_scaled_dot_product_attention(
        q, k, v, attn_mask = attn_mask, dropout_p = dropout_p
      )
      # Restore original dtype and shape
      return(out$view(q_shape))
    }

    # Reshape output back: (flat_bs*seqlen_q, nheads, headdim) -> (flat_bs, seqlen_q, nheads, headdim)
    out <- out$view(c(flat_bs, seqlen_q, nheads, headdim))$transpose(2L, 3L)$to(dtype = orig_dtype)
  } else {
    # Standard scaled dot-product attention
    out <- torch_scaled_dot_product_attention(
      q, k, v, attn_mask = attn_mask, dropout_p = dropout_p
    )
  }

  # Restore original batch shape
  out$view(q_shape)
}


#' Multi-head attention with support for rotary position embeddings
#'
#' @param query Query tensor of shape `(.., tgt_len, embed_dim)`.
#' @param num_heads Integer. Number of attention heads.
#' @param in_proj_weight Tensor. Combined weight matrix for Q, K, V input projections.
#' @param in_proj_bias Tensor or NULL. Combined bias vector for input projections.
#' @param dropout_p Numeric. Dropout probability applied to attention weights.
#' @param out_proj_weight Tensor. Output projection weight matrix.
#' @param out_proj_bias Tensor or NULL. Output projection bias vector.
#' @param key Optional key tensor of shape `(.., src_len, embed_dim)`.
#'   Required when `cached_kv` is `NULL`.
#' @param value Optional value tensor of shape `(.., src_len, embed_dim)`.
#'   Required when `cached_kv` is `NULL`.
#' @param cached_kv Optional `KVCacheEntry` for pre-computed key/value projections.
#'   When provided:
#'   - `key` and `value` parameters are ignored
#'   - Only query projection is computed
#'   - `cached_kv$key` shape: `(.., num_heads, src_len, head_dim)`
#'   - `cached_kv$value` shape: `(.., num_heads, src_len, head_dim)`
#'   - RoPE is applied only to queries (keys should already have RoPE applied)
#' @param training Logical. Whether the model is in training mode (affects dropout)
#'   (default: `TRUE`).
#' @param key_padding_mask Optional mask of shape `(.., src_len)` that identifies
#'   padding elements in the key sequence:
#'   - Binary masks: `TRUE` values indicate positions to ignore
#'   - Float masks: values are directly added to attention scores
#' @param attn_mask Optional attention mask of shape `(tgt_len, src_len)` or
#'   `(.., num_heads, tgt_len, src_len)`.
#' @param rope Optional `RotaryEmbedding` for rotary positional encoding.
#' @param ssmax_layer Optional `nn_module`. If provided, applies scalable softmax
#'   (SSMax) scaling to queries before attention computation.
#' @param need_kv Logical. If `TRUE` and `cached_kv` is `NULL`, also returns
#'   the computed K and V projections along with the attention output
#'   (default: `FALSE`).
#'
#' @return If `need_kv` is `FALSE` or `cached_kv` is provided:
#'   Attention output tensor of shape `(.., tgt_len, embed_dim)`.
#'   If `need_kv` is `TRUE` and `cached_kv` is `NULL`:
#'   List of `(attn_output, k, v)` where:
#'   - `attn_output`: shape `(.., tgt_len, embed_dim)`
#'   - `k`: shape `(.., num_heads, src_len, head_dim)`
#'   - `v`: shape `(.., num_heads, src_len, head_dim)`
#'
#' @keywords internal
multi_head_attention_forward <- function(
  query,
  num_heads,
  in_proj_weight,
  in_proj_bias,
  dropout_p,
  out_proj_weight,
  out_proj_bias,
  key = NULL,
  value = NULL,
  cached_kv = NULL,
  training = TRUE,
  key_padding_mask = NULL,
  attn_mask = NULL,
  rope = NULL,
  ssmax_layer = NULL,
  need_kv = FALSE
) {
  # Extract shape information, supporting arbitrary batch dimensions
  query_shape <- query$shape
  ndim <- query$ndim

  # R: extract batch_shape as all but last 2 dimensions
  if (ndim > 2L) {
    batch_shape <- query_shape[seq_len(ndim - 2L)]
  } else {
    batch_shape <- integer(0L)
  }

  tgt_len <- query_shape[ndim - 1]
  embed_dim <- query_shape[ndim]

  head_dim <- embed_dim %/% num_heads
  if (head_dim * num_heads != embed_dim) {
    value_error("embed_dim {embed_dim} not divisible by num_heads {num_heads}")
  }

  if (is.null(cached_kv)) {
    # Standard: project Q, K, V jointly
    if (is.null(key) || is.null(value)) {
      value_error("key and value must be provided when cached_kv is NULL")
    }

    key_shape <- key$shape
    src_len <- key_shape[key$ndim - 1]

    if (!identical(key$shape, value$shape)) {
      value_error("key shape {key$shape} does not match value shape {value$shape}")
    }

    # R equivalent of F._in_projection_packed: manual projection
    # Split in_proj_weight and in_proj_bias into Q, K, V portions
    q_weight <- in_proj_weight[seq_len(embed_dim), ..]
    k_weight <- in_proj_weight[(embed_dim + 1L):(2L * embed_dim), ..]
    v_weight <- in_proj_weight[(2L * embed_dim + 1L):(3L * embed_dim), ..]

    if (!is.null(in_proj_bias)) {
      q_bias <- in_proj_bias[seq_len(embed_dim)]
      k_bias <- in_proj_bias[(embed_dim + 1L):(2L * embed_dim)]
      v_bias <- in_proj_bias[(2L * embed_dim + 1L):(3L * embed_dim)]
    } else {
      q_bias <- NULL
      k_bias <- NULL
      v_bias <- NULL
    }

    # Apply linear projections
    q <- nnf_linear(query, q_weight, q_bias)
    k <- nnf_linear(key, k_weight, k_bias)
    v <- nnf_linear(value, v_weight, v_bias)

    # Reshape and transpose: (.., seq_len, embed_dim) -> (.., num_heads, seq_len, head_dim)
    # New shape: c(batch_shape, seq_len, num_heads, head_dim)
    new_shape_q <- c(batch_shape, tgt_len, num_heads, head_dim)
    new_shape_kv <- c(batch_shape, src_len, num_heads, head_dim)

    q <- q$view(new_shape_q)$transpose(length(new_shape_q) - 2L, length(new_shape_q) - 1L)
    k <- k$view(new_shape_kv)$transpose(length(new_shape_kv) - 2L, length(new_shape_kv) - 1L)
    v <- v$view(new_shape_kv)$transpose(length(new_shape_kv) - 2L, length(new_shape_kv) - 1L)

    # Apply RoPE if provided
    if (!is.null(rope)) {
      q <- rope$rotate_queries_or_keys(q)
      k <- rope$rotate_queries_or_keys(k)
    }
  } else {
    # Use cached K/V, project Q only
    k <- cached_kv$key
    v <- cached_kv$value
    src_len <- k$size(-2L)

    # Project query only: use first embed_dim rows of in_proj_weight/bias
    q_weight <- in_proj_weight[seq_len(embed_dim), ..]
    q_bias <- if (!is.null(in_proj_bias)) in_proj_bias[seq_len(embed_dim)] else NULL

    q <- nnf_linear(query, q_weight, q_bias)

    # Reshape and transpose query
    new_shape_q <- c(batch_shape, tgt_len, num_heads, head_dim)
    q <- q$view(new_shape_q)$transpose(length(new_shape_q) - 2L, length(new_shape_q) - 1L)

    # Apply RoPE to query only (keys should already have RoPE applied during cache store)
    if (!is.null(rope)) {
      q <- rope$rotate_queries_or_keys(q)
    }
  }

  # Disable dropout during evaluation
  if (!training) {
    dropout_p <- 0
  }

  # Process attention mask
  correct_2d_shape <- c(tgt_len, src_len)
  correct_nd_shape <- c(batch_shape, num_heads, tgt_len, src_len)

  if (!is.null(attn_mask)) {
    mask_ndim <- length(attn_mask$shape)

    if (mask_ndim == 2L) {
      if (!identical(attn_mask$shape, correct_2d_shape)) {
        value_error("2D attn_mask should have shape {correct_2d_shape}, but got {attn_mask$shape}")
      }
      # Expand to full batch shape: (tgt_len, src_len) -> (.., num_heads, tgt_len, src_len)
      attn_mask <- attn_mask$unsqueeze(1L)$unsqueeze(1L)  # Add head and batch dims
      for (i in seq_along(batch_shape)) {
        attn_mask <- attn_mask$unsqueeze(1L)  # Prepend batch dims
      }
      attn_mask <- attn_mask$expand(correct_nd_shape)
    } else if (mask_ndim == length(correct_nd_shape)) {
      if (!identical(attn_mask$shape, correct_nd_shape)) {
        value_error("{length(correct_nd_shape)}D attn_mask should have shape {correct_nd_shape}, but got {attn_mask$shape}")
      }
    } else {
      value_error("attn_mask must be 2D or {length(correct_nd_shape)}D, got {mask_ndim}D")
    }
  }

  # Process key padding mask
  if (!is.null(key_padding_mask)) {
    expected_kpm_shape <- c(batch_shape, src_len)
    if (!identical(key_padding_mask$shape, expected_kpm_shape)) {
      value_error("key_padding_mask should have shape {expected_kpm_shape}, but got {key_padding_mask$shape}")
    }

    # Expand to attention mask shape: (.., src_len) -> (.., 1, 1, src_len) -> (.., num_heads, tgt_len, src_len)
    kpm_expanded <- key_padding_mask
    for (i in seq_len(length(batch_shape) + 1L)) {
      kpm_expanded <- kpm_expanded$unsqueeze(-2L)  # Add dims before src_len
    }
    kpm_expanded <- kpm_expanded$expand(c(batch_shape, num_heads, tgt_len, src_len))

    if (is.null(attn_mask)) {
      attn_mask <- kpm_expanded
    } else {
      # Combine masks: additive for float masks, logical OR would need conversion
      attn_mask <- attn_mask + kpm_expanded
    }
  }

  # Apply scaled dot-product attention with flattened batch
  attn_output <- sdpa_with_flattened_batch(
    q, k, v, attn_mask, dropout_p, ssmax_layer = ssmax_layer
  )  # (.., nh, tgt_len, hs)

  # Reshape and project output: (.., nh, tgt_len, hs) -> (.., tgt_len, embed_dim)
  # Transpose back: move nh dim to after seq_len, then flatten nh*hs -> embed_dim
  attn_output <- attn_output$transpose(length(attn_output$shape) - 2L, length(attn_output$shape) - 1L)
  attn_output <- attn_output$contiguous()$view(c(batch_shape, tgt_len, embed_dim))

  # Final output projection
  attn_output <- nnf_linear(attn_output, out_proj_weight, out_proj_bias)

  # Return K/V if requested and not using cache
  if (need_kv && is.null(cached_kv)) {
    return(list(attn_output, k, v))
  }

  attn_output
}
