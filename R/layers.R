#' @importFrom torch nn_module nn_linear nn_layer_norm nn_identity nn_init_zeros_ nn_init_trunc_normal_
#' @importFrom torch nn_parameter torch_empty torch_randn torch_full_like torch_zeros_like
#' @importFrom torch torch_cat torch_stack torch_arange torch_zeros torch_ones
#' @importFrom torch nn_multihead_attention nnf_one_hot nnf_linear nnf_pad
#' @importFrom torch torch_isnan torch_isinf
#' @importFrom torch torch_long torch_float torch_rand torch_randint
NULL


# #' Node in the hierarchical classification tree
# #'
# #' reuse data.tree Node data model
# #'
# #' @keywords internal
# ClassNode <- data.tree::Node

#' One-hot encoding combined with linear projection
#'
#' Combines one-hot encoding and linear projection in a single efficient operation
#' to convert categorical indices to embeddings.
#'
#' @param num_classes Integer. Number of distinct categories for one-hot encoding.
#' @param embed_dim Integer. Output embedding dimension.
#'
#' @return A `nn_module` that transforms integer indices to dense embeddings.
#'
#' @examples
#' \dontrun{
#' encoder <- one_hot_and_linear(num_classes = 10L, embed_dim = 32L)
#' indices <- torch_randint(1L, 11L, c(2L, 20L))  # [batch, seq_len]
#' embeddings <- encoder(indices)  # [batch, seq_len, embed_dim]
#' }
#'
#' @export
one_hot_and_linear <- nn_module(
  "OneHotAndLinear",
  inherit = nn_linear,
  initialize = function(num_classes, embed_dim) {
    # Parent nn_linear: maps one-hot (num_classes) to embed_dim
    super$initialize(num_classes, embed_dim)
    self$num_classes <- num_classes
    self$embed_dim <- embed_dim
  },

  # Transform integer indices to dense embeddings
  #
  # @param src Integer tensor of shape `(batch_size, sequence_length)` containing
  #   category indices.
  # @return Embedded representation of shape `(batch_size, sequence_length, embed_dim)`.
  forward = function(src) {
    # Convert indices to one-hot vectors and apply linear projection
    # R: one_hot expects 0-based indices for consistency with Python logic
    one_hot <- nnf_one_hot(src$to(dtype = torch_long()), self$num_classes)
    # Cast to src dtype and apply linear projection
    nnf_linear(one_hot$to(dtype = src$dtype), self$weight, self$bias)
  }
)


#' Linear layer that handles inputs flagged with a skip value
#'
#' First applies the linear transformation to all inputs, then replaces outputs
#' for inputs where all values equal `skip_value` with the `skip_value`.
#'
#' @param in_features Integer. Size of each input sample.
#' @param out_features Integer. Size of each output sample.
#' @param bias Logical. If `FALSE`, the layer will not learn an additive bias
#'   (default: `TRUE`).
#' @param skip_value Numeric. Value used to mark inputs that should be skipped
#'   (default: `-100.0`).
#'
#' @return A `nn_module` that applies linear transformation with skip handling.
#'
#' @examples
#' \dontrun{
#' layer <- skippable_linear(in_features = 10L, out_features = 20L)
#' x <- torch_randn(32L, 10L)
#' x[1:5, ] <- -100.0  # Mark rows to skip
#' out <- layer(x)  # Skipped rows remain -100.0 in output
#' }
#'
#' @export
skippable_linear <- nn_module(
  "SkippableLinear",
  inherit = nn_linear,
  initialize = function(in_features, out_features, bias = TRUE, skip_value = -100.0) {
    super$initialize(in_features, out_features, bias = bias)
    self$skip_value <- skip_value
  },

  # Forward pass that handles inputs flagged with `skip_value`
  #
  # @param src Input tensor of shape `(.., in_features)`.
  # @return Output tensor of shape `(.., out_features)` where rows corresponding
  #   to skipped inputs are filled with `skip_value`.
  forward = function(src) {
    out <- nnf_linear(src, self$weight, self$bias)

    # Check which rows have ALL values equal to skip_value
    # R: all() along last dimension, keepdims for broadcasting
    skip_mask <- torch_min(src == self$skip_value, dim = -1L)[[1]]


    if (torch_max(skip_mask)$item()) {
      # Replace skipped rows with skip_value
      # R: indexing with boolean mask on first dimension
      out[skip_mask, ..] <- self$skip_value
    }

    out
  }
)


#' Enhanced multi-head attention with RoPE, scalable softmax, and KV caching
#'
#' @param embed_dim Integer. Model dimension (total size of each attention head combined).
#' @param num_heads Integer. Number of attention heads.
#' @param dropout Numeric. Dropout probability applied to attention weights (default: `0.0`).
#' @param ssmax Logical or character. Type of scalable softmax to use:
#'   - `FALSE` or `"none"`: No scaling applied
#'   - `TRUE` or `"qassmax-mlp-elementwise"`: Elementwise query-aware scaling
#'   - Other strings: specific scalable softmax type
#'   (default: `FALSE`)
#'
#' @return A `nn_module` for multi-head attention with advanced features.
#'
#' @note The implementation always uses `batch_first = TRUE`, so input tensors have
#'   shape `(.., seq_len, embed_dim)`.
#'
#' @references
#' \enumerate{
#'   \item Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
#'   \item Liu et al., "Scalable-Softmax Is Superior for Attention"
#' }
#'
#' @export
multihead_attention <- nn_module(
  "MultiheadAttention",
  inherit = nn_multihead_attention,
  initialize = function(embed_dim, num_heads, dropout = 0.0, ssmax = FALSE) {
    # Parent nn_multihead_attention with batch_first = TRUE
    super$initialize(
      embed_dim = embed_dim,
      num_heads = num_heads,
      dropout = dropout,
      batch_first = TRUE
    )

    # Normalize ssmax argument
    if (is.logical(ssmax)) {
      ssmax <- if (ssmax) "qassmax-mlp-elementwise" else "none"
    }

    # Create scalable softmax layer
    self$ssmax_layer <- create_ssmax_layer(
      ssmax_type = ssmax,
      num_heads = num_heads,
      embed_dim = embed_dim
    )
  },

  # Compute multi-head attention
  #
  # @param query Query tensor of shape `(.., tgt_len, embed_dim)`.
  # @param key Optional key tensor of shape `(.., src_len, embed_dim)`.
  #   Required when `cached_kv` is `NULL`.
  # @param value Optional value tensor of shape `(.., src_len, embed_dim)`.
  #   Required when `cached_kv` is `NULL`.
  # @param cached_kv Optional `KVCacheEntry` for pre-computed key/value projections.
  #   When provided, `key` and `value` parameters are ignored.
  # @param key_padding_mask Optional mask of shape `(.., src_len)` that identifies
  #   padding elements in the key sequence:
  #   - Binary masks: `TRUE` values indicate positions to ignore
  #   - Float masks: values are directly added to attention scores
  # @param attn_mask Optional attention mask of shape `(tgt_len, src_len)` or
  #   `(.., num_heads, tgt_len, src_len)`.
  # @param rope Optional `RotaryEmbedding` for rotary positional encoding.
  # @param need_kv Logical. If `TRUE` and `cached_kv` is `NULL`, also returns
  #   the computed K and V projections along with the attention output
  #   (default: `FALSE`).
  #
  # @return If `need_kv` is `FALSE` or `cached_kv` is provided:
  #   Attention output of shape `(.., tgt_len, embed_dim)`.
  #   If `need_kv` is `TRUE` and `cached_kv` is `NULL`:
  #   List of `(attn_output, k, v)` where:
  #   - `attn_output`: shape `(.., tgt_len, embed_dim)`
  #   - `k`: shape `(.., num_heads, src_len, head_dim)`
  #   - `v`: shape `(.., num_heads, src_len, head_dim)`
  forward = function(
    query,
    key = NULL,
    value = NULL,
    cached_kv = NULL,
    key_padding_mask = NULL,
    attn_mask = NULL,
    rope = NULL,
    need_kv = FALSE
  ) {
    # Canonical mask handling (internal PyTorch helpers, adapted for R)
    key_padding_mask <- .canonical_mask(
      mask = key_padding_mask,
      mask_name = "key_padding_mask",
      other_type = .none_or_dtype(attn_mask),
      other_name = "src_mask",
      target_type = query$dtype
    )

    attn_mask <- .canonical_mask(
      mask = attn_mask,
      mask_name = "attn_mask",
      other_type = NULL,
      other_name = "",
      target_type = query$dtype,
      check_other = FALSE
    )

    # Delegate to shared implementation
    multi_head_attention_forward(
      query = query,
      num_heads = self$num_heads,
      in_proj_weight = self$in_proj_weight,
      in_proj_bias = self$in_proj_bias,
      dropout = self$dropout,
      out_proj_weight = self$out_proj$weight,
      out_proj_bias = self$out_proj$bias,
      key = key,
      value = value,
      cached_kv = cached_kv,
      training = self$training,
      key_padding_mask = key_padding_mask,
      attn_mask = attn_mask,
      rope = rope,
      ssmax_layer = self$ssmax_layer,
      need_kv = need_kv
    )
  }
)


#' Attention block supporting RoPE, scalable softmax, and KV caching
#'
#' @param d_model Integer. Model dimension.
#' @param nhead Integer. Number of attention heads.
#' @param dim_feedforward Integer. Dimension of the feedforward network.
#' @param dropout Numeric. Dropout probability (default: `0.0`).
#' @param activation Character or function. Activation function for feedforward:
#'   `"relu"`, `"gelu"`, or a unary callable (default: `"gelu"`).
#' @param norm_first Logical. If `TRUE`, uses pre-norm architecture
#'   (LayerNorm before attention and feedforward) (default: `TRUE`).
#' @param bias_free_ln Logical. If `TRUE`, removes bias from all LayerNorm layers
#'   (default: `FALSE`).
#' @param ssmax Logical or character. Type of scalable softmax to use in attention
#'   (default: `FALSE`).
#'
#' @return A `nn_module` for transformer encoder layer with advanced features.
#'
#' @export
multihead_attention_block <- nn_module(
  "MultiheadAttentionBlock",
  initialize = function(
    d_model,
    nhead,
    dim_feedforward,
    dropout = 0.0,
    activation = "gelu",
    norm_first = TRUE,
    bias_free_ln = FALSE,
    ssmax = FALSE
  ) {
    # Store configuration
    self$d_model <- d_model
    self$nhead <- nhead
    self$norm_first <- norm_first

    # Self-attention with optional SSMax
    self$attn <- multihead_attention(d_model, nhead, dropout, ssmax)

    # Feedforward network
    self$linear1 <- nn_linear(d_model, dim_feedforward)
    self$dropout <- nn_dropout(dropout)
    self$linear2 <- nn_linear(dim_feedforward, d_model)

    # Activation function (store as character for later use in forward)
    self$activation_name <- if (is.character(activation)) activation else "custom"
    if (!is.character(activation)) {
      self$activation_fn <- activation
    }

    # Layer normalization
    self$norm1 <- nn_layer_norm(d_model, elementwise_affine = !bias_free_ln)
    self$norm2 <- nn_layer_norm(d_model, elementwise_affine = !bias_free_ln)

    # Dropout for residual connections
    self$dropout1 <- nn_dropout(dropout)
    self$dropout2 <- nn_dropout(dropout)

    # Initialize weights for stable training
    self$init_weights()
  },

  # Initialize projection layers to zero for stable training
  # @keywords internal
  init_weights = function() {
    nn_init_zeros_(self$attn$out_proj$weight)
    nn_init_zeros_(self$attn$out_proj$bias)
    nn_init_zeros_(self$linear2$weight)
    nn_init_zeros_(self$linear2$bias)
  },

  # Process input through attention
  #
  # @param q Query tensor of shape `(.., tgt_len, d_model)`.
  # @param k Optional key tensor of shape `(.., src_len, d_model)`.
  #   If `NULL`, uses `q` for self-attention.
  # @param v Optional value tensor of shape `(.., src_len, d_model)`.
  #   If `NULL`, uses `q` for self-attention.
  # @param cached_kv Optional `KVCacheEntry` for pre-computed K/V projections.
  # @param key_padding_mask Optional mask of shape `(.., src_len)`.
  # @param attn_mask Optional attention mask.
  # @param train_size Optional integer. When provided (requires `k = NULL` and
  #   `v = NULL`), the full sequence is used as query while only the first
  #   `train_size` positions serve as key/value.
  # @param rope Optional `RotaryEmbedding` for rotary positional encoding.
  # @param need_kv Logical. If `TRUE`, also returns the computed K and V
  #   projections (default: `FALSE`).
  #
  # @return If `need_kv` is `FALSE`: Output tensor of shape `(.., tgt_len, d_model)`.
  #   If `need_kv` is `TRUE`: List of `(output, k, v)`.
  forward = function(
    q,
    k = NULL,
    v = NULL,
    cached_kv = NULL,
    key_padding_mask = NULL,
    attn_mask = NULL,
    train_size = NULL,
    rope = NULL,
    need_kv = FALSE
  ) {
    # Handle train_size: restrict k/v to training portion if specified
    if (is.null(train_size)) {
      k <- if (is.null(k)) q else k
      v <- if (is.null(v)) q else v
    } else {
      if (!is.null(k) || !is.null(v)) {
        value_error("k and v must be NULL when train_size is provided")
      }
      # R: 1-based inclusive slicing for training portion
      k <- v <- q[.., seq_len(train_size), ..]
    }

    k_proj <- NULL
    v_proj <- NULL
    use_cache <- !is.null(cached_kv)

    if (self$norm_first) {
      # Pre-norm: normalize first, then apply attention
      q_normed <- self$norm1(q)

      if (use_cache) {
        attn <- self$.attn_block(
          q_normed, cached_kv = cached_kv, key_padding_mask = key_padding_mask,
          attn_mask = attn_mask, rope = rope
        )
      } else {
        if (is.null(train_size)) {
          k_normed <- if (identical(k, q)) q_normed else self$norm1(k)
          v_normed <- if (identical(v, k)) k_normed else self$norm1(v)
        } else {
          # R: slice training portion
          k_normed <- v_normed <- q_normed[.., seq_len(train_size), ..]
        }

        attn_result <- self$.attn_block(
          q_normed, k_normed, v_normed,
          key_padding_mask = key_padding_mask,
          attn_mask = attn_mask,
          rope = rope,
          need_kv = need_kv
        )

        if (need_kv && is.list(attn_result)) {
          attn <- attn_result[[1L]]
          k_proj <- attn_result[[2L]]
          v_proj <- attn_result[[3L]]
        } else {
          attn <- attn_result
        }
      }

      # Residual connection + feedforward
      x <- q + attn
      x <- x + self$.ff_block(self$norm2(x))
    } else {
      # Post-norm: attention first, then normalize
      if (use_cache) {
        attn <- self$.attn_block(
          q, cached_kv = cached_kv, key_padding_mask = key_padding_mask,
          attn_mask = attn_mask, rope = rope
        )
      } else {
        attn_result <- self$.attn_block(
          q, k, v,
          key_padding_mask = key_padding_mask,
          attn_mask = attn_mask,
          rope = rope,
          need_kv = need_kv
        )

        if (need_kv && is.list(attn_result)) {
          attn <- attn_result[[1L]]
          k_proj <- attn_result[[2L]]
          v_proj <- attn_result[[3L]]
        } else {
          attn <- attn_result
        }
      }

      x <- self$norm1(q + attn)
      x <- self$norm2(x + self$.ff_block(x))
    }

    if (need_kv && !is.null(k_proj)) {
      return(list(x, k_proj, v_proj))
    }

    x
  },

  # Internal attention block helper
  # @keywords internal
  .attn_block = function(
    q, k = NULL, v = NULL, cached_kv = NULL,
    key_padding_mask = NULL, attn_mask = NULL, rope = NULL, need_kv = FALSE
  ) {
    result <- self$attn(
      q, k, v,
      cached_kv = cached_kv,
      key_padding_mask = key_padding_mask,
      attn_mask = attn_mask,
      rope = rope,
      need_kv = need_kv
    )

    if (need_kv && is.list(result)) {
      # Apply dropout to attention output only
      attn <- self$dropout1(result[[1L]])
      return(list(attn, result[[2L]], result[[3L]]))
    }

    self$dropout1(result)
  },

  # Internal feedforward block helper
  # @keywords internal
  .ff_block = function(x) {
    x <- self$linear1(x)
    # Apply activation
    x <- if (self$activation_name == "relu") {
      nnf_relu(x)
    } else if (self$activation_name == "gelu") {
      nnf_gelu(x)
    } else if (exists("activation_fn", envir = self)) {
      self$activation_fn(x)
    } else {
      nnf_gelu(x)  # default
    }
    x <- self$dropout(x)
    x <- self$linear2(x)
    self$dropout2(x)
  }
)


#' Induced Self-Attention Block for efficient O(n) attention
#'
#' This module implements a bottleneck attention mechanism using a small set of
#' learned inducing points that mediate interactions between input elements.
#' The complexity is reduced from O(n²) to O(n) by:
#' \enumerate{
#'   \item Projecting inputs onto inducing points (size m << n)
#'   \item Propagating information through these inducing points
#'   \item Projecting back to the original sequence
#' }
#'
#' @param d_model Integer. Model dimension.
#' @param nhead Integer. Number of attention heads.
#' @param dim_feedforward Integer. Dimension of the feedforward network.
#' @param num_inds Integer. Number of inducing points (controls capacity vs. efficiency).
#' @param dropout Numeric. Dropout probability (default: `0.0`).
#' @param activation Character or function. Activation function for feedforward
#'   (default: `"gelu"`).
#' @param norm_first Logical. If `TRUE`, uses pre-norm architecture (default: `TRUE`).
#' @param bias_free_ln Logical. If `TRUE`, removes bias from all LayerNorm layers
#'   (default: `FALSE`).
#' @param skip_value Numeric. Value used to mark inputs that should be skipped
#'   (default: `-100.0`).
#' @param ssmax Logical or character. Type of scalable softmax to use in attention.
#'   Note that only the first attention layer uses SSMax (default: `FALSE`).
#'
#' @return A `nn_module` for induced self-attention.
#'
#' @references
#' Lee et al. "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks", ICML 2019
#'
#' @export
induced_self_attention_block <- nn_module(
  "InducedSelfAttentionBlock",
  initialize = function(
    d_model,
    nhead,
    dim_feedforward,
    num_inds,
    dropout = 0.0,
    activation = "gelu",
    norm_first = TRUE,
    bias_free_ln = FALSE,
    ssmax = FALSE,
    skip_value = -100.0
  ) {
    self$skip_value <- skip_value

    # Normalize ssmax argument
    if (is.logical(ssmax)) {
      ssmax <- if (ssmax) "qassmax-mlp-elementwise" else "none"
    }

    # Two-stage attention mechanism
    # First stage: only first layer uses ssmax
    self$multihead_attn1 <- multihead_attention_block(
      d_model, nhead, dim_feedforward, dropout, activation, norm_first, bias_free_ln, ssmax
    )
    # Second stage: no ssmax
    self$multihead_attn2 <- multihead_attention_block(
      d_model, nhead, dim_feedforward, dropout, activation, norm_first, bias_free_ln
    )

    # Learnable inducing points
    self$num_inds <- num_inds
    self$ind_vectors <- nn_parameter(torch_empty(num_inds, d_model))
    nn_init_trunc_normal_(self$ind_vectors, std = 0.02)
  },

  # Apply induced self-attention to input sequence
  #
  # @param src Input tensor of shape `(.., seq_len, d_model)`.
  # @param train_size Optional integer. Position to split input into training/test data.
  # @return Output tensor with same shape as input.
  # @keywords internal
  .induced_attention = function(src, train_size = NULL) {
    # Extract batch shape and model dimension
    src_shape <- src$shape
    ndim <- length(src_shape)
    batch_shape <- src_shape[seq_len(ndim - 2L)]  # All but last two dims
    d_model <- src_shape[ndim]

    # Expand inducing vectors to match batch dimensions
    ind_vectors <- self$ind_vectors
    for (i in seq_along(batch_shape)) {
      ind_vectors <- ind_vectors$unsqueeze(1L)  # Add dim at position 1
    }
    ind_vectors <- ind_vectors$expand(c(batch_shape, self$num_inds, d_model))

    # First attention: inducing points attend to src
    if (is.null(train_size)) {
      hidden <- self$multihead_attn1(ind_vectors, src, src)
    } else {
      # R: 1-based inclusive slicing for training portion
      hidden <- self$multihead_attn1(
        ind_vectors,
        src[.., seq_len(train_size), ..],
        src[.., seq_len(train_size), ..]
      )
    }

    # Second attention: src attends to hidden (inducing points)
    self$multihead_attn2(src, hidden, hidden)
  },

  # Main forward: apply induced self-attention with skip value handling
  #
  # @param src Input tensor of shape `(.., seq_len, d_model)`.
  # @param train_size Optional integer. Position to split input into training/test data.
  #   When provided, inducing points only attend to training data in the first
  #   attention stage to prevent information leakage.
  # @return Output tensor with same shape as input.
  forward = function(src, train_size = NULL) {
    # Identify rows where ALL values equal skip_value (across feature dims)
    # R: all() along last two dimensions (seq_len, d_model)
    # Apply min twice: first along dim -1, then along dim -2
    skip_check <- (src == self$skip_value)$min(dim = -1L)[[1]]$min(dim = -1L)[[1]]
    skip_mask <- skip_check > 0

    if (torch_max(skip_mask)$item()) {
      if (torch_min(skip_mask)$item() > 0) {
        # All batches are skipped: return filled tensor
        torch_full_like(src, self$skip_value)
      } else {
        # Mixed: compute only for non-skipped batches
        out <- torch_zeros_like(src)
        # R: boolean indexing on batch dimension
        out[!skip_mask, ..] <- self$.induced_attention(src[!skip_mask, ..], train_size)
        # Restore skip values
        out[skip_mask, ..] <- self$skip_value
        out
      }
    } else {
      # No skips: standard forward
      self$.induced_attention(src, train_size)
    }
  },

  # Apply induced self-attention with optional caching
  #
  # @param src Input tensor of shape `(.., seq_len, d_model)`.
  # @param col_cache `KVCache` object for storing/retrieving K/V of the second attention layer.
  # @param block_idx Integer. Index of this block for cache key.
  # @param train_size Optional integer. Position to split input into training/test data.
  # @param use_cache Logical. Whether to use cached values (default: `FALSE`).
  # @param store_cache Logical. Whether to store computed values in cache (default: `TRUE`).
  # @return Output tensor with same shape as input.
  # @keywords internal
  .induced_attention_with_cache = function(
    src, col_cache, block_idx, train_size = NULL, use_cache = FALSE, store_cache = TRUE
  ) {
    # Extract batch shape and model dimension
    src_shape <- src$shape
    ndim <- length(src_shape)
    batch_shape <- src_shape[seq_len(ndim - 2L)]
    d_model <- src_shape[ndim]

    # Expand inducing vectors
    ind_vectors <- self$ind_vectors
    for (i in seq_along(batch_shape)) {
      ind_vectors <- ind_vectors$unsqueeze(1L)
    }
    ind_vectors <- ind_vectors$expand(c(batch_shape, self$num_inds, d_model))

    out <- NULL

    if (use_cache) {
      if (!block_idx %in% names(col_cache$kv)) {
        runtime_error("Cache miss for kv at ISAB {block_idx}")
      }
      out <- self$multihead_attn2(src, cached_kv = col_cache$kv[[as.character(block_idx)]])
    }

    if (store_cache) {
      if (is.null(train_size)) {
        value_error("train_size must be provided when store_cache = TRUE")
      }
      # First attention: inducing points attend to training portion only
      hidden <- self$multihead_attn1(
        ind_vectors,
        src[.., seq_len(train_size), ..],
        src[.., seq_len(train_size), ..]
      )
      # Second attention: get K/V projections for caching
      result <- self$multihead_attn2(src, hidden, hidden, need_kv = TRUE)

      if (is.list(result)) {
        out <- result[[1L]]
        k_proj <- result[[2L]]
        v_proj <- result[[3L]]
        # Store in cache
        col_cache$kv[[as.character(block_idx)]] <- kv_cache_entry(key = k_proj, value = v_proj)
      } else {
        out <- result
      }
    }

    out
  },

  # Forward with caching support and skip value handling
  #
  # @param src Input tensor of shape `(.., seq_len, d_model)`.
  # @param col_cache `KVCache` object for storing/retrieving hidden tensors and K/V.
  # @param block_idx Integer. Index of this block for cache key.
  # @param train_size Optional integer. Position to split input into training/test data.
  # @param use_cache Logical. Whether to use cached values (default: `FALSE`).
  # @param store_cache Logical. Whether to store computed values in cache (default: `TRUE`).
  # @return Output tensor with same shape as input.
  forward_with_cache = function(
    src, col_cache, block_idx, train_size = NULL, use_cache = FALSE, store_cache = TRUE
  ) {
    # Validate mutual exclusivity
    if (!xor(use_cache, store_cache)) {
      value_error("Exactly one of use_cache or store_cache must be TRUE")
    }

    if (store_cache && is.null(train_size)) {
      value_error("train_size must be provided when store_cache = TRUE")
    }

    # Identify skip rows
    skip_mask <- torch_min(src == self$skip_value, dim = c(-2L, -1L)) > 0

    if (torch_min(skip_mask)$item() > 0) {
      # All skipped: return filled tensor
      return(torch_full_like(src, self$skip_value))
    }

    # Compute with caching
    out <- self$.induced_attention_with_cache(
      src, col_cache, block_idx, train_size, use_cache, store_cache
    )

    # Restore skip values in output if any were skipped
    if (torch_max(skip_mask)$item()) {
      out[skip_mask, ..] <- self$skip_value
    }

    out
  }
)


#' Canonical mask helper for attention
#'
#' @keywords internal
.canonical_mask <- function(mask, mask_name, other_type, other_name, target_type, check_other = TRUE) {
  if (is.null(mask)) {
    return(NULL)
  }

  # Ensure mask has correct dtype
  if (mask$dtype != target_type) {
    mask <- mask$to(dtype = target_type)
  }

  mask
}

#' Check if value is NULL or extract dtype from tensor
#'
#' @keywords internal
.none_or_dtype <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  x$dtype
}
