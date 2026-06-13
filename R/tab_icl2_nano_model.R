#' @importFrom torch nn_module nn_embedding nn_module_list nn_parameter nn_layer_norm nn_gelu nn_sequential nn_init_uniform_ nn_init_zeros_
#' @importFrom torch torch_arange torch_stack torch_randn torch_randint torch_cat torch_transpose torch_reshape torch_empty torch_load
#' @importFrom torch torch_linspace torch_sin torch_cos torch_log torch_tanh torch_long torch_scaled_dot_product_attention
#' @importFrom torch nnf_linear nn_linear torch_tensor
#' @importFrom cli cli_inform cli_warn cli_abort
#' @importFrom glue glue
#' @importFrom purrr map reduce
#' @keywords internal
NULL

.get_mlp <- function(n_in, n_hidden, n_out) {
  nn_sequential(
    nn_linear(n_in, n_hidden),
    nn_gelu(),
    nn_linear(n_hidden, n_out)
  )
}

ClassEmbedding <- torch::nn_module(
  "ClassEmbedding",

  initialize = function(num_embeddings, embedding_dim) {
    self$embedding <- nn_embedding(num_embeddings, embedding_dim)
    bound <- 1 / sqrt(self$embedding$num_embeddings)
    nn_init_uniform_(self$embedding$weight, -bound, bound)
  },

  forward = function(y) {
    # squeeze last dim, convert to long for embedding lookup
    idx <- y$squeeze(-1)$to(dtype = torch_long())
    self$embedding(idx)
  }
)


NanoRope <- torch::nn_module(
  "Rope",

  initialize = function(head_dim, theta = 100000) {
    self$half <- head_dim %/% 2
    inv_freq <- theta ^ torch_linspace(0, -1, self$half + 1)[1:self$half]
    self$register_buffer("inv_freq", inv_freq, persistent = FALSE)
    self$register_buffer("sin", torch_empty(0), persistent = FALSE)
    self$register_buffer("cos", torch_empty(0), persistent = FALSE)
  },

  forward = function(x) {
    # x shape: (batch, heads, seq_len, head_dim)
    seq_len <- x$shape[3]

    # Recompute rotation buffers once if empty or too short
    if (self$cos$numel() == 0 || !identical(self$cos$device, x$device) || self$cos$shape[1] < seq_len) {
      pos <- torch_arange(seq_len, device = x$device, dtype = self$inv_freq$dtype)
      angles <- pos[, NULL] * self$inv_freq[NULL, ]
      self$register_buffer("sin", angles$sin())
      self$register_buffer("cos", angles$cos())
    }

    sin <- self$sin[1:seq_len, ..]
    cos <- self$cos[1:seq_len, ..]

    # Split x into two halves along the last dimension
    x1 <- x[.., 1:self$half]
    x2 <- x[.., (self$half + 1):N]


    # Apply rotary encoding
    torch_cat(list(x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim = 4)$to(x$dtype)
  }
)

QASSMax <- torch::nn_module(
  "QASSMax",

  initialize = function(num_heads, head_dim, n_hidden = 64L) {
    self$base_mlp <- .get_mlp(1, n_hidden, num_heads * head_dim)
    self$query_mlp <- .get_mlp(head_dim, n_hidden, head_dim)
    nn_init_zeros_(self$query_mlp[[3]]$weight)
    nn_init_zeros_(self$query_mlp[[3]]$bias)
  },

  forward = function(q, n) {
    num_heads <- q$shape[2]
    head_dim <- q$shape[4]

    logn <- torch_tensor(log(max(1, n)))$view(c(1, 1))
    base_mod <- self$base_mlp(logn)$view(c(1, num_heads, 1, head_dim))
    query_mod <- 1 + torch_tanh(self$query_mlp(q))

    base_mod * query_mod * q
  }
)

NanoTransformerBlock <- torch::nn_module(
  "TransformerBlock",

  initialize = function(embed_dim, num_heads, use_rope = FALSE, ssmax = FALSE) {
    self$mha <- nn_multihead_attention(embed_dim = embed_dim, num_heads = num_heads)
    self$in_proj_weight <- self$mha$in_proj_weight
    self$in_proj_bias <- self$mha$in_proj_bias
    self$out_proj <- self$mha$out_proj
    self$num_heads <- num_heads
    self$head_dim <- embed_dim %/% num_heads

    self$rope <- if (use_rope) NanoRope(head_dim = embed_dim %/% num_heads, theta = 100000) else NULL
    self$ssmax_layer <- if (ssmax) QASSMax(num_heads = num_heads, head_dim = embed_dim %/% num_heads) else NULL
    self$mlp <- .get_mlp(embed_dim, embed_dim * 2, embed_dim)
    self$ln_attn <- nn_layer_norm(embed_dim)
    self$ln_mlp <- nn_layer_norm(embed_dim)
  },

  forward = function(q, kv = NULL, q_max_idx = NULL, kv_max_idx = NULL) {
    x <- q
    q_norm <- self$ln_attn(q)
    kv_norm <- if (is.null(kv)) q_norm else self$ln_attn(kv)

    if (!is.null(kv_max_idx)) {
      kv_norm <- kv_norm[, 1:kv_max_idx, ..]
    }
    if (!is.null(q_max_idx)) {
      x <- x[, 1:q_max_idx, ..]
      q_norm <- q_norm[, 1:q_max_idx, ..]
    }

    x <- x + self$.attn(q_norm, kv_norm)
    x + self$mlp(self$ln_mlp(x))
  },

  .attn = function(q, k) {
    e <- self$head_dim * self$num_heads

    # Manually split the fused projection weights & biases
    w_q <- self$in_proj_weight[1:e, ..]
    w_k <- self$in_proj_weight[(e + 1):(2 * e), ..]
    w_v <- self$in_proj_weight[(2 * e + 1):(3 * e), ..]

    b_q <- if (!is.null(self$in_proj_bias)) self$in_proj_bias[1:e] else NULL
    b_k <- if (!is.null(self$in_proj_bias)) self$in_proj_bias[(e + 1):(2 * e)] else NULL
    b_v <- if (!is.null(self$in_proj_bias)) self$in_proj_bias[(2 * e + 1):(3 * e)] else NULL

    # Compute Q, K, V projections (replaces _in_projection_packed)
    q_proj <- nnf_linear(q, w_q, b_q)
    k_proj <- nnf_linear(k, w_k, b_k)
    v_proj <- nnf_linear(k, w_v, b_v)

    # Reshape to multi-head: (batch, seq, embed_dim) -> (batch, heads, seq, head_dim)
    to_multihead <- function(t) {
      t$view(c(t$shape[1], t$shape[2], self$num_heads, self$head_dim))$transpose(2, 3)
    }
    q_h <- to_multihead(q_proj)
    k_h <- to_multihead(k_proj)
    v_h <- to_multihead(v_proj)

    if (!is.null(self$ssmax_layer)) {
      q_h <- self$ssmax_layer(q = q_h, n = k_h$size(3))
    }

    if (!is.null(self$rope)) {
      q_h <- self$rope(q_h)
      k_h <- self$rope(k_h)
    }

    # scaled_dot_product_attention natively accepts (batch, heads, seq, head_dim)
    attn_out <- torch_scaled_dot_product_attention(q_h, k_h, v_h)

    # Reshape back to (batch, seq, embed_dim)
    out <- attn_out$transpose(2, 3)$contiguous()$view(c(attn_out$shape[1], attn_out$shape[3], e))
    self$out_proj(out)
  },

  row_attn = function(q, kv = NULL, ...) {
    n_batch <- q$shape[1]
    n_rows <- q$shape[2]
    embed_dim <- q$shape[4]

    # Flatten batch and rows: (B, R, C, D) -> (B*R, C, D)
    q_flat <- q$reshape(c(n_batch * n_rows, -1, embed_dim))
    kv_flat <- if (is.null(kv)) NULL else kv$reshape(c(n_batch * n_rows, -1, embed_dim))

    result <- self$forward(q_flat, kv_flat, ...)
    result$reshape(c(n_batch, n_rows, -1, embed_dim))
  },

  col_attn = function(q, kv = NULL, ...) {
    # Transpose rows and cols: (B, R, C, D) -> (B, C, R, D)
    q_t <- q$transpose(2, 3)
    kv_t <- if (is.null(kv)) NULL else kv$transpose(2, 3)
    result <- self$row_attn(q_t, kv_t, ...)
    result$transpose(2, 3)
  }
)

NanoInducedTransformerBlock <- torch::nn_module(
  "InducedTransformerBlock",

  initialize = function(embed_dim, num_heads, n_inducing, ssmax = FALSE) {
    self$tfm1 <- NanoTransformerBlock(embed_dim = embed_dim, num_heads = num_heads, ssmax = ssmax)
    self$tfm2 <- NanoTransformerBlock(embed_dim = embed_dim, num_heads = num_heads)
    self$inducing_vectors <- nn_parameter(0.02 * torch_randn(1, n_inducing, embed_dim))
  },

  forward = function(q, kv = NULL, q_max_idx = NULL, kv_max_idx = NULL) {
    batch_size <- q$shape[1]
    n_inducing <- self$inducing_vectors$shape[2]
    embed_dim <- self$inducing_vectors$shape[3]
    inducing_expanded <- self$inducing_vectors$expand(c(batch_size, n_inducing, embed_dim))
    kv_out <- self$tfm1(inducing_expanded, if (is.null(kv)) q else kv, kv_max_idx = kv_max_idx)
    self$tfm2(q, kv_out, q_max_idx = q_max_idx)
  },

  row_attn = function(q, kv = NULL, ...) {
    self$tfm2$row_attn(q, kv, ...)
  },

  col_attn = function(q, kv = NULL, ...) {
    self$tfm2$col_attn(q, kv, ...)
  }
)

#' NanoTabICL v2 Model
#'
#' A compact tabular in-context learning model with column, row, and ICL attention blocks.
#'
#' @param max_classes Maximum number of classes (0 for regression)
#' @param out_dim Output dimension (n_classes for classification, n_quantiles for regression)
#' @param embed_dim Embedding dimension for features
#' @param col_n_block Number of column transformer blocks
#' @param row_n_block Number of row transformer blocks
#' @param icl_n_block Number of ICL transformer blocks
#' @param col_n_head Number of attention heads for column blocks
#' @param row_n_head Number of attention heads for row blocks
#' @param icl_n_head Number of attention heads for ICL blocks
#' @param feature_group_size Size of feature groups for repeated grouping
#' @param col_n_cls Number of CLS tokens per column
#' @param row_n_cls Number of inducing vectors for column attention
#' @return An nn_module ready for training/inference
#' @export
NanoTabICLv2 <- torch::nn_module(
  "NanoTabICLv2",

  initialize = function(
    max_classes,
    out_dim,
    embed_dim = 128L,
    col_n_block = 3L,
    row_n_block = 3L,
    icl_n_block = 12L,
    col_n_head = 8L,
    row_n_head = 8L,
    icl_n_head = 8L,
    feature_group_size = 3L,
    col_n_cls = 4L,
    row_n_cls = 128L
  ) {
    self$feature_group_size <- feature_group_size
    icl_dim <- embed_dim * col_n_cls

    self$x_embed <- nn_linear(feature_group_size, embed_dim)
    self$y_embed_in <- if (max_classes > 0) {
      ClassEmbedding(max_classes, embed_dim)
    } else {
      nn_linear(1, embed_dim)
    }
    self$y_embed_icl <- if (max_classes > 0) {
      ClassEmbedding(max_classes, icl_dim)
    } else {
      nn_linear(1, icl_dim)
    }

    self$col_blocks <- nn_module_list(
      purrr::map(seq_len(col_n_block), function(i) {
        NanoInducedTransformerBlock(
          embed_dim = embed_dim,
          num_heads = col_n_head,
          n_inducing = row_n_cls,
          ssmax = TRUE
        )
      })
    )

    self$row_blocks <- nn_module_list(
      purrr::map(seq_len(row_n_block), function(i) {
        NanoTransformerBlock(
          embed_dim = embed_dim,
          num_heads = row_n_head,
          use_rope = TRUE
        )
      })
    )

    self$icl_blocks <- nn_module_list(
      purrr::map(seq_len(icl_n_block), function(i) {
        NanoTransformerBlock(
          embed_dim = icl_dim,
          num_heads = icl_n_head,
          ssmax = TRUE
        )
      })
    )

    self$row_cls_tokens <- nn_parameter(0.02 * torch_randn(1, 1, col_n_cls, embed_dim))
    self$row_ln <- nn_layer_norm(embed_dim)
    self$out_ln <- nn_layer_norm(icl_dim)
    self$out_mlp <- .get_mlp(icl_dim, icl_dim * 2, out_dim)
  },

  forward = function(x, y) {
    n_batch <- x$shape[1]
    n_rows <- x$shape[2]
    n_cols <- x$shape[3]
    n_train <- y$shape[2]

    # Standardize x based on training subset only
    train_std <- x[, 1:n_train, ..]$std(dim = 2, unbiased = FALSE, keepdim = TRUE)
    x <- x / (train_std + 1e-8)

    # Feature grouping with cyclic shifts
    idxs <- torch_arange(n_cols, dtype = torch_long(), device = x$device)
    grouped <- purrr::map(seq_len(self$feature_group_size), function(i) {
      shift <- (2^i - 1) %% n_cols
      # Use torch_long() to ensure integer indexing
      indices <- ((idxs + shift - 1L) %% n_cols + 1L)$to(dtype = torch_long())
      x[.., indices]
    })
    x_grouped <- torch_stack(grouped, dim = 4)

    emb <- self$x_embed(x_grouped)

    # Add y embedding to training rows
    y_expanded <- y[.., NULL, NULL]
    emb[, 1:n_train, ..] <- emb[, 1:n_train, ..] + self$y_embed_in(y_expanded)

    # Column attention blocks (index-based reduce)
    n_col_blocks <- length(self$col_blocks)
    emb <- purrr::reduce(seq_len(n_col_blocks), function(acc, i) {
      self$col_blocks[[i]]$col_attn(acc, kv_max_idx = n_train)
    }, .init = emb)

    # Row attention blocks ; concat CLS tokens and row attention
    cls_shape <- self$row_cls_tokens$shape
    # TODO this seems to bypass the "Repeated Feature Grouping"
    cls_expanded <- self$row_cls_tokens$expand(c(n_batch, n_rows, cls_shape[3], cls_shape[4]))
    emb <- torch_cat(list(cls_expanded, emb), dim = 3)

    n_row_blocks <- length(self$row_blocks)
    if (n_row_blocks > 1) {
      emb <- purrr::reduce(seq_len(n_row_blocks - 1), function(acc, i) {
        self$row_blocks[[i]]$row_attn(acc)
      }, .init = emb)
    }

    # Last row block: only keep CLS token outputs
    n_cls <- cls_shape[3]
    emb <- self$row_blocks[[n_row_blocks]]$row_attn(emb, q_max_idx = n_cls)

    # Normalize and flatten CLS tokens
    emb <- self$row_ln(emb)$flatten(start_dim = 3, end_dim = 4)

    # Add y embedding again for ICL attention
    y_icl_expanded <- y[.., NULL]
    emb[, 1:n_train, ..] <- emb[, 1:n_train, ..] + self$y_embed_icl(y_icl_expanded)

    # ICL attention blocks
    n_icl_blocks <- length(self$icl_blocks)
    if (n_icl_blocks > 1) {
      emb <- purrr::reduce(seq_len(n_icl_blocks - 1), function(acc, i) {
        self$icl_blocks[[i]]$forward(acc, kv_max_idx = n_train)
      }, .init = emb)
    }

    # Final block: test attends to train only
    last_block <- self$icl_blocks[[n_icl_blocks]]
    emb_test <- emb[, (n_train + 1):n_rows, ..]
    emb_train <- emb[, 1:n_train, ..]
    emb <- last_block(emb_test, emb_train)

    self$out_mlp(self$out_ln(emb))
  }
)
