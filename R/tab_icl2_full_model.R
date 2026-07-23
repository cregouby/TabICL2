#' @importFrom torch nn_module nn_embedding nn_module_list nn_parameter nn_layer_norm
#' @importFrom torch nn_gelu nn_sequential nn_init_uniform_ nn_init_zeros_ nn_linear
#' @importFrom torch torch_arange torch_stack torch_randn torch_randint torch_cat
#' @importFrom torch torch_transpose torch_reshape torch_empty torch_load torch_unique_consecutive
#' @importFrom torch torch_linspace torch_sin torch_cos torch_log torch_tanh torch_long
#' @importFrom torch torch_scaled_dot_product_attention nnf_linear torch_tensor
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
    idx <- y$squeeze(-1)$to(dtype = torch_long())
    self$embedding(idx)
  }
)

Rope <- torch::nn_module(
  "Rope",
  initialize = function(head_dim, theta = 100000, interleaved = FALSE) {
    self$half <- head_dim %/% 2L
    self$interleaved <- interleaved
    inv_freq <- theta ^ torch_linspace(0, -1, self$half + 1L)[1L:self$half]
    self$register_buffer("inv_freq", inv_freq, persistent = FALSE)
    self$register_buffer("sin", torch_empty(0), persistent = FALSE)
    self$register_buffer("cos", torch_empty(0), persistent = FALSE)
  },
  forward = function(x) {
    seq_len <- x$shape[3L]
    if (self$cos$numel() == 0 || !identical(self$cos$device, x$device) || self$cos$shape[1L] < seq_len) {
      pos <- torch_arange(seq_len, device = x$device, dtype = self$inv_freq$dtype)
      angles <- pos[, NULL] * self$inv_freq[NULL, ]
      self$register_buffer("sin", angles$sin())
      self$register_buffer("cos", angles$cos())
    }
    sin <- self$sin[seq_len(seq_len), ..]
    cos <- self$cos[seq_len(seq_len), ..]

    x1 <- x[.., 1L:self$half]
    x2 <- x[.., (self$half + 1L):N]

    torch_cat(list(x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim = 4L)$to(x$dtype)
  }
)

QASSMax <- torch::nn_module(
  "QASSMax",
  initialize = function(num_heads, head_dim, n_hidden = 64L) {
    self$base_mlp <- .get_mlp(1L, n_hidden, num_heads * head_dim)
    self$query_mlp <- .get_mlp(head_dim, n_hidden, head_dim)
    nn_init_zeros_(self$query_mlp[[3L]]$weight)
    nn_init_zeros_(self$query_mlp[[3L]]$bias)
  },
  forward = function(q, n) {
    num_heads <- q$shape[2L]
    head_dim <- q$shape[4L]
    logn <- torch_tensor(log(max(1, n)))$view(c(1L, 1L))
    base_mod <- self$base_mlp(logn)$view(c(1L, num_heads, 1L, head_dim))
    query_mod <- 1 + torch_tanh(self$query_mlp(q))
    base_mod * query_mod * q
  }
)

TransformerBlock <- torch::nn_module(
  "TransformerBlock",
  initialize = function(embed_dim, num_heads, use_rope = FALSE, rope_base = 100000,
                        rope_interleaved = FALSE, ssmax = FALSE) {
    self$mha <- nn_multihead_attention(embed_dim = embed_dim, num_heads = num_heads)
    self$in_proj_weight <- self$mha$in_proj_weight
    self$in_proj_bias <- self$mha$in_proj_bias
    self$out_proj <- self$mha$out_proj
    self$num_heads <- num_heads
    self$head_dim <- embed_dim %/% num_heads

    self$rope <- if (use_rope) {
      Rope(head_dim = self$head_dim, theta = rope_base, interleaved = rope_interleaved)
    } else {
      NULL
    }
    self$ssmax_layer <- if (isTRUE(ssmax)) {
      QASSMax(num_heads = num_heads, head_dim = self$head_dim)
    } else {
      NULL
    }
    self$mlp <- .get_mlp(embed_dim, embed_dim * 2L, embed_dim)
    self$ln_attn <- nn_layer_norm(embed_dim)
    self$ln_mlp <- nn_layer_norm(embed_dim)
  },
  forward = function(q, kv = NULL, q_max_idx = NULL, kv_max_idx = NULL) {
    x <- q
    q_norm <- self$ln_attn(q)
    kv_norm <- if (is.null(kv)) q_norm else self$ln_attn(kv)

    if (!is.null(kv_max_idx)) kv_norm <- kv_norm[, seq_len(kv_max_idx), ..]
    if (!is.null(q_max_idx)) {
      x <- x[, seq_len(q_max_idx), ..]
      q_norm <- q_norm[, seq_len(q_max_idx), ..]
    }

    x <- x + self$.attn(q_norm, kv_norm)
    x + self$mlp(self$ln_mlp(x))
  },
  .attn = function(q, k) {
    e <- self$head_dim * self$num_heads
    w_q <- self$in_proj_weight[seq_len(e), ..]
    w_k <- self$in_proj_weight[(e + 1L):(2L * e), ..]
    w_v <- self$in_proj_weight[(2L * e + 1L):(3L * e), ..]

    b_q <- if (!is.null(self$in_proj_bias)) self$in_proj_bias[seq_len(e)] else NULL
    b_k <- if (!is.null(self$in_proj_bias)) self$in_proj_bias[(e + 1L):(2L * e)] else NULL
    b_v <- if (!is.null(self$in_proj_bias)) self$in_proj_bias[(2L * e + 1L):(3L * e)] else NULL

    q_proj <- nnf_linear(q, w_q, b_q)
    k_proj <- nnf_linear(k, w_k, b_k)
    v_proj <- nnf_linear(k, w_v, b_v)

    to_multihead <- function(t) {
      t$view(c(t$shape[1L], t$shape[2L], self$num_heads, self$head_dim))$transpose(2L, 3L)
    }
    q_h <- to_multihead(q_proj)
    k_h <- to_multihead(k_proj)
    v_h <- to_multihead(v_proj)

    if (!is.null(self$ssmax_layer)) q_h <- self$ssmax_layer(q = q_h, n = k_h$size(3L))
    if (!is.null(self$rope)) {
      q_h <- self$rope(q_h)
      k_h <- self$rope(k_h)
    }

    attn_out <- torch_scaled_dot_product_attention(q_h, k_h, v_h)
    out <- attn_out$transpose(2L, 3L)$contiguous()$view(c(attn_out$shape[1L], attn_out$shape[3L], e))
    self$out_proj(out)
  },
  row_attn = function(q, kv = NULL, ...) {
    n_batch <- q$shape[1L]
    n_rows <- q$shape[2L]
    embed_dim <- q$shape[4L]
    # (B, R, C, D) -> (B*R, C, D)
    q_flat <- q$reshape(c(n_batch * n_rows, -1L, embed_dim))
    kv_flat <- if (is.null(kv)) NULL else kv$reshape(c(n_batch * n_rows, -1L, embed_dim))
    result <- self$forward(q_flat, kv_flat, ...)
    # (B*R, C, D) ->  (B, R, C, D)
    result$reshape(c(n_batch, n_rows, -1L, embed_dim))
  },
  col_attn = function(q, kv = NULL, ...) {
    # (B, R, C, D) -> (B, C, R, D)
    q_t <- q$transpose(2L, 3L)
    kv_t <- if (is.null(kv)) NULL else kv$transpose(2L, 3L)
    result <- self$row_attn(q_t, kv_t, ...)
    # (B, C, R, D) -> (B, R, C, D)
    result$transpose(2L, 3L)
  }
)

InducedTransformerBlock <- torch::nn_module(
  "InducedTransformerBlock",
  initialize = function(embed_dim, num_heads, n_inducing, ssmax = FALSE) {
    self$tfm1 <- TransformerBlock(embed_dim = embed_dim, num_heads = num_heads, ssmax = ssmax)
    self$tfm2 <- TransformerBlock(embed_dim = embed_dim, num_heads = num_heads)
    self$inducing_vectors <- nn_parameter(0.02 * torch_randn(1L, n_inducing, embed_dim))
  },
  forward = function(q, kv = NULL, q_max_idx = NULL, kv_max_idx = NULL) {
    batch_size <- q$shape[1L]
    n_inducing <- self$inducing_vectors$shape[2L]
    embed_dim <- self$inducing_vectors$shape[3L]
    inducing_expanded <- self$inducing_vectors$expand(c(batch_size, n_inducing, embed_dim))
    kv_out <- self$tfm1(inducing_expanded, if (is.null(kv)) q else kv, kv_max_idx = kv_max_idx)
    self$tfm2(q, kv_out, q_max_idx = q_max_idx)
  },
  row_attn = function(q, kv = NULL, ...) self$tfm2$row_attn(q, kv, ...),
  col_attn = function(q, kv = NULL, ...) self$tfm2$col_attn(q, kv, ...)
)


RowInteractor <- torch::nn_module(
  "RowInteractor",
  initialize = function(embed_dim, num_blocks, nhead, num_cls, rope_base, rope_interleaved,
                        dropout, activation, norm_first, bias_free_ln, recompute) {
    # self$cls_tokens will be registered as "row_interactor.cls_tokens"
    self$cls_tokens <- nn_parameter(0.02 * torch_randn(1L, 1L, num_cls, embed_dim))
    self$row_blocks <- nn_module_list(
      purrr::map(seq_len(num_blocks), function(i) {
        TransformerBlock(
          embed_dim = embed_dim, num_heads = nhead, use_rope = TRUE,
          rope_base = rope_base, rope_interleaved = rope_interleaved, ssmax = FALSE
        )
      })
    )
    icl_dim <- embed_dim * num_cls
    self$out_ln <- nn_layer_norm(icl_dim)
  },
  forward = function(emb, d = NULL, mgr_config = NULL) {
    n_batch <- emb$shape[1L]
    cls_shape <- self$cls_tokens$shape

    # Concat CLS tokens: (B, T, n_cls, E) + (B, T, H, E) -> (B, T, n_cls + H, E)
    cls_expanded <- self$cls_tokens$expand(c(n_batch, emb$shape[2L], cls_shape[3L], cls_shape[4L]))
    x <- torch_cat(list(cls_expanded, emb), dim = 3L)

    n_row_blocks <- length(self$row_blocks)
    if (n_row_blocks > 1L) {
      x <- purrr::reduce(seq_len(n_row_blocks - 1L), function(acc, i) {
        self$row_blocks[[i]]$row_attn(acc)
      }, .init = x)
    }

    # Last block keeps only CLS tokens.
    # row_attn transposes to (B, n_cls+H, T, E), keeps q_max_idx -> (B, n_cls, T, E)
    n_cls <- cls_shape[3L]
    x <- self$row_blocks[[n_row_blocks]]$row_attn(x, q_max_idx = n_cls)

    # (B, n_cls, T, E) ->  -> flatten -> (B, T, n_cls * E)
    x <- x$flatten(start_dim = 3L, end_dim = 4L)

    self$out_ln(x)
  }
)

ICLPredictor <- torch::nn_module(
  "ICLPredictor",
  initialize = function(out_dim, max_classes, d_model, num_blocks, nhead, dim_feedforward,
                        dropout, activation, norm_first, bias_free_ln, ssmax, recompute) {
    self$y_embed_icl <- if (max_classes > 0L) {
      ClassEmbedding(max_classes, d_model)
    } else {
      nn_linear(1L, d_model)
    }
    self$icl_blocks <- nn_module_list(
      purrr::map(seq_len(num_blocks), function(i) {
        TransformerBlock(
          embed_dim = d_model, num_heads = nhead, use_rope = FALSE, ssmax = ssmax
        )
      })
    )
    self$out_ln <- nn_layer_norm(d_model)
    self$out_mlp <- .get_mlp(d_model, d_model * 2L, out_dim)

    if (max_classes == 0L) {
      # Supposé défini ailleurs dans le package
      self$quantile_dist <- QuantileToDistribution(num_quantiles = out_dim)
    }
  },
  forward = function(emb, y_train, return_logits = TRUE, softmax_temperature = 0.9, mgr_config = NULL) {
    n_train <- y_train$shape[2L]
    n_rows <- emb$shape[2L] # Shape is now (B, T, icl_dim) thanks to RowInteractor fix

    y_emb <- self$y_embed_icl(y_train) # (B, train_size, icl_dim)
    # Add y embedding to training rows
    emb_train <- emb$narrow(2L, 1L, n_train)
    emb_train_added <- emb_train + y_emb

    if (n_train < n_rows) {
      emb_test <- emb$narrow(2L, n_train + 1L, n_rows - n_train)
      emb <- torch_cat(list(emb_train_added, emb_test), dim = 2L)
    } else {
      emb <- emb_train_added
    }

    n_icl_blocks <- length(self$icl_blocks)
    if (n_icl_blocks > 1L) {
      emb <- purrr::reduce(seq_len(n_icl_blocks - 1L), function(acc, i) {
        self$icl_blocks[[i]]$forward(acc, kv_max_idx = n_train)
      }, .init = emb)
    }

    # Final block: test attends to train only
    last_block <- self$icl_blocks[[n_icl_blocks]]
    emb_test <- emb[, (n_train + 1L):n_rows, ..]
    emb_train <- emb[, seq_len(n_train), ..]
    emb <- last_block(emb_test, emb_train)

    out <- self$out_mlp(self$out_ln(emb))

    if (self$max_classes > 0L && !return_logits) {
      out <- torch_nn_functional_softmax(out / softmax_temperature, dim = -1L)
    }
    out
  }
)

#' TabICL: A Tabular In-Context Learning Foundation Model
#'

#' TabICL: A Tabular In-Context Learning Foundation Model
#'
#' TabICL is a transformer-based architecture for in-context learning on tabular data
#' to make predictions without fine-tuning. It processes tabular data through three
#' sequential stages:
#'
#' \enumerate{
#'   \item Column-wise embedding creates distribution-aware embeddings.
#'   \item Row-wise interaction captures interactions between features within each row.
#'   \item Dataset-wise in-context learning to learn patterns from labeled examples
#'         and make predictions.
#' }
#'
#' This class is the underlying raw torch module for TabICL. It is not intended to be
#' used directly. Instead, use the classes from the top-level \code{tabicl} package
#' such as \code{TabICLClassifier} or \code{TabICLRegressor} that wrap this class to
#' include the necessary preprocessing of input features and postprocessing of
#' predictions.
#'
#' @description
#' TabICL is a transformer-based architecture for in-context learning on tabular data.
#' This class is the underlying raw torch module.
#'
#' @param max_classes Integer, default `10L`. 0 for regression.
#' @param num_quantiles Integer, default `999L`. Number of quantiles for regression.
#' @param embed_dim Integer, default `128L`. Model dimension.
#' @param col_n_block Integer, default `3L`.
#' @param col_n_head Integer, default `8L`.
#' @param col_n_cls Integer, default `128L`.
#' @param col_affine Logical, default `FALSE`.
#' @param col_feature_group Character, default `"same"`.
#' @param col_feature_group_size Integer, default `3L`.
#' @param col_target_aware Logical, default `TRUE`.
#' @param col_ssmax Character, default `"qassmax-mlp-elementwise"`.
#' @param row_n_block Integer, default `3L`.
#' @param row_n_head Integer, default `8L`.
#' @param row_n_cls Integer, default `4L`.
#' @param row_rope_base Float, default `100000`.
#' @param row_rope_interleaved Logical, default `FALSE`.
#' @param icl_n_block Integer, default `12L`.
#' @param icl_n_head Integer, default `8L`.
#' @param icl_ssmax Character, default `"qassmax-mlp-elementwise"`.
#' @param ff_factor Integer, default `2L`.
#' @param dropout Float, default `0`.
#' @param activation Character, default `"gelu"`.
#' @param norm_first Logical, default `TRUE`.
#' @param bias_free_ln Logical, default `FALSE`.
#' @param recompute Logical, default `FALSE`.
#'
#' @return An `nn_module` instance of class `TabICL`.
#'
#' @section Methods:
#' \subsection{Usage}{
#' \preformatted{
#' model <- TabICL$new(max_classes = 10L)
#' model$forward(X, y_train)
#' model$predict_stats(X, y_train, output_type = "mean")
#' model$forward_with_cache(X_train, y_train, store_cache = TRUE)
#' model$predict_stats_with_cache(X_test = X_test, use_cache = TRUE)
#' model$has_cache()
#' model$clear_cache()
#' }
#' }
#'
#' \subsection{\code{forward(X, y_train, d, embed_with_test, feature_shuffles,
#'   return_logits, softmax_temperature, inference_config)}}{
#' Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.
#' Dispatches to \code{train_forward()} in training mode and \code{inference_forward()}
#' in evaluation mode.
#' \itemize{
#'   \item \code{X}: Input tensor of shape \code{(B, TT, H)} where B is the number of tables,
#'     TT is the number of samples (rows), and H is the number of features (columns). The
#'     first \code{train_size} positions contain training samples, and the remaining positions
#'     contain test samples.
#'   \item \code{y_train}: Training labels of shape \code{(B, train_size)}.
#'   \item \code{d}: Optional tensor. The number of features per dataset. Used only in
#'     training mode.
#'   \item \code{embed_with_test}: Logical, default \code{FALSE}. If \code{TRUE}, allow
#'     training samples to attend to test samples during embedding.
#'   \item \code{feature_shuffles}: Optional list of integer vectors. Feature shuffle
#'     patterns for each table in the batch. Used only in inference mode.
#'   \item \code{return_logits}: Logical, default \code{TRUE}. If \code{TRUE}, return raw
#'     logits instead of probabilities. Used only in inference mode.
#'   \item \code{softmax_temperature}: Float, default \code{0.9}. Temperature for the softmax
#'     function. Used only in inference mode.
#'   \item \code{inference_config}: An \code{InferenceConfig} object. Used only in inference
#'     mode.
#' }
#' Returns a tensor. For training mode: predictions of shape \code{(B, test_size, out_dim)}.
#' For inference mode: logits or probabilities of shape \code{(B, test_size, num_classes)} for
#' classification, or predictions of shape \code{(B, test_size, num_quantiles)} for regression.
#' }
#'
#' \subsection{\code{predict_stats(X, y_train, output_type, alphas,
#'   embed_with_test, inference_config)}}{
#' Compute summary statistics from predicted quantiles. Only applicable for regression
#' tasks (\code{max_classes = 0}).
#' \itemize{
#'   \item \code{output_type}: Character string or character vector. Supported values:
#'     \code{"mean"}, \code{"variance"}, \code{"median"}, \code{"quantiles"},
#'     \code{"raw_quantiles"}. If a vector, returns a named list.
#'   \item \code{alphas}: Optional numeric vector. Probability levels for quantile output.
#'     Default: \code{c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}.
#' }
#' Returns a tensor (single output type) or a named list of tensors (multiple output types).
#' Output shapes: \code{"mean"}, \code{"variance"}, \code{"median"} return
#' \code{(B, test_size)}; \code{"quantiles"} returns \code{(B, test_size, len(alphas))};
#' \code{"raw_quantiles"} returns \code{(B, test_size, num_quantiles)}.
#' }
#'
#' \subsection{\code{forward_with_cache(X_train, y_train, X_test, return_logits,
#'   softmax_temperature, use_cache, store_cache, cache, cache_mode,
#'   inference_config)}}{
#' Forward pass with caching support for efficient inference. Two caching modes are
#' supported:
#' \itemize{
#'   \item \code{"kv"}: Cache KV projections from both column embedding and ICL transformer
#'     layers. Fastest inference but uses more memory.
#'   \item \code{"repr"}: Cache column embedding KV projections and row interaction outputs.
#'     Uses approximately 24x less memory for the ICL part, at the cost of re-running the
#'     ICL transformer.
#' }
#' Exactly one of \code{use_cache} or \code{store_cache} must be \code{TRUE}. When
#' \code{store_cache = TRUE}, requires \code{X_train} and \code{y_train}. When
#' \code{use_cache = TRUE}, requires \code{X_test} and a populated cache.
#' \itemize{
#'   \item \code{X_train}: Optional tensor of shape \code{(B, train_size, H)}. Required when
#'     \code{store_cache = TRUE}.
#'   \item \code{y_train}: Optional tensor of shape \code{(B, train_size)}. Required when
#'     \code{store_cache = TRUE}.
#'   \item \code{X_test}: Optional tensor of shape \code{(B, test_size, H)}. Required when
#'     \code{use_cache = TRUE}.
#'   \item \code{return_logits}: Logical, default \code{TRUE}.
#'   \item \code{softmax_temperature}: Float, default \code{0.9}.
#'   \item \code{use_cache}: Logical, default \code{FALSE}.
#'   \item \code{store_cache}: Logical, default \code{TRUE}.
#'   \item \code{cache}: Optional \code{TabICLCache}. If provided, equivalent to setting
#'     \code{use_cache = TRUE} and \code{store_cache = FALSE}.
#'   \item \code{cache_mode}: Character string, default \code{"kv"}. Caching strategy.
#'   \item \code{inference_config}: Optional \code{InferenceConfig}.
#' }
#' Returns predictions of shape \code{(B, test_size, out_dim)}, or \code{NULL} if
#' \code{store_cache = TRUE} and \code{X_test} is not provided.
#' }
#'
#' \subsection{\code{predict_stats_with_cache(X_train, y_train, X_test, output_type,
#'   alphas, use_cache, store_cache, cache, cache_mode, inference_config)}}{
#' Compute summary statistics from predicted quantiles with KV caching. Only applicable
#' for regression tasks (\code{max_classes = 0}). Parameters and return value are the same
#' as \code{predict_stats()}, with additional caching parameters matching
#' \code{forward_with_cache()}. Returns \code{NULL} if \code{store_cache = TRUE} and
#' \code{X_test} is not provided.
#' }
#'
#' \subsection{\code{has_cache()}}{
#' Check if a valid cache is stored. Returns a logical value.
#' }
#'
#' \subsection{\code{clear_cache()}}{
#' Clear the stored cache. Called for its side effect; returns \code{NULL} invisibly.
#' }
#'
#' @seealso \code{\link{column_embedding}}, \code{\link{RowInteraction}},
#'   \code{\link{ICLearning}}, \code{\link{QuantileToDistribution}},
#'   \code{\link{TabICLCache}}, \code{\link{InferenceConfig}}
#'
#' @export
TabICLv2 <- torch::nn_module(
  "TabICLv2",

  initialize = function(
    max_classes = 10L, num_quantiles = 999L, embed_dim = 128L,
    col_n_block = 3L, col_n_head = 8L, col_n_cls = 128L,
    col_affine = FALSE, col_feature_group = "same", col_feature_group_size = 3L,
    col_target_aware = TRUE, col_ssmax = "qassmax-mlp-elementwise",
    row_n_block = 3L, row_n_head = 8L, row_n_cls = 4L,
    row_rope_base = 100000, row_rope_interleaved = FALSE,
    icl_n_block = 12L, icl_n_head = 8L, icl_ssmax = "qassmax-mlp-elementwise",
    ff_factor = 2L, dropout = 0, activation = "gelu",
    norm_first = TRUE, bias_free_ln = FALSE, recompute = FALSE
  ) {
    icl_dim <- as.integer(embed_dim * row_n_cls)

    if (max_classes == 0L) {
      if (num_quantiles <= 0L) {
        value_error("For regression (max_classes = 0), num_quantiles must be > 0.")
      }
      out_dim <- num_quantiles
    } else {
      out_dim <- max_classes
    }

    # Store hyperparameters for reference
    self$max_classes <- max_classes
    self$num_quantiles <- num_quantiles
    self$embed_dim <- embed_dim

    # Sub-modules
    self$col_embedder <- ColEmbedding(
      embed_dim = embed_dim, num_blocks = col_n_block, nhead = col_n_head,
      num_inds = col_n_cls, dim_feedforward = embed_dim * ff_factor,
      dropout = dropout, activation = activation, norm_first = norm_first,
      bias_free_ln = bias_free_ln, affine = col_affine, feature_group = col_feature_group,
      feature_group_size = col_feature_group_size, target_aware = col_target_aware,
      max_classes = max_classes, reserve_cls_tokens = row_n_cls,
      ssmax = col_ssmax, recompute = recompute
    )

    self$row_interactor <- RowInteractor(
      embed_dim = embed_dim, num_blocks = row_n_block, nhead = row_n_head,
      num_cls = row_n_cls, rope_base = row_rope_base, rope_interleaved = row_rope_interleaved,
      dropout = dropout, activation = activation, norm_first = norm_first,
      bias_free_ln = bias_free_ln, recompute = recompute
    )

    self$icl_predictor <- ICLPredictor(
      out_dim = out_dim, max_classes = max_classes, d_model = icl_dim,
      num_blocks = icl_n_block, nhead = icl_n_head, dim_feedforward = icl_dim * ff_factor,
      dropout = dropout, activation = activation, norm_first = norm_first,
      bias_free_ln = bias_free_ln, ssmax = icl_ssmax, recompute = recompute
    )

    self$._cache <- NULL
  },

  has_cache = function() {
    !is.null(self$._cache) && !self$._cache$is_empty()
  },

  clear_cache = function() {
    self$._cache <- NULL
    invisible(NULL)
  },

  train_forward = function(X, y_train, d = NULL, embed_with_test = FALSE) {
    train_size <- y_train$shape[2L]
    if (train_size > X$shape[2L]) {
      value_error("Number of training samples ({train_size}) exceeds total samples ({X$shape[2L]}).")
    }

    if (!is.null(d) && length(torch_unique_consecutive(d)[[1]]) == 1L && as.numeric(d[[1L]]) == X$shape[3L]) {
      d <- NULL
    }

    representations <- self$row_interactor(
      self$col_embedder(X, y_train = y_train, d = d, embed_with_test = embed_with_test),
      d = d
    )

    self$icl_predictor(representations, y_train = y_train)
  },

  inference_forward = function(
    X, y_train, feature_shuffles = NULL, embed_with_test = FALSE,
    return_logits = TRUE, softmax_temperature = 0.9, inference_config = NULL
  ) {
    train_size <- y_train$shape[2L]
    if (train_size > X$shape[2L]) {
      value_error("Number of training samples ({train_size}) exceeds total samples ({X$shape[2L]}).")
    }

    if (is.null(inference_config)) {
      inference_config <- inference_config()
    }

    representations <- self$row_interactor(
      self$col_embedder(
        X, y_train = y_train, embed_with_test = embed_with_test,
        feature_shuffles = feature_shuffles, mgr_config = inference_config$COL_CONFIG
      ),
      mgr_config = inference_config$ROW_CONFIG
    )

    self$icl_predictor(
      representations, y_train = y_train, return_logits = return_logits,
      softmax_temperature = softmax_temperature, mgr_config = inference_config$ICL_CONFIG
    )
  },

  forward = function(
    X, y_train, d = NULL, embed_with_test = FALSE, feature_shuffles = NULL,
    return_logits = TRUE, softmax_temperature = 0.9, inference_config = NULL
  ) {
    if (self$training) {
      self$train_forward(X, y_train, d = d, embed_with_test = embed_with_test)
    } else {
      self$inference_forward(
        X, y_train, feature_shuffles = feature_shuffles, embed_with_test = embed_with_test,
        return_logits = return_logits, softmax_temperature = softmax_temperature,
        inference_config = inference_config
      )
    }
  },

  predict_stats = function(X, y_train, output_type = "mean", alphas = NULL, embed_with_test = FALSE, inference_config = NULL) {
    if (self$max_classes != 0L) {
      value_error("predict_stats is only applicable for regression tasks (max_classes = 0).")
    }

    raw_quantiles <- self$inference_forward(X, y_train, embed_with_test = embed_with_test, inference_config = inference_config)
    dist <- self$icl_predictor$quantile_dist(raw_quantiles)
    mono_quantiles <- dist$quantiles

    if (is.character(output_type) && length(output_type) == 1L) output_type <- list(output_type)
    results <- list()

    if ("mean" %in% output_type) results[["mean"]] <- mono_quantiles$mean(dim = -1L)
    if ("variance" %in% output_type) results[["variance"]] <- mono_quantiles$var(dim = -1L)
    if ("median" %in% output_type) {
      results[["median"]] <- dist$icdf(alpha = torch_tensor(0.5, device = mono_quantiles$device, dtype = mono_quantiles$dtype))
    }
    if ("quantiles" %in% output_type) {
      if (is.null(alphas)) alphas <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
      results[["quantiles"]] <- dist$icdf(alpha = torch_tensor(alphas, device = mono_quantiles$device, dtype = mono_quantiles$dtype))
    }
    if ("raw_quantiles" %in% output_type) results[["raw_quantiles"]] <- mono_quantiles

    if (length(output_type) == 1L) return(results[[output_type[[1L]]]])
    results
  },

  forward_with_cache = function(
    X_train = NULL, y_train = NULL, X_test = NULL, return_logits = TRUE,
    softmax_temperature = 0.9, use_cache = FALSE, store_cache = TRUE,
    cache = NULL, cache_mode = "kv", inference_config = NULL
  ) {
    if (!is.null(cache)) {
      use_cache <- TRUE
      store_cache <- FALSE
      self$._cache <- cache
    }

    if (use_cache == store_cache) {
      value_error("Exactly one of use_cache or store_cache must be TRUE.")
    }
    if (!(cache_mode %in% c("kv", "repr"))) {
      value_error("cache_mode must be 'kv' or 'repr', got '{cache_mode}'.")
    }
    if (is.null(inference_config)) inference_config <- inference_config()

    if (store_cache) {
      if (is.null(X_train) || is.null(y_train)) {
        value_error("X_train and y_train are required when store_cache = TRUE.")
      }
      num_classes <- if (self$max_classes > 0L) length(torch_unique(y_train[1L, , drop = FALSE])) else 0L
      self$._cache <- TabICLCache(train_shape = X_train$shape, num_classes = num_classes)

      X <- if (is.null(X_test)) X_train else torch_cat(list(X_train, X_test), dim = 1L)
    }

    if (use_cache) {
      if (is.null(X_test)) value_error("X_test is required when use_cache = TRUE.")
      if (is.null(self$._cache) || self$._cache$is_empty()) {
        runtime_error("No cache available. Call with store_cache = TRUE first.")
      }
      X <- X_test
      y_train <- NULL
    }

    representations <- self$row_interactor(
      self$col_embedder$forward_with_cache(
        X, col_cache = self$._cache$col_cache, y_train = y_train,
        use_cache = use_cache, store_cache = store_cache, mgr_config = inference_config$COL_CONFIG
      ),
      mgr_config = inference_config$ROW_CONFIG
    )

    if (cache_mode == "repr") {
      if (store_cache) {
        train_size <- y_train$shape[2L]
        representations <- self$icl_predictor$prepare_repr_cache(representations, y_train)
        self$._cache$row_repr <- representations[, seq_len(train_size), .., drop = FALSE]
        if (is.null(X_test)) return(NULL)
      } else {
        train_repr <- self$._cache$row_repr
        train_size <- train_repr$shape[2L]
        representations <- torch_cat(list(train_repr$to(device = representations$device), representations), dim = 1L)
      }
      out <- self$icl_predictor$forward_with_repr_cache(
        representations, train_size = train_size, num_classes = self$._cache$num_classes,
        return_logits = return_logits, softmax_temperature = softmax_temperature,
        mgr_config = inference_config$ICL_CONFIG
      )
    } else {
      out <- self$icl_predictor$forward_with_cache(
        representations, icl_cache = self$._cache$icl_cache, y_train = y_train,
        num_classes = self$._cache$num_classes, return_logits = return_logits,
        softmax_temperature = softmax_temperature, use_cache = use_cache,
        store_cache = store_cache, mgr_config = inference_config$ICL_CONFIG
      )
      if (is.null(X_test)) return(NULL)
    }
    out
  },

  predict_stats_with_cache = function(
    X_train = NULL, y_train = NULL, X_test = NULL, output_type = "mean",
    alphas = NULL, use_cache = FALSE, store_cache = TRUE, cache = NULL,
    cache_mode = "kv", inference_config = NULL
  ) {
    if (self$max_classes != 0L) {
      value_error("predict_stats_with_cache is only applicable for regression tasks.")
    }

    raw_quantiles <- self$forward_with_cache(
      X_train = X_train, y_train = y_train, X_test = X_test, use_cache = use_cache,
      store_cache = store_cache, cache = cache, cache_mode = cache_mode,
      inference_config = inference_config
    )

    if (is.null(raw_quantiles)) return(NULL)

    dist <- self$icl_predictor$quantile_dist(raw_quantiles)
    mono_quantiles <- dist$quantiles

    if (is.character(output_type) && length(output_type) == 1L) output_type <- list(output_type)
    results <- list()

    if ("mean" %in% output_type) results[["mean"]] <- mono_quantiles$mean(dim = -1L)
    if ("variance" %in% output_type) results[["variance"]] <- mono_quantiles$var(dim = -1L)
    if ("median" %in% output_type) {
      results[["median"]] <- dist$icdf(alpha = torch_tensor(0.5, device = mono_quantiles$device, dtype = mono_quantiles$dtype))
    }
    if ("quantiles" %in% output_type) {
      if (is.null(alphas)) alphas <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
      results[["quantiles"]] <- dist$icdf(alpha = torch_tensor(alphas, device = mono_quantiles$device, dtype = mono_quantiles$dtype))
    }
    if ("raw_quantiles" %in% output_type) results[["raw_quantiles"]] <- mono_quantiles

    if (length(output_type) == 1L) return(results[[output_type[[1L]]]])
    results
  }
)
