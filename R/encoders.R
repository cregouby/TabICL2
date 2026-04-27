#' @title Encoder
#' @name Encoder
#' @description Stack of multihead attention blocks.
#'
#' @param num_blocks Integer. Number of multihead attention blocks in the stack.
#' @param d_model Integer. Model dimension.
#' @param nhead Integer. Number of attention heads and should be a divisor of
#'   \code{d_model}.
#' @param dim_feedforward Integer. Dimension of the feedforward network in each
#'   block.
#' @param dropout Float, default \code{0}. Dropout probability.
#' @param activation Character string or unary function, default \code{"gelu"}.
#'   The activation function used in the feedforward network.
#' @param norm_first Logical, default \code{TRUE}. If \code{TRUE}, uses pre-norm
#'   architecture (LayerNorm before attention and feedforward).
#' @param bias_free_ln Logical, default \code{FALSE}. If \code{TRUE}, removes
#'   bias from all LayerNorm layers.
#' @param use_rope Logical, default \code{FALSE}. Whether to use rotary
#'   positional encoding.
#' @param rope_base Integer, default \code{100000L}. A base scaling factor for
#'   rotary position encoding.
#' @param rope_interleaved Logical, default \code{TRUE}. If \code{TRUE}, uses
#'   interleaved rotation where dimension pairs are (1,2), (3,4), etc.
#'   If \code{FALSE}, uses non-interleaved rotation where the embedding is split
#'   into first half and second half.
#' @param ssmax Logical or character string, default \code{FALSE}. Type of
#'   scalable softmax to use. If \code{TRUE}, equivalent to
#'   \code{"qassmax-mlp-elementwise"}. If \code{FALSE}, equivalent to
#'   \code{"none"}. If a string, uses the specified scalable softmax type.
#'   Options include: \code{"none"}, \code{"ssmax"}, \code{"ssmax-mlp"},
#'   \code{"ssmax-mlp-elementwise"}, \code{"qassmax-mlp"},
#'   \code{"qassmax-mlp-elementwise"}.
#' @param recompute Logical, default \code{FALSE}. If \code{TRUE}, uses gradient
#'   checkpointing to save memory at the cost of additional computation.
#'
#' @return An \code{nn_module} instance of class \code{Encoder}.
#'
#' @section Methods:
#' \subsection{Usage}{
#' \preformatted{
#' enc <- Encoder(num_blocks = 6L, d_model = 128L, nhead = 8L, dim_feedforward = 256L)
#' enc$forward(src, train_size = NULL)
#' enc$forward_with_cache(src, icl_cache, train_size, use_cache, store_cache)
#' }
#' }
#'
#' @seealso \code{\link{MultiheadAttentionBlock}}, \code{\link{RotaryEmbedding}},
#'   \code{\link{KVCache}}, \code{\link{KVCacheEntry}}
#'
#' @export
#' @importFrom torch nn_module nn_module_list
Encoder <- nn_module(
  "Encoder",
  initialize = function(
    num_blocks,
    d_model,
    nhead,
    dim_feedforward,
    dropout         = 0,
    activation      = "gelu",
    norm_first      = TRUE,
    bias_free_ln    = FALSE,
    use_rope        = FALSE,
    rope_base       = 100000L,
    rope_interleaved = TRUE,
    ssmax           = FALSE,
    recompute       = FALSE
  ) {
    if (d_model %% nhead != 0) {
      stop(
        paste0("d_model (", d_model, ") must be divisible by nhead (", nhead, ")."),
        call. = FALSE
      )
    }

    self$blocks <- nn_module_list(lapply(
      seq_len(num_blocks),
      function(.) {
        MultiheadAttentionBlock(
          d_model        = d_model,
          nhead          = nhead,
          dim_feedforward = dim_feedforward,
          dropout        = dropout,
          activation     = activation,
          norm_first     = norm_first,
          bias_free_ln   = bias_free_ln,
          ssmax          = ssmax
        )
      }
    ))

    if (use_rope) {
      self$rope <- RotaryEmbedding(
        dim        = d_model %/% nhead,
        theta      = rope_base,
        interleaved = rope_interleaved
      )
    } else {
      self$rope <- NULL
    }

    self$recompute <- recompute
  },

  #' @description Process input through the stacked blocks.
  #'
  #' @param src Input tensor of shape \code{(..., seq_len, d_model)}.
  #' @param train_size Positive integer or \code{NULL}. When provided, queries
  #'   attend only to the first \code{train_size} positions. Useful in the ICL
  #'   transformer where only training samples serve as context.
  #'
  #' @return Output tensor with same shape as \code{src}.
  forward = function(src, train_size = NULL) {
    out <- src
    for (i in seq_along(self$blocks)) {
      block <- self$blocks[[i]]
      if (self$recompute) {
        out <- checkpoint(
          function(x) block(q = x, train_size = train_size, rope = self$rope),
          out,
          use_reentrant = FALSE
        )
      } else {
        out <- block(q = out, train_size = train_size, rope = self$rope)
      }
    }
    out
  },

  #' @description Process input through the stacked blocks with KV caching
  #'   support.
  #'
  #' When \code{store_cache = TRUE}, this method processes the full sequence and
  #' stores K/V projections from training data (positions \code{1:train_size}) at
  #' each layer.
  #'
  #' When \code{use_cache = TRUE}, this method assumes \code{src} only contains
  #' test data and uses cached K/V from training data for attention at each layer.
  #'
  #' @param src Input tensor of shape \code{(..., seq_len, d_model)}.
  #' @param icl_cache A \code{KVCache} object for storing/retrieving K/V
  #'   projections per layer.
  #' @param train_size Positive integer or \code{NULL}. When provided, queries
  #'   attend only to the first \code{train_size} positions.
  #' @param use_cache Logical, default \code{FALSE}. Whether to use cached values
  #'   to avoid redundant computation.
  #' @param store_cache Logical, default \code{TRUE}. Whether to store computed
  #'   values in cache.
  #'
  #' @return Output tensor with same shape as \code{src}.
  #'
  #' @details Exactly one of \code{use_cache} or \code{store_cache} must be
  #'   \code{TRUE}. \code{train_size} must be provided when
  #'   \code{store_cache = TRUE}.
  forward_with_cache = function(
    src,
    icl_cache,
    train_size  = NULL,
    use_cache   = FALSE,
    store_cache = TRUE
  ) {
    if (use_cache == store_cache) {
      stop(
        "Exactly one of use_cache or store_cache must be TRUE.",
        call. = FALSE
      )
    }

    if (store_cache && is.null(train_size)) {
      stop(
        "train_size must be provided when store_cache = TRUE.",
        call. = FALSE
      )
    }

    out <- src
    for (layer_idx in seq_along(self$blocks)) {
      block <- self$blocks[[layer_idx]]
      if (use_cache) {
        # src is already test-only data; no train_size needed
        out <- block(
          q       = out,
          rope    = self$rope,
          cached_kv = icl_cache$kv[[layer_idx]]
        )
      } else {
        # Compute K/V for training data and store in cache
        result <- block(
          q         = out,
          train_size = train_size,
          rope      = self$rope,
          need_kv   = TRUE
        )
        out     <- result[[1L]]
        k_proj  <- result[[2L]]
        v_proj  <- result[[3L]]
        icl_cache$kv[[layer_idx]] <- KVCacheEntry(key = k_proj, value = v_proj)
      }
    }

    out
  }
)


#' @title SetTransformer
#' @name SetTransformer
#' @description Stack of induced self-attention blocks.
#'
#' A set transformer uses induced self-attention mechanism to efficiently process
#' variable-sized sets while maintaining permutation invariance.
#'
#' @param num_blocks Integer. Number of induced self-attention blocks in the
#'   stack.
#' @param d_model Integer. Model dimension.
#' @param nhead Integer. Number of attention heads and should be a divisor of
#'   \code{d_model}.
#' @param dim_feedforward Integer. Dimension of the feedforward network in each
#'   block.
#' @param num_inds Integer, default \code{16L}. Number of inducing points used
#'   in self-attention blocks.
#' @param dropout Float, default \code{0}. Dropout probability.
#' @param activation Character string or unary function, default \code{"gelu"}.
#'   The activation function used in the feedforward network.
#' @param norm_first Logical, default \code{TRUE}. If \code{TRUE}, uses pre-norm
#'   architecture (LayerNorm before attention and feedforward).
#' @param bias_free_ln Logical, default \code{FALSE}. If \code{TRUE}, removes
#'   bias from all LayerNorm layers.
#' @param ssmax Logical or character string, default \code{FALSE}. Type of
#'   scalable softmax to use in attention. Note that only the first attention
#'   layer of the induced self-attention blocks uses SSMax. If \code{TRUE},
#'   equivalent to \code{"qassmax-mlp-elementwise"}. If \code{FALSE}, equivalent
#'   to \code{"none"}. If a string, uses the specified scalable softmax type.
#'   Options include: \code{"none"}, \code{"ssmax"}, \code{"ssmax-mlp"},
#'   \code{"ssmax-mlp-elementwise"}, \code{"qassmax-mlp"},
#'   \code{"qassmax-mlp-elementwise"}.
#' @param recompute Logical, default \code{FALSE}. If \code{TRUE}, uses gradient
#'   checkpointing to save memory at the cost of additional computation.
#'
#' @return An \code{nn_module} instance of class \code{SetTransformer}.
#'
#' @section Methods:
#' \subsection{Usage}{
#' \preformatted{
#' st <- SetTransformer(num_blocks = 3L, d_model = 128L, nhead = 8L,
#'                      dim_feedforward = 256L, num_inds = 128L)
#' st$forward(src, train_size = NULL)
#' st$forward_with_cache(src, col_cache, train_size, use_cache, store_cache)
#' }
#' }
#'
#' @references Lee et al. "Set Transformer: A Framework for Attention-based
#'   Permutation-Invariant Neural Networks", ICML 2019.
#'
#' @seealso \code{\link{InducedSelfAttentionBlock}}, \code{\link{KVCache}},
#'   \code{\link{KVCacheEntry}}
#'
#' @export
set_transformer <- nn_module(
  "SetTransformer",

  initialize = function(
    num_blocks,
    d_model,
    nhead,
    dim_feedforward,
    num_inds       = 16L,
    dropout        = 0,
    activation     = "gelu",
    norm_first     = TRUE,
    bias_free_ln   = FALSE,
    ssmax          = FALSE,
    recompute      = FALSE
  ) {
    if (d_model %% nhead != 0) {
      stop(
        paste0("d_model (", d_model, ") must be divisible by nhead (", nhead, ")."),
        call. = FALSE
      )
    }

    self$blocks <- nn_module_list(lapply(
      seq_len(num_blocks),
      function(.) {
        induced_self_attention_block(
          d_model        = d_model,
          nhead          = nhead,
          dim_feedforward = dim_feedforward,
          num_inds       = num_inds,
          dropout        = dropout,
          activation     = activation,
          norm_first     = norm_first,
          bias_free_ln   = bias_free_ln,
          ssmax          = ssmax
        )
      }
    ))

    self$recompute <- recompute
  },

  #' @description Process input through the stacked induced self-attention
  #'   blocks.
  #'
  #' @param src Input tensor of shape \code{(..., seq_len, d_model)}.
  #' @param train_size Positive integer or \code{NULL}. Position to split the
  #'   input into training and test data. When provided, inducing points will
  #'   only attend to training data in the first attention stage of induced
  #'   self-attention blocks to prevent information leakage.
  #'
  #' @return Output tensor with same shape as \code{src}.
  forward = function(src, train_size = NULL) {
    out <- src
    for (i in seq_along(self$blocks)) {
      block <- self$blocks[[i]]
      if (self$recompute) {
        out <- checkpoint(
          function(x) block(x, train_size = train_size),
          out,
          use_reentrant = FALSE
        )
      } else {
        out <- block(out, train_size = train_size)
      }
    }
    out
  },

  #' @description Process input through the stacked ISAB blocks with KV caching
  #'   support.
  #'
  #' Each block has two attention stages:
  #' \enumerate{
  #'   \item Stage 1: Inducing points attend to training data, producing
  #'     \code{hidden}.
  #'   \item Stage 2: Input attends to \code{hidden}, producing the output.
  #' }
  #'
  #' We cache the K/V projections of \code{hidden} for Stage 2, which allows
  #' test samples to reuse the cached K/V without recomputing Stage 1.
  #'
  #' When \code{store_cache = TRUE}:
  #' \enumerate{
  #'   \item Runs Stage 1: inducing points attend to training data to produce
  #'     \code{hidden}.
  #'   \item Caches K/V projections of \code{hidden} for each block.
  #'   \item Runs Stage 2: all samples attend to \code{hidden}.
  #' }
  #'
  #' When \code{use_cache = TRUE}:
  #' \enumerate{
  #'   \item Skips Stage 1 (uses cached K/V from \code{hidden}).
  #'   \item Runs Stage 2: test samples attend to cached K/V.
  #' }
  #'
  #' @param src Input tensor of shape \code{(..., seq_len, d_model)}.
  #' @param col_cache A \code{KVCache} object for storing/retrieving K/V
  #'   projections of \code{hidden}.
  #' @param train_size Positive integer or \code{NULL}. Position to split the
  #'   input into training and test data. If storing cache, it must be provided
  #'   to ensure the cache is populated with training data correctly. If using
  #'   cache, it is ignored.
  #' @param use_cache Logical, default \code{FALSE}. Whether to use cached
  #'   values to avoid redundant computation.
  #' @param store_cache Logical, default \code{TRUE}. Whether to store computed
  #'   values in cache.
  #'
  #' @return Output tensor with same shape as \code{src}.
  #'
  #' @details Exactly one of \code{use_cache} or \code{store_cache} must be
  #'   \code{TRUE}. \code{train_size} must be provided when
  #'   \code{store_cache = TRUE}.
  forward_with_cache = function(
    src,
    col_cache,
    train_size  = NULL,
    use_cache   = FALSE,
    store_cache = TRUE
  ) {
    if (use_cache == store_cache) {
      stop(
        "Exactly one of use_cache or store_cache must be TRUE.",
        call. = FALSE
      )
    }

    if (store_cache && is.null(train_size)) {
      stop(
        "train_size must be provided when store_cache = TRUE.",
        call. = FALSE
      )
    }

    out <- src
    for (block_idx in seq_along(self$blocks)) {
      block <- self$blocks[[block_idx]]
      out <- block$forward_with_cache(
        out,
        col_cache,
        block_idx,
        train_size,
        use_cache,
        store_cache
      )
    }

    out
  }
)
