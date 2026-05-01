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
#' @param max_classes Integer, default \code{10L}. Determines the task type and output
#'   behavior:
#'   \itemize{
#'     \item If \code{max_classes = 0}: The model performs regression using quantile
#'       prediction.
#'     \item If \code{max_classes > 0}: The model performs classification. This value
#'       specifies the number of classes the model supports natively. If the number of
#'       classes in the dataset exceeds this value, mixed-radix ensembling is used during
#'       column-wise embedding and hierarchical classification is used during in-context
#'       learning.
#'   }
#' @param num_quantiles Integer, default \code{999L}. Number of quantiles to predict for
#'   regression tasks. Only used when \code{max_classes = 0}. The model directly predicts
#'   these quantile values.
#' @param embed_dim Integer, default \code{128L}. Model dimension used in the column / row
#'   embedding transformers. For the in-context learning transformer, the dimension is this
#'   value multiplied by the number of CLS tokens.
#' @param col_num_blocks Integer, default \code{3L}. Number of induced self-attention blocks
#'   in the column embedding transformer.
#' @param col_nhead Integer, default \code{8L}. Number of attention heads in the column
#'   embedding transformer.
#' @param col_num_inds Integer, default \code{128L}. Number of inducing points in the column
#'   embedding transformer.
#' @param col_affine Logical, default \code{FALSE}. If \code{TRUE}, computes embeddings as:
#'   \code{features * W + b}. If \code{FALSE}, directly uses the set transformer output as
#'   embeddings.
#' @param col_feature_group Logical or character string. Feature grouping mode:
#'   \itemize{
#'     \item \code{FALSE}: No grouping.
#'     \item \code{TRUE} or \code{"same"}: Group through circular permutation (output has
#'       same number of groups as features).
#'     \item \code{"valid"}: Group through padding and reshaping (output may have fewer
#'       groups).
#'   }
#'   Default: \code{"same"}.
#' @param col_feature_group_size Integer, default \code{3L}. Number of features per group
#'   when feature grouping is enabled.
#' @param col_target_aware Logical, default \code{TRUE}. If \code{TRUE}, incorporates target
#'   information into column-wise embeddings.
#' @param col_ssmax Logical or character string, default \code{"qassmax-mlp-elementwise"}.
#'   Type of scalable softmax to use in the column embedding transformer. Note that only the
#'   first attention layer of the induced self-attention blocks uses SSMax. If \code{TRUE},
#'   equivalent to \code{"qassmax-mlp-elementwise"}. If \code{FALSE}, equivalent to
#'   \code{"none"}. If a string, uses the specified scalable softmax type. Options include:
#'   \itemize{
#'     \item \code{"none"}: No scaling applied.
#'     \item \code{"ssmax"}: Scaled attention with learnable per-head parameter.
#'     \item \code{"ssmax-mlp"}: Uses MLP to compute scaling factors based on sequence length.
#'     \item \code{"ssmax-mlp-elementwise"}: Elementwise scaling per head dimension using MLP.
#'     \item \code{"qassmax-mlp"}: Query-aware scaling with base and query MLPs.
#'     \item \code{"qassmax-mlp-elementwise"}: Elementwise query-aware scaling.
#'   }
#' @param row_num_blocks Integer, default \code{3L}. Number of attention blocks in the row
#'   interaction transformer.
#' @param row_nhead Integer, default \code{8L}. Number of attention heads in the row
#'   interaction transformer.
#' @param row_num_cls Integer, default \code{4L}. Number of learnable CLS tokens used to
#'   aggregate feature information per row.
#' @param row_rope_base Float, default \code{100000}. Base scaling factor for rotary position
#'   encoding in the row interaction transformer.
#' @param row_rope_interleaved Logical, default \code{FALSE}. If \code{TRUE}, uses
#'   interleaved rotation where dimension pairs are (0,1), (2,3), etc. If \code{FALSE}, uses
#'   non-interleaved rotation where the embedding is split into first half and second half.
#' @param icl_num_blocks Integer, default \code{12L}. Number of transformer blocks in the
#'   in-context learning transformer.
#' @param icl_nhead Integer, default \code{8L}. Number of attention heads in the in-context
#'   learning transformer.
#' @param icl_ssmax Logical or character string, default \code{"qassmax-mlp-elementwise"}.
#'   Type of scalable softmax to use in the in-context learning transformer. Same options as
#'   \code{col_ssmax}.
#' @param ff_factor Integer, default \code{2L}. Expansion factor for feedforward networks
#'   across all components.
#' @param dropout Float, default \code{0}. Dropout probability across all components.
#' @param activation Character string or function, default \code{"gelu"}. Activation function
#'   used throughout the model.
#' @param norm_first Logical, default \code{TRUE}. If \code{TRUE}, uses pre-norm architecture
#'   across all components.
#' @param bias_free_ln Logical, default \code{FALSE}. If \code{TRUE}, removes bias from all
#'   LayerNorm layers.
#' @param recompute Logical, default \code{FALSE}. If \code{TRUE}, uses gradient checkpointing
#'   to save memory at the cost of additional computation.
#'
#' @return An \code{nn_module} instance of class \code{TabICL}.
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
#' @seealso \code{\link{ColEmbedding}}, \code{\link{RowInteraction}},
#'   \code{\link{ICLearning}}, \code{\link{QuantileToDistribution}},
#'   \code{\link{TabICLCache}}, \code{\link{InferenceConfig}}
#'
#' @export
TabICL <- nn_module(

  classname = "TabICL",

  # -------------------------------------------------------------------
  # Initialization
  # -------------------------------------------------------------------
  initialize = function(
    max_classes      = 10L,
    num_quantiles    = 999L,
    embed_dim        = 128L,
    col_num_blocks   = 3L,
    col_nhead        = 8L,
    col_num_inds     = 128L,
    col_affine       = FALSE,
    col_feature_group          = "same",
    col_feature_group_size     = 3L,
    col_target_aware           = TRUE,
    col_ssmax        = "qassmax-mlp-elementwise",
    row_num_blocks   = 3L,
    row_nhead        = 8L,
    row_num_cls      = 4L,
    row_rope_base    = 100000,
    row_rope_interleaved       = FALSE,
    icl_num_blocks   = 12L,
    icl_nhead        = 8L,
    icl_ssmax        = "qassmax-mlp-elementwise",
    ff_factor        = 2L,
    dropout          = 0,
    activation       = "gelu",
    norm_first       = TRUE,
    bias_free_ln     = FALSE,
    recompute        = FALSE
  ) {
    icl_dim <- as.integer(embed_dim * row_num_cls)  # CLS tokens are concatenated for ICL

    # -- Determine task type ---------------------------------------------------
    if (max_classes == 0L) {
      # Regression
      if (num_quantiles <= 0L) {
        value_error(
          "For regression (max_classes = 0), num_quantiles must be greater than 0.")
      }
      out_dim <- num_quantiles
      self$quantile_dist <- QuantileToDistribution(num_quantiles = num_quantiles)
    } else {
      # Classification
      out_dim <- max_classes
    }

    # -- Store hyperparameters -------------------------------------------------
    self$max_classes            <- max_classes
    self$num_quantiles          <- num_quantiles
    self$embed_dim              <- embed_dim
    self$col_num_blocks         <- col_num_blocks
    self$col_nhead              <- col_nhead
    self$col_num_inds           <- col_num_inds
    self$col_affine             <- col_affine
    self$col_feature_group      <- col_feature_group
    self$col_feature_group_size <- col_feature_group_size
    self$col_target_aware       <- col_target_aware
    self$col_ssmax              <- col_ssmax
    self$row_num_blocks         <- row_num_blocks
    self$row_nhead              <- row_nhead
    self$row_num_cls            <- row_num_cls
    self$row_rope_base          <- row_rope_base
    self$row_rope_interleaved   <- row_rope_interleaved
    self$icl_num_blocks         <- icl_num_blocks
    self$icl_nhead              <- icl_nhead
    self$icl_ssmax              <- icl_ssmax
    self$ff_factor              <- ff_factor
    self$dropout                <- dropout
    self$activation             <- activation
    self$norm_first             <- norm_first
    self$bias_free_ln           <- bias_free_ln

    # -- Sub-modules -----------------------------------------------------------
    self$col_embedder <- ColEmbedding(
      embed_dim           = embed_dim,
      num_blocks          = col_num_blocks,
      nhead               = col_nhead,
      num_inds            = col_num_inds,
      dim_feedforward     = embed_dim * ff_factor,
      dropout             = dropout,
      activation          = activation,
      norm_first          = norm_first,
      bias_free_ln        = bias_free_ln,
      affine              = col_affine,
      feature_group       = col_feature_group,
      feature_group_size  = col_feature_group_size,
      target_aware        = col_target_aware,
      max_classes         = max_classes,
      reserve_cls_tokens  = row_num_cls,
      ssmax               = col_ssmax,
      recompute           = recompute
    )

    self$row_interactor <- RowInteraction(
      embed_dim           = embed_dim,
      num_blocks          = row_num_blocks,
      nhead               = row_nhead,
      dim_feedforward     = embed_dim * ff_factor,
      num_cls             = row_num_cls,
      rope_base           = row_rope_base,
      rope_interleaved    = row_rope_interleaved,
      dropout             = dropout,
      activation          = activation,
      norm_first          = norm_first,
      bias_free_ln        = bias_free_ln,
      recompute           = recompute
    )

    self$icl_predictor <- ICLearning(
      out_dim            = out_dim,
      max_classes        = max_classes,
      d_model            = icl_dim,
      num_blocks         = icl_num_blocks,
      nhead              = icl_nhead,
      dim_feedforward    = icl_dim * ff_factor,
      dropout            = dropout,
      activation         = activation,
      norm_first         = norm_first,
      bias_free_ln       = bias_free_ln,
      ssmax              = icl_ssmax,
      recompute          = recompute
    )

    # KV cache for efficient inference
    self$._cache <- NULL
  },

  # -------------------------------------------------------------------
  # Cache helpers
  # -------------------------------------------------------------------

  has_cache = function() {
    #' @description Check if a valid cache is stored.
    #' @return A logical value: \code{TRUE} when a non-empty cache exists.
    !is.null(self$._cache) && !self$._cache$is_empty()
  },

  clear_cache = function() {
    #' @description Clear the stored cache.
    #' @return \code{NULL}, invisibly.
    self$._cache <- NULL
    invisible(NULL)
  },

  # -------------------------------------------------------------------
  # Training forward
  # -------------------------------------------------------------------

  train_forward = function(
    X,
    y_train,
    d               = NULL,
    embed_with_test = FALSE
  ) {
    #' @description Column-wise embedding -> row-wise interaction -> dataset-wise
    #'   in-context learning for training.
    #' @param X Input tensor of shape \code{(B, TT, H)}. The first
    #'   \code{train_size} positions contain training samples and the remaining
    #'   positions contain test samples.
    #' @param y_train Training labels of shape \code{(B, train_size)}.
    #' @param d Optional integer tensor. The number of features per dataset.
    #' @param embed_with_test Logical. If \code{TRUE}, allow training samples
    #'   to attend to test samples during embedding.
    #' @return Predictions tensor of shape \code{(B, test_size, out_dim)}.
    #'   For regression (\code{max_classes = 0}): \code{out_dim = num_quantiles}.
    #'   For classification (\code{max_classes > 0}): \code{out_dim = max_classes}.

    B <- X$shape[1L]
    TT <- X$shape[2L]
    H <- X$shape[3L]
    train_size <- y_train$shape[2L]

    if (train_size > TT) {
      stop(
        "Number of training samples exceeds total samples.",
        call. = FALSE
      )
    }

    # Check if d is provided and has the same length as the number of features
    if (
      !is.null(d) &&
      length(torch_unique(d)) == 1L &&
      as.numeric(d[[1L]]) == H
    ) {
      d <- NULL
    }

    # Column-wise embedding -> Row-wise interaction
    representations <- self$row_interactor(
      self$col_embedder(
        X,
        y_train        = y_train,
        d              = d,
        embed_with_test = embed_with_test
      ),
      d = d
    )

    # Dataset-wise in-context learning
    self$icl_predictor(representations, y_train = y_train)
  },

  # -------------------------------------------------------------------
  # Inference forward
  # -------------------------------------------------------------------

  inference_forward = function(
    X,
    y_train,
    feature_shuffles       = NULL,
    embed_with_test        = FALSE,
    return_logits          = TRUE,
    softmax_temperature    = 0.9,
    inference_config       = NULL
  ) {
    #' @description Column-wise embedding -> row-wise interaction -> dataset-wise
    #'   in-context learning for inference.
    #' @param X Input tensor of shape \code{(B, TT, H)}.
    #' @param y_train Training labels of shape \code{(B, train_size)}.
    #' @param feature_shuffles Optional list of integer vectors. A list of feature
    #'   shuffle patterns for each table in the batch. When provided, indicates that
    #'   \code{X} contains the same table with different feature orders.
    #' @param embed_with_test Logical, default \code{FALSE}.
    #' @param return_logits Logical, default \code{TRUE}. If \code{TRUE}, return raw
    #'   logits instead of probabilities.
    #' @param softmax_temperature Float, default \code{0.9}.
    #' @param inference_config An \code{InferenceConfig} object.
    #' @return For regression: predictions of shape \code{(B, test_size, num_quantiles)}.
    #'   For classification: logits or probabilities of shape
    #'   \code{(B, test_size, num_classes)}.

    train_size <- y_train$shape[2L]

    if (train_size > X$shape[2L]) {
      stop(
        "Number of training samples exceeds total samples.",
        call. = FALSE
      )
    }

    if (is.null(inference_config)) {
      inference_config <- inference_config()
    }

    # Column-wise embedding -> Row-wise interaction
    representations <- self$row_interactor(
      self$col_embedder(
        X,
        y_train           = y_train,
        embed_with_test   = embed_with_test,
        feature_shuffles  = feature_shuffles,
        mgr_config        = inference_config$COL_CONFIG
      ),
      mgr_config = inference_config$ROW_CONFIG
    )

    # Dataset-wise in-context learning
    self$icl_predictor(
      representations,
      y_train            = y_train,
      return_logits      = return_logits,
      softmax_temperature = softmax_temperature,
      mgr_config         = inference_config$ICL_CONFIG
    )
  },

  # -------------------------------------------------------------------
  # Main forward (dispatches to train or inference)
  # -------------------------------------------------------------------

  forward = function(
    X,
    y_train,
    d                      = NULL,
    embed_with_test        = FALSE,
    feature_shuffles       = NULL,
    return_logits          = TRUE,
    softmax_temperature    = 0.9,
    inference_config       = NULL
  ) {
    #' @description Column-wise embedding -> row-wise interaction -> dataset-wise
    #'   in-context learning. Dispatches to \code{train_forward()} in training mode
    #'   and \code{inference_forward()} in evaluation mode.
    #' @param X Input tensor of shape \code{(B, TT, H)}.
    #' @param y_train Training labels of shape \code{(B, train_size)}.
    #' @param d Optional tensor. Used only in training mode.
    #' @param embed_with_test Logical, default \code{FALSE}.
    #' @param feature_shuffles Optional list of integer vectors. Used only in
    #'   inference mode.
    #' @param return_logits Logical, default \code{TRUE}. Used only in inference mode.
    #' @param softmax_temperature Float, default \code{0.9}. Used only in inference
    #'   mode.
    #' @param inference_config An \code{InferenceConfig} object. Used only in
    #'   inference mode.
    #' @return A tensor whose shape depends on mode and task type (see
    #'   \code{train_forward()} and \code{inference_forward()}).

    if (self$training) {
      out <- self$train_forward(
        X, y_train,
        d               = d,
        embed_with_test = embed_with_test
      )
    } else {
      out <- self$inference_forward(
        X,
        y_train,
        feature_shuffles    = feature_shuffles,
        embed_with_test     = embed_with_test,
        return_logits       = return_logits,
        softmax_temperature = softmax_temperature,
        inference_config    = inference_config
      )
    }

    out
  },

  # -------------------------------------------------------------------
  # Predict stats (regression only)
  # -------------------------------------------------------------------

  predict_stats = function(
    X,
    y_train,
    output_type        = "mean",
    alphas             = NULL,
    embed_with_test    = FALSE,
    inference_config   = NULL
  ) {
    #' @description Compute summary statistics from predicted quantiles. Only
    #'   applicable for regression tasks (\code{max_classes = 0}).
    #' @param X Input tensor of shape \code{(B, TT, H)}.
    #' @param y_train Training labels of shape \code{(B, train_size)}.
    #' @param output_type Character string or character vector determining the
    #'   type of output. Supported values:
    #'   \itemize{
    #'     \item \code{"mean"}: Mean of the predicted quantiles.
    #'     \item \code{"variance"}: Variance of the predicted quantiles.
    #'     \item \code{"median"}: Median via inverse CDF interpolation.
    #'     \item \code{"quantiles"}: Specific quantiles via inverse CDF.
    #'     \item \code{"raw_quantiles"}: The raw (monotonised) quantile tensor.
    #'   }
    #'   If a vector of length > 1, returns a named list.
    #' @param alphas Optional numeric vector of probability levels. Only used when
    #'   \code{"quantiles"} is in \code{output_type}. Default:
    #'   \code{c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}.
    #' @param embed_with_test Logical, default \code{FALSE}.
    #' @param inference_config An \code{InferenceConfig} object.
    #' @return A tensor (single output type) or a named list of tensors.

    if (self$max_classes != 0L) {
      stop(
        "predict_stats is only applicable for regression tasks.",
        call. = FALSE
      )
    }

    raw_quantiles <- self$inference_forward(
      X, y_train,
      embed_with_test = embed_with_test,
      inference_config = inference_config
    ) # shape: (B, test_size, num_quantiles)

    dist <- self$quantile_dist(raw_quantiles)
    raw_quantiles <- dist$quantiles  # dist ensures quantiles are monotonic

    # Normalise output_type to a list
    if (is.character(output_type) && length(output_type) == 1L) {
      output_type <- list(output_type)
    }
    results <- list()

    if ("mean" %in% output_type) {
      results[["mean"]] <- raw_quantiles$mean(dim = -1L)
    }
    if ("variance" %in% output_type) {
      results[["variance"]] <- raw_quantiles$var(dim = -1L)
    }
    if ("median" %in% output_type) {
      results[["median"]] <- dist$icdf(
        alpha = torch_tensor(
          0.5,
          device = raw_quantiles$device,
          dtype  = raw_quantiles$dtype
        )
      )
    }
    if ("quantiles" %in% output_type) {
      if (is.null(alphas)) {
        alphas <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
      }
      results[["quantiles"]] <- dist$icdf(
        alpha = torch_tensor(
          alphas,
          device = raw_quantiles$device,
          dtype  = raw_quantiles$dtype
        )
      )
    }
    if ("raw_quantiles" %in% output_type) {
      results[["raw_quantiles"]] <- raw_quantiles
    }

    if (length(output_type) == 1L) {
      return(results[[output_type[[1L]]]])
    }

    results
  },

  # -------------------------------------------------------------------
  # Cached forward
  # -------------------------------------------------------------------

    #' @description Forward pass with caching support for efficient inference.
    #'   Enables caching of training data computations to speed up repeated
    #'   inference on the same training context.
    #' @param X_train Optional tensor of shape \code{(B, train_size, H)}.
    #'   Required when \code{store_cache = TRUE}.
    #' @param y_train Optional tensor of shape \code{(B, train_size)}.
    #'   Required when \code{store_cache = TRUE}.
    #' @param X_test Optional tensor of shape \code{(B, test_size, H)}.
    #'   Required when \code{use_cache = TRUE}.
    #' @param return_logits Logical, default \code{TRUE}.
    #' @param softmax_temperature Float, default \code{0.9}.
    #' @param use_cache Logical, default \code{FALSE}.
    #' @param store_cache Logical, default \code{TRUE}.
    #' @param cache Optional \code{TabICLCache}.
    #' @param cache_mode Character string, default \code{"kv"}.
    #' @param inference_config Optional \code{InferenceConfig}.
    #' @return Predictions tensor of shape \code{(B, test_size, out_dim)}, or
    #'   \code{NULL} if \code{store_cache = TRUE} and \code{X_test} is not
    #'   provided.
  forward_with_cache = function(
    X_train              = NULL,
    y_train              = NULL,
    X_test               = NULL,
    return_logits        = TRUE,
    softmax_temperature  = 0.9,
    use_cache            = FALSE,
    store_cache          = TRUE,
    cache                = NULL,
    cache_mode           = "kv",
    inference_config     = NULL
  ) {

    if (!is.null(cache)) {
      use_cache   <- TRUE
      store_cache <- FALSE
      self$._cache <- cache
    }

    if (use_cache == store_cache) {
      stop(
        "Exactly one of use_cache or store_cache must be TRUE.",
        call. = FALSE
      )
    }

    if (!(cache_mode %in% c("kv", "repr"))) {
      stop(
        paste0("cache_mode must be 'kv' or 'repr', got '", cache_mode, "'."),
        call. = FALSE
      )
    }

    if (is.null(inference_config)) {
      inference_config <- inference_config()
    }

    # Auto-detect cache mode from cache contents
    if (
      use_cache &&
      !is.null(self$._cache) &&
      !is.null(self$._cache$cache_type) &&
      self$._cache$cache_type == "repr"
    ) {
      cache_mode <- "repr"
    }

    if (store_cache) {
      if (is.null(X_train) || is.null(y_train)) {
        stop(
          "X_train and y_train are required when store_cache = TRUE.",
          call. = FALSE
        )
      }

      # Initialise cache based on training data
      num_classes <- if (self$max_classes > 0L) {
        length(torch_unique(y_train[1L, , drop = FALSE]))
      } else {
        0L
      }
      self$._cache <- TabICLCache(
        train_shape = X_train$shape,
        num_classes = num_classes
      )

      X <- if (is.null(X_test)) X_train else {
        torch_cat(list(X_train, X_test), dim = 1L)
      }
    }

    if (use_cache) {
      if (is.null(X_test)) {
        stop(
          "X_test is required when use_cache = TRUE.",
          call. = FALSE
        )
      }

      if (is.null(self$._cache) || self$._cache$is_empty()) {
        stop(
          "No cache available. Call with store_cache = TRUE first.",
          call. = FALSE
        )
      }

      X       <- X_test
      y_train <- NULL
    }

    # Column-wise embedding with cache support -> Row-wise interaction
    representations <- self$row_interactor(
      self$col_embedder$forward_with_cache(
        X,
        col_cache     = self$._cache$col_cache,
        y_train       = y_train,
        use_cache     = use_cache,
        store_cache   = store_cache,
        mgr_config    = inference_config$COL_CONFIG
      ),
      mgr_config = inference_config$ROW_CONFIG
    )

    # Dataset-wise in-context learning
    if (cache_mode == "repr") {
      if (store_cache) {
        train_size <- y_train$shape[2L]

        # Bake y_train into train portion of representations
        representations <- self$icl_predictor$prepare_repr_cache(
          representations, y_train
        )
        # Slice out train rows (R is 1-indexed: cols 1..train_size)
        self$._cache$row_repr <- representations[, 1:train_size, drop = FALSE]

        if (is.null(X_test)) {
          return(NULL)
        }
      } else {
        # Concatenate cached train representations with test representations
        train_repr  <- self$._cache$row_repr
        train_size  <- train_repr$shape[2L]
        representations <- torch_cat(
          list(
            train_repr$to(device = representations$device),
            representations
          ),
          dim = 1L
        )
      }

      out <- self$icl_predictor$forward_with_repr_cache(
        representations,
        train_size         = train_size,
        num_classes        = self$._cache$num_classes,
        return_logits      = return_logits,
        softmax_temperature = softmax_temperature,
        mgr_config         = inference_config$ICL_CONFIG
      )
    } else {
      out <- self$icl_predictor$forward_with_cache(
        representations,
        icl_cache          = self$._cache$icl_cache,
        y_train            = y_train,
        num_classes        = self$._cache$num_classes,
        return_logits      = return_logits,
        softmax_temperature = softmax_temperature,
        use_cache          = use_cache,
        store_cache        = store_cache,
        mgr_config         = inference_config$ICL_CONFIG
      )

      if (is.null(X_test)) {
        return(NULL)
      }
    }

    out
  },

  # -------------------------------------------------------------------
  # Predict stats with cache (regression only)
  # -------------------------------------------------------------------

  #' @description Compute summary statistics from predicted quantiles with KV
  #'   caching. Only applicable for regression tasks (\code{max_classes = 0}).
  #'   Delegates to \code{forward_with_cache()} and then applies the same
  #'   summary logic as \code{predict_stats()}.
  #' @param X_train Optional tensor of shape \code{(B, train_size, H)}.
  #' @param y_train Optional tensor of shape \code{(B, train_size)}.
  #' @param X_test Optional tensor of shape \code{(B, test_size, H)}.
  #' @param output_type Character string or character vector. See
  #'   \code{predict_stats()} for supported values.
  #' @param alphas Optional numeric vector of probability levels.
  #' @param use_cache Logical, default \code{FALSE}.
  #' @param store_cache Logical, default \code{TRUE}.
  #' @param cache Optional \code{TabICLCache}.
  #' @param cache_mode Character string, default \code{"kv"}.
  #' @param inference_config Optional \code{InferenceConfig}.
  #' @return A tensor, a named list of tensors, or \code{NULL} (when
  #'   \code{store_cache = TRUE} and \code{X_test} is not provided).
  predict_stats_with_cache = function(
    X_train              = NULL,
    y_train              = NULL,
    X_test               = NULL,
    output_type          = "mean",
    alphas               = NULL,
    use_cache            = FALSE,
    store_cache          = TRUE,
    cache                = NULL,
    cache_mode           = "kv",
    inference_config     = NULL
  ) {

    if (self$max_classes != 0L) {
      stop(
        "predict_stats_with_cache is only applicable for regression tasks.",
        call. = FALSE
      )
    }

    raw_quantiles <- self$forward_with_cache(
      X_train       = X_train,
      y_train       = y_train,
      X_test        = X_test,
      use_cache     = use_cache,
      store_cache   = store_cache,
      cache         = cache,
      cache_mode    = cache_mode,
      inference_config = inference_config
    )

    if (is.null(raw_quantiles)) {
      return(NULL)
    }

    dist <- self$quantile_dist(raw_quantiles)
    raw_quantiles <- dist$quantiles

    # Normalise output_type to a list
    if (is.character(output_type) && length(output_type) == 1L) {
      output_type <- list(output_type)
    }
    results <- list()

    if ("mean" %in% output_type) {
      results[["mean"]] <- raw_quantiles$mean(dim = -1L)
    }
    if ("variance" %in% output_type) {
      results[["variance"]] <- raw_quantiles$var(dim = -1L)
    }
    if ("median" %in% output_type) {
      results[["median"]] <- dist$icdf(
        alpha = torch_tensor(
          0.5,
          device = raw_quantiles$device,
          dtype  = raw_quantiles$dtype
        )
      )
    }
    if ("quantiles" %in% output_type) {
      if (is.null(alphas)) {
        alphas <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
      }
      results[["quantiles"]] <- dist$icdf(
        alpha = torch_tensor(
          alphas,
          device = raw_quantiles$device,
          dtype  = raw_quantiles$dtype
        )
      )
    }
    if ("raw_quantiles" %in% output_type) {
      results[["raw_quantiles"]] <- raw_quantiles
    }

    if (length(output_type) == 1L) {
      return(results[[output_type[[1L]]]])
    }

    results
  }
)
