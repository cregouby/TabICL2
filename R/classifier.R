#' @importFrom torch torch_tensor with_no_grad
#' @importFrom R6 R6Class
#' @importFrom cli cli_inform cli_warn
#' @importFrom utils download.file
#' @keywords internal
NULL

.ckpt_v1   <- "tabicl-classifier-v1-20250208.ckpt"
.ckpt_v1_1 <- "tabicl-classifier-v1.1-20250506.ckpt"
.ckpt_v2   <- "tabicl-classifier-v2-20260212.ckpt"

.hf_download <- function(filename, dest_path = NULL, allow_auto_download = TRUE) {
  repo_id  <- "jingang/TabICL"
  url      <- paste0("https://huggingface.co/", repo_id, "/resolve/main/", filename)

  if (is.null(dest_path)) {
    cache_dir <- file.path(path.expand("~"), ".cache", "tabicl")
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
    dest_path <- file.path(cache_dir, filename)
  }

  if (file.exists(dest_path)) return(dest_path)

  if (!allow_auto_download) {
    value_error(
      "Checkpoint '{filename}' not found and automatic download is disabled."
    )
  }

  cli_inform("Downloading checkpoint '{filename}' from Hugging Face Hub ({repo_id}).")
  download.file(url, destfile = dest_path, mode = "wb", quiet = TRUE)
  dest_path
}

#' TabICL Classifier
#'
#' Tabular In-Context Learning (TabICL) classifier with an ensemble of
#' transformed dataset views.
#'
#' @param n_estimators Number of ensemble members, default \code{8L}.
#' @param norm_methods Normalization methods (character vector or \code{NULL}
#'   for \code{c("none", "power")}).
#' @param feat_shuffle_method Feature permutation strategy: \code{"latin"},
#'   \code{"shift"}, \code{"random"}, or \code{"none"}.
#' @param class_shuffle_method Class label permutation strategy (same options).
#' @param outlier_threshold Z-score threshold for outlier clipping.
#' @param softmax_temperature Temperature \eqn{\tau} for softmax.
#' @param average_logits Average logits (TRUE) or probabilities (FALSE).
#' @param support_many_classes Enable many-class support when classes exceed
#'   \code{max_classes}.
#' @param batch_size Ensemble batch size; \code{NULL} processes all at once.
#' @param kv_cache Cache mode: \code{FALSE}, \code{TRUE}/\code{"kv"}, or
#'   \code{"repr"}.
#' @param model_path Path to checkpoint file, or \code{NULL} to use HF Hub
#'   cache.
#' @param allow_auto_download Download checkpoint automatically if missing.
#' @param checkpoint_version Checkpoint filename on HF Hub.
#' @param device Device string or \code{torch_device}, or \code{NULL} for auto.
#' @param use_amp Automatic mixed precision: \code{"auto"}, \code{TRUE}, or
#'   \code{FALSE}.
#' @param use_fa3 Flash Attention 3: \code{"auto"}, \code{TRUE}, or
#'   \code{FALSE}.
#' @param offload_mode Column embedding offload mode.
#' @param disk_offload_dir Directory for disk offloading.
#' @param random_state RNG seed.
#' @param n_jobs CPU thread count for inference on CPU (\code{NULL} = default).
#' @param verbose Print progress messages.
#' @param inference_config \code{InferenceConfig} object or named list, or
#'   \code{NULL}.
#'
#' @export
TabICLClassifier <- R6::R6Class(
  "TabICLClassifier",
  inherit = TabICLBaseEstimator,

  public = list(

    #' @field n_estimators Number of ensemble members.
    n_estimators         = 8L,
    #' @field norm_methods Normalisation methods.
    norm_methods         = NULL,
    #' @field feat_shuffle_method Feature permutation strategy.
    feat_shuffle_method  = "latin",
    #' @field class_shuffle_method Class permutation strategy.
    class_shuffle_method = "shift",
    #' @field outlier_threshold Outlier Z-score threshold.
    outlier_threshold    = 4.0,
    #' @field softmax_temperature Softmax temperature.
    softmax_temperature  = 0.9,
    #' @field average_logits Average logits (TRUE) or probabilities (FALSE).
    average_logits       = TRUE,
    #' @field support_many_classes Enable many-class support.
    support_many_classes = TRUE,
    #' @field batch_size Ensemble batch size.
    batch_size           = 8L,
    #' @field kv_cache KV cache mode.
    kv_cache             = FALSE,
    #' @field model_path Path to checkpoint file.
    model_path           = NULL,
    #' @field allow_auto_download Allow automatic download.
    allow_auto_download  = TRUE,
    #' @field checkpoint_version Checkpoint filename.
    checkpoint_version   = "tabicl-classifier-v2-20260212.ckpt",
    #' @field n_jobs CPU thread count.
    n_jobs               = NULL,
    #' @field random_state RNG seed.
    random_state         = 42L,

    # --- Fitted attributes (set by $fit()) ---
    #' @field classes_ Sorted class labels seen during \code{$fit()}.
    classes_             = NULL,
    #' @field n_classes_ Number of classes.
    n_classes_           = NULL,
    #' @field n_features_in_ Number of features after encoding.
    n_features_in_       = NULL,
    #' @field n_samples_in_ Number of training samples.
    n_samples_in_        = NULL,
    #' @field feature_names_in_ Column names when X is a data frame, else NULL.
    feature_names_in_    = NULL,
    #' @field X_encoder_ Fitted \code{TransformToNumerical}.
    X_encoder_           = NULL,
    #' @field ensemble_generator_ Fitted \code{EnsembleGenerator}.
    ensemble_generator_  = NULL,
    #' @field model_ Loaded \code{TabICL} nn_module.
    model_               = NULL,
    #' @field model_path_ Path to the loaded checkpoint file.
    model_path_          = NULL,
    #' @field model_config_ Config list from the checkpoint.
    model_config_        = NULL,
    #' @field cache_mode_ Resolved caching mode (\code{"kv"}, \code{"repr"}, or \code{NULL}).
    cache_mode_          = NULL,
    #' @field model_kv_cache_ Pre-computed KV caches keyed by norm method, or \code{NULL}.
    model_kv_cache_      = NULL,

    #' @description Create a new \code{TabICLClassifier}.
    initialize = function(
      n_estimators         = 8L,
      norm_methods         = NULL,
      feat_shuffle_method  = "latin",
      class_shuffle_method = "shift",
      outlier_threshold    = 4.0,
      softmax_temperature  = 0.9,
      average_logits       = TRUE,
      support_many_classes = TRUE,
      batch_size           = 8L,
      kv_cache             = FALSE,
      model_path           = NULL,
      allow_auto_download  = TRUE,
      checkpoint_version   = "tabicl-classifier-v2-20260212.ckpt",
      device               = NULL,
      use_amp              = "auto",
      use_fa3              = "auto",
      offload_mode         = "auto",
      disk_offload_dir     = NULL,
      random_state         = 42L,
      n_jobs               = NULL,
      verbose              = FALSE,
      inference_config     = NULL
    ) {
      super$initialize(
        device           = device,
        use_amp          = use_amp,
        use_fa3          = use_fa3,
        verbose          = as.logical(verbose),
        offload_mode     = offload_mode,
        disk_offload_dir = disk_offload_dir,
        inference_config = inference_config
      )
      self$n_estimators         <- as.integer(n_estimators)
      self$norm_methods         <- norm_methods
      self$feat_shuffle_method  <- feat_shuffle_method
      self$class_shuffle_method <- class_shuffle_method
      self$outlier_threshold    <- as.numeric(outlier_threshold)
      self$softmax_temperature  <- as.numeric(softmax_temperature)
      self$average_logits       <- as.logical(average_logits)
      self$support_many_classes <- as.logical(support_many_classes)
      self$batch_size           <- if (is.null(batch_size)) NULL else as.integer(batch_size)
      self$kv_cache             <- kv_cache
      self$model_path           <- model_path
      self$allow_auto_download  <- as.logical(allow_auto_download)
      self$checkpoint_version   <- checkpoint_version
      self$n_jobs               <- n_jobs
      self$random_state         <- random_state
      invisible(self)
    },

    #' @description Fit the classifier to training data.
    #'
    #' Prepares the model for prediction by encoding class labels, transforming
    #' features, fitting the ensemble generator, loading the pre-trained TabICL
    #' model, and optionally pre-computing KV caches.
    #'
    #' @param X array-like of shape \code{(n_samples, n_features)}.
    #' @param y Target class labels of length \code{n_samples}.
    #' @return Invisibly \code{self}.
    fit = function(X, y) {
      if (is.null(y))
        value_error("y must not be NULL.")

      if (is.data.frame(X) && !is.null(colnames(X)))
        self$feature_names_in_ <- colnames(X)

      self$n_samples_in_ <- nrow(X)

      self$.resolve_device()
      self$.build_inference_config()

      private$.load_model()
      self$model_$to(self$device_)

      # Encode class labels to 0-indexed integers
      self$classes_  <- sort(unique(y))
      y_encoded      <- match(y, self$classes_) - 1L
      self$n_classes_ <- length(self$classes_)

      if (self$n_classes_ > self$model_$max_classes) {
        if (!isFALSE(self$kv_cache) && !is.null(self$kv_cache))
          value_error(
            "KV caching is not supported when the number of classes ({self$n_classes_}) ",
            "exceeds the model's max_classes ({self$model_$max_classes})."
          )
        if (!self$support_many_classes)
          value_error(
            "Number of classes ({self$n_classes_}) exceeds model's max_classes ",
            "({self$model_$max_classes}). Enable support_many_classes = TRUE."
          )
        if (self$verbose > 0L)
          cli_inform(
            "Many-class strategy enabled: {self$n_classes_} classes exceed model's {self$model_$max_classes} native limit."
          )
      }

      self$X_encoder_ <- TransformToNumerical$new(verbose = self$verbose > 0L)
      X_num           <- self$X_encoder_$fit_transform(X)
      self$n_features_in_ <- ncol(X_num)

      self$ensemble_generator_ <- EnsembleGenerator$new(
        classification       = TRUE,
        n_estimators         = self$n_estimators,
        norm_methods         = self$norm_methods %||% c("none", "power"),
        feat_shuffle_method  = self$feat_shuffle_method,
        class_shuffle_method = self$class_shuffle_method,
        outlier_threshold    = self$outlier_threshold,
        random_state         = self$random_state
      )
      self$ensemble_generator_$fit(X_num, y_encoded)

      self$model_kv_cache_ <- NULL
      self$cache_mode_     <- NULL

      if (!isFALSE(self$kv_cache) && !is.null(self$kv_cache)) {
        self$cache_mode_ <- if (isTRUE(self$kv_cache) || identical(self$kv_cache, "kv")) {
          "kv"
        } else if (identical(self$kv_cache, "repr")) {
          "repr"
        } else {
          value_error(
            "Invalid kv_cache '{self$kv_cache}'. Use FALSE, TRUE, 'kv', or 'repr'."
          )
        }
        private$.build_kv_cache()
      }

      invisible(self)
    },

    #' @description Predict class probabilities.
    #'
    #' @param X array-like of shape \code{(n_samples, n_features)}. Columns
    #'   that are entirely \code{NA} are treated as masked features (SHAP).
    #' @return Numeric matrix of shape \code{(n_samples, n_classes)}.
    predict_proba = function(X) {
      if (is.null(self$classes_))
        value_error("Classifier not fitted. Call $fit() first.")
      if (is.atomic(X) && is.null(dim(X)))
        value_error("X must not be one-dimensional. Reshape your data.")

      has_kv_cache     <- !is.null(self$model_kv_cache_)
      has_training_data <- !is.null(self$ensemble_generator_$X_)
      if (!has_kv_cache && !has_training_data)
        runtime_error(
          "Cannot predict: no KV cache and no training data available."
        )

      old_threads <- NULL
      if (!is.null(self$n_jobs)) {
        old_threads <- torch::torch_get_num_threads()
        n_logical   <- parallel::detectCores(logical = TRUE)
        n_threads   <- if (self$n_jobs > 0L) {
          min(n_logical, self$n_jobs)
        } else {
          max(1L, n_logical + 1L + self$n_jobs)
        }
        torch::torch_set_num_threads(n_threads)
        on.exit(torch::torch_set_num_threads(old_threads), add = TRUE)
      }

      # Detect all-NaN columns (SHAP feature masking)
      feature_mask <- if (is.data.frame(X)) {
        vapply(X, function(col) all(is.na(col)), logical(1L))
      } else {
        arr <- as.matrix(X)
        apply(arr, 2L, function(col) all(is.na(col)))
      }
      if (!any(feature_mask)) feature_mask <- NULL

      if (!is.null(feature_mask)) {
        if (is.data.frame(X)) {
          X[, feature_mask] <- 0.0
        } else {
          X[, feature_mask] <- 0.0
        }
      }

      X_num     <- self$X_encoder_$transform(X)
      use_cache <- has_kv_cache && is.null(feature_mask)

      all_outputs <- list()

      if (use_cache) {
        test_data <- self$ensemble_generator_$transform(X_num, mode = "test")
        for (method in names(test_data)) {
          Xs_test  <- test_data[[method]]$X
          kv       <- self$model_kv_cache_[[method]]
          all_outputs <- c(all_outputs, private$.batch_forward_with_cache(Xs_test, kv))
        }
      } else {
        data <- self$ensemble_generator_$transform(
          X_num, mode = "both", feature_mask = feature_mask
        )
        for (method in names(data)) {
          Xs <- data[[method]]$X
          ys <- data[[method]]$y
          all_outputs <- c(all_outputs, private$.batch_forward(Xs, ys))
        }
      }

      # Class shuffles are stored in the same order as ensemble outputs
      class_shuffles <- unlist(
        self$ensemble_generator_$class_shuffles_, recursive = FALSE
      )
      n_estimators <- length(class_shuffles)

      n_test    <- nrow(all_outputs[[1L]])
      n_classes <- ncol(all_outputs[[1L]])
      avg       <- matrix(0.0, nrow = n_test, ncol = n_classes)

      for (i in seq_len(n_estimators)) {
        avg <- avg + all_outputs[[i]][, class_shuffles[[i]], drop = FALSE]
      }
      avg <- avg / n_estimators

      if (self$average_logits)
        avg <- softmax(avg, axis = 1L, temperature = self$softmax_temperature)

      avg / rowSums(avg)
    },

    #' @description Predict class labels.
    #'
    #' @param X array-like of shape \code{(n_samples, n_features)}.
    #' @return Vector of predicted class labels.
    predict = function(X) {
      proba <- self$predict_proba(X)
      y_idx <- max.col(proba, ties.method = "first") - 1L
      self$classes_[y_idx + 1L]
    }
  ),

  private = list(

    .load_model = function() {
      valid <- c(.ckpt_v1, .ckpt_v1_1, .ckpt_v2)
      filename <- self$checkpoint_version
      if (!filename %in% valid)
        value_error(
          "Invalid checkpoint_version '{filename}'. Valid options: {paste(valid, collapse = ', ')}."
        )

      path <- if (is.null(self$model_path)) {
        .hf_download(filename, allow_auto_download = self$allow_auto_download)
      } else {
        mp <- self$model_path
        if (file.exists(mp)) {
          mp
        } else if (self$allow_auto_download) {
          .hf_download(
            filename, dest_path = mp, allow_auto_download = TRUE
          )
        } else {
          value_error(
            "Checkpoint not found at '{mp}' and allow_auto_download is FALSE."
          )
        }
      }

      checkpoint <- torch::torch_load(path, device = "cpu")

      if (is.null(checkpoint[["config"]]))
        stop("Checkpoint missing 'config' key.", call. = FALSE)
      if (is.null(checkpoint[["state_dict"]]))
        stop("Checkpoint missing 'state_dict' key.", call. = FALSE)

      self$model_path_   <- path
      self$model_config_ <- checkpoint[["config"]]
      self$model_        <- do.call(TabICL, as.list(checkpoint[["config"]]))
      self$model_$load_state_dict(checkpoint[["state_dict"]])
      self$model_$eval()
      invisible(NULL)
    },

    .build_kv_cache = function() {
      train_data           <- self$ensemble_generator_$transform(X = NULL, mode = "train")
      self$model_kv_cache_ <- list()

      for (method in names(train_data)) {
        Xs        <- train_data[[method]]$X
        ys        <- train_data[[method]]$y
        n_total   <- dim(Xs)[1L]
        bs        <- self$batch_size %||% n_total
        n_batches <- ceiling(n_total / bs)
        caches    <- vector("list", n_batches)

        offset <- 0L
        for (b in seq_len(n_batches)) {
          start <- offset + 1L
          end   <- min(offset + bs, n_total)

          X_t <- torch_tensor(Xs[start:end, , , drop = FALSE])$float()$to(self$device_)
          y_t <- torch_tensor(ys[start:end, , drop = FALSE])$float()$to(self$device_)

          with_no_grad({
            self$model_$forward_with_cache(
              X_train          = X_t,
              y_train          = y_t,
              use_cache        = FALSE,
              store_cache      = TRUE,
              cache_mode       = self$cache_mode_,
              inference_config = self$inference_config_
            )
          })
          caches[[b]] <- self$model_$._cache
          self$model_$clear_cache()
          offset <- end
        }

        self$model_kv_cache_[[method]] <- tabicl_cache_concat(caches)
      }
      invisible(NULL)
    },

    # Returns a list of [test_size, n_classes] matrices, one per estimator.
    .batch_forward = function(Xs, ys) {
      n_total   <- dim(Xs)[1L]
      bs        <- self$batch_size %||% n_total
      n_batches <- ceiling(n_total / bs)
      outputs   <- vector("list", n_total)
      offset    <- 0L

      for (b in seq_len(n_batches)) {
        start <- offset + 1L
        end   <- min(offset + bs, n_total)
        cur   <- end - start + 1L

        X_t <- torch_tensor(Xs[start:end, , , drop = FALSE])$float()$to(self$device_)
        y_t <- torch_tensor(ys[start:end, , drop = FALSE])$float()$to(self$device_)

        with_no_grad({
          out <- self$model_(
            X                   = X_t,
            y_train             = y_t,
            return_logits       = self$average_logits,
            softmax_temperature = self$softmax_temperature,
            inference_config    = self$inference_config_
          )
        })

        out_arr <- as.array(out$float()$cpu())
        for (i in seq_len(cur)) {
          outputs[[start + i - 1L]] <- out_arr[i, , ]
        }
        offset <- end
      }
      outputs
    },

    # Returns a list of [test_size, n_classes] matrices, one per estimator.
    .batch_forward_with_cache = function(Xs_test, kv_cache) {
      n_total   <- dim(Xs_test)[1L]
      bs        <- self$batch_size %||% n_total
      n_batches <- ceiling(n_total / bs)
      outputs   <- vector("list", n_total)
      offset    <- 0L

      for (b in seq_len(n_batches)) {
        start <- offset + 1L
        end   <- min(offset + bs, n_total)
        cur   <- end - start + 1L

        cache_sub <- kv_cache$slice_batch(start, end)
        X_t       <- torch_tensor(Xs_test[start:end, , , drop = FALSE])$float()$to(self$device_)

        with_no_grad({
          out <- self$model_$forward_with_cache(
            X_test              = X_t,
            cache               = cache_sub,
            return_logits       = self$average_logits,
            softmax_temperature = self$softmax_temperature,
            inference_config    = self$inference_config_
          )
        })

        out_arr <- as.array(out$float()$cpu())
        for (i in seq_len(cur)) {
          outputs[[start + i - 1L]] <- out_arr[i, , ]
        }
        offset <- end
      }
      outputs
    }
  )
)
