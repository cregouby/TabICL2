#' @importFrom R6 R6Class
#' @importFrom torch torch_load torch_tensor with_no_grad
#' @importFrom torch torch_set_num_threads torch_get_num_threads
#' @importFrom magrittr %>%
NULL

#' Split an array along its first axis into roughly equal chunks.
#'
#' @param arr A matrix or array.
#' @param n `integer(1)` Number of chunks.
#'
#' @return A list of sub-arrays.
#' @keywords internal
.split_first_axis <- function(arr, n) {
  total <- dim(arr)[1L]
  if (n >= total) return(list(arr))

  sizes <- rep(total %/% n, n)
  remainder <- total %% n
  if (remainder > 0L) sizes[1L:remainder] <- sizes[1L:remainder] + 1L

  offset <- 1L
  lapply(sizes, function(size) {
    rows <- seq(offset, length.out = size)
    offset <<- offset + size
    ndim <- length(dim(arr))
    idx  <- as.list(rep(TRUE, ndim))
    idx[[1L]] <- rows
    idx$drop <- FALSE
    do.call(`[`, c(list(arr), idx))
  })
}

#' Concatenate a list of arrays along their first axis.
#'
#' @param arrays List of matrices or arrays with compatible dimensions.
#'
#' @return A single concatenated array.
#' @keywords internal
.concat_first_axis <- function(arrays) {
  if (length(arrays) == 0L) return(NULL)
  if (length(arrays) == 1L) return(arrays[[1L]])

  ndim <- length(dim(arrays[[1L]]))
  if (ndim <= 2L) {
    do.call(rbind, arrays)
  } else {
    do.call(abind::abind, c(arrays, list(along = 1L)))
  }
}

#' @title TabICLRegressor -- Tabular In-Context Learning Regressor
#'
#' @description scikit-learn-compatible regressor that applies TabICL
#'   to tabular data regression.  Uses an ensemble of transformed
#'   dataset views (different normalizations and feature permutations)
#'   to improve predictions.
#'
#' @section Parameters:
#' \describe{
#'   \item{`n_estimators`}{`integer(1)` Number of ensemble estimators.
#'     Default: \code{8L}.}
#'   \item{`norm_methods`}{`character` or \code{NULL}. Normalization
#'     methods: \code{"none"}, \code{"power"}, \code{"quantile"},
#'     \code{"quantile_rtdl"}, \code{"robust"}.  Default: \code{NULL}
#'     (uses \code{c("none", "power")}).}
#'   \item{`feat_shuffle_method`}{`character(1)` Feature permutation:
#'     \code{"none"}, \code{"shift"}, \code{"random"},
#'     \code{"latin"}.  Default: \code{"latin"}.}
#'   \item{`outlier_threshold`}{`double(1)` Z-score threshold for
#'     outlier clipping.  Default: \code{4.0}.}
#'   \item{`batch_size`}{`integer(1)` or \code{NULL}. Batch size for
#'     inference.  Default: \code{8L}.}
#'   \item{`kv_cache`}{`logical(1)` or `character(1)`. Caching mode:
#'     \code{FALSE} (none), \code{TRUE}/\code{"kv"} (full KV cache),
#'     \code{"repr"} (row representations).  Default: \code{FALSE}.}
#'   \item{`model_path`}{`character(1)` or \code{NULL}. Path to
#'     checkpoint.  Default: \code{NULL}.}
#'   \item{`allow_auto_download`}{`logical(1)`. Default: \code{TRUE}.}
#'   \item{`checkpoint_version`}{`character(1)`. Default:
#'     \code{"tabicl-regressor-v2-20260212.ckpt"}.}
#'   \item{`device`, `use_amp`, `use_fa3`, `offload_mode`,
#'     `disk_offload_dir`, `random_state`, `n_jobs`, `verbose`,
#'     `inference_config`}{See \code{\link{TabICLBaseEstimator}}.}
#' }
#'
#' @section Fitted Attributes:
#' \describe{
#'   \item{`y_scaler_`}{StandardScaler for target values.}
#'   \item{`X_encoder_`}{TransformToNumerical for features.}
#'   \item{`ensemble_generator_`}{Fitted EnsembleGenerator.}
#'   \item{`model_`}{Loaded TabICL model.}
#'   \item{`model_path_`}{Path to checkpoint file.}
#'   \item{`model_config_`}{Checkpoint config dict.}
#'   \item{`cache_mode_`}{Resolved cache mode or \code{NULL}.}
#'   \item{`model_kv_cache_`}{Named list of TabICLCache or \code{NULL}.}
#' }
#'
#' @seealso \code{\link{TabICLBaseEstimator}}
#' @export
TabICLRegressor <- R6Class(
  classname = "TabICLRegressor",
  inherit = TabICLBaseEstimator,

  public = list(
   #' @field n_estimators Number of ensemble estimators.
    n_estimators = 8L,

    #' @field norm_methods Normalization methods (character vector or NULL).
    norm_methods = NULL,

    #' @field feat_shuffle_method Feature permutation strategy.
    feat_shuffle_method = "latin",

    #' @field outlier_threshold Z-score threshold for outlier clipping.
    outlier_threshold = 4.0,

    #' @field batch_size Batch size for inference (integer or NULL).
    batch_size = 8L,

    #' @field kv_cache Caching mode (logical or string).
    kv_cache = FALSE,

    #' @field model_path Path to checkpoint file.
    model_path = NULL,

    #' @field allow_auto_download Whether to auto-download checkpoint.
    allow_auto_download = TRUE,

    #' @field checkpoint_version Checkpoint version string.
    checkpoint_version = "tabicl-regressor-v2-20260212.ckpt",

    #' @field y_scaler_ StandardScaler for target values (set by \code{fit}).
    y_scaler_ = NULL,

    #' @field model_ Loaded TabICL model (set by \code{fit}).
    model_ = NULL,

    #' @field model_path_ Resolved checkpoint path (set by \code{fit}).
    model_path_ = NULL,

    #' @field model_config_ Checkpoint config dict (set by \code{fit}).
    model_config_ = NULL,

    #' @field cache_mode_ Resolved cache mode string or NULL.
    cache_mode_ = NULL,

    #' @field model_kv_cache_ Named list of TabICLCache keyed by
    #'   normalization method, or \code{NULL}.
    model_kv_cache_ = NULL,

    #' @field X_encoder_ TransformToNumerical encoder (set by \code{fit}).
    X_encoder_ = NULL,

    #' @field ensemble_generator_ EnsembleGenerator (set by \code{fit}).
    ensemble_generator_ = NULL,

    #' @description Create a new TabICLRegressor.
    initialize = function(
      n_estimators        = 8L,
      norm_methods        = NULL,
      feat_shuffle_method = "latin",
      outlier_threshold   = 4.0,
      batch_size          = 8L,
      kv_cache            = FALSE,
      model_path          = NULL,
      allow_auto_download = TRUE,
      checkpoint_version  = "tabicl-regressor-v2-20260212.ckpt",
      device              = NULL,
      use_amp             = "auto",
      use_fa3             = "auto",
      offload_mode        = "auto",
      disk_offload_dir    = NULL,
      random_state        = 42L,
      n_jobs              = NULL,
      verbose             = FALSE,
      inference_config    = NULL
    ) {
      self$n_estimators        <- n_estimators
      self$norm_methods        <- norm_methods
      self$feat_shuffle_method <- feat_shuffle_method
      self$outlier_threshold   <- outlier_threshold
      self$batch_size          <- batch_size
      self$kv_cache            <- kv_cache
      self$model_path          <- model_path
      self$allow_auto_download <- allow_auto_download
      self$checkpoint_version  <- checkpoint_version
      self$n_jobs              <- n_jobs
      self$random_state        <- random_state

      # Pass base-class params through the parent initializer
      super$initialize(
        device           = device,
        use_amp          = use_amp,
        use_fa3          = use_fa3,
        offload_mode     = offload_mode,
        disk_offload_dir = disk_offload_dir,
        verbose          = verbose,
        inference_config = inference_config
      )

      invisible(self)
    },

    #' @description Fit the regressor to training data.
    #'
    #' Prepares the model for prediction by scaling targets, encoding
    #' features, fitting the ensemble generator, loading the pre-trained
    #' TabICL model, and optionally building KV caches.
    #'
    #' @param X `matrix` or `data.frame` of shape
    #'   \code{(n_samples, n_features)}.
    #' @param y `numeric` vector of length \code{n_samples}.
    #'
    #' @return \code{self}, invisibly.
    fit = function(X, y) {
      if (is.null(y)) {
        value_error(
          "This regressor requires y to be passed, but the target y is NULL."
        )
      }

      X <- validate_data(self, X, y, dtype = NULL, skip_check_array = TRUE)
      y <- validate_data(self, X, y, dtype = NULL, skip_check_array = TRUE)[[2L]]

      # Ensure y is numeric float32
      storage.mode(y) <- "single"

      # Flatten column-vector y
      if (is.matrix(y) && ncol(y) == 1L) {
        cli_warn(
          "A column-vector y was passed when a 1d array was expected. ",
          "Please change the shape of y to (n_samples, ), for example ",
          "using as.vector()."
        )
        y <- as.vector(y)
      }

      # Device setup
      self$.resolve_device()

      # Inference config
      self$n_samples_in_ <- nrow(X)
      self$.build_inference_config()

      # Load model
      self$.load_model()
      self$model_$to(self$device_)

      # Scale target values
      self$y_scaler_ <- StandardScaler()
      y_scaled <- as.vector(
        self$y_scaler_$fit_transform(matrix(y, ncol = 1L))
      )

      # Transform input features
      self$X_encoder_ <- TransformToNumerical(verbose = self$verbose)
      X <- self$X_encoder_$fit_transform(X)

      # Fit ensemble generator
      self$ensemble_generator_ <- EnsembleGenerator(
        classification       = FALSE,
        n_estimators         = self$n_estimators,
        norm_methods         = self$norm_methods %||% c("none", "power"),
        feat_shuffle_method  = self$feat_shuffle_method,
        outlier_threshold    = self$outlier_threshold,
        random_state         = self$random_state
      )
      self$ensemble_generator_$fit(X, y_scaled)

      # KV cache
      self$model_kv_cache_ <- NULL
      if (!identical(self$kv_cache, FALSE)) {
        self$cache_mode_ <- if (identical(self$kv_cache, TRUE) || identical(self$kv_cache, "kv")) {
          "kv"
        } else if (identical(self$kv_cache, "repr")) {
          "repr"
        } else {
          value_error(
            "Invalid kv_cache value '{self$kv_cache}'. Expected FALSE, TRUE, 'kv', or 'repr'."
          )
        }
        self$.build_kv_cache()
      }

      invisible(self)
    },

    #' Download a TabICL checkpoint manually
    #'
    #' Pre-download a model checkpoint to the cache directory. Useful for
    #' offline environments or pre-warming the cache.
    #'
    #' @param checkpoint_version Checkpoint filename (e.g., "tabicl-classifier-v2-20260212.ckpt")
    #' @param cache_dir Optional cache directory within ~/.cache/torch/ (default: tabicl)
    #' @param progress Show download progress
    #' @return Path to the downloaded file
    #' @export
    download_checkpoint = function(checkpoint_version, cache_dir = NULL, progress = TRUE) {
      info <- .get_checkpoint_info(checkpoint_version)
      .download_and_cache(
        url = info[1],
        filename = checkpoint_version,
        md5 = info[2],
        size_hint = info[3],
        cache_dir = cache_dir,
        progress = progress
      )
    },

    #' @description Load the pre-trained TabICL model from a checkpoint.
    #'
    #' Resolves the checkpoint source via \code{model_path} and
    #' \code{checkpoint_version}, downloading from Hugging Face Hub if
    #' necessary.
    #'
    #' @return \code{self}, invisibly.
    .load_model = function(progress = TRUE) {
      path <- .resolve_checkpoint_path(
        model_path = self$model_path,
        checkpoint_version = self$checkpoint_version,
        allow_auto_download = self$allow_auto_download,
        progress = progress && self$verbose
      )

      cli_inform("Loading checkpoint from {.file {path}}")

      checkpoint <- tryCatch(
        torch_load(path, device = "cpu"),
        error = function(e) {
          runtime_error("Failed to load checkpoint: {e$message}")
        }
      )

      if (is.null(checkpoint[["config"]])) {
        runtime_error("Checkpoint missing 'config' key.")
      }
      if (is.null(checkpoint[["state_dict"]])) {
        runtime_error("Checkpoint missing 'state_dict' key.")
      }

      self$model_path_   <- path
      self$model_config_ <- checkpoint[["config"]]

      self$model_ <- do.call(TabICL, as.list(checkpoint[["config"]]))
      self$model_$load_state_dict(checkpoint[["state_dict"]])
      self$model_$eval()

      cli_inform("Model loaded successfully with {length(checkpoint[['state_dict']])} parameters")
      invisible(NULL)
    },
    #' @description Pre-compute KV caches for training data across all
    #'   ensemble batches.
    .build_kv_cache = function() {
      train_data <- self$ensemble_generator_$transform(
        X = NULL, mode = "train"
      )
      self$model_kv_cache_ <- list()

      names(train_data) %>% lapply(function(norm_method) {
        pair <- train_data[[norm_method]]
        Xs   <- pair[[1L]]
        ys   <- pair[[2L]]

        batch_size <- self$batch_size %||% nrow(Xs)
        n_batches  <- ceiling(nrow(Xs) / batch_size)
        Xs_split   <- .split_first_axis(Xs, n_batches)
        ys_split   <- .split_first_axis(as.matrix(ys), n_batches)

        caches <- Map(list, Xs_split, ys_split) %>%
          lapply(function(p) {
            X_batch <- torch_tensor(as.array(p[[1L]]), dtype = torch_float32())$to(self$device_)
            y_batch <- torch_tensor(as.array(p[[2L]]), dtype = torch_float32())$to(self$device_)
            with_no_grad({
              self$model_$predict_stats_with_cache(
                X_train          = X_batch,
                y_train          = y_batch,
                use_cache        = FALSE,
                store_cache      = TRUE,
                cache_mode       = self$cache_mode_,
                inference_config = self$inference_config_
              )
            })
            self$model_$._cache
          })

        self$model_$clear_cache()
        self$model_kv_cache_[[norm_method]] <- tabicl_cache_concat(caches)
      })

      invisible(self)
    },

    #' @description Batched forward pass without KV cache.
    #'
    #' @param Xs `array(n_datasets, n_samples, n_features)`.
    #' @param ys `array(n_datasets, train_size)`.
    #' @param output_type `character` vector of output types.
    #' @param alphas `numeric` vector of quantile levels, or \code{NULL}.
    #'
    #' @return R array or named list of arrays.
    .batch_forward = function(Xs, ys, output_type, alphas = NULL) {
      batch_size <- self$batch_size %||% dim(Xs)[1L]
      n_batches  <- ceiling(dim(Xs)[1L] / batch_size)
      Xs_list    <- .split_first_axis(Xs, n_batches)
      ys_list    <- .split_first_axis(as.matrix(ys), n_batches)

      batch_outputs <- Map(list, Xs_list, ys_list) %>%
        lapply(function(pair) {
          X_batch <- torch_tensor(as.array(pair[[1L]]), dtype = torch_float32())$to(self$device_)
          y_batch <- torch_tensor(as.array(pair[[2L]]), dtype = torch_float32())$to(self$device_)
          with(torch_no_grad(), {
            self$model_$predict_stats(
              X_batch, y_batch,
              output_type      = output_type,
              alphas           = alphas,
              inference_config = self$inference_config_
            )
          })
        })

      # Extract and concatenate per output type
      results <- output_type %>%
        lapply(function(key) {
          batch_outputs %>%
            lapply(function(out) {
              val <- if (is.list(out) && !is_torch_tensor(out)) out[[key]] else out
              val$float()$cpu()
            }) %>%
            torch_cat(dim = 1L) %>%
            `$`(numpy())
        })
      names(results) <- output_type

      if (length(output_type) == 1L) return(results[[1L]])
      results
    },

    #' @description Batched forward pass using a pre-computed KV cache.
    #'
    #' @param Xs `array(n_datasets, test_size, n_features)`.
    #' @param kv_cache A \code{TabICLCache} for all estimators.
    #' @param output_type `character` vector of output types.
    #' @param alphas `numeric` vector or \code{NULL}.
    #'
    #' @return R array or named list of arrays.
    .batch_forward_with_cache = function(Xs, kv_cache, output_type, alphas = NULL) {
      n_total    <- dim(Xs)[1L]
      batch_size <- self$batch_size %||% n_total
      n_batches  <- ceiling(n_total / batch_size)
      Xs_list    <- .split_first_axis(Xs, n_batches)

      # Compute per-batch offsets for slicing the cache
      offsets <- c(1L, cumsum(rep(batch_size, n_batches - 1L)) + 1L)
      if (n_batches > 1L) {
        sizes <- lapply(seq_len(n_batches - 1L), function(i) {
          min(batch_size, n_total - offsets[i] + 1L)
        })
        sizes[[n_batches]] <- n_total - offsets[[n_batches]] + 1L
      } else {
        sizes <- list(n_total)
      }

      batch_outputs <- seq_len(n_batches) %>%
        lapply(function(i) {
          start <- offsets[[i]]
          end   <- start + sizes[[i]] - 1L
          cache_subset <- kv_cache$slice_batch(start, end)
          X_batch <- torch_tensor(as.array(Xs_list[[i]]), dtype = torch_float32())$to(self$device_)
          with(torch_no_grad(), {
            self$model_$predict_stats_with_cache(
              X_test           = X_batch,
              output_type      = output_type,
              alphas           = alphas,
              cache            = cache_subset,
              inference_config = self$inference_config_
            )
          })
        })

      # Extract and concatenate per output type
      results <- output_type %>%
        lapply(function(key) {
          batch_outputs %>%
            lapply(function(out) {
              val <- if (is.list(out) && !is_torch_tensor(out)) out[[key]] else out
              val$float()$cpu()
            }) %>%
            torch_cat(dim = 1L) %>%
            `$`(numpy())
        })
      names(results) <- output_type

      if (length(output_type) == 1L) return(results[[1L]])
      results
    },

    #' @description Predict target values for test samples.
    #'
    #' Applies the ensemble of TabICL models, averages predictions
    #' across ensemble members, and inverse-transforms to original
    #' scale.
    #'
    #' @param X `matrix`, `data.frame`, or array of shape
    #'   \code{(n_samples, n_features)}.
    #' @param output_type `character` vector.  One or more of
    #'   \code{"mean"}, \code{"median"}, \code{"quantiles"},
    #'   \code{"raw_quantiles"}.  Default: \code{"mean"}.
    #' @param alphas `numeric` vector of quantile levels, or
    #'   \code{NULL}.
    #'
    #' @return R array of shape \code{(n_samples,)} or
    #'   \code{(n_samples, n_quantiles)}, or a named list when
    #'   multiple output types are requested.
    predict = function(X, output_type = "mean", alphas = NULL) {
      check_is_fitted(self)

      # Reject 1D input
      if ((is.matrix(X) || is.array(X)) && length(dim(X)) == 1L) {
        value_error(
          "The provided input X is one-dimensional. Reshape your data."
        )
      }

      # Check prerequisites
      has_kv_cache     <- !is.null(self$model_kv_cache_)
      has_training_data <- !is.null(self$ensemble_generator_) &&
        !is.null(self$ensemble_generator_$X_)

      if (!has_kv_cache && !has_training_data) {
        runtime_error(
          "Cannot predict: this estimator was saved without training data ",
          "and has no KV cache. Re-fit or load from a file saved with ",
          "save_training_data=TRUE or save_kv_cache=TRUE."
        )
      }

      # Thread management
      old_n_threads <- NULL
      if (!is.null(self$n_jobs)) {
        stopifnot(self$n_jobs != 0)
        old_n_threads <- torch_get_num_threads()
        n_logical <- parallel::detectCores()
        n_threads <- if (self$n_jobs > 0) {
          if (self$n_jobs > n_logical) {
            cli_warn(
              "TabICL got n_jobs={self$n_jobs} but only {n_logical} ",
              "logical cores. Only {n_logical} threads will be used."
            )
          }
          max(n_logical, self$n_jobs)
        } else {
          max(1L, n_logical + 1L + self$n_jobs)
        }
        torch_set_num_threads(n_threads)
      }

      X <- validate_data(self, X, reset = FALSE, dtype = NULL, skip_check_array = TRUE)

      # Detect all-NaN columns (SHAP feature masking)
      feature_mask <- if (inherits(X, "data.frame")) {
        mask <- sapply(X, function(col) all(is.na(col)))
        if (!any(mask)) NULL else mask
      } else {
        arr <- as.matrix(X)
        if (is.numeric(arr)) {
          mask <- apply(arr, 2L, function(col) all(is.na(col)))
          if (!any(mask)) NULL else mask
        } else {
          NULL
        }
      }

      # Fill masked columns so transformers don't choke
      if (!is.null(feature_mask)) {
        if (inherits(X, "data.frame")) {
          X[, feature_mask] <- 0.0
        } else {
          X[, feature_mask] <- 0.0
        }
      }

      X <- self$X_encoder_$transform(X)

      # Ensure output_type is a list
      output_type <- as.list(output_type)
      output_type_names <- unlist(output_type)

      # Decide cache path
      use_cache <- !is.null(self$model_kv_cache_) && is.null(feature_mask)

      if (use_cache) {
        test_data   <- self$ensemble_generator_$transform(X, mode = "test")
        norm_methods <- names(test_data)

        raw_results <- setNames(
          lapply(norm_methods, function(nm) {
            kv_cache <- self$model_kv_cache_[[nm]]
            Xs_test  <- test_data[[nm]][[1L]]
            self$.batch_forward_with_cache(
              Xs_test, kv_cache,
              output_type = output_type, alphas = alphas
            )
          }),
          nm = norm_methods
        )
      } else {
        data        <- self$ensemble_generator_$transform(
          X, mode = "both", feature_mask = feature_mask
        )
        norm_methods <- names(data)

        raw_results <- setNames(
          lapply(norm_methods, function(nm) {
            pair <- data[[nm]]
            self$.batch_forward(
              pair[[1L]], pair[[2L]],
              output_type = output_type, alphas = alphas
            )
          }),
          nm = norm_methods
        )
      }

      # Aggregate across ensemble members, inverse-transform, average
      final_results <- setNames(
        lapply(output_type_names, function(key) {
          per_method <- lapply(raw_results, function(r) {
            if (is.list(r)) r[[key]] else r
          })

          arr <- .concat_first_axis(per_method)
          n_est  <- dim(arr)[1L]
          n_samp <- dim(arr)[2L]

          if (length(dim(arr)) == 2L) {
            # mean / median / variance: (n_estimators, n_samples)
            arr <- self$y_scaler_$inverse_transform(
              matrix(arr, ncol = 1L)
            ) %>% matrix(nrow = n_est, ncol = n_samp)
            colMeans(arr)
          } else {
            # quantiles: (n_estimators, n_samples, n_quantiles)
            n_q  <- dim(arr)[3L]
            arr <- self$y_scaler_$inverse_transform(
              matrix(arr, ncol = 1L)
            ) %>% array(dim = c(n_est, n_samp, n_q))
            apply(arr, c(2L, 3L), mean)
          }
        }),
        nm = output_type_names
      )

      # Restore thread count
      if (!is.null(old_n_threads)) {
        torch_set_num_threads(old_n_threads)
      }

      if (length(output_type_names) == 1L) return(final_results[[1L]])
      final_results
    }
  )
)
