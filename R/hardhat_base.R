#' @importFrom torch torch_device torch_float32 cuda_is_available
#'   torch_float16 torch_bfloat16 nn_module
#' @importFrom cli cli_inform cli_warn cli_abort
#' @importFrom glue glue
#' @importFrom utils packageVersion
#' @keywords internal
NULL

#' TabICL Base Estimator Class
#'
#' Base class for TabICL scikit-learn compatible estimators.
#' Provides shared functionality for both classifier and regressor.
#'
#' @importFrom R6 R6Class
#' @export
TabICLBaseEstimator <- R6::R6Class(
  "TabICLBaseEstimator",

  public = list(

    #' @field device Device specification (string, torch_device, or NULL)
    device = NULL,

    #' @field use_amp Automatic mixed precision setting ("auto", TRUE, or FALSE)
    use_amp = "auto",

    #' @field use_fa3 Flash Attention 3 setting ("auto", TRUE, or FALSE)
    use_fa3 = "auto",

    #' @field verbose Verbosity level
    verbose = 0,

    #' @field offload_mode Offloading mode for memory management
    offload_mode = NULL,

    #' @field disk_offload_dir Directory for disk offloading
    disk_offload_dir = NULL,

    #' @field inference_config Inference configuration (list or InferenceConfig)
    inference_config = NULL,

    # --- Fitted attributes (set by subclass $fit()) ---
    #' @field device_ Resolved torch device (set by \code{.resolve_device()}).
    device_ = NULL,
    #' @field inference_config_ Resolved InferenceConfig (set by \code{.build_inference_config()}).
    inference_config_ = NULL,
    #' @field n_samples_in_ Number of training samples (set by \code{$fit()}).
    n_samples_in_ = NULL,
    #' @field n_features_in_ Number of training features (set by \code{$fit()}).
    n_features_in_ = NULL,

    #' Initialize the estimator
    #'
    #' @param device Device specification
    #' @param use_amp AMP setting
    #' @param use_fa3 FA3 setting
    #' @param verbose Verbosity
    #' @param offload_mode Offloading mode
    #' @param disk_offload_dir Offload directory
    #' @param inference_config Inference config
    initialize = function(device = NULL, use_amp = "auto", use_fa3 = "auto",
                         verbose = 0, offload_mode = NULL,
                         disk_offload_dir = NULL, inference_config = NULL) {
      self$device <- device
      self$use_amp <- use_amp
      self$use_fa3 <- use_fa3
      self$verbose <- verbose
      self$offload_mode <- offload_mode
      self$disk_offload_dir <- disk_offload_dir
      self$inference_config <- inference_config
      invisible(self)
    },

    #' Get additional tags for the estimator
    #'
    #' @return A list with estimator tags
    .more_tags = function() {
      list(non_deterministic = TRUE)
    },

    #' Get sklearn-compatible tags
    #'
    #' @return A list with tags
    .sklearn_tags = function() {
      list(non_deterministic = TRUE)
    },

    #' Resolve the target device from the init parameter
    #'
    #' TODO : improve it through tabnet:::get_device_from_config
    #' @return A proper torch_device()
    #' @keywords internal
    .resolve_device = function() {
      self$device_ <- if (is.null(self$device)) {
        torch_device(if (cuda_is_available()) "cuda" else "cpu")
      } else if (is.character(self$device)) {
        torch_device(self$device)
      } else {
        self$device
      }
    },

    #' Resolve the "auto" option for use_amp and use_fa3
    #'
    #' @return A list with use_amp and use_fa3 boolean values
    #' @keywords internal
    .resolve_amp_fa3 = function() {
      n_samples <- if (!is.null(self$n_samples_in_)) {
        self$n_samples_in_
      } else {
        0
      }
      n_features <- if (!is.null(self$n_features_in_)) {
        self$n_features_in_
      } else {
        0
      }
      small_data <- n_samples < 1024 && n_features < 60

      # AMP resolution
      use_amp <- if (identical(self$use_amp, "auto")) {
        !small_data
      } else {
        as.logical(self$use_amp)
      }

      # FA3 resolution
      use_fa3 <- if (identical(self$use_fa3, "auto")) {
        if (small_data) {
          FALSE
        } else if (!use_amp) {
          TRUE
        } else {
          n_samples >= 10240
        }
      } else {
        as.logical(self$use_fa3)
      }

      list(use_amp = use_amp, use_fa3 = use_fa3)
    },

    #' Build the inference configuration from init parameters
    #'
    #' @return A proper config list
    #' @keywords internal
    .build_inference_config = function() {
      amp_fa3 <- self$.resolve_amp_fa3()
      use_amp <- amp_fa3$use_amp
      use_fa3 <- amp_fa3$use_fa3

      init_config <- list(
        COL_CONFIG = list(
          device = self$device_,
          use_amp = use_amp,
          use_fa3 = use_fa3,
          verbose = self$verbose,
          offload = self$offload_mode,
          disk_offload_dir = self$disk_offload_dir
        ),
        ROW_CONFIG = list(
          device = self$device_,
          use_amp = use_amp,
          use_fa3 = use_fa3,
          verbose = self$verbose
        ),
        ICL_CONFIG = list(
          device = self$device_,
          use_amp = use_amp,
          use_fa3 = use_fa3,
          verbose = self$verbose
        )
      )

      if (is.null(self$inference_config)) {
        self$inference_config_ <- InferenceConfig$new()
        self$inference_config_$update_from_dict(init_config)
      } else if (is.list(self$inference_config)) {
        self$inference_config_ <- InferenceConfig$new()
        purrr::iwalk(self$inference_config, function(value, key) {
          if (key %in% names(init_config)) {
            init_config[[key]] <- append(init_config[[key]], value)
          }
        })
        self$inference_config_$update_from_dict(init_config)
      } else {
        self$inference_config_ <- self$inference_config
      }
      invisible(self)
    },

    #' Move KV cache to the current device, auto-upcasting if needed
    #'
    #' @return nothing
    #' @keywords internal
    .move_cache_to_device = function() {
      if (!exists("model_kv_cache_", envir = self, inherits = FALSE) ||
          is.null(self$model_kv_cache_)) {
        return(invisible(NULL))
      }

      amp_fa3 <- self$.resolve_amp_fa3()
      use_amp <- amp_fa3$use_amp

      needs_upcast <- self$device_$type %in% c("cpu", "mps") || !use_amp
      upcast_dtype <- if (needs_upcast) torch_float32() else NULL

      # Warn once if we are actually upcasting reduced-precision tensors
      if (!is.null(upcast_dtype)) {
        first_cache <- self$model_kv_cache_[[1]]
        cache_dtype <- first_cache$col_cache$kv[[1]]$key$dtype
        if (!identical(cache_dtype, torch_float32())) {
          reason <- if (self$device_$type %in% c("cpu", "mps")) {
            paste0(toupper(self$device_$type), " does not support float16/bfloat16 attention")
          } else {
            "AMP is not enabled"
          }
          cli_warn(
            "KV cache contains {cache_dtype} tensors (typically from AMP). Automatically upcasting to float32 because {reason}."
          )
        }
      }

      device_cache <- purrr::map(self$model_kv_cache_, function(cache) {
        if (is.null(upcast_dtype)) {
          cache$to(self$device_)
        } else {
          cache$to(self$device_, dtype = upcast_dtype)
        }
      })
      self$model_kv_cache_ <- device_cache
      invisible(self)
    },

    #' Save the fitted estimator to a file
    #'
    #' @param path File path to save to
    #' @param save_model_weights Include model weights in file
    #' @param save_training_data Include training data
    #' @param save_kv_cache Include KV cache
    #' @keywords internal
    save = function(path, save_model_weights = FALSE, save_training_data = TRUE,
                   save_kv_cache = TRUE) {
      # check_is_fitted equivalent would go here

      has_kv_cache <- exists("model_kv_cache_", envir = self, inherits = FALSE) &&
                      !is.null(self$model_kv_cache_)
      if (!save_training_data && !(save_kv_cache && has_kv_cache)) {
        value_error(
          "Cannot exclude training data when KV cache is not available or not being saved. Either set save_training_data=TRUE, or set kv_cache=TRUE during init and save_kv_cache=TRUE."
        )
      }

      # Set temporary flags for serialization
      private$._save_model_weights <- save_model_weights
      private$._save_kv_cache <- save_kv_cache
      private$._save_training_data <- save_training_data

      tryCatch({
        path <- normalizePath(path, mustWork = FALSE)
        dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
        saveRDS(self, file = path, version = 2)
      }, finally = {
        private$._save_model_weights <- NULL
        private$._save_kv_cache <- NULL
        private$._save_training_data <- NULL
      })
      invisible(self)
    }
  ),

  private = list(
    ._save_model_weights = NULL,
    ._save_kv_cache = NULL,
    ._save_training_data = NULL
  )
)

#' Compute softmax values with temperature scaling
#'
#' Computes softmax(x / temperature) using array operations.
#'
#' @param x Input array of logits
#' @param axis Axis along which to compute softmax
#' @param temperature Temperature scaling parameter
#' @return Softmax probabilities array
#' @export
softmax <- function(x, axis = -1, temperature = 0.9) {
  if (is.null(dim(x))) {
    stop("x must be an array or matrix")
  }

  nd <- length(dim(x))
  if (axis < 0) axis <- nd + axis + 1
  if (axis < 1 || axis > nd) {
    stop("axis must be between 1 and ", nd)
  }

  x <- x / temperature
  x_max <- apply(x, MARGIN = axis, FUN = max, na.rm = TRUE)
  e_x <- exp(sweep(x, MARGIN = axis, STATS = x_max, FUN = "-"))
  e_x_sum <- apply(e_x, MARGIN = axis, FUN = sum, na.rm = TRUE)
  sweep(e_x, MARGIN = axis, STATS = e_x_sum, FUN = "/")
}
