#' @importFrom R6 R6Class
#' @importFrom rlang is_scalar_character is_scalar_integer is_scalar_logical is_scalar_double
#' @keywords internal
NULL

# Allowed keys for MgrConfig
.MGRCFG_ALLOWED_KEYS <- c(
  "device", "use_amp", "use_fa3", "verbose",
  "min_batch_size", "safety_factor",
  "offload", "auto_offload_threshold",
  "cpu_safety_factor", "max_pinned_memory_mb",
  "disk_offload_dir", "disk_min_free_mb", "disk_flush_mb",
  "disk_cleanup", "disk_file_prefix", "disk_dtype", "disk_safety_factor",
  "use_async", "async_depth"
)

# Validators: each returns TRUE if value is acceptable.
# verbose accepts integer 0/1 as well as logical (base class stores it as int).
.MGRCFG_SPECS <- list(
  device = list(
    check   = function(x) is.null(x) || is_scalar_character(x) || inherits(x, "torch_device"),
    coerce  = NULL,
    msg     = "device must be NULL, a character string, or a torch_device"
  ),
  use_amp = list(
    check   = is_scalar_logical,
    coerce  = as.logical,
    msg     = "use_amp must be a logical"
  ),
  use_fa3 = list(
    check   = is_scalar_logical,
    coerce  = as.logical,
    msg     = "use_fa3 must be a logical"
  ),
  verbose = list(
    check   = is_scalar_logical,
    coerce  = as.logical,
    msg     = "verbose must be a logical"
  ),
  min_batch_size = list(
    check   = function(x) is_scalar_integer(x) && x >= 1,
    coerce  = as.integer,
    msg     = "min_batch_size must be a number >= 1"
  ),
  safety_factor = list(
    check   = function(x) is_scalar_double(x) && x >= 0 && x <= 1,
    coerce  = NULL,
    msg     = "safety_factor must be a number in [0, 1]"
  ),
  offload = list(
    check   = function(x) {
      (is_scalar_logical(x)) ||
        (is_scalar_character(x) && x %in% c("auto", "gpu", "cpu", "disk"))
    },
    coerce  = NULL,
    msg     = "offload must be a logical or one of 'auto', 'gpu', 'cpu', 'disk'"
  ),
  auto_offload_threshold = list(
    check   = function(x) is_scalar_double(x) && x >= 0 && x <= 1,
    coerce  = NULL,
    msg     = "auto_offload_threshold must be a number in [0, 1]"
  ),
  cpu_safety_factor = list(
    check   = function(x) is_scalar_double(x) && x >= 0 && x <= 1,
    coerce  = NULL,
    msg     = "cpu_safety_factor must be a number in [0, 1]"
  ),
  max_pinned_memory_mb = list(
    check   = function(x) is_scalar_double(x) && x >= 0,
    coerce  = NULL,
    msg     = "max_pinned_memory_mb must be a non-negative number"
  ),
  disk_offload_dir = list(
    check   = function(x) is.null(x) || is_scalar_character(x),
    coerce  = NULL,
    msg     = "disk_offload_dir must be NULL or a character string"
  ),
  disk_min_free_mb = list(
    check   = function(x) is_scalar_double(x) && x >= 0,
    coerce  = NULL,
    msg     = "disk_min_free_mb must be a non-negative number"
  ),
  disk_flush_mb = list(
    check   = function(x) is_scalar_double(x) && x >= 0,
    coerce  = NULL,
    msg     = "disk_flush_mb must be a non-negative number"
  ),
  disk_cleanup = list(
    check   = is_scalar_logical,
    coerce  = as.logical,
    msg     = "disk_cleanup must be a logical"
  ),
  disk_file_prefix = list(
    check   = is_scalar_character,
    coerce  = NULL,
    msg     = "disk_file_prefix must be a character string"
  ),
  disk_dtype = list(
    check   = function(x) is.null(x) || inherits(x, "torch_dtype"),
    coerce  = NULL,
    msg     = "disk_dtype must be NULL or a torch dtype"
  ),
  disk_safety_factor = list(
    check   = function(x) is_scalar_double(x) && x >= 0 && x <= 1,
    coerce  = NULL,
    msg     = "disk_safety_factor must be a number in [0, 1]"
  ),
  use_async = list(
    check   = is_scalar_logical,
    coerce  = as.logical,
    msg     = "use_async must be a logical"
  ),
  async_depth = list(
    check   = function(x) is_scalar_integer(x) && x >= 1,
    coerce  = as.integer,
    msg     = "async_depth must be a number >= 1"
  )
)

.mgrcfg_validate_and_coerce <- function(key, value) {
  if (!key %in% .MGRCFG_ALLOWED_KEYS)
    stop(
      paste0("Invalid config key '", key, "'. Allowed keys: ",
             paste(.MGRCFG_ALLOWED_KEYS, collapse = ", ")),
      call. = FALSE
    )
  spec <- .MGRCFG_SPECS[[key]]
  if (!spec$check(value))
    stop(paste0(spec$msg, ". Got: ", class(value)[1L]), call. = FALSE)
  if (!is.null(spec$coerce)) spec$coerce(value) else value
}

#' Create a validated manager configuration (named list)
#'
#' Validates and returns a named list of \code{inference_manager} configuration
#' parameters.  Each parameter is checked for type and range; see
#' \code{\link{InferenceConfig}} for the full parameter documentation.
#'
#' @param ... Named configuration parameters.
#' @return A named list with class \code{"MgrConfig"}.
#' @export
MgrConfig <- function(...) {
  params <- list(...)
  if (length(params) > 0L && is.null(names(params)))
    stop("All arguments to MgrConfig() must be named.", call. = FALSE)

  out <- vector("list", length(params))
  names(out) <- names(params)
  for (key in names(params)) {
    out[[key]] <- .mgrcfg_validate_and_coerce(key, params[[key]])
  }
  structure(out, class = c("MgrConfig", "list"))
}

#' Convert a MgrConfig to a plain named list
#' @param x A \code{MgrConfig} object.
#' @param ... Ignored.
#' @export
as.list.MgrConfig <- function(x, ...) {
  unclass(x)
}

#' Update a MgrConfig list in place (returns updated copy)
#'
#' @param x A \code{MgrConfig} or named list.
#' @param updates A named list or \code{MgrConfig} with new values.
#' @return Updated \code{MgrConfig}.
#' @keywords internal
.mgrcfg_update <- function(x, updates) {
  if (inherits(updates, "MgrConfig") || is.list(updates)) {
    for (key in names(updates)) {
      x[[key]] <- .mgrcfg_validate_and_coerce(key, updates[[key]])
    }
  } else {
    stop("updates must be a named list or MgrConfig.", call. = FALSE)
  }
  if (!inherits(x, "MgrConfig")) class(x) <- c("MgrConfig", "list")
  x
}

# Default MgrConfig values ------------------------------------------------

.col_config_defaults <- function() {
  MgrConfig(
    device                = NULL,
    use_amp               = TRUE,
    use_fa3               = TRUE,
    verbose               = FALSE,
    min_batch_size        = 1L,
    safety_factor         = 0.8,
    offload               = "auto",
    auto_offload_threshold = 0.5,
    cpu_safety_factor     = 0.85,
    max_pinned_memory_mb  = 32768.0,
    disk_offload_dir      = NULL,
    disk_min_free_mb      = 1024.0,
    disk_flush_mb         = 8192.0,
    disk_cleanup          = TRUE,
    disk_file_prefix      = "",
    disk_dtype            = NULL,
    disk_safety_factor    = 0.95,
    use_async             = TRUE,
    async_depth           = 4L
  )
}

.row_config_defaults <- function() {
  MgrConfig(
    device                = NULL,
    use_amp               = TRUE,
    use_fa3               = TRUE,
    verbose               = FALSE,
    min_batch_size        = 1L,
    safety_factor         = 0.8,
    offload               = FALSE,
    auto_offload_threshold = 0.5,
    cpu_safety_factor     = 0.85,
    max_pinned_memory_mb  = 32768.0,
    disk_offload_dir      = NULL,
    disk_min_free_mb      = 1024.0,
    disk_flush_mb         = 8192.0,
    disk_cleanup          = TRUE,
    disk_file_prefix      = "",
    disk_dtype            = NULL,
    disk_safety_factor    = 0.95,
    use_async             = TRUE,
    async_depth           = 4L
  )
}

.icl_config_defaults <- .row_config_defaults  # Same defaults as ROW


#' InferenceConfig: Container for TabICL inference configuration
#'
#' Holds \code{\link{MgrConfig}} objects for the three transformer components
#' of TabICL: column-wise embedding (\code{COL_CONFIG}), row-wise interaction
#' (\code{ROW_CONFIG}), and in-context learning (\code{ICL_CONFIG}).
#'
#' @export
InferenceConfig <- R6::R6Class(
  "InferenceConfig",

  public = list(
    #' @field COL_CONFIG \code{MgrConfig} for the column embedding transformer.
    COL_CONFIG = NULL,
    #' @field ROW_CONFIG \code{MgrConfig} for the row interaction transformer.
    ROW_CONFIG = NULL,
    #' @field ICL_CONFIG \code{MgrConfig} for the in-context learning transformer.
    ICL_CONFIG = NULL,

    #' @description Create a new \code{InferenceConfig} with default settings.
    #'
    #' @param COL_CONFIG A \code{MgrConfig}, named list, or \code{NULL} for
    #'   defaults.
    #' @param ROW_CONFIG A \code{MgrConfig}, named list, or \code{NULL} for
    #'   defaults.
    #' @param ICL_CONFIG A \code{MgrConfig}, named list, or \code{NULL} for
    #'   defaults.
    initialize = function(COL_CONFIG = NULL, ROW_CONFIG = NULL, ICL_CONFIG = NULL) {
      self$COL_CONFIG <- private$.init_sub(COL_CONFIG, .col_config_defaults)
      self$ROW_CONFIG <- private$.init_sub(ROW_CONFIG, .row_config_defaults)
      self$ICL_CONFIG <- private$.init_sub(ICL_CONFIG, .icl_config_defaults)
      invisible(self)
    },

    #' @description Update sub-configurations from a nested named list.
    #'
    #' @param config_dict A named list with keys \code{"COL_CONFIG"},
    #'   \code{"ROW_CONFIG"}, and/or \code{"ICL_CONFIG"}, each mapping to a
    #'   named list of parameter overrides.
    update_from_dict = function(config_dict) {
      allowed <- c("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")
      for (key in names(config_dict)) {
        if (!key %in% allowed)
          stop(
            paste0("Invalid InferenceConfig key '", key, "'. Allowed: ",
                   paste(allowed, collapse = ", ")),
            call. = FALSE
          )
        self[[key]] <- .mgrcfg_update(self[[key]], config_dict[[key]])
      }
      invisible(self)
    }
  ),

  private = list(
    .init_sub = function(arg, default_fn) {
      if (is.null(arg)) {
        default_fn()
      } else if (is.list(arg)) {
        .mgrcfg_update(default_fn(), arg)
      } else if (inherits(arg, "MgrConfig")) {
        arg
      } else {
        stop(
          paste0("Config argument must be NULL, a named list, or a MgrConfig. Got: ",
                 class(arg)[1L]),
          call. = FALSE
        )
      }
    }
  )
)

#' Create an InferenceConfig with default settings
#'
#' Convenience alias for \code{InferenceConfig$new()}.
#'
#' @param ... Arguments forwarded to \code{InferenceConfig$new()}.
#' @return An \code{InferenceConfig} object.
#' @export
inference_config <- function(...) InferenceConfig$new(...)
