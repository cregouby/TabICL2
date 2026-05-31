# Default settings for each transformer stage --------------------------------

.icfg_col_defaults <- function() {
  list(
    device                 = NULL,
    use_amp                = TRUE,
    use_fa3                = TRUE,
    verbose                = FALSE,
    min_batch_size         = 1L,
    safety_factor          = 0.8,
    offload                = "auto",
    auto_offload_threshold = 0.5,
    cpu_safety_factor      = 0.85,
    max_pinned_memory_mb   = 32768.0,
    disk_offload_dir       = NULL,
    disk_min_free_mb       = 1024.0,
    disk_flush_mb          = 8192.0,
    disk_cleanup           = TRUE,
    disk_file_prefix       = "",
    disk_dtype             = NULL,
    disk_safety_factor     = 0.95,
    use_async              = TRUE,
    async_depth            = 4L
  )
}

.icfg_row_defaults <- function() {
  d <- .icfg_col_defaults()
  d$offload <- FALSE
  d
}

.icfg_merge <- function(base, overrides) {
  if (length(overrides) == 0L) return(base)
  bad_keys <- setdiff(names(overrides), names(base))
  if (length(bad_keys) > 0L) {
    cli_abort("Invalid inference config key{?s}: {.val {bad_keys}}")
  }
  utils::modifyList(base, overrides)
}



#' Control TabICL2 inference execution
#'
#' @param device A character string or `NULL` for the torch device (e.g.,
#'   `"cpu"`, `"cuda"`, `"mps"`). Applied to all three transformer stages.
#' @param use_amp A logical. Enable automatic mixed precision. Applied to all
#'   stages.
#' @param use_fa3 A logical. Enable Flash Attention 3 kernels. Applied to all
#'   stages.
#' @param verbose A logical. Enable verbose logging. Applied to all stages.
#' @param min_batch_size A positive integer. Minimum batch size passed to the
#'   memory manager. Applied to all stages.
#' @param safety_factor A number in \[0, 1\]. Memory safety margin for batch
#'   size estimation. Applied to all stages.
#' @param offload A logical or one of `"auto"`, `"gpu"`, `"cpu"`, `"disk"`.
#'   Offload strategy for the column-embedding (COL) stage. Row-interaction
#'   (ROW) and in-context-learning (ICL) stages default to `FALSE` unless
#'   overridden via `row_config`/`icl_config`.
#' @param ignore_pretraining_limits A logical. Bypass the default data size
#'   limits on training set rows (50,000) and predictors (2,000).
#' @param col_config A named list of additional settings for the COL stage,
#'   overriding any of the defaults or parameters above.
#' @param row_config A named list of additional settings for the ROW stage.
#' @param icl_config A named list of additional settings for the ICL stage.
#' @param ... Additional named arguments stored in the returned control object.
#'   Use this for settings not covered above (e.g. `auto_offload_threshold`,
#'   `disk_offload_dir`, `use_async`, etc.) that apply uniformly. Per-stage
#'   values must go in `col_config`, `row_config`, or `icl_config`.
#' @return A list with class `"tab_icl2_control"` containing resolved
#'   per-stage settings in `$col`, `$row`, and `$icl`, plus
#'   `$ignore_pretraining_limits`.
#' @examples
#' tab_icl2_control()
#' tab_icl2_control(use_amp = FALSE, offload = "cpu")
#' tab_icl2_control(col_config = list(disk_offload_dir = tempdir()))
#' tab_icl2_control(ignore_pretraining_limits = TRUE)
#' @export
tab_icl2_control <- function(
  device                    = NULL,
  use_amp                   = TRUE,
  use_fa3                   = TRUE,
  verbose                   = FALSE,
  min_batch_size            = 1L,
  safety_factor             = 0.8,
  offload                   = "auto",
  ignore_pretraining_limits = FALSE,
  col_config                = list(),
  row_config                = list(),
  icl_config                = list(),
  ...
) {
  if (!is.null(device)) check_string(device)
  check_bool(use_amp)
  check_bool(use_fa3)
  check_bool(verbose)
  check_number_whole(min_batch_size, min = 1)
  if (!is.numeric(safety_factor) || length(safety_factor) != 1L ||
      safety_factor < 0 || safety_factor > 1) {
    cli_abort("{.arg safety_factor} must be a number in [0, 1].")
  }
  check_bool(ignore_pretraining_limits)

  offload_msg <- paste0(
    "{.arg offload} must be a single logical or one of ",
    "{.val {c('auto', 'gpu', 'cpu', 'disk')}}."
  )
  if (length(offload) != 1L) cli_abort(offload_msg)
  if (!is.logical(offload) &&
      !(is.character(offload) && offload %in% c("auto", "gpu", "cpu", "disk"))) {
    cli_abort(offload_msg)
  }

  if (!is.list(col_config)) cli_abort("{.arg col_config} must be a list.")
  if (!is.list(row_config)) cli_abort("{.arg row_config} must be a list.")
  if (!is.list(icl_config)) cli_abort("{.arg icl_config} must be a list.")

  dot_args <- rlang::list2(...)
  reserved <- c(
    "device", "use_amp", "use_fa3", "verbose", "min_batch_size",
    "safety_factor", "offload", "ignore_pretraining_limits",
    "col_config", "row_config", "icl_config"
  )
  conflicts <- intersect(names(dot_args), reserved)
  if (length(conflicts) > 0L) {
    cli_abort(
      "Argument{?s} {.arg {conflicts}} must be passed as named argument{?s}, not via {.code ...}."
    )
  }

  top_level <- list(
    device         = device,
    use_amp        = use_amp,
    use_fa3        = use_fa3,
    verbose        = verbose,
    min_batch_size = as.integer(min_batch_size),
    safety_factor  = safety_factor
  )

  col <- .icfg_merge(
    .icfg_merge(.icfg_col_defaults(), c(top_level, list(offload = offload))),
    col_config
  )
  row <- .icfg_merge(.icfg_merge(.icfg_row_defaults(), top_level), row_config)
  icl <- .icfg_merge(.icfg_merge(.icfg_row_defaults(), top_level), icl_config)

  args <- c(
    list(
      col                       = col,
      row                       = row,
      icl                       = icl,
      ignore_pretraining_limits = ignore_pretraining_limits
    ),
    dot_args
  )
  class(args) <- "tab_icl2_control"
  args
}

#' @export
print.tab_icl2_control <- function(x, ...) {
  defaults <- tab_icl2_control()
  stages   <- c("col", "row", "icl")

  cli_inform("inference control for {.fn tab_icl2}")

  # Non-default top-level (non-stage) parameters
  top_keys          <- setdiff(names(x), stages)
  top_default_keys  <- setdiff(names(defaults), stages)
  common_top        <- intersect(top_keys, top_default_keys)
  extra_top         <- setdiff(top_keys, top_default_keys)
  non_default_top <- c(
    purrr::map2_lgl(x[common_top], defaults[common_top], ~ !identical(.x, .y)),
    stats::setNames(rep(TRUE, length(extra_top)), extra_top)
  )
  if (any(non_default_top)) {
    cat("\n")
    cli_inform("non-default top-level arguments:")
    xsub <- x[non_default_top]
    lst <- purrr::map2(
      names(xsub),
      xsub,
      ~ cli::format_inline("{.arg {.x}}: {.val {.y}}")
    )
    names(lst) <- rep("*", length(lst))
    cli::cli_bullets(lst)
  }

  # Non-default per-stage parameters
  has_any <- FALSE
  for (stage in stages) {
    stage_defaults <- defaults[[stage]]
    stage_current  <- x[[stage]]
    common <- intersect(names(stage_current), names(stage_defaults))
    non_default <- purrr::map2_lgl(
      stage_current[common],
      stage_defaults[common],
      ~ !identical(.x, .y)
    )
    if (any(non_default)) {
      if (!has_any) cat("\n")
      has_any <- TRUE
      cli_inform("{toupper(stage)} stage non-default arguments:")
      xsub <- stage_current[non_default]
      lst <- purrr::map2(
        names(xsub),
        xsub,
        ~ cli::format_inline("{.arg {.x}}: {.val {.y}}")
      )
      names(lst) <- rep("*", length(lst))
      cli::cli_bullets(lst)
    }
  }

  invisible(x)
}

#' @export
#' @rdname tab_icl2_control
inference_config <- function(...) tab_icl2_control(...)
