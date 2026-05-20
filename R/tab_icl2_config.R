#' Configure TabICL2 model architecture
#'
#' @param embed_dim A positive integer. Embedding dimension shared across all
#'   transformer stages.
#' @param col_n_block A positive integer. Number of column-wise transformer
#'   blocks.
#' @param row_n_block A positive integer. Number of row-wise transformer blocks.
#' @param icl_n_block A positive integer. Number of in-context-learning
#'   transformer blocks.
#' @param col_n_head A positive integer. Number of attention heads for column
#'   blocks.
#' @param row_n_head A positive integer. Number of attention heads for row
#'   blocks.
#' @param icl_n_head A positive integer. Number of attention heads for ICL
#'   blocks.
#' @param feature_group_size A positive integer. Size of feature groups for
#'   cyclic repeated grouping.
#' @param col_n_cls A positive integer. Number of CLS tokens per column,
#'   concatenated to produce the row representation fed into the ICL stage.
#' @param row_n_cls A positive integer. Number of inducing vectors used in
#'   the induced self-attention column blocks.
#' @param num_quantiles A positive integer (>= 5). Number of quantiles
#'   predicted by the regression head.
#' @param ... Additional named arguments stored in the returned config object.
#'   Use this for architecture settings not covered by the named parameters
#'   above (e.g., arguments added in newer versions of the model).
#' @return A list with class `"tab_icl2_config"` containing one element per
#'   architecture parameter, all coerced to integer.
#' @examples
#' tab_icl2_config()
#' tab_icl2_config(embed_dim = 256L, icl_n_block = 8L)
#' @export
tab_icl2_config <- function(
  embed_dim          = 128L,
  col_n_block        = 3L,
  row_n_block        = 3L,
  icl_n_block        = 12L,
  col_n_head         = 8L,
  row_n_head         = 8L,
  icl_n_head         = 8L,
  feature_group_size = 3L,
  col_n_cls          = 4L,
  row_n_cls          = 128L,
  num_quantiles      = 30L,
  ...
) {
  check_number_whole(embed_dim,          min = 1)
  check_number_whole(col_n_block,        min = 1)
  check_number_whole(row_n_block,        min = 1)
  check_number_whole(icl_n_block,        min = 1)
  check_number_whole(col_n_head,         min = 1)
  check_number_whole(row_n_head,         min = 1)
  check_number_whole(icl_n_head,         min = 1)
  check_number_whole(feature_group_size, min = 1)
  check_number_whole(col_n_cls,          min = 1)
  check_number_whole(row_n_cls,          min = 1)
  check_number_whole(num_quantiles,      min = 5)

  dot_args <- rlang::list2(...)
  reserved <- c(
    "embed_dim", "col_n_block", "row_n_block", "icl_n_block",
    "col_n_head", "row_n_head", "icl_n_head", "feature_group_size",
    "col_n_cls", "row_n_cls", "num_quantiles"
  )
  conflicts <- intersect(names(dot_args), reserved)
  if (length(conflicts) > 0L) {
    cli_abort(
      "Argument{?s} {.arg {conflicts}} must be passed as named argument{?s}, not via {.code ...}."
    )
  }

  args <- c(
    list(
      embed_dim          = as.integer(embed_dim),
      col_n_block        = as.integer(col_n_block),
      row_n_block        = as.integer(row_n_block),
      icl_n_block        = as.integer(icl_n_block),
      col_n_head         = as.integer(col_n_head),
      row_n_head         = as.integer(row_n_head),
      icl_n_head         = as.integer(icl_n_head),
      feature_group_size = as.integer(feature_group_size),
      col_n_cls          = as.integer(col_n_cls),
      row_n_cls          = as.integer(row_n_cls),
      num_quantiles      = as.integer(num_quantiles)
    ),
    dot_args
  )

  class(args) <- "tab_icl2_config"
  args
}

#' @export
print.tab_icl2_config <- function(x, ...) {
  defaults <- tab_icl2_config()
  common   <- intersect(names(x), names(defaults))
  extra    <- setdiff(names(x), names(defaults))

  non_default <- c(
    purrr::map2_lgl(x[common], defaults[common], ~ !identical(.x, .y)),
    stats::setNames(rep(TRUE, length(extra)), extra)
  )

  cli_inform("architecture config for {.fn tab_icl2}")
  if (any(non_default)) {
    cat("\n")
    cli_inform("non-default arguments:")
    xsub <- x[non_default]
    lst <- purrr::map2(
      names(xsub),
      xsub,
      ~ cli::format_inline("{.arg {.x}}: {.val {.y}}")
    )
    names(lst) <- rep("*", length(lst))
    cli::cli_bullets(lst)
  }

  invisible(x)
}
