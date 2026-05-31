row_limits <- 50000
col_limits <- 2000
cls_limits <- 10

check_data_constraints <- function(x, y, control) {
  lvls <- levels(y)

  x_dims <- dim(x)
  if (x_dims[1] > row_limits && !control$ignore_pretraining_limits) {
    cli_abort(
      call = NULL,
      c(
        i = "There are {format(x_dims[1], big.mark = ',')} rows in the training set.",
        i = "TabICLv2 is intended for training set sizes <=
        {format(row_limits, big.mark = ',')} rows.",
        i = "Consider setting the option {.arg ignore_pretraining_limits} to
        {.val TRUE} or subset the training size using the
        {.arg training_set_limit} argument."
      )
    )
  }
  if (x_dims[2] > col_limits && !control$ignore_pretraining_limits) {
    cli_abort(
      call = NULL,
      c(
        i = "There are {format(x_dims[2], big.mark = ',')} predictors in the training set.",
        i = "TabICLv2 is intended for <= {col_limits} predictors.",
        i = "Consider setting the option {.arg ignore_pretraining_limits} to {.val TRUE} or subset the training size."
      )
    )
  }

  if (!is.null(lvls) && length(lvls) > cls_limits) {
    cli_abort(
      call = NULL,
      c(
        i = "There are {length(lvls)} classes in the outcome.",
        x = "TabICLv2 is intended for <= {cls_limits} classes and won't work with more."
      )
    )
  }

  invisible(NULL)
}

# Sampling down the data for data constraints

sample_indicies <- function(molded, size_limit = row_limits) {
  num_rows <- nrow(molded$outcomes)
  if (num_rows <= size_limit) {
    return(integer(0))
  }

  dat <- molded$outcomes %>%
    dplyr::mutate(.row_order = dplyr::row_number()) %>%
    rlang::set_names(c("outcome", ".row_order"))

  is_factor <- is.factor(dat$outcome)

  if (is_factor) {
    data_subset <- dat %>%
      dplyr::group_by(.data$outcome) %>%
      dplyr::group_nest(keep = TRUE) %>%
      dplyr::mutate(
        size = purrr::map_int(data, nrow),
        sample_prop = .data$size / num_rows,
        sample_num = ceiling(.data$sample_prop * size_limit),
        data = purrr::map2(.data, .data$sample_num, ~ dplyr::slice_sample(.x, n = .y))
      )
  } else {
    data_subset <- dat %>%
      dplyr::mutate(quantile = dplyr::ntile(.data$outcome, n = 4)) %>%
      dplyr::group_by(.data$quantile) %>%
      dplyr::group_nest(keep = TRUE) %>%
      dplyr::mutate(
        size = purrr::map_int(data, nrow),
        sample_prop = .data$size / num_rows,
        sample_num = ceiling(.data$sample_prop * size_limit),
        data = purrr::map2(.data, .data$sample_num, ~ dplyr::slice_sample(.x, n = .y))
      )
  }

  purrr::map_dfr(data_subset$data, ~.x) %>%
    dplyr::arrange(.data$.row_order) %>%
    dplyr::select(.data$.row_order) %>%
    dplyr::slice(1:size_limit) %>%
    purrr::pluck(".row_order")
}


