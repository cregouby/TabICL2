#' Fit a TabICL2 model.
#'
#' `tab_icl2()` applies data to a pre-estimated deep learning model defined by
#' Qu _et al_ (2026). This model emulates Bayesian inference for
#' regression and classification models.
#'
#' @param x Depending on the context:
#'
#'   * A __data frame__ of predictors.
#'   * A __matrix__ of predictors.
#'   * A __recipe__ specifying a set of preprocessing steps
#'     created from [recipes::recipe()].
#'
#' @param y When `x` is a __data frame__ or __matrix__, `y` is the outcome
#' specified as:
#'
#'   * A __data frame__ with 1 numeric column.
#'   * A __matrix__ with 1 numeric column.
#'   * A numeric __vector__ for regression or a __factor__ for classification.
#'
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @param formula A formula specifying the outcome terms on the left-hand side,
#' and the predictor terms on the right-hand side.
#'
#' @param num_quantiles An integer for the number of quantiles in case of regression.
#'  Default is `30L`.
#'
#' @param version A character string for the model version. Currently unsupported.
#'
#' @param control A list of options produced by [control_tab_icl2()].
#'
#' @param ... Not currently used, but required for extensibility.
#'
#' @details
#'
#' ## Computing Requirements
#'
#' This model can be used with or without a graphics processing unit (GPU).
#' However, it is fairly limited when used with a CPU (and no GPU). There might
#' be additional data size limitation warnings with CPU computations, and,
#' understandably, the execution time is much longer. CPU computations can also
#' consume a significant amount of system memory, depending on the size of your
#' data.
#'
#' GPUs using CUDA (Compute Unified Device Architecture) are most effective.
#' Limited testing with others has shown that GPUs with Metal Performance
#' Shaders (MPS) instructions (e.g., Apple GPUs) have limited utility for these
#' specific computations and might be slower than the CPU for some data sets.
#'
#' ## Data
#'
#' Be default, there are limits to the training data dimensions:
#'
#'   * Version 2.0: number of training set samples (10,000) and, the number of
#'   predictors (500). There is an unchangeable limit to the number of classes
#'   (10).
#'
#'   * Version 2.5: number of training set samples (50,000) and, the number of
#'   predictors (2,000). There is an unchangeable limit to the number of classes
#'   (10).
#'
#' Predictors do not require preprocessing; missing values and factor vectors
#' are allowed.
#'
#' ## Model Selection
#'
#' ### Selecting a model version
#'
#' Use the `version` argument to select a specific released model version. For
#' example:
#'
#' \preformatted{
#'   # Use version 2.0
#'   mod <- tab_icl2(predictors, outcome, version = "v2")
#'
#'   # Use version 2.5
#'   mod <- tab_icl2(predictors, outcome, version = "v2.5")
#' }
#'
#' ### Pointing to a local model file
#'
#' If you have a model file on disk (e.g., downloaded for offline use), pass
#' its path via `control_tab_icl2(model_path = ...)`:
#'
#' \preformatted{
#'   ctrl <- control_tab_icl2(model_path = "/path/to/model_file.ckpt")
#'   mod  <- tab_icl2(predictors, outcome, control = ctrl)
#' }
#'
#' Note that `version` and `model_path` are mutually exclusive: if `version`
#' is set, it overwrites any `model_path` supplied through `control`.
#'
#' ## Calculations
#'
#' For the `softmax_temperature` value, the softmax terms are:
#'
#' \preformatted{
#' exp(value / softmax_temperature)
#' }
#'
#' A value of `softmax_temperature = 1` results in a plain softmax value.
#'
#' @return
#'
#' A `tab_icl2` object with elements:
#'
#'   * `fit`: the python object containing the model.
#'   * `levels`: a character string of class levels (or NULL for regression)
#'   * `training`: a vector with the training set dimensions.
#'   * `logging`: any R or python messages produced by the computations.
#'   * `blueprint`: am object produced by [hardhat::mold()] used to process
#'      new data during prediction.
#'
#' @references
#'
#' Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan.
#' "TabICLv2: A better, faster, scalable, and open tabular foundation model."
#'  _arXiv preprint_ arXiv:2602.11139 (2026).
#'
#' Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan.
#' "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data." _arXiv preprint_
#' arXiv:2502.05564 (2025).
#'
#' @seealso [control_tab_icl2()], [predict.tab_icl2()]
#' @examples
#' predictors <- mtcars[, -1]
#' outcome <- mtcars[, 1]
#'
#' \dontrun{
#' if (torch_is_installed() & interactive()) {
#'  # XY interface
#'  mod <- tab_icl2(predictors, outcome)
#'
#'  # Formula interface
#'  mod2 <- tab_icl2(mpg ~ ., mtcars)
#'
#'  # Recipes interface
#'  if (rlang::is_installed("recipes")) {
#'   suppressPackageStartupMessages(library(recipes))
#'   rec <-
#'    recipe(mpg ~ ., mtcars) %>%
#'    step_log(disp)
#'
#'   mod3 <- tab_icl2(rec, mtcars)
#'   mod3
#'  }
#' }
#' }
#'
#' @export
tab_icl2 <- function(x, ...) {
  UseMethod("tab_icl2")
}

#' @export
#' @rdname tab_icl2
tab_icl2.default <- function(x, ...) {
  cli_abort("{.fn tab_icl2} is not defined for {cls class(x)}.")
}

# XY method - data frame

#' @export
#' @rdname tab_icl2
tab_icl2.data.frame <- function(
  x,
  y,
  num_quantiles = 30L,
  # softmax_temperature = 0.9,
  # balance_probabilities = FALSE,
  # average_before_softmax = FALSE,
  # training_set_limit = 10000,
  version = NULL,
  control = control_tab_icl2(),
  ...
) {
  options <- control
  options$num_quantiles <- num_quantiles
  # options$softmax_temperature <- softmax_temperature
  # options$balance_probabilities <- balance_probabilities
  # options$average_before_softmax <- average_before_softmax
  options <- check_fit_args(options)
  # check_number_whole(training_set_limit, min = 2, allow_infinite = TRUE)

  processed <- hardhat::mold(x, y)
  tr_ind <- sample_indicies(processed, size_limit = 1e4)
  if (length(tr_ind) > 0) {
    processed$predictors <- processed$predictors[tr_ind, , drop = FALSE]
    processed$outcomes <- processed$outcomes[tr_ind, , drop = FALSE]
  }

  tab_icl2_bridge(processed, options, version = version, ...)
}

# XY method - matrix

#' @export
#' @rdname tab_icl2
tab_icl2.matrix <- function(
  x,
  y,
  num_quantiles = 30L,
  # softmax_temperature = 0.9,
  # balance_probabilities = FALSE,
  # average_before_softmax = FALSE,
  # training_set_limit = 10000,
  version = NULL,
  control = control_tab_icl2(),
  ...
) {
  options <- control
  options$num_quantiles <- num_quantiles
  # options$softmax_temperature <- softmax_temperature
  # options$balance_probabilities <- balance_probabilities
  # options$average_before_softmax <- average_before_softmax
  options <- check_fit_args(options)
  # check_number_whole(training_set_limit, min = 2, allow_infinite = TRUE)

  processed <- hardhat::mold(x, y)
  tr_ind <- sample_indicies(processed, size_limit = 1e4)
  if (length(tr_ind) > 0) {
    processed$predictors <- processed$predictors[tr_ind, , drop = FALSE]
    processed$outcomes <- processed$outcomes[tr_ind, , drop = FALSE]
  }

  tab_icl2_bridge(processed, options, version = version, ...)
}

# Formula method

#' @export
#' @rdname tab_icl2
tab_icl2.formula <- function(
  formula,
  data,
  num_quantiles = 30L,
  # softmax_temperature = 0.9,
  # balance_probabilities = FALSE,
  # average_before_softmax = FALSE,
  # training_set_limit = 10000,
  version = NULL,
  control = control_tab_icl2(),
  ...
) {
  options <- control
  options$num_quantiles <- num_quantiles
  # options$softmax_temperature <- softmax_temperature
  # options$balance_probabilities <- balance_probabilities
  # options$average_before_softmax <- average_before_softmax
  options <- check_fit_args(options)
  # check_number_whole(training_set_limit, min = 2, allow_infinite = TRUE)

  # Do not convert factors to indicators:
  bp <- hardhat::default_formula_blueprint(
    intercept = FALSE,
    allow_novel_levels = FALSE,
    indicators = "none",
    composition = "tibble"
  )
  processed <- hardhat::mold(formula, data, blueprint = bp)
  tr_ind <- sample_indicies(processed, size_limit = 1e4)
  if (length(tr_ind) > 0) {
    processed$predictors <- processed$predictors[tr_ind, , drop = FALSE]
    processed$outcomes <- processed$outcomes[tr_ind, , drop = FALSE]
  }

  tab_icl2_bridge(processed, options, version = version, ...)
}

# Recipe method

#' @export
#' @rdname tab_icl2
tab_icl2.recipe <- function(
  x,
  data,
  num_quantiles = 30L,
  # softmax_temperature = 0.9,
  # balance_probabilities = FALSE,
  # average_before_softmax = FALSE,
  # training_set_limit = 10000,
  version = NULL,
  control = control_tab_icl2(),
  ...
) {
  options <- control
  options$num_quantiles <- num_quantiles
  # options$softmax_temperature <- softmax_temperature
  # options$balance_probabilities <- balance_probabilities
  # options$average_before_softmax <- average_before_softmax
  options <- check_fit_args(options)
  # check_number_whole(training_set_limit, min = 2, allow_infinite = TRUE)

  processed <- hardhat::mold(x, data)
  tr_ind <- sample_indicies(processed, size_limit = 1e4)
  if (length(tr_ind) > 0) {
    processed$predictors <- processed$predictors[tr_ind, , drop = FALSE]
    processed$outcomes <- processed$outcomes[tr_ind, , drop = FALSE]
  }

  tab_icl2_bridge(processed, options, version = version, ...)
}

# ------------------------------------------------------------------------------
# Bridge

tab_icl2_bridge <- function(processed, options, version = NULL, ...) {
  rlang::check_dots_empty()

  if (!is.null(version)) {
    check_model_version(version)
  }

  predictors <- processed$predictors
  outcome <- processed$outcomes[[1]]

  check_data_constraints(predictors, outcome, options)

  res <- tab_icl2_impl(predictors, outcome, options, version = version)

  new_tab_icl(
    fit = res$fit,
    levels = res$lvls,
    training = res$train,
    logging = res$logging,
    blueprint = processed$blueprint
  )
}

# ------------------------------------------------------------------------------
# Implementation

tab_icl2_impl <- function(x, y, opts, version = NULL) {

  if (is.factor(y)) {
    max_classes <- out_dim <- nlevels(y)
    y_tt <- torch_tensor(as.numeric(y))

  } else if (is.numeric(y)) {
    max_classes <- 0L
    out_dim <- opts$num_quantiles
    y_tt <- torch_tensor(y)
  }
  mod_obj <- NanoTabICLv2(max_classes = max_classes, out_dim = out_dim)
  model_fit <- mod_obj(torch_tensor(as.matrix(x)), y_tt)

  # TODO check for failures
  res <- list(
    fit = model_fit,
    lvls = levels(y),
    train = dim(x),
    logging = c(r = msgs)
  )
  class(res) <- c("tab_icl2")
  res
}

#' @export
print.tab_icl2 <- function(x, ...) {
  type <- ifelse(is.null(x$levels), "Regression", "Classification")
  cli_inform("TabICL2 {type} Model")
  cat("\n")
  cli_inform("Training set\n\n")
  cli_inform(c(i = "{x$training[1]} data point{?s}"))
  cli_inform(c(i = "{x$training[2]} predictor{?s}"))

  if (!is.null(x$levels)) {
    cli_inform(c(i = "class levels: {.val {x$levels}}"))
  }

  invisible(x)
}

check_fit_args <- function(opts, call = rlang::caller_env()) {
  check_number_whole(
    # These arg names are deliberately different
    opts$num_quantiles,
    arg = "num_quantiles",
    min = 5,
    call = call
  )
  # opts$n_estimators <- as.integer(opts$n_estimators)
  #
  # check_number_decimal(
  #   opts$softmax_temperature,
  #   arg = "softmax_temperature",
  #   min = .Machine$double.eps,
  #   call = call
  # )
  #
  # check_logical(
  #   opts$balance_probabilities,
  #   arg = "balance_probabilities",
  #   call = call
  # )
  #
  # check_logical(
  #   opts$average_before_softmax,
  #   arg = "average_before_softmax",
  #   call = call
  # )

  # ------------------------------------------------------------------------------
  # There have been some argument name differences in the python package versions

  arg_names <- names(opts)

  # ------------------------------------------------------------------------------

  opts
}
