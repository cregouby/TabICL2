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
#'   * A __rsplit__ object from `rsample` package containing both the predictors and the outcome.
#'
#' @param formula A formula specifying the outcome terms on the left-hand side,
#' and the predictor terms on the right-hand side.
#'
#' @param config A list of model architecture options produced by
#'   [tab_icl2_config()].
#'
#' @param control A list of inference execution options produced by
#'   [tab_icl2_control()].
#'
#' @param version A character string for the model version. Currently
#'   unsupported.
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
#' Predictors do not require preprocessing; missing values and factor vectors
#' are allowed.
#'
#' ## Model Selection
#'
#' ### Selecting a model version
#'
#' Use the `version` argument to select a specific pretrained model version. For
#' example:
#'
#' \preformatted{
#'   # Use classifier version 2
#'   mod <- tab_icl2(predictors, outcome, version = "tabicl_classifier_v2")
#'
#'   # Use regressor version 2
#'   mod <- tab_icl2(predictors, outcome, version = "tabicl_regressor_v2")
#' }
#'
#' ### Pointing to a local model file
#'
#' If you have a model file on disk (e.g., downloaded for offline use), pass
#' its path via `tab_icl2_control(model_path = ...)`:
#'
#' \preformatted{
#'   control <- tab_icl2_control(model_path = "/path/to/model_file.ckpt")
#'   mod     <- tab_icl2(predictors, outcome, control = control)
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
#'   * `fit`: the object containing the model.
#'   * `levels`: a character string of class levels (or NULL for regression)
#'   * `training`: a vector with the training set dimensions.
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
#' @seealso [tab_icl2_config()], [tab_icl2_control()], [predict.tab_icl_v2()]
#' @examples
#' predictors <- mtcars[, -1]
#' outcome <- mtcars[, 1]
#' mtcars_split <- rsample::initial_split(mtcars)
#'
#' \dontrun{
#' if (torch_is_installed() & interactive()) {
#'  # XY interface
#'  mod <- tab_icl2(predictors, outcome)
#'  mod
#'
#'  # Formula interface
#'  mod2 <- tab_icl2(mpg ~ ., mtcars_split)
#'  mod2
#'
#'  # Recipes interface
#'  if (rlang::is_installed("recipes")) {
#'   suppressPackageStartupMessages(library(recipes))
#'   rec <-
#'    recipe(mpg ~ ., mtcars) %>%
#'    step_log(disp)
#'
#'   mod3 <- tab_icl2(rec, mtcars_split)
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
  config  = tab_icl2_config(),
  control = tab_icl2_control(),
  version = NULL,
  ...
) {
  options <- c(as.list(config), as.list(control))
  options <- check_fit_args(options)

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
  config  = tab_icl2_config(),
  control = tab_icl2_control(),
  version = NULL,
  ...
) {
  options <- c(as.list(config), as.list(control))
  options <- check_fit_args(options)

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
  config  = tab_icl2_config(),
  control = tab_icl2_control(),
  version = NULL,
  ...
) {
  options <- c(as.list(config), as.list(control))
  options <- check_fit_args(options)
  if (!inherits(data, "rsplit")) {
    cli::cli_abort(
      "With {.cls formula}}, the data object should be a rsample {.cls rsplit}, not
			{.cls {class(data)}}.",
      call = call
    )
  }

  # Do not convert factors to indicators:
  bp <- hardhat::default_formula_blueprint(
    intercept = FALSE,
    allow_novel_levels = FALSE,
    indicators = "none",
    composition = "tibble"
  )
  processed <- hardhat::mold(formula, rsample::training(data), blueprint = bp)
  processed_ts <- hardhat::forge(rsample::testing(data), blueprint = processed$blueprint)
  processed$predictors <- bind_rows(processed$predictors, processed_ts$predictor)
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
  config  = tab_icl2_config(),
  control = tab_icl2_control(),
  version = NULL,
  ...
) {
  options <- c(as.list(config), as.list(control))
  options <- check_fit_args(options)
  if (!inherits(data, "rsplit")) {
    cli::cli_abort(
      "With {.cls {class(x)}}, the data object should be a rsample {.cls rsplit}, not
			{.cls {class(data)}}.",
      call = call
    )
  }

  processed <- hardhat::mold(x, rsample::training(data))
  processed_ts <- hardhat::forge(rsample::testing(data), processed$blueprint)
  processed$predictors <- bind_rows(processed$predictors, processed_ts$predictor)
  tr_ind <- sample_indicies(processed, size_limit = 1e4)
  if (length(tr_ind) > 0) {
    processed$predictors <- processed$predictors[tr_ind, , drop = FALSE]
    processed$outcomes <- processed$outcomes[tr_ind, , drop = FALSE]
  }

  tab_icl2_bridge(processed, options, version = version)
}

# ------------------------------------------------------------------------------
# Bridge

tab_icl2_bridge <- function(processed, options, version = NULL, ...) {
  rlang::check_dots_empty()

  if (!is.null(version)) {
    check_model_version(version)
  }

  predictors <- processed$predictors
  outcome <- processed$outcomes

  check_data_constraints(predictors, outcome, options)

  res <- tab_icl2_impl(predictors, outcome, options, version = version)

  new_tab_icl2(
    fit = res$fit,
    levels = res$levels,
    training = res$train,
    blueprint = processed$blueprint,
    t_predictors = predictors,
    t_outcome = outcome
  )
}

# ------------------------------------------------------------------------------
# Implementation

tab_icl2_impl <- function(x, y, opts, version = NULL) {

  # TODO need to turn into a proper torch_dataset
  batch <- resolve_data(x, y)
  is_classification <- !is.null(batch$output_lvls)

  if (!is.null(version)) {
    is_cls_ckpt <- grepl("classifier", version, fixed = TRUE)
    is_reg_ckpt <- grepl("regressor",  version, fixed = TRUE)
    if (is_cls_ckpt && !is_classification) {
      cli_abort(
        c("{.val {version}} is a classifier checkpoint.",
          i = "Supply a factor outcome or switch to a regressor version.")
      )
    }
    if (is_reg_ckpt && is_classification) {
      cli_abort(
        c("{.val {version}} is a regressor checkpoint.",
          i = "Supply a numeric outcome or switch to a classifier version.")
      )
    }

    ckpt_path <- .resolve_checkpoint_path(
      model_path          = opts[["model_path"]],
      checkpoint_version  = version,
      allow_auto_download = TRUE
    )
    mod_obj <- torch::torch_load(ckpt_path)

  } else {
    arch <- list(
      embed_dim          = opts$embed_dim,
      col_n_block        = opts$col_n_block,
      row_n_block        = opts$row_n_block,
      icl_n_block        = opts$icl_n_block,
      col_n_head         = opts$col_n_head,
      row_n_head         = opts$row_n_head,
      icl_n_head         = opts$icl_n_head,
      feature_group_size = opts$feature_group_size,
      col_n_cls          = opts$col_n_cls,
      row_n_cls          = opts$row_n_cls
    )

    if (is_classification) {
      mod_obj <- do.call(
        NanoTabICLv2,
        c(arch, list(max_classes = sum(batch$output_dim), out_dim = sum(batch$output_dim)))
      )
    } else {
      mod_obj <- do.call(
        NanoTabICLv2,
        c(arch, list(max_classes = 0L, out_dim = opts$num_quantiles))
      )
    }
  }

  if (is_classification) {
    class(mod_obj) <- c(class(mod_obj), "tab_icl_v2.classifier")
  } else {
    class(mod_obj) <- c(class(mod_obj), "tab_icl_v2.regressor")
  }

  res <- list(
    fit    = mod_obj,
    levels = batch$output_lvls,
    train  = c(batch$y$shape[1], ncol(x))
  )
  class(res) <- c("tab_icl2")
  res
}

#' Transforms input data into a list of_tensors and parameters for model input
#'
#' The 3 torch tensors being
#' $x , $x_na_mask, $y
#'  and parameters being
#' cat_idx the vector of x categorical predictor index
#' cat_dims the vector of number of levels of each x categorical predictor
#' input_dim  the number of col in `x`
#' output_dim the `ncol(y)` in case of (multi-outcome) regression or
#'            the `nlevels(y)` in case of classification or
#'            the vector of `nlevels(y)` in case of multi-outcome classification
#'
#' @param x a data frame
#' @param y a response vector
#' @noRd
resolve_data <- function(x, y) {
  cat_idx <- which(sapply(x, is.factor))
  cat_dims <- sapply(cat_idx, function(i) nlevels(x[[i]]))
  # convert factors into integers
  if (length(cat_idx) > 0) {
    x[,cat_idx] <- sapply(cat_idx, function(i) as.integer(x[[i]]))
  } else {
    # prevent empty cat idx
    cat_idx <- 0L
    cat_dims <- 0L
  }
  x_tensor <- torch::torch_tensor(as.matrix(x), dtype = torch::torch_float())
  x_na_mask <- x %>% is.na %>% as.matrix %>% torch::torch_tensor(dtype = torch::torch_bool())

  # convert factors to integers, based on the class of target first column
  # TODO do not assume but assert type-consistency of all y cols
  if (is.factor(y[[1]])) {
    y_tensor <- torch::torch_tensor(sapply(y, function(i) as.integer(i)))
    if (is.atomic(y)) {
      output_lvls <- levels(y)
      output_dim <- nlevels(y)
    } else {
      output_lvls <- sapply(y, function(i) levels(i))
      output_dim <- sapply(y, function(i) nlevels(i))
    }
  } else {
    y_tensor <- torch::torch_tensor(as.matrix(y))
    output_lvls <- NULL
    output_dim <- ncol(y)
  }
  input_dim <- ncol(x)

  list(x = x_tensor, x_na_mask = x_na_mask, y = y_tensor$to(dtype = torch::torch_float())$squeeze(),
       input_dim = input_dim,
       cat_idx = cat_idx, cat_dims = cat_dims,
       output_lvls = output_lvls, output_dim = output_dim)
}

#' @export
print.tab_icl2 <- function(x, ...) {
  type <- ifelse(is.null(x$levels), "Regression", "Classification")
  cli_inform("TabICLv2 {type} Model")
  cat("\n")
  cli_inform("Training set\n\n")
  cli_inform(c(i = "{x$train[1]} data point{?s}"))
  cli_inform(c(i = "{x$train[2]} predictor{?s}"))

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
