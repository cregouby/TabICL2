#' Predict using `TabICL2`
#'
#' @param object,x A `tab_icl2` object.
#'
#' @param new_data A data frame or matrix of new predictors.
#'
#' @param ... Not used, but required for extensibility.
#'
#' @return
#'
#' [predict()] returns a tibble of predictions and [augment()] appends the
#' columns in `new_data`. In either case, the number of rows in the tibble is
#' guaranteed to be the same as the number of rows in `new_data`.
#'
#' For regression data, the prediction is in the column `.pred`. For
#' classification, the class predictions are in `.pred_class` and the
#' probability estimates are in columns with the pattern `.pred_{level}` where
#' `level` is the levels of the outcome factor vector.
#'
#' @examples
#' # Minimal example for quick execution
#' car_train <- mtcars[ 1:5,   ]
#' car_test  <- mtcars[6, -1]
#'
#' \dontrun{
#' # Fit
#' if (torch_is_installed() & interactive()) {
#'  mod <- tab_icl2(mpg ~ cyl + log(drat), car_train)
#'
#'  # Predict
#'  predict(mod, car_test)
#'  augment(mod, car_test)
#' }
#' }
#'
#' @export
predict.tab_icl2 <- function(object, new_data, ...) {
  rlang::check_dots_empty()
  forged <- hardhat::forge(new_data, object$blueprint)$predictors
  res <- predict(object$fit, forged, object$levels)
  res
}

# ------------------------------------------------------------------------------
# Implementation

#' @export
predict.tab_icl2.regressor <- function(
  object,
  new_data,
  levels,
  ...
) {

    res <- object$predict(new_data)
    tibble::tibble(.pred = as.vector(res))

}

#' @export
predict.tab_icl2.classifier <- function(
  object,
  new_data,
  levels,
  ...
) {
  res <- object$predict_proba(new_data)
  colnames(res) <- paste0(".pred_", object$classes_)
  cls_ind <- apply(res, 1, which.max)
  res <- tibble::as_tibble(res)
  # TabICL2 will reorder the class levels; if the original factor has levels "b"
  # and "a", object$classes_ will have c("a", "b)
  res$.pred_class <- factor(object$classes_[cls_ind], levels = levels)

  res
}

#' @export
#' @rdname predict.tab_icl2
augment.tab_icl2 <- function(x, new_data, ...) {
  new_data <- tibble::new_tibble(new_data)
  res <- predict(x, new_data)
  res <- cbind(res, new_data)
  tibble::new_tibble(res)
}
