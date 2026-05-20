#' Predict using `TabICL2`
#'
#' @param object,x A `tab_icl_v2` object.
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
#' car_split <- rsample::initial_split(mtcars[ 1:6,   ])
#'
#' \dontrun{
#' # Fit
#' if (torch_is_installed() & interactive()) {
#'  mod <- tab_icl2(mpg ~ cyl + log(drat), car_split)
#'
#'  # Predict
#'  predict(mod, testing(car_split))
#'  augment(mod, testing(car_split))
#' }
#' }
#'
#' @export
predict.tab_icl_v2 <- function(object, new_data, ...) {
  rlang::check_dots_empty()

  forged <- hardhat::forge(new_data, object$blueprint)$predictors
  # TODO need to turn into a proper torch_dataset
  batch <- resolve_data(bind_rows(object$t_predictors, forged),
                        object$t_outcome)
  res <- predict(object$fit, batch, object$levels, object$training)
  res
}

# ------------------------------------------------------------------------------
# Implementation

#' @export
predict.tab_icl_v2.regressor <- function(object, new_data, levels, train_dim, ...) {

  raw_quantiles <- object(new_data$x$unsqueeze(1), new_data$y$unsqueeze(1))
  # remove the training set rows
  raw_quantiles <- raw_quantiles[, (train_dim[1] + 1):N, ]$squeeze()
  # TODO using num_quantiles assumes normalized outcome. To be documented and transported in options as well.
  num_quantiles <- raw_quantiles$shape[raw_quantiles$ndim]
  q2dist_nn <- quantile_to_distribution(num_quantiles = num_quantiles)
  dist <- q2dist_nn(raw_quantiles)
  tibble::tibble(.pred = as.numeric(dist$mean()))

}

#' @export
predict.tab_icl_v2.classifier <- function(object, new_data, levels, train_dim, ...) {

  res <- object(new_data$x$unsqueeze(1), new_data$y$unsqueeze(1))
  # remove the trainingset and move back to R
  res <- as_array(res[, (train_dim[1] + 1):N, ]$squeeze())

  colnames(res) <- paste0(".pred_", levels)
  cls_ind <- apply(res, 1, which.max)
  res <- tibble::as_tibble(res)
  # TabICL2 will reorder the class levels; if the original factor has levels "b"
  # and "a", object$classes_ will have c("a", "b)
  res$.pred_class <- factor(levels[cls_ind], levels = levels)

  res
}

#' @export
#' @rdname predict.tab_icl2
augment.tab_icl_v2 <- function(x, new_data, ...) {
  new_data <- tibble::new_tibble(new_data)
  res <- predict(x, new_data)
  res <- cbind(res, new_data)
  tibble::new_tibble(res)
}
