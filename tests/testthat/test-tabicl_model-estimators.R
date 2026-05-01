# Tests for TabICL estimators (Classifier and Regressor)
# Transcribed from Python pytest

test_that("TabICLClassifier supports NaNs", {

  est <- TabICLClassifier$new(random_state = 0L)

  X <- matrix(
    c(
      1.0, NA, 3.0,
      4.0, 5.0, NA,
      7.0, 8.0, 9.0,
      NA, 11.0, 12.0
    ),
    nrow = 4L, ncol = 3L, byrow = TRUE
  )

  y <- c(0L, 1L, 0L, 1L)

  est <- est$fit(X, y)
  y_pred <- est$predict(X)

  expect_equal(length(y_pred), length(y))
})


test_that("TabICLRegressor supports NaNs", {
  skip_if_not(exists("TabICLRegressor"), "TabICLRegressor not yet implemented")

  est <- TabICLRegressor$new(random_state = 0L)

  X <- matrix(
    c(
      1.0, NA, 3.0,
      4.0, 5.0, NA,
      7.0, 8.0, 9.0,
      NA, 11.0, 12.0
    ),
    nrow = 4L, ncol = 3L, byrow = TRUE
  )

  y <- c(0.1, 1.2, 2.3, 3.4)

  est <- est$fit(X, y)
  y_pred <- est$predict(X)

  expect_equal(length(y_pred), length(y))
})


test_that("TabICLClassifier supports logical inputs", {

  est <- TabICLClassifier$new(random_state = 0L)

  X <- matrix(
    c(
      TRUE, FALSE, TRUE,
      FALSE, TRUE, FALSE,
      TRUE, TRUE, FALSE,
      FALSE, FALSE, TRUE
    ),
    nrow = 4L, ncol = 3L, byrow = TRUE
  )

  y <- c(0L, 1L, 0L, 1L)

  est <- est$fit(X, y)
  y_pred <- est$predict(X)

  expect_equal(length(y_pred), length(y))
})


test_that("TabICLRegressor supports logical inputs", {
  skip_if_not(exists("TabICLRegressor"), "TabICLRegressor not yet implemented")

  est <- TabICLRegressor$new(random_state = 0L)

  X <- matrix(
    c(
      TRUE, FALSE, TRUE,
      FALSE, TRUE, FALSE,
      TRUE, TRUE, FALSE,
      FALSE, FALSE, TRUE
    ),
    nrow = 4L, ncol = 3L, byrow = TRUE
  )

  y <- c(0.1, 1.2, 2.3, 3.4)

  est <- est$fit(X, y)
  y_pred <- est$predict(X)

  expect_equal(length(y_pred), length(y))
})


test_that("TabICLClassifier supports mixed numeric types", {

  est <- TabICLClassifier$new(random_state = 0L)

  # In R, mixed integer and numeric automatically becomes numeric
  X <- matrix(
    c(
      1, 2.5, 3,
      4, 5.5, 6,
      7, 8.5, 9,
      10, 11.5, 12
    ),
    nrow = 4L, ncol = 3L, byrow = TRUE
  )

  y <- c(0L, 1L, 0L, 1L)

  est <- est$fit(X, y)
  y_pred <- est$predict(X)

  expect_equal(length(y_pred), length(y))
})


test_that("TabICLRegressor supports mixed numeric types", {
  skip_if_not(exists("TabICLRegressor"), "TabICLRegressor not yet implemented")

  est <- TabICLRegressor$new(random_state = 0L)

  # In R, mixed integer and numeric automatically becomes numeric
  X <- matrix(
    c(
      1, 2.5, 3,
      4, 5.5, 6,
      7, 8.5, 9,
      10, 11.5, 12
    ),
    nrow = 4L, ncol = 3L, byrow = TRUE
  )

  y <- c(0.1, 1.2, 2.3, 3.4)

  est <- est$fit(X, y)
  y_pred <- est$predict(X)

  expect_equal(length(y_pred), length(y))
})


test_that("TabICLClassifier supports character inputs", {

  est <- TabICLClassifier$new(random_state = 0L)

  X <- matrix(
    c(
      "1.0", "2.0", "3.0",
      "4.0", "5.0", "6.0",
      "7.0", "8.0", "9.0",
      "10.0", "11.0", "12.0"
    ),
    nrow = 4L, ncol = 3L, byrow = TRUE
  )

  y <- c(0L, 1L, 0L, 1L)

  est <- est$fit(X, y)
  y_pred <- est$predict(X)

  expect_equal(length(y_pred), length(y))
})


test_that("TabICLRegressor supports character inputs", {
  skip_if_not(exists("TabICLRegressor"), "TabICLRegressor not yet implemented")

  est <- TabICLRegressor$new(random_state = 0L)

  X <- matrix(
    c(
      "1.0", "2.0", "3.0",
      "4.0", "5.0", "6.0",
      "7.0", "8.0", "9.0",
      "10.0", "11.0", "12.0"
    ),
    nrow = 4L, ncol = 3L, byrow = TRUE
  )

  y <- c(0.1, 1.2, 2.3, 3.4)

  est <- est$fit(X, y)
  y_pred <- est$predict(X)

  expect_equal(length(y_pred), length(y))
})
