test_that("TabICLClassifier handles string inputs in a dataframe", {
  n <- 20
  set.seed(0)

  X <- data.frame(
    num1 = rnorm(n),
    num2 = rnorm(n),
    str = sample(c("a", "b", "c"), size = n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  y <- sample(0:1, size = n, replace = TRUE)

  est <- TabICLClassifier$new()

  expect_no_error(est$fit(X, y))
  expect_no_error(preds <- est$predict(X))

  expect_tensor(preds)
  expect_tensor_shape(preds, n)
})

test_that("TabICLClassifier handles string inputs in a heterogeneous matrix-like list", {
  n <- 20
  set.seed(0)

  X <- matrix(sample(LETTERS, size = n * 7, replace = TRUE),ncol = 7)
  y <- sample(0:1, size = n, replace = TRUE)

  est <- TabICLClassifier$new()

  expect_no_error(est$fit(X, y))
  expect_no_error(preds <- est$predict(X))

  expect_tensor(preds)
  expect_tensor_shape(preds, n)
})

test_that("TabICLRegressor handles string inputs in a dataframe", {
  skip_if_not(exists("TabICLRegressor"), "TabICLRegressor not yet implemented")
  n <- 20
  set.seed(0)

  X <- data.frame(
    num1 = rnorm(n),
    num2 = rnorm(n),
    str1 = sample(c("a", "b", "c"), size = n, replace = TRUE),
    str2 = sample(c("x", "y", "z"), size = n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  y <- rnorm(n)

  est <- TabICLRegressor$new()

  expect_no_error(est$fit(X, y))
  expect_no_error(preds <- est$predict(X))

  expect_tensor(preds)
  expect_tensor_shape(preds, n)
})

test_that("TabICLRegressor handles string inputs in a heterogeneous matrix-like list", {
  skip_if_not(exists("TabICLRegressor"), "TabICLRegressor not yet implemented")
  n <- 20
  set.seed(0)

  X <- data.frame(
    num1 = rnorm(n),
    num2 = rnorm(n),
    str1 = sample(c("a", "b", "c"), size = n, replace = TRUE),
    str2 = sample(c("x", "y", "z"), size = n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  y <- rnorm(n)

  est <- TabICLRegressor$new()

  expect_no_error(est$fit(X, y))
  expect_no_error(preds <- est$predict(X))

  expect_tensor(preds)
  expect_tensor_shape(preds, n)
})
