test_that("TabICLClassifier handles string inputs in a dataframe", {
  skip_if_not(exists("TabICLClassifier"), "TabICLClassifier not yet implemented")
  n <- 20
  set.seed(0)
  
  X <- data.frame(
    num1 = rnorm(n),
    num2 = rnorm(n),
    obj_str = sample(c("a", "b", "c"), size = n, replace = TRUE),
    str_dtype = sample(c("x", "y", "z"), size = n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  y <- sample(0:1, size = n, replace = TRUE)
  
  est <- TabICLClassifier()
  
  expect_type(X$obj_str, "character")
  
  expect_error({
    est$fit(X, y)
    preds <- est$predict(X)
  }, regexp = NA)
  
  expect_tensor(preds)
  expect_equal(length(preds), n)
})

test_that("TabICLClassifier handles string inputs in a heterogeneous matrix-like list", {
  skip_if_not(exists("TabICLClassifier"), "TabICLClassifier not yet implemented")
  n <- 20
  set.seed(0)
  
  X <- data.frame(
    num1 = rnorm(n),
    num2 = rnorm(n),
    obj_str = sample(c("a", "b", "c"), size = n, replace = TRUE),
    str_other = sample(c("u", "v", "w"), size = n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  y <- sample(0:1, size = n, replace = TRUE)
  
  est <- TabICLClassifier()
  
  expect_error({
    est$fit(X, y)
    preds <- est$predict(X)
  }, regexp = NA)
  
  expect_tensor(preds)
  expect_equal(length(preds), n)
})

test_that("TabICLRegressor handles string inputs in a dataframe", {
  skip_if_not(exists("TabICLRegressor"), "TabICLRegressor not yet implemented")
  n <- 20
  set.seed(0)
  
  X <- data.frame(
    num1 = rnorm(n),
    num2 = rnorm(n),
    obj_str = sample(c("a", "b", "c"), size = n, replace = TRUE),
    str_dtype = sample(c("x", "y", "z"), size = n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  y <- rnorm(n)
  
  est <- TabICLRegressor()
  
  expect_type(X$obj_str, "character")
  
  expect_error({
    est$fit(X, y)
    preds <- est$predict(X)
  }, regexp = NA)
  
  expect_tensor(preds)
  expect_equal(length(preds), n)
})

test_that("TabICLRegressor handles string inputs in a heterogeneous matrix-like list", {
  skip_if_not(exists("TabICLRegressor"), "TabICLRegressor not yet implemented")
  n <- 20
  set.seed(0)
  
  X <- data.frame(
    num1 = rnorm(n),
    num2 = rnorm(n),
    obj_str = sample(c("a", "b", "c"), size = n, replace = TRUE),
    str_other = sample(c("u", "v", "w"), size = n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  y <- rnorm(n)
  
  est <- TabICLRegressor()
  
  expect_error({
    est$fit(X, y)
    preds <- est$predict(X)
  }, regexp = NA)
  
  expect_tensor(preds)
  expect_equal(length(preds), n)
})