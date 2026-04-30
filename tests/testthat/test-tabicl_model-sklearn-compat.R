# Tests for sklearn compatibility and KV cache functionality
# Transcribed from Python pytest

# Helper function to generate classification data
make_classification <- function(n_samples, n_features, random_state = NULL) {
  if (!is.null(random_state)) {
    set.seed(random_state)
  }

  X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)
  y <- sample(c(0L, 1L), size = n_samples, replace = TRUE)

  list(X = X, y = y)
}

# Helper function to generate regression data
make_regression <- function(n_samples, n_features, random_state = NULL) {
  if (!is.null(random_state)) {
    set.seed(random_state)
  }

  X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)
  beta <- rnorm(n_features)
  y <- as.vector(X %*% beta + rnorm(n_samples, sd = 0.1))

  list(X = X, y = y)
}


# Sklearn-compatible estimator tests
# Note: R does not have a direct equivalent to sklearn's parametrize_with_checks
# These tests should be expanded manually based on specific requirements

test_that("TabICLClassifier basic sklearn compatibility", {
  skip_if_not(exists("TabICLClassifier"), "TabICLClassifier not yet implemented")

  # n_estimators=2 ensures full preprocessing and ensembling pipeline is tested
  est <- TabICLClassifier(n_estimators = 2L)

  # Test that estimator has required methods
  expect_true(is.function(est$fit) || is.environment(est))
  expect_true(is.function(est$predict) || is.environment(est))

  # Basic fit/predict test
  data <- make_classification(n_samples = 50L, n_features = 5L, random_state = 42L)
  X <- data$X
  y <- data$y

  est <- est$fit(X, y)
  y_pred <- est$predict(X)

  expect_equal(length(y_pred), length(y))
})


test_that("TabICLRegressor basic sklearn compatibility", {
  skip_if_not(exists("TabICLRegressor"), "TabICLRegressor not yet implemented")

  # n_estimators=2 ensures full preprocessing and ensembling pipeline is tested
  est <- TabICLRegressor(n_estimators = 2L)

  # Test that estimator has required methods
  expect_true(is.function(est$fit) || is.environment(est))
  expect_true(is.function(est$predict) || is.environment(est))

  # Basic fit/predict test
  data <- make_regression(n_samples = 50L, n_features = 5L, random_state = 42L)
  X <- data$X
  y <- data$y

  est <- est$fit(X, y)
  y_pred <- est$predict(X)

  expect_equal(length(y_pred), length(y))
})


# KV Cache tests for Classifier

test_that("TabICLClassifier KV cache mode 'kv' matches no cache", {
  skip_if_not(exists("TabICLClassifier"), "TabICLClassifier not yet implemented")

  data <- make_classification(n_samples = 50L, n_features = 5L, random_state = 42L)
  X <- data$X
  y <- data$y

  X_train <- X[1:40, , drop = FALSE]
  X_test <- X[41:50, , drop = FALSE]
  y_train <- y[1:40]

  # Without cache
  clf <- TabICLClassifier(n_estimators = 2L)
  clf <- clf$fit(X_train, y_train)
  pred_no_cache <- clf$predict_proba(X_test)

  # With kv cache
  clf_cached <- TabICLClassifier(n_estimators = 2L, kv_cache = "kv")
  clf_cached <- clf_cached$fit(X_train, y_train)
  pred_cached <- clf_cached$predict_proba(X_test)

  # Predictions should match within tolerance
  expect_equal(pred_no_cache, pred_cached, tolerance = 1e-4)
})


test_that("TabICLClassifier KV cache mode 'repr' matches no cache", {
  skip_if_not(exists("TabICLClassifier"), "TabICLClassifier not yet implemented")

  data <- make_classification(n_samples = 50L, n_features = 5L, random_state = 42L)
  X <- data$X
  y <- data$y

  X_train <- X[1:40, , drop = FALSE]
  X_test <- X[41:50, , drop = FALSE]
  y_train <- y[1:40]

  # Without cache
  clf <- TabICLClassifier(n_estimators = 2L)
  clf <- clf$fit(X_train, y_train)
  pred_no_cache <- clf$predict_proba(X_test)

  # With repr cache
  clf_cached <- TabICLClassifier(n_estimators = 2L, kv_cache = "repr")
  clf_cached <- clf_cached$fit(X_train, y_train)
  pred_cached <- clf_cached$predict_proba(X_test)

  # Predictions should match within tolerance
  expect_equal(pred_no_cache, pred_cached, tolerance = 1e-4)
})


# KV Cache tests for Regressor

test_that("TabICLRegressor KV cache mode 'kv' matches no cache", {
  skip_if_not(exists("TabICLRegressor"), "TabICLRegressor not yet implemented")

  data <- make_regression(n_samples = 50L, n_features = 5L, random_state = 42L)
  X <- data$X
  y <- data$y

  X_train <- X[1:40, , drop = FALSE]
  X_test <- X[41:50, , drop = FALSE]
  y_train <- y[1:40]

  # Without cache
  reg <- TabICLRegressor(n_estimators = 2L)
  reg <- reg$fit(X_train, y_train)
  pred_no_cache <- reg$predict(X_test)

  # With kv cache
  reg_cached <- TabICLRegressor(n_estimators = 2L, kv_cache = "kv")
  reg_cached <- reg_cached$fit(X_train, y_train)
  pred_cached <- reg_cached$predict(X_test)

  # Relaxed tolerance: kv cache changes float32 computation order
  expect_equal(pred_no_cache, pred_cached, tolerance = 1e-4)
})


test_that("TabICLRegressor KV cache mode 'repr' matches no cache", {
  skip_if_not(exists("TabICLRegressor"), "TabICLRegressor not yet implemented")

  data <- make_regression(n_samples = 50L, n_features = 5L, random_state = 42L)
  X <- data$X
  y <- data$y

  X_train <- X[1:40, , drop = FALSE]
  X_test <- X[41:50, , drop = FALSE]
  y_train <- y[1:40]

  # Without cache
  reg <- TabICLRegressor(n_estimators = 2L)
  reg <- reg$fit(X_train, y_train)
  pred_no_cache <- reg$predict(X_test)

  # With repr cache
  reg_cached <- TabICLRegressor(n_estimators = 2L, kv_cache = "repr")
  reg_cached <- reg_cached$fit(X_train, y_train)
  pred_cached <- reg_cached$predict(X_test)

  # Relaxed tolerance: kv cache changes float32 computation order
  expect_equal(pred_no_cache, pred_cached, tolerance = 1e-4)
})
