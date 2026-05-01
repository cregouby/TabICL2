set.seed(42)

# Helpers ------------------------------------------------------------------

make_X <- function(n = 50L, p = 4L) {
  matrix(rnorm(n * p), n, p)
}

make_X_with_outliers <- function(n = 100L, p = 3L) {
  X <- matrix(rnorm(n * p), n, p)
  X[1L, ] <- 100
  X
}

# TransformToNumerical -----------------------------------------------------

test_that("TransformToNumerical handles numeric matrix passthrough", {
  X <- make_X()
  t <- TransformToNumerical$new()
  Xt <- t$fit_transform(X)
  expect_true(is.matrix(Xt))
  expect_equal(dim(Xt), dim(X))
  expect_equal(Xt, X)
})

test_that("TransformToNumerical encodes factor columns in data.frame", {
  df <- data.frame(
    a = factor(c("cat", "dog", "cat", "bird")),
    b = c(1.0, 2.0, 3.0, 4.0),
    stringsAsFactors = FALSE
  )
  t  <- TransformToNumerical$new()
  Xt <- t$fit_transform(df)
  expect_true(is.matrix(Xt))
  expect_equal(ncol(Xt), 2L)
  # Unknown value at transform time becomes -1
  df2 <- data.frame(a = factor("whale"), b = 5.0, stringsAsFactors = FALSE)
  Xt2 <- t$transform(df2)
  expect_equal(unname(Xt2[1L, 1L]), -1)
})

test_that("TransformToNumerical imputes NA in numeric columns", {
  df <- data.frame(a = c(1.0, NA, 3.0))
  t  <- TransformToNumerical$new()
  Xt <- t$fit_transform(df)
  expect_false(anyNA(Xt))
})

# UniqueFeatureFilter ------------------------------------------------------

test_that("UniqueFeatureFilter removes constant columns", {
  X <- cbind(matrix(rnorm(50 * 3), 50, 3), rep(1, 50))
  f  <- UniqueFeatureFilter$new()
  Xf <- f$fit_transform(X)
  expect_equal(ncol(Xf), 3L)
  expect_equal(f$n_features_out_, 3L)
})

test_that("UniqueFeatureFilter keeps all columns when n_samples <= threshold", {
  X <- matrix(c(1, 1), nrow = 1L)
  f  <- UniqueFeatureFilter$new(threshold = 1L)
  Xf <- f$fit_transform(X)
  expect_equal(ncol(Xf), ncol(X))
})

test_that("UniqueFeatureFilter errors on wrong feature count at transform", {
  X <- make_X(50L, 4L)
  f  <- UniqueFeatureFilter$new()$fit(X)
  expect_error(f$transform(make_X(10L, 3L)), "features")
})

# OutlierRemover -----------------------------------------------------------

test_that("OutlierRemover clips extreme values", {
  X  <- make_X_with_outliers()
  or <- OutlierRemover$new(threshold = 4.0)
  Xt <- or$fit_transform(X)
  # Outlier at row 1 should be pulled inward
  expect_true(all(abs(Xt[1L, ]) < 100))
})

test_that("OutlierRemover output is same shape as input", {
  X  <- make_X()
  or <- OutlierRemover$new()
  Xt <- or$fit_transform(X)
  expect_equal(dim(Xt), dim(X))
})

# CustomStandardScaler -----------------------------------------------------

test_that("CustomStandardScaler produces ~zero mean and ~unit variance", {
  X   <- make_X(200L, 5L) * 10 + 3
  css <- CustomStandardScaler$new()
  Xt  <- css$fit_transform(X)
  expect_true(all(abs(colMeans(Xt)) < 0.01))
  expect_true(all(abs(apply(Xt, 2L, sd) - 1) < 0.01))
})

test_that("CustomStandardScaler clips output", {
  X   <- make_X(50L, 2L)
  css <- CustomStandardScaler$new(clip_min = -1, clip_max = 1)
  Xt  <- css$fit_transform(X)
  expect_true(all(Xt >= -1))
  expect_true(all(Xt <= 1))
})

test_that("CustomStandardScaler inverse_transform recovers original", {
  X   <- make_X(50L, 3L)
  css <- CustomStandardScaler$new(clip_min = -1e6, clip_max = 1e6)
  Xt  <- css$fit_transform(X)
  Xr  <- css$inverse_transform(Xt)
  expect_equal(Xr, X, tolerance = 1e-10)
})

test_that("CustomStandardScaler handles vector input", {
  x   <- rnorm(50)
  css <- CustomStandardScaler$new()
  xt  <- css$fit_transform(x)
  expect_true(is.vector(xt))
  expect_equal(length(xt), 50L)
})

# RTDLQuantileTransformer --------------------------------------------------

test_that("RTDLQuantileTransformer output has same shape", {
  X   <- make_X(100L, 5L)
  rqt <- RTDLQuantileTransformer$new(random_state = 1L)
  Xt  <- rqt$fit_transform(X)
  expect_equal(dim(Xt), dim(X))
})

test_that("RTDLQuantileTransformer output is approximately standard normal", {
  X   <- matrix(rexp(2000), 200, 10)
  rqt <- RTDLQuantileTransformer$new(n_quantiles = 100L, random_state = 1L)
  Xt  <- rqt$fit_transform(X)
  # Median of column means should be close to 0
  expect_true(abs(median(colMeans(Xt))) < 0.3)
})

# PreprocessingPipeline ----------------------------------------------------

test_that("PreprocessingPipeline (power) runs without error", {
  X  <- make_X(80L, 4L)
  pp <- PreprocessingPipeline$new(normalization_method = "power")
  Xt <- pp$fit_transform(X)
  expect_equal(dim(Xt), dim(X))
})

test_that("PreprocessingPipeline (quantile) runs without error", {
  X  <- make_X(80L, 4L)
  pp <- PreprocessingPipeline$new(normalization_method = "quantile")
  Xt <- pp$fit_transform(X)
  expect_equal(dim(Xt), dim(X))
})

test_that("PreprocessingPipeline (quantile_rtdl) runs without error", {
  X  <- make_X(80L, 4L)
  pp <- PreprocessingPipeline$new(normalization_method = "quantile_rtdl",
                                   random_state = 42L)
  Xt <- pp$fit_transform(X)
  expect_equal(dim(Xt), dim(X))
})

test_that("PreprocessingPipeline (robust) runs without error", {
  X  <- make_X(80L, 4L)
  pp <- PreprocessingPipeline$new(normalization_method = "robust")
  Xt <- pp$fit_transform(X)
  expect_equal(dim(Xt), dim(X))
})

test_that("PreprocessingPipeline (none) runs without error", {
  X  <- make_X(80L, 4L)
  pp <- PreprocessingPipeline$new(normalization_method = "none")
  Xt <- pp$fit_transform(X)
  expect_equal(dim(Xt), dim(X))
})

test_that("PreprocessingPipeline transform matches X_transformed_ for train data", {
  X  <- make_X(60L, 3L)
  pp <- PreprocessingPipeline$new(normalization_method = "power")
  pp$fit(X)
  expect_equal(pp$transform(X), pp$X_transformed_, tolerance = 1e-8)
})

test_that("PreprocessingPipeline errors on unknown method", {
  pp <- PreprocessingPipeline$new(normalization_method = "bogus")
  expect_error(pp$fit(make_X()), "normalization_method")
})

# Shuffler -----------------------------------------------------------------

test_that("Shuffler 'none' returns one identity permutation", {
  s <- Shuffler$new(5L, method = "none")
  p <- s$shuffle(10L)
  expect_equal(length(p), 1L)
  expect_equal(p[[1L]], 1:5)
})

test_that("Shuffler 'shift' returns circular permutations", {
  s <- Shuffler$new(4L, method = "shift", random_state = 1L)
  p <- s$shuffle(10L)
  expect_equal(length(p), 4L)
  for (perm in p) expect_setequal(perm, 1:4)
})

test_that("Shuffler 'random' returns valid permutations", {
  s <- Shuffler$new(6L, method = "random", random_state = 99L)
  p <- s$shuffle(5L)
  expect_equal(length(p), 5L)
  for (perm in p) expect_setequal(perm, 1:6)
})

test_that("Shuffler 'latin' returns valid permutations", {
  s <- Shuffler$new(4L, method = "latin", random_state = 7L)
  p <- s$shuffle(8L)
  for (perm in p) expect_setequal(perm, 1:4)
})

test_that("Shuffler falls back to 'random' when n_elements > max_elements_for_latin", {
  s <- Shuffler$new(10L, method = "latin", max_elements_for_latin = 5L,
                    random_state = 1L)
  p <- s$shuffle(3L)
  for (perm in p) expect_setequal(perm, 1:10)
})

# EnsembleGenerator --------------------------------------------------------

test_that("EnsembleGenerator fit runs for classification", {
  X <- make_X(60L, 5L)
  y <- sample(0:2, 60L, replace = TRUE)
  eg <- EnsembleGenerator$new(classification = TRUE, n_estimators = 4L,
                               norm_methods = "none", random_state = 1L)
  expect_no_error(eg$fit(X, y))
  expect_equal(eg$n_features_in_, 5L)
  expect_equal(eg$n_classes_, 3L)
})

test_that("EnsembleGenerator fit runs for regression", {
  X <- make_X(60L, 5L)
  y <- rnorm(60L)
  eg <- EnsembleGenerator$new(classification = FALSE, n_estimators = 3L,
                               norm_methods = "none", random_state = 2L)
  expect_no_error(eg$fit(X, y))
})

test_that("EnsembleGenerator transform mode='both' returns correct dimensions", {
  n_train <- 40L; n_test <- 10L; p <- 5L; n_est <- 3L
  X_train <- make_X(n_train, p)
  X_test  <- make_X(n_test,  p)
  y       <- sample(0:1, n_train, replace = TRUE)
  eg <- EnsembleGenerator$new(classification = TRUE, n_estimators = n_est,
                               norm_methods = "none", random_state = 3L)
  eg$fit(X_train, y)
  out <- eg$transform(X_test, mode = "both")
  method_out <- out[[1L]]
  expect_equal(dim(method_out$X), c(n_est, n_train + n_test, p))
  expect_equal(dim(method_out$y), c(n_est, n_train))
})

test_that("EnsembleGenerator transform mode='train' returns correct dimensions", {
  n_train <- 30L; p <- 4L; n_est <- 2L
  X_train <- make_X(n_train, p)
  y       <- rnorm(n_train)
  eg <- EnsembleGenerator$new(classification = FALSE, n_estimators = n_est,
                               norm_methods = "none", random_state = 4L)
  eg$fit(X_train, y)
  out <- eg$transform(mode = "train")
  method_out <- out[[1L]]
  expect_equal(dim(method_out$X), c(n_est, n_train, p))
  expect_equal(dim(method_out$y), c(n_est, n_train))
})

test_that("EnsembleGenerator transform mode='test' returns correct dimensions", {
  n_train <- 30L; n_test <- 8L; p <- 4L; n_est <- 2L
  X_train <- make_X(n_train, p)
  X_test  <- make_X(n_test,  p)
  y       <- rnorm(n_train)
  eg <- EnsembleGenerator$new(classification = FALSE, n_estimators = n_est,
                               norm_methods = "none", random_state = 5L)
  eg$fit(X_train, y)
  out <- eg$transform(X_test, mode = "test")
  method_out <- out[[1L]]
  expect_equal(dim(method_out$X), c(n_est, n_test, p))
})

test_that("EnsembleGenerator removes constant column via UniqueFeatureFilter", {
  p       <- 4L; n_train <- 50L
  X_train <- cbind(make_X(n_train, p - 1L), rep(0, n_train))
  X_test  <- cbind(make_X(10L, p - 1L), rep(0, 10L))
  y       <- rnorm(n_train)
  eg <- EnsembleGenerator$new(classification = FALSE, n_estimators = 2L,
                               norm_methods = "none", random_state = 6L)
  eg$fit(X_train, y)
  expect_equal(eg$n_features_in_, p - 1L)
  out <- eg$transform(X_test, mode = "both")
  expect_equal(dim(out[[1L]]$X)[[3L]], p - 1L)
})

test_that("EnsembleGenerator errors when mode requires X but X is NULL", {
  X <- make_X(); y <- rnorm(nrow(X))
  eg <- EnsembleGenerator$new(classification = FALSE, n_estimators = 2L,
                               norm_methods = "none")$fit(X, y)
  expect_error(eg$transform(X = NULL, mode = "both"), "required")
})

test_that("EnsembleGenerator feature_mask drops masked columns", {
  p       <- 5L; n_train <- 40L; n_test <- 8L
  X_train <- make_X(n_train, p)
  X_test  <- make_X(n_test,  p)
  y       <- rnorm(n_train)
  eg <- EnsembleGenerator$new(classification = FALSE, n_estimators = 2L,
                               norm_methods = "none", random_state = 7L)
  eg$fit(X_train, y)
  # Mask the last feature
  mask <- c(FALSE, FALSE, FALSE, FALSE, TRUE)
  out  <- eg$transform(X_test, mode = "both", feature_mask = mask)
  expect_equal(dim(out[[1L]]$X)[[3L]], p - 1L)
})

test_that("EnsembleGenerator class shuffle remaps y correctly", {
  n <- 30L
  X <- make_X(n, 3L)
  y <- as.numeric(rep(0:2, each = 10L))
  eg <- EnsembleGenerator$new(classification = TRUE, n_estimators = 4L,
                               norm_methods = "none", random_state = 8L)
  eg$fit(X, y)
  out <- eg$transform(mode = "train")
  y_out <- out[[1L]]$y
  # Each row of y_out should be a permutation of original class indices
  for (i in seq_len(nrow(y_out))) expect_setequal(unique(y_out[i, ]), 0:2)
})
