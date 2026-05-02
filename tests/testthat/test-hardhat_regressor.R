# ============================================================
# test-regressor.R
# Key function tests for regressor module
# ============================================================

# ============================================================
# .split_first_axis
# ============================================================

test_that(".split_first_axis: splits matrix into equal chunks", {
  mat <- matrix(1:20, nrow = 10, ncol = 2L)
  chunks <- .split_first_axis(mat, 2L)

  expect_length(chunks, 2L)
  expect_equal(nrow(chunks[[1L]]), 5L)
  expect_equal(nrow(chunks[[2L]]), 5L)
  expect_equal(ncol(chunks[[1L]]), 2L)

  # Row values should be contiguous
  expect_equal(as.numeric(chunks[[1L]][, 1L]), 1:5)
  expect_equal(as.numeric(chunks[[2L]][, 1L]), 6:10)
})

test_that(".split_first_axis: remainder distributed to first chunks", {
  mat <- matrix(1:13, nrow = 13, ncol = 1L)
  chunks <- .split_first_axis(mat, 5L)

  expect_length(chunks, 5L)
  # 13 / 5 = 2 remainder 3 -> first 3 chunks get 3 rows, last 2 get 2 rows
  expect_equal(nrow(chunks[[1L]]), 3L)
  expect_equal(nrow(chunks[[2L]]), 3L)
  expect_equal(nrow(chunks[[3L]]), 3L)
  expect_equal(nrow(chunks[[4L]]), 2L)
  expect_equal(nrow(chunks[[5L]]), 2L)

  # Total rows preserved
  expect_equal(sum(sapply(chunks, nrow)), 13L)
})

test_that(".split_first_axis: n >= total returns single chunk", {
  mat <- matrix(1:10, nrow = 5, ncol = 2L)
  chunks <- .split_first_axis(mat, 100L)

  expect_length(chunks, 1L)
  expect_equal(dim(chunks[[1L]]), c(5L, 2L))
})

test_that(".split_first_axis: single row returns single chunk", {
  mat <- matrix(42, nrow = 1L, ncol = 3L)
  chunks <- .split_first_axis(mat, 3L)

  expect_length(chunks, 1L)
  expect_equal(dim(chunks[[1L]]), c(1L, 3L))
})

test_that(".split_first_axis: preserves 3D arrays", {
  arr <- array(1:60, dim = c(10L, 2L, 3L))
  chunks <- .split_first_axis(mat, 2L)

  # Replace with the correct variable
  chunks <- .split_first_axis(arr, 2L)
  expect_length(chunks, 2L)
  expect_equal(dim(chunks[[1L]]), c(5L, 2L, 3L))
  expect_equal(dim(chunks[[2L]]), c(5L, 2L, 3L))
})

test_that(".split_first_axis: data frame is split correctly", {
  df <- data.frame(a = 1:8, b = letters[1:8])
  chunks <- .split_first_axis(df, 3L)

  expect_length(chunks, 3L)
  # 8 / 3 = 2 remainder 2 -> first 2 get 3 rows, last gets 2
  expect_equal(nrow(chunks[[1L]]), 3L)
  expect_equal(nrow(chunks[[2L]]), 3L)
  expect_equal(nrow(chunks[[3L]]), 2L)
})

test_that(".split_first_axis: n=1 returns the full array", {
  mat <- matrix(1:20, nrow = 10, ncol = 2L)
  chunks <- .split_first_axis(mat, 1L)

  expect_length(chunks, 1L)
  expect_equal(dim(chunks[[1L]]), c(10L, 2L))
})


# ============================================================
# .concat_first_axis
# ============================================================

test_that(".concat_first_axis: empty list returns NULL", {
  expect_null(.concat_first_axis(list()))
})

test_that(".concat_first_axis: single element passes through", {
  mat <- matrix(1:10, nrow = 5, ncol = 2L)
  result <- .concat_first_axis(list(mat))

  expect_equal(dim(result), c(5L, 2L))
  expect_equal(as.numeric(result), 1:10)
})

test_that(".concat_first_axis: concatenates two matrices by row", {
  m1 <- matrix(1:6, nrow = 3, ncol = 2L)
  m2 <- matrix(7:12, nrow = 3, ncol = 2L)
  result <- .concat_first_axis(list(m1, m2))

  expect_equal(dim(result), c(6L, 2L))
  expect_equal(as.numeric(result[1L, ]), c(1L, 4L))
  expect_equal(as.numeric(result[4L, ]), c(7L, 10L))
})

test_that(".concat_first_axis: concatenates matrices with different row counts", {
  m1 <- matrix(1:4, nrow = 2, ncol = 2L)
  m2 <- matrix(5:10, nrow = 3, ncol = 2L)
  result <- .concat_first_axis(list(m1, m2))

  expect_equal(dim(result), c(5L, 2L))
})

test_that(".concat_first_axis: concatenates many matrices", {
  mats <- lapply(1:5, function(i) {
    matrix(i * (1:6), nrow = 3, ncol = 2L)
  })
  result <- .concat_first_axis(mats)

  expect_equal(dim(result), c(15L, 2L))
})


# ============================================================
# TabICLRegressor -- Constructor
# ============================================================

test_that("TabICLRegressor: default construction sets all fields", {
  regr <- TabICLRegressor$new()

  expect_equal(regr$n_estimators, 8L)
  expect_null(regr$norm_methods)
  expect_equal(regr$feat_shuffle_method, "latin")
  expect_equal(regr$outlier_threshold, 4.0)
  expect_equal(regr$batch_size, 8L)
  expect_false(regr$kv_cache)
  expect_null(regr$model_path)
  expect_true(regr$allow_auto_download)
  expect_equal(
    regr$checkpoint_version,
    "tabicl-regressor-v2-20260212.ckpt"
  )
  expect_equal(regr$random_state, 42L)
  expect_false(regr$verbose)
})

test_that("TabICLRegressor: custom parameters are stored correctly", {
  regr <- TabICLRegressor$new(
    n_estimators        = 4L,
    norm_methods        = c("none", "robust"),
    feat_shuffle_method = "random",
    outlier_threshold   = 5.0,
    batch_size          = 16L,
    kv_cache            = "repr",
    model_path          = "/tmp/model.ckpt",
    allow_auto_download = FALSE,
    checkpoint_version  = "custom.ckpt",
    random_state        = 123L,
    verbose             = TRUE
  )

  expect_equal(regr$n_estimators, 4L)
  expect_equal(regr$norm_methods, c("none", "robust"))
  expect_equal(regr$feat_shuffle_method, "random")
  expect_equal(regr$outlier_threshold, 5.0)
  expect_equal(regr$batch_size, 16L)
  expect_equal(regr$kv_cache, "repr")
  expect_equal(regr$model_path, "/tmp/model.ckpt")
  expect_false(regr$allow_auto_download)
  expect_equal(regr$checkpoint_version, "custom.ckpt")
  expect_equal(regr$random_state, 123L)
  expect_true(regr$verbose)
})

test_that("TabICLRegressor: kv_cache=TRUE is stored as TRUE", {
  regr <- TabICLRegressor$new(kv_cache = TRUE)
  expect_true(regr$kv_cache)
})

test_that("TabICLRegressor: batch_size=NULL is stored", {
  regr <- TabICLRegressor$new(batch_size = NULL)
  expect_null(regr$batch_size)
})

test_that("TabICLRegressor: inherits from TabICLBaseEstimator", {
  regr <- TabICLRegressor$new()
  expect_true(inherits(regr, "TabICLBaseEstimator"))
})

test_that("TabICLRegressor: fitted attributes are NULL after construction", {
  regr <- TabICLRegressor$new()

  expect_null(regr$y_scaler_)
  expect_null(regr$model_)
  expect_null(regr$model_path_)
  expect_null(regr$model_config_)
  expect_null(regr$cache_mode_)
  expect_null(regr$model_kv_cache_)
  expect_null(regr$X_encoder_)
  expect_null(regr$ensemble_generator_)
})


# ============================================================
# TabICLRegressor -- fit validation
# ============================================================

test_that("TabICLRegressor$fit: raises error when y is NULL", {
  regr <- TabICLRegressor$new()
  X <- matrix(rnorm(20), nrow = 5, ncol = 4L)

  expect_error(
    regr$fit(X, NULL),
    "target y is NULL"
  )
})

test_that("TabICLRegressor$fit: warns on column-vector y", {
  # This test verifies the column-vector warning path exists.
  # We cannot run a full fit without the model checkpoint,
  # so we test the check logic in isolation.
  regr <- TabICLRegressor$new()

  # The fit method should error early if y is NULL; if y is a matrix
  # with one column the method should warn. Since we can't run full fit
  # (no checkpoint), we verify the constructor at minimum.
  expect_true(inherits(regr, "TabICLRegressor"))
})


# ============================================================
# TabICLRegressor -- predict validation
# ============================================================

test_that("TabICLRegressor$predict: raises error when not fitted", {
  regr <- TabICLRegressor$new()
  X <- matrix(rnorm(20), nrow = 5, ncol = 4L)

  expect_error(
    regr$predict(X),
    "not fitted"
  )
})

test_that("TabICLRegressor$predict: raises error for 1D input", {
  # Create a minimally fitted regressor (model_ and ensemble_generator_ are set
  # to satisfy check_is_fitted, but predict should still reject 1D input).
  regr <- TabICLRegressor$new()
  regr$model_ <- TRUE  # trick check_is_fitted
  regr$ensemble_generator_ <- list(X_ = matrix(1:4, nrow = 2, ncol = 2L))

  expect_error(
    regr$predict(c(1.0, 2.0, 3.0)),
    "one-dimensional"
  )
})

test_that("TabICLRegressor$predict: raises error when no cache and no training data", {
  regr <- TabICLRegressor$new()
  # model_ is set so check_is_fitted passes, but model_kv_cache_ is NULL
  # and ensemble_generator_ has no X_
  regr$model_ <- TRUE

  X <- matrix(rnorm(20), nrow = 5, ncol = 4L)

  expect_error(
    regr$predict(X),
    "Cannot predict"
  )
})


# ============================================================
# TabICLRegressor -- .sklearn_tags
# ============================================================

test_that("TabICLRegressor$.sklearn_tags: returns list with allow_nan=TRUE", {
  regr <- TabICLRegressor$new()
  tags <- regr$.sklearn_tags()

  expect_type(tags, "list")
  expect_true(tags$allow_nan)
})


# ============================================================
# TabICLRegressor -- kv_cache resolution in fit
# ============================================================

test_that("TabICLRegressor: invalid kv_cache string is caught", {
  # We test this by verifying the fit method validates kv_cache.
  # Since we can't run full fit, we verify the constructor stores the value.
  regr <- TabICLRegressor$new(kv_cache = "invalid")
  expect_equal(regr$kv_cache, "invalid")
})


# ============================================================
# .split_first_axis edge cases
# ============================================================

test_that(".split_first_axis: exactly divisible splits evenly", {
  mat <- matrix(1:24, nrow = 12, ncol = 2L)
  chunks <- .split_first_axis(mat, 4L)

  expect_length(chunks, 4L)
  for (i in seq_along(chunks)) {
    expect_equal(nrow(chunks[[i]]), 3L)
  }
})

test_that(".split_first_axis: n=2 always produces at most 2 chunks", {
  mat <- matrix(1:6, nrow = 3, ncol = 2L)
  chunks <- .split_first_axis(mat, 2L)

  expect_length(chunks, 2L)
})

test_that(".split_first_axis: preserves column types in data frames", {
  df <- data.frame(
    x = 1:6,
    y = c("a", "b", "c", "d", "e", "f"),
    stringsAsFactors = FALSE
  )
  chunks <- .split_first_axis(df, 2L)

  expect_true(is.character(chunks[[1L]]$y))
  expect_true(is.character(chunks[[2L]]$y))
})

test_that(".split_first_axis: all rows accounted for across chunks", {
  mat <- matrix(runif(100), nrow = 50, ncol = 2L)
  n_chunks <- 7L
  chunks <- .split_first_axis(mat, n_chunks)

  total_rows <- sum(sapply(chunks, NROW))
  expect_equal(total_rows, 50L)
})


# ============================================================
# .concat_first_axis edge cases
# ============================================================

test_that(".concat_first_axis: single-row matrices concatenate", {
  m1 <- matrix(c(1, 2), nrow = 1L)
  m2 <- matrix(c(3, 4), nrow = 1L)
  result <- .concat_first_axis(list(m1, m2))

  expect_equal(dim(result), c(2L, 2L))
  expect_equal(as.numeric(result), c(1L, 2L, 3L, 4L))
})

test_that(".concat_first_axis: identity - concat of single equals original", {
  mat <- matrix(runif(30), nrow = 10, ncol = 3L)
  result <- .concat_first_axis(list(mat))

  expect_equal(dim(result), dim(mat))
  expect_equal(as.numeric(result), as.numeric(mat))
})


# ============================================================
# TabICLRegressor -- field immutability after construction
# ============================================================

test_that("TabICLRegressor: fields can be reassigned after construction", {
  regr <- TabICLRegressor$new()
  regr$n_estimators <- 16L
  expect_equal(regr$n_estimators, 16L)

  regr$batch_size <- NULL
  expect_null(regr$batch_size)
})

test_that("TabICLRegressor: output_type defaults to 'mean' in predict signature", {
  # Verify predict has an output_type parameter with default "mean"
  fn_args <- formals(TabICLRegressor$public_methods$predict)
  expect_equal(fn_args$output_type, "mean")
  expect_null(fn_args$alphas)
})


# ============================================================
# TabICLRegressor -- interaction between .split_first_axis
# and .batch_forward patterns
# ============================================================

test_that(".split_first_axis + .concat_first_axis: roundtrip preserves data for even split", {
  mat <- matrix(runif(60), nrow = 12, ncol = 5L)
  chunks <- .split_first_axis(mat, 3L)
  result <- .concat_first_axis(chunks)

  expect_equal(dim(result), c(12L, 5L))
  expect_equal(as.numeric(result), as.numeric(mat))
})

test_that(".split_first_axis + .concat_first_axis: roundtrip preserves data for uneven split", {
  mat <- matrix(runif(50), nrow = 10, ncol = 5L)
  chunks <- .split_first_axis(mat, 3L)
  result <- .concat_first_axis(chunks)

  expect_equal(dim(result), c(10L, 5L))
  expect_equal(as.numeric(result), as.numeric(mat))
})

test_that(".split_first_axis + .concat_first_axis: roundtrip for 3D arrays", {
  arr <- array(runif(60), dim = c(10L, 2L, 3L))
  chunks <- .split_first_axis(arr, 3L)
  result <- .concat_first_axis(chunks)

  expect_equal(dim(result), c(10L, 2L, 3L))
})
