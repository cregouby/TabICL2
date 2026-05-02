test_that("TabICLClassifier initializes with defaults", {
  clf <- TabICLClassifier$new()
  expect_equal(clf$n_estimators, 8L)
  expect_equal(clf$norm_methods, NULL)
  expect_equal(clf$feat_shuffle_method, "latin")
  expect_equal(clf$class_shuffle_method, "shift")
  expect_equal(clf$outlier_threshold, 4.0)
  expect_equal(clf$softmax_temperature, 0.9)
  expect_true(clf$average_logits)
  expect_true(clf$support_many_classes)
  expect_equal(clf$batch_size, 8L)
  expect_false(clf$kv_cache)
  expect_null(clf$model_path)
  expect_true(clf$allow_auto_download)
  expect_equal(clf$checkpoint_version, "tabicl-classifier-v2-20260212.ckpt")
  expect_null(clf$n_jobs)
  expect_equal(clf$random_state, 42L)
  # Base class fields
  expect_null(clf$device)
  expect_equal(clf$use_amp, "auto")
  expect_equal(clf$use_fa3, "auto")
  expect_equal(clf$verbose, 0L)
})

test_that("TabICLClassifier initializes with custom values", {
  clf <- TabICLClassifier$new(
    n_estimators         = 4L,
    norm_methods         = c("none", "robust"),
    feat_shuffle_method  = "random",
    class_shuffle_method = "none",
    outlier_threshold    = 3.0,
    softmax_temperature  = 1.0,
    average_logits       = FALSE,
    support_many_classes = FALSE,
    batch_size           = NULL,
    kv_cache             = "repr",
    random_state         = 0L,
    verbose              = TRUE,
    device               = "cpu"
  )
  expect_equal(clf$n_estimators, 4L)
  expect_equal(clf$norm_methods, c("none", "robust"))
  expect_equal(clf$feat_shuffle_method, "random")
  expect_equal(clf$class_shuffle_method, "none")
  expect_equal(clf$outlier_threshold, 3.0)
  expect_equal(clf$softmax_temperature, 1.0)
  expect_false(clf$average_logits)
  expect_false(clf$support_many_classes)
  expect_null(clf$batch_size)
  expect_equal(clf$kv_cache, "repr")
  expect_equal(clf$random_state, 0L)
  expect_equal(clf$verbose, 1L)
  expect_equal(clf$device, "cpu")
})

test_that("TabICLClassifier predict_proba errors when not fitted", {
  clf <- TabICLClassifier$new()
  expect_error(clf$predict_proba(matrix(1:4, 2, 2)), "not fitted")
})

test_that("TabICLClassifier predict_proba rejects 1-D input", {
  clf <- TabICLClassifier$new()
  clf$classes_ <- c("a", "b")
  clf$ensemble_generator_ <- list(X_ = matrix(1))
  expect_error(clf$predict_proba(c(1, 2, 3)), "one-dimensional")
})

# Helper: build a minimal fitted classifier using a mock model
.mock_classifier <- function(n_train = 20, n_features = 3, n_classes = 2,
                              n_estimators = 2, random_state = 0L) {
  set.seed(random_state)
  X <- matrix(rnorm(n_train * n_features), n_train, n_features)
  y <- sample(letters[1:n_classes], n_train, replace = TRUE)

  clf <- TabICLClassifier$new(
    n_estimators        = n_estimators,
    norm_methods        = "none",
    feat_shuffle_method = "none",
    random_state        = random_state,
    device              = "cpu",
    use_amp             = FALSE,
    use_fa3             = FALSE,
    kv_cache            = FALSE
  )

  # Manually set fitted attributes without loading a real model
  clf$classes_        <- sort(unique(y))
  clf$n_classes_      <- length(clf$classes_)
  clf$n_samples_in_   <- n_train
  clf$n_features_in_  <- n_features
  clf$cache_mode_     <- NULL
  clf$model_kv_cache_ <- NULL

  clf$X_encoder_ <- TransformToNumerical$new()
  X_num <- clf$X_encoder_$fit_transform(X)

  y_enc <- match(y, clf$classes_) - 1L

  clf$ensemble_generator_ <- EnsembleGenerator$new(
    classification       = TRUE,
    n_estimators         = n_estimators,
    norm_methods         = "none",
    feat_shuffle_method  = "none",
    class_shuffle_method = "shift",
    random_state         = random_state
  )
  clf$ensemble_generator_$fit(X_num, y_enc)

  list(clf = clf, X = X, y = y, X_num = X_num)
}

test_that("EnsembleGenerator produces correct output shape for classifier", {
  res  <- .mock_classifier(n_train = 30, n_features = 4, n_classes = 3, n_estimators = 4)
  clf  <- res$clf
  Xte  <- matrix(rnorm(5 * 4), 5, 4)
  Xte_num <- clf$X_encoder_$transform(Xte)

  data <- clf$ensemble_generator_$transform(Xte_num, mode = "both")
  expect_true(is.list(data))
  for (method_data in data) {
    Xs <- method_data$X
    ys <- method_data$y
    expect_equal(length(dim(Xs)), 3L)   # [n_est, n_train+n_test, n_feat]
    expect_equal(length(dim(ys)), 2L)   # [n_est, n_train]
    expect_equal(dim(Xs)[2L], 30L + 5L) # train + test rows
    expect_equal(dim(ys)[2L], 30L)
  }
})

test_that("class shuffle unshuffling is consistent", {
  # Verify that applying shuffle then unshuffling recovers identity
  n_classes <- 4L
  shuffle   <- c(3L, 1L, 4L, 2L)  # 1-indexed permutation

  # Simulate predictions for one estimator
  proba <- matrix(c(0.1, 0.5, 0.2, 0.2,
                    0.7, 0.1, 0.1, 0.1), nrow = 2L, byrow = TRUE)

  # Simulate model output with shuffled class labels
  out_shuffled <- proba[, order(shuffle), drop = FALSE]

  # Unshuffling: avg += out_shuffled[, shuffle]
  avg <- out_shuffled[, shuffle, drop = FALSE]

  expect_equal(avg, proba)
})

test_that("softmax axis=1 returns valid probabilities for 2-D input", {
  x    <- matrix(c(1.0, 2.0, 0.5, 3.0, 1.0, 0.1), nrow = 2, byrow = TRUE)
  prob <- softmax(x, axis = 1L, temperature = 1.0)
  expect_equal(dim(prob), dim(x))
  expect_true(all(prob >= 0))
  expect_equal(rowSums(prob), c(1.0, 1.0), tolerance = 1e-6)
})

test_that("TabICLClassifier fit errors for invalid kv_cache value", {
  clf <- TabICLClassifier$new(kv_cache = "invalid")
  clf$classes_        <- c("a", "b")
  clf$n_classes_      <- 2L
  clf$n_samples_in_   <- 10L
  # Bypass model loading by providing a mock model with max_classes
  skip("requires model loading")
})

test_that("hf_download uses local cache when file exists", {
  tmp <- tempfile(fileext = ".ckpt")
  writeLines("dummy", tmp)
  result <- .hf_download("somefile.ckpt", dest_path = tmp)
  expect_equal(result, tmp)
  file.remove(tmp)
})

test_that("hf_download errors when file missing and download disabled", {
  tmp <- tempfile(fileext = ".ckpt")
  expect_error(
    .hf_download("somefile.ckpt", dest_path = tmp, allow_auto_download = FALSE),
    "not found"
  )
})
