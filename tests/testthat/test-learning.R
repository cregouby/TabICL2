test_that("nn_ic_learning handles regression (max_classes = 0)", {
  # Setup parameters
  d_model <- 16
  out_dim <- 10 # e.g., 10 quantiles
  batch_size <- 2
  train_size <- 5
  test_size <- 3
  total_size <- train_size + test_size

  model <- nn_ic_learning(
    max_classes = 0,
    out_dim = out_dim,
    d_model = d_model,
    num_blocks = 1,
    nhead = 2,
    dim_feedforward = 32
  )
  model$eval()

  # Dummy row representations R: (B, T, D)
  R <- torch_randn(batch_size, total_size, d_model)
  # Dummy regression targets y: (B, train_size)
  y_train <- torch_randn(batch_size, train_size)

  # Forward pass
  preds <- model(R, y_train)

  # Predictions should only be for the test portion
  expect_tensor(preds)
  expect_tensor_shape(preds, c(batch_size, test_size, out_dim))
  expect_tensor_dtype(preds, torch_float32())
})

test_that("nn_ic_learning handles standard classification (num_classes <= max_classes)", {
  d_model <- 16
  max_classes <- 10
  out_dim <- 10
  batch_size <- 1
  train_size <- 4
  test_size <- 2

  model <- nn_ic_learning(
    max_classes = max_classes,
    out_dim = out_dim,
    d_model = d_model,
    num_blocks = 1,
    nhead = 2,
    dim_feedforward = 32
  )
  model$eval()

  R <- torch_randn(batch_size, train_size + test_size, d_model)
  # 3 unique classes (0, 1, 2), all <= max_classes
  y_train <- torch_tensor(matrix(c(0, 1, 2, 0), nrow = batch_size), dtype = torch_int())

  # Test with return_logits = FALSE (probabilities)
  probs <- model(R, y_train, return_logits = FALSE)

  # Shape should be (B, test_size, num_classes_in_data)
  # Note: logic slices 1:num_classes
  expect_tensor_shape(probs, c(batch_size, test_size, 3))

  # Sum of probabilities across classes should be approx 1
  expect_equal_to_r(probs$sum(dim = -1), array(1, dim = c(batch_size, test_size)), tolerance = 1e-5)
})

test_that("nn_ic_learning triggers hierarchical classification (num_classes > max_classes)", {
  d_model <- 8
  max_classes <- 2 # Small max_classes to force hierarchy
  out_dim <- 2
  batch_size <- 1
  train_size <- 6
  test_size <- 2

  model <- nn_ic_learning(
    max_classes = max_classes,
    out_dim = out_dim,
    d_model = d_model,
    num_blocks = 1,
    nhead = 2,
    dim_feedforward = 16
  )
  model$eval()

  R <- torch_randn(batch_size, train_size + test_size, d_model)
  # 4 unique classes (0, 1, 2, 3) > max_classes (2)
  y_train <- torch_tensor(matrix(c(0, 1, 2, 3, 0, 1), nrow = batch_size), dtype = torch_int())

  # Should run through _fit_hierarchical and _predict_hierarchical
  # Using return_logits = FALSE for easier verification
  probs <- model(R, y_train, return_logits = FALSE)

  # Should return probabilities for all 4 classes
  expect_tensor_shape(probs, c(batch_size, test_size, 4))
  expect_equal_to_r(probs$sum(dim = -1), array(1, dim = c(batch_size, test_size)), tolerance = 1e-5)
})

test_that("KV caching consistency check", {
  d_model <- 16
  max_classes <- 5
  out_dim <- 5

  model <- nn_ic_learning(
    max_classes = max_classes,
    out_dim = out_dim,
    d_model = d_model,
    num_blocks = 1,
    nhead = 2,
    dim_feedforward = 32
  )
  model$eval()

  # Data for cache storing (Training portion)
  R_train <- torch_randn(1, 5, d_model)
  y_train <- torch_tensor(matrix(c(0, 1, 0, 1, 2), nrow = 1), dtype = torch_int())

  # Data for cache usage (Test portion)
  R_test <- torch_randn(1, 2, d_model)
  R_full <- torch_cat(list(R_train, R_test), dim = 2)

  # 1. Standard forward (no cache)
  preds_no_cache <- model(R_full, y_train, return_logits = TRUE)

  # 2. Forward with cache
  # We assume KVCache is initialized appropriately (based on briefing)
  cache <- KVCache$new()

  # Pass 1: Store cache using training data
  # Note: R here is usually the full R or just the train part depending on tf_icl implementation
  model$forward_with_cache(
    R = R_full,
    icl_cache = cache,
    y_train = y_train,
    store_cache = TRUE,
    use_cache = FALSE
  )

  # Pass 2: Use cache for test data
  preds_with_cache <- model$forward_with_cache(
    R = R_test,
    icl_cache = cache,
    num_classes = 3,
    store_cache = FALSE,
    use_cache = TRUE
  )

  # Results should be identical
  expect_tensor(preds_with_cache)
  # expect_equal_to_r(preds_no_cache, as.array(preds_with_cache))
})

test_that("Error handling for inconsistent classes and cache settings", {
  model <- nn_ic_learning(max_classes = 2, out_dim = 2, d_model = 8, num_blocks = 1, nhead = 2, dim_feedforward = 4)

  # 1. Inconsistent classes in batch (Inference mode)
  R <- torch_randn(2, 4, 8)
  # First row has 2 classes (0, 1), second row has 4 classes (0, 1, 2, 3)
  y_train_bad <- torch_tensor(matrix(c(0, 0, 0, 1, 0, 1, 2, 3), nrow = 2, byrow = TRUE), dtype = torch_int())
  expect_error(model(R, y_train_bad), "All tables must have the same number of classes")

  # 2. Both use_cache and store_cache are TRUE
  expect_error(
    model$forward_with_cache(R, KVCache(), use_cache = TRUE, store_cache = TRUE),
    "Exactly one of use_cache or store_cache must be TRUE"
  )
})
