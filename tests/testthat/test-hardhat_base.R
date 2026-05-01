# if (!exists("InferenceConfig", mode = "function")) {
#   InferenceConfig <- R6::R6Class(
#     "InferenceConfig",
#     public = list(
#       update_from_dict = function(config) {
#         purrr::iwalk(config, function(value, key) {
#           self[[key]] <- value
#         })
#         invisible(self)
#       }
#     )
#   )
# }


test_that("softmax computes correct probabilities with default temperature", {
  x <- matrix(1:6, nrow = 2)
  result <- softmax(x)
  expect_equal(dim(result), dim(x))
  row_sums <- rowSums(result)
  expect_true(all(abs(row_sums - 1) < 1e-6))
  expect_true(all(result >= 0 & result <= 1))
})

test_that("softmax handles temperature scaling", {
  x <- matrix(1:6, nrow = 2)
  result_low_temp <- softmax(x, temperature = 0.1)
  result_high_temp <- softmax(x, temperature = 10)
  expect_true(max(result_low_temp) > max(result_high_temp))
})

test_that("softmax handles different axes", {
  x <- array(1:24, dim = c(2, 3, 4))
  result_axis1 <- softmax(x, axis = 1)
  result_axis2 <- softmax(x, axis = 2)
  result_axis3 <- softmax(x, axis = 3)
  expect_true(all(abs(apply(result_axis1, c(2, 3), sum) - 1) < 1e-6))
  expect_true(all(abs(apply(result_axis2, c(1, 3), sum) - 1) < 1e-6))
  expect_true(all(abs(apply(result_axis3, c(1, 2), sum) - 1) < 1e-6))
})

test_that("softmax handles numerical stability with large values", {
  x <- matrix(c(1000, 1001, 1002), nrow = 1)
  result <- softmax(x)
  expect_true(all(is.finite(result)))
  expect_true(all(abs(sum(result) - 1) < 1e-6))
})

test_that("TabICLBaseEstimator initializes with default parameters", {
  estimator <- TabICLBaseEstimator$new()
  expect_null(estimator$device)
  expect_equal(estimator$use_amp, "auto")
  expect_equal(estimator$use_fa3, "auto")
  expect_equal(estimator$verbose, 0)
  expect_null(estimator$offload_mode)
  expect_null(estimator$disk_offload_dir)
  expect_null(estimator$inference_config)
})

test_that("TabICLBaseEstimator initializes with custom parameters", {
  estimator <- TabICLBaseEstimator$new(
    device = "cuda:0",
    use_amp = TRUE,
    use_fa3 = FALSE,
    verbose = 2,
    offload_mode = "cpu",
    disk_offload_dir = "/tmp/cache"
  )
  expect_equal(estimator$device, "cuda:0")
  expect_true(estimator$use_amp)
  expect_false(estimator$use_fa3)
  expect_equal(estimator$verbose, 2)
  expect_equal(estimator$offload_mode, "cpu")
  expect_equal(estimator$disk_offload_dir, "/tmp/cache")
})

test_that(".resolve_device handles NULL device auto-detect", {
  estimator <- TabICLBaseEstimator$new(device = NULL)
  estimator$.resolve_device()
  expect_true(inherits(estimator$device_, "torch_device"))
  expect_true(estimator$device_$type %in% c("cpu", "cuda"))
})

test_that(".resolve_device handles string device specification", {
  estimator <- TabICLBaseEstimator$new(device = "cpu")
  estimator$.resolve_device()
  expect_equal(estimator$device_$type, "cpu")
})

test_that(".resolve_device handles torch_device object", {
  if (cuda_is_available()) {
    device_obj <- torch_device("cuda:0")
  } else {
    device_obj <- torch_device("cpu")
  }
  estimator <- TabICLBaseEstimator$new(device = device_obj)
  estimator$.resolve_device()
  expect_equal(estimator$device_, device_obj)
})

test_that(".resolve_amp_fa3 small data regime auto mode", {
  estimator <- TabICLBaseEstimator$new(use_amp = "auto", use_fa3 = "auto")
  estimator$n_samples_in_ <- 500
  estimator$n_features_in_ <- 30
  result <- estimator$.resolve_amp_fa3()
  expect_false(result$use_amp)
  expect_false(result$use_fa3)
})

test_that(".resolve_amp_fa3 medium data regime auto mode", {
  estimator <- TabICLBaseEstimator$new(use_amp = "auto", use_fa3 = "auto")
  estimator$n_samples_in_ <- 5000
  estimator$n_features_in_ <- 30
  result <- estimator$.resolve_amp_fa3()
  expect_true(result$use_amp)
  expect_false(result$use_fa3)
})

test_that(".resolve_amp_fa3 large data regime auto mode", {
  estimator <- TabICLBaseEstimator$new(use_amp = "auto", use_fa3 = "auto")
  estimator$n_samples_in_ <- 15000
  estimator$n_features_in_ <- 100
  result <- estimator$.resolve_amp_fa3()
  expect_true(result$use_amp)
  expect_true(result$use_fa3)
})

test_that(".resolve_amp_fa3 explicit boolean values override auto", {
  estimator <- TabICLBaseEstimator$new(use_amp = FALSE, use_fa3 = TRUE)
  estimator$n_samples_in_ <- 15000
  estimator$n_features_in_ <- 100
  result <- estimator$.resolve_amp_fa3()
  expect_false(result$use_amp)
  expect_true(result$use_fa3)
})

test_that(".resolve_amp_fa3 FA3 fallback when AMP explicitly disabled", {
  estimator <- TabICLBaseEstimator$new(use_amp = FALSE, use_fa3 = "auto")
  estimator$n_samples_in_ <- 5000
  estimator$n_features_in_ <- 30
  result <- estimator$.resolve_amp_fa3()
  expect_false(result$use_amp)
  expect_true(result$use_fa3)
})

test_that(".resolve_amp_fa3 handles missing n_samples_in_ n_features_in_", {
  estimator <- TabICLBaseEstimator$new(use_amp = "auto", use_fa3 = "auto")
  result <- estimator$.resolve_amp_fa3()
  expect_false(result$use_amp)
  expect_false(result$use_fa3)
})

test_that(".build_inference_config creates config from NULL inference_config", {
  estimator <- TabICLBaseEstimator$new(device = "cpu", verbose = 1)
  estimator$n_samples_in_ <- 1000
  estimator$n_features_in_ <- 50
  estimator$.resolve_device()
  estimator$.build_inference_config()
  expect_true(exists("inference_config_", envir = estimator, inherits = FALSE))
  expect_true(inherits(estimator$inference_config_, "InferenceConfig"))
})

test_that(".build_inference_config merges dict inference_config", {
  estimator <- TabICLBaseEstimator$new(
    device = "cpu",
    verbose = 1,
    inference_config = list(
      COL_CONFIG = list(extra_col_param = "test")
    )
  )
  estimator$n_samples_in_ <- 1000
  estimator$n_features_in_ <- 50
  estimator$.resolve_device()
  estimator$.build_inference_config()
  expect_true(exists("inference_config_", envir = estimator, inherits = FALSE))
  expect_true("extra_col_param" %in% names(estimator$inference_config_$COL_CONFIG))
})

test_that(".build_inference_config passes through InferenceConfig object", {
  custom_config <- InferenceConfig$new()
  estimator <- TabICLBaseEstimator$new(inference_config = custom_config)
  estimator$.resolve_device()
  estimator$.build_inference_config()
  expect_identical(estimator$inference_config_, custom_config)
})

test_that(".move_cache_to_device handles NULL cache gracefully", {
  estimator <- TabICLBaseEstimator$new(device = "cpu")
  estimator$.resolve_device()
  expect_silent(estimator$.move_cache_to_device())
})

test_that("save raises error when excluding training data without KV cache", {
  estimator <- TabICLBaseEstimator$new()
  estimator$model_kv_cache_ <- NULL
  expect_error(
    estimator$save("/tmp/test.rds", save_training_data = FALSE, save_kv_cache = FALSE),
    "Cannot exclude training data"
  )
})

test_that("save allows excluding training data when KV cache is present", {
  estimator <- TabICLBaseEstimator$new()
  estimator$model_kv_cache_ <- list(method1 = list(dummy = "cache"))
  tmp_path <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp_path), add = TRUE)
  expect_silent(
    estimator$save(tmp_path, save_training_data = FALSE, save_kv_cache = TRUE)
  )
  expect_true(file.exists(tmp_path))
})

test_that("save creates parent directories if needed", {
  estimator <- TabICLBaseEstimator$new()
  estimator$model_kv_cache_ <- list(method1 = list(dummy = "cache"))
  tmp_dir <- tempfile()
  tmp_path <- file.path(tmp_dir, "subdir", "model.rds")
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)
  expect_silent(estimator$save(tmp_path))
  expect_true(file.exists(tmp_path))
})

test_that(".more_tags returns non_deterministic flag", {
  estimator <- TabICLBaseEstimator$new()
  tags <- estimator$.more_tags()
  expect_true(is.list(tags))
  expect_true(tags$non_deterministic)
})

test_that(".sklearn_tags returns non_deterministic flag", {
  estimator <- TabICLBaseEstimator$new()
  tags <- estimator$.sklearn_tags()
  expect_true(is.list(tags))
  expect_true(tags$non_deterministic)
})

test_that("softmax handles empty input gracefully", {
  x <- numeric(0)
  expect_error(softmax(x), "argument is of length zero")
})

test_that("softmax handles single value", {
  x <- matrix(5, nrow = 1, ncol = 1)
  result <- softmax(x)
  expect_equal(result, 1)
})

test_that(".resolve_amp_fa3 handles boundary conditions", {
  estimator <- TabICLBaseEstimator$new(use_amp = "auto", use_fa3 = "auto")
  estimator$n_samples_in_ <- 1023
  estimator$n_features_in_ <- 59
  result <- estimator$.resolve_amp_fa3()
  expect_false(result$use_amp)
  expect_false(result$use_fa3)
  estimator$n_samples_in_ <- 1024
  result <- estimator$.resolve_amp_fa3()
  expect_true(result$use_amp)
})

test_that("error wrappers preserve call stack info", {
  expect_error({
    tryCatch({
      value_error("Test error with {val}", val = 123)
    }, error = function(e) {
      expect_match(conditionMessage(e), "Test error with 123")
      stop(e)
    })
  })
})

test_that("save method manages private flags correctly", {
  estimator <- TabICLBaseEstimator$new()
  estimator$model_kv_cache_ <- list(method1 = list(dummy = "cache"))
  tmp_path <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp_path), add = TRUE)
  expect_false(exists("._save_model_weights", envir = estimator$.__enclos_env__$private, inherits = FALSE))
  estimator$save(tmp_path, save_model_weights = TRUE, save_kv_cache = FALSE)
  expect_null(estimator$.__enclos_env__$private$._save_model_weights)
  expect_null(estimator$.__enclos_env__$private$._save_kv_cache)
})

test_that("full workflow init resolve build config", {
  estimator <- TabICLBaseEstimator$new(
    device = "cpu",
    use_amp = "auto",
    use_fa3 = "auto",
    verbose = 1
  )
  estimator$n_samples_in_ <- 5000
  estimator$n_features_in_ <- 40
  estimator$.resolve_device()
  amp_fa3 <- estimator$.resolve_amp_fa3()
  estimator$.build_inference_config()
  expect_true(inherits(estimator$device_, "torch_device"))
  expect_equal(estimator$device_$type, "cpu")
  expect_true(amp_fa3$use_amp)
  expect_false(amp_fa3$use_fa3)
  expect_true(exists("inference_config_", envir = estimator, inherits = FALSE))
})

test_that("softmax output compatible with expect_tensor helpers", {
  x <- matrix(rnorm(20), nrow = 4, ncol = 5)
  result <- softmax(x)
  expect_equal(dim(result), c(4, 5))
  expect_equal(typeof(result), "double")
  expect_true(all(result >= 0 & result <= 1))
  expect_true(all(abs(rowSums(result) - 1) < 1e-6))
})
