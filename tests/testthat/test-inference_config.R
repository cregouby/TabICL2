test_that("MgrConfig validates and stores valid parameters", {
  cfg <- MgrConfig(
    use_amp        = TRUE,
    use_fa3        = FALSE,
    verbose        = FALSE,
    min_batch_size = 2L,
    safety_factor  = 0.8,
    offload        = "auto"
  )
  expect_s3_class(cfg, "MgrConfig")
  expect_type(cfg, "list")
  expect_identical(cfg$use_amp, TRUE)
  expect_identical(cfg$use_fa3, FALSE)
  expect_identical(cfg$offload, "auto")
  expect_identical(cfg$min_batch_size, 2L)
})

test_that("MgrConfig coerces numeric verbose to logical", {
  cfg <- MgrConfig(verbose = 0L)
  expect_identical(cfg$verbose, FALSE)
  cfg2 <- MgrConfig(verbose = 1L)
  expect_identical(cfg2$verbose, TRUE)
})

test_that("MgrConfig coerces numeric use_amp to logical", {
  cfg <- MgrConfig(use_amp = 0)
  expect_identical(cfg$use_amp, FALSE)
})

test_that("MgrConfig rejects invalid key", {
  expect_error(MgrConfig(invalid_key = 1), "Invalid config key")
})

test_that("MgrConfig rejects invalid type", {
  expect_error(MgrConfig(use_amp = "yes"), "use_amp")
  expect_error(MgrConfig(safety_factor = 2.0), "safety_factor")
  expect_error(MgrConfig(min_batch_size = 0), "min_batch_size")
})

test_that("MgrConfig rejects offload value not in allowed set", {
  expect_error(MgrConfig(offload = "invalid"), "offload")
})

test_that("MgrConfig accepts logical offload values", {
  cfg <- MgrConfig(offload = FALSE)
  expect_identical(cfg$offload, FALSE)
  cfg2 <- MgrConfig(offload = TRUE)
  expect_identical(cfg2$offload, TRUE)
})

test_that("MgrConfig accepts NULL for nullable fields", {
  cfg <- MgrConfig(device = NULL, disk_offload_dir = NULL, disk_dtype = NULL)
  expect_null(cfg$device)
  expect_null(cfg$disk_offload_dir)
})

test_that("MgrConfig accepts character device", {
  cfg <- MgrConfig(device = "cpu")
  expect_equal(cfg$device, "cpu")
})

test_that("MgrConfig is usable directly with do.call (no as.list needed)", {
  cfg <- MgrConfig(use_amp = TRUE, verbose = FALSE)
  # MgrConfig is a named list: $-access works
  expect_identical(cfg$use_amp, TRUE)
  expect_true(is.list(cfg))
})

test_that(".mgrcfg_update applies overrides correctly", {
  base <- MgrConfig(use_amp = TRUE, verbose = FALSE, min_batch_size = 1L)
  updated <- .mgrcfg_update(base, list(use_amp = FALSE, min_batch_size = 4L))
  expect_identical(updated$use_amp, FALSE)
  expect_identical(updated$min_batch_size, 4L)
  expect_identical(updated$verbose, FALSE)  # unchanged
})

test_that(".mgrcfg_update validates overrides", {
  base <- MgrConfig(use_amp = TRUE)
  expect_error(.mgrcfg_update(base, list(invalid_key = 1)), "Invalid config key")
})

test_that("InferenceConfig initialises with all three sub-configs", {
  cfg <- InferenceConfig$new()
  expect_s3_class(cfg, "InferenceConfig")
  expect_s3_class(cfg$COL_CONFIG, "MgrConfig")
  expect_s3_class(cfg$ROW_CONFIG, "MgrConfig")
  expect_s3_class(cfg$ICL_CONFIG, "MgrConfig")
})

test_that("InferenceConfig COL_CONFIG has offload = 'auto' by default", {
  cfg <- InferenceConfig$new()
  expect_equal(cfg$COL_CONFIG$offload, "auto")
})

test_that("InferenceConfig ROW_CONFIG and ICL_CONFIG have offload = FALSE by default", {
  cfg <- InferenceConfig$new()
  expect_identical(cfg$ROW_CONFIG$offload, FALSE)
  expect_identical(cfg$ICL_CONFIG$offload, FALSE)
})

test_that("InferenceConfig accepts MgrConfig argument", {
  custom <- MgrConfig(use_amp = FALSE, verbose = TRUE)
  cfg <- InferenceConfig$new(COL_CONFIG = custom)
  expect_identical(cfg$COL_CONFIG$use_amp, FALSE)
  expect_identical(cfg$COL_CONFIG$verbose, TRUE)
})

test_that("InferenceConfig accepts named list argument", {
  cfg <- InferenceConfig$new(ROW_CONFIG = list(use_amp = FALSE))
  expect_identical(cfg$ROW_CONFIG$use_amp, FALSE)
  # Other defaults are preserved
  expect_true(cfg$ROW_CONFIG$use_fa3)
})

test_that("InferenceConfig rejects invalid sub-config type", {
  expect_error(InferenceConfig$new(COL_CONFIG = "bad"), "MgrConfig")
})

test_that("InferenceConfig$update_from_dict applies updates to sub-configs", {
  cfg <- InferenceConfig$new()
  cfg$update_from_dict(list(
    COL_CONFIG = list(use_amp = FALSE, verbose = TRUE),
    ICL_CONFIG = list(safety_factor = 0.5)
  ))
  expect_identical(cfg$COL_CONFIG$use_amp, FALSE)
  expect_identical(cfg$COL_CONFIG$verbose, TRUE)
  expect_equal(cfg$ICL_CONFIG$safety_factor, 0.5)
  # Unchanged configs
  expect_true(cfg$ROW_CONFIG$use_amp)
})

test_that("InferenceConfig$update_from_dict rejects invalid keys", {
  cfg <- InferenceConfig$new()
  expect_error(cfg$update_from_dict(list(INVALID = list())), "Invalid InferenceConfig key")
})

test_that("inference_config() is an alias for InferenceConfig$new()", {
  cfg <- inference_config()
  expect_s3_class(cfg, "InferenceConfig")
  expect_s3_class(cfg$COL_CONFIG, "MgrConfig")
})

test_that("MgrConfig is usable with do.call on inference_manager$configure", {
  mgr <- inference_manager$new(enc_name = "tf_col", out_dim = 64L)
  cfg <- MgrConfig(
    device         = "cpu",
    use_amp        = FALSE,
    use_fa3        = FALSE,
    verbose        = FALSE,
    min_batch_size = 1L,
    safety_factor  = 0.8,
    offload        = FALSE
  )
  expect_silent(do.call(mgr$configure, as.list(cfg)))
})
