test_that("tab_icl2_control() returns a classed list with col/row/icl stages", {
  cfg <- tab_icl2_control()
  expect_s3_class(cfg, "tab_icl2_control")
  expect_type(cfg, "list")
  expect_true(all(c("col", "row", "icl", "ignore_pretraining_limits") %in% names(cfg)))
  expect_type(cfg$col, "list")
  expect_type(cfg$row, "list")
  expect_type(cfg$icl, "list")
})

test_that("COL stage defaults to offload = 'auto'", {
  cfg <- tab_icl2_control()
  expect_equal(cfg$col$offload, "auto")
})

test_that("ROW and ICL stages default to offload = FALSE", {
  cfg <- tab_icl2_control()
  expect_identical(cfg$row$offload, FALSE)
  expect_identical(cfg$icl$offload, FALSE)
})

test_that("ignore_pretraining_limits defaults to FALSE", {
  cfg <- tab_icl2_control()
  expect_identical(cfg$ignore_pretraining_limits, FALSE)
})

test_that("ignore_pretraining_limits = TRUE is accepted and stored", {
  cfg <- tab_icl2_control(ignore_pretraining_limits = TRUE)
  expect_identical(cfg$ignore_pretraining_limits, TRUE)
})

test_that("ignore_pretraining_limits must be logical", {
  expect_error(tab_icl2_control(ignore_pretraining_limits = "yes"), "ignore_pretraining_limits")
})

test_that("top-level params propagate to all stages", {
  cfg <- tab_icl2_control(use_amp = FALSE, verbose = TRUE, device = "cpu")
  expect_identical(cfg$col$use_amp, FALSE)
  expect_identical(cfg$row$use_amp, FALSE)
  expect_identical(cfg$icl$use_amp, FALSE)
  expect_identical(cfg$col$verbose, TRUE)
  expect_identical(cfg$col$device, "cpu")
})

test_that("offload param applies only to COL stage", {
  cfg <- tab_icl2_control(offload = "cpu")
  expect_equal(cfg$col$offload, "cpu")
  expect_identical(cfg$row$offload, FALSE)
  expect_identical(cfg$icl$offload, FALSE)
})

test_that("offload = FALSE and offload = TRUE are accepted", {
  cfg_f <- tab_icl2_control(offload = FALSE)
  expect_identical(cfg_f$col$offload, FALSE)
  cfg_t <- tab_icl2_control(offload = TRUE)
  expect_identical(cfg_t$col$offload, TRUE)
})

test_that("col_config overrides apply to COL stage only", {
  cfg <- tab_icl2_control(col_config = list(use_amp = FALSE, verbose = TRUE))
  expect_identical(cfg$col$use_amp, FALSE)
  expect_identical(cfg$col$verbose, TRUE)
  expect_identical(cfg$row$use_amp, TRUE)
  expect_identical(cfg$icl$use_amp, TRUE)
})

test_that("row_config and icl_config apply to their respective stages", {
  cfg <- tab_icl2_control(
    row_config = list(safety_factor = 0.5),
    icl_config = list(min_batch_size = 4L)
  )
  expect_equal(cfg$row$safety_factor, 0.5)
  expect_identical(cfg$icl$min_batch_size, 4L)
  expect_equal(cfg$col$safety_factor, 0.8)
  expect_identical(cfg$row$min_batch_size, 1L)
})

test_that("col_config can override offload for COL stage", {
  cfg <- tab_icl2_control(col_config = list(offload = "disk"))
  expect_equal(cfg$col$offload, "disk")
})

test_that("row_config can override offload for ROW stage", {
  cfg <- tab_icl2_control(row_config = list(offload = "auto"))
  expect_equal(cfg$row$offload, "auto")
})

test_that("invalid col_config key is rejected", {
  expect_error(
    tab_icl2_control(col_config = list(invalid_key = 1)),
    "Invalid inference config key"
  )
})

test_that("invalid offload value is rejected", {
  expect_error(tab_icl2_control(offload = "invalid"), "offload")
})

test_that("safety_factor out of range is rejected", {
  expect_error(tab_icl2_control(safety_factor = 1.5),  "safety_factor")
  expect_error(tab_icl2_control(safety_factor = -0.1), "safety_factor")
})

test_that("use_amp must be logical", {
  expect_error(tab_icl2_control(use_amp = "yes"), "use_amp")
})

test_that("non-list col_config is rejected", {
  expect_error(tab_icl2_control(col_config = "bad"), "col_config")
})

test_that("min_batch_size must be a positive integer", {
  expect_error(tab_icl2_control(min_batch_size = 0L), "min_batch_size")
  expect_error(tab_icl2_control(min_batch_size = "a"), "min_batch_size")
})

test_that("... args are stored in returned list", {
  cfg <- tab_icl2_control(my_extra = 42L)
  expect_identical(cfg$my_extra, 42L)
})

test_that("min_batch_size coerces to integer", {
  cfg <- tab_icl2_control(min_batch_size = 2)
  expect_identical(cfg$col$min_batch_size, 2L)
})

test_that("inference_config() is an alias for tab_icl2_control()", {
  cfg <- inference_config()
  expect_s3_class(cfg, "tab_icl2_control")
  expect_true(all(c("col", "row", "icl") %in% names(cfg)))
})

test_that("print works without error", {
  expect_no_error(print(tab_icl2_control()))
  expect_no_error(print(tab_icl2_control(use_amp = FALSE, ignore_pretraining_limits = TRUE)))
})
