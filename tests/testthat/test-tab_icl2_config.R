test_that("tab_icl2_config() returns a classed list", {
  cfg <- tab_icl2_config()
  expect_s3_class(cfg, "tab_icl2_config")
  expect_type(cfg, "list")
})

test_that("tab_icl2_config() contains all architecture params as integers", {
  cfg <- tab_icl2_config()
  int_params <- c(
    "embed_dim", "col_n_block", "row_n_block", "icl_n_block",
    "col_n_head", "row_n_head", "icl_n_head", "feature_group_size",
    "col_n_cls", "row_n_cls", "num_quantiles"
  )
  expect_named(cfg, int_params)
  for (p in int_params) {
    expect_type(cfg[[p]], "integer")
  }
})

test_that("tab_icl2_config() defaults match NanoTabICLv2 defaults", {
  cfg <- tab_icl2_config()
  expect_identical(cfg$embed_dim,          128L)
  expect_identical(cfg$col_n_block,        3L)
  expect_identical(cfg$row_n_block,        3L)
  expect_identical(cfg$icl_n_block,        12L)
  expect_identical(cfg$feature_group_size, 3L)
  expect_identical(cfg$col_n_cls,          4L)
  expect_identical(cfg$row_n_cls,          128L)
  expect_identical(cfg$num_quantiles,      30L)
})

test_that("tab_icl2_config() coerces numeric to integer", {
  cfg <- tab_icl2_config(embed_dim = 256, col_n_block = 4)
  expect_identical(cfg$embed_dim,   256L)
  expect_identical(cfg$col_n_block, 4L)
})

test_that("tab_icl2_config() rejects non-positive integers", {
  expect_error(tab_icl2_config(embed_dim = 0L),     "embed_dim")
  expect_error(tab_icl2_config(col_n_block = 0L),   "col_n_block")
  expect_error(tab_icl2_config(num_quantiles = 4L), "num_quantiles")
})

test_that("tab_icl2_config() rejects non-numeric inputs", {
  expect_error(tab_icl2_config(embed_dim = "big"), "embed_dim")
  expect_error(tab_icl2_config(icl_n_head = TRUE), "icl_n_head")
})

test_that("tab_icl2_config() ... args are stored in returned list", {
  cfg <- tab_icl2_config(my_extra = 99L)
  expect_identical(cfg$my_extra, 99L)
})

test_that("tab_icl2_config() print works without error", {
  expect_no_error(print(tab_icl2_config()))
  expect_no_error(print(tab_icl2_config(embed_dim = 256L)))
})
