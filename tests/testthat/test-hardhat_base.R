# tests/testthat/helper-base.R

.mock_kv_cache <- function(device = "cpu", dtype = NULL) {
  list(
    col_cache = list(
      kv = list(
        layer1 = list(
          key = structure(list(dtype = dtype %||% torch_float32()), class = "mock_tensor"),
          value = structure(list(dtype = dtype %||% torch_float32()), class = "mock_tensor")
        )
      )
    ),
    to = function(device, dtype = NULL) {
      self
    }
  )
}

.skip_if_no_backend <- function(backend) {
  if (backend == "cuda" && !torch_cuda_is_available()) {
    testthat::skip("CUDA not available")
  }
  if (backend == "mps" && !torch_mps_is_available()) {
    testthat::skip("MPS not available")
  }
}

expect_tensor_properties <- function(x, expected_shape = NULL, expected_dtype = NULL) {
  if (inherits(x, "torch_tensor")) {
    if (!is.null(expected_shape)) {
      expect_tensor_shape(x, expected_shape)
    }
    if (!is.null(expected_dtype)) {
      expect_tensor_dtype(x, expected_dtype)
    }
  } else {
    if (!is.null(expected_shape)) {
      expect_equal(dim(x), expected_shape)
    }
    if (!is.null(expected_dtype)) {
      expect_equal(typeof(x), expected_dtype)
    }
  }
}