test_that("rotate_half_interleaved preserves shape and rotates correctly", {
  # Create test tensor: [2, 4] → pairs (1,2) and (3,4) in 1-based
  x <- torch_tensor(matrix(c(1, 2, 3, 4, 5, 6, 7, 8), nrow = 2, ncol = 4, byrow = TRUE),
                    dtype = torch_float())

  result <- rotate_half_interleaved(x)

  # Expected: [-2, 1, -4, 3] for first row, [-6, 5, -8, 7] for second
  expected_vals <- matrix(c(-2, 1, -4, 3, -6, 5, -8, 7), nrow = 2, ncol = 4, byrow = TRUE)

  expect_tensor_shape(result, c(2,4)) # same shape as x
  expect_equal_to_r(result, expected_vals)
})

test_that("rotate_half_contiguous works correctly", {
  x <- torch_tensor(matrix(c(1, 2, 3, 4, 5, 6, 7, 8), nrow = 2, ncol = 4, byrow = TRUE),
                    dtype = torch_float())

  result <- rotate_half_contiguous(x)

  # Split [1,2,3,4] → [1,2] and [3,4] → rotate → [-3,-4,1,2]
  expected_vals <- matrix(c(-3, -4, 1, 2, -7, -8, 5, 6), nrow = 2, ncol = 4, byrow = TRUE)

  expect_equal_to_r(result, expected_vals)
})


test_that("apply_rotary_emb preserves dtype and shape", {
  # Simple test: rotate a small tensor
  dim <- 4L
  seq_len <- 3L
  batch_size <- 2L
  freqs <- torch_tensor(matrix(c(0.1, 0.2, 0.3, 0.4), nrow = seq_len, ncol = dim %/% 2L),
                        dtype = torch_float())
  t <- torch_randn(batch_size, seq_len, dim)  # [batch=2, seq=3, dim=4]

  result <- apply_rotary_emb(freqs, t, interleaved = TRUE)

  expect_tensor_shape(result, c(2,3,4)) # same shape as t
  expect_tensor_dtype(result, torch_float())

  # Output should be numerically different from input (rotation happened)
  expect_false(torch_allclose(result, t, atol = 1e-1))
})

test_that("RotaryEmbedding initializes without error", {
  # Should not throw "attempt to apply non-function"
  expect_silent({
    rope <- RotaryEmbedding(dim = 64L)
  })

  expect_s3_class(rope, "RotaryEmbedding")
  expect_s3_class(rope, "nn_module")

  # Check that buffers are registered
  state <- rope$state_dict()

  # expect_true("freqs" %in% names(state))
  expect_false("cached_freqs" %in% names(state))
  expect_false("cached_scales" %in% names(state))
  expect_false("dummy" %in% names(state))

  # Check caches are accessibles and initialized
  expect_null(rope$cached_freqs)
  expect_null(rope$cached_scales)
  expect_tensor(rope$dummy)

  # Check cache is fed after forward
  positions <- torch_arange(10L)
  bin <- rope$forward(positions, seq_len = 10)  # starts caching

  # cached_freqs doit être un tenseur (plus NULL)
  expect_tensor(rope$cached_freqs)
  expect_tensor_shape(rope$cached_freqs, c(10, 64))

  # Check device tracking works
  if (requireNamespace("torch", quietly = TRUE) && torch::cuda_is_available()) {
    rope_cuda <- rope$to(device = "cuda")
    expect_equal(rope_cuda$device, "cuda")
  }
})

test_that("RotaryEmbedding forward produces expected frequencies", {
  rope <- RotaryEmbedding(dim = 8L, freqs_for = "lang", theta = 10000)
  positions <- torch_arange(5L)  # [0,1,2,3,4] in 0-based logic

  freqs <- rope$forward(positions)

  expect_tensor_shape(freqs, c(5, 8))

  # Check no NaN/Inf using proper tensor → scalar conversion
  expect_false(torch_max(torch_isnan(freqs))$item())
  expect_false(torch_max(torch_isinf(freqs))$item())

  # Check frequencies are finite and reasonable
  expect_true(torch_min(freqs > -100 & freqs < 100)$item())

})
