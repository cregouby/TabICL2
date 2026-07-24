# tests/testthat/test-rope.R

test_that("rotate_half_interleaved preserves shape and rotates correctly", {
  # Input: [1, 2, 3, 4] -> pairs (1,2) and (3,4)
  # Expected: [-2, 1, -4, 3]
  x <- torch_tensor(matrix(c(1, 2, 3, 4, 5, 6, 7, 8), nrow = 2, ncol = 4, byrow = TRUE),
                    dtype = torch_float())

  result <- TabICL2:::rotate_half_interleaved(x)

  expect_tensor(result)
  expect_tensor_shape(result, c(2L, 4L))
  expect_tensor_dtype(result, torch_float())

  expected <- torch_tensor(matrix(c(-2, 1, -4, 3, -6, 5, -8, 7), nrow = 2, ncol = 4, byrow = TRUE),
                           dtype = torch_float())
  expect_equal_to_r(result, as.array(expected))
})

test_that("rotate_half_contiguous preserves shape and rotates correctly", {
  # Input: [1, 2, 3, 4] -> halves [1, 2] and [3, 4]
  # Expected: [-3, -4, 1, 2]
  x <- torch_tensor(matrix(c(1, 2, 3, 4, 5, 6, 7, 8), nrow = 2, ncol = 4, byrow = TRUE),
                    dtype = torch_float())

  result <- TabICL2:::rotate_half_contiguous(x)

  expect_tensor(result)
  expect_tensor_shape(result, c(2L, 4L))

  expected <- torch_tensor(matrix(c(-3, -4, 1, 2, -7, -8, 5, 6), nrow = 2, ncol = 4, byrow = TRUE),
                           dtype = torch_float())
  expect_equal_to_r(result, as.array(expected))
})

test_that("apply_rotary_emb preserves shape and dtype (interleaved)", {
  dim <- 4L
  seq_len <- 3L
  batch_size <- 2L

  freqs <- torch_randn(seq_len, dim, dtype = torch_float())
  t <- torch_randn(batch_size, seq_len, dim, dtype = torch_float32()) # float32 alias

  result <- TabICL2:::apply_rotary_emb(freqs, t, interleaved = TRUE)

  expect_tensor(result)
  expect_tensor_shape(result, t$shape)
  expect_tensor_dtype(result, t$dtype)

  # Ensure rotation actually happened (output != input)
  expect_false(torch_allclose(result, t, atol = 1e-1))

  # Ensure no NaNs
  expect_equal_to_r(torch_sum(torch_isnan(result)), 0L)
})

test_that("apply_rotary_emb preserves shape and dtype (contiguous)", {
  dim <- 4L
  seq_len <- 3L
  batch_size <- 2L
  half_dim <- dim %/% 2L

  # For contiguous, freqs has shape (seq_len, half_dim)
  freqs <- torch_randn(seq_len, half_dim, dtype = torch_float())
  t <- torch_randn(batch_size, seq_len, dim, dtype = torch_float())

  result <- TabICL2:::apply_rotary_emb(freqs, t, interleaved = FALSE)

  expect_tensor(result)
  expect_tensor_shape(result, t$shape)
  expect_tensor_dtype(result, t$dtype)
  expect_equal_to_r(torch_sum(torch_isnan(result)), 0L)
})

test_that("apply_rotary_emb throws error on dimension mismatch", {
  freqs <- torch_randn(3L, 8L) # rot_dim = 8
  t <- torch_randn(2L, 3L, 4L) # last_dim = 4

  expect_error(
    TabICL2:::apply_rotary_emb(freqs, t, interleaved = TRUE),
    class = "value_error"
  )
})

test_that("RotaryEmbedding initializes correctly with default params", {
  rope <- RotaryEmbedding(dim = 64L)

  expect_s3_class(rope, "nn_module")
  expect_true("RotaryEmbedding" %in% class(rope))

  # Check parameters and buffers
  expect_tensor(rope$freqs)
  expect_true(rope$freqs$requires_grad == FALSE)
  expect_null(rope$cached_freqs)
  expect_null(rope$cached_scales)
  expect_equal(rope$interleaved, TRUE)
  expect_equal(rope$use_xpos, FALSE)
})

test_that("RotaryEmbedding initializes correctly with XPOS", {
  rope <- RotaryEmbedding(dim = 64L, use_xpos = TRUE)

  expect_true(is_torch_tensor(rope$scale))
  expect_equal(rope$scale_base, 512)
})

test_that("RotaryEmbedding initializes correctly with different freqs_for", {
  rope_pixel <- RotaryEmbedding(dim = 64L, freqs_for = "pixel", max_freq = 10)
  expect_tensor(rope_pixel$freqs)

  rope_const <- RotaryEmbedding(dim = 64L, freqs_for = "constant", num_freqs = 5L)
  expect_tensor(rope_const$freqs)
  expect_equal(rope_const$freqs$shape, c(5L))
})

test_that("RotaryEmbedding forward computes and caches frequencies", {
  rope <- RotaryEmbedding(dim = 32L, cache_if_possible = TRUE)
  seq_len <- 10L

  # First call: should compute and cache
  freqs1 <- rope$forward(torch_arange(seq_len), seq_len = seq_len)

  expect_tensor(freqs1)
  expect_tensor_shape(freqs1, c(seq_len, 32L))
  expect_tensor(rope$cached_freqs)

  # Second call with same seq_len: should use cache (no error, same shape)
  freqs2 <- rope$forward(torch_arange(seq_len), seq_len = seq_len)
  expect_tensor_shape(freqs2, c(seq_len, 32L))
})

test_that("RotaryEmbedding rotate_queries_or_keys works", {
  rope <- RotaryEmbedding(dim = 64L)
  t <- torch_randn(2L, 10L, 64L) # (batch, seq, dim)

  result <- rope$rotate_queries_or_keys(t)

  expect_tensor(result)
  expect_tensor_shape(result, t$shape)
  expect_tensor_dtype(result, t$dtype)
})

test_that("RotaryEmbedding rotate_queries_and_keys works with XPOS", {
  rope <- RotaryEmbedding(dim = 64L, use_xpos = TRUE)
  q <- torch_randn(2L, 10L, 64L)
  k <- torch_randn(2L, 10L, 64L)

  result <- rope$rotate_queries_and_keys(q, k)

  expect_type(result, "list")
  expect_length(result, 2L)

  rotated_q <- result[[1L]]
  rotated_k <- result[[2L]]

  expect_tensor_shape(rotated_q, q$shape)
  expect_tensor_shape(rotated_k, k$shape)
  expect_tensor_dtype(rotated_q, q$dtype)
  expect_tensor_dtype(rotated_k, k$dtype)
})

test_that("RotaryEmbedding rotate_queries_with_cached_keys handles different lengths", {
  rope <- RotaryEmbedding(dim = 64L, use_xpos = TRUE)
  q <- torch_randn(2L, 4L, 64L)   # q shorter
  k <- torch_randn(2L, 10L, 64L)  # k longer

  result <- rope$rotate_queries_with_cached_keys(q, k)

  expect_type(result, "list")
  expect_equal(length(result), 2L)
  expect_tensor_shape(result[[1L]], q$shape)
  expect_tensor_shape(result[[2L]], k$shape)
})

test_that("RotaryEmbedding rotate_queries_and_keys throws error without XPOS", {
  rope <- RotaryEmbedding(dim = 64L, use_xpos = FALSE)
  q <- torch_randn(2L, 8L, 64L)
  k <- torch_randn(2L, 10L, 64L)

  expect_error(
    rope$rotate_queries_and_keys(q, k),
    class = "runtime_error"
  )
})

test_that("RotaryEmbedding get_seq_pos applies interpolation", {
  rope <- RotaryEmbedding(dim = 32L, interpolate_factor = 2.0)
  pos <- rope$get_seq_pos(4L, device = "cpu", dtype = torch_float())

  expect_tensor(pos)
  expect_equal_to_r(pos, c(0.0, 0.5, 1.0, 1.5))
})

test_that("RotaryEmbedding custom_freqs overrides default", {
  custom <- torch_randn(16L)
  rope <- RotaryEmbedding(dim = 32L, custom_freqs = custom)

  expect_equal_to_r(rope$freqs, as.array(custom))
})
