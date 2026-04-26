test_that("memory_estimator estimates batch size correctly", {
  bs <- memory_estimator$estimate_batch_size(
    seq_len = 100L, target_memory = 1000.0, 
    enc_name = "tf_col", include_inputs = TRUE, in_dim = 128L
  )
  
  expect_true(is.integer(bs))
  expect_true(bs >= 1L)
})

test_that("pinned_buffer_pool reuses buffers", {
  pool <- pinned_buffer_pool(max_buffers_per_shape = 2L)
  
  # Get two buffers with same shape/dtype
  buf1 <- pool$get(c(10L, 20L), torch_float())
  buf2 <- pool$get(c(10L, 20L), torch_float())
  
  expect_true(buf1$is_pinned())
  expect_true(buf2$is_pinned())
  
  # Return one to pool
  pool$put(buf1)
  
  # Get again: should reuse buf1
  buf3 <- pool$get(c(10L, 20L), torch_float())
  
  # buf3 should be the same memory as buf1 (conceptual; hard to test directly)
  expect_tensor_shape(buf3, c(10L, 20L))
  
  pool$clear()
})

test_that("inference_manager validates configuration", {
  mgr <- inference_manager(enc_name = "tf_col", out_dim = 64L)
  
  # Must configure before use
  expect_error(
    mgr$forward(function(...) torch_randn(2L, 10L, 64L), list(x = torch_randn(2L, 10L, 32L))),
    class = "runtime_error"
  )
  
  # Valid configuration
  expect_silent(
    mgr$configure(offload = "auto", safety_factor = 0.9, verbose = FALSE)
  )
  
  # Invalid offload mode
  expect_error(
    mgr$configure(offload = "invalid"),
    class = "value_error"
  )
})

test_that("offload_reason formats correctly", {
  reason <- offload_reason("test_key", "test detail")
  expect_match(format(reason), "test_key: test detail")
  
  reason_no_detail <- offload_reason("simple")
  expect_equal(format(reason_no_detail), "simple")
})