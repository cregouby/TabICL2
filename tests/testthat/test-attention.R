test_that("sdpa_with_flattened_batch preserves shapes", {
  # Simple test: single batch, single head
  q <- torch_randn(2L, 4L, 8L)   # [batch=2, tgt_len=4, head_dim=8]
  k <- torch_randn(2L, 6L, 8L)   # [batch=2, src_len=6, head_dim=8]
  v <- torch_randn(2L, 6L, 8L)

  # Reshape to add head dimension: (batch, seq, head_dim) -> (batch, 1, seq, head_dim)
  q <- q$unsqueeze(2L)  # [2, 1, 4, 8]
  k <- k$unsqueeze(2L)  # [2, 1, 6, 8]
  v <- v$unsqueeze(2L)

  out <- sdpa_with_flattened_batch(q, k, v)

  expect_tensor_shape(out, c(2L, 1L, 4L, 8L))
  expect_tensor_dtype(out, q$dtype)
})

test_that("multi_head_attention_forward handles caching", {
  embed_dim <- 32L
  num_heads <- 4L
  head_dim <- embed_dim %/% num_heads

  # Create dummy projections
  in_proj_weight <- torch_randn(3L * embed_dim, embed_dim)
  in_proj_bias <- torch_randn(3L * embed_dim)
  out_proj_weight <- torch_randn(embed_dim, embed_dim)
  out_proj_bias <- torch_randn(embed_dim)

  query <- torch_randn(2L, 10L, embed_dim)
  key <- torch_randn(2L, 15L, embed_dim)
  value <- torch_randn(2L, 15L, embed_dim)

  # Standard forward: should return (attn_output, k, v) when need_kv=TRUE
  result <- multi_head_attention_forward(
    query, num_heads, in_proj_weight, in_proj_bias,
    dropout_p = 0.0, out_proj_weight, out_proj_bias,
    key = key, value = value, need_kv = TRUE
  )

  expect_true(is.list(result))
  expect_equal(length(result), 3L)

  attn_out <- result[[1L]]
  k_proj <- result[[2L]]
  v_proj <- result[[3L]]

  expect_tensor_shape(attn_out, c(2L, 10L, embed_dim))
  expect_tensor_shape(k_proj, c(2L, num_heads, 15L, head_dim))
  expect_tensor_shape(v_proj, c(2L, num_heads, 15L, head_dim))
})

test_that("multi_head_attention_forward validates mask shapes", {
  embed_dim <- 16L
  num_heads <- 2L

  in_proj_weight <- torch_randn(3L * embed_dim, embed_dim)
  in_proj_bias <- torch_randn(3L * embed_dim)
  out_proj_weight <- torch_randn(embed_dim, embed_dim)
  out_proj_bias <- torch_randn(embed_dim)

  query <- torch_randn(1L, 5L, embed_dim)
  key <- torch_randn(1L, 7L, embed_dim)
  value <- torch_randn(1L, 7L, embed_dim)

  # Invalid 2D mask shape
  bad_mask <- torch_randn(3L, 3L)  # Wrong shape

  expect_error(
    multi_head_attention_forward(
      query, num_heads, in_proj_weight, in_proj_bias,
      dropout_p = 0, out_proj_weight, out_proj_bias,
      key = key, value = value, attn_mask = bad_mask
    ),
    class = "value_error"
  )
})
