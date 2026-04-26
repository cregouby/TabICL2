test_that("skippable_linear handles skip values correctly", {
  layer <- skippable_linear(in_features = 4L, out_features = 8L, skip_value = -100.0)

  # Input with some skipped rows
  x <- torch_randn(10L, 4L)
  x[1:3, ] <- -100.0  # Mark first 3 rows as skipped
  x[1:3, 1:3] <- -100.0  # Mark first 3 rows as non-skipped

  out <- layer(x)

  # Output shape preserved
  expect_tensor_shape(out, c(10L, 8L))

  # Skipped rows remain at skip_value
  expect_equal_to_r(out[1:3, ], array(-100.0, dim = c(3,8)))

  # Non-skipped rows are transformed (not exactly -100)
  expect_false(torch_min(out[4:10, ] == -100.0)$item() == 1)
})

test_that("one_hot_and_linear produces correct embeddings", {
  encoder <- one_hot_and_linear(num_classes = 5L, embed_dim = 16L)
  indices <- torch_randint(1L, 6L, c(2L, 10L))  # [batch, seq], values 1-5

  embeddings <- encoder(indices)

  expect_tensor_shape(embeddings, c(2L, 10L, 16L))
  expect_tensor_dtype(embeddings, indices$dtype)

  # Check no NaN/Inf
  expect_false(torch_max(torch_isnan(embeddings))$item())
})
