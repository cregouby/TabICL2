test_that("col_embedding initializes and forwards correctly", {
  emb <- col_embedding(
    embed_dim = 16L, num_blocks = 1L, nhead = 2L,
    dim_feedforward = 32L, num_inds = 8L, affine = TRUE
  )

  expect_s3_class(emb, "nn_module")
  expect_true("ColEmbedding" %in% class(emb))

  # Vérifier que les sous-modules sont bien des nn_module
  expect_s3_class(emb$in_linear, "nn_module")
  expect_s3_class(emb$tf_col, "nn_module")

  # Forward pass
  B <- 2L; T <- 20L; H <- 10L; train_size <- 15L
  X <- torch_randn(B, T, H)
  y_train <- torch_randint(1L, 6L, c(B, train_size))

  emb$train()
  out <- emb(X, y_train)  # Appel direct, forward implicite

  expect_tensor_shape(out, c(B, T, H + 4L, 16L))
  expect_false(any(torch_isnan(out)$item()))
})

test_that("ColEmbedding forward preserves shapes", {
  emb <- ColEmbedding$new(
    embed_dim = 16L, num_blocks = 1L, nhead = 2L,
    dim_feedforward = 32L, num_inds = 8L
  )

  B <- 2L; T <- 20L; H <- 10L; train_size <- 15L
  X <- torch_randn(B, T, H)
  y_train <- torch_randint(1L, 6L, c(B, train_size))

  # Training mode
  emb$train()
  out <- emb$forward(X, y_train)

  # Output: (B, T, H + reserve_cls_tokens, embed_dim)
  expect_tensor_shape(out, c(B, T, H + 4L, 16L))

  # Inference mode
  emb$eval()
  out_inf <- emb$forward(X, y_train)
  expect_tensor_shape(out_inf, c(B, T, H + 4L, 16L))
})
