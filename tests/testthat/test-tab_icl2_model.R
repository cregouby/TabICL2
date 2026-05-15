test_that("NanoTabICLv2 initializes with default parameters", {
  model <- NanoTabICLv2(max_classes = 10L, out_dim = 10L)
  expect_true(inherits(model, "nn_module"))
  expect_equal(model$feature_group_size, 3L)
  expect_tensor_shape(model$row_cls_tokens, c(1, 1, 4, 128))
})

test_that("NanoTabICLv2 initializes for regression mode", {
  model <- NanoTabICLv2(max_classes = 0L, out_dim = 5L)
  expect_true(inherits(model$y_embed_in, "nn_linear"))
  expect_true(inherits(model$y_embed_icl, "nn_linear"))
})

test_that("NanoTabICLv2 initializes for classification mode", {
  model <- NanoTabICLv2(max_classes = 10L, out_dim = 10L)
  expect_true(inherits(model$y_embed_in, "nn_module"))
  expect_true(inherits(model$y_embed_icl, "nn_module"))
})

test_that("NanoTabICLv2 forward pass basic classification", {
  model <- NanoTabICLv2(max_classes = 10L, out_dim = 10L)
  model$eval()
  x <- torch_randn(2L, 24L, 3L)
  y <- torch_randint(1, 10L, size = c(2L, 16L))
  output <- model(x, y)
  expect_tensor(output)
  expect_tensor_shape(output, c(2L, 8L, 10L))
})

test_that("NanoTabICLv2 forward pass basic regression", {
  model <- NanoTabICLv2(max_classes = 0L, out_dim = 3L)
  model$eval()
  x <- torch_randn(1L, 20L, 5L)
  y <- torch_randn(1L, 12L)
  output <- model(x, y)
  expect_tensor_shape(output, c(1L, 8L, 3L))
})

test_that("NanoTabICLv2 handles single batch", {
  model <- NanoTabICLv2(max_classes = 5L, out_dim = 5L)
  model$eval()
  x <- torch_randn(1L, 10L, 4L)
  y <- torch_randint(1, 5L, size = c(1L, 6L))
  output <- model(x, y)
  expect_tensor_shape(output, c(1L, 4L, 5L))
})

test_that("NanoTabICLv2 handles single feature column", {
  model <- NanoTabICLv2(max_classes = 3L, out_dim = 3L, feature_group_size = 1L)
  model$eval()
  x <- torch_randn(2L, 8L, 1L)
  y <- torch_randint(1, 3L, size = c(2L, 5L))
  output <- model(x, y)
  expect_tensor_shape(output, c(2L, 3L, 3L))
})

test_that("NanoTabICLv2 handles all training no test", {
  model <- NanoTabICLv2(max_classes = 4L, out_dim = 4L)
  model$eval()
  x <- torch_randn(1L, 10L, 3L)
  y <- torch_randint(1, 4L, size = c(1L, 10L))
  output <- model(x, y)
  expect_tensor_shape(output, c(1L, 0L, 4L))
})

test_that("NanoTabICLv2 output dtype matches input", {
  model <- NanoTabICLv2(max_classes = 5L, out_dim = 5L)
  model$eval()
  x_f32 <- torch_randn(2L, 12L, 3L, dtype = torch_float32())
  y <- torch_randint(1, 5L, size = c(2L, 8L))
  output <- model(x_f32, y)
  expect_tensor_dtype(output, torch_float32())
})

test_that("NanoTabICLv2 gradient flow training mode", {
  model <- NanoTabICLv2(max_classes = 3L, out_dim = 3L)
  model$train()
  x <- torch_randn(2L, 10L, 4L, requires_grad = TRUE)
  y <- torch_randint(1, 3L, size = c(2L, 6L))
  output <- model(x, y)
  loss <- output$sum()
  loss$backward()
  expect_true(!is.null(x$grad))
  expect_tensor_shape(x$grad, c(2L, 10L, 4L))
})

test_that("ClassEmbedding initializes with uniform bounds", {
  emb <- ClassEmbedding(num_embeddings = 10L, embedding_dim = 16L)
  bound <- 1 / sqrt(10)
  weights <- as_array(emb$embedding$weight)
  expect_true(all(weights >= -bound - 1e-6))
  expect_true(all(weights <= bound + 1e-6))
})

test_that("ClassEmbedding forward handles integer labels", {
  emb <- ClassEmbedding(num_embeddings = 5L, embedding_dim = 8L)
  y <- torch_tensor(c(1L, 3L, 5L, 2L), dtype = torch_long())
  output <- emb(y)
  expect_tensor_shape(output, c(4L, 8L))
})

test_that("ClassEmbedding forward handles multi-label (2D labels)", {
  emb <- ClassEmbedding(num_embeddings = 5L, embedding_dim = 8L)
  y <- torch_randint(1, 5L, size = c(2L, 3L, 1L), dtype = torch_long())
  output <- emb(y)
  expect_tensor_shape(output, c(2L, 3L, 8L))
})

test_that("Rope initializes with correct buffers", {
  rope <- Rope(head_dim = 16L, theta = 10000.0)
  expect_equal(rope$half, 8L)
  expect_tensor_shape(rope$inv_freq, c(8L))
  expect_equal(rope$sin$numel(), 0L)
  expect_equal(rope$cos$numel(), 0L)
})

test_that("Rope forward computes rotary encoding", {
  rope <- Rope(head_dim = 8L, theta = 10000.0)
  x <- torch_randn(1L, 2L, 5L, 8L)
  output <- rope(x)
  expect_tensor_shape(output, c(1L, 2L, 5L, 8L))
  expect_equal_to_r(output$norm(p = 2L, dim = -1), as_array(x$norm(p = 2L, dim = -1)), tolerance = 1e-5)
})

test_that("Rope forward caches positional encodings", {
  rope <- Rope(head_dim = 8L, theta = 10000.0)
  x1 <- torch_randn(1L, 1L, 3L, 8L)
  ttrash <- rope(x1)
  expect_true(rope$sin$numel() > 0)
  expect_true(rope$cos$numel() > 0)
  x2 <- torch_randn(1L, 1L, 3L, 8L)
  output2 <- rope(x2)
  expect_tensor(output2)
})

test_that("QASSMax initializes with zeroed query modulation", {
  ssmax <- QASSMax(num_heads = 4L, head_dim = 16L, n_hidden = 32L)
  query_weight <- as_array(ssmax$query_mlp[[3]]$weight)
  query_bias <- as_array(ssmax$query_mlp[[3]]$bias)
  expect_true(all(abs(query_weight) < 1e-6))
  expect_true(all(abs(query_bias) < 1e-6))
})

test_that("QASSMax forward scales with sequence length", {
  torch::with_torch_manual_seed({
    ssmax <- QASSMax(num_heads = 2L, head_dim = 8L)
    q <- torch_randn(1L, 2L, 10L, 8L)
    },
    seed = 92)
  out_short <- ssmax(q, n = 10L)
  out_long <- ssmax(q, n = 100L)
  expect_tensor_shape(out_short, c(1L, 2L, 10L, 8L))
  expect_tensor_shape(out_long, c(1L, 2L, 10L, 8L))
  ratio <- (out_long / (out_short + 1e-8))$mean()$item()
  expect_true(ratio > 0.5 && ratio < 2.5)
})

test_that("TransformerBlock initializes components", {
  block <- TransformerBlock(embed_dim = 32L, num_heads = 4L, use_rope = TRUE, ssmax = TRUE)
  expect_true(inherits(block$rope, "nn_module"))
  expect_true(inherits(block$ssmax_layer, "nn_module"))
  expect_true(inherits(block$mlp, "nn_sequential"))
  expect_true(inherits(block$ln_attn, "nn_layer_norm"))
  expect_true(inherits(block$ln_mlp, "nn_layer_norm"))
})

test_that("TransformerBlock forward basic", {
  block <- TransformerBlock(embed_dim = 16L, num_heads = 2L)
  q <- torch_randn(2L, 5L, 16L)
  output <- block(q)
  expect_tensor_shape(output, c(2L, 5L, 16L))
})

test_that("TransformerBlock forward with kv and max indices", {
  block <- TransformerBlock(embed_dim = 16L, num_heads = 2L)
  q <- torch_randn(1L, 8L, 16L)
  kv <- torch_randn(1L, 12L, 16L)
  output <- block(q, kv, q_max_idx = 6L, kv_max_idx = 10L)
  expect_tensor_shape(output, c(1L, 6L, 16L))
})

test_that("TransformerBlock row_attn preserves batch and row dims", {
  block <- TransformerBlock(embed_dim = 16L, num_heads = 2L)
  q <- torch_randn(2L, 3L, 4L, 16L)
  output <- block$row_attn(q)
  expect_tensor_shape(output, c(2L, 3L, 4L, 16L))
})

test_that("TransformerBlock col_attn transposes correctly", {
  block <- TransformerBlock(embed_dim = 16L, num_heads = 2L)
  q <- torch_randn(2L, 3L, 4L, 16L)
  output <- block$col_attn(q)
  expect_tensor_shape(output, c(2L, 3L, 4L, 16L))
})

test_that("InducedTransformerBlock initializes", {
  block <- InducedTransformerBlock(embed_dim = 24L, num_heads = 3L, n_inducing = 16L, ssmax = TRUE)
  expect_tensor_shape(block$inducing_vectors, c(1L, 16L, 24L))
  expect_true(inherits(block$tfm1, "nn_module"))
  expect_true(inherits(block$tfm2, "nn_module"))
})

test_that("InducedTransformerBlock forward uses inducing vectors", {
  block <- InducedTransformerBlock(embed_dim = 16L, num_heads = 2L, n_inducing = 8L)
  q <- torch_randn(2L, 10L, 16L)
  output <- block(q)
  expect_tensor_shape(output, c(2L, 10L, 16L))
})

test_that("InducedTransformerBlock col_attn delegates to tfm2", {
  block <- InducedTransformerBlock(embed_dim = 16L, num_heads = 2L, n_inducing = 8L)
  q <- torch_randn(2L, 3L, 4L, 16L)
  output <- block$col_attn(q)
  expect_tensor_shape(output, c(2L, 3L, 4L, 16L))
})

test_that("NanoTabICLv2 feature grouping creates correct input shape", {
  model <- NanoTabICLv2(max_classes = 5L, out_dim = 5L, feature_group_size = 2L)
  n_cols <- 6L
  x <- torch_randn(1L, 8L, n_cols)
  idxs <- torch_arange(n_cols, dtype = torch_long())
  grouped <- purrr::map(seq_len(2), function(i) {
    shift <- (2^i - 1) %% n_cols
    indices <- ((idxs + shift - 1L) %% n_cols + 1L)$to(dtype = torch_long())
    x[.., indices]
  })
  x_grouped <- torch_stack(grouped, dim = -1)
  expect_tensor_shape(x_grouped, c(1L, 8L, n_cols, 2L))
})

test_that("NanoTabICLv2 standardization uses training subset only", {
  model <- NanoTabICLv2(max_classes = 3L, out_dim = 3L)
  x <- torch_randn(1L, 10L, 4L)
  x[, 1:6, ..] <- x[, 1:6, ..] * 10 + 5
  train_std <- x[, 1:6, ..]$std(dim = 2, unbiased = FALSE, keepdim = TRUE)
  x_norm <- x / (train_std + 1e-8)
  train_std_after <- x_norm[, 1:6, ..]$std(dim = 2, unbiased = FALSE, keepdim = TRUE)
  expect_true(all(abs(as_array(train_std_after) - 1) < 0.01))
})

test_that("NanoTabICLv2 CLS token expansion matches batch and rows", {
  model <- NanoTabICLv2(max_classes = 4L, out_dim = 4L, n_cls_cols = 3L)
  n_batch <- 2L
  n_rows <- 5L
  expanded <- model$row_cls_tokens$expand(c(n_batch, n_rows, -1, -1))
  expect_tensor_shape(expanded, c(n_batch, n_rows, 3L, 128L))
})

test_that("NanoTabICLv2 output MLP produces correct final dimension", {
  embed_dim = 16L
  n_cls_cols = 2L
  model <- NanoTabICLv2(max_classes = 6L, out_dim = 6L, embed_dim = embed_dim, n_cls_cols = n_cls_cols)
  icl_dim <- embed_dim * n_cls_cols
  expect_tensor_shape(model$out_mlp[[1]]$weight, c(64L, icl_dim))
  expect_tensor_shape(model$out_mlp[[3]]$weight, c(6L, 64L))
})

test_that("NanoTabICLv2 model eval mode disables gradients", {
  model <- NanoTabICLv2(max_classes = 3L, out_dim = 3L)
  model$eval()
  x <- torch_randn(1L, 8L, 4L)
  y <- torch_randint(1, 3L, size = c(1L, 5L))
  with_no_grad({
    output <- model(x, y)
  })
  expect_false(output$requires_grad)
})

test_that("NanoTabICLv2 handles different embed_dim values", {
  purrr::walk(c(64L, 128L, 256L), function(dim) {
    model <- NanoTabICLv2(max_classes = 5L, out_dim = 5L, embed_dim = dim)
    expect_tensor_shape(model$x_embed$weight, c(dim, 3L))
    expect_tensor_shape(model$row_cls_tokens, c(1L, 1L, 4L, dim))
  })
})

test_that("NanoTabICLv2 handles different block counts", {
  model <- NanoTabICLv2(
    max_classes = 4L, out_dim = 4L,
    col_num_blocks = 1L, row_num_blocks = 2L, icl_num_blocks = 4L
  )
  expect_equal(length(model$col_blocks), 1L)
  expect_equal(length(model$row_blocks), 2L)
  expect_equal(length(model$icl_blocks), 4L)
})

test_that("NanoTabICLv2 handles different nhead values", {
  model <- NanoTabICLv2(
    max_classes = 3L, out_dim = 3L,
    col_nhead = 4L, row_nhead = 2L, icl_nhead = 8L, embed_dim = 32L
  )
  expect_equal(model$col_blocks[[1]]$tfm1$num_heads, 4L)
  expect_equal(model$row_blocks[[1]]$num_heads, 2L)
  expect_equal(model$icl_blocks[[1]]$num_heads, 8L)
})

test_that("NanoTabICLv2 output is finite for random input", {
  model <- NanoTabICLv2(max_classes = 10L, out_dim = 10L)
  model$eval()
  x <- torch_randn(3L, 30L, 5L)
  y <- torch_randint(1, 10L, size = c(3L, 20L))
  output <- model(x, y)
  expect_true(all(is.finite(as_array(output))))
})

test_that("NanoTabICLv2 output varies with input", {
  model <- NanoTabICLv2(max_classes = 5L, out_dim = 5L)
  model$eval()
  x1 <- torch_randn(1L, 12L, 4L)
  x2 <- x1 + 0.1
  y <- torch_randint(1, 5L, size = c(1L, 8L))
  out1 <- model(x1, y)
  out2 <- model(x2, y)
  expect_false(identical(as_array(out1), as_array(out2)))
})

test_that("NanoTabICLv2 reproducibility with fixed seed", {
  torch::with_torch_manual_seed({
    model1 <- NanoTabICLv2(max_classes = 3L, out_dim = 3L)
    x1 <- torch_randn(1L, 10L, 3L)
    y1 <- torch_randint(1, 3L, size = c(1L, 6L))
    out1 <- model1(x1, y1)
  }, seed = 1792)
  torch::with_torch_manual_seed({
    model2 <- NanoTabICLv2(max_classes = 3L, out_dim = 3L)
    x2 <- torch_randn(1L, 10L, 3L)
    y2 <- torch_randint(1, 3L, size = c(1L, 6L))
    out2 <- model2(x2, y2)
  }, seed = 1792)
  expect_equal_to_r(out1, as_array(out2), tolerance = 1e-6)
})

test_that("NanoTabICLv2 state dict contains expected keys", {
  model <- NanoTabICLv2(max_classes = 4L, out_dim = 4L)
  state <- model$state_dict()
  expected_keys <- c(
    "x_embed.weight", "x_embed.bias",
    "y_embed_in.embedding.weight",
    "y_embed_icl.embedding.weight",
    "row_cls_tokens",
    "row_ln.weight", "row_ln.bias",
    "out_ln.weight", "out_ln.bias"
  )
  purrr::walk(expected_keys, function(key) {
    expect_true(key %in% names(state))
  })
})

test_that("NanoTabICLv2 load_state_dict restores weights", {
  model1 <- NanoTabICLv2(max_classes = 3L, out_dim = 3L)
  model1$eval()
  state <- model1$state_dict()
  model2 <- NanoTabICLv2(max_classes = 3L, out_dim = 3L)
  model2$load_state_dict(state)
  x <- torch_randn(1L, 8L, 4L)
  y <- torch_randint(1, 3L, size = c(1L, 5L))
  out1 <- model1(x, y)
  out2 <- model2(x, y)
  expect_equal_to_r(out1, as_array(out2), tolerance = 1e-6)
})
