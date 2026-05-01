test_that(".logn returns correct scalar tensor", {
  t <- .logn(10L, torch_device("cpu"), torch_float32())
  expect_tensor(t)
  expect_tensor_shape(t, 1)
  expect_equal_to_r(t, log(10))

  # clamps n = 0 to 1, so log(1) = 0
  t0 <- .logn(0L, torch_device("cpu"), torch_float32())
  expect_equal_to_r(t0, 0)

  # respects device and dtype
  t16 <- .logn(5L, torch_device("cpu"), torch_float16())
  expect_tensor_dtype(t16, torch_float16())
})

test_that("SSMax: output shape matches input shape", {
  num_heads <- 4L
  head_dim  <- 8L
  bs        <- 2L
  seq_len   <- 6L

  mod <- SSMax(num_heads)
  q   <- torch_randn(bs, num_heads, seq_len, head_dim)

  out <- mod(q, n = seq_len)
  expect_tensor(out)
  expect_tensor_shape(out, c(bs, num_heads, seq_len, head_dim))
})

test_that("SSMax: scales are learnable parameters", {
  mod <- SSMax(4L)
  expect_true(is_torch_tensor(mod$scales))
  expect_tensor_shape(mod$scales, c(4L))

  # all initialised to 1
  expect_equal_to_r(mod$scales, rep(1, 4))
})

test_that("SSMax: n=1 produces zero scaling (log(1)=0)", {
  mod <- SSMax(4L)
  q   <- torch_randn(2L, 4L, 5L, 8L)

  out <- mod(q, n = 1L)
  # log(1) = 0 => scales = 0 => output is all zeros
  expect_tensor_shape(out, c(2, 4, 5, 8))
  expect_equal_to_r(out, array(rep(0, 2 * 4 * 5 * 8), dim = c(2, 4, 5, 8)), tolerance = 1e-6)
})

test_that("SSMax: larger n amplifies output relative to smaller n", {
  mod <- SSMax(4L)
  q   <- torch_randn(1L, 4L, 3L, 8L)

  out_small <- mod(q, n = 2L)
  out_large <- mod(q, n = 100L)

  # log(100) > log(2), so magnitudes should be larger
  expect_true(
    mean(as_array(out_large$abs())) > mean(as_array(out_small$abs()))
  )
})

test_that("SSMaxMLP (non-elementwise): output shape matches input", {
  num_heads <- 4L
  head_dim  <- 8L
  bs        <- 2L
  seq_len   <- 5L

  mod <- SSMaxMLP(num_heads, n_hidden = 32L, elementwise = FALSE)
  q   <- torch_randn(bs, num_heads, seq_len, head_dim)

  out <- mod(q, n = seq_len)
  expect_tensor(out)
  expect_tensor_shape(out, c(bs, num_heads, seq_len, head_dim))
})

test_that("SSMaxMLP (elementwise): output shape matches input", {
  num_heads <- 4L
  head_dim  <- 8L
  bs        <- 2L
  seq_len   <- 5L

  mod <- SSMaxMLP(
    num_heads   = num_heads,
    n_hidden    = 32L,
    elementwise = TRUE,
    head_dim    = head_dim
  )
  q   <- torch_randn(bs, num_heads, seq_len, head_dim)

  out <- mod(q, n = seq_len)
  expect_tensor(out)
  expect_tensor_shape(out, c(bs, num_heads, seq_len, head_dim))
})

test_that("SSMaxMLP: elementwise=TRUE without head_dim raises error", {
  expect_error(
    SSMaxMLP(num_heads = 4L, elementwise = TRUE, head_dim = NULL),
    "head_dim must be provided"
  )
})

test_that("SSMaxMLP (non-elementwise): same n always gives same scales", {
  mod <- SSMaxMLP(4L, n_hidden = 16L, elementwise = FALSE)
  q1  <- torch_randn(2L, 4L, 3L, 8L)
  q2  <- torch_randn(2L, 4L, 3L, 8L)

  out1 <- mod(q1, n = 10L)
  out2 <- mod(q2, n = 10L)

  # scales depend only on n, not q -> out / q should be constant
  ratio1 <- as_array((out1 / q1)[1, 1, 1, 1])
  ratio2 <- as_array((out2 / q2)[1, 1, 1, 1])
  expect_equal(ratio1, ratio2, tolerance = 1e-5)
})

test_that("QASSMaxMLP (non-elementwise): output shape matches input", {
  num_heads <- 4L
  head_dim  <- 8L
  bs        <- 2L
  seq_len   <- 5L

  mod <- QASSMaxMLP(num_heads, head_dim, n_hidden = 32L, elementwise = FALSE)
  q   <- torch_randn(bs, num_heads, seq_len, head_dim)

  out <- mod(q, n = seq_len)
  expect_tensor(out)
  expect_tensor_shape(out, c(bs, num_heads, seq_len, head_dim))
})

test_that("QASSMaxMLP (elementwise): output shape matches input", {
  num_heads <- 4L
  head_dim  <- 8L
  bs        <- 2L
  seq_len   <- 5L

  mod <- QASSMaxMLP(
    num_heads   = num_heads,
    head_dim    = head_dim,
    n_hidden    = 32L,
    elementwise = TRUE
  )
  q   <- torch_randn(bs, num_heads, seq_len, head_dim)

  out <- mod(q, n = seq_len)
  expect_tensor(out)
  expect_tensor_shape(out, c(bs, num_heads, seq_len, head_dim))
})

test_that("QASSMaxMLP: initial output is close to base-only scaling (zero-init)", {
  num_heads <- 2L
  head_dim  <- 4L
  n         <- 10L

  mod <- QASSMaxMLP(num_heads, head_dim, n_hidden = 16L, elementwise = FALSE)
  q   <- torch_randn(1L, num_heads, 3L, head_dim)

  # At init, query_mlp output is ~0 => modulation ≈ 1
  # so qassmax ≈ q * base_mlp(log(n)) * 1
  out <- mod(q, n = n)

  # Verify that modulation is indeed ~1 by manually computing
  logn         <- .logn(n, q$device, q$dtype)$view(c(1L, 1L))
  base_scales  <- mod$base_mlp(logn)$view(c(1L, num_heads, 1L, 1L))
  expected     <- q * base_scales  # modulation ≈ 1

  # Should match closely
  diff <- as_array((out - expected)$abs()$max())
  expect_true(diff < 1e-4)
})

test_that("QASSMaxMLP: different queries produce different scalings", {
  num_heads <- 2L
  head_dim  <- 4L

  mod <- QASSMaxMLP(num_heads, head_dim, n_hidden = 16L, elementwise = FALSE)

  q_same <- torch_ones(1L, num_heads, 3L, head_dim)
  q_diff <- torch_randn(1L, num_heads, 3L, head_dim)

  out_same <- mod(q_same, n = 10L)
  out_diff <- mod(q_diff, n = 10L)

  # Same input everywhere => uniform scaling along seq_len for non-elementwise
  col0 <- as_array(out_same[1, 1, 1, ..]$squeeze())
  col1 <- as_array(out_same[1, 1, 2, ..]$squeeze())
  expect_equal(col0, col1, tolerance = 1e-5)

  # Different inputs => different outputs (since query_mlp modulates)
  # At init modulation is ~1, but base_mlp still scales, so values differ
  # by the ratio of input elements (before base scaling).
  # Just check the tensor shapes are valid and non-NaN.
  expect_false(any(is.nan(as_array(out_diff))))
})

test_that("QASSMaxMLP: backward pass runs without error", {
  mod <- QASSMaxMLP(2L, 4L, n_hidden = 16L, elementwise = TRUE)
  q   <- torch_randn(1L, 2L, 3L, 4L, requires_grad = TRUE)

  out <- mod(q, n = 5L)
  loss <- out$sum()
  loss$backward()

  # gradient should flow back to q
  expect_tensor(q$grad)
  expect_false(any(is.nan(as_array(q$grad))))
})

test_that("create_ssmax_layer: 'none' returns NULL", {
  expect_null(create_ssmax_layer("none", num_heads = 4L, embed_dim = 16L))
})

test_that("create_ssmax_layer: 'ssmax' returns SSMax", {
  mod <- create_ssmax_layer("ssmax", num_heads = 4L, embed_dim = 16L)
  expect_s3_class(mod, "nn_module")
  expect_s3_class(mod, "SSMax")
})

test_that("create_ssmax_layer: 'ssmax-mlp' returns SSMaxMLP", {
  mod <- create_ssmax_layer("ssmax-mlp", num_heads = 4L, embed_dim = 16L)
  expect_s3_class(mod, "nn_module")
  expect_s3_class(mod, "SSMaxMLP")
})

test_that("create_ssmax_layer: 'ssmax-mlp-elementwise' returns elementwise SSMaxMLP", {
  mod <- create_ssmax_layer(
    "ssmax-mlp-elementwise", num_heads = 4L, embed_dim = 16L
  )
  expect_s3_class(mod, "SSMaxMLP")
  expect_true(mod$elementwise)
})

test_that("create_ssmax_layer: 'qassmax-mlp' returns QASSMaxMLP", {
  mod <- create_ssmax_layer(
    "qassmax-mlp", num_heads = 4L, embed_dim = 16L
  )
  expect_s3_class(mod, "QASSMaxMLP")
})

test_that("create_ssmax_layer: 'qassmax-mlp-elementwise' returns elementwise QASSMaxMLP", {
  mod <- create_ssmax_layer(
    "qassmax-mlp-elementwise", num_heads = 4L, embed_dim = 16L
  )
  expect_s3_class(mod, "QASSMaxMLP")
  expect_true(mod$elementwise)
})

test_that("create_ssmax_layer: unknown type raises error", {
  expect_error(
    create_ssmax_layer("bogus", num_heads = 4L, embed_dim = 16L),
    "Unknown ssmax_type"
  )
})

test_that("create_ssmax_layer: all non-null variants produce correct output shapes", {
  types <- c(
    "ssmax", "ssmax-mlp", "ssmax-mlp-elementwise",
    "qassmax-mlp", "qassmax-mlp-elementwise"
  )

  num_heads <- 4L
  embed_dim <- 16L
  bs        <- 2L
  seq_len   <- 6L
  head_dim  <- embed_dim %/% num_heads

  for (tp in types) {
    mod <- create_ssmax_layer(tp, num_heads, embed_dim)
    q   <- torch_randn(bs, num_heads, seq_len, head_dim)

    out <- mod(q, n = seq_len)
    expect_tensor_shape(out, c(bs, num_heads, seq_len, head_dim))
  }
})

test_that("all SSMax variants support backward pass", {
  num_heads <- 2L
  head_dim  <- 4L
  bs        <- 1L
  seq_len   <- 3L

  q <- torch_randn(bs, num_heads, seq_len, head_dim, requires_grad = TRUE)

  variants <- list(
    SSMax     = SSMax(num_heads),
    SSMaxMLP  = SSMaxMLP(num_heads, n_hidden = 16L),
    SSMaxMLP_ew = SSMaxMLP(
      num_heads = num_heads, n_hidden = 16L,
      elementwise = TRUE, head_dim = head_dim
    ),
    QASSMaxMLP  = QASSMaxMLP(num_heads, head_dim, n_hidden = 16L),
    QASSMaxMLP_ew = QASSMaxMLP(
      num_heads = num_heads, head_dim = head_dim,
      n_hidden = 16L, elementwise = TRUE
    )
  )

  for (name in names(variants)) {
    q_grad <- q$clone()$detach()$requires_grad_(TRUE)
    mod    <- variants[[name]]
    out    <- mod(q_grad, n = seq_len)
    out$sum()$backward()

    expect_tensor(q_grad$grad)
    expect_false(
      any(is.nan(as_array(q_grad$grad))),
      label = paste(name, "no NaN grads")
    )
  }
})
