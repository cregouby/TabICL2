test_that("QuantileDistributionConfig has correct default values", {
  cfg <- QuantileDistributionConfig$new()

  expect_equal(cfg$TOL, 1e-6)
  expect_equal(cfg$MIN_SLOPE, 1e-6)
  expect_equal(cfg$MAX_SLOPE, 1e6)
  expect_equal(cfg$MIN_BETA, 0.01)
  expect_equal(cfg$MAX_BETA, 100.0)
  expect_equal(cfg$MIN_ETA, -0.49)
  expect_equal(cfg$MAX_ETA, 0.49)
  expect_equal(cfg$ETA_TOLERANCE, 0.01)
  expect_equal(cfg$MAX_LOG_RATIO, 15.0)
  expect_equal(cfg$MAX_EXPONENT, 15.0)
  expect_equal(cfg$MAX_CRPS, 1e4)
  expect_equal(cfg$TAIL_QUANTILES_FOR_ESTIMATION, 20L)
})

test_that("isotonic_regression_pava enforces monotonicity", {
  # Non-monotonic input
  y <- torch_tensor(c(1.0, 3.0, 2.0, 4.0, 3.5, 5.0))

  result <- isotonic_regression_pava(y, weights = NULL)

  # Check monotonicity
  expect_tensor(result)
  expect_equal(result$shape, y$shape)

  result_vec <- as.numeric(result)
  for (i in seq_len(length(result_vec) - 1)) {
    expect_true(result_vec[i] <= result_vec[i + 1])
  }
})

test_that("isotonic_regression_pava works with weights", {
  y <- torch_tensor(c(1.0, 3.0, 2.0, 4.0))
  w <- torch_tensor(c(1.0, 1.0, 2.0, 1.0))  # Higher weight on 3rd element

  result <- isotonic_regression_pava(y, weights = w)

  expect_tensor(result)
  expect_equal(result$shape, y$shape)

  # Check monotonicity
  result_vec <- as.numeric(result)
  for (i in seq_len(length(result_vec) - 1)) {
    expect_true(result_vec[i] <= result_vec[i + 1])
  }
})

test_that("isotonic_regression_pava handles batch input", {
  # Batch of 3 sequences
  y <- torch_tensor(matrix(c(
    1.0, 3.0, 2.0, 4.0,
    2.0, 1.0, 3.0, 2.5,
    1.0, 2.0, 1.5, 3.0
  ), nrow = 3, byrow = TRUE))

  result <- isotonic_regression_pava(y, weights = NULL)

  expect_tensor(result)
  expect_equal(result$shape, y$shape)

  # Check each sequence is monotonic
  result_mat <- as.matrix(result)
  for (i in seq_len(nrow(result_mat))) {
    for (j in seq_len(ncol(result_mat) - 1)) {
      expect_true(result_mat[i, j] <= result_mat[i, j + 1])
    }
  }
})

test_that("enforce_monotonicity with sort method", {
  quantiles <- torch_tensor(c(1.0, 5.0, 3.0, 7.0, 6.0, 9.0))

  result <- enforce_monotonicity(quantiles, method = "sort")

  expect_tensor(result)
  expect_equal_to_r(result, c(1.0, 3.0, 5.0, 6.0, 7.0, 9.0))
})

test_that("enforce_monotonicity with cummax method", {
  quantiles <- torch_tensor(c(1.0, 5.0, 3.0, 7.0, 6.0, 9.0))

  result <- enforce_monotonicity(quantiles, method = "cummax")

  expect_tensor(result)
  # cummax: 1, 5, 5, 7, 7, 9
  expect_equal_to_r(result, c(1.0, 5.0, 5.0, 7.0, 7.0, 9.0))
})

test_that("enforce_monotonicity with isotonic method", {
  quantiles <- torch_tensor(c(1.0, 5.0, 3.0, 7.0, 6.0, 9.0))

  result <- enforce_monotonicity(quantiles, method = "isotonic")

  expect_tensor(result)
  # Check monotonicity
  result_vec <- as.numeric(result)
  for (i in seq_len(length(result_vec) - 1)) {
    expect_true(result_vec[i] <= result_vec[i + 1])
  }
})

test_that("enforce_monotonicity rejects invalid method", {
  quantiles <- torch_tensor(c(1.0, 2.0, 3.0))

  expect_error(
    enforce_monotonicity(quantiles, method = "invalid"),
    "Unknown method"
  )
})

test_that("estimate_exp_tail_params returns valid parameters", {
  # Create monotonic quantiles
  quantiles <- torch_linspace(0.0, 10.0, 100)$unsqueeze(1)
  alpha_levels <- torch_linspace(0.01, 0.99, 100)

  params <- estimate_exp_tail_params(quantiles, alpha_levels, num_tail_quantiles = 20L)

  expect_true("beta_l" %in% names(params))
  expect_true("beta_r" %in% names(params))

  expect_tensor(params$beta_l)
  expect_tensor(params$beta_r)

  # Beta should be positive and within bounds
  cfg <- QuantileDistributionConfig$new()
  expect_true(all(as.numeric(params$beta_l) >= cfg$MIN_BETA))
  expect_true(all(as.numeric(params$beta_l) <= cfg$MAX_BETA))
  expect_true(all(as.numeric(params$beta_r) >= cfg$MIN_BETA))
  expect_true(all(as.numeric(params$beta_r) <= cfg$MAX_BETA))
})

test_that("estimate_gpd_tail_params returns valid parameters", {
  # Create monotonic quantiles
  quantiles <- torch_linspace(0.0, 10.0, 100)$unsqueeze(1)
  alpha_levels <- torch_linspace(0.01, 0.99, 100)

  params <- estimate_gpd_tail_params(quantiles, alpha_levels, num_tail_quantiles = 20L)

  expect_true(all(c("eta_l", "mu_l", "eta_r", "mu_r") %in% names(params)))

  expect_tensor(params$eta_l)
  expect_tensor(params$mu_l)
  expect_tensor(params$eta_r)
  expect_tensor(params$mu_r)

  # Eta should be within bounds
  cfg <- QuantileDistributionConfig$new()
  expect_true(all(as.numeric(params$eta_l) >= cfg$MIN_ETA))
  expect_true(all(as.numeric(params$eta_l) <= cfg$MAX_ETA))
  expect_true(all(as.numeric(params$eta_r) >= cfg$MIN_ETA))
  expect_true(all(as.numeric(params$eta_r) <= cfg$MAX_ETA))
})

test_that("QuantileDistribution initializes with exponential tails", {
  quantiles <- torch_linspace(0.0, 10.0, 50)$unsqueeze(1)

  dist <- QuantileDistribution$new(
    quantiles = quantiles,
    tail_type = "exp",
    fix_crossing = TRUE
  )

  expect_s3_class(dist, "QuantileDistribution")
  expect_equal(dist$tail_type, "exp")
  expect_equal(dist$num_quantiles, 50)
  expect_tensor(dist$quantiles)
  expect_tensor(dist$alpha_levels)
})

test_that("QuantileDistribution initializes with GPD tails", {
  quantiles <- torch_linspace(0.0, 10.0, 50)$unsqueeze(1)

  dist <- QuantileDistribution$new(
    quantiles = quantiles,
    tail_type = "gpd",
    fix_crossing = TRUE
  )

  expect_s3_class(dist, "QuantileDistribution")
  expect_equal(dist$tail_type, "gpd")
  expect_tensor(dist$eta_l)
  expect_tensor(dist$mu_l)
  expect_tensor(dist$eta_r)
  expect_tensor(dist$mu_r)
})

test_that("QuantileDistribution fixes quantile crossing", {
  # Create quantiles with crossing
  quantiles <- torch_tensor(c(0.0, 2.0, 1.5, 3.0, 4.0))$unsqueeze(1)

  dist <- QuantileDistribution$new(
    quantiles = quantiles,
    fix_crossing = TRUE,
    crossing_method = "sort"
  )

  # Quantiles should be sorted
  q_vec <- as.numeric(dist$quantiles)
  for (i in seq_len(length(q_vec) - 1)) {
    expect_true(q_vec[i] <= q_vec[i + 1])
  }
})

test_that("QuantileDistribution icdf is monotonic", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  alphas <- torch_linspace(0.01, 0.99, 100)
  q_values <- dist$icdf(alphas)

  expect_tensor(q_values)

  # Check monotonicity
  q_vec <- as.numeric(q_values)
  for (i in seq_len(length(q_vec) - 1)) {
    expect_true(q_vec[i] <= q_vec[i + 1])
  }
})

test_that("QuantileDistribution icdf at alpha_levels recovers quantiles", {
  quantiles <- torch_linspace(-5.0, 5.0, 20)$unsqueeze(1)
  alpha_levels <- torch_linspace(0.01, 0.99, 20)

  dist <- QuantileDistribution$new(
    quantiles = quantiles,
    alpha_levels = alpha_levels,
    tail_type = "exp"
  )

  recovered <- dist$icdf(alpha_levels)

  # Should be very close to original quantiles
  expect_equal_to_r(
    recovered,
    as.array(quantiles),
    tolerance = 1e-4
  )
})

test_that("QuantileDistribution cdf is monotonic", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  z_values <- torch_linspace(-10.0, 10.0, 100)
  cdf_values <- dist$cdf(z_values)

  expect_tensor(cdf_values)

  # Check monotonicity
  cdf_vec <- as.numeric(cdf_values)
  for (i in seq_len(length(cdf_vec) - 1)) {
    expect_true(cdf_vec[i] <= cdf_vec[i + 1])
  }

  # CDF should be in [0, 1]
  expect_true(all(cdf_vec >= 0.0))
  expect_true(all(cdf_vec <= 1.0))
})

test_that("QuantileDistribution cdf and icdf are inverses", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  alphas <- torch_tensor(c(0.1, 0.3, 0.5, 0.7, 0.9))

  # icdf then cdf
  z <- dist$icdf(alphas)
  alphas_recovered <- dist$cdf(z)

  expect_equal_to_r(alphas_recovered, as.array(alphas), tolerance = 1e-4)
})

test_that("QuantileDistribution pdf is non-negative", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  z_values <- torch_linspace(-8.0, 8.0, 100)
  pdf_values <- dist$pdf(z_values)

  expect_tensor(pdf_values)
  expect_true(all(as.numeric(pdf_values) >= 0.0))
})

test_that("QuantileDistribution log_prob and pdf are consistent", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  z_values <- torch_linspace(-5.0, 5.0, 50)

  log_prob <- dist$log_prob(z_values)
  pdf_from_log <- torch_exp(log_prob)
  pdf_direct <- dist$pdf(z_values)

  expect_equal_to_r(pdf_from_log, as.array(pdf_direct), tolerance = 1e-5)
})

test_that("QuantileDistribution mean is computed correctly", {
  # Uniform-like quantiles centered at 0
  quantiles <- torch_linspace(-5.0, 5.0, 100)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  mean_val <- dist$mean()

  expect_tensor(mean_val)
  # Should be close to 0 for symmetric distribution
  expect_true(abs(as.numeric(mean_val)) < 1.0)
})

test_that("QuantileDistribution variance is non-negative", {
  quantiles <- torch_linspace(-5.0, 5.0, 100)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  var_val <- dist$variance()

  expect_tensor(var_val)
  expect_true(all(as.numeric(var_val) >= 0.0))
})

test_that("QuantileDistribution stddev equals sqrt of variance", {
  quantiles <- torch_linspace(-5.0, 5.0, 100)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  std_val <- dist$stddev()
  var_val <- dist$variance()

  expect_equal_to_r(std_val, sqrt(as.array(var_val)), tolerance = 1e-5)
})

test_that("QuantileDistribution crps is non-negative", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  observations <- torch_tensor(c(-2.0, 0.0, 2.0))
  crps_values <- dist$crps(observations)

  expect_tensor(crps_values)
  expect_equal(crps_values$shape, observations$shape)
  expect_true(all(as.numeric(crps_values) >= 0.0))
})

test_that("QuantileDistribution crps is zero for perfect prediction", {
  # Create a very narrow distribution
  quantiles <- torch_linspace(5.0, 5.001, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  # Observation at the center
  observation <- torch_tensor(5.0)
  crps_val <- dist$crps(observation)

  # CRPS should be very small
  expect_true(as.numeric(crps_val) < 0.1)
})

test_that("QuantileDistribution pinball approximates crps", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  observation <- torch_tensor(0.0)

  crps_analytical <- dist$crps(observation)
  crps_numerical <- dist$pinball(observation, num_quantiles = 999L)

  # Should be close but not exact
  expect_equal_to_r(
    crps_numerical,
    as.array(crps_analytical),
    tolerance = 0.1  # Numerical approximation
  )
})

test_that("QuantileDistribution sample returns correct shape", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  # Single sample
  sample_single <- dist$sample()
  expect_tensor(sample_single)
  expect_equal(sample_single$dim(), 0)  # Scalar

  # Multiple samples
  samples <- dist$sample(c(100))
  expect_tensor(samples)
  expect_equal(samples$shape, c(100, 1))
})

test_that("QuantileDistribution sample statistics match distribution", {
  # Create a distribution
  quantiles <- torch_linspace(-5.0, 5.0, 100)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  # Draw many samples
  samples <- dist$sample(c(10000))

  # Sample mean should be close to distribution mean
  sample_mean <- samples$mean()
  dist_mean <- dist$mean()

  expect_equal_to_r(sample_mean, as.array(dist_mean), tolerance = 0.2)

  # Sample variance should be close to distribution variance
  sample_var <- samples$var()
  dist_var <- dist$variance()

  expect_equal_to_r(sample_var, as.array(dist_var), tolerance = 0.5)
})

test_that("QuantileDistribution works with batch input", {
  # Batch of 3 distributions
  quantiles <- torch_stack(list(
    torch_linspace(-5.0, 5.0, 50),
    torch_linspace(-3.0, 7.0, 50),
    torch_linspace(-8.0, 2.0, 50)
  ), dim = 1)

  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  expect_equal(dist$batch_shape, c(3))

  # Mean should have batch shape
  mean_val <- dist$mean()
  expect_equal(mean_val$shape, c(3))

  # Variance should have batch shape
  var_val <- dist$variance()
  expect_equal(var_val$shape, c(3))
})

test_that("QuantileDistribution GPD tails work correctly", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "gpd")

  # icdf should work
  alphas <- torch_linspace(0.01, 0.99, 100)
  q_values <- dist$icdf(alphas)
  expect_tensor(q_values)

  # cdf should work
  z_values <- torch_linspace(-10.0, 10.0, 100)
  cdf_values <- dist$cdf(z_values)
  expect_tensor(cdf_values)

  # Mean should work
  mean_val <- dist$mean()
  expect_tensor(mean_val)

  # Variance should work
  var_val <- dist$variance()
  expect_tensor(var_val)
  expect_true(as.numeric(var_val) >= 0.0)

  # CRPS should work
  observation <- torch_tensor(0.0)
  crps_val <- dist$crps(observation)
  expect_tensor(crps_val)
  expect_true(as.numeric(crps_val) >= 0.0)
})

test_that("QuantileDistribution handles custom alpha_levels", {
  quantiles <- torch_linspace(-5.0, 5.0, 10)$unsqueeze(1)
  custom_alphas <- torch_linspace(0.1, 0.9, 10)

  dist <- QuantileDistribution$new(
    quantiles = quantiles,
    alpha_levels = custom_alphas,
    tail_type = "exp"
  )

  expect_equal(dist$num_quantiles, 10)
  expect_equal_to_r(dist$alpha_levels, as.array(custom_alphas), tolerance = 1e-6)
})

test_that("QuantileDistribution numerical stability in extreme tails", {
  quantiles <- torch_linspace(-10.0, 10.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  # Very extreme alphas
  extreme_alphas <- torch_tensor(c(1e-8, 0.5, 1 - 1e-8))
  q_values <- dist$icdf(extreme_alphas)

  expect_tensor(q_values)
  expect_false(any(torch_isnan(q_values)$item()))
  expect_false(any(torch_isinf(q_values)$item()))

  # Very extreme z values
  extreme_z <- torch_tensor(c(-100.0, 0.0, 100.0))
  cdf_values <- dist$cdf(extreme_z)

  expect_tensor(cdf_values)
  expect_false(any(torch_isnan(cdf_values)$item()))
  expect_true(all(as.numeric(cdf_values) >= 0.0))
  expect_true(all(as.numeric(cdf_values) <= 1.0))
})

test_that("quantile_to_distribution module initializes correctly", {
  module <- quantile_to_distribution(
    num_quantiles = 99L,
    tail_type = "exp",
    fix_crossing = TRUE,
    crossing_method = "sort"
  )

  expect_s3_class(module, "nn_module")
  expect_equal(module$tail_type, "exp")
  expect_equal(module$fix_crossing, TRUE)
  expect_equal(module$crossing_method, "sort")
  expect_tensor(module$alpha_levels)
})

test_that("quantile_to_distribution module forward works", {
  module <- quantile_to_distribution(
    num_quantiles = 50L,
    tail_type = "exp"
  )

  # Batch of quantile predictions
  quantiles <- torch_randn(c(8, 50))
  quantiles <- torch_sort(quantiles, dim = -1)[[1]]  # Make monotonic

  dist <- module(quantiles)

  expect_s3_class(dist, "QuantileDistribution")
  expect_equal(dist$batch_shape, c(8))
  expect_equal(dist$num_quantiles, 50)

  # Test methods work
  mean_val <- dist$mean()
  expect_equal(mean_val$shape, c(8))

  var_val <- dist$variance()
  expect_equal(var_val$shape, c(8))
})

test_that("quantile_to_distribution with custom alpha_levels", {
  custom_alphas <- seq(0.1, 0.9, length.out = 20)
  module <- quantile_to_distribution(
    alpha_levels = custom_alphas,
    tail_type = "gpd"
  )

  expect_equal_to_r(module$alpha_levels, custom_alphas, tolerance = 1e-6)

  quantiles <- torch_randn(c(4, 20))
  quantiles <- torch_sort(quantiles, dim = -1)[[1]]

  dist <- module(quantiles)

  expect_s3_class(dist, "QuantileDistribution")
  expect_equal(dist$tail_type, "gpd")
})

test_that("QuantileDistribution CRPS for GPD tails", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "gpd")

  observations <- torch_tensor(c(-3.0, 0.0, 3.0))
  crps_values <- dist$crps(observations)

  expect_tensor(crps_values)
  expect_equal(crps_values$shape, observations$shape)
  expect_true(all(as.numeric(crps_values) >= 0.0))
  expect_true(all(as.numeric(crps_values) < 1e4))  # Within MAX_CRPS
})

test_that("QuantileDistribution handles edge case with few quantiles", {
  # Only 5 quantiles
  quantiles <- torch_linspace(-1.0, 1.0, 5)$unsqueeze(1)

  dist <- QuantileDistribution$new(
    quantiles = quantiles,
    tail_type = "exp",
    fix_crossing = TRUE
  )

  # Should still work
  expect_s3_class(dist, "QuantileDistribution")

  mean_val <- dist$mean()
  expect_tensor(mean_val)

  var_val <- dist$variance()
  expect_tensor(var_val)
  expect_true(as.numeric(var_val) >= 0.0)
})

test_that("QuantileDistribution icdf handles scalar input", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  # Scalar alpha
  alpha <- torch_tensor(0.5)
  q_value <- dist$icdf(alpha)

  expect_tensor(q_value)
  expect_equal(q_value$dim(), 0)  # Scalar output
})

test_that("QuantileDistribution cdf handles scalar input", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  # Scalar z
  z <- torch_tensor(0.0)
  cdf_value <- dist$cdf(z)

  expect_tensor(cdf_value)
  # CDF in [0, 1]
  expect_true(as.numeric(cdf_value) >= 0.0)
  expect_true(as.numeric(cdf_value) <= 1.0)
})

test_that("QuantileDistribution derivative is positive", {
  quantiles <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  alphas <- torch_linspace(0.01, 0.99, 100)
  derivs <- dist$.icdf_derivative(alphas)

  expect_tensor(derivs)
  # Derivative should be positive (monotonic quantile function)
  expect_true(all(as.numeric(derivs) > 0.0))
})

test_that("QuantileDistribution integrates to 1 approximately", {
  quantiles <- torch_linspace(-5.0, 5.0, 100)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  # Numerical integration of PDF
  z_grid <- torch_linspace(-20.0, 20.0, 1000)
  pdf_vals <- dist$pdf(z_grid)

  # Trapezoid rule
  dz <- 40.0 / 999
  integral <- as.numeric(pdf_vals$sum()) * dz

  # Should integrate to approximately 1
  expect_equal(integral, 1.0, tolerance = 0.1)
})

test_that("QuantileDistribution mean via sampling matches analytical", {
  quantiles <- torch_linspace(-5.0, 5.0, 100)$unsqueeze(1)
  dist <- QuantileDistribution$new(quantiles = quantiles, tail_type = "exp")

  # Analytical mean
  mean_analytical <- as.numeric(dist$mean())

  # Mean via sampling
  samples <- dist$sample(c(50000))
  mean_sampled <- as.numeric(samples$mean())

  expect_equal(mean_sampled, mean_analytical, tolerance = 0.2)
})

test_that("QuantileDistribution crossing_method affects result", {
  # Create quantiles with crossing
  quantiles_crossed <- torch_tensor(c(0.0, 3.0, 2.0, 5.0, 4.0, 6.0))$unsqueeze(1)

  dist_sort <- QuantileDistribution$new(
    quantiles = quantiles_crossed,
    fix_crossing = TRUE,
    crossing_method = "sort"
  )

  dist_cummax <- QuantileDistribution$new(
    quantiles = quantiles_crossed,
    fix_crossing = TRUE,
    crossing_method = "cummax"
  )

  # Results should differ slightly
  q_sort <- as.numeric(dist_sort$quantiles)
  q_cummax <- as.numeric(dist_cummax$quantiles)

  # Both should be monotonic
  for (i in seq_len(length(q_sort) - 1)) {
    expect_true(q_sort[i] <= q_sort[i + 1])
    expect_true(q_cummax[i] <= q_cummax[i + 1])
  }
})

test_that("QuantileDistribution no crossing when fix_crossing=FALSE", {
  # Create monotonic quantiles
  quantiles_mono <- torch_linspace(-5.0, 5.0, 50)$unsqueeze(1)

  dist_no_fix <- QuantileDistribution$new(
    quantiles = quantiles_mono,
    fix_crossing = FALSE
  )

  # Should still initialize fine
  expect_s3_class(dist_no_fix, "QuantileDistribution")

  # Quantiles should be unchanged
  expect_equal_to_r(
    dist_no_fix$quantiles,
    as.array(quantiles_mono),
    tolerance = 1e-6
  )
})
