#' Configuration constants for Quantile Distribution
#'
#' @description
#' Numerical stability parameters, tail parameters, computational limits,
#' and tail inference parameters for quantile distribution estimation.
#'
#' @section Numerical Stability Parameters:
#' - `TOL`: General numerical tolerance (default: 1e-6)
#' - `MIN_SLOPE`: Minimum allowed slope dQ/dα (default: 1e-6)
#' - `MAX_SLOPE`: Maximum allowed slope dQ/dα (default: 1e6)
#'
#' @section Tail Parameters:
#' - `MIN_BETA`: Minimum tail scale parameter β (default: 0.01)
#' - `MAX_BETA`: Maximum tail scale parameter β (default: 100.0)
#' - `MIN_ETA`: Minimum GPD shape parameter η (default: -0.49)
#' - `MAX_ETA`: Maximum GPD shape parameter η (default: 0.49)
#' - `ETA_TOLERANCE`: Threshold for treating η ≈ 0 (default: 0.01)
#'
#' @section Computational Limits:
#' - `MAX_LOG_RATIO`: Maximum log(ratio) (default: 15.0)
#' - `MAX_EXPONENT`: Maximum exponent before exp() (default: 15.0)
#' - `MAX_CRPS`: Maximum CRPS value (default: 1e4)
#'
#' @section Tail Inference Parameters:
#' - `TAIL_QUANTILES_FOR_ESTIMATION`: Number of tail quantiles (default: 20)
#'
#' @export
QuantileDistributionConfig <- R6::R6Class(
  "QuantileDistributionConfig",
  public = list(
    #' @field TOL General numerical tolerance (default: 1e-6).
    TOL = 1e-6,
    #' @field MIN_SLOPE Minimum allowed slope dQ/dα (default: 1e-6).
    MIN_SLOPE = 1e-6,
    #' @field MAX_SLOPE Maximum allowed slope dQ/dα (default: 1e6).
    MAX_SLOPE = 1e6,
    #' @field MIN_BETA Minimum tail scale parameter β (default: 0.01).
    MIN_BETA = 0.01,
    #' @field MAX_BETA Maximum tail scale parameter β (default: 100.0).
    MAX_BETA = 100.0,
    #' @field MIN_ETA Minimum GPD shape parameter η (default: -0.49).
    MIN_ETA = -0.49,
    #' @field MAX_ETA Maximum GPD shape parameter η (default: 0.49).
    MAX_ETA = 0.49,
    #' @field ETA_TOLERANCE Threshold for treating η ≈ 0 (default: 0.01).
    ETA_TOLERANCE = 0.01,
    #' @field MAX_LOG_RATIO Maximum log(ratio) (default: 15.0).
    MAX_LOG_RATIO = 15.0,
    #' @field MAX_EXPONENT Maximum exponent before exp() (default: 15.0).
    MAX_EXPONENT = 15.0,
    #' @field MAX_CRPS Maximum CRPS value (default: 1e4).
    MAX_CRPS = 1e4,
    #' @field TAIL_QUANTILES_FOR_ESTIMATION Number of tail quantiles (default: 20).
    TAIL_QUANTILES_FOR_ESTIMATION = 20L
  )
)

#' Pool Adjacent Violators Algorithm (PAVA) for isotonic regression
#'
#' @description
#' Solves: argmin_x Σ w_i(y_i - x_i)² subject to x_i ≤ x_{i+1}
#'
#' @param y Tensor. Input values to make monotonically non-decreasing.
#'   Shape: `(*batch_shape, n)`.
#' @param weights Tensor or NULL. Case weights (must be positive).
#'   Shape: `(*batch_shape, n)` or NULL for uniform weights.
#'
#' @return Tensor. Isotonic (monotonically non-decreasing) values.
#'   Shape: `(*batch_shape, n)`.
#'
#' @keywords internal
isotonic_regression_pava <- function(y, weights = NULL) {
  # For R implementation, use simpler approach
  # Can be optimized later with Rcpp if needed

  batch_shape <- y$shape[-length(y$shape)]
  n <- y$shape[length(y$shape)]
  device <- y$device
  dtype <- y$dtype

  if (length(batch_shape) == 0) {
    # Single sequence
    return(.pava_single(y, weights))
  }

  # Batch processing
  y_flat <- y$reshape(c(-1, n))
  batch_size <- y_flat$shape[1]

  result_list <- list()
  if (is.null(weights)) {
    for (i in seq_len(batch_size)) {
      result_list[[i]] <- .pava_single(y_flat[i, ..], NULL)
    }
  } else {
    w_flat <- weights$reshape(c(-1, n))
    for (i in seq_len(batch_size)) {
      result_list[[i]] <- .pava_single(y_flat[i, ..], w_flat[i, ..])
    }
  }

  result <- torch_stack(result_list, dim = 1)
  result$reshape(c(batch_shape, n))
}

#' Single-sequence PAVA
#' @keywords internal
.pava_single <- function(y, w = NULL) {
  n <- y$shape[1]
  if (n <= 1) return(y$clone())

  # Convert to R vectors for processing
  y_vec <- as.numeric(y)
  if (is.null(w)) {
    w_vec <- rep(1, n)
  } else {
    w_vec <- as.numeric(w)
  }

  # PAVA algorithm
  block_values <- numeric(n)
  block_weights <- numeric(n)
  block_ends <- integer(n)
  num_blocks <- 0

  for (i in seq_len(n)) {
    num_blocks <- num_blocks + 1
    block_values[num_blocks] <- y_vec[i]
    block_weights[num_blocks] <- w_vec[i]
    block_ends[num_blocks] <- i

    # Merge while violation exists
    while (num_blocks > 1 && block_values[num_blocks - 1] > block_values[num_blocks]) {
      v1 <- block_values[num_blocks - 1]
      w1 <- block_weights[num_blocks - 1]
      v2 <- block_values[num_blocks]
      w2 <- block_weights[num_blocks]
      end2 <- block_ends[num_blocks]

      merged_w <- w1 + w2
      merged_v <- (v1 * w1 + v2 * w2) / merged_w

      block_values[num_blocks - 1] <- merged_v
      block_weights[num_blocks - 1] <- merged_w
      block_ends[num_blocks - 1] <- end2
      num_blocks <- num_blocks - 1
    }
  }

  # Reconstruct
  result <- numeric(n)
  start <- 1
  for (b in seq_len(num_blocks)) {
    val <- block_values[b]
    end <- block_ends[b]
    result[start:end] <- val
    start <- end + 1
  }

  torch_tensor(result, dtype = y$dtype, device = y$device)
}

#' Enforce monotonicity of quantiles to fix crossing
#'
#' @param quantiles Tensor. Predicted quantiles that may have crossing violations.
#'   Shape: `(*batch_shape, num_quantiles)`.
#' @param method Character. Method for fixing crossing:
#'   - "sort": Sort values (fast, default)
#'   - "isotonic": Pool Adjacent Violators (optimal L2, O(N))
#'   - "cummax": Cumulative maximum (fast but distorts distribution)
#' @param weights Tensor or NULL. Case weights for PAVA method.
#'   Shape: `(*batch_shape, num_quantiles)` or NULL.
#'
#' @return Tensor. Monotonically non-decreasing quantiles.
#'   Shape: `(*batch_shape, num_quantiles)`.
#'
#' @export
enforce_monotonicity <- function(quantiles, method = "sort", weights = NULL) {
  if (method == "isotonic") {
    isotonic_regression_pava(quantiles, weights)
  } else if (method == "cummax") {
    torch_cummax(quantiles, dim = -1)[[1]]
  } else if (method == "sort") {
    torch_sort(quantiles, dim = -1)[[1]]
  } else {
    cli_abort("Unknown method: {method}. Use 'isotonic', 'cummax', or 'sort'.")
  }
}

#' Estimate exponential tail parameters using log-space linear regression
#'
#' @param quantiles Tensor. Quantile values after monotonicity correction.
#'   Shape: `(*batch_shape, num_quantiles)`.
#' @param alpha_levels Tensor. Probability levels corresponding to quantiles.
#'   Shape: `(num_quantiles,)`.
#' @param num_tail_quantiles Integer. Number of quantiles in each tail to use.
#'
#' @return List with:
#'   - `beta_l`: Left tail scale parameter. Shape: `(*batch_shape,)`.
#'   - `beta_r`: Right tail scale parameter. Shape: `(*batch_shape,)`.
#'
#' @keywords internal
estimate_exp_tail_params <- function(quantiles, alpha_levels, num_tail_quantiles = 20L) {
  cfg <- QuantileDistributionConfig$new()
  n <- quantiles$shape[length(quantiles$shape)]
  k <- min(num_tail_quantiles, n %/% 4L)

  # === Left tail: Q(α) = β_L·ln(α) + c_L ===
  alpha_left <- alpha_levels[1:k]
  q_left <- quantiles[.., 1:k]

  # Log-transform alpha
  ln_alpha_left <- torch_log(alpha_left$clamp(min = cfg$TOL))

  # Linear regression: β = Cov(Q, ln(α)) / Var(ln(α))
  ln_alpha_mean <- ln_alpha_left$mean()
  ln_alpha_centered <- ln_alpha_left - ln_alpha_mean

  q_left_mean <- q_left$mean(dim = -1, keepdim = TRUE)
  q_left_centered <- q_left - q_left_mean

  # Compute covariance and variance
  cov_left <- (q_left_centered * ln_alpha_centered)$mean(dim = -1)
  var_ln_alpha_left <- (ln_alpha_centered^2)$mean()

  beta_l <- cov_left / var_ln_alpha_left$clamp(min = cfg$TOL)
  beta_l <- torch_clamp(beta_l$abs(), min = cfg$MIN_BETA, max = cfg$MAX_BETA)

  # === Right tail: Q(α) = -β_R·ln(1-α) + c_R ===
  alpha_right <- alpha_levels[(n - k + 1):n]
  q_right <- quantiles[.., (n - k + 1):n]

  # Log-transform (1 - alpha)
  ln_one_minus_alpha <- torch_log((1 - alpha_right)$clamp(min = cfg$TOL))

  # Linear regression
  ln_1ma_mean <- ln_one_minus_alpha$mean()
  ln_1ma_centered <- ln_one_minus_alpha - ln_1ma_mean

  q_right_mean <- q_right$mean(dim = -1, keepdim = TRUE)
  q_right_centered <- q_right - q_right_mean

  cov_right <- (q_right_centered * ln_1ma_centered)$mean(dim = -1)
  var_ln_1ma <- (ln_1ma_centered^2)$mean()

  # Note: Q = -β·ln(1-α) + c, so coefficient is -β
  beta_r <- -cov_right / var_ln_1ma$clamp(min = cfg$TOL)
  beta_r <- torch_clamp(beta_r$abs(), min = cfg$MIN_BETA, max = cfg$MAX_BETA)

  list(beta_l = beta_l, beta_r = beta_r)
}

#' Estimate GPD tail parameters using Pickands-like estimator
#'
#' @param quantiles Tensor. Quantile values after monotonicity correction.
#'   Shape: `(*batch_shape, num_quantiles)`.
#' @param alpha_levels Tensor. Probability levels.
#'   Shape: `(num_quantiles,)`.
#' @param num_tail_quantiles Integer. Number of quantiles in each tail.
#'
#' @return List with:
#'   - `eta_l`: Left tail shape parameter. Shape: `(*batch_shape,)`.
#'   - `mu_l`: Left tail scale parameter. Shape: `(*batch_shape,)`.
#'   - `eta_r`: Right tail shape parameter. Shape: `(*batch_shape,)`.
#'   - `mu_r`: Right tail scale parameter. Shape: `(*batch_shape,)`.
#'
#' @keywords internal
estimate_gpd_tail_params <- function(quantiles, alpha_levels, num_tail_quantiles = 20L) {
  cfg <- QuantileDistributionConfig$new()
  n <- quantiles$shape[length(quantiles$shape)]
  k <- min(num_tail_quantiles, n %/% 4L)

  # First get exponential estimates for μ (scale parameter)
  exp_params <- estimate_exp_tail_params(quantiles, alpha_levels, num_tail_quantiles)
  beta_l <- exp_params$beta_l
  beta_r <- exp_params$beta_r

  # === Left tail - Pickands-like estimation ===
  idx_1 <- 1L
  idx_2 <- (k %/% 3L) + 1L
  idx_3 <- (2L * k %/% 3L) + 1L

  q1_left <- quantiles[.., idx_1]
  q2_left <- quantiles[.., idx_2]
  q3_left <- quantiles[.., idx_3]

  alpha_1 <- alpha_levels[idx_1]
  alpha_2 <- alpha_levels[idx_2]
  alpha_3 <- alpha_levels[idx_3]

  # Q spacing ratios
  delta_q_12 <- q2_left - q1_left
  delta_q_23 <- q3_left - q2_left

  # Log-alpha spacing ratios
  ln_alpha_12 <- torch_log((alpha_2 / alpha_1)$clamp(min = cfg$TOL))
  ln_alpha_23 <- torch_log((alpha_3 / alpha_2)$clamp(min = cfg$TOL))

  # For exponential tail: ΔQ_12 / ΔQ_23 = ln(α2/α1) / ln(α3/α2)
  expected_q_ratio <- ln_alpha_12 / ln_alpha_23$clamp(min = cfg$TOL)
  actual_q_ratio <- delta_q_12 / delta_q_23$clamp(min = cfg$TOL)

  # η estimate: deviation from exponential behavior
  ratio_deviation <- actual_q_ratio / expected_q_ratio$clamp(min = cfg$TOL)

  # Clamp intermediate values for numerical stability
  log_ratio <- torch_log(ratio_deviation$clamp(min = cfg$TOL))
  log_ratio_clamped <- torch_clamp(log_ratio, min = -cfg$MAX_LOG_RATIO, max = cfg$MAX_LOG_RATIO)

  eta_l_raw <- torch_where(
    ratio_deviation > cfg$TOL,
    log_ratio_clamped / ln_alpha_12$abs()$clamp(min = cfg$TOL),
    torch_zeros_like(ratio_deviation)
  )
  # Ensure no NaN or Inf values before clamping
  eta_l_raw <- torch_where(torch_isnan(eta_l_raw) | torch_isinf(eta_l_raw), torch_zeros_like(eta_l_raw), eta_l_raw)
  eta_l <- torch_clamp(eta_l_raw, min = cfg$MIN_ETA, max = cfg$MAX_ETA)
  mu_l <- beta_l

  # === Right tail - Pickands-like estimation ===
  idx_1_r <- n
  idx_2_r <- n - (k %/% 3L)
  idx_3_r <- n - (2L * k %/% 3L)

  q1_right <- quantiles[.., idx_1_r]
  q2_right <- quantiles[.., idx_2_r]
  q3_right <- quantiles[.., idx_3_r]

  alpha_1_r <- alpha_levels[idx_1_r]
  alpha_2_r <- alpha_levels[idx_2_r]
  alpha_3_r <- alpha_levels[idx_3_r]

  # For right tail, larger α = larger Q
  delta_q_12_r <- q1_right - q2_right
  delta_q_23_r <- q2_right - q3_right

  # Use (1-α) ratios
  one_m_1 <- (1 - alpha_1_r)$clamp(min = cfg$TOL)
  one_m_2 <- (1 - alpha_2_r)$clamp(min = cfg$TOL)
  one_m_3 <- (1 - alpha_3_r)$clamp(min = cfg$TOL)

  ln_1ma_12 <- torch_log((one_m_2 / one_m_1)$clamp(min = cfg$TOL))
  ln_1ma_23 <- torch_log((one_m_3 / one_m_2)$clamp(min = cfg$TOL))

  expected_q_ratio_r <- ln_1ma_12 / ln_1ma_23$clamp(min = cfg$TOL)
  actual_q_ratio_r <- delta_q_12_r / delta_q_23_r$clamp(min = cfg$TOL)

  ratio_deviation_r <- actual_q_ratio_r / expected_q_ratio_r$clamp(min = cfg$TOL)

  # Clamp intermediate values for numerical stability
  log_ratio_r <- torch_log(ratio_deviation_r$clamp(min = cfg$TOL))
  log_ratio_r_clamped <- torch_clamp(log_ratio_r, min = -cfg$MAX_LOG_RATIO, max = cfg$MAX_LOG_RATIO)

  eta_r_raw <- torch_where(
    ratio_deviation_r > cfg$TOL,
    log_ratio_r_clamped / ln_1ma_12$abs()$clamp(min = cfg$TOL),
    torch_zeros_like(ratio_deviation_r)
  )
  # Ensure no NaN or Inf values before clamping
  eta_r_raw <- torch_where(torch_isnan(eta_r_raw) | torch_isinf(eta_r_raw), torch_zeros_like(eta_r_raw), eta_r_raw)
  eta_r <- torch_clamp(eta_r_raw, min = cfg$MIN_ETA, max = cfg$MAX_ETA)
  mu_r <- beta_r

  list(eta_l = eta_l, mu_l = mu_l, eta_r = eta_r, mu_r = mu_r)
}

#' Probability distribution constructed from predicted quantiles
#'
#' @description
#' Wraps a set of predicted quantiles into a proper distribution with:
#' - Monotonicity enforcement (fixes quantile crossing)
#' - Tail extrapolation (exponential or GPD) with data-inferred parameters
#' - Analytical statistics (CDF, PDF, CRPS, mean, variance)
#'
#' @importFrom R6 R6Class
#' @export
QuantileDistribution <- R6::R6Class(
  "QuantileDistribution",
  public = list(
    #' @field cfg Configuration object
    cfg = NULL,
    #' @field tol Numerical tolerance
    tol = NULL,
    #' @field tail_type Type of tail ("exp" or "gpd")
    tail_type = NULL,
    #' @field batch_shape Batch shape
    batch_shape = NULL,
    #' @field num_quantiles Number of quantiles
    num_quantiles = NULL,
    #' @field alpha_levels Probability levels
    alpha_levels = NULL,
    #' @field quantiles Quantile values
    quantiles = NULL,

    # Spline fields
    #' @field alpha_lo Lower alpha boundaries
    alpha_lo = NULL,
    #' @field alpha_hi Upper alpha boundaries
    alpha_hi = NULL,
    #' @field q_lo Lower quantile boundaries
    q_lo = NULL,
    #' @field q_hi Upper quantile boundaries
    q_hi = NULL,
    #' @field delta_alpha Alpha differences
    delta_alpha = NULL,
    #' @field delta_q Quantile differences
    delta_q = NULL,
    #' @field slopes Segment slopes
    slopes = NULL,
    #' @field num_segments Number of segments
    num_segments = NULL,
    #' @field alpha_l Left boundary alpha
    alpha_l = NULL,
    #' @field alpha_r Right boundary alpha
    alpha_r = NULL,
    #' @field q_l Left boundary quantile
    q_l = NULL,
    #' @field q_r Right boundary quantile
    q_r = NULL,
    #' @field alpha_lo_1d 1D alpha lower boundaries
    alpha_lo_1d = NULL,
    #' @field alpha_hi_1d 1D alpha upper boundaries
    alpha_hi_1d = NULL,

    # Tail fields (exponential)
    #' @field beta_l Left tail beta
    beta_l = NULL,
    #' @field beta_r Right tail beta
    beta_r = NULL,
    #' @field tail_a_l Left tail coefficient a
    tail_a_l = NULL,
    #' @field tail_b_l Left tail coefficient b
    tail_b_l = NULL,
    #' @field tail_a_r Right tail coefficient a
    tail_a_r = NULL,
    #' @field tail_b_r Right tail coefficient b
    tail_b_r = NULL,

    # Tail fields (GPD)
    #' @field eta_l Left tail GPD shape
    eta_l = NULL,
    #' @field mu_l Left tail GPD scale
    mu_l = NULL,
    #' @field eta_r Right tail GPD shape
    eta_r = NULL,
    #' @field mu_r Right tail GPD scale
    mu_r = NULL,

    #' @description Initialize QuantileDistribution
    #' @param quantiles Tensor. Predicted quantile values.
    #' @param alpha_levels Tensor or NULL. Probability levels.
    #' @param tail_type Character. "exp" or "gpd".
    #' @param fix_crossing Logical. Fix quantile crossing?
    #' @param crossing_method Character. Method for fixing crossing.
    initialize = function(quantiles,
                         alpha_levels = NULL,
                         tail_type = "exp",
                         fix_crossing = TRUE,
                         crossing_method = "sort") {
      self$cfg <- QuantileDistributionConfig$new()
      self$tol <- self$cfg$TOL
      self$tail_type <- tail_type

      # Store shapes
      self$batch_shape <- quantiles$shape[-quantiles$ndim]
      self$num_quantiles <- quantiles$shape[quantiles$ndim]

      # Default alpha levels
      if (is.null(alpha_levels)) {
        alpha_levels <- torch_linspace(
          0.0, 1.0, self$num_quantiles + 2L,
          device = quantiles$device,
          dtype = quantiles$dtype
        )[2:(self$num_quantiles + 1L)]
      }
      self$alpha_levels <- alpha_levels

      # Fix quantile crossing
      if (fix_crossing) {
        quantiles <- enforce_monotonicity(quantiles, method = crossing_method)
      }

      self$quantiles <- quantiles

      # Setup internal structures
      self$.setup_spline()
      self$.setup_tails()
    },

    #' @description Setup linear spline segments
    .setup_spline = function() {
      # Expand alpha_levels for batch operations
      alpha <- self$alpha_levels
      n_batch_dims <- length(self$batch_shape)
      for (i in seq_len(n_batch_dims)) {
        alpha <- alpha$unsqueeze(1L)
      }
      if (n_batch_dims > 0) {
        expand_shape <- c(self$batch_shape, -1L)
        alpha <- alpha$expand(expand_shape)
      }

      # Segment boundaries
      n <- self$num_quantiles
      self$alpha_lo <- alpha[.., 1:(n - 1L)]
      self$alpha_hi <- alpha[.., 2:n]
      self$q_lo <- self$quantiles[.., 1:(n - 1L)]
      self$q_hi <- self$quantiles[.., 2:n]

      # Segment properties
      self$delta_alpha <- self$alpha_hi - self$alpha_lo
      self$delta_q <- self$q_hi - self$q_lo
      self$slopes <- self$delta_q / torch_clamp(self$delta_alpha, min = self$tol)

      self$num_segments <- self$num_quantiles - 1L

      # Boundary values
      self$alpha_l <- as.numeric(self$alpha_levels[1])
      self$alpha_r <- as.numeric(self$alpha_levels[self$num_quantiles])
      self$q_l <- self$quantiles[.., 1]
      self$q_r <- self$quantiles[.., self$num_quantiles]

      # 1D boundaries for searchsorted
      self$alpha_lo_1d <- self$alpha_levels[1:(n - 1L)]
      self$alpha_hi_1d <- self$alpha_levels[2:n]
    },

    #' @description Setup tail parameters
    .setup_tails = function() {
      cfg <- self$cfg
      device <- self$quantiles$device
      dtype <- self$quantiles$dtype

      num_tail_q <- cfg$TAIL_QUANTILES_FOR_ESTIMATION

      if (self$tail_type == "exp") {
        # Estimate β using log-space regression
        params <- estimate_exp_tail_params(
          self$quantiles,
          self$alpha_levels,
          num_tail_quantiles = num_tail_q
        )
        self$beta_l <- params$beta_l
        self$beta_r <- params$beta_r

        # Compute tail coefficients
        # Left: Q(α) = a_l·ln(α) + b_l where a_l = β_l
        alpha_l_safe <- max(self$alpha_l, self$tol)
        self$tail_a_l <- self$beta_l
        self$tail_b_l <- self$q_l - self$tail_a_l * torch_log(
          torch_tensor(alpha_l_safe, device = device, dtype = dtype)
        )

        # Right: Q(α) = a_r·ln(1-α) + b_r where a_r = -β_r
        alpha_r_safe <- min(self$alpha_r, 1 - self$tol)
        self$tail_a_r <- -self$beta_r
        self$tail_b_r <- self$q_r - self$tail_a_r * torch_log(
          torch_tensor(1 - alpha_r_safe, device = device, dtype = dtype)
        )
      } else {
        # GPD tails
        params <- estimate_gpd_tail_params(
          self$quantiles,
          self$alpha_levels,
          num_tail_quantiles = num_tail_q
        )
        self$eta_l <- params$eta_l
        self$mu_l <- params$mu_l
        self$eta_r <- params$eta_r
        self$mu_r <- params$mu_r
      }
    },

    #' @description Compute quantile function Q(α) = F^(-1)(α)
    #' @param alpha Tensor. Probability levels.
    #' @return Tensor. Quantile values Q(α).
    icdf = function(alpha) {
      squeeze_output <- FALSE
      if (alpha$dim() == 0) {
        alpha <- alpha$unsqueeze(1L)
        squeeze_output <- TRUE
      }

      alpha_shape <- alpha$shape

      # Expand alpha for batch dimensions if needed
      if (alpha$dim() == 1) {
        alpha_expanded <- alpha
        for (i in seq_along(self$batch_shape)) {
          alpha_expanded <- alpha_expanded$unsqueeze(1L)
        }
        expand_shape <- c(self$batch_shape, alpha_shape)
        alpha_expanded <- alpha_expanded$expand(expand_shape)
      } else {
        alpha_expanded <- alpha
      }

      # Compute in each region
      q_left <- self$.icdf_left_tail(alpha_expanded)
      q_right <- self$.icdf_right_tail(alpha_expanded)
      q_spline <- self$.icdf_spline(alpha_expanded)

      # Select based on region
      result <- torch_where(
        alpha_expanded < self$alpha_l,
        q_left,
        torch_where(alpha_expanded > self$alpha_r, q_right, q_spline)
      )

      if (squeeze_output) {
        # For scalar input, extract the single element
        if (result$dim() == 2 && all(result$shape == c(1, 1))) {
          result <- result[1, 1]
        } else if (result$dim() == 1 && result$shape[1] == 1) {
          result <- result[1]
        } else {
          result <- result$squeeze()
        }
      }

      result
    },

    #' @description Expand batch parameter to match alpha shape
    #' @param param Tensor. Parameter estimates of the tail function.
    #' @param alpha Tensor. Probability levels.
    .expand_to_alpha = function(param, alpha) {
      n_expand <- alpha$dim() - param$dim()
      result <- param
      for (i in seq_len(n_expand)) {
        result <- result$unsqueeze(-1L)
      }
      result$expand_as(alpha)
    },

    #' @description Left tail quantile
    #' @param alpha Tensor. Probability levels.
    .icdf_left_tail = function(alpha) {
      if (self$tail_type == "exp") {
        a <- self$.expand_to_alpha(self$tail_a_l, alpha)
        b <- self$.expand_to_alpha(self$tail_b_l, alpha)
        alpha_safe <- torch_clamp(alpha, min = self$tol)
        return(a * torch_log(alpha_safe) + b)
      } else {
        return(self$.icdf_gpd_left(alpha))
      }
    },

    #' @description Right tail quantile
    #' @param alpha Tensor. Probability levels.
    .icdf_right_tail = function(alpha) {
      if (self$tail_type == "exp") {
        a <- self$.expand_to_alpha(self$tail_a_r, alpha)
        b <- self$.expand_to_alpha(self$tail_b_r, alpha)
        one_m_alpha <- torch_clamp(1 - alpha, min = self$tol)
        return(a * torch_log(one_m_alpha) + b)
      } else {
        return(self$.icdf_gpd_right(alpha))
      }
    },

    #' @description GPD left tail
    #' @param alpha Tensor. Probability levels.
    .icdf_gpd_left = function(alpha) {
      cfg <- self$cfg
      eta <- self$.expand_to_alpha(self$eta_l, alpha)
      mu <- self$.expand_to_alpha(self$mu_l, alpha)
      q_L <- self$.expand_to_alpha(self$q_l, alpha)

      is_exp_approx <- torch_abs(eta) < cfg$ETA_TOLERANCE
      eta_safe <- torch_where(
        is_exp_approx,
        torch_full_like(eta, cfg$ETA_TOLERANCE),
        eta
      )

      alpha_safe <- torch_clamp(alpha, min = self$tol)
      ratio <- self$alpha_l / alpha_safe
      log_ratio <- torch_clamp(torch_log(ratio), max = cfg$MAX_LOG_RATIO)
      ratio_pow <- torch_exp(torch_clamp(eta_safe * log_ratio, max = cfg$MAX_EXPONENT))

      gpd_result <- q_L - mu / eta_safe * (ratio_pow - 1.0)
      exp_result <- q_L - mu * log_ratio

      torch_where(is_exp_approx, exp_result, gpd_result)
    },

    #' @description GPD right tail
    #' @param alpha Tensor. Probability levels.
    .icdf_gpd_right = function(alpha) {
      cfg <- self$cfg
      eta <- self$.expand_to_alpha(self$eta_r, alpha)
      mu <- self$.expand_to_alpha(self$mu_r, alpha)
      q_R <- self$.expand_to_alpha(self$q_r, alpha)

      is_exp_approx <- torch_abs(eta) < cfg$ETA_TOLERANCE
      eta_safe <- torch_where(
        is_exp_approx,
        torch_full_like(eta, cfg$ETA_TOLERANCE),
        eta
      )

      one_m_alpha <- torch_clamp(1 - alpha, min = self$tol)
      ratio <- (1 - self$alpha_r) / one_m_alpha
      log_ratio <- torch_clamp(torch_log(ratio), max = cfg$MAX_LOG_RATIO)
      ratio_pow <- torch_exp(torch_clamp(eta_safe * log_ratio, max = cfg$MAX_EXPONENT))

      gpd_result <- q_R + mu / eta_safe * (ratio_pow - 1.0)
      exp_result <- q_R + mu * log_ratio

      torch_where(is_exp_approx, exp_result, gpd_result)
    },

    #' @description Piecewise linear quantile interpolation
    #' @param alpha Tensor. Probability levels.
    .icdf_spline = function(alpha) {
      seg_idx <- (
        torch_searchsorted(
          self$alpha_lo_1d$contiguous(),
          alpha$contiguous(),
          right = TRUE
        ) - 1L
      )
      seg_idx <- seg_idx$clamp(1L, self$num_segments)

      q_lo_g <- self$q_lo$gather(-1L, seg_idx)
      q_hi_g <- self$q_hi$gather(-1L, seg_idx)
      alpha_lo_g <- self$alpha_lo$gather(-1L, seg_idx)
      alpha_hi_g <- self$alpha_hi$gather(-1L, seg_idx)

      t <- (alpha - alpha_lo_g) / (alpha_hi_g - alpha_lo_g)$clamp(min = self$tol)
      result <- q_lo_g + t$clamp(0.0, 1.0) * (q_hi_g - q_lo_g)

      q_r_exp <- self$.expand_to_alpha(self$q_r, alpha)
      torch_where(alpha >= self$alpha_r, q_r_exp, result)
    },

    #' @description Compute CDF F(z) = P(Z ≤ z)
    #' @param z Tensor. Values at which to evaluate CDF.
    #' @return Tensor. CDF values in \[0, 1\].
    cdf = function(z) {
      # Handle 1D input by broadcasting
      if (z$dim() == 1 && length(self$batch_shape) > 0) {
        if (!identical(z$shape, self$batch_shape)) {
          z <- z$unsqueeze(1L)$expand(c(self$batch_shape, -1L))
        }
      }

      n_extra <- z$dim() - length(self$batch_shape)

      q_l_exp <- self$q_l
      q_r_exp <- self$q_r
      for (i in seq_len(n_extra)) {
        q_l_exp <- q_l_exp$unsqueeze(-1L)
        q_r_exp <- q_r_exp$unsqueeze(-1L)
      }

      cdf_left <- self$.cdf_left_tail(z)
      cdf_right <- self$.cdf_right_tail(z)
      cdf_spline <- self$.cdf_spline(z)

      torch_where(
        z < q_l_exp,
        cdf_left,
        torch_where(z > q_r_exp, cdf_right, cdf_spline)
      )
    },

    #' @description Expand batch parameter to match z shape
    #' @param param Tensor. Parameter estimates of the tail function.
    #' @param z Tensor. Values at which to evaluate CDF.
    .expand_to_z = function(param, z) {
      n_expand <- z$dim() - param$dim()
      result <- param
      for (i in seq_len(n_expand)) {
        result <- result$unsqueeze(-1L)
      }
      result$expand_as(z)
    },

    #' @description CDF in left tail region
    #' @param z Tensor. Values at which to evaluate CDF.
    .cdf_left_tail = function(z) {
      cfg <- self$cfg
      if (self$tail_type == "exp") {
        a <- self$.expand_to_z(self$tail_a_l, z)
        b <- self$.expand_to_z(self$tail_b_l, z)
        a_safe <- torch_clamp(a$abs(), min = self$tol)
        log_alpha <- torch_clamp((z - b) / a_safe, max = 0.0)
        alpha <- torch_exp(log_alpha)
        return(torch_clamp(alpha, min = 0.0, max = self$alpha_l))
      } else {
        eta <- self$.expand_to_z(self$eta_l, z)
        mu <- self$.expand_to_z(self$mu_l, z)
        q_L <- self$.expand_to_z(self$q_l, z)

        is_exp <- torch_abs(eta) < cfg$ETA_TOLERANCE
        eta_safe <- torch_where(is_exp, torch_full_like(eta, cfg$ETA_TOLERANCE), eta)
        mu_safe <- torch_clamp(mu, min = self$tol)

        psi <- torch_clamp((q_L - z) / mu_safe, min = 0.0, max = cfg$MAX_EXPONENT)
        inner <- torch_clamp(1.0 + eta_safe * psi, min = self$tol)
        exp_arg <- torch_clamp(
          -torch_log(inner) / eta_safe,
          min = -cfg$MAX_EXPONENT,
          max = cfg$MAX_EXPONENT
        )

        gpd_alpha <- self$alpha_l * torch_exp(exp_arg)
        exp_alpha <- self$alpha_l * torch_exp(-psi)
        alpha <- torch_where(is_exp, exp_alpha, gpd_alpha)
        return(torch_clamp(alpha, min = 0.0, max = self$alpha_l))
      }
    },

    #' @description CDF in right tail region
    #' @param z Tensor. Values at which to evaluate CDF.
    .cdf_right_tail = function(z) {
      cfg <- self$cfg
      if (self$tail_type == "exp") {
        a <- self$.expand_to_z(self$tail_a_r, z)
        b <- self$.expand_to_z(self$tail_b_r, z)
        a_safe <- torch_clamp((-a)$abs(), min = self$tol)
        log_one_m <- torch_clamp((z - b) / (-a_safe), max = 0.0)
        one_m_alpha <- torch_exp(log_one_m)
        return(torch_clamp(1.0 - one_m_alpha, min = self$alpha_r, max = 1.0))
      } else {
        eta <- self$.expand_to_z(self$eta_r, z)
        mu <- self$.expand_to_z(self$mu_r, z)
        q_R <- self$.expand_to_z(self$q_r, z)

        is_exp <- torch_abs(eta) < cfg$ETA_TOLERANCE
        eta_safe <- torch_where(is_exp, torch_full_like(eta, cfg$ETA_TOLERANCE), eta)
        mu_safe <- torch_clamp(mu, min = self$tol)

        psi <- torch_clamp((z - q_R) / mu_safe, min = 0.0, max = cfg$MAX_EXPONENT)
        inner <- torch_clamp(1.0 + eta_safe * psi, min = self$tol)
        exp_arg <- torch_clamp(
          -torch_log(inner) / eta_safe,
          min = -cfg$MAX_EXPONENT,
          max = cfg$MAX_EXPONENT
        )

        one_m_r <- 1 - self$alpha_r
        gpd_one_m <- one_m_r * torch_exp(exp_arg)
        exp_one_m <- one_m_r * torch_exp(-psi)
        one_m <- torch_where(is_exp, exp_one_m, gpd_one_m)
        return(torch_clamp(1.0 - one_m, min = self$alpha_r, max = 1.0))
      }
    },

    #' @description CDF in spline region
    #' @param z Tensor. Values at which to evaluate CDF.
    .cdf_spline = function(z) {
      # Handle z: (*batch_shape,) or (*batch_shape, n)
      added_dim <- z$dim() == length(self$batch_shape)
      if (added_dim) {
        z <- z$unsqueeze(-1L)
      }

      seg_idx <- (
        torch_searchsorted(
          self$q_lo$contiguous(),
          z$contiguous(),
          right = TRUE
        ) - 1L
      )
      seg_idx <- seg_idx$clamp(1L, self$num_segments)

      q_lo_g <- self$q_lo$gather(-1L, seg_idx)
      q_hi_g <- self$q_hi$gather(-1L, seg_idx)
      alpha_lo_g <- self$alpha_lo$gather(-1L, seg_idx)
      alpha_hi_g <- self$alpha_hi$gather(-1L, seg_idx)

      t <- (z - q_lo_g) / (q_hi_g - q_lo_g)$clamp(min = self$tol)
      result <- alpha_lo_g + t$clamp(0.0, 1.0) * (alpha_hi_g - alpha_lo_g)

      q_r_exp <- self$.expand_to_z(self$q_r, z)
      result <- torch_where(
        z >= q_r_exp,
        torch_tensor(self$alpha_r, device = z$device, dtype = z$dtype),
        result
      )

      if (added_dim) {
        result <- result$squeeze(-1L)
      }
      result
    },

    #' @description Compute dQ/dα derivative
    #' @param alpha Tensor. Probability levels.
    .icdf_derivative = function(alpha) {
      cfg <- self$cfg

      # Expand alpha for batch dimensions if needed (same logic as icdf)
      alpha_shape <- alpha$shape
      if (alpha$dim() == 1 && length(self$batch_shape) > 0) {
        alpha_expanded <- alpha
        for (i in seq_along(self$batch_shape)) {
          alpha_expanded <- alpha_expanded$unsqueeze(1L)
        }
        expand_shape <- c(self$batch_shape, alpha_shape)
        alpha_expanded <- alpha_expanded$expand(expand_shape)
      } else {
        alpha_expanded <- alpha
      }

      alpha_l_t <- torch_tensor(self$alpha_l, device = alpha$device, dtype = alpha$dtype)
      alpha_r_t <- torch_tensor(self$alpha_r, device = alpha$device, dtype = alpha$dtype)

      if (alpha_expanded$dim() > length(self$batch_shape)) {
        alpha_l_t <- self$.expand_to_alpha(alpha_l_t$expand(self$batch_shape), alpha_expanded)
        alpha_r_t <- self$.expand_to_alpha(alpha_r_t$expand(self$batch_shape), alpha_expanded)
      }

      deriv_left <- self$.deriv_left_tail(alpha_expanded)
      deriv_right <- self$.deriv_right_tail(alpha_expanded)
      deriv_spline <- self$.deriv_spline(alpha_expanded)

      deriv <- torch_where(
        alpha_expanded < alpha_l_t,
        deriv_left,
        torch_where(alpha_expanded > alpha_r_t, deriv_right, deriv_spline)
      )

      torch_clamp(deriv, min = cfg$MIN_SLOPE, max = cfg$MAX_SLOPE)
    },

    #' @description Left tail derivative
    #' @param alpha Tensor. Probability levels.
    .deriv_left_tail = function(alpha) {
      if (self$tail_type == "exp") {
        a <- self$.expand_to_alpha(self$tail_a_l, alpha)
        alpha_safe <- torch_clamp(alpha, min = self$tol)
        return(a / alpha_safe)
      } else {
        cfg <- self$cfg
        eta <- self$.expand_to_alpha(self$eta_l, alpha)
        mu <- self$.expand_to_alpha(self$mu_l, alpha)

        is_exp <- torch_abs(eta) < cfg$ETA_TOLERANCE
        eta_safe <- torch_where(is_exp, torch_full_like(eta, cfg$ETA_TOLERANCE), eta)

        alpha_safe <- torch_clamp(alpha, min = self$tol)
        ratio <- self$alpha_l / alpha_safe
        log_ratio <- torch_clamp(torch_log(ratio), max = cfg$MAX_LOG_RATIO)
        ratio_pow <- torch_exp(torch_clamp(eta_safe * log_ratio, max = cfg$MAX_EXPONENT))

        gpd_deriv <- mu * ratio_pow / alpha_safe
        exp_deriv <- mu / alpha_safe
        return(torch_where(is_exp, exp_deriv, gpd_deriv))
      }
    },

    #' @description Right tail derivative
    #' @param alpha Tensor. Probability levels.
    .deriv_right_tail = function(alpha) {
      if (self$tail_type == "exp") {
        a <- self$.expand_to_alpha(self$tail_a_r, alpha)
        one_m <- torch_clamp(1 - alpha, min = self$tol)
        return((-a) / one_m)
      } else {
        cfg <- self$cfg
        eta <- self$.expand_to_alpha(self$eta_r, alpha)
        mu <- self$.expand_to_alpha(self$mu_r, alpha)

        is_exp <- torch_abs(eta) < cfg$ETA_TOLERANCE
        eta_safe <- torch_where(is_exp, torch_full_like(eta, cfg$ETA_TOLERANCE), eta)

        one_m <- torch_clamp(1 - alpha, min = self$tol)
        ratio <- (1 - self$alpha_r) / one_m
        log_ratio <- torch_clamp(torch_log(ratio), max = cfg$MAX_LOG_RATIO)
        ratio_pow <- torch_exp(torch_clamp(eta_safe * log_ratio, max = cfg$MAX_EXPONENT))

        gpd_deriv <- mu * ratio_pow / one_m
        exp_deriv <- mu / one_m
        return(torch_where(is_exp, exp_deriv, gpd_deriv))
      }
    },

    #' @description Spline derivative
    #' @param alpha Tensor. Probability levels.
    .deriv_spline = function(alpha) {
      added_dim <- alpha$dim() == length(self$batch_shape)
      if (added_dim) {
        alpha <- alpha$unsqueeze(-1L)
      }

      seg_idx <- (
        torch_searchsorted(
          self$alpha_lo_1d$contiguous(),
          alpha$contiguous(),
          right = TRUE
        ) - 1L
      )
      seg_idx <- seg_idx$clamp(1L, self$num_segments)

      result <- self$slopes$gather(-1L, seg_idx)

      # Return 1.0 when alpha is below all segments
      no_segment <- alpha < self$alpha_levels[1]
      result <- torch_where(no_segment, torch_ones_like(result), result)

      if (added_dim) {
        result <- result$squeeze(-1L)
      }
      result
    },

    #' @description Compute log PDF
    #' @param z Tensor. Values at which to evaluate log density.
    #' @return Tensor. Log probability density values.
    log_prob = function(z) {
      # Handle 1D input
      if (z$dim() == 1 && length(self$batch_shape) > 0) {
        if (!identical(z$shape, self$batch_shape)) {
          z <- z$unsqueeze(1L)$expand(c(self$batch_shape, -1L))
        }
      }

      alpha <- self$cdf(z)
      q_deriv <- self$.icdf_derivative(alpha)
      -torch_log(q_deriv)
    },

    #' @description Compute PDF
    #' @param z Tensor. Values at which to evaluate PDF.
    #' @return Tensor. PDF values (non-negative).
    pdf = function(z) {
      torch_exp(self$log_prob(z))
    },

    #' @description Compute mean E\[Z\]
    #' @return Tensor. Expected value. Shape: `(*batch_shape,)`.
    mean = function() {
      if (self$tail_type == "exp") {
        self$.mean_exp_analytical()
      } else {
        self$.mean_gpd_analytical()
      }
    },

    #' @description Analytical mean for exponential tails
    .mean_exp_analytical = function() {
      # Left tail
      left_int <- self$alpha_l * (self$q_l - self$tail_a_l)

      # Spline: trapezoid rule
      spline_int <- (self$delta_alpha * (self$q_lo + self$q_hi) / 2)$sum(dim = -1L)

      # Right tail
      right_int <- (1 - self$alpha_r) * (self$q_r - self$tail_a_r)

      left_int + spline_int + right_int
    },

    #' @description Analytical mean for GPD tails
    .mean_gpd_analytical = function() {
      cfg <- self$cfg

      eta_l_safe <- torch_clamp(self$eta_l, min = cfg$ETA_TOLERANCE, max = 1 - cfg$ETA_TOLERANCE)
      eta_r_safe <- torch_clamp(self$eta_r, min = cfg$ETA_TOLERANCE, max = 1 - cfg$ETA_TOLERANCE)

      # Left
      left_int <- self$alpha_l * (self$q_l - self$mu_l / (1 - eta_l_safe))

      # Spline
      spline_int <- (self$delta_alpha * (self$q_lo + self$q_hi) / 2)$sum(dim = -1L)

      # Right
      right_int <- (1 - self$alpha_r) * (self$q_r + self$mu_r / (1 - eta_r_safe))

      left_int + spline_int + right_int
    },

    #' @description Compute variance
    #' @return Tensor. Variance (non-negative). Shape: `(*batch_shape,)`.
    variance = function() {
      if (self$tail_type == "exp") {
        self$.variance_exp_analytical()
      } else {
        self$.variance_gpd_analytical()
      }
    },

    #' @description Analytical variance for exponential tails
    .variance_exp_analytical = function() {
      a_l <- self$tail_a_l
      a_r <- self$tail_a_r

      # E[Z²] left
      e_z2_left <- self$alpha_l * (self$q_l^2 - 2 * a_l * self$q_l + 2 * a_l^2)

      # E[Z²] spline
      e_z2_spline <- (self$delta_alpha * (self$q_lo^2 + self$q_lo * self$q_hi + self$q_hi^2) / 3)$sum(dim = -1L)

      # E[Z²] right
      e_z2_right <- (1 - self$alpha_r) * (self$q_r^2 - 2 * a_r * self$q_r + 2 * a_r^2)

      e_z2 <- e_z2_left + e_z2_spline + e_z2_right
      e_z <- self$.mean_exp_analytical()

      torch_clamp(e_z2 - e_z^2, min = 0.0)
    },

    #' @description Analytical variance for GPD tails
    .variance_gpd_analytical = function() {
      cfg <- self$cfg

      eta_l_safe <- torch_clamp(self$eta_l, min = cfg$ETA_TOLERANCE, max = 0.49)
      eta_r_safe <- torch_clamp(self$eta_r, min = cfg$ETA_TOLERANCE, max = 0.49)

      # Left tail E[Z²]
      c_l <- self$q_l + self$mu_l / eta_l_safe
      d_l <- self$mu_l / eta_l_safe
      one_m_eta_l <- 1 - eta_l_safe
      one_m_2eta_l <- torch_clamp(1 - 2 * eta_l_safe, min = cfg$ETA_TOLERANCE)
      e_z2_left <- self$alpha_l * (c_l^2 - 2 * c_l * d_l / one_m_eta_l + d_l^2 / one_m_2eta_l)

      # Spline E[Z²]
      e_z2_spline <- (self$delta_alpha * (self$q_lo^2 + self$q_lo * self$q_hi + self$q_hi^2) / 3)$sum(dim = -1L)

      # Right tail E[Z²]
      c_r <- self$q_r - self$mu_r / eta_r_safe
      d_r <- self$mu_r / eta_r_safe
      one_m_eta_r <- 1 - eta_r_safe
      one_m_2eta_r <- torch_clamp(1 - 2 * eta_r_safe, min = cfg$ETA_TOLERANCE)
      e_z2_right <- (1 - self$alpha_r) * (c_r^2 + 2 * c_r * d_r / one_m_eta_r + d_r^2 / one_m_2eta_r)

      e_z2 <- e_z2_left + e_z2_spline + e_z2_right
      e_z <- self$.mean_gpd_analytical()

      torch_clamp(e_z2 - e_z^2, min = 0.0)
    },

    #' @description Compute standard deviation
    #' @return Tensor. Standard deviation. Shape: `(*batch_shape,)`.
    stddev = function() {
      torch_sqrt(torch_clamp(self$variance(), min = self$tol))
    },

    #' @description Compute analytical CRPS (Continuous Ranked Probability Score)
    #' @param z Tensor. Observation values. Shape: `(*batch_shape,)` or `(*batch_shape, ...)`.
    #' @return Tensor. CRPS values (non-negative, lower is better). Shape: same as z.
    crps = function(z) {
      cfg <- self$cfg

      alpha_z <- self$cdf(z)

      crps_left <- self$.crps_left_tail(z, alpha_z)
      crps_spline <- self$.crps_spline(z, alpha_z)
      crps_right <- self$.crps_right_tail(z, alpha_z)

      torch_clamp(crps_left + crps_spline + crps_right, min = 0.0, max = cfg$MAX_CRPS)$view(z$shape)
    },

    #' @description CRPS contribution from left tail
    #' @param z Tensor. Observation values. Shape: `(*batch_shape,)` or `(*batch_shape, ...)`.
    #' @param alpha_z Tensor. Probability levels.
    .crps_left_tail = function(z, alpha_z) {
      if (self$tail_type == "exp") {
        self$.crps_left_tail_exp(z, alpha_z)
      } else {
        self$.crps_left_tail_gpd(z, alpha_z)
      }
    },

    #' @description Exponential left tail CRPS
    #' @param z Tensor. Observation values. Shape: `(*batch_shape,)` or `(*batch_shape, ...)`.
    #' @param alpha_z Tensor. Probability levels.
    .crps_left_tail_exp = function(z, alpha_z) {
      a <- self$.expand_to_z(self$tail_a_l, z)
      b <- self$.expand_to_z(self$tail_b_l, z)
      q_L <- self$.expand_to_z(self$q_l, z)
      alpha_L <- self$alpha_l

      alpha_L_safe <- max(alpha_L, self$tol)
      alpha_tilde <- torch_clamp(alpha_z, min = self$tol, max = alpha_L_safe)

      ln_alpha_L <- torch_log(torch_tensor(alpha_L_safe, device = z$device, dtype = z$dtype))

      term1 <- (z - b) * (alpha_L_safe^2 - 2 * alpha_L_safe + 2 * alpha_tilde)
      term2 <- alpha_L_safe^2 * a * (-ln_alpha_L + 0.5)
      term2 <- term2 + 2 * torch_where(
        z < q_L,
        alpha_L_safe * a * (ln_alpha_L - 1) + alpha_tilde * (-z + b + a),
        torch_zeros_like(z)
      )

      term1 + term2
    },

    #' @description GPD left tail CRPS
    #' @param z Tensor. Observation values. Shape: `(*batch_shape,)` or `(*batch_shape, ...)`.
    #' @param alpha_z Tensor. Probability levels.
    .crps_left_tail_gpd = function(z, alpha_z) {
      cfg <- self$cfg

      eta <- self$.expand_to_z(self$eta_l, z)
      mu <- self$.expand_to_z(self$mu_l, z)
      q_L <- self$.expand_to_z(self$q_l, z)
      alpha_L <- self$alpha_l

      is_exp <- torch_abs(eta) < cfg$ETA_TOLERANCE
      eta_safe <- torch_where(is_exp, torch_full_like(eta, cfg$ETA_TOLERANCE), eta)

      two_m_eta <- torch_clamp((2.0 - eta_safe)$abs(), min = cfg$ETA_TOLERANCE)
      two_m_eta <- torch_where(2.0 - eta_safe >= 0, two_m_eta, -two_m_eta)
      mu_over_2me <- torch_clamp(mu / two_m_eta, min = -cfg$MAX_CRPS, max = cfg$MAX_CRPS)

      alpha_sq <- alpha_L^2
      simple_crps <- alpha_sq * (z - q_L + mu_over_2me)

      alpha_tilde <- torch_clamp(alpha_z, min = self$tol, max = alpha_L - self$tol)
      psi <- torch_clamp((q_L - z) / torch_clamp(mu, min = self$tol), min = 0.0, max = cfg$MAX_EXPONENT)
      TT <- torch_clamp(1.0 + eta_safe * psi, min = self$tol)
      exp_power <- torch_clamp(
        (1.0 - 2.0 / eta_safe) * torch_log(TT),
        min = -cfg$MAX_EXPONENT,
        max = cfg$MAX_EXPONENT
      )
      T_power <- torch_exp(exp_power)

      I2 <- alpha_sq * mu_over_2me * T_power
      I1 <- 2.0 * (q_L - z) * (alpha_L - alpha_tilde) + alpha_sq * mu_over_2me * (1.0 - T_power)
      gpd_crps <- I1 + I2

      in_tail <- z < q_L
      torch_where(in_tail, gpd_crps, simple_crps)
    },

    #' @description CRPS contribution from right tail
    #' @param z Tensor. Observation values. Shape: `(*batch_shape,)` or `(*batch_shape, ...)`.
    #' @param alpha_z Tensor. Probability levels.
    .crps_right_tail = function(z, alpha_z) {
      if (self$tail_type == "exp") {
        self$.crps_right_tail_exp(z, alpha_z)
      } else {
        self$.crps_right_tail_gpd(z, alpha_z)
      }
    },

    #' @description Exponential right tail CRPS
    #' @param z Tensor. Observation values. Shape: `(*batch_shape,)` or `(*batch_shape, ...)`.
    #' @param alpha_z Tensor. Probability levels.
    .crps_right_tail_exp = function(z, alpha_z) {
      a <- self$.expand_to_z(self$tail_a_r, z)
      b <- self$.expand_to_z(self$tail_b_r, z)
      q_R <- self$.expand_to_z(self$q_r, z)
      alpha_R <- self$alpha_r

      alpha_R_safe <- min(alpha_R, 1 - self$tol)
      one_m_R <- max(1 - alpha_R_safe, self$tol)
      alpha_tilde <- torch_clamp(alpha_z, min = alpha_R_safe, max = 1 - self$tol)

      ln_one_m_R <- torch_log(torch_tensor(one_m_R, device = z$device, dtype = z$dtype))

      term1 <- (z - b) * (-1 - alpha_R_safe^2 + 2 * alpha_tilde)
      term2 <- a * (-0.5 * (alpha_R_safe + 1)^2 + (alpha_R_safe^2 - 1) * ln_one_m_R + 2 * alpha_tilde)
      term2 <- term2 + 2 * torch_where(
        z > q_R,
        (1 - alpha_tilde) * (z - b),
        a * one_m_R * ln_one_m_R
      )

      term1 + term2
    },

    #' @description GPD right tail CRPS
    #' @param z Tensor. Observation values. Shape: `(*batch_shape,)` or `(*batch_shape, ...)`.
    #' @param alpha_z Tensor. Probability levels.
    .crps_right_tail_gpd = function(z, alpha_z) {
      cfg <- self$cfg

      eta <- self$.expand_to_z(self$eta_r, z)
      mu <- self$.expand_to_z(self$mu_r, z)
      q_R <- self$.expand_to_z(self$q_r, z)
      alpha_R <- self$alpha_r

      is_exp <- torch_abs(eta) < cfg$ETA_TOLERANCE
      eta_safe <- torch_where(is_exp, torch_full_like(eta, cfg$ETA_TOLERANCE), eta)

      two_m_eta <- torch_clamp((2.0 - eta_safe)$abs(), min = cfg$ETA_TOLERANCE)
      two_m_eta <- torch_where(2.0 - eta_safe >= 0, two_m_eta, -two_m_eta)
      mu_over_2me <- torch_clamp(mu / two_m_eta, min = -cfg$MAX_CRPS, max = cfg$MAX_CRPS)

      one_m_R <- max(1 - alpha_R, self$tol)
      one_m_sq <- one_m_R^2

      simple_crps <- one_m_sq * (q_R - z + mu_over_2me)

      alpha_tilde <- torch_clamp(alpha_z, min = alpha_R + self$tol, max = 1 - self$tol)
      psi <- torch_clamp((z - q_R) / torch_clamp(mu, min = self$tol), min = 0.0, max = cfg$MAX_EXPONENT)
      TT <- torch_clamp(1.0 + eta_safe * psi, min = self$tol)
      exp_power <- torch_clamp(
        (1.0 - 2.0 / eta_safe) * torch_log(TT),
        min = -cfg$MAX_EXPONENT,
        max = cfg$MAX_EXPONENT
      )
      T_power <- torch_exp(exp_power)

      I2 <- one_m_sq * mu_over_2me * T_power
      I1 <- 2.0 * (z - q_R) * (alpha_tilde - alpha_R) - one_m_sq * mu_over_2me * (1.0 - T_power)
      gpd_crps <- I1 + I2

      in_tail <- z > q_R
      torch_where(in_tail, gpd_crps, simple_crps)
    },

    #' @description CRPS contribution from spline region
    #' @param z Tensor. Observation values. Shape: `(*batch_shape,)` or `(*batch_shape, ...)`.
    #' @param alpha_z Tensor. Probability levels.
    .crps_spline = function(z, alpha_z) {
      z_shape <- z$shape
      if (z$ndim < alpha_z$ndim) {
        n_missing <- alpha_z$ndim - z$ndim
        z <- z$view(c(rep(1L, n_missing), z_shape))
      }
      n_extra <- z$dim() - length(self$batch_shape)
      seg_dim <- length(self$batch_shape) + 1L

      # Expand segment data
      alpha_i <- self$alpha_lo
      alpha_ip1 <- self$alpha_hi
      q_i <- self$q_lo
      m <- self$slopes

      lapply(seq_len(n_extra), function(i) {
        alpha_i <<- alpha_i$unsqueeze(-1L)
        alpha_ip1 <<- alpha_ip1$unsqueeze(-1L)
        q_i <<- q_i$unsqueeze(-1L)
        m <<- m$unsqueeze(-1L)
      })

      z_exp <- z$unsqueeze(seg_dim)
      alpha_z_exp <- alpha_z$unsqueeze(seg_dim)

      # Manually expand to compatible shapes for broadcasting
      # alpha_z_exp: (*batch, 1, *obs) -> (*batch, n_seg, *obs)
      # alpha_i: (*batch, n_seg, 1) -> (*batch, n_seg, *obs)
      # Compute target shape from alpha_z_exp shape, replacing the segment dimension
      alpha_z_shape <- as.list(alpha_z_exp$shape)
      alpha_z_shape[[seg_dim]] <- self$num_segments
      target_shape <- unlist(alpha_z_shape)

      alpha_z_broadcast <- alpha_z_exp$expand(target_shape)
      alpha_i_broadcast <- alpha_i$expand(target_shape)
      alpha_ip1_broadcast <- alpha_ip1$expand(target_shape)

      # Unified formula via clamp trick - compute mask and use it to select values
      lower_mask <- (alpha_z_broadcast < alpha_i_broadcast)$to(dtype = alpha_z_exp$dtype)
      upper_mask <- (alpha_z_broadcast > alpha_ip1_broadcast)$to(dtype = alpha_z_exp$dtype)
      middle_mask <- 1.0 - lower_mask - upper_mask

      r <- lower_mask * alpha_i_broadcast + middle_mask * alpha_z_broadcast + upper_mask * alpha_ip1_broadcast

      r2 <- r^2
      r3 <- r^3
      ai2 <- alpha_i^2
      ai3 <- alpha_i^3
      aip12 <- alpha_ip1^2
      aip13 <- alpha_ip1^3

      # I1 = ∫_{α_i}^r 2α(z-Q)dα
      I1 <- (z_exp - q_i) * (r2 - ai2) - 2 * m * (r3 / 3 - alpha_i * r2 / 2 + ai3 / 6)

      # I2 = ∫_r^{α_{i+1}} 2(1-α)(Q-z)dα
      A <- q_i - z_exp
      diff_a <- alpha_ip1 - r
      diff_a2 <- aip12 - r2
      diff_a3 <- aip13 - r3

      int_Qmz <- A * diff_a + m * (diff_a2 / 2 - alpha_i * diff_a)
      int_aQmz <- A * diff_a2 / 2 + m * (diff_a3 / 3 - alpha_i * diff_a2 / 2)

      I2 <- 2 * int_Qmz - 2 * int_aQmz

      seg_crps <- I1 + I2
      seg_crps$sum(dim = seg_dim)
    },

    #' @description Numerical CRPS via pinball loss (for validation)
    #' @param z Tensor. Observation values. Shape: `(*batch_shape,)`.
    #' @param num_quantiles Integer. Number of quantile levels for integration.
    #' @return Tensor. Approximate CRPS values. Shape: `(*batch_shape,)`.
    pinball = function(z, num_quantiles = 999L) {
      device <- z$device
      dtype <- z$dtype
      alphas <- torch_linspace(0.0, 1.0, num_quantiles + 2L, device = device, dtype = dtype)[2:(num_quantiles + 1L)]
      pred_q <- self$icdf(alphas)
      z_exp <- z$unsqueeze(-1L)
      diff <- z_exp - pred_q
      loss <- torch_where(diff >= 0, alphas * diff, (alphas - 1) * diff)
      2 * loss$mean(dim = -1L)
    },

    #' @description Draw samples from the distribution
    #' @param sample_shape torch.Size. Shape of the sample.
    #' @return Tensor. Samples from the distribution.
    sample = function(sample_shape = NULL) {
      if (is.null(sample_shape)) {
        n_samples <- 1L
      } else {
        n_samples <- max(1L, prod(sample_shape))
      }

      # Generate uniforms and apply inverse CDF
      u <- torch_rand(
        c(self$batch_shape, n_samples),
        device = self$quantiles$device,
        dtype = self$quantiles$dtype
      )
      q <- self$icdf(u)

      if (is.null(sample_shape)) {
        # For single sample, extract the single element
        if (q$dim() == 2 && all(q$shape == c(1, 1))) {
          return(q[1, 1])
        } else if (q$dim() == 1 && q$shape[1] == 1) {
          return(q[1])
        } else {
          return(q$squeeze())
        }
      }

      # Reshape
      n_batch <- length(self$batch_shape)
      q <- q$view(c(self$batch_shape, sample_shape))
      perm <- c(
        seq(from = n_batch + 1, length.out = length(sample_shape)),
        seq_len(n_batch)
      )
      q$permute(perm)
    }
  )
)

#' Module wrapper for QuantileDistribution
#'
#' @description
#' Converts predicted quantiles to QuantileDistribution with analytical statistics.
#'
#' @param alpha_levels Numeric vector or NULL. Quantile levels.
#' @param num_quantiles Integer. Number of quantile levels (if alpha_levels not provided).
#' @param tail_type Character. "exp" or "gpd".
#' @param fix_crossing Logical. Enforce monotonicity?
#' @param crossing_method Character. Method for fixing crossing.
#'
#' @importFrom torch nn_module
#' @export
quantile_to_distribution <- nn_module(
  "QuantileToDistribution",
  initialize = function(alpha_levels = NULL,
                       num_quantiles = 999L,
                       tail_type = "exp",
                       fix_crossing = TRUE,
                       crossing_method = "sort") {
    self$tail_type <- tail_type
    self$fix_crossing <- fix_crossing
    self$crossing_method <- crossing_method

    # Register alpha levels as buffer
    if (is.null(alpha_levels)) {
      self$alpha_levels <- torch_linspace(0.0, 1.0, num_quantiles + 2L)[2:(num_quantiles + 1L)]
    } else {
      self$alpha_levels <- torch_tensor(alpha_levels)
    }
  },

  forward = function(quantiles) {
    QuantileDistribution$new(
      quantiles = quantiles,
      alpha_levels = self$alpha_levels$to(device = quantiles$device, dtype = quantiles$dtype),
      tail_type = self$tail_type,
      fix_crossing = self$fix_crossing,
      crossing_method = self$crossing_method
    )
  }
)
