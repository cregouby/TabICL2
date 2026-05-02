#' @importFrom R6 R6Class
#' @importFrom recipes recipe prep bake add_step step rand_id recipes_eval_select is_trained sel2char print_step check_type step_zv step_normalize
#' @importFrom recipes step_YeoJohnson all_predictors all_numeric_predictors all_nominal_predictors
#' @importFrom recipes step_impute_mean step_integer step_novel step_unknown step_string2factor
#' @importFrom rlang enquos

.to_num_matrix <- function(X) {
  if (is.data.frame(X)) X <- as.matrix(X)
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  X
}

.col_nanstd <- function(X, ddof = 1L) {
  apply(X, 2L, function(col) {
    x <- col[!is.na(col)]
    n <- length(x)
    if (n <= ddof) return(0)
    sqrt(sum((x - mean(x))^2) / (n - ddof))
  })
}

# Shared bake logic for quantile-to-normal steps
.bake_quantile_normal <- function(object, new_data) {
  col_names <- colnames(object$quantiles)
  BOUNDS    <- 1e-7
  for (j in seq_along(col_names)) {
    col <- pmax(pmin(new_data[[col_names[j]]],
                     max(object$quantiles[, j])),
                min(object$quantiles[, j]))
    cdf <- approx(x = object$quantiles[, j], y = object$references,
                  xout = col, method = "linear", rule = 2, ties = "ordered")$y
    new_data[[col_names[j]]] <- qnorm(pmax(pmin(cdf, 1 - BOUNDS), BOUNDS))
  }
  new_data
}

# Latin-square helpers for Shuffler
.rls <- function(symbols) {
  n <- length(symbols)
  if (n == 1L) return(list(symbols))
  idx    <- sample.int(n, 1L)
  sym    <- symbols[[idx]]
  square <- .rls(symbols[-idx])
  square[[n]] <- square[[1L]]
  for (i in seq_len(n)) square[[i]] <- append(square[[i]], sym, after = i - 1L)
  square
}

.shuffle_transpose_square <- function(square) {
  n     <- length(square)
  square <- square[sample.int(n)]
  trans  <- lapply(seq_len(n), function(i) sapply(square, `[[`, i))
  trans[sample.int(n)]
}

.permutations <- function(n) {
  if (n == 1L) return(matrix(1L, 1L, 1L))
  prev <- .permutations(n - 1L)
  m    <- nrow(prev)
  out  <- matrix(0L, m * n, n)
  for (i in seq_len(n)) {
    rows        <- ((i - 1L) * m + 1L):(i * m)
    out[rows, 1L] <- i
    out[rows, -1L] <- prev + (prev >= i)
  }
  out
}

#' Normalize and clip numeric predictors
#'
#' Computes z = (x - mean) / (sd + epsilon) and clips to \[`clip_min`, `clip_max`\].
#'
#' @param recipe A recipe object.
#' @param ... Tidyselect column selectors.
#' @param clip_min Lower clip bound (default -100).
#' @param clip_max Upper clip bound (default 100).
#' @param epsilon Denominator stabiliser (default 1e-6).
#' @param na_rm Remove NAs when computing statistics.
#' @param role Not used.
#' @param trained Logical; set by `prep()`.
#' @param means Named numeric vector of means (set by `prep()`).
#' @param sds Named numeric vector of standard deviations (set by `prep()`).
#' @param skip Skip during `bake()`.
#' @param id Step identifier.
#' @return Updated recipe.
#' @export
step_normalize_clip <- function(recipe, ...,
                                clip_min = -100, clip_max = 100,
                                epsilon = 1e-6, na_rm = TRUE,
                                role = NA, trained = FALSE,
                                means = NULL, sds = NULL,
                                skip = FALSE, id = rand_id("normalize_clip")) {
  add_step(recipe, step_normalize_clip_new(
    terms = enquos(...), clip_min = clip_min, clip_max = clip_max,
    epsilon = epsilon, na_rm = na_rm, role = role, trained = trained,
    means = means, sds = sds, skip = skip, id = id
  ))
}

step_normalize_clip_new <- function(terms, clip_min, clip_max, epsilon, na_rm,
                                    role, trained, means, sds, skip, id) {
  step("normalize_clip", terms = terms, clip_min = clip_min, clip_max = clip_max,
       epsilon = epsilon, na_rm = na_rm, role = role, trained = trained,
       means = means, sds = sds, skip = skip, id = id)
}

#' @export
prep.step_normalize_clip <- function(x, training, info = NULL, ...) {
  col_names <- recipes_eval_select(x$terms, training, info)
  check_type(training[, col_names, drop = FALSE], quant = TRUE)
  means <- vapply(training[, col_names, drop = FALSE],
                  mean, na.rm = x$na_rm, FUN.VALUE = numeric(1L))
  sds   <- vapply(training[, col_names, drop = FALSE],
                  sd,   na.rm = x$na_rm, FUN.VALUE = numeric(1L))
  step_normalize_clip_new(
    terms = x$terms, clip_min = x$clip_min, clip_max = x$clip_max,
    epsilon = x$epsilon, na_rm = x$na_rm, role = x$role, trained = TRUE,
    means = setNames(means, col_names), sds = setNames(sds, col_names),
    skip = x$skip, id = x$id
  )
}

#' @export
bake.step_normalize_clip <- function(object, new_data, ...) {
  col_names <- names(object$means)
  for (nm in col_names) {
    z <- (new_data[[nm]] - object$means[[nm]]) / (object$sds[[nm]] + object$epsilon)
    new_data[[nm]] <- pmax(pmin(z, object$clip_max), object$clip_min)
  }
  new_data
}

#' @export
print.step_normalize_clip <- function(x, width = max(20, options()$width - 30), ...) {
  title <- glue::glue("Normalize+clip [e={x$epsilon}, {x$clip_min},{x$clip_max}] ")
  print_step(x$terms, x$terms, x$trained, title, width)
  invisible(x)
}

#' @export
tidy.step_normalize_clip <- function(x, ...) {
  if (is_trained(x)) {
    tibble::tibble(
      terms     = rep(names(x$means), 2L),
      statistic = c(rep("mean", length(x$means)), rep("sd", length(x$sds))),
      value     = c(unname(x$means), unname(x$sds)),
      id        = x$id
    )
  } else {
    tibble::tibble(terms = sel2char(x$terms), statistic = NA_character_,
                   value = NA_real_, id = x$id)
  }
}

#' Clip outliers using a two-stage Z-score approach
#'
#' Values beyond `threshold` standard deviations are first zeroed for
#' re-estimation; final clipping uses log-based soft bounds.
#'
#' @param recipe A recipe object.
#' @param ... Tidyselect column selectors.
#' @param threshold Z-score threshold (default 4.0).
#' @param role Not used.
#' @param trained Logical; set by `prep()`.
#' @param lower_bounds Named vector of lower bounds (set by `prep()`).
#' @param upper_bounds Named vector of upper bounds (set by `prep()`).
#' @param skip Skip during `bake()`.
#' @param id Step identifier.
#' @return Updated recipe.
#' @export
step_clip_outliers <- function(recipe, ...,
                               threshold = 4.0,
                               role = NA, trained = FALSE,
                               lower_bounds = NULL, upper_bounds = NULL,
                               skip = FALSE, id = rand_id("clip_outliers")) {
  add_step(recipe, step_clip_outliers_new(
    terms = enquos(...), threshold = threshold, role = role, trained = trained,
    lower_bounds = lower_bounds, upper_bounds = upper_bounds, skip = skip, id = id
  ))
}

step_clip_outliers_new <- function(terms, threshold, role, trained,
                                   lower_bounds, upper_bounds, skip, id) {
  step("clip_outliers", terms = terms, threshold = threshold, role = role,
       trained = trained, lower_bounds = lower_bounds, upper_bounds = upper_bounds,
       skip = skip, id = id)
}

#' @export
prep.step_clip_outliers <- function(x, training, info = NULL, ...) {
  col_names <- recipes_eval_select(x$terms, training, info)
  check_type(training[, col_names, drop = FALSE], quant = TRUE)
  X    <- as.matrix(training[, col_names, drop = FALSE])
  ddof <- if (nrow(X) > 1L) 1L else 0L
  thr  <- x$threshold
  # Stage 1: initial bounds
  m1 <- colMeans(X, na.rm = TRUE)
  s1 <- pmax(.col_nanstd(X, ddof), 1e-6)
  Xc <- X
  Xc[sweep(X, 2L, m1 - thr * s1, "<") | sweep(X, 2L, m1 + thr * s1, ">")] <- NA_real_
  # Stage 2: recompute on clean data
  m2 <- colMeans(Xc, na.rm = TRUE)
  s2 <- pmax(.col_nanstd(Xc, ddof), 1e-6)
  step_clip_outliers_new(
    terms = x$terms, threshold = thr, role = x$role, trained = TRUE,
    lower_bounds = setNames(m2 - thr * s2, col_names),
    upper_bounds = setNames(m2 + thr * s2, col_names),
    skip = x$skip, id = x$id
  )
}

#' @export
bake.step_clip_outliers <- function(object, new_data, ...) {
  col_names <- names(object$lower_bounds)
  for (nm in col_names) {
    x <- new_data[[nm]]
    x <- pmax(-log1p(abs(x)) + object$lower_bounds[[nm]], x)
    x <- pmin( log1p(abs(x)) + object$upper_bounds[[nm]], x)
    new_data[[nm]] <- x
  }
  new_data
}

#' @export
print.step_clip_outliers <- function(x, width = max(20, options()$width - 30), ...) {
  title <- glue::glue("Clip outliers (threshold = {x$threshold}) for ")
  print_step(x$terms, x$terms, x$trained, title, width)
  invisible(x)
}

#' @export
tidy.step_clip_outliers <- function(x, ...) {
  if (is_trained(x)) {
    tibble::tibble(
      terms       = rep(names(x$lower_bounds), 2L),
      statistic   = c(rep("lower_bound", length(x$lower_bounds)),
                      rep("upper_bound", length(x$upper_bounds))),
      value       = c(unname(x$lower_bounds), unname(x$upper_bounds)),
      id          = x$id
    )
  } else {
    tibble::tibble(terms = sel2char(x$terms), statistic = NA_character_,
                   value = NA_real_, id = x$id)
  }
}

#' Robust scaling using median and IQR
#'
#' Centres by median and scales by IQR / (qnorm(0.75) - qnorm(0.25)) when
#' `unit_variance = TRUE`, matching sklearn's `RobustScaler(unit_variance=True)`.
#'
#' @param recipe A recipe object.
#' @param ... Tidyselect column selectors.
#' @param unit_variance Scale by IQR / normal-IQR so variance ≈ 1 for Gaussian data.
#' @param role Not used.
#' @param trained Logical; set by `prep()`.
#' @param centers Named vector of medians (set by `prep()`).
#' @param scales Named vector of scale factors (set by `prep()`).
#' @param skip Skip during `bake()`.
#' @param id Step identifier.
#' @return Updated recipe.
#' @export
step_robust_scale <- function(recipe, ...,
                              unit_variance = TRUE,
                              role = NA, trained = FALSE,
                              centers = NULL, scales = NULL,
                              skip = FALSE, id = rand_id("robust_scale")) {
  add_step(recipe, step_robust_scale_new(
    terms = enquos(...), unit_variance = unit_variance, role = role,
    trained = trained, centers = centers, scales = scales, skip = skip, id = id
  ))
}

step_robust_scale_new <- function(terms, unit_variance, role, trained,
                                  centers, scales, skip, id) {
  step("robust_scale", terms = terms, unit_variance = unit_variance, role = role,
       trained = trained, centers = centers, scales = scales, skip = skip, id = id)
}

#' @export
prep.step_robust_scale <- function(x, training, info = NULL, ...) {
  col_names <- recipes_eval_select(x$terms, training, info)
  check_type(training[, col_names, drop = FALSE], quant = TRUE)
  X    <- as.matrix(training[, col_names, drop = FALSE])
  ctr  <- apply(X, 2L, median, na.rm = TRUE)
  iqrs <- apply(X, 2L, IQR,    na.rm = TRUE)
  sf   <- if (x$unit_variance) qnorm(0.75) - qnorm(0.25) else 1
  scl  <- pmax(iqrs / sf, 1e-10)
  step_robust_scale_new(
    terms = x$terms, unit_variance = x$unit_variance, role = x$role, trained = TRUE,
    centers = setNames(ctr, col_names), scales = setNames(scl, col_names),
    skip = x$skip, id = x$id
  )
}

#' @export
bake.step_robust_scale <- function(object, new_data, ...) {
  col_names <- names(object$centers)
  for (nm in col_names)
    new_data[[nm]] <- (new_data[[nm]] - object$centers[[nm]]) / object$scales[[nm]]
  new_data
}

#' @export
print.step_robust_scale <- function(x, width = max(20, options()$width - 30), ...) {
  print_step(x$terms, x$terms, x$trained, "Robust scaling for ", width)
  invisible(x)
}

#' @export
tidy.step_robust_scale <- function(x, ...) {
  if (is_trained(x)) {
    tibble::tibble(
      terms     = rep(names(x$centers), 2L),
      statistic = c(rep("center", length(x$centers)), rep("scale", length(x$scales))),
      value     = c(unname(x$centers), unname(x$scales)),
      id        = x$id
    )
  } else {
    tibble::tibble(terms = sel2char(x$terms), statistic = NA_character_,
                   value = NA_real_, id = x$id)
  }
}

#' Map numeric predictors to a standard normal distribution via quantiles
#'
#' Estimates `n_quantiles` equally-spaced quantiles from training data and uses
#' linear interpolation + `qnorm()` to normalise new data.
#'
#' @param recipe A recipe object.
#' @param ... Tidyselect column selectors.
#' @param n_quantiles Maximum number of quantiles (default 1000).
#' @param role Not used.
#' @param trained Logical; set by `prep()`.
#' @param quantiles Per-feature quantile matrix (set by `prep()`).
#' @param references Probability levels vector (set by `prep()`).
#' @param skip Skip during `bake()`.
#' @param id Step identifier.
#' @return Updated recipe.
#' @export
step_quantile_normal <- function(recipe, ...,
                                 n_quantiles = 1000L,
                                 role = NA, trained = FALSE,
                                 quantiles = NULL, references = NULL,
                                 skip = FALSE, id = rand_id("quantile_normal")) {
  add_step(recipe, step_quantile_normal_new(
    terms = enquos(...), n_quantiles = as.integer(n_quantiles), role = role,
    trained = trained, quantiles = quantiles, references = references,
    skip = skip, id = id
  ))
}

step_quantile_normal_new <- function(terms, n_quantiles, role, trained,
                                     quantiles, references, skip, id) {
  step("quantile_normal", terms = terms, n_quantiles = n_quantiles, role = role,
       trained = trained, quantiles = quantiles, references = references,
       skip = skip, id = id)
}

#' @export
prep.step_quantile_normal <- function(x, training, info = NULL, ...) {
  col_names <- recipes_eval_select(x$terms, training, info)
  check_type(training[, col_names, drop = FALSE], quant = TRUE)
  X    <- as.matrix(training[, col_names, drop = FALSE])
  n_q  <- min(nrow(X), x$n_quantiles)
  refs <- seq(0, 1, length.out = n_q)
  quants <- apply(X, 2L, function(col)
    quantile(col, probs = refs, na.rm = TRUE, type = 7L))
  colnames(quants) <- col_names
  step_quantile_normal_new(
    terms = x$terms, n_quantiles = x$n_quantiles, role = x$role, trained = TRUE,
    quantiles = quants, references = refs, skip = x$skip, id = x$id
  )
}

#' @export
bake.step_quantile_normal <- function(object, new_data, ...)
  .bake_quantile_normal(object, new_data)

#' @export
print.step_quantile_normal <- function(x, width = max(20, options()$width - 30), ...) {
  print_step(x$terms, x$terms, x$trained, "Quantile-to-normal transform for ", width)
  invisible(x)
}

#' @export
tidy.step_quantile_normal <- function(x, ...) {
  if (is_trained(x)) {
    nf  <- ncol(x$quantiles)
    nq  <- length(x$references)
    tibble::tibble(
      terms          = rep(colnames(x$quantiles), each = nq),
      quantile_level = rep(x$references, nf),
      value          = as.vector(x$quantiles),
      id             = x$id
    )
  } else {
    tibble::tibble(terms = sel2char(x$terms), quantile_level = NA_real_,
                   value = NA_real_, id = x$id)
  }
}

#' RTDL-style quantile-to-normal transform with training noise
#'
#' Adds proportional Gaussian noise before fitting quantiles and dynamically
#' sets `n_quantiles = max(min(n / 30, n_quantiles), 10)`.
#'
#' @param recipe A recipe object.
#' @param ... Tidyselect column selectors.
#' @param noise Noise magnitude relative to feature sd (default 1e-3).
#' @param n_quantiles Maximum quantile count (default 1000).
#' @param random_state Integer seed for noise generation.
#' @param role Not used.
#' @param trained Logical; set by `prep()`.
#' @param quantiles Per-feature quantile matrix (set by `prep()`).
#' @param references Probability levels vector (set by `prep()`).
#' @param skip Skip during `bake()`.
#' @param id Step identifier.
#' @return Updated recipe.
#' @export
step_quantile_normal_rtdl <- function(recipe, ...,
                                      noise = 1e-3, n_quantiles = 1000L,
                                      random_state = NULL,
                                      role = NA, trained = FALSE,
                                      quantiles = NULL, references = NULL,
                                      skip = FALSE,
                                      id = rand_id("quantile_normal_rtdl")) {
  add_step(recipe, step_quantile_normal_rtdl_new(
    terms = enquos(...), noise = noise, n_quantiles = as.integer(n_quantiles),
    random_state = random_state, role = role, trained = trained,
    quantiles = quantiles, references = references, skip = skip, id = id
  ))
}

step_quantile_normal_rtdl_new <- function(terms, noise, n_quantiles, random_state,
                                          role, trained, quantiles, references, skip, id) {
  step("quantile_normal_rtdl", terms = terms, noise = noise,
       n_quantiles = n_quantiles, random_state = random_state, role = role,
       trained = trained, quantiles = quantiles, references = references,
       skip = skip, id = id)
}

#' @export
prep.step_quantile_normal_rtdl <- function(x, training, info = NULL, ...) {
  col_names <- recipes_eval_select(x$terms, training, info)
  check_type(training[, col_names, drop = FALSE], quant = TRUE)
  X   <- as.matrix(training[, col_names, drop = FALSE])
  n_q <- max(min(nrow(X) %/% 30L, x$n_quantiles), 10L)
  if (x$noise > 0) {
    if (!is.null(x$random_state)) set.seed(x$random_state)
    stds      <- apply(X, 2L, sd, na.rm = TRUE)
    noise_std <- x$noise / pmax(stds, x$noise)
    X <- X + matrix(rnorm(length(X), sd = rep(noise_std, each = nrow(X))),
                    nrow = nrow(X))
  }
  refs   <- seq(0, 1, length.out = n_q)
  quants <- apply(X, 2L, function(col)
    quantile(col, probs = refs, na.rm = TRUE, type = 7L))
  colnames(quants) <- col_names
  step_quantile_normal_rtdl_new(
    terms = x$terms, noise = x$noise, n_quantiles = x$n_quantiles,
    random_state = x$random_state, role = x$role, trained = TRUE,
    quantiles = quants, references = refs, skip = x$skip, id = x$id
  )
}

#' @export
bake.step_quantile_normal_rtdl <- function(object, new_data, ...)
  .bake_quantile_normal(object, new_data)

#' @export
print.step_quantile_normal_rtdl <- function(x, width = max(20, options()$width - 30), ...) {
  print_step(x$terms, x$terms, x$trained, "RTDL quantile-to-normal for ", width)
  invisible(x)
}

#' @export
tidy.step_quantile_normal_rtdl <- function(x, ...) {
  if (is_trained(x)) {
    nf  <- ncol(x$quantiles)
    nq  <- length(x$references)
    tibble::tibble(
      terms          = rep(colnames(x$quantiles), each = nq),
      quantile_level = rep(x$references, nf),
      value          = as.vector(x$quantiles),
      id             = x$id
    )
  } else {
    tibble::tibble(terms = sel2char(x$terms), quantile_level = NA_real_,
                   value = NA_real_, id = x$id)
  }
}

.named_df <- function(X, col_names) {
  df <- as.data.frame(X)
  colnames(df) <- col_names
  df
}

#' Transform non-numerical data to a numeric matrix
#'
#' Detects and converts categorical, boolean, and text columns to ordinal
#' integers (unknown/NA → -1) and imputes numeric columns with column means.
#'
#' @param verbose Whether to print column classification information.
#' @export
TransformToNumerical <- R6Class(
  "TransformToNumerical",
  public = list(
    #' @field verbose Print column classification info.
    verbose = FALSE,

    #' @description Create a new `TransformToNumerical`.
    #' @param verbose Logical.
    initialize = function(verbose = FALSE) {
      self$verbose <- verbose
    },

    #' @description Fit to training data.
    #' @param X Matrix or data.frame.
    #' @param y Ignored.
    #' @return Invisibly `self`.
    fit = function(X, y = NULL) {
      private$.is_df <- is.data.frame(X)
      if (!private$.is_df) {
        Xm <- .to_num_matrix(X)
        private$.is_categorical <- !is.numeric(X) && !is.matrix(X)
        if (private$.is_categorical) {
          vals <- as.character(Xm)
          private$.levels <- list(sort(unique(vals[!is.na(vals)])))
        }
        private$.col_means <- colMeans(Xm, na.rm = TRUE)
        return(invisible(self))
      }
      cat_cols <- names(Filter(
        function(col) inherits(col, c("character", "factor", "logical")), X))
      num_cols <- names(Filter(is.numeric, X))
      private$.cat_cols   <- cat_cols
      private$.num_cols   <- num_cols
      private$.levels     <- lapply(setNames(cat_cols, cat_cols), function(nm) {
        vals <- as.character(X[[nm]])
        sort(unique(vals[!is.na(vals)]))
      })
      num_mat <- as.matrix(X[num_cols])
      storage.mode(num_mat) <- "double"
      private$.col_means <- colMeans(num_mat, na.rm = TRUE)
      if (self$verbose) {
        cli_inform("Columns classified as categorical: {cat_cols}")
        cli_inform("Columns classified as continuous: {num_cols}")
        dropped <- setdiff(names(X), c(cat_cols, num_cols))
        if (length(dropped) > 0L)
          cli_inform("Columns not used due to data type: {dropped}")
      }
      invisible(self)
    },

    #' @description Transform data to numeric matrix.
    #' @param X Matrix or data.frame.
    #' @return Numeric matrix.
    transform = function(X) {
      if (is.null(private$.is_df))
        value_error("Transformer has not been fitted. Call $fit() first.")
      if (!private$.is_df) {
        Xm <- as.matrix(X)
        if (private$.is_categorical) {
          vals    <- as.character(Xm)
          encoded <- match(vals, private$.levels[[1L]]) - 1L
          encoded[is.na(encoded)] <- -1L
          return(matrix(encoded, nrow = nrow(Xm)))
        }
        Xm <- .to_num_matrix(Xm)
        Xm[is.na(Xm)] <- 0
        return(Xm)
      }
      cat_mat <- matrix(0L, nrow(X), length(private$.cat_cols))
      for (i in seq_along(private$.cat_cols)) {
        nm  <- private$.cat_cols[i]
        enc <- match(as.character(X[[nm]]), private$.levels[[nm]]) - 1L
        enc[is.na(enc)] <- -1L
        cat_mat[, i] <- enc
      }
      num_mat <- as.matrix(X[private$.num_cols])
      storage.mode(num_mat) <- "double"
      for (j in seq_along(private$.num_cols)) {
        na_idx <- is.na(num_mat[, j])
        num_mat[na_idx, j] <- private$.col_means[j]
      }
      cbind(cat_mat, num_mat)
    },

    #' @description Fit then transform.
    #' @param X Matrix or data.frame.
    #' @param y Ignored.
    #' @return Numeric matrix.
    fit_transform = function(X, y = NULL) { self$fit(X, y); self$transform(X) }
  ),
  private = list(
    .is_df = NULL, .is_categorical = FALSE,
    .cat_cols = NULL, .num_cols = NULL,
    .levels = NULL, .col_means = NULL
  )
)

#' UniqueFeatureFilter: remove features with fewer than `threshold + 1` unique values
#'
#' Wraps `recipes::step_zv()` internally.  For `threshold = 1` (default) this
#' matches sklearn's behaviour exactly.
#'
#' @param threshold Features with unique values `<= threshold` are removed.
#'   When `n_samples <= threshold` all features are kept.
#' @export
UniqueFeatureFilter <- R6Class(
  "UniqueFeatureFilter",
  public = list(
    #' @field threshold Uniqueness threshold.
    threshold = 1L,
    #' @field features_to_keep_ Logical mask of retained features (set after `$fit()`).
    features_to_keep_ = NULL,
    #' @field n_features_in_ Input feature count (set after `$fit()`).
    n_features_in_ = NULL,
    #' @field n_features_out_ Output feature count (set after `$fit()`).
    n_features_out_ = NULL,

    #' @description Create a new `UniqueFeatureFilter`.
    #' @param threshold Integer uniqueness threshold.
    initialize = function(threshold = 1L) {
      self$threshold <- as.integer(threshold)
      },

    #' @description Fit: identify zero-variance features.
    #' @param X Numeric matrix.
    #' @param y Ignored.
    #' @return Invisibly `self`.
    fit = function(X, y = NULL) {
      X  <- .to_num_matrix(X)
      cn <- private$.make_colnames(ncol(X))
      colnames(X) <- cn
      # step_zv only removes single-value features (threshold=1 case)
      # for threshold>1 we fall back to manual uniqueness counting
      if (self$threshold == 1L && nrow(X) > self$threshold) {
        df  <- .named_df(X, cn)
        rec <- recipe(~ ., data = df) %>% step_zv(all_predictors())
        private$.recipe <- prep(rec, training = df, verbose = FALSE)
        removed <- private$.recipe$steps[[1L]]$removals
      } else if (nrow(X) <= self$threshold) {
        removed <- character(0L)
      } else {
        removed <- cn[apply(X, 2L, function(col) length(unique(col)) <= self$threshold)]
      }
      self$n_features_in_    <- ncol(X)
      self$features_to_keep_ <- !cn %in% removed
      self$n_features_out_   <- sum(self$features_to_keep_)
      invisible(self)
    },

    #' @description Drop low-variance features.
    #' @param X Numeric matrix.
    #' @return Filtered numeric matrix.
    transform = function(X) {
      if (is.null(self$features_to_keep_))
        value_error("Filter has not been fitted. Call $fit() first.")
      X <- .to_num_matrix(X)
      if (ncol(X) != self$n_features_in_)
        value_error("Expected {self$n_features_in_} features but got {ncol(X)}.")
      X[, self$features_to_keep_, drop = FALSE]
    },

    #' @description Fit then transform.
    #' @param X Numeric matrix.
    #' @param y Ignored.
    #' @return Filtered numeric matrix.
    fit_transform = function(X, y = NULL) { self$fit(X, y); self$transform(X) }
  ),
  private = list(
    .recipe = NULL,
    .make_colnames = function(p) paste0("X", seq_len(p))
  )
)

#' OutlierRemover: clip extreme values using a two-stage Z-score approach
#'
#' Wraps `step_clip_outliers()` in a recipes pipeline.
#'
#' @param threshold Z-score threshold for outlier detection.
#' @export
OutlierRemover <- R6Class(
  "OutlierRemover",
  public = list(
    #' @field threshold Z-score threshold.
    threshold = 4.0,
    #' @field lower_bounds_ Per-feature lower clip bounds (set after `$fit()`).
    lower_bounds_ = NULL,
    #' @field upper_bounds_ Per-feature upper clip bounds (set after `$fit()`).
    upper_bounds_ = NULL,

    #' @description Create a new `OutlierRemover`.
    #' @param threshold Numeric Z-score threshold.
    initialize = function(threshold = 4.0) {
      self$threshold <- threshold
    },

    #' @description Fit: compute clipping bounds.
    #' @param X Numeric matrix.
    #' @param y Ignored.
    #' @return Invisibly `self`.
    fit = function(X, y = NULL) {
      X  <- .to_num_matrix(X)
      cn <- paste0("X", seq_len(ncol(X)))
      df <- .named_df(X, cn)
      rec <- recipe(~ ., data = df) %>%
        step_clip_outliers(all_predictors(), threshold = self$threshold)
      private$.recipe    <- prep(rec, training = df, verbose = FALSE)
      stp                <- private$.recipe$steps[[1L]]
      self$lower_bounds_ <- stp$lower_bounds
      self$upper_bounds_ <- stp$upper_bounds
      private$.col_names <- cn
      invisible(self)
    },

    #' @description Clip values using learned log-based soft bounds.
    #' @param X Numeric matrix.
    #' @return Clipped numeric matrix.
    transform = function(X) {
      if (is.null(private$.recipe))
        value_error("OutlierRemover has not been fitted. Call $fit() first.")
      X  <- .to_num_matrix(X)
      df <- .named_df(X, private$.col_names)
      as.matrix(bake(private$.recipe, new_data = df))
    },

    #' @description Fit then transform.
    #' @param X Numeric matrix.
    #' @param y Ignored.
    #' @return Clipped numeric matrix.
    fit_transform = function(X, y = NULL) { self$fit(X, y); self$transform(X) }
  ),
  private = list(.recipe = NULL, .col_names = NULL)
)

#' CustomStandardScaler: z-score standardisation with output clipping
#'
#' Wraps `step_normalize_clip()`: z = (x - mean) / (sd + epsilon), clipped.
#'
#' @param clip_min Lower bound for clipped output.
#' @param clip_max Upper bound for clipped output.
#' @param epsilon Small constant added to sd to avoid division by zero.
#' @export
CustomStandardScaler <- R6Class(
  "CustomStandardScaler",
  public = list(
    #' @field clip_min Lower clip bound.
    clip_min = -100,
    #' @field clip_max Upper clip bound.
    clip_max = 100,
    #' @field epsilon Denominator stabiliser.
    epsilon = 1e-6,
    #' @field mean_ Per-feature means (set after `$fit()`).
    mean_ = NULL,
    #' @field scale_ Per-feature sds (set after `$fit()`).
    scale_ = NULL,

    #' @description Create a new `CustomStandardScaler`.
    #' @param clip_min Numeric lower clip bound.
    #' @param clip_max Numeric upper clip bound.
    #' @param epsilon Numeric denominator stabiliser.
    initialize = function(clip_min = -100, clip_max = 100, epsilon = 1e-6) {
      self$clip_min <- clip_min; self$clip_max <- clip_max; self$epsilon <- epsilon
    },

    #' @description Fit: compute mean and sd.
    #' @param X Numeric matrix or vector.
    #' @param y Ignored.
    #' @return Invisibly `self`.
    fit = function(X, y = NULL) {
      is_vec <- is.vector(X)
      if (is_vec) X <- matrix(X, ncol = 1L)
      X  <- .to_num_matrix(X)
      cn <- paste0("X", seq_len(ncol(X)))
      df <- .named_df(X, cn)
      rec <- recipe(~ ., data = df) %>%
        step_normalize_clip(all_predictors(),
                            clip_min = self$clip_min, clip_max = self$clip_max,
                            epsilon  = self$epsilon)
      private$.recipe <- prep(rec, training = df, verbose = FALSE)
      stp             <- private$.recipe$steps[[1L]]
      self$mean_      <- stp$means
      self$scale_     <- stp$sds
      private$.col_names <- cn
      invisible(self)
    },

    #' @description Standardise and clip.
    #' @param X Numeric matrix or vector.
    #' @return Scaled, clipped output (same shape as input).
    transform = function(X) {
      if (is.null(private$.recipe))
        value_error("Scaler has not been fitted. Call $fit() first.")
      is_vec <- is.vector(X)
      if (is_vec) X <- matrix(X, ncol = 1L)
      X  <- .to_num_matrix(X)
      df <- .named_df(X, private$.col_names)
      out <- as.matrix(bake(private$.recipe, new_data = df))
      if (is_vec) as.numeric(out) else out
    },

    #' @description Inverse transform: recover original scale.
    #' @param X Numeric matrix or vector.
    #' @return Data in original scale.
    inverse_transform = function(X) {
      if (is.null(self$mean_))
        value_error("Scaler has not been fitted. Call $fit() first.")
      is_vec <- is.vector(X)
      if (is_vec) X <- matrix(X, ncol = 1L)
      X   <- .to_num_matrix(X)
      out <- sweep(sweep(X, 2L, self$scale_ + self$epsilon, "*"), 2L, self$mean_, "+")
      if (is_vec) as.numeric(out) else out
    },

    #' @description Fit then transform.
    #' @param X Numeric matrix or vector.
    #' @param y Ignored.
    #' @return Scaled, clipped output.
    fit_transform = function(X, y = NULL) { self$fit(X, y); self$transform(X) }
  ),
  private = list(.recipe = NULL, .col_names = NULL)
)

#' RTDLQuantileTransformer: quantile transformer adapted for tabular deep learning
#'
#' Wraps `step_quantile_normal_rtdl()` in a recipe.
#'
#' @param noise Noise magnitude relative to feature standard deviations.
#' @param n_quantiles Maximum number of quantiles.
#' @param output_distribution Target distribution (`"normal"` only).
#' @param random_state Integer seed or `NULL`.
#' @export
RTDLQuantileTransformer <- R6Class(
  "RTDLQuantileTransformer",
  public = list(
    #' @field noise Noise magnitude.
    noise = 1e-3,
    #' @field n_quantiles Maximum quantile count.
    n_quantiles = 1000L,
    #' @field output_distribution Target distribution.
    output_distribution = "normal",
    #' @field random_state RNG seed.
    random_state = NULL,

    #' @description Create a new `RTDLQuantileTransformer`.
    #' @param noise Numeric noise level.
    #' @param n_quantiles Integer maximum quantile count.
    #' @param output_distribution `"normal"` (uniform not yet supported).
    #' @param random_state Integer seed or `NULL`.
    initialize = function(noise = 1e-3, n_quantiles = 1000L,
                          output_distribution = "normal", random_state = NULL) {
      self$noise               <- noise
      self$n_quantiles         <- as.integer(n_quantiles)
      self$output_distribution <- output_distribution
      self$random_state        <- random_state
    },

    #' @description Fit: compute quantiles from (optionally noised) training data.
    #' @param X Numeric matrix.
    #' @param y Ignored.
    #' @return Invisibly `self`.
    fit = function(X, y = NULL) {
      X  <- .to_num_matrix(X)
      cn <- paste0("X", seq_len(ncol(X)))
      df <- .named_df(X, cn)
      rec <- recipe(~ ., data = df) %>%
        step_quantile_normal_rtdl(all_predictors(),
                                  noise        = self$noise,
                                  n_quantiles  = self$n_quantiles,
                                  random_state = self$random_state)
      private$.recipe    <- prep(rec, training = df, verbose = FALSE)
      private$.col_names <- cn
      invisible(self)
    },

    #' @description Transform using the fitted quantile recipe.
    #' @param X Numeric matrix.
    #' @return Transformed matrix.
    transform = function(X) {
      if (is.null(private$.recipe))
        value_error("Transformer has not been fitted. Call $fit() first.")
      X  <- .to_num_matrix(X)
      df <- .named_df(X, private$.col_names)
      as.matrix(bake(private$.recipe, new_data = df))
    },

    #' @description Fit then transform.
    #' @param X Numeric matrix.
    #' @param y Ignored.
    #' @return Transformed matrix.
    fit_transform = function(X, y = NULL) { self$fit(X, y); self$transform(X) }
  ),
  private = list(.recipe = NULL, .col_names = NULL)
)

#' PreprocessingPipeline: normalisation and outlier clipping for tabular data
#'
#' Builds a `recipes` pipeline combining:
#' 1. `step_normalize_clip` (z-score + clip)
#' 2. A normalisation step (`step_YeoJohnson`, `step_quantile_normal`,
#'    `step_quantile_normal_rtdl` + `step_normalize`, `step_robust_scale`,
#'    or nothing for `"none"`)
#' 3. `step_clip_outliers`
#'
#' @param normalization_method One of `"power"`, `"quantile"`,
#'   `"quantile_rtdl"`, `"robust"`, or `"none"`.
#' @param outlier_threshold Z-score threshold for outlier detection.
#' @param random_state Integer seed or `NULL`.
#' @export
PreprocessingPipeline <- R6Class(
  "PreprocessingPipeline",
  public = list(
    #' @field normalization_method Normalisation method.
    normalization_method = "power",
    #' @field outlier_threshold Outlier Z-score threshold.
    outlier_threshold = 4.0,
    #' @field random_state RNG seed.
    random_state = NULL,
    #' @field X_transformed_ Transformed training data (set after `$fit()`).
    X_transformed_ = NULL,

    #' @description Create a new `PreprocessingPipeline`.
    #' @param normalization_method Character normalisation method.
    #' @param outlier_threshold Numeric outlier threshold.
    #' @param random_state Integer seed or `NULL`.
    initialize = function(normalization_method = "power",
                          outlier_threshold = 4.0,
                          random_state = NULL) {
      self$normalization_method <- normalization_method
      self$outlier_threshold    <- outlier_threshold
      self$random_state         <- random_state
    },

    #' @description Fit the pipeline.
    #' @param X Numeric matrix.
    #' @param y Ignored.
    #' @return Invisibly `self`.
    fit = function(X, y = NULL) {
      X  <- .to_num_matrix(X)
      cn <- paste0("X", seq_len(ncol(X)))
      colnames(X) <- cn
      df <- as.data.frame(X)
      rec <- recipe(~ ., data = df) %>%
        step_normalize_clip(all_predictors())
      rec <- private$.add_norm_step(rec)
      rec <- rec %>%
        step_clip_outliers(all_predictors(), threshold = self$outlier_threshold)
      private$.recipe    <- prep(rec, training = df, verbose = FALSE)
      private$.col_names <- cn
      self$X_transformed_ <- as.matrix(bake(private$.recipe, new_data = NULL))
      colnames(self$X_transformed_) <- NULL
      invisible(self)
    },

    #' @description Apply the pipeline.
    #' @param X Numeric matrix.
    #' @return Preprocessed matrix.
    transform = function(X) {
      if (is.null(private$.recipe))
        value_error("Pipeline has not been fitted. Call $fit() first.")
      X  <- .to_num_matrix(X)
      df <- .named_df(X, private$.col_names)
      out <- as.matrix(bake(private$.recipe, new_data = df))
      colnames(out) <- NULL
      out
    },

    #' @description Fit then transform.
    #' @param X Numeric matrix.
    #' @param y Ignored.
    #' @return Preprocessed matrix.
    fit_transform = function(X, y = NULL) { self$fit(X, y); self$transform(X) }
  ),
  private = list(
    .recipe = NULL, .col_names = NULL,
    .add_norm_step = function(rec) {
      switch(
        self$normalization_method,
        "none"          = rec,
        "power"         = rec %>% step_YeoJohnson(all_predictors()),
        "quantile"      = rec %>% step_quantile_normal(all_predictors()),
        "quantile_rtdl" = rec %>%
          step_quantile_normal_rtdl(all_predictors(), random_state = self$random_state) %>%
          step_normalize(all_predictors()),
        "robust"        = rec %>% step_robust_scale(all_predictors()),
        value_error("Unknown normalization_method: {self$normalization_method}")
      )
    }
  )
)

#' Shuffler: generate permutations for ensemble diversity
#'
#' @param n_elements Number of elements to permute.
#' @param method One of `"latin"`, `"shift"`, `"random"`, or `"none"`.
#' @param max_elements_for_latin Maximum `n_elements` before falling back to
#'   `"random"` (Latin square recursion depth scales with `n_elements`).
#' @param random_state Integer seed or `NULL`.
#' @export
Shuffler <- R6Class(
  "Shuffler",
  public = list(
    #' @field n_elements Number of elements.
    n_elements = NULL,
    #' @field method Shuffling method.
    method = "latin",
    #' @field max_elements_for_latin Latin square size limit.
    max_elements_for_latin = 4000L,
    #' @field random_state RNG seed.
    random_state = NULL,

    #' @description Create a new `Shuffler`.
    #' @param n_elements Integer number of elements.
    #' @param method Shuffling strategy.
    #' @param max_elements_for_latin Integer size limit for Latin squares.
    #' @param random_state Integer seed or `NULL`.
    initialize = function(n_elements, method = "latin",
                          max_elements_for_latin = 4000L, random_state = NULL) {
      self$n_elements             <- as.integer(n_elements)
      self$method                 <- method
      self$max_elements_for_latin <- as.integer(max_elements_for_latin)
      self$random_state           <- random_state
    },

    #' @description Generate `n_estimators` shuffling patterns.
    #' @param n_estimators Number of permutations to generate.
    #' @return A list of integer permutation vectors (1-indexed).
    shuffle = function(n_estimators) {
      if (!is.null(self$random_state)) set.seed(self$random_state)
      indices <- seq_len(self$n_elements)
      method  <- if (self$n_elements > self$max_elements_for_latin &&
                     self$method == "latin") "random" else self$method
      if (method == "none" || n_estimators == 1L) return(list(indices))
      if (method == "shift") {
        lapply(seq_len(self$n_elements), function(i)
          c(tail(indices, i), head(indices, self$n_elements - i)))
      } else if (method == "random") {
        if (self$n_elements <= 5L) {
          all_p <- .permutations(self$n_elements)
          lapply(sample.int(nrow(all_p), min(n_estimators, nrow(all_p))),
                 function(i) as.integer(all_p[i, ]))
        } else {
          lapply(seq_len(n_estimators), function(...) sample(indices))
        }
      } else if (method == "latin") {
        shuffled <- .shuffle_transpose_square(.rls(as.list(indices)))
        lapply(shuffled, as.integer)
      } else {
        value_error("Unknown method '{method}'. Use 'shift', 'random', 'latin', or 'none'.")
      }
    }
  )
)

#' EnsembleGenerator: generate ensemble variants for robust tabular prediction
#'
#' Creates diverse dataset variants through (1) multiple normalisation methods,
#' (2) feature-order permutations, and (3) for classification, class-label
#' shuffles.
#'
#' @param classification Logical; `TRUE` for classification tasks.
#' @param n_estimators Number of ensemble variants.
#' @param norm_methods Character vector of normalisation methods, or `NULL` for
#'   `c("none", "power")`.
#' @param feat_shuffle_method Feature permutation strategy.
#' @param class_shuffle_method Class permutation strategy (classification only).
#' @param outlier_threshold Z-score threshold for outlier detection.
#' @param random_state Integer seed or `NULL`.
#' @export
EnsembleGenerator <- R6Class(
  "EnsembleGenerator",
  public = list(
    #' @field classification Classification task flag.
    classification = NULL,
    #' @field n_estimators Number of ensemble variants.
    n_estimators = NULL,
    #' @field norm_methods Normalisation methods.
    norm_methods = NULL,
    #' @field feat_shuffle_method Feature shuffle strategy.
    feat_shuffle_method = "latin",
    #' @field class_shuffle_method Class shuffle strategy.
    class_shuffle_method = "shift",
    #' @field outlier_threshold Outlier Z-score threshold.
    outlier_threshold = 4.0,
    #' @field random_state RNG seed.
    random_state = NULL,
    #' @field n_features_in_ Post-filter feature count (set after `$fit()`).
    n_features_in_ = NULL,
    #' @field n_classes_ Class count for classification (set after `$fit()`).
    n_classes_ = NULL,
    #' @field unique_filter_ Fitted `UniqueFeatureFilter` (set after `$fit()`).
    unique_filter_ = NULL,
    #' @field preprocessors_ Named list of fitted `PreprocessingPipeline`s (set after `$fit()`).
    preprocessors_ = NULL,
    #' @field ensemble_configs_ Named list of shuffle configs (set after `$fit()`).
    ensemble_configs_ = NULL,
    #' @field feature_shuffles_ Named list of feature permutations (set after `$fit()`).
    feature_shuffles_ = NULL,
    #' @field class_shuffles_ Named list of class permutations (set after `$fit()`).
    class_shuffles_ = NULL,
    #' @field X_ Training features after unique-feature filtering (set after `$fit()`).
    X_ = NULL,
    #' @field y_ Training targets (set after `$fit()`).
    y_ = NULL,

    #' @description Create a new `EnsembleGenerator`.
    #' @param classification Logical.
    #' @param n_estimators Integer.
    #' @param norm_methods Character vector or `NULL`.
    #' @param feat_shuffle_method Character.
    #' @param class_shuffle_method Character.
    #' @param outlier_threshold Numeric.
    #' @param random_state Integer or `NULL`.
    initialize = function(classification, n_estimators,
                          norm_methods = NULL,
                          feat_shuffle_method = "latin",
                          class_shuffle_method = "shift",
                          outlier_threshold = 4.0,
                          random_state = NULL) {
      if (!class_shuffle_method %in% c("none", "shift", "random", "latin"))
        value_error("Invalid class_shuffle_method: '{class_shuffle_method}'.")
      self$classification       <- classification
      self$n_estimators         <- as.integer(n_estimators)
      self$norm_methods         <- norm_methods
      self$feat_shuffle_method  <- feat_shuffle_method
      self$class_shuffle_method <- class_shuffle_method
      self$outlier_threshold    <- outlier_threshold
      self$random_state         <- random_state
    },

    #' @description Fit: filter features, generate configs, fit preprocessors.
    #' @param X Numeric matrix of training features.
    #' @param y Numeric or integer target vector.
    #' @return Invisibly `self`.
    fit = function(X, y) {
      X <- .to_num_matrix(X)
      y <- as.numeric(y)
      if (length(y) != nrow(X))
        value_error("X has {nrow(X)} rows but y has {length(y)} elements.")
      if (!is.null(self$random_state)) set.seed(self$random_state)
      private$.norm_methods <- if (is.null(self$norm_methods)) {
        c("none", "power")
      } else {
        as.character(self$norm_methods)
      }
      self$unique_filter_ <- UniqueFeatureFilter$new()
      X                   <- self$unique_filter_$fit_transform(X)
      self$X_             <- X
      self$y_             <- y
      self$n_features_in_ <- ncol(X)
      if (self$classification) self$n_classes_ <- length(unique(y))
      gen                    <- private$.generate_ensemble()
      self$ensemble_configs_ <- gen$ensemble_configs
      self$feature_shuffles_ <- gen$X_shuffle_dict
      if (self$classification) self$class_shuffles_ <- gen$y_pattern_dict
      self$preprocessors_ <- setNames(
        lapply(names(self$ensemble_configs_), function(method) {
          pp <- PreprocessingPipeline$new(
            normalization_method = method,
            outlier_threshold    = self$outlier_threshold,
            random_state         = self$random_state)
          pp$fit(X)
          pp
        }),
        names(self$ensemble_configs_)
      )
      invisible(self)
    },

    #' @description Create ensemble data variants.
    #'
    #' @param X Numeric matrix of test features, or `NULL` when `mode = "train"`.
    #' @param mode `"both"` (train + test), `"train"`, or `"test"`.
    #' @param feature_mask Logical vector over original features; masked columns
    #'   (TRUE) are zeroed and dropped from the output.
    #' @return Named list keyed by normalisation method.  Each element contains
    #'   `X` (3-D array `[variants, samples, features]`) and, unless
    #'   `mode = "test"`, `y` (matrix `[variants, n_train]`).
    transform = function(X = NULL, mode = "both", feature_mask = NULL) {
      if (is.null(self$ensemble_configs_))
        value_error("Generator has not been fitted. Call $fit() first.")
      if (!mode %in% c("both", "train", "test"))
        value_error("Invalid mode '{mode}'. Use 'both', 'train', or 'test'.")
      kept_cols <- NULL
      if (!is.null(feature_mask)) {
        filtered_mask <- feature_mask[self$unique_filter_$features_to_keep_]
        kept_cols     <- !filtered_mask
        idx_map       <- cumsum(kept_cols)
        idx_map[!kept_cols] <- NA_integer_
        private$.masked_shuffles <- setNames(
          lapply(names(self$ensemble_configs_), function(method) {
            lapply(self$ensemble_configs_[[method]], function(cfg) {
              sh <- cfg$feat_shuffle
              as.integer(idx_map[sh[kept_cols[sh]]])
            })
          }),
          names(self$ensemble_configs_)
        )
      }
      if (mode == "train") return(private$.make_train(kept_cols))
      if (is.null(X))
        value_error("X is required when mode is 'test' or 'both'.")
      X <- self$unique_filter_$transform(.to_num_matrix(X))
      if (!is.null(feature_mask)) X[, filtered_mask] <- 0
      if (mode == "test") return(private$.make_test(X, kept_cols))
      private$.make_both(X, kept_cols)
    }
  ),

  private = list(
    .norm_methods    = NULL,
    .masked_shuffles = NULL,

    .generate_ensemble = function() {
      feat_shuffler <- Shuffler$new(n_elements = self$n_features_in_,
                                    method = self$feat_shuffle_method,
                                    random_state = self$random_state)
      X_shuffles <- feat_shuffler$shuffle(self$n_estimators)
      y_patterns <- if (self$classification) {
        Shuffler$new(n_elements = self$n_classes_,
                     method = self$class_shuffle_method,
                     random_state = self$random_state)$shuffle(self$n_estimators)
      } else {
        list(NULL)
      }
      configs <- vector("list", length(X_shuffles) * length(y_patterns))
      k <- 1L
      for (xs in X_shuffles) for (yp in y_patterns) {
        configs[[k]] <- list(feat_shuffle = xs, y_pattern = yp); k <- k + 1L
      }
      configs <- sample(configs)
      combos  <- vector("list", length(configs) * length(private$.norm_methods))
      k <- 1L
      for (cfg in configs) for (method in private$.norm_methods) {
        combos[[k]] <- list(shuffle = cfg, method = method); k <- k + 1L
      }
      combos <- combos[seq_len(min(length(combos), self$n_estimators))]
      used <- unique(sapply(combos, `[[`, "method"))
      ens  <- setNames(vector("list", length(used)), used)
      xsd  <- setNames(vector("list", length(used)), used)
      ypd  <- setNames(vector("list", length(used)), used)
      for (method in used) {
        mc <- Filter(function(x) x$method == method, combos)
        ens[[method]] <- lapply(mc, `[[`, "shuffle")
        xsd[[method]] <- lapply(mc, function(x) x$shuffle$feat_shuffle)
        ypd[[method]] <- lapply(mc, function(x) x$shuffle$y_pattern)
      }
      list(ensemble_configs = ens, X_shuffle_dict = xsd, y_pattern_dict = ypd)
    },

    .get_feat_shuffle = function(method, i) {
      if (!is.null(private$.masked_shuffles))
        return(private$.masked_shuffles[[method]][[i]])
      self$ensemble_configs_[[method]][[i]]$feat_shuffle
    },

    .stack_X = function(mats) {
      arr <- array(0, dim = c(length(mats), nrow(mats[[1L]]), ncol(mats[[1L]])))
      for (i in seq_along(mats)) arr[i, , ] <- mats[[i]]
      arr
    },

    .apply_y_pattern = function(y, y_pattern) {
      if (self$classification && !is.null(y_pattern))
        y_pattern[as.integer(y) + 1L] - 1L
      else
        y
    },

    .make_train = function(kept_cols) {
      setNames(lapply(names(self$ensemble_configs_), function(method) {
        Xpp  <- self$preprocessors_[[method]]$X_transformed_
        if (!is.null(kept_cols)) Xpp <- Xpp[, kept_cols, drop = FALSE]
        cfgs  <- self$ensemble_configs_[[method]]
        Xmats <- lapply(seq_along(cfgs), function(i)
          Xpp[, private$.get_feat_shuffle(method, i), drop = FALSE])
        ymats <- lapply(cfgs, function(cfg) private$.apply_y_pattern(self$y_, cfg$y_pattern))
        list(X = private$.stack_X(Xmats), y = do.call(rbind, ymats))
      }), names(self$ensemble_configs_))
    },

    .make_test = function(Xtest, kept_cols) {
      setNames(lapply(names(self$ensemble_configs_), function(method) {
        Xpp  <- self$preprocessors_[[method]]$transform(Xtest)
        if (!is.null(kept_cols)) Xpp <- Xpp[, kept_cols, drop = FALSE]
        cfgs  <- self$ensemble_configs_[[method]]
        Xmats <- lapply(seq_along(cfgs), function(i)
          Xpp[, private$.get_feat_shuffle(method, i), drop = FALSE])
        list(X = private$.stack_X(Xmats))
      }), names(self$ensemble_configs_))
    },

    .make_both = function(Xtest, kept_cols) {
      setNames(lapply(names(self$ensemble_configs_), function(method) {
        pp     <- self$preprocessors_[[method]]
        Xtrain <- pp$X_transformed_
        Xte    <- pp$transform(Xtest)
        if (!is.null(kept_cols)) {
          Xtrain <- Xtrain[, kept_cols, drop = FALSE]
          Xte    <- Xte[,   kept_cols, drop = FALSE]
        }
        Xall  <- rbind(Xtrain, Xte)
        cfgs  <- self$ensemble_configs_[[method]]
        Xmats <- lapply(seq_along(cfgs), function(i)
          Xall[, private$.get_feat_shuffle(method, i), drop = FALSE])
        ymats <- lapply(cfgs, function(cfg) private$.apply_y_pattern(self$y_, cfg$y_pattern))
        list(X = private$.stack_X(Xmats), y = do.call(rbind, ymats))
      }), names(self$ensemble_configs_))
    }
  )
)
