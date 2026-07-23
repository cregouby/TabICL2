#' @importFrom torch nn_module nn_parameter
#' @importFrom torch torch_arange torch_linspace torch_ones torch_cat torch_stack
#' @importFrom torch torch_float torch_long torch_einsum
#' @importFrom torch torch_repeat_interleave
#' @keywords internal
NULL

#' Check if a value exists (is not NULL)
#'
#' @param val Value to check
#' @return Logical. `TRUE` if `val` is not `NULL`, `FALSE` otherwise.
#' @keywords internal
.exists <- function(val) {
  !is.null(val)
}

#' Return value or default fallback
#'
#' @param val Value to check
#' @param d Default value
#' @return `val` if not `NULL`, otherwise `d`.
#' @keywords internal
.default <- function(val, d) {
  if (.exists(val)) val else d
}

#' Interleaved rotation: pairs (1,2), (3,4), (5,6), etc. (1-based indexing)
#'
#' Given input `[.., d]`, rearranges to `[.., d/2, 2]` and rotates pairs.
#' Used by default in most RoPE implementations (e.g., LLaMA).
#'
#' @param x Input tensor of shape `[.., d]` where `d` is even.
#' @return Rotated tensor with same shape as `x`.
#' @keywords internal
rotate_half_interleaved <- function(x) {
  shape <- x$shape
  last_dim <- length(shape)
  d <- shape[last_dim]
  half <- d %/% 2L

  # Reshape to [.., half, 2]
  new_shape <- c(shape[seq_len(last_dim - 1L)], half, 2L)
  x_reshaped <- x$view(new_shape)

  # Unbind last dim (1-based: 1 and 2)
  x1 <- x_reshaped[.., 1L]
  x2 <- x_reshaped[.., 2L]

  # Stack (-x2, x1)
  rotated <- torch_stack(list(-x2, x1), dim = -1L)

  # Reshape back to [.., d]
  rotated$view(shape)
}

#' Contiguous rotation: split into first/second halves (1-based indexing)
#'
#' Given input `[.., d]`, splits into `[.., 1:(d/2)]` and `[.., (d/2+1):d]`
#' and rotates. Returns `[-x2, x1]`.
#'
#' @param x Input tensor of shape `[.., d]` where `d` is even.
#' @return Rotated tensor with same shape as `x`.
#' @keywords internal
rotate_half_contiguous <- function(x) {
  last_dim <- length(x$shape)
  d <- x$shape[last_dim]
  half <- d %/% 2L

  x1 <- x[.., seq_len(half)]
  x2 <- x[.., (half + 1L):d]

  torch_cat(list(-x2, x1), dim = -1L)
}

#' Apply rotary embeddings to tensor
#'
#' Computes \eqn{x \cdot \cos(\theta) + \text{rotate}(x) \cdot \sin(\theta)}
#' for the portion of the input that overlaps with the frequency tensor.
#'
#' @param freqs Frequency tensor for rotation.
#' @param t Input tensor to rotate.
#' @param start_index Starting index for rotation in the last dimension (default: `1L`).
#' @param scale Scaling factor for the rotation (default: `1.0`).
#' @param seq_dim Sequence dimension (default: `-2L`).
#' @param interleaved If `TRUE`, uses interleaved rotation (default: `TRUE`).
#' @return Rotated tensor, same shape as `t`.
#' @keywords internal
apply_rotary_emb <- function(freqs, t, start_index = 1L, scale = 1.0, seq_dim = -2L, interleaved = TRUE) {
  dtype <- t$dtype

  # Handle sequence truncation for 3D tensors
  if (t$ndim == 3L) {
    seq_len <- t$size(seq_dim)
    n_freqs <- freqs$size(1L)
    if (n_freqs > seq_len) {
      start_idx <- n_freqs - seq_len + 1L
      freqs <- freqs[start_idx:n_freqs, , drop = FALSE]
    }
  }

  if (interleaved) {
    rot_dim <- freqs$shape[length(freqs$shape)]
  } else {
    rot_dim <- freqs$shape[length(freqs$shape)] * 2L
  }

  # R: end_index is inclusive
  end_index <- start_index + rot_dim - 1L
  last_dim_size <- t$size(-1L)

  if (rot_dim > last_dim_size) {
    value_error("feature dimension {last_dim_size} is not of sufficient size to rotate in all the positions {rot_dim}")
  }

  # Split t into three parts
  t_left <- if (start_index > 1L) t[.., seq_len(start_index - 1L)] else NULL
  t_middle <- t[.., start_index:end_index]
  t_right <- if (end_index < last_dim_size) t[.., (end_index + 1L):last_dim_size] else NULL

  # Apply rotary embeddings
  if (interleaved) {
    t_transformed <- (t_middle * freqs$cos() * scale) +
      (rotate_half_interleaved(t_middle) * freqs$sin() * scale)
  } else {
    cos_freq <- torch_cat(list(freqs$cos(), freqs$cos()), dim = -1L) * scale
    sin_freq <- torch_cat(list(freqs$sin(), freqs$sin()), dim = -1L) * scale
    t_transformed <- (t_middle * cos_freq) +
      (rotate_half_contiguous(t_middle) * sin_freq)
  }

  # Concatenate parts
  parts <- list()
  if (!is.null(t_left)) parts <- c(parts, list(t_left))
  parts <- c(parts, list(t_transformed))
  if (!is.null(t_right)) parts <- c(parts, list(t_right))

  out <- torch_cat(parts, dim = -1L)
  out$to(dtype = dtype)
}

#' Rotary Positional Embeddings Module
#'
#' Rotary embeddings encode positional information in a way that allows
#' continuous rotation of embeddings, enhancing the model's ability to
#' capture long-range dependencies and positional relations.
#'
#' @param dim Integer. The dimension of the embeddings.
#' @param interleaved Logical. If `TRUE`, uses interleaved rotation (default: `TRUE`).
#' @param custom_freqs Tensor or `NULL`. Custom frequency tensor (default: `NULL`).
#' @param freqs_for Character. One of `"lang"`, `"pixel"`, or `"constant"` (default: `"lang"`).
#' @param theta Numeric. Base scaling factor (default: `10000`).
#' @param max_freq Numeric. Maximum frequency for pixel-based embeddings (default: `10`).
#' @param num_freqs Integer. Number of frequencies for `"constant"` mode (default: `1L`).
#' @param learned_freq Logical. If `TRUE`, frequencies are learnable parameters (default: `FALSE`).
#' @param use_xpos Logical. If `TRUE`, uses extrapolatable rotary embeddings (XPOS) (default: `FALSE`).
#' @param xpos_scale_base Numeric. Base scaling factor used for XPOS (default: `512`).
#' @param interpolate_factor Numeric. Factor by which sequence length is interpolated (default: `1.0`).
#' @param theta_rescale_factor Numeric. Rescaling factor applied to theta (default: `1.0`).
#' @param seq_before_head_dim Logical. If `TRUE`, sequences are handled before head dim (default: `FALSE`).
#' @param cache_if_possible Logical. If `TRUE`, caches computed frequencies (default: `TRUE`).
#'
#' @return An `nn_module` object.
#'
#' @export
RotaryEmbedding <- torch::nn_module(
  "RotaryEmbedding",

  initialize = function(
    dim,
    interleaved = TRUE,
    custom_freqs = NULL,
    freqs_for = "lang",
    theta = 10000,
    max_freq = 10,
    num_freqs = 1L,
    learned_freq = FALSE,
    use_xpos = FALSE,
    xpos_scale_base = 512,
    interpolate_factor = 1.0,
    theta_rescale_factor = 1.0,
    seq_before_head_dim = FALSE,
    cache_if_possible = TRUE
  ) {
    # NTK-aware scaling
    theta <- theta * (theta_rescale_factor ^ (dim / (dim - 2)))

    self$freqs_for <- freqs_for
    self$cache_if_possible <- cache_if_possible
    self$learned_freq <- learned_freq
    self$seq_before_head_dim <- seq_before_head_dim
    self$default_seq_dim <- if (seq_before_head_dim) -3L else -2L
    self$interpolate_factor <- interpolate_factor
    self$interleaved <- interleaved
    self$use_xpos <- use_xpos

    if (interpolate_factor < 1.0) {
      value_error("interpolate_factor must be >= 1.0, got {interpolate_factor}")
    }

    # Compute frequencies
    if (.exists(custom_freqs)) {
      freqs <- custom_freqs
    } else if (freqs_for == "lang") {
      # R: torch_arange is 0-based if we start at 0, but we need to slice carefully
      indices <- torch_arange(0L, dim, step = 2L, dtype = torch_float())
      freqs <- 1.0 / (theta ^ (indices[seq_len(dim %/% 2L)] / dim))
    } else if (freqs_for == "pixel") {
      freqs <- torch_linspace(1.0, max_freq / 2, dim %/% 2L) * pi
    } else if (freqs_for == "constant") {
      freqs <- torch_ones(num_freqs)$to(dtype = torch_float())
    }

    # Register buffers
    self$register_buffer("cached_freqs", NULL, persistent = FALSE)
    self$register_buffer("cached_scales", NULL, persistent = FALSE)
    self$register_buffer("dummy", torch_tensor(0, device = "cpu"), persistent = FALSE)

    self$freqs <- nn_parameter(freqs, requires_grad = learned_freq)

    # XPOS initialization
    if (use_xpos) {
      indices <- torch_arange(0L, dim, step = 2L, dtype = torch_float())
      scale <- (indices + 0.4 * dim) / (1.4 * dim)
      self$scale_base <- xpos_scale_base
      self$register_buffer("scale", scale, persistent = FALSE)
    } else {
      self$scale_base <- xpos_scale_base
      self$register_buffer("scale", NULL, persistent = FALSE)
    }
  },

  # Get the device where module parameters are stored
  #
  # @return Character string indicating the device.
  device = function() {
    self$dummy$device
  },

  # Compute sequence positions with interpolation
  #
  # @param seq_len Integer. Length of the sequence.
  # @param device Device where tensors should be placed.
  # @param dtype Data type for the output tensor.
  # @param offset Integer. Offset to add to positions (default: `0L`).
  # @return Tensor of shape `(seq_len,)` with interpolated positions.
  get_seq_pos = function(seq_len, device, dtype, offset = 0L) {
    (torch_arange(0L, seq_len, device = device, dtype = dtype) + offset) / self$interpolate_factor
  },

  # Rotate queries or keys independently
  #
  # @param t Input tensor to rotate.
  # @param seq_dim Integer. Sequence dimension (default: uses class default).
  # @param offset Integer. Offset for sequence positions (default: `0L`).
  # @param scale Optional scaling tensor for XPOS.
  # @return Rotated tensor with same shape as input.
  rotate_queries_or_keys = function(t, seq_dim = NULL, offset = 0L, scale = NULL) {
    seq_dim <- .default(seq_dim, self$default_seq_dim)

    if (self$use_xpos && is.null(scale)) {
      runtime_error("you must use `rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings")
    }

    device <- t$device
    dtype <- t$dtype

    seq_dim_pos <- if (seq_dim < 0) t$ndim + seq_dim + 1L else seq_dim
    seq_len <- t$shape[seq_dim_pos]

    seq <- self$get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

    if (self$interleaved) {
      freqs <- self$forward(seq, seq_len = seq_len, offset = offset)
    } else {
      # Non-interleaved: freqs shape is (seq_len, half)
      freqs <- torch_einsum("..., f -> ...f", list(seq$to(dtype = self$freqs$dtype), self$freqs))
    }

    if (seq_dim == -3L) {
      # Rearrange "n d -> n 1 d"
      freqs <- freqs$unsqueeze(2L)
    }

    apply_rotary_emb(freqs, t, scale = .default(scale, 1.0), seq_dim = seq_dim, interleaved = self$interleaved)
  },

  # Rotate queries with pre-cached keys (for incremental decoding)
  #
  # @param q Query tensor.
  # @param k Key tensor.
  # @param seq_dim Integer. Sequence dimension.
  # @param offset Integer. Offset for sequence positions.
  # @return List of two tensors: `(rotated_q, rotated_k)`.
  rotate_queries_with_cached_keys = function(q, k, seq_dim = NULL, offset = 0L) {
    dtype <- q$dtype
    device <- q$device
    seq_dim <- .default(seq_dim, self$default_seq_dim)

    q_len <- q$shape[seq_dim]
    k_len <- k$shape[seq_dim]

    if (q_len > k_len) {
      value_error("query length {q_len} must be <= key length {k_len}")
    }

    q_scale <- 1.0
    k_scale <- 1.0

    if (self$use_xpos) {
      seq <- self$get_seq_pos(k_len, dtype = dtype, device = device)
      # R slicing: (k_len - q_len + 1) to k_len
      start_idx <- k_len - q_len + 1L
      q_scale <- self$get_scale(seq[start_idx:k_len])$to(dtype = dtype)
      k_scale <- self$get_scale(seq)$to(dtype = dtype)
    }

    rotated_q <- self$rotate_queries_or_keys(q, seq_dim = seq_dim, scale = q_scale, offset = k_len - q_len + offset)
    rotated_k <- self$rotate_queries_or_keys(k, seq_dim = seq_dim, scale = k_scale^(-1))

    list(
      rotated_q$to(dtype = q$dtype),
      rotated_k$to(dtype = k$dtype)
    )
  },

  # Rotate both queries and keys together (for XPOS)
  #
  # @param q Query tensor.
  # @param k Key tensor.
  # @param seq_dim Integer. Sequence dimension.
  # @return List of two tensors: `(rotated_q, rotated_k)`.
  rotate_queries_and_keys = function(q, k, seq_dim = NULL) {
    seq_dim <- .default(seq_dim, self$default_seq_dim)

    if (!self$use_xpos) {
      runtime_error("rotate_queries_and_keys requires use_xpos = TRUE")
    }

    device <- q$device
    dtype <- q$dtype

    seq_dim_pos <- if (seq_dim < 0) q$ndim + seq_dim + 1L else seq_dim
    seq_len <- q$shape[seq_dim_pos]

    seq <- self$get_seq_pos(seq_len, dtype = dtype, device = device)

    if (self$interleaved) {
      freqs <- self$forward(seq, seq_len = seq_len)
    } else {
      freqs <- torch_einsum("..., f -> ...f", list(seq$to(dtype = self$freqs$dtype), self$freqs))
    }

    scale <- self$get_scale(seq, seq_len = seq_len)$to(dtype = dtype)

    if (seq_dim == -3L) {
      freqs <- freqs$unsqueeze(2L)
      scale <- scale$unsqueeze(2L)
    }

    rotated_q <- apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim, interleaved = self$interleaved)
    rotated_k <- apply_rotary_emb(freqs, k, scale = scale^(-1), seq_dim = seq_dim, interleaved = self$interleaved)

    list(
      rotated_q$to(dtype = q$dtype),
      rotated_k$to(dtype = k$dtype)
    )
  },

  # Compute XPOS scale factors
  #
  # @param t Position tensor (1D).
  # @param seq_len Optional sequence length for caching.
  # @param offset Integer. Offset for positions.
  # @return Scale tensor of shape `(seq_len, dim)` for XPOS.
  get_scale = function(t, seq_len = NULL, offset = 0L) {
    if (!self$use_xpos) {
      runtime_error("get_scale requires use_xpos = TRUE")
    }

    should_cache <- self$cache_if_possible && .exists(seq_len)
    t_len <- t$size(1L)

    if (should_cache && !is.null(self$cached_scales) && (seq_len + offset) <= self$cached_scales$size(1L)) {
      start_idx <- offset + 1L
      end_idx <- offset + seq_len
      return(self$cached_scales[start_idx:end_idx, , drop = FALSE])
    }

    # R: positions are 1-based, convert to 0-based for formula: (1:t_len) - 1
    positions_0based <- (seq_len(t_len) - 1L)
    power <- (positions_0based - t_len %/% 2L) / self$scale_base

    # Rearrange "n -> n 1"
    power_tensor <- torch_tensor(power, device = t$device, dtype = t$dtype)
    scale <- self$scale ^ power_tensor$unsqueeze(-1L)

    # Repeat to full dim (equivalent to torch.cat([scale, scale], dim=-1))
    scale <- torch_cat(list(scale, scale), dim = -1L)

    if (should_cache) {
      self$register_buffer("cached_scales", scale, persistent = FALSE)
    }

    scale
  },

  # Forward pass: compute rotary frequency tensor
  #
  # @param t Position tensor (1D or broadcastable).
  # @param seq_len Optional sequence length for caching.
  # @param offset Integer. Offset for positions.
  # @return Frequency tensor of shape `(.., dim)` for rotary embeddings.
  forward = function(t, seq_len = NULL, offset = 0L) {
    should_cache <- (
      self$cache_if_possible &&
        !self$learned_freq &&
        .exists(seq_len) &&
        self$freqs_for != "pixel"
    )

    if (should_cache && !is.null(self$cached_freqs) && (offset + seq_len) <= self$cached_freqs$size(1L)) {
      start_idx <- offset + 1L
      end_idx <- offset + seq_len
      return(self$cached_freqs[start_idx:end_idx, , drop = FALSE]$detach())
    }

    freqs <- self$freqs

    # Einsum: ".., f -> .. f"
    freqs <- torch_einsum("..., f -> ...f", list(t$to(dtype = freqs$dtype), freqs))

    # Repeat: ".. n -> .. (n r)", r=2
    # Equivalent to unsqueeze last dim, expand to 2, then flatten last two dims
    freqs <- freqs$unsqueeze(-1L)$expand(c(freqs$shape, 2L))$flatten(start_dim = length(freqs$shape))

    if (should_cache) {
      self$register_buffer("cached_freqs", freqs$detach(), persistent = FALSE)
    }

    freqs
  }
)
