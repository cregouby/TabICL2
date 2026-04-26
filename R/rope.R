#' Rotary Positional Embedding for R {torch}
#'
#' Implementation of rotary positional embeddings (RoPE) for transformer models.
#' Ported from https://github.com/lucidrains/rotary-embedding-torch
#'
#' @name rotary_embedding
#' @keywords internal
NULL


#' Check if a value exists (is not NULL)
#'
#' @param val Value to check
#' @return `TRUE` if `val` is not `NULL`, `FALSE` otherwise
#' @keywords internal
exists_val <- function(val) {
  !is.null(val)
}


#' Return value or default fallback
#'
#' @param val Value to check
#' @param d Default value to return if `val` is `NULL`
#' @return `val` if not `NULL`, otherwise `d`
#' @keywords internal
default_val <- function(val, d) {
  if (exists_val(val)) val else d
}


#' Broadcast and concatenate a list of tensors
#'
#' Equivalent to the `broadcat` utility from tortoise-tts.
#'
#' @param tensors List of tensors to broadcast and concatenate
#' @param dim Dimension along which to concatenate (default: `-1L`)
#' @return Concatenated tensor after broadcasting
#' @keywords internal
broadcat <- function(tensors, dim = -1L) {
  broadcasted <- torch_broadcast_tensors(tensors)
  torch_cat(broadcasted, dim = dim)
}


#' Interleaved rotation: pairs (1,2), (3,4), (5,6), ... (1-based indexing)
#'
#' Given input `[.., d]`, rearranges to `[.., d/2, 2]` and rotates pairs.
#' Used by default in most RoPE implementations (e.g. LLaMA).
#'
#' @details
#' In Python (0-based): pairs are (0,1), (2,3), ...
#' In R (1-based): pairs are (1,2), (3,4), ... — same logical pairing,
#' just different index notation.
#'
#' @param x Input tensor of shape `[.., d]` where `d` is even
#' @return Rotated tensor with same shape as `x`
#' @keywords internal
rotate_half_interleaved <- function(x) {
  # Rearrange: [..., d] -> [..., d/2, 2]
  x <- einops::rearrange(x, "... (d r) -> ... d r", r = 2L)
  x1 <- x[.., 1L]  # R: first element of pair
  x2 <- x[.., 2L]  # R: second element of pair
  # Rotate: [-x2, x1]
  x <- torch_stack(list(-x2, x1), dim = -1L)
  # Rearrange back: [..., d/2, 2] -> [..., d]
  einops::rearrange(x, "... d r -> ... (d r)")
}


#' Contiguous rotation: split into first/second halves (1-based indexing)
#'
#' Given input `[..., d]`, splits into `[..., 1:(d/2)]` and `[..., (d/2+1):d]`
#' and rotates. Returns `[-x2, x1]` where `x1` is the first half and
#' `x2` is the second half.
#'
#' @param x Input tensor of shape `[..., d]` where `d` is even
#' @return Rotated tensor with same shape as `x`
#' @keywords internal
rotate_half_contiguous <- function(x) {
  last_dim <- length(x$shape)
  d <- x$shape[last_dim]
  half <- d %/% 2L

  # R indexing: 1-based, inclusive ranges
  x1 <- x[.., seq_len(half)]                    # [..., 1:half]
  x2 <- x[.., (half + 1L):d]                    # [..., (half+1):d]
  torch_cat(list(-x2, x1), dim = -1L)
}


#' Apply rotary embeddings to a tensor
#'
#' Computes \eqn{x \cdot \cos(\theta) + \text{rotate}(x) \cdot \sin(\theta)}
#' for the portion of the input that overlaps with the frequency tensor.
#'
#' @param freqs Frequency tensor for rotation. For interleaved mode, shape is
#'   `(..., dim)` where `dim = 2 * half`. For non-interleaved mode,
#'   shape is `(..., half)`.
#' @param t Input tensor to rotate
#' @param start_index Starting index for rotation in the last dimension
#'   (default: `1L` for R's 1-based indexing)
#' @param scale Scaling factor for the rotation (default: `1.0`)
#' @param seq_dim Sequence dimension (default: `-2L`)
#' @param interleaved If `TRUE`, uses interleaved rotation where dimension
#'   pairs are `(1,2), (3,4)`, etc. If `FALSE`, uses non-interleaved rotation
#'   where the embedding is split into first half `[1:(d/2)]` and
#'   second half `[(d/2+1):d]` (default: `TRUE`)
#' @return Rotated tensor, same shape as `t`
#' @keywords internal
apply_rotary_emb <- function(freqs, t, start_index = 1L, scale = 1.0,
                             seq_dim = -2L, interleaved = TRUE) {
  dtype <- t$dtype

  # Handle sequence truncation for 3D tensors
  if (t$ndim == 3L) {
    seq_len <- t$size(seq_dim)
    n_freqs <- freqs$size(1L)
    # R indexing: negative indexing works differently than Python
    # Python: freqs[-seq_len:] → R: freqs[(N-seq_len+1):N]
    if (n_freqs > seq_len) {
      freqs <- freqs[(n_freqs - seq_len + 1L):N, .. , drop = FALSE]
    }
  }
  # Determine rotation dimension
  if (interleaved) {
    rot_dim <- freqs$shape[length(freqs$shape)]
  } else {
    # Non-interleaved: freqs has shape (.., half), need full dim
    rot_dim <- freqs$shape[length(freqs$shape)] * 2L
  }

  # R: end_index is inclusive, so subtract 1 from Python formula
  end_index <- start_index + rot_dim - 1L

  # Safety check
  last_dim_size <- t$size(-1L)
  if(rot_dim > last_dim_size) {
    value_error("feature dimension {last_dim_size} is not of sufficient size to rotate in all the positions {rot_dim}")
  }

  # Split t into three parts: left, middle (transformed), right
  # R indexing: 1-based, inclusive ranges
  t_left <- if (start_index > 1L) t[.., seq_len(start_index - 1L)] else NULL
  t_middle <- t[.., start_index:end_index]
  t_right <- if (end_index < last_dim_size) t[.., (end_index + 1L):last_dim_size] else NULL

  # Apply rotary embeddings: x*cos + rotate(x)*sin
  if (interleaved) {
    # Interleaved mode: freqs has shape (..., rot_dim)
    t_transformed <- (t_middle * freqs$cos() * scale) +
      (rotate_half_interleaved(t_middle) * freqs$sin() * scale)
  } else {
    # Non-interleaved: freqs has shape (..., half), expand to full dim
    cos_freq <- torch_cat(list(freqs$cos(), freqs$cos()), dim = -1L) * scale
    sin_freq <- torch_cat(list(freqs$sin(), freqs$sin()), dim = -1L) * scale
    t_transformed <- (t_middle * cos_freq) +
      (rotate_half_contiguous(t_middle) * sin_freq)
  }

  # Concatenate parts back together
  parts <- list()
  if (!is.null(t_left)) parts <- c(parts, list(t_left))
  parts <- c(parts, list(t_transformed))
  if (!is.null(t_right)) parts <- c(parts, list(t_right))

  out <- torch_cat(parts, dim = -1L)

  # Restore original dtype (R torch equivalent of autocast management)
  out$to(dtype = dtype)
}


#' Apply learned rotations helper
#'
#' @param rotations Rotation tensor
#' @param t Input tensor to rotate
#' @param start_index Starting index for rotation (default: `1L`)
#' @param freq_ranges Optional frequency ranges tensor
#' @return Rotated tensor
#' @keywords internal
apply_learned_rotations <- function(rotations, t, start_index = 1L, freq_ranges = NULL) {
  if (exists_val(freq_ranges)) {
    rotations <- torch_einsum("..., f -> ... f", list(rotations, freq_ranges))
    rotations <- einops::rearrange(rotations, "... r f -> ... (r f)")
  }

  rotations <- einops::einops.repeat(rotations, "... n -> ... (n r)", r = 2L)
  apply_rotary_emb(rotations, t, start_index = start_index)
}

#' Rotary Embedding R6 Class
#'
#' Rotary positional embeddings for use in transformer models.
#'
#' Rotary embeddings encode positional information in a way that allows
#' continuous rotation of embeddings, enhancing the model's ability to
#' capture long-range dependencies and positional relations.
#'
#' @section Fields (private):
#' \describe{
#'   \item{freqs_for}{Character. Type of frequencies: "lang", "pixel", or "constant".}
#'   \item{cache_if_possible}{Logical. Whether to cache computed frequencies.}
#'   \item{cached_freqs}{Tensor or NULL. Cached frequency tensor.}
#'   \item{cached_scales}{Tensor or NULL. Cached scale tensor for XPOS.}
#'   \item{freqs}{nn_parameter. Frequency tensor (learnable or fixed).}
#'   \item{learned_freq}{Logical. Whether frequencies are learnable.}
#'   \item{dummy}{Tensor. Dummy buffer for device tracking.}
#'   \item{seq_before_head_dim}{Logical. Sequence dimension placement.}
#'   \item{default_seq_dim}{Integer. Default sequence dimension (-2 or -3).}
#'   \item{interpolate_factor}{Numeric. Sequence interpolation factor.}
#'   \item{interleaved}{Logical. Use interleaved rotation mode.}
#'   \item{use_xpos}{Logical. Use extrapolatable XPOS embeddings.}
#'   \item{scale_base}{Numeric. Base scaling factor for XPOS.}
#'   \item{scale}{Tensor or NULL. Scale tensor for XPOS.}
#' }
#'
#' @examples
#' \dontrun{
#' # Create a rotary embedding module
#' rope <- RotaryEmbedding$new(dim = 64L)
#'
#' # Apply to a tensor [batch, seq_len, dim]
#' x <- torch_randn(2L, 10L, 64L)
#' x_rotated <- rope$rotate_queries_or_keys(x)
#'
#' # With XPOS for length extrapolation
#' rope_xpos <- RotaryEmbedding$new(dim = 64L, use_xpos = TRUE)
#' q <- torch_randn(2L, 8L, 64L)
#' k <- torch_randn(2L, 10L, 64L)
#' result <- rope_xpos$rotate_queries_and_keys(q, k)
#' rotated_q <- result[[1]]
#' rotated_k <- result[[2]]
#' }
#'

#' Initialize a RotaryEmbedding module
#'
#' @param dim Integer. Dimension of the embeddings (must be even for interleaved mode).
#' @param interleaved Logical. If `TRUE`, uses interleaved rotation where
#'   dimension pairs are `(1,2), (3,4)`, etc. If `FALSE`, uses non-interleaved
#'   rotation splitting embedding into halves (default: `TRUE`).
#' @param custom_freqs Tensor or NULL. Custom frequency tensor. If provided,
#'   overrides default frequency computation (default: `NULL`).
#' @param freqs_for Character. One of `"lang"`, `"pixel"`, or `"constant"`
#'   specifying the frequency generation strategy (default: `"lang"`).
#' @param theta Numeric. Base scaling factor for rotary embeddings (default: `10000`).
#' @param max_freq Numeric. Maximum frequency for pixel-based embeddings (default: `10`).
#' @param num_freqs Integer. Number of frequencies for `"constant"` mode (default: `1L`).
#' @param learned_freq Logical. If `TRUE`, frequencies are learnable parameters (default: `FALSE`).
#' @param use_xpos Logical. If `TRUE`, uses extrapolatable XPOS embeddings (default: `FALSE`).
#' @param xpos_scale_base Numeric. Base scaling factor for XPOS (default: `512`).
#' @param interpolate_factor Numeric. Factor to interpolate sequence length (default: `1.0`).
#' @param theta_rescale_factor Numeric. Rescaling factor for theta (NTK-aware scaling) (default: `1.0`).
#' @param seq_before_head_dim Logical. If `TRUE`, sequence dimension comes before head dimension (default: `FALSE`).
#' @param cache_if_possible Logical. If `TRUE`, cache computed frequencies for efficiency (default: `TRUE`).
#' @return A new `RotaryEmbedding` object (invisibly)
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
    # NTK-aware theta rescaling for longer sequences
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    theta <- theta * (theta_rescale_factor ^ (dim / (dim - 2)))

    private$freqs_for <- freqs_for

    # Compute initial frequencies based on strategy
    if (exists_val(custom_freqs)) {
      freqs <- custom_freqs
    } else if (freqs_for == "lang") {
      # Language frequencies: 1/theta^(2i/d) for i in 0,1,...,d/2-1
      # Python: torch.arange(0, dim, 2)[:dim//2] → [0,2,4,...,dim-2] (0-based)
      # R: seq(1, dim, by=2)[1:(dim/2)] → [1,3,5,...,dim-1] (1-based)
      # Formula uses 0-based indices, so subtract 1
      indices_0based <- seq(0L, dim - 1L, by = 2L)[seq_len(dim %/% 2L)]
      freqs <- torch_ones(1) / (theta ^ (indices_0based / dim))
    } else if (freqs_for == "pixel") {
      # Pixel frequencies: linearly spaced for image data
      freqs <- torch_linspace(1.0, max_freq / 2, dim %/% 2L) * pi
    } else if (freqs_for == "constant") {
      # Constant frequencies: all ones
      freqs <- torch_ones(num_freqs)$float()
    }

    private$cache_if_possible <- cache_if_possible

    # Initialize non-persistent buffers for caching
    self$register_buffer("cached_freqs", NULL, persistent = FALSE)
    self$register_buffer("cached_scales", NULL, persistent = FALSE)

    # Register frequencies as parameter (learnable or fixed)
    private$freqs <- nn_parameter(freqs, requires_grad = learned_freq)
    private$learned_freq <- learned_freq

    # Dummy tensor for device tracking (R torch equivalent of Python's register_buffer)
    self$register_buffer("dummy", torch_tensor(0, device = "cpu"), persistent = FALSE)

    # Sequence dimension configuration
    private$seq_before_head_dim <- seq_before_head_dim
    private$default_seq_dim <- if (seq_before_head_dim) -3L else -2L

    # Interpolation factor validation
    stopifnot(interpolate_factor >= 1.0)
    private$interpolate_factor <- interpolate_factor

    # Rotation mode
    private$interleaved <- interleaved

    # XPOS configuration
    private$use_xpos <- use_xpos
    if (!use_xpos) {
      self$register_buffer("scale", NULL, persistent = FALSE)
      return(invisible(self))
    }

    # XPOS scale computation
    # Python: torch.arange(0, dim, 2) → R: use 0-based indices for formula consistency
    indices_0based <- seq(0L, dim - 1L, by = 2L)
    scale <- ((indices_0based + 0.4 * dim) / (1.4 * dim))
    private$scale_base <- xpos_scale_base
    self$register_buffer("scale", scale, persistent = FALSE)

    invisible(self)
  },

  #Get the device where module parameters are stored
  #
  # @return Character string indicating the device (e.g., `"cpu"`, `"cuda"`)
  device = function() {
    private$dummy$device
  },

  #Store a non-persistent buffer
  #
  # Internal helper to register temporary tensors that shouldn't be saved
  # in model state dicts.
  #
  # @param key Character. Name of the buffer.
  # @param value Tensor or NULL. Value to store.
  # @keywords internal
  tmp_store = function(key, value) {
    self$register_buffer(key, value, persistent = FALSE)
  },

  # Compute sequence positions with interpolation
  #
  # @param seq_len Integer. Length of the sequence.
  # @param device Device where tensors should be placed.
  # @param dtype Data type for the output tensor.
  # @param offset Integer. Offset to add to positions (default: `0L`).
  # @return Tensor of shape `(seq_len,)` with interpolated positions
  get_seq_pos = function(seq_len, device, dtype, offset = 0L) {
    (torch_arange(seq_len, device = device, dtype = dtype) + offset) / private$interpolate_factor
  },

  # Rotate queries or keys independently
  #
  # @param t Input tensor to rotate.
  # @param seq_dim Integer. Sequence dimension (default: uses class default).
  # @param offset Integer. Offset for sequence positions (default: `0L`).
  # @param scale Optional scaling tensor for XPOS (required if `use_xpos=TRUE`).
  # @return Rotated tensor with same shape as input
  rotate_queries_or_keys = function(t, seq_dim = NULL, offset = 0L, scale = NULL) {
    seq_dim <- default_val(seq_dim, private$default_seq_dim)

    # XPOS requires explicit scale when rotating queries/keys separately
    stopifnot(!private$use_xpos || exists_val(scale),
              "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings")

    device <- t$device
    dtype <- t$dtype
    seq_len <- t$shape[seq_dim]

    # Get positions and compute frequencies
    seq <- self$get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset)

    if (private$interleaved) {
      freqs <- self$forward(seq, seq_len = seq_len, offset = offset)
    } else {
      # Non-interleaved: freqs shape is (seq_len, half), no repetition yet
      freqs <- torch_einsum("..., f -> ... f",
                            list(seq$to(dtype = private$freqs$dtype), private$freqs))
    }

    # Adjust frequency tensor shape for batch dimensions if needed
    if (seq_dim == -3L) {
      freqs <- einops::rearrange(freqs, "n d -> n 1 d")
    }

    apply_rotary_emb(freqs, t, scale = default_val(scale, 1.0),
                     seq_dim = seq_dim, interleaved = private$interleaved)
  },

  # Rotate queries with pre-cached keys (for incremental decoding)
  #
  # @param q Query tensor.
  # @param k Key tensor (must be at least as long as `q`).
  # @param seq_dim Integer. Sequence dimension (default: uses class default).
  # @param offset Integer. Offset for sequence positions (default: `0L`).
  # @return List of two tensors: `(rotated_q, rotated_k)`
  rotate_queries_with_cached_keys = function(q, k, seq_dim = NULL, offset = 0L) {
    dtype <- q$dtype
    device <- q$device
    seq_dim <- default_val(seq_dim, private$default_seq_dim)

    q_len <- q$shape[seq_dim]
    k_len <- k$shape[seq_dim]
    stopifnot(q_len <= k_len,
              sprintf("query length %d must be <= key length %d", q_len, k_len))

    q_scale <- 1.0
    k_scale <- 1.0

    if (private$use_xpos) {
      seq <- self$get_seq_pos(k_len, dtype = dtype, device = device)

      # Extract scales for query portion (last q_len positions)
      if (q_len > 0) {
        start_idx <- k_len - q_len + 1L  # R: 1-based inclusive
        q_scale <- self$get_scale(seq[start_idx:k_len])$to(dtype = dtype)
      }
      k_scale <- self$get_scale(seq)$to(dtype = dtype)
    }

    # Rotate query with offset to align with key positions
    rotated_q <- self$rotate_queries_or_keys(q, seq_dim = seq_dim,
                                             scale = q_scale,
                                             offset = k_len - q_len + offset)
    # Rotate key with inverse scale
    rotated_k <- self$rotate_queries_or_keys(k, seq_dim = seq_dim,
                                             scale = k_scale^(-1))

    list(
      rotated_q$to(dtype = q$dtype),
      rotated_k$to(dtype = k$dtype)
    )
  },

  # Rotate both queries and keys together (for XPOS)
  #
  # @param q Query tensor.
  # @param k Key tensor.
  # @param seq_dim Integer. Sequence dimension (default: uses class default).
  # @return List of two tensors: `(rotated_q, rotated_k)`
  rotate_queries_and_keys = function(q, k, seq_dim = NULL) {
    seq_dim <- default_val(seq_dim, private$default_seq_dim)

    stopifnot(private$use_xpos,
              "rotate_queries_and_keys requires use_xpos = TRUE")

    device <- q$device
    dtype <- q$dtype
    seq_len <- q$shape[seq_dim]

    seq <- self$get_seq_pos(seq_len, dtype = dtype, device = device)

    if (private$interleaved) {
      freqs <- self$forward(seq, seq_len = seq_len)
    } else {
      freqs <- torch_einsum("..., f -> ... f",
                            list(seq$to(dtype = private$freqs$dtype), private$freqs))
    }

    scale <- self$get_scale(seq, seq_len = seq_len)$to(dtype = dtype)

    # Adjust shapes for batch dimensions
    if (seq_dim == -3L) {
      freqs <- einops::rearrange(freqs, "n d -> n 1 d")
      scale <- einops::rearrange(scale, "n d -> n 1 d")
    }

    # Apply rotation with scale for query, inverse scale for key
    rotated_q <- apply_rotary_emb(freqs, q, scale = scale,
                                  seq_dim = seq_dim, interleaved = private$interleaved)
    rotated_k <- apply_rotary_emb(freqs, k, scale = scale^(-1),
                                  seq_dim = seq_dim, interleaved = private$interleaved)

    list(
      rotated_q$to(dtype = q$dtype),
      rotated_k$to(dtype = k$dtype)
    )
  },

  # Compute XPOS scale factors
  #
  # @param t Position tensor (1D).
  # @param seq_len Optional sequence length for caching.
  # @param offset Integer. Offset for positions (default: `0L`).
  # @return Scale tensor of shape `(seq_len, dim)` for XPOS
  get_scale = function(t, seq_len = NULL, offset = 0L) {
    stopifnot(private$use_xpos, "get_scale requires use_xpos = TRUE")

    should_cache <- private$cache_if_possible && exists_val(seq_len)

    # Check cache first
    if (should_cache && exists_val(private$cached_scales)) {
      cached_len <- private$cached_scales$size(1L)
      if ((offset + seq_len) <= cached_len) {
        start_idx <- offset + 1L  # R: 1-based
        end_idx <- offset + seq_len
        return(private$cached_scales[start_idx:end_idx, , drop = FALSE])
      }
    }

    # Compute scale: (pos - seq_len/2) / scale_base
    # R: positions are 1-based, convert to 0-based for formula
    positions_0based <- (seq_along(t) - 1L)  # [0, 1, 2, ..., len(t)-1]
    power <- (positions_0based - length(t) %/% 2L) / private$scale_base

    # Rearrange and compute scale
    power_tensor <- torch_tensor(power, device = t$device, dtype = t$dtype)
    scale <- private$scale ^ einops::rearrange(power_tensor, "n -> n 1")

    # Duplicate for full dimension (interleaved-style expansion)
    scale <- torch_cat(list(scale, scale), dim = -1L)

    # Cache if appropriate
    if (should_cache) {
      self$tmp_store("cached_scales", scale)
    }

    scale
  },

  # Compute axial frequencies for multi-dimensional positional encoding
  #
  # @param ... Integer dimensions for each axis (e.g., height, width for images).
  # @return Combined frequency tensor for axial positions
  get_axial_freqs = function(...) {
    dims <- list(...)
    all_freqs <- list()

    for (ind in seq_along(dims)) {
      dim <- dims[[ind]]

      # Position values based on frequency type
      if (private$freqs_for == "pixel") {
        pos <- torch_linspace(-1, 1, steps = dim, device = self$device)
      } else {
        pos <- torch_arange(dim, device = self$device)
      }

      # Compute frequencies for this axis
      freqs <- self$forward(pos, seq_len = dim)

      # Prepare broadcasting: expand freqs to have singleton dims for other axes
      # This is complex in R; using einops for clarity
      expand_pattern <- paste0("...", paste(rep("1", length(dims) - 1), collapse = " "), "d")
      # Note: This is simplified; full implementation may need custom broadcasting logic
      all_freqs[[ind]] <- freqs
    }

    # Broadcast and concatenate along feature dimension
    all_freqs <- torch_broadcast_tensors(all_freqs)
    torch_cat(all_freqs, dim = -1L)
  },

  # Forward pass: compute rotary frequency tensor
  #
  # @param t Position tensor (1D or broadcastable).
  # @param seq_len Optional sequence length for caching.
  # @param offset Integer. Offset for positions (default: `0L`).
  # @return Frequency tensor of shape `(..., dim)` for rotary embeddings
  #
  # @note
  # The Python `@torch.autocast("cuda", enabled=False)` decorator disables
  # mixed precision for this function. In R {torch}, handle dtype explicitly
  # by converting inputs to appropriate precision before calling.
  forward = function(t, seq_len = NULL, offset = 0L) {
    should_cache <- (
      private$cache_if_possible &&
        !private$learned_freq &&
        exists_val(seq_len) &&
        private$freqs_for != "pixel"
    )

    # Check cache first
    if (should_cache && exists_val(private$cached_freqs)) {
      cached_len <- private$cached_freqs$size(1L)
      if ((offset + seq_len) <= cached_len) {
        start_idx <- offset + 1L  # R: 1-based inclusive
        end_idx <- offset + seq_len
        return(private$cached_freqs[start_idx:end_idx, , drop = FALSE]$detach())
      }
    }

    freqs <- private$freqs

    # Compute outer product: positions × base frequencies
    freqs <- torch_einsum("..., f -> ... f",
                          list(t$to(dtype = freqs$dtype), freqs))

    # Repeat each frequency twice for interleaved rotation: [f0, f0, f1, f1, ...]
    freqs <- einops::einops.repeat(freqs, "... n -> ... (n r)", r = 2L)

    # Cache if appropriate (detach to avoid gradient tracking)
    if (should_cache) {
      self$tmp_store("cached_freqs", freqs$detach())
    }

    freqs
  },

  private = list(
    freqs_for = NULL,
    cache_if_possible = NULL,
    cached_freqs = NULL,
    cached_scales = NULL,
    freqs = NULL,
    learned_freq = NULL,
    dummy = NULL,
    seq_before_head_dim = NULL,
    default_seq_dim = NULL,
    interpolate_factor = NULL,
    interleaved = NULL,
    use_xpos = NULL,
    scale_base = NULL,
    scale = NULL
  )
)
