# Translated from TabICL2 python/_model/ssmax.py


#' @importFrom torch nn_module nn_parameter nn_sequential
#' @importFrom torch nn_linear nn_gelu
#' @importFrom torch torch_tensor torch_ones torch_tanh
#' @importFrom torch nn_init_zeros_
NULL

#' Compute log(n) as a scalar tensor on the correct device/dtype.
#'
#' Clamps \code{n} to at least 1 before taking the logarithm to
#' avoid fp16 overflow for very large sequence lengths.
#'
#' @param n `integer(1)` Sequence length.
#' @param device Device on which to place the tensor.
#' @param dtype Torch dtype of the resulting tensor.
#'
#' @return A 0-d tensor containing \code{log(max(n, 1))}.
#'
#' @noRd
.logn <- function(n, device, dtype) {
  torch_tensor(log(max(n, 1)), device = device, dtype = dtype)
}

#' SSMax -- Scalable Softmax with Learnable Per-Head Scaling
#'
#' @description Applies scaling to queries:
#'   \eqn{q_{\text{scaled}} = q \cdot (s \cdot \log n)},
#'   where \eqn{s} is a learnable per-head parameter.
#'
#' @param num_heads `integer(1)` Number of attention heads.
#'
#' @return An \code{\link[torch]{nn_module}} object (class generator).
#'
#' @seealso \code{\link{create_ssmax_layer}}
#' @export
SSMax <- nn_module(
  classname = "SSMax",

  initialize = function(num_heads) {
    self$scales <- nn_parameter(torch_ones(num_heads))
  },

  #' @description Apply SSMax scaling to queries.
  #'
  #' @param q `tensor` of shape \code{(bs, n_heads, seq_len, head_dim)}.
  #'   Query tensor after projection.
  #' @param n `integer(1)` Source sequence length.
  #'
  #' @return Scaled query tensor, same shape as \code{q}.
  forward = function(q, n) {
    logn <- .logn(n, q$device, q$dtype)
    scales <- self$scales$view(c(1L, -1L, 1L, 1L)) * logn
    q * scales
  }
)

#' SSMaxMLP -- Scalable Softmax with MLP-Computed Scaling
#'
#' @description Applies scaling to queries:
#'   \eqn{q_{\text{scaled}} = q \cdot \text{mlp}(\log n)},
#'   where a small MLP learns to map sequence length to scaling factors.
#'
#' @param num_heads `integer(1)` Number of attention heads.
#' @param n_hidden `integer(1)` Number of hidden units in the MLP.
#'   Default: \code{64L}.
#' @param elementwise `logical(1)` If \code{TRUE}, apply elementwise
#'   scaling per head dimension so that each element in the head
#'   dimension gets its own scaling factor.  Default: \code{FALSE}.
#' @param head_dim `integer(1)` or \code{NULL}. Dimension of each
#'   attention head. Required when \code{elementwise = TRUE}.
#'
#' @return An \code{\link[torch]{nn_module}} object (class generator).
#'
#' @seealso \code{\link{create_ssmax_layer}}
#' @export
SSMaxMLP <- nn_module(
  classname = "SSMaxMLP",

  initialize = function(
    num_heads,
    n_hidden    = 64L,
    elementwise = FALSE,
    head_dim    = NULL
  ) {
    self$elementwise <- elementwise

    if (elementwise) {
      if (is.null(head_dim)) {
        value_error(
          "head_dim must be provided when elementwise=TRUE"
        )
      }
      out_dim <- num_heads * head_dim
    } else {
      out_dim <- num_heads
    }

    self$mlp <- nn_sequential(
      nn_linear(1L, n_hidden),
      nn_gelu(),
      nn_linear(n_hidden, out_dim)
    )

    self$num_heads <- num_heads
  },

  #' @description Apply SSMax-MLP scaling to queries.
  #'
  #' @param q `tensor` of shape \code{(bs, n_heads, seq_len, head_dim)}.
  #'   Query tensor after projection.
  #' @param n `integer(1)` Source sequence length.
  #'
  #' @return Scaled query tensor, same shape as \code{q}.
  forward = function(q, n) {
    logn <- .logn(n, q$device, q$dtype)$view(c(1L, 1L))
    scales <- self$mlp(logn)

    if (self$elementwise) {
      # scales: (1, num_heads * head_dim) -> (1, num_heads, 1, head_dim)
      head_dim <- q$size(-1L)
      scales <- scales$view(c(1L, self$num_heads, 1L, head_dim))
    } else {
      # scales: (1, num_heads) -> (1, num_heads, 1, 1)
      scales <- scales$view(c(1L, self$num_heads, 1L, 1L))
    }

    q * scales
  }
)


#' QASSMaxMLP -- Query-Aware Scalable Softmax with MLPs
#'
#' @description Applies scaling to queries using two MLPs:
#'
#' \eqn{q_{\text{scaled}} = q \cdot \text{base\_mlp}(\log n)
#'   \cdot (1 + \tanh(\text{query\_mlp}(q)))}
#'
#' The \emph{base MLP} learns length-dependent scaling while the
#' \emph{query MLP} learns query-dependent modulation.  The query
#' MLP is zero-initialised so that the initial modulation is the
#' identity (\eqn{1 + 0 = 1}).
#'
#' @param num_heads `integer(1)` Number of attention heads.
#' @param head_dim `integer(1)` Dimension of each attention head.
#' @param n_hidden `integer(1)` Number of hidden units in each MLP.
#'   Default: \code{64L}.
#' @param elementwise `logical(1)` If \code{TRUE}, apply elementwise
#'   scaling per head dimension.  Default: \code{FALSE}.
#'
#' @return An \code{\link[torch]{nn_module}} object (class generator).
#'
#' @seealso \code{\link{create_ssmax_layer}}
#' @export
QASSMaxMLP <- nn_module(
  classname = "QASSMaxMLP",

  initialize = function(
    num_heads,
    head_dim,
    n_hidden    = 64L,
    elementwise = FALSE
  ) {
    self$num_heads   <- num_heads
    self$head_dim    <- head_dim
    self$elementwise <- elementwise

    if (elementwise) {
      base_out_dim  <- num_heads * head_dim
      query_out_dim <- head_dim
    } else {
      base_out_dim  <- num_heads
      query_out_dim <- 1L
    }

    self$base_mlp <- nn_sequential(
      nn_linear(1L, n_hidden),
      nn_gelu(),
      nn_linear(n_hidden, base_out_dim)
    )

    self$query_mlp <- nn_sequential(
      nn_linear(head_dim, n_hidden),
      nn_gelu(),
      nn_linear(n_hidden, query_out_dim)
    )

    # Zero-initialise the last layer so initial modulation is zero
    # (1 + tanh(0) = 1, i.e. identity at init).
    last_layer <- self$query_mlp[[length(self$query_mlp)]]
    nn_init_zeros_(last_layer$weight)
    nn_init_zeros_(last_layer$bias)
  },

  #' @description Apply QASSMax scaling to queries.
  #'
  #' @param q `tensor` of shape \code{(bs, n_heads, seq_len, head_dim)}.
  #'   Query tensor after projection.
  #' @param n `integer(1)` Source sequence length.
  #'
  #' @return Scaled query tensor, same shape as \code{q}.
  forward = function(q, n) {
    logn <- .logn(n, q$device, q$dtype)$view(c(1L, 1L))

    if (self$elementwise) {
      # base_scales: (1, num_heads * head_dim) -> (1, num_heads, 1, head_dim)
      base_scales <- self$base_mlp(logn)$view(
        c(1L, self$num_heads, 1L, self$head_dim)
      )
      # modulation: (bs, n_heads, seq_len, head_dim)
      modulation <- 1 + torch_tanh(self$query_mlp(q))
    } else {
      # base_scales: (1, num_heads) -> (1, num_heads, 1, 1)
      base_scales <- self$base_mlp(logn)$view(
        c(1L, self$num_heads, 1L, 1L)
      )
      # modulation: (bs, n_heads, seq_len, 1)
      modulation <- 1 + torch_tanh(self$query_mlp(q))
    }

    scales <- base_scales * modulation

    q * scales
  }
)


#' Create an SSMax Layer by Type
#'
#' @description Factory function that instantiates the appropriate
#'   \code{SSMax*} module (or \code{NULL}) from a string identifier.
#'
#' @param ssmax_type `character(1)` One of:
#'   \code{"none"}, \code{"ssmax"}, \code{"ssmax-mlp"},
#'   \code{"ssmax-mlp-elementwise"}, \code{"qassmax-mlp"},
#'   \code{"qassmax-mlp-elementwise"}.
#' @param num_heads `integer(1)` Number of attention heads.
#' @param embed_dim `integer(1)` Total embedding dimension
#'   (\code{num_heads * head_dim}).
#'
#' @return An \code{nn_module} instance, or \code{NULL} when
#'   \code{ssmax_type = "none"}.
#'
#' @seealso \code{\link{SSMax}}, \code{\link{SSMaxMLP}},
#'   \code{\link{QASSMaxMLP}}
#' @export
create_ssmax_layer <- function(ssmax_type, num_heads, embed_dim) {
  if (ssmax_type == "none") {
    return(NULL)
  } else if (ssmax_type == "ssmax") {
    return(SSMax(num_heads))
  } else if (ssmax_type == "ssmax-mlp") {
    return(SSMaxMLP(num_heads))
  } else if (ssmax_type == "ssmax-mlp-elementwise") {
    return(
      SSMaxMLP(
        num_heads   = num_heads,
        head_dim    = embed_dim %/% num_heads,
        elementwise = TRUE
      )
    )
  } else if (ssmax_type == "qassmax-mlp") {
    return(
      QASSMaxMLP(
        num_heads = num_heads,
        head_dim  = embed_dim %/% num_heads
      )
    )
  } else if (ssmax_type == "qassmax-mlp-elementwise") {
    return(
      QASSMaxMLP(
        num_heads   = num_heads,
        head_dim    = embed_dim %/% num_heads,
        elementwise = TRUE
      )
    )
  } else {
    value_error("Unknown ssmax_type '{ssmax_type}'")
  }
}
