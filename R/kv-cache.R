#' @importFrom torch torch_cat torch_zeros
NULL

#' Compute the number of bytes per element for common torch dtypes.
#'
#' @param tensor A torch tensor.
#' @return `integer(1)` Bytes per element.
#' @keywords internal
.element_size <- function(tensor) {
  dtype_str <- sub("^torch\\.", "", tolower(as.character(tensor$dtype)))
  sizes <- c(
    float16 = 2L, bfloat16 = 2L, float32 = 4L, float64 = 8L,
    int8 = 1L, int16 = 2L, int32 = 4L, int64 = 8L,
    uint8 = 1L, bool = 1L, complex32 = 8L, complex64 = 16L
  )
  if (dtype_str %in% names(sizes)) {
    return(sizes[[dtype_str]])
  }
  4L # default fallback (float32)
}

.numel <- function(tensor) {
  tensor$view(-1)$shape

}
#' KVCacheEntry -- Single Key-Value Cache Entry
#'
#' @description An R6 class holding optional cached key and value
#'   projections for one attention layer.  Supports slicing, device
#'   transfer, concatenation and in-place assignment.
#'
#' @export
KVCacheEntry <- R6::R6Class(
  classname = "KVCacheEntry",

  public = list(

    #' @field key Cached key tensor of shape
    #'   \code{(batch, num_heads, seq_len, head_dim)}, or \code{NULL}.
    key = NULL,

    #' @field value Cached value tensor of shape
    #'   \code{(batch, num_heads, seq_len, head_dim)}, or \code{NULL}.
    value = NULL,

    #' @description Create a new (possibly empty) cache entry.
    #' @param key `tensor` or \code{NULL}.
    #' @param value `tensor` or \code{NULL}.
    initialize = function(key = NULL, value = NULL) {
      self$key   <- key
      self$value <- value
    },

    #' @description Check whether this entry contains valid (non-NULL)
    #'   key and value tensors.
    #' @return `logical(1)`
    is_valid = function() {
      !is.null(self$key) && !is.null(self$value)
    },

    #' @description Slice key and value along the batch dimension
    #'   (dim 1 in R, dim 0 in Python).
    #' @param start `integer(1)` 1-indexed start position (inclusive).
    #' @param end `integer(1)` 1-indexed end position (inclusive).
    #' @return A new \code{KVCacheEntry} with sliced tensors, or an
    #'   empty entry if this entry is not valid.
    subset = function(start, end) {
      if (!self$is_valid()) {
        return(KVCacheEntry$new())
      }
      KVCacheEntry$new(
        key   = self$key[start:end, ..],
        value = self$value[start:end, ..]
      )
    },

    #' @description Write a batch slice into this entry (in-place).
    #' @param start `integer(1)` 1-indexed start position (inclusive).
    #' @param end `integer(1)` 1-indexed end position (inclusive).
    #' @param other A valid \code{KVCacheEntry} to copy from.
    assign = function(start, end, other) {
      if (other$is_valid() && self$is_valid()) {
        self$key[start:end, ..]   <- other$key
        self$value[start:end, ..] <- other$value
      }
    },

    #' @description Move this entry to a device and optionally cast dtype.
    #' @param device Target device (e.g. \code{"cpu"}, \code{"cuda:0"}).
    #' @param dtype Target torch dtype, or \code{NULL} to preserve.
    #' @return A new \code{KVCacheEntry}.
    to = function(device, dtype = NULL) {
      if (!self$is_valid()) {
        return(KVCacheEntry$new())
      }
      KVCacheEntry$new(
        key   = self$key$to(device = device, dtype = dtype),
        value = self$value$to(device = device, dtype = dtype)
      )
    }
  )
)


#' Concatenate Multiple KVCacheEntry Objects
#'
#' @description Concatenate the key and value tensors of several
#'   \code{\link{KVCacheEntry}} objects along a given dimension.
#'
#' @param entries `list` of \code{KVCacheEntry} objects.
#' @param dim `integer(1)` Dimension to concatenate along (batch
#'   dimension).  Default: \code{1L}.
#'
#' @return A new \code{KVCacheEntry} with concatenated tensors, or
#'   an empty entry if no valid entries are provided.
#'
#' @seealso \code{\link{KVCacheEntry}}
#' @export
kv_cache_entry_concat <- function(entries, dim = 1L) {
  keys   <- lapply(entries, function(e) e$key)[
    vapply(entries, function(e) e$is_valid(), logical(1L))
  ]
  values <- lapply(entries, function(e) e$value)[
    vapply(entries, function(e) e$is_valid(), logical(1L))
  ]
  if (length(keys) == 0L) {
    return(KVCacheEntry$new())
  }
  KVCacheEntry$new(
    key   = torch_cat(keys,   dim = dim),
    value = torch_cat(values, dim = dim)
  )
}

#' KVCache -- Per-Layer Key-Value Cache
#'
#' @description An R6 class that maps layer/block indices
#'   (\code{integer}) to \code{\link{KVCacheEntry}} objects.
#'   Supports slicing, device transfer, concatenation and
#'   pre-allocation.
#'
#' @export
KVCache <- R6::R6Class(
  classname = "KVCache",

  public = list(

    #' @field kv Named \code{list} mapping layer indices (as
    #'   character strings) to \code{KVCacheEntry} objects.
    kv = list(),

    #' @description Create a new (empty) KVCache.
    #' @param kv Named list of \code{KVCacheEntry} objects.
    initialize = function(kv = list()) {
      self$kv <- kv
    },

    #' @description Check whether any layer entry contains valid data.
    #' @return `logical(1)`
    is_populated = function() {
      if (length(self$kv) == 0L) return(FALSE)
      any(vapply(self$kv, function(e) e$is_valid(), logical(1L)))
    },

    #' @description Slice all entries along the batch dimension.
    #' @param start `integer(1)` 1-indexed start (inclusive).
    #' @param end `integer(1)` 1-indexed end (inclusive).
    #' @return A new \code{KVCache} with sliced entries.
    subset = function(start, end) {
      sliced_kv <- lapply(self$kv, function(entry) {
        entry$subset(start, end)
      })
      KVCache$new(kv = sliced_kv)
    },

    #' @description Write batch-sliced entries into this pre-allocated
    #'   cache (in-place).
    #' @param start `integer(1)` 1-indexed start (inclusive).
    #' @param end `integer(1)` 1-indexed end (inclusive).
    #' @param other A \code{KVCache} whose entries are copied in.
    assign = function(start, end, other) {
      for (idx in names(other$kv)) {
        if (idx %in% names(self$kv)) {
          target <- self$kv[[idx]]
          if (!target$is_valid()) {
            value_error(
              "Cannot write to cache index {idx} because it is not valid."
            )
          }
          target$assign(
            start, end,
            other$kv[[idx]]$to(
              device = target$key$device,
              dtype  = target$key$dtype
            )
          )
        }
      }
    },

    #' @description Move all entries to a device and optionally cast dtype.
    #' @param device Target device.
    #' @param dtype Target torch dtype, or \code{NULL}.
    #' @return A new \code{KVCache}.
    to = function(device, dtype = NULL) {
      moved_kv <- lapply(self$kv, function(entry) {
        entry$to(device, dtype)
      })
      KVCache$new(kv = moved_kv)
    },

    #' @description Pre-allocate K/V tensors based on shapes from a
    #'   reference cache.  The last three dimensions are taken from the
    #'   reference entry; the leading dimensions are set to
    #'   \code{batch_shape}.
    #'
    #' @param reference A \code{KVCache} from a single batch whose
    #'   entry shapes serve as a template.
    #' @param batch_shape `integer` vector for the leading dimensions.
    #' @param device Device string (default \code{"cpu"}).
    #' @param dtype Target dtype, or \code{NULL} to use reference dtype.
    preallocate = function(reference, batch_shape, device = "cpu", dtype = NULL) {
      for (idx in names(reference$kv)) {
        ref_entry <- reference$kv[[idx]]
        if (ref_entry$is_valid()) {
          target_dtype <- if (!is.null(dtype)) dtype else ref_entry$key$dtype
          ref_size <- ref_entry$key$size()
          # Keep last 3 dims: (num_heads, seq_len, head_dim)
          tail_dims <- ref_size[(length(ref_size) - 2L):length(ref_size)]
          key_shape   <- c(batch_shape, tail_dims)
          value_shape <- c(batch_shape, tail_dims)
          self$kv[[idx]] <- KVCacheEntry$new(
            key   = do.call(
              torch_zeros,
              c(as.list(key_shape),   list(dtype = target_dtype, device = device))
            ),
            value = do.call(
              torch_zeros,
              c(as.list(value_shape), list(dtype = target_dtype, device = device))
            )
          )
        }
      }
    }
  )
)


#' Concatenate Multiple KVCache Objects
#'
#' @description Concatenate entries across several
#'   \code{\link{KVCache}} objects at each shared layer index.
#'
#' @param caches `list` of \code{KVCache} objects.
#' @param dim `integer(1)` Dimension to concatenate along
#'   (batch dimension).  Default: \code{1L}.
#'
#' @return A new \code{KVCache}.
#'
#' @seealso \code{\link{KVCache}}, \code{\link{KVCacheEntry}}
#' @export
kv_cache_concat <- function(caches, dim = 1L) {
  all_indices <- character(0L)
  for (cache in caches) {
    all_indices <- union(all_indices, names(cache$kv))
  }
  merged_kv <- list()
  for (idx in sort(all_indices)) {
    entries <- lapply(
      caches[.caches_has_index(caches, idx)],
      function(c) c$kv[[idx]]
    )
    merged_kv[[idx]] <- kv_cache_entry_concat(entries, dim = dim)
  }
  KVCache$new(kv = merged_kv)
}


#' @keywords internal
.caches_has_index <- function(caches, idx) {
  vapply(caches, function(c) idx %in% names(c$kv), logical(1L))
}


#' TabICLCache -- Top-Level Cache for the TabICL Model
#'
#' @description Aggregates caches for the three major TabICL
#'   components:
#'   \itemize{
#'     \item \strong{ColEmbedding} cache (ISAB blocks in column
#'           embedding).
#'     \item \strong{ICLearning} cache (Encoder layers in the ICL
#'           transformer).
#'     \item \strong{Row representations} cached as a single tensor.
#'   }
#'
#' @export
TabICLCache <- R6::R6Class(
  classname = "TabICLCache",

  public = list(

    #' @field col_cache A \code{\link{KVCache}} for ColEmbedding.
    col_cache = NULL,

    #' @field row_repr Cached row representations, or \code{NULL}.
    row_repr = NULL,

    #' @field icl_cache A \code{\link{KVCache}} for ICLearning.
    icl_cache = NULL,

    #' @field train_shape `integer(3)` \code{c(batch_size, train_size,
    #'   num_features)} describing the training data shape the cache
    #'   was built with.
    train_shape = c(0L, 0L, 0L),

    #' @field num_classes `integer(1)` or \code{NULL}. Number of
    #'   classes (0 or \code{NULL} for regression).
    num_classes = NULL,

    #' @description Create a new TabICLCache.
    #' @param col_cache A \code{KVCache}, or \code{NULL} (default:
    #'   empty \code{KVCache}).
    #' @param row_repr `tensor` or \code{NULL}.
    #' @param icl_cache A \code{KVCache}, or \code{NULL} (default:
    #'   empty \code{KVCache}).
    #' @param train_shape `integer(3)`.
    #' @param num_classes `integer(1)` or \code{NULL}.
    initialize = function(
      col_cache   = NULL,
      row_repr    = NULL,
      icl_cache   = NULL,
      train_shape = c(0L, 0L, 0L),
      num_classes = NULL
    ) {
      self$col_cache   <- if (is.null(col_cache)) KVCache$new() else col_cache
      self$row_repr    <- row_repr
      self$icl_cache   <- if (is.null(icl_cache)) KVCache$new() else icl_cache
      self$train_shape <- train_shape
      self$num_classes <- num_classes
    },

    #' @description Return the cache type: \code{"kv"}, \code{"repr"},
    #'   or \code{"empty"}.
    #' @return `character(1)`
    cache_type = function() {
      if (!is.null(self$row_repr)) {
        return("repr")
      }
      col_populated <- !is.null(self$col_cache) && length(self$col_cache$kv) > 0L
      icl_populated <- !is.null(self$icl_cache) && length(self$icl_cache$kv) > 0L
      if (col_populated || icl_populated) {
        return("kv")
      }
      "empty"
    },

    #' @description Return the approximate memory occupied by cached
    #'   tensors in megabytes.
    #' @return `numeric(1)`
    cache_size_mb = function() {
      total <- 0L
      if (!is.null(self$col_cache)) {
        for (kv in self$col_cache$kv) {
          if (!is.null(kv$key))   total <- total + .numel(kv$key)   * .element_size(kv$key)
          if (!is.null(kv$value)) total <- total + .numel(kv$value) * .element_size(kv$value)
        }
      }
      if (!is.null(self$row_repr)) {
        total <- total + .numel(self$row_repr) * .element_size(self$row_repr)
      }
      if (!is.null(self$icl_cache)) {
        for (kv in self$icl_cache$kv) {
          if (!is.null(kv$key))   total <- total + .numel(kv$key)   * .element_size(kv$key)
          if (!is.null(kv$value)) total <- total + .numel(kv$value) * .element_size(kv$value)
        }
      }
      total %/% (1024L * 1024L)
    },

    #' @description Check whether this cache is empty (no col_cache
    #'   entries, no row_repr, no icl_cache entries).
    #' @return `logical(1)`
    is_empty = function() {
      col_empty <- is.null(self$col_cache) || length(self$col_cache$kv) == 0L
      row_empty <- is.null(self$row_repr)
      icl_empty <- is.null(self$icl_cache) || length(self$icl_cache$kv) == 0L
      col_empty && row_empty && icl_empty
    },

    #' @description Slice this cache along the batch dimension.
    #' @param start `integer(1)` 1-indexed start (inclusive).
    #' @param end `integer(1)` 1-indexed end (inclusive).
    #' @return A new \code{TabICLCache} with sliced tensors.
    slice_batch = function(start, end) {
      TabICLCache$new(
        col_cache   = if (!is.null(self$col_cache) && length(self$col_cache$kv) > 0L) {
          self$col_cache$subset(start, end)
        } else {
          KVCache$new()
        },
        row_repr    = if (!is.null(self$row_repr)) self$row_repr[start:end, ..] else NULL,
        icl_cache   = if (!is.null(self$icl_cache) && length(self$icl_cache$kv) > 0L) {
          self$icl_cache$subset(start, end)
        } else {
          KVCache$new()
        },
        train_shape = c(end - start + 1L, self$train_shape[2L], self$train_shape[3L]),
        num_classes = self$num_classes
      )
    },

    #' @description Move all cached tensors to a device and
    #'   optionally cast dtype.
    #' @param device Target device.
    #' @param dtype Target torch dtype, or \code{NULL}.
    #' @return A new \code{TabICLCache}.
    to = function(device, dtype = NULL) {
      TabICLCache$new(
        col_cache   = if (!is.null(self$col_cache) && length(self$col_cache$kv) > 0L) {
          self$col_cache$to(device, dtype)
        } else {
          KVCache$new()
        },
        row_repr    = if (!is.null(self$row_repr)) {
          self$row_repr$to(device = device, dtype = dtype)
        } else {
          NULL
        },
        icl_cache   = if (!is.null(self$icl_cache) && length(self$icl_cache$kv) > 0L) {
          self$icl_cache$to(device, dtype)
        } else {
          KVCache$new()
        },
        train_shape = self$train_shape,
        num_classes = self$num_classes
      )
    }
  )
)


#' Concatenate Multiple TabICLCache Objects
#'
#' @description Concatenate the \code{col_cache}, \code{row_repr},
#'   and \code{icl_cache} of several \code{\link{TabICLCache}}
#'   objects along the batch dimension.
#'
#' @param caches `list` of \code{TabICLCache} objects.
#' @param dim `integer(1)` Dimension to concatenate along
#'   (batch dimension).  Default: \code{1L}.
#'
#' @return A new \code{TabICLCache}.
#'
#' @seealso \code{\link{TabICLCache}}, \code{\link{KVCache}}
#' @export
tabicl_cache_concat <- function(caches, dim = 1L) {
  col_caches <- Filter(function(c) {
    !is.null(c$col_cache) && length(c$col_cache$kv) > 0L
  }, caches)
  row_reprs <- Filter(function(c) !is.null(c$row_repr), caches)
  icl_caches <- Filter(function(c) {
    !is.null(c$icl_cache) && length(c$icl_cache$kv) > 0L
  }, caches)

  total_batch <- sum(vapply(caches, function(c) c$train_shape[1L], integer(1L)))
  train_size  <- caches[[1L]]$train_shape[2L]
  n_features  <- caches[[1L]]$train_shape[3L]

  TabICLCache$new(
    col_cache = if (length(col_caches) > 0L) {
      kv_cache_concat(lapply(col_caches, function(c) c$col_cache), dim = dim)
    } else {
      KVCache$new()
    },
    row_repr  = if (length(row_reprs) > 0L) {
      torch_cat(lapply(row_reprs, function(c) c$row_repr), dim = dim)
    } else {
      NULL
    },
    icl_cache = if (length(icl_caches) > 0L) {
      kv_cache_concat(lapply(icl_caches, function(c) c$icl_cache), dim = dim)
    } else {
      KVCache$new()
    },
    train_shape = c(total_batch, train_size, n_features),
    num_classes = caches[[1L]]$num_classes
  )
}
