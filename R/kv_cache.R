#' TabICL Key-Value Cache Structures
#'
#' @description
#' Cache data structures for storing key-value projections from attention layers,
#' enabling efficient inference by reusing computed values across test samples.
#'
#' The caching strategy focuses on:
#' \enumerate{
#'   \item ColEmbedding: Cache K/V of the second attention layer of ISAB blocks
#'   \item ICLearning: Cache K/V from training data at each layer of the ICL transformer
#' }
#'
#' @docType package
#' @name tabicl-cache
#' @keywords internal
#' @importFrom R6 R6Class
"_PACKAGE"

#' A Single Key-Value Cache Entry for an Attention Layer
#'
#' @description
#' Stores cached key and value projections from an attention layer.
#'
#' @details
#' Key and value tensors have shape \code{(batch, num_heads, seq_len, head_dim)}.
#'
#' @format An [R6::R6Class] object.
#'
#' @examples
#' \dontrun{
#'   entry <- KVCacheEntry$new(
#'     key = torch_randn(2, 4, 10, 64),
#'     value = torch_randn(2, 4, 10, 64)
#'   )
#'   entry$is_valid()
#' }
#'
#' @export
KVCacheEntry <- R6::R6Class(
  classname = "KVCacheEntry",

  public = list(

    #' @field key Cached key projections. A \code{torch_tensor} or \code{NULL}.
    key = NULL,

    #' @field value Cached value projections. A \code{torch_tensor} or \code{NULL}.
    value = NULL,

    #' @description
    #' Create a new KVCacheEntry.
    #' @param key Optional \code{torch_tensor}.
    #' @param value Optional \code{torch_tensor}.
    initialize = function(key = NULL, value = NULL) {
      self$key <- key
      self$value <- value
    },

    #' @description
    #' Check if this cache entry contains valid data.
    #' @return Logical scalar.
    is_valid = function() {
      !is.null(self$key) && !is.null(self$value)
    },

    #' @description
    #' Slice key/value along batch dimensions.
    #'
    #' Returns a new KVCacheEntry with sliced tensors, or an empty entry
    #' if this entry is not valid.
    #'
    #' @param drop drop unitary dimension
    #' @param indices Indices for slicing. Can be an integer vector, a
    #'   \code{torch_tensor}, or use \code{..} for all elements.
    #' @return A new \code{KVCacheEntry} instance.
    slice = function(indices, drop=FALSE) {
      if (!self$is_valid()) {
        return(KVCacheEntry$new())
      }
      KVCacheEntry$new(
        key = self$key[indices, drop=drop],
        value = self$value[indices, drop=drop]
      )
    },

    #' @description
    #' Write a batch slice into this entry.
    #'
    #' @param indices Indices for writing.
    #' @param other A \code{KVCacheEntry} instance to write from.
    #' @return Invisible self (for method chaining).
    write = function(indices, other) {
      if (other$is_valid() && self$is_valid()) {
        self$key[indices] <- other$key
        self$value[indices] <- other$value
      }
      invisible(self)
    },

    #' @description
    #' Move this entry to the given device and optionally cast dtype.
    #'
    #' Returns a new KVCacheEntry.
    #'
    #' @param device Device string, e.g. \code{"cpu"} or \code{"cuda"}.
    #' @param dtype Optional \code{torch_dtype}.
    #' @return A new \code{KVCacheEntry} instance.
    to = function(device, dtype = NULL) {
      if (!self$is_valid()) {
        return(KVCacheEntry$new())
      }
      KVCacheEntry$new(
        key = self$key$to(device = device, dtype = dtype),
        value = self$value$to(device = device, dtype = dtype)
      )
    }
  )
)


#' Concatenate Multiple KVCacheEntry Objects
#'
#' @description
#' Concatenate multiple \code{KVCacheEntry} objects along a dimension.
#'
#' @param entries List of \code{KVCacheEntry} instances. All valid entries
#'   are concatenated; invalid ones are silently skipped.
#' @param dim Integer. Dimension to concatenate along. Default is \code{1}
#'   (batch dimension in R 1-based indexing; corresponds to Python dim 0).
#' @return A new \code{KVCacheEntry} with concatenated key and value tensors.
#'   Returns an empty entry if no valid entries are found.
#'
#' @examples
#' \dontrun{
#'   e1 <- KVCacheEntry$new(key = torch_randn(2, 4, 10, 64),
#'                          value = torch_randn(2, 4, 10, 64))
#'   e2 <- KVCacheEntry$new(key = torch_randn(3, 4, 10, 64),
#'                          value = torch_randn(3, 4, 10, 64))
#'   e_cat <- kv_cache_entry_concat(list(e1, e2), dim = 1)
#' }
#'
#' @export
kv_cache_entry_concat <- function(entries, dim = 1) {
  keys <- lapply(entries, function(e) {
    if (e$is_valid()) e$key else NULL
  })
  values <- lapply(entries, function(e) {
    if (e$is_valid()) e$value else NULL
  })

  keys <- Filter(Negate(is.null), keys)
  values <- Filter(Negate(is.null), values)

  if (length(keys) == 0) {
    return(KVCacheEntry$new())
  }

  KVCacheEntry$new(
    key = torch_cat(keys, dim = dim),
    value = torch_cat(values, dim = dim)
  )
}


#' Base Class for Key-Value Caches
#'
#' @description
#' Provides common structure and operations for caches that store
#' mappings from layer/block index to cached key-value projections.
#'
#' @details
#' Layer indices are stored as character strings in the named list \code{kv},
#' since R lists cannot have integer names that start from 0.
#'
#' @format An [R6::R6Class] object.
#'
#' @examples
#' \dontrun{
#'   cache <- KVCache$new()
#'   cache$kv[["1"]] <- KVCacheEntry$new(
#'     key = torch_randn(2, 4, 10, 64),
#'     value = torch_randn(2, 4, 10, 64)
#'   )
#'   cache$is_populated()
#' }
#'
#' @export
KVCache <- R6::R6Class(
  classname = "KVCache",

  public = list(

    #' @field kv Named list mapping layer index (character) to \code{KVCacheEntry}.
    kv = list(),

    #' @description
    #' Create a new KVCache.
    #' @param kv Optional named list of \code{KVCacheEntry} instances.
    initialize = function(kv = list()) {
      self$kv <- kv
    },

    #' @description
    #' Check if this cache has valid entries.
    #'
    #' Returns \code{TRUE} when the cache contains data (use_cache mode).
    #' Returns \code{FALSE} when the cache is empty (store_cache mode).
    #'
    #' @return Logical scalar.
    is_populated = function() {
      if (length(self$kv) == 0) return(FALSE)
      any(vapply(self$kv, function(entry) entry$is_valid(), logical(1)))
    },

    #' @description
    #' Slice all entries along batch dimensions.
    #'
    #' Returns a new cache of the same class with sliced entries.
    #'
    #' @param indices Indices for slicing.
    #' @return A new \code{KVCache} instance.
    slice = function(indices) {
      sliced_kv <- lapply(self$kv, function(entry) entry$slice(indices))
      KVCache$new(kv = sliced_kv)
    },

    #' @description
    #' Write batch-sliced entries into this pre-allocated cache.
    #'
    #' @param indices Indices for writing.
    #' @param other A \code{KVCache} instance to write from.
    #' @return Invisible self (for method chaining).
    write = function(indices, other) {
      for (idx in names(other$kv)) {
        if (idx %in% names(self$kv)) {
          target <- self$kv[[idx]]
          if (!target$is_valid()) {
            stop("Cannot write to cache index ", idx, " because it is not valid.")
          }
          other_entry <- other$kv[[idx]]
          target$write(indices, other_entry$to(
            device = target$key$device,
            dtype = target$key$dtype
          ))
        }
      }
      invisible(self)
    },

    #' @description
    #' Move all entries to the given device and optionally cast dtype.
    #'
    #' Returns a new cache of the same class.
    #'
    #' @param device Device string.
    #' @param dtype Optional \code{torch_dtype}.
    #' @return A new \code{KVCache} instance.
    to = function(device, dtype = NULL) {
      moved_kv <- lapply(self$kv, function(entry) entry$to(device, dtype))
      KVCache$new(kv = moved_kv)
    },

    #' @description
    #' Pre-allocate entries in this cache based on shapes from a reference.
    #'
    #' K/V tensors always have shape \code{(*batch, num_heads, seq_len, head_dim)}.
    #' This method keeps the last three dimensions from the reference entry and
    #' prepends \code{batch_shape} as the leading dimensions.
    #'
    #' @param reference A \code{KVCache} from a single batch whose entry shapes
    #'   are used as a template.
    #' @param batch_shape Integer vector. The full batch shape for leading dims.
    #' @param device Device on which to allocate tensors. Default \code{"cpu"}.
    #' @param dtype Optional \code{torch_dtype}. If \code{NULL}, uses reference
    #'   entry dtype.
    #' @return Invisible self (for method chaining).
    preallocate = function(reference, batch_shape, device = "cpu", dtype = NULL) {
      for (idx in names(reference$kv)) {
        ref_entry <- reference$kv[[idx]]
        if (ref_entry$is_valid()) {
          target_dtype <- if (is.null(dtype)) ref_entry$key$dtype else dtype

          ref_key_shape <- ref_entry$key$shape
          ref_value_shape <- ref_entry$value$shape

          n_dims <- length(ref_key_shape)
          # Keep last 3 dimensions (same in R and Python for trailing dims)
          trailing_dims <- ref_key_shape[(n_dims - 2):n_dims]
          key_shape <- c(batch_shape, trailing_dims)

          trailing_dims_v <- ref_value_shape[(n_dims - 2):n_dims]
          value_shape <- c(batch_shape, trailing_dims_v)

          self$kv[[idx]] <- KVCacheEntry$new(
            key = torch_zeros(key_shape, dtype = target_dtype, device = device),
            value = torch_zeros(value_shape, dtype = target_dtype, device = device)
          )
        }
      }
      invisible(self)
    }
  )
)


#' Concatenate Multiple KVCache Objects
#'
#' @description
#' Concatenate multiple \code{KVCache} objects along a dimension.
#'
#' @param caches List of \code{KVCache} instances. All must have the same
#'   layer indices.
#' @param dim Integer. Dimension to concatenate along (batch dimension).
#'   Default is \code{1} (R 1-based; corresponds to Python dim 0).
#' @return A new \code{KVCache} with concatenated entries at each layer index.
#'
#' @export
kv_cache_concat <- function(caches, dim = 1) {
  all_indices <- unique(unlist(lapply(caches, function(c) names(c$kv))))
  merged_kv <- list()

  for (idx in sort(all_indices)) {
    entries <- lapply(caches, function(c) {
      if (idx %in% names(c$kv)) c$kv[[idx]] else NULL
    })
    entries <- Filter(Negate(is.null), entries)
    if (length(entries) > 0) {
      merged_kv[[idx]] <- kv_cache_entry_concat(entries, dim = dim)
    }
  }

  KVCache$new(kv = merged_kv)
}



#' Top-Level Cache Container for the Entire TabICL Model
#'
#' @description
#' Aggregates caches for different components of TabICL:
#' \itemize{
#'   \item ColEmbedding cache (for ISAB blocks in column embedding)
#'   \item ICLearning cache (for Encoder layers in the ICL transformer)
#' }
#'
#' @details
#' The \code{train_shape} field stores \code{(batch_size, train_size, num_features)}
#' as a 3-element integer vector.
#'
#' @format An [R6::R6Class] object.
#'
#' @examples
#' \dontrun{
#'   cache <- TabICLCache$new(
#'     train_shape = c(4, 100, 20),
#'     num_classes = 10
#'   )
#'   cache$cache_type()
#' }
#'
#' @export
TabICLCache <- R6::R6Class(
  classname = "TabICLCache",

  public = list(

    #' @field col_cache Optional \code{KVCache} for ColEmbedding ISAB blocks.
    col_cache = NULL,

    #' @field row_repr Optional \code{torch_tensor}. Cached row representations.
    row_repr = NULL,

    #' @field icl_cache Optional \code{KVCache} for ICLearning Encoder layers.
    icl_cache = NULL,

    #' @field train_shape Integer vector of length 3:
    #'   \code{(batch_size, train_size, num_features)}.
    train_shape = c(0, 0, 0),

    #' @field num_classes Optional integer. Number of classes in classification
    #'   tasks (0 for regression). Stored when caching to ensure consistent
    #'   output shape during cache use.
    num_classes = NULL,

    #' @description
    #' Create a new TabICLCache.
    #'
    #' Initializes sub-caches automatically if not provided.
    #'
    #' @param col_cache Optional \code{KVCache}.
    #' @param row_repr Optional \code{torch_tensor}.
    #' @param icl_cache Optional \code{KVCache}.
    #' @param train_shape Integer vector of length 3. Default \code{c(0, 0, 0)}.
    #' @param num_classes Optional integer.
    initialize = function(col_cache = NULL, row_repr = NULL,
                          icl_cache = NULL,
                          train_shape = c(0, 0, 0),
                          num_classes = NULL) {
      self$col_cache <- if (is.null(col_cache)) KVCache$new() else col_cache
      self$row_repr <- row_repr
      self$icl_cache <- if (is.null(icl_cache)) KVCache$new() else icl_cache
      self$train_shape <- train_shape
      self$num_classes <- num_classes
    },

    #' @description
    #' Return the cache type.
    #'
    #' @return Character string: \code{"kv"}, \code{"repr"}, or \code{"empty"}.
    cache_type = function() {
      if (!is.null(self$row_repr)) {
        return("repr")
      }
      col_populated <- !is.null(self$col_cache) && self$col_cache$is_populated()
      icl_populated <- !is.null(self$icl_cache) && self$icl_cache$is_populated()
      if (col_populated || icl_populated) {
        return("kv")
      }
      "empty"
    },

    #' @description
    #' Return the memory occupied by cached tensors in MB.
    #'
    #' @return Integer.
    cache_size_mb = function() {
      total <- 0

      # Count memory from ColEmbedding
      if (!is.null(self$col_cache)) {
        for (kv in self$col_cache$kv) {
          if (!is.null(kv$key)) {
            total <- total + as.numeric(kv$key$numel()) * kv$key$element_size()
          }
          if (!is.null(kv$value)) {
            total <- total + as.numeric(kv$value$numel()) * kv$value$element_size()
          }
        }
      }

      # Count memory from row representations
      if (!is.null(self$row_repr)) {
        total <- total + as.numeric(self$row_repr$numel()) * self$row_repr$element_size()
      }

      # Count memory from ICLearning
      if (!is.null(self$icl_cache)) {
        for (kv in self$icl_cache$kv) {
          if (!is.null(kv$key)) {
            total <- total + as.numeric(kv$key$numel()) * kv$key$element_size()
          }
          if (!is.null(kv$value)) {
            total <- total + as.numeric(kv$value$numel()) * kv$value$element_size()
          }
        }
      }

      as.integer(total %/% (1024 * 1024))
    },

    #' @description
    #' Check if the cache is empty.
    #'
    #' @return Logical scalar.
    is_empty = function() {
      col_empty <- is.null(self$col_cache) || !self$col_cache$is_populated()
      row_empty <- is.null(self$row_repr)
      icl_empty <- is.null(self$icl_cache) || !self$icl_cache$is_populated()

      col_empty && row_empty && icl_empty
    },

    #' @description
    #' Slice this cache along the batch dimension (dim 1 in R).
    #'
    #' @param start Integer. Start index of the batch slice (1-based, inclusive).
    #' @param end Integer. End index of the batch slice (1-based, exclusive).
    #' @return A new \code{TabICLCache} with sliced tensors (views of original).
    slice_batch = function(start, end) {
      # R uses 1-based indexing; Python slice(start, end) -> R[start:(end-1)]
      # torch indexing in R: tensor[start:(end-1)] works correctly
      indices <- start:(end - 1)

      TabICLCache$new(
        col_cache = if (!is.null(self$col_cache)) {
          self$col_cache$slice(indices)
        } else {
          KVCache$new()
        },
        row_repr = if (!is.null(self$row_repr)) {
          self$row_repr[indices]
        } else {
          NULL
        },
        icl_cache = if (!is.null(self$icl_cache)) {
          self$icl_cache$slice(indices)
        } else {
          KVCache$new()
        },
        train_shape = c(end - start, self$train_shape[2], self$train_shape[3]),
        num_classes = self$num_classes
      )
    },

    #' @description
    #' Move all cached tensors to the given device and optionally cast dtype.
    #'
    #' @param device Device string, e.g. \code{"cpu"} or \code{"cuda:0"}.
    #' @param dtype Optional \code{torch_dtype}. If \code{NULL}, preserves
    #'   existing dtype.
    #' @return A new \code{TabICLCache} with all tensors on the target device.
    to = function(device, dtype = NULL) {
      TabICLCache$new(
        col_cache = if (!is.null(self$col_cache)) {
          self$col_cache$to(device, dtype)
        } else {
          KVCache$new()
        },
        row_repr = if (!is.null(self$row_repr)) {
          self$row_repr$to(device = device, dtype = dtype)
        } else {
          NULL
        },
        icl_cache = if (!is.null(self$icl_cache)) {
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
#' @description
#' Concatenate multiple \code{TabICLCache} objects along the batch dimension.
#'
#' @param caches List of \code{TabICLCache} instances.
#' @param dim Integer. Dimension to concatenate along (batch dimension).
#'   Default is \code{1} (R 1-based; corresponds to Python dim 0).
#' @return A new \code{TabICLCache} with concatenated caches.
#'
#' @export
tabicl_cache_concat <- function(caches, dim = 1) {
  col_caches <- lapply(caches, function(c) c$col_cache)
  col_caches <- Filter(Negate(is.null), col_caches)

  row_reprs <- lapply(caches, function(c) c$row_repr)
  row_reprs <- Filter(Negate(is.null), row_reprs)

  icl_caches <- lapply(caches, function(c) c$icl_cache)
  icl_caches <- Filter(Negate(is.null), icl_caches)

  if (length(caches) == 0) {
    return(TabICLCache$new())
  }
  total_batch <- as.integer(sum(vapply(caches, function(c) as.integer(c$train_shape[1]), integer(1))))
  train_size <- caches[[1]]$train_shape[2]
  n_features <- caches[[1]]$train_shape[3]

  TabICLCache$new(
    col_cache = if (length(col_caches) > 0) {
      kv_cache_concat(col_caches, dim = dim)
    } else {
      KVCache$new()
    },
    row_repr = if (length(row_reprs) > 0) {
      torch_cat(row_reprs, dim = dim)
    } else {
      NULL
    },
    icl_cache = if (length(icl_caches) > 0) {
      kv_cache_concat(icl_caches, dim = dim)
    } else {
      KVCache$new()
    },
    train_shape = c(total_batch, train_size, n_features),
    num_classes = caches[[1]]$num_classes
  )
}
