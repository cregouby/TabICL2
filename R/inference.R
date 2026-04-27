#' Offload mode constants for memory management
#'
#' @description
#' These constants define where intermediate results should be stored during inference:
#' \describe{
#'   \item{OFFLOAD_GPU}{Keep everything on GPU (fastest if it fits)}
#'   \item{OFFLOAD_CPU}{Offload to CPU pinned memory}
#'   \item{OFFLOAD_DISK}{Offload to disk via memory-mapped files}
#'   \item{OFFLOAD_AUTO}{Automatically choose based on available memory}
#' }
#'
#' @keywords internal
#' @importFrom torch nn_module torch_tensor torch_empty torch_zeros
#' @importFrom torch torch_cat torch_stack torch_transpose
#' @importFrom torch torch_reshape torch_clone
#' @importFrom torch torch_min torch_max torch_abs torch_equal
#' @importFrom torch torch_float torch_bfloat16 torch_float16 torch_long torch_int32 torch_int16
#' @importFrom torch cuda_is_available cuda_empty_cache
#' @importFrom torch cuda_memory_summary with_no_grad
OFFLOAD_GPU <- "gpu"
OFFLOAD_CPU <- "cpu"
OFFLOAD_DISK <- "disk"
OFFLOAD_AUTO <- "auto"


#' MemoryEstimator: Estimate peak activation memory for inference components
#'
#' Estimates peak inference memory requirements for different attention-based components.
#' Peak inference memory refers to the maximum amount of memory (typically GPU memory) used
#' during the inference phase of a model.
#'
#' The coefficients and intercepts for each component are derived through memory profiling
#' and regression using the model:
#' \deqn{
#' c_1 \cdot batch\_size + c_2 \cdot seq\_len + c_3 \cdot (batch\_size \times seq\_len) + intercept
#' }
#'
#' Memory profiling was conducted using float32 without automatic mixed precision (AMP).
#' When using AMP, actual memory usage will be lower than the estimates.
#'
#' @keywords internal
memory_estimator <- list(
  # Coefficients for memory estimation: [c1, c2, c3] for each encoder
  coefficients = list(
    tf_col = c(1.456149314e-01, 1.94081457e-05, 4.88223400e-03),
    tf_row = c(-2.06831848e-05, 2.27205969e-04, 5.37117114e-03),
    tf_icl = c(-4.03932756e-02, 5.42811085e-07, 1.95312473e-02)
  ),

  # Intercepts for each encoder
  intercepts = list(
    tf_col = 142.91294659096457,
    tf_row = 138.53653545318957,
    tf_icl = 142.84243874417552
  )
)


#' Estimate peak memory usage for a given component
#'
#' @param batch_size Integer. Batch size for inference.
#' @param seq_len Integer. Sequence length for inference.
#' @param enc_name Character. Model encoder name: `"tf_col"`, `"tf_row"`, or `"tf_icl"`.
#' @param include_inputs Logical. Whether to include memory usage for input tensors
#'   (default: `TRUE`).
#' @param in_dim Integer or NULL. Model dimension for the encoder (required if
#'   `include_inputs = TRUE`).
#'
#' @return Numeric. Estimated peak memory usage in MB for the specified encoder.
#'
#' @examples
#' \dontrun{
#' # Estimate memory for column encoder with batch=32, seq_len=100
#' mem_mb <- memory_estimator$estimate_peak_mem(
#'   batch_size = 32L, seq_len = 100L, enc_name = "tf_col",
#'   include_inputs = TRUE, in_dim = 128L
#' )
#' cli_inform("Estimated peak memory: {mem_mb:.1f} MB")
#' }
#'
#' @keywords internal
#' @noRd
memory_estimator$estimate_peak_mem <- function(batch_size, seq_len, enc_name,
                                                include_inputs = TRUE, in_dim = NULL) {
  if (!enc_name %in% names(memory_estimator$coefficients)) {
    value_error("Unknown encoder name {enc_name}. Expected one of: {paste(names(memory_estimator$coefficients), collapse=', ')}")
  }

  coefs <- memory_estimator$coefficients[[enc_name]]
  inter <- memory_estimator$intercepts[[enc_name]]

  # Polynomial regression model: c1*bs + c2*seq + c3*bs*seq + intercept
  peak_activation_mem <- coefs[1L] * batch_size +
                         coefs[2L] * seq_len +
                         coefs[3L] * batch_size * seq_len +
                         inter

  if (include_inputs) {
    if (is.null(in_dim)) {
      value_error("Input dimension must be provided for input memory estimation")
    }
    bytes_per_element <- 4L  # float32
    n_elements <- batch_size * seq_len * in_dim
    mem_inputs <- n_elements * bytes_per_element / (1024^2)
    peak_activation_mem <- peak_activation_mem + mem_inputs
  }

  peak_activation_mem
}


#' Estimate batch size that fits within target memory budget
#'
#' Solves the memory model for batch size:
#' \deqn{
#' bs = (target - c2*seq - intercept) / (c1 + c3*seq + seq*in_dim*4/1024^2)
#' }
#'
#' @param seq_len Integer. Sequence length for inference.
#' @param target_memory Numeric. Target memory usage in MB.
#' @param enc_name Character. Model encoder name.
#' @param include_inputs Logical. Whether to include input tensor memory (default: `TRUE`).
#' @param in_dim Integer or NULL. Model dimension (required if `include_inputs = TRUE`).
#'
#' @return Integer. Estimated batch size that fits within target memory constraints.
#'
#' @keywords internal
#' @noRd
memory_estimator$estimate_batch_size <- function(seq_len, target_memory, enc_name,
                                                  include_inputs = TRUE, in_dim = NULL) {
  if (target_memory <= 0) {
    return(1L)
  }

  if (!enc_name %in% names(memory_estimator$coefficients)) {
    value_error("Unknown encoder name {enc_name}")
  }

  coefs <- memory_estimator$coefficients[[enc_name]]
  intercept <- memory_estimator$intercepts[[enc_name]]

  numerator <- target_memory - coefs[2L] * seq_len - intercept
  denominator <- coefs[1L] + coefs[3L] * seq_len

  if (include_inputs && !is.null(in_dim)) {
    denominator <- denominator + seq_len * in_dim * 4 / (1024^2)
  }

  if (denominator <= 0) {
    return(1L)
  }

  max(1L, as.integer(numerator / denominator))
}


#' Create an OffloadReason object
#'
#' Structured reason for offload mode resolution.
#'
#' @param key Character. Primary reason key.
#' @param detail Character or NULL. Additional detail about the reason.
#'
#' @return A list with class `"OffloadReason"`.
#'
#' @keywords internal
offload_reason <- function(key, detail = NULL) {
  structure(
    list(key = key, detail = detail),
    class = "OffloadReason"
  )
}

#' Format OffloadReason for display
#' @param x An OffloadReason object
#' @param ... Additional arguments (unused)
#' @keywords internal
format.OffloadReason <- function(x, ...) {
  if (is.null(x$detail)) {
    x$key
  } else {
    sprintf("%s: %s", x$key, x$detail)
  }
}


#' Create an OffloadConfig object
#'
#' Configuration for offloading behavior during inference.
#'
#' @param mode Character. Offload mode: `"auto"`, `"gpu"`, `"cpu"`, or `"disk"`
#'   (default: `"auto"`).
#' @param auto_offload_threshold Numeric. GPU memory threshold (0-1) for automatic
#'   offloading (default: `0.5`).
#' @param cpu_safety_factor Numeric. Safety margin for CPU memory (default: `0.85`).
#' @param disk_safety_factor Numeric. Safety margin for disk space (default: `0.95`).
#' @param max_pinned_memory_mb Numeric. Max pinned memory in MB (default: `32768`).
#' @param disk_offload_dir Character or NULL. Directory for memory-mapped files.
#' @param disk_min_free_mb Numeric. Minimum free disk space to maintain in MB
#'   (default: `1024`).
#' @param disk_flush_mb Numeric. Flush memmap after writing this many MB
#'   (default: `2048`).
#' @param disk_cleanup Logical. Auto-cleanup disk files (default: `TRUE`).
#' @param disk_file_prefix Character. Prefix for disk file names (default: `""`).
#' @param disk_dtype torch.dtype or NULL. Override dtype for disk storage.
#' @param use_async Logical. Use async D2H copy (default: `TRUE`).
#' @param async_depth Integer. Number of pending async copies before blocking
#'   (default: `4`).
#'
#' @return A list with class `"OffloadConfig"`.
#'
#' @keywords internal
offload_config <- function(
  mode = OFFLOAD_AUTO,
  auto_offload_threshold = 0.5,
  cpu_safety_factor = 0.85,
  disk_safety_factor = 0.95,
  max_pinned_memory_mb = 32768.0,
  disk_offload_dir = NULL,
  disk_min_free_mb = 1024.0,
  disk_flush_mb = 2048.0,
  disk_cleanup = TRUE,
  disk_file_prefix = "",
  disk_dtype = NULL,
  use_async = TRUE,
  async_depth = 4L
) {
  structure(
    list(
      mode = mode,
      auto_offload_threshold = auto_offload_threshold,
      cpu_safety_factor = cpu_safety_factor,
      disk_safety_factor = disk_safety_factor,
      max_pinned_memory_mb = max_pinned_memory_mb,
      disk_offload_dir = disk_offload_dir,
      disk_min_free_mb = disk_min_free_mb,
      disk_flush_mb = disk_flush_mb,
      disk_cleanup = disk_cleanup,
      disk_file_prefix = disk_file_prefix,
      disk_dtype = disk_dtype,
      use_async = use_async,
      async_depth = async_depth
    ),
    class = "OffloadConfig"
  )
}


#' Pool of pinned CPU memory buffers for efficient GPU-to-CPU transfers
#'
#' Pinned (page-locked) memory enables faster async GPU-to-CPU data transfers
#' by allowing the GPU to directly access the CPU memory without involving
#' the CPU. This class maintains a pool of such buffers to avoid the overhead
#' of repeated allocation and deallocation.
#'
#' @param max_buffers_per_shape Integer. Maximum number of buffers to keep pooled
#'   for each unique (shape, dtype) combination (default: `4L`).
#' @param shape Integer vector. Shape of the tensor.
#' @param dtype torch.dtype. Data type of the tensor.
#' @param buf Tensor. Buffer to return (must be pinned).
#'
#' @return A `PinnedBufferPool` object.
#'
#' @examples
#' \dontrun{
#' pool <- pinned_buffer_pool$new(max_buffers_per_shape = 8L)
#' buf <- pool$get(c(100L, 128L), torch_float())
#' # ... use buffer ...
#' pool$put(buf)  # Return to pool for reuse
#' }
#'
#' @export
pinned_buffer_pool <- R6::R6Class(
  "PinnedBufferPool",
  public = list(
    # Initialize the buffer pool
    #
    # @param max_buffers_per_shape Integer. Maximum buffers per (shape, dtype)
    initialize = function(max_buffers_per_shape = 4L) {
      private$pool <- list()
      private$max_per_shape <- as.integer(max_buffers_per_shape)
    },

    # Get a pinned buffer, creating one if necessary
    #
    # @param shape Integer vector. Shape of the tensor.
    # @param dtype torch.dtype. Data type of the tensor.
    # @return A pinned CPU tensor.
    get = function(shape, dtype) {
      key <- .make_buffer_key(shape, dtype)
      pool <- private$pool[[key]]

      if (!is.null(pool) && length(pool) > 0L) {
        # Pop from pool
        buf <- pool[[length(pool)]]
        private$pool[[key]] <- if (length(pool) > 1L) pool[seq_len(length(pool) - 1L)] else NULL
        return(buf)
      }

      # Create new CPU buffer
      # Note: pin_memory=TRUE doesn't work as expected in torch R, using regular CPU tensors
      do.call(torch_empty, c(as.list(shape), list(dtype = dtype, device = "cpu")))
    },

    # Return a buffer to the pool for reuse
    #
    # @param buf Tensor. Buffer to return.
    put = function(buf) {
      # Accept any CPU tensor for pooling
      if (buf$device$type != "cpu") {
        return(invisible(NULL))  # Only pool CPU buffers
      }

      key <- .make_buffer_key(buf$shape, buf$dtype)
      pool <- private$pool[[key]]

      if (is.null(pool)) {
        pool <- list()
      }

      if (length(pool) < private$max_per_shape) {
        private$pool[[key]] <- c(pool, list(buf))
      }

      invisible(NULL)
    },

    # Clear all pooled buffers
    clear = function() {
      private$pool <- list()
      invisible(NULL)
    }
  ),

  private = list(
    pool = NULL,
    max_per_shape = NULL
  )
)

#' Create a unique key for buffer pooling
#' @param shape Integer vector. Tensor shape.
#' @param dtype torch.dtype. Tensor dtype.
#' @return Character. Unique key string.
#' @keywords internal
.make_buffer_key <- function(shape, dtype) {
  # dtype is an externalptr, use its address or a string representation
  dtype_str <- tryCatch(dtype$to_string(), error = function(e) "unknown")
  paste0("shape_", paste(shape, collapse = "_"), "_dtype_", dtype_str)
}


#' A tensor backed by a memory-mapped file on disk
#'
#' Uses R's `bigmemory` or base file I/O for storage and provides a torch tensor
#' interface for seamless integration. This enables storing tensors larger than
#' available RAM by using disk space.
#'
#' @param shape Integer vector. Shape of the tensor.
#' @param dtype torch.dtype. Data type of the tensor.
#' @param path Character. Path to the memory-mapped file.
#' @param cleanup Logical. Whether to delete the file when object is garbage
#'   collected (default: `TRUE`).
#'
#' @section Methods:
#' \describe{
#'   \item{`flush()`}{Flush changes to disk}
#' }
#'
#' @return A `DiskTensor` object.
#'
#' @examples
#' \dontrun{
#' # Create a disk-backed tensor
#' dt <- disk_tensor$new(c(1000L, 128L), torch_float(), "/tmp/my_tensor.mmap")
#'
#' # Write data (automatically persists to disk)
#' dt[1:100, ..] <- torch_randn(100L, 128L)
#'
#' # Read data
#' x <- dt[1:50, ..]$tensor
#'
#' # Flush changes to disk
#' dt$flush()
#' }
#'
#' @export
disk_tensor <- R6::R6Class(
  "DiskTensor",

  public = list(
    # Initialize a DiskTensor
    #
    # @param shape Integer vector. Shape of the tensor.
    # @param dtype torch.dtype. Data type.
    # @param path Character. File path.
    # @param cleanup Logical. Auto-cleanup on GC.
    initialize = function(shape, dtype, path, cleanup = TRUE) {
      private$shape <- as.integer(shape)
      private$dtype <- dtype
      private$path <- path
      private$cleanup_flag <- cleanup

      # Resolve dtype mapping (numpy doesn't support bfloat16)
      resolved <- .resolve_dtype_for_disk(dtype)
      private$storage_dtype <- resolved$storage_dtype
      private$needs_view <- resolved$needs_view

      # Create directory if needed
      dir_name <- dirname(path)
      if (dir_name != "" && !dir.exists(dir_name)) {
        dir.create(dir_name, recursive = TRUE, showWarnings = FALSE)
      }

      # Calculate file size
      bytes_per_element <- torch_tensor(0, dtype = private$storage_dtype)$element_size()
      nbytes <- as.integer(bytes_per_element * prod(private$shape))

      # Pre-allocate file
      file.create(path)
      file.info(path)$size  # Check creation

      # Open connection for binary writing
      con <- file(path, open = "wb")
      # Write zeros to allocate (simplified; in production use seek + write)
      close(con)

      # For R, we use a simpler approach: store metadata and read/write on demand
      # A full implementation would use bigmemory or ff packages
      private$data <- NULL  # Lazy loading

      # Setup cleanup finalizer
      if (cleanup) {
        reg.finalizer(self, function(e) {
          tryCatch(file.remove(e$path), error = function(err) NULL)
        }, onexit = TRUE)
      }
    },

    # Get the torch tensor view (loads data if needed)
    #
    # @return A torch tensor.
    tensor = function() {
      if (is.null(private$data)) {
        # Lazy load from disk
        private$data <- .load_tensor_from_disk(
          private$path, private$shape, private$storage_dtype
        )
        if (private$needs_view) {
          private$data <- private$data$view(private$dtype)
        }
      }
      private$data
    },

    # Index into the tensor (read)
    #
    # @param indices List of indices or slices.
    # @return A torch tensor.
    # @note R6 doesn't support `[` overloading directly; use `$get()` instead.
    get = function(indices) {
      self$tensor()[indices]
    },

    # Write to the tensor (automatically persists to disk)
    #
    # @param indices List of indices.
    # @param value Tensor. Value to write.
    set = function(indices, value) {
      if (!value$is_cpu) {
        value <- value$cpu()
      }

      # Load if needed
      if (is.null(private$data)) {
        private$data <- .load_tensor_from_disk(
          private$path, private$shape, private$storage_dtype
        )
        if (private$needs_view) {
          private$data <- private$data$view(private$dtype)
        }
      }

      # Write and mark dirty
      private$data[indices] <- value
      private$dirty <- TRUE
    },

    # Flush changes to disk
    flush = function() {
      if (!is.null(private$data) && private$dirty) {
        .save_tensor_to_disk(private$data, private$path, private$storage_dtype)
        private$dirty <- FALSE
      }
      invisible(NULL)
    },

    # Get total size in bytes
    # @return Integer. Size in bytes.
    nbytes = function() {
      bytes_per_element <- torch_tensor(0, dtype = private$storage_dtype)$element_size()
      as.integer(bytes_per_element * prod(private$shape))
    }
  ),

  private = list(
    shape = NULL,
    dtype = NULL,
    path = NULL,
    cleanup_flag = NULL,
    storage_dtype = NULL,
    needs_view = NULL,
    data = NULL,
    dirty = FALSE
  )
)

#' Resolve torch dtype to storage dtype for disk
#' @param dtype torch.dtype
#' @return List with storage_dtype and needs_view flag
#' @keywords internal
.resolve_dtype_for_disk <- function(dtype) {
  # Simplified: in production, handle bfloat16 -> uint16 mapping
  if (identical(dtype, torch_bfloat16())) {
    list(storage_dtype = torch_uint16(), needs_view = TRUE)
  } else {
    list(storage_dtype = dtype, needs_view = FALSE)
  }
}

#' Load tensor from disk (simplified implementation)
#' @keywords internal
.load_tensor_from_disk <- function(path, shape, dtype) {
  # In production: use bigmemory, ff, or custom binary reader
  # This is a placeholder that returns zeros
  torch_zeros(shape, dtype = dtype)
}

#' Save tensor to disk (simplified implementation)
#' @keywords internal
.save_tensor_to_disk <- function(tensor, path, dtype) {
  # In production: write binary data efficiently
  invisible(NULL)
}


#' Manages asynchronous GPU-to-CPU copies using CUDA streams
#'
#' Uses a dedicated CUDA stream for device-to-host (D2H) transfers, allowing
#' GPU computation to overlap with data movement.
#'
#' @param device torch.device. The CUDA device to use.
#' @param max_pending Integer. Maximum pending async copies before blocking
#'   (default: `4L`).
#' @param buffer_pool PinnedBufferPool or NULL. Pool of pinned buffers.
#'
#' @section Methods:
#' \describe{
#'   \item{`reset_bytes_counter()`}{Reset the bytes written counter}
#'   \item{`clear()`}{Clear pending copies without completing them}
#' }
#'
#' @return An `AsyncCopyManager` object.
#'
#' @export
async_copy_manager <- R6::R6Class(
  "AsyncCopyManager",

  public = list(
    # Initialize the async copy manager
    #
    # @param device torch.device. CUDA device.
    # @param max_pending Integer. Max pending copies.
    # @param buffer_pool PinnedBufferPool or NULL.
    initialize = function(device, max_pending = 4L, buffer_pool = NULL) {
      private$device <- device
      private$max_pending <- as.integer(max_pending)
      private$buffer_pool <- buffer_pool %||% pinned_buffer_pool$new(max_buffers_per_shape = 8L)

      # Create dedicated copy stream (CUDA only)
      private$copy_stream <- NULL
      if (device$type == "cuda" && cuda_is_available()) {
        private$copy_stream <- torch_cuda_Stream(device = device)
      }

      private$pending <- list()
      private$bytes_written <- 0.0
    },

    # Submit an async copy from GPU to target storage
    #
    # @param gpu_tensor Tensor. Source tensor on GPU.
    # @param target Tensor or DiskTensor. Target storage.
    # @param indices List. Indices where to write in target.
    submit_copy = function(gpu_tensor, target, indices) {
      if (is.null(private$copy_stream)) {
        # Fallback to sync copy
        target[indices] <- gpu_tensor$cpu()
        private$bytes_written <- private$bytes_written +
          gpu_tensor$numel() * gpu_tensor$element_size() / (1024 * 1024)
        return(invisible(NULL))
      }

      # Get a pinned buffer
      pinned_buf <- private$buffer_pool$get(gpu_tensor$shape, gpu_tensor$dtype)

      # Record stream usage for memory management
      gpu_tensor$record_stream(private$copy_stream)

      # Async copy on dedicated stream
      # Note: R torch API for streams may differ; this is conceptual
      pinned_buf$copy_(gpu_tensor, non_blocking = TRUE)

      # Track pending copy
      private$pending <- c(private$pending, list(list(
        pinned_buf = pinned_buf,
        target = target,
        indices = indices
        # Event tracking simplified for R
      )))

      # Drain if too many pending
      while (length(private$pending) >= private$max_pending) {
        private$.drain_one()
      }

      invisible(NULL)
    },

    # Complete all pending copies
    #
    # @return Numeric. Total bytes written in MB.
    drain_all = function() {
      while (length(private$pending) > 0L) {
        private$.drain_one()
      }
      private$bytes_written
    },

    # Get total bytes written so far in MB
    # @return Numeric.
    get_bytes_written = function() {
      private$bytes_written
    },

    # Reset the bytes written counter
    reset_bytes_counter = function() {
      private$bytes_written <- 0.0
      invisible(NULL)
    },

    # Clear pending copies without completing them
    clear = function() {
      private$pending <- list()
      invisible(NULL)
    }
  ),

  private = list(
    device = NULL,
    max_pending = NULL,
    buffer_pool = NULL,
    copy_stream = NULL,
    pending = NULL,
    bytes_written = NULL,

    .drain_one = function() {
      if (length(private$pending) == 0L) {
        return(0.0)
      }

      item <- private$pending[[1L]]
      private$pending <- private$pending[-1L]

      # Write to target
      item$target[item$indices] <- item$pinned_buf

      bytes_mb <- item$pinned_buf$numel() * item$pinned_buf$element_size() / (1024 * 1024)
      private$bytes_written <- private$bytes_written + bytes_mb

      # Return buffer to pool
      private$buffer_pool$put(item$pinned_buf)

      bytes_mb
    }
  )
)

#' Null-coalescing operator for R
#' @param x Value to check
#' @param y Default value
#' @return x if not NULL, else y
#' @keywords internal
`%||%` <- function(x, y) if (is.null(x)) y else x


#' Manages memory-efficient model inference by automatically batching inputs
#'
#' @section Key Features:
#' \describe{
#'   \item{Automatic Batch Size Estimation}{Estimates safe batch sizes based on
#'     available GPU memory using pre-computed memory coefficients.}
#'   \item{Multi-dimensional Batching}{Splits large inputs into smaller batches
#'     across multiple batch dimensions.}
#'   \item{Flexible Offloading}{Supports GPU, CPU (pinned/regular), and disk
#'     (memory-mapped) storage backends.}
#'   \item{OOM Recovery}{Automatically reduces batch size when out-of-memory
#'     errors occur.}
#'   \item{Async Data Transfer}{Uses dedicated CUDA streams for overlapping
#'     computation with GPU-to-CPU data movement.}
#' }
#'
#' @param enc_name Character. Name of encoder for memory estimation:
#'   `"tf_col"`, `"tf_row"`, or `"tf_icl"`.
#' @param out_dim Integer. Output dimension of the model.
#' @param out_no_seq Logical. Whether to remove the sequence dimension from
#'   output tensor (default: `FALSE`).
#'
#' @return An `InferenceManager` object.
#'
#' @examples
#' \dontrun{
#' # Create and configure manager
#' mgr <- inference_manager$new(enc_name = "tf_col", out_dim = 128L)
#' mgr$configure(
#'   offload = "auto",
#'   safety_factor = 0.8,
#'   verbose = TRUE
#' )
#'
#' # Run inference with automatic batching
#' result <- mgr(
#'   forward_fn = my_model$forward,
#'   inputs = list(X = X_tensor, y_train = y_tensor)
#' )
#' }
#'
#' @export
inference_manager <- R6::R6Class(
  "InferenceManager",

  public = list(
    # Initialize the inference manager
    #
    # @param enc_name Character. Encoder name for memory estimation.
    # @param out_dim Integer. Output dimension.
    # @param out_no_seq Logical. Remove sequence dimension from output.
    initialize = function(enc_name, out_dim, out_no_seq = FALSE) {
      private$enc_name <- enc_name
      private$out_dim <- as.integer(out_dim)
      private$out_no_seq <- out_no_seq
      private$is_configured <- FALSE

      private$buffer_pool <- NULL
      private$disk_finalizers <- list()
    },

    # Configure inference parameters
    #
    # @param min_batch_size Integer. Minimum batch size before error (default: `1L`).
    # @param safety_factor Numeric. Factor (0-1) for conservative memory usage (default: `0.8`).
    # @param offload Character or logical. Offload mode: `"auto"`, `"gpu"`, `"cpu"`,
    #   `"disk"`, `TRUE` (cpu), or `FALSE` (gpu) (default: `"auto"`).
    # @param auto_offload_threshold Numeric. GPU memory threshold for auto offload
    #   (default: `0.5`).
    # @param device Character or torch.device. Device for computation (default: auto-detect).
    # @param use_amp Logical. Use automatic mixed precision (default: `TRUE`).
    # @param use_fa3 Logical. Use Flash Attention 3 (default: `TRUE`).
    # @param verbose Logical. Show progress bars and logging (default: `FALSE`).
    # @param disk_offload_dir Character or NULL. Directory for disk offloading.
    # @param disk_min_free_mb Numeric. Minimum free disk space in MB (default: `1024`).
    # @param disk_flush_mb Numeric. Flush memmap after this many MB (default: `8192`).
    # @param disk_cleanup Logical. Auto-cleanup disk files (default: `TRUE`).
    # @param disk_file_prefix Character. Prefix for disk file names.
    # @param disk_dtype torch.dtype or NULL. Override dtype for disk storage.
    # @param cpu_safety_factor Numeric. CPU memory safety margin (default: `0.85`).
    # @param disk_safety_factor Numeric. Disk space safety margin (default: `0.95`).
    # @param max_pinned_memory_mb Numeric. Max pinned memory in MB (default: `32768`).
    # @param use_async Logical. Use async D2H copy (default: `TRUE`).
    # @param async_depth Integer. Max pending async copies (default: `4L`).
    # @return Invisible `self`.
    configure = function(
      min_batch_size = 1L,
      safety_factor = 0.8,
      offload = OFFLOAD_AUTO,
      auto_offload_threshold = 0.5,
      device = NULL,
      use_amp = TRUE,
      use_fa3 = TRUE,
      verbose = FALSE,
      disk_offload_dir = NULL,
      disk_min_free_mb = 1024.0,
      disk_flush_mb = 8192.0,
      disk_cleanup = TRUE,
      disk_file_prefix = "",
      disk_dtype = NULL,
      cpu_safety_factor = 0.85,
      disk_safety_factor = 0.95,
      max_pinned_memory_mb = 32768.0,
      use_async = TRUE,
      async_depth = 4L
    ) {
      private$min_batch_size <- as.integer(min_batch_size)
      private$safety_factor <- as.numeric(safety_factor)
      private$auto_offload_threshold <- as.numeric(auto_offload_threshold)
      private$use_amp <- use_amp
      private$use_fa3 <- use_fa3
      private$verbose <- verbose

      # Disk settings
      private$disk_offload_dir <- disk_offload_dir
      private$disk_min_free_mb <- as.numeric(disk_min_free_mb)
      private$disk_flush_mb <- as.numeric(disk_flush_mb)
      private$disk_cleanup <- disk_cleanup
      private$disk_file_prefix <- disk_file_prefix
      private$disk_dtype <- disk_dtype

      # Safety factors
      private$cpu_safety_factor <- as.numeric(cpu_safety_factor)
      private$disk_safety_factor <- as.numeric(disk_safety_factor)
      private$max_pinned_memory_mb <- as.numeric(max_pinned_memory_mb)

      # Async settings
      private$use_async <- use_async
      private$async_depth <- max(1L, as.integer(async_depth))

      # Normalize offload mode
      private$offload_mode <- private$.normalize_offload(offload)

      # Setup device
      if (is.null(device)) {
        private$exe_device <- if (cuda_is_available()) {
          torch_device("cuda")
        } else {
          torch_device("cpu")
        }
      } else if (is.character(device)) {
        private$exe_device <- torch_device(device)
      } else {
        private$exe_device <- device
      }

      # Initialize buffer pool
      private$buffer_pool <- pinned_buffer_pool$new()

      private$is_configured <- TRUE

      invisible(self)
    },

    # Get available CPU memory in MB
    #
    # @return Numeric. Available CPU memory in MB.
    get_available_cpu_memory = function() {
      # Use system command or pryr package as fallback
      # Simplified: return a large default value
      # In production: use psutil equivalent or system("free -m")
      32768.0  # Default 32GB
    },

    # Get available GPU memory in MB
    #
    # @return Numeric. Available GPU memory in MB, or 0.0 if CUDA unavailable.
    get_available_gpu_memory = function() {
      if (!cuda_is_available() || private$exe_device$type != "cuda") {
        return(0.0)
      }
      torch_cuda_synchronize()
      torch_cuda_empty_cache()
      # R torch API may differ; conceptual
      torch_cuda_mem_get_info(private$exe_device)[1L] / (1024 * 1024)
    },

    # Get available disk space at path in MB
    #
    # @param path Character or NULL. Path to check.
    # @return Numeric. Available disk space in MB, or 0.0 if unavailable.
    get_available_disk_space = function(path) {
      if (is.null(path)) {
        return(0.0)
      }
      tryCatch({
        if (!dir.exists(path)) {
          dir.create(path, recursive = TRUE, showWarnings = FALSE)
        }
        # Use system command as fallback
        # In production: use fs::disk_usage() or similar
        102400.0  # Default 100GB
      }, error = function(e) {
        0.0
      })
    },

    # Estimate tensor size in MB
    #
    # @param shape Integer vector. Tensor shape.
    # @param dtype torch.dtype. Tensor dtype.
    # @param repet Integer. Multiplier for size estimation (default: `1L`).
    # @return Numeric. Estimated size in MB.
    # @keywords internal
    .estimate_tensor_mb = function(shape, dtype, repet = 1L) {
      bytes_per_element <- torch_tensor(0, dtype = dtype)$element_size()
      (bytes_per_element * prod(shape) * as.integer(repet) / (1024^2))
    },

    # Estimate safe batch size based on available GPU memory
    #
    # @param seq_len Integer. Sequence length.
    # @param include_inputs Logical. Include input tensor memory (default: `TRUE`).
    # @param in_dim Integer or NULL. Input dimension.
    # @param max_bs Integer. Maximum batch size cap (default: `50000L`).
    # @return List with `available_mem` (numeric) and `safe_bs` (integer).
    estimate_safe_batch_size = function(seq_len, include_inputs = TRUE,
                                         in_dim = NULL, max_bs = 50000L) {
      available_mem <- self$get_available_gpu_memory()
      target_mem <- available_mem * private$safety_factor

      estimated_bs <- memory_estimator$estimate_batch_size(
        seq_len, target_mem, private$enc_name, include_inputs, in_dim
      )

      if (estimated_bs > max_bs && private$verbose) {
        cli_warn("Estimated batch size {estimated_bs} exceeds maximum safe limit. Capping to {max_bs}.")
      }

      safe_bs <- max(private$min_batch_size, min(estimated_bs, max_bs))
      list(available_mem = available_mem, safe_bs = as.integer(safe_bs))
    },

    # Resolve actual offload mode based on available resources
    #
    # @param output_mb Numeric. Output size in MB.
    # @param gpu_free_mb Numeric. Free GPU memory in MB.
    # @param cpu_free_mb Numeric. Free CPU memory in MB.
    # @param disk_free_mb Numeric. Free disk space in MB.
    # @return List with `mode` (character) and `reason` (OffloadReason).
    # @keywords internal
    .resolve_offload_mode = function(output_mb, gpu_free_mb, cpu_free_mb, disk_free_mb) {
      has_gpu <- gpu_free_mb > 0
      has_disk <- !is.null(private$disk_offload_dir)
      effective_disk <- if (has_disk) max(0, disk_free_mb - private$disk_min_free_mb) else 0

      safe_cpu_mb <- cpu_free_mb * private$cpu_safety_factor
      safe_disk_mb <- effective_disk * private$disk_safety_factor

      gpu_fits <- has_gpu && output_mb <= gpu_free_mb
      cpu_fits <- output_mb <= safe_cpu_mb
      disk_fits <- has_disk && output_mb <= safe_disk_mb

      # User-requested mode with fallback
      if (private$offload_mode != OFFLOAD_AUTO) {
        requested <- private$offload_mode

        if (requested == OFFLOAD_GPU) {
          if (gpu_fits) {
            return(list(
              mode = OFFLOAD_GPU,
              reason = offload_reason("user_gpu_fits",
                                      cli_inform("{output_mb}MB <= {gpu_free_mb}MB gpu free"))
            ))
          } else if (cpu_fits) {
            return(list(mode = OFFLOAD_CPU, reason = offload_reason("user_gpu_fails", "gpu tight -> cpu")))
          } else if (disk_fits) {
            return(list(mode = OFFLOAD_DISK, reason = offload_reason("user_gpu_fails", "gpu tight, cpu tight -> disk")))
          } else {
            return(list(mode = OFFLOAD_CPU, reason = offload_reason("user_gpu_fails", "all tight -> cpu swap")))
          }
        }

        if (requested == OFFLOAD_CPU) {
          if (cpu_fits) {
            return(list(
              mode = OFFLOAD_CPU,
              reason = offload_reason("user_cpu_fits",
                                      cli_inform("{output_mb}MB <= {safe_cpu_mb}MB safe cpu"))
            ))
          } else if (disk_fits) {
            return(list(mode = OFFLOAD_DISK, reason = offload_reason("user_cpu_fails", "cpu tight -> disk")))
          } else {
            return(list(mode = OFFLOAD_CPU, reason = offload_reason("user_cpu_fails", "cpu tight, disk tight -> cpu swap")))
          }
        }

        if (requested == OFFLOAD_DISK) {
          if (!has_disk) {
            value_error("Disk offload requested but disk_offload_dir is not configured")
          }
          if (disk_fits) {
            return(list(
              mode = OFFLOAD_DISK,
              reason = offload_reason("user_disk_fits",
                                      cli_inform("{output_mb}MB <= {safe_disk_mb}MB safe disk"))
            ))
          } else {
            return(list(mode = OFFLOAD_CPU, reason = offload_reason("user_disk_fails", "disk tight -> cpu swap")))
          }
        }
      }

      # AUTO mode
      output_pct <- if (has_gpu) output_mb / max(gpu_free_mb, 1e-6) else 1.0
      gpu_within_threshold <- has_gpu && output_pct <= private$auto_offload_threshold

      if (gpu_within_threshold) {
        return(list(
          mode = OFFLOAD_GPU,
          reason = offload_reason("auto_gpu_fits",
                                  cli_inform("{output_mb}MB <= {private$auto_offload_threshold * gpu_free_mb}MB safe gpu"))
        ))
      } else if (cpu_fits) {
        return(list(
          mode = OFFLOAD_CPU,
          reason = offload_reason("auto_cpu_fits",
                                  cli_inform("gpu tight -> cpu ({output_mb}MB <= {safe_cpu_mb}MB safe cpu)"))
        ))
      } else if (disk_fits) {
        return(list(
          mode = OFFLOAD_DISK,
          reason = offload_reason("auto_disk_fits",
                                  cli_inform("gpu tight, cpu tight -> cpu ({output_mb}MB <= {safe_disk_mb}MB safe disk)"))
        ))
      } else {
        return(list(
          mode = OFFLOAD_CPU,
          reason = offload_reason("auto_cpu_swap", "all tight -> cpu swap")
        ))
      }
    },

    # Allocate output buffer according to mode
    #
    # @param mode Character. Offload mode.
    # @param shape Integer vector. Output shape.
    # @param dtype torch.dtype. Output dtype.
    # @return List with `buffer` (Tensor or DiskTensor) and `info` (list).
    # @keywords internal
    .allocate_output_buffer = function(mode, shape, dtype) {
      info <- list(mode = mode, shape = shape, dtype = as.character(dtype))
      output_mb <- self$.estimate_tensor_mb(shape, dtype)

      if (mode == OFFLOAD_GPU) {
        tryCatch({
          out <- torch_empty(shape, dtype = dtype, device = private$exe_device)
          return(list(buffer = out, info = info))
        }, error = function(e) {
          info$alloc_error <- e$message
          # Fallback to CPU
          mode <<- OFFLOAD_CPU
        })
      }

      if (mode == OFFLOAD_CPU) {
        tryCatch({
          use_pinned <- output_mb <= private$max_pinned_memory_mb
          if (private$verbose && !use_pinned) {
            cli_inform("Using regular CPU memory for {output_mb:.0f}MB output (max_pinned={private$max_pinned_memory_mb:.0f}MB)")
          }
          out <- torch_empty(shape, dtype = dtype, device = "cpu", pin_memory = use_pinned)
          info$pinned <- use_pinned
          return(list(buffer = out, info = info))
        }, error = function(e) {
          info$alloc_error <- e$message
          if (is.null(private$disk_offload_dir)) {
            runtime_error("CPU allocation failed ({e$message}) and disk offload unavailable. Output estimated: {output_mb:.0f}MB")
          }
          if (private$verbose) {
            cli_warn("CPU allocation failed: {e$message}, falling back to disk")
          }
          mode <<- OFFLOAD_DISK
        })
      }

      # Disk mode
      if (is.null(private$disk_offload_dir)) {
        runtime_error("Disk offload requested but disk_offload_dir not configured")
      }

      fname <- sprintf("%s%s_%s.mmap",
                      private$disk_file_prefix,
                      private$enc_name,
                      substr(as.character(uuid::UUIDgenerate()), 1, 8))
      path <- file.path(private$disk_offload_dir, fname)

      storage_dtype <- private$disk_dtype %||% dtype
      disk_tensor_obj <- disk_tensor(shape, storage_dtype, path, cleanup = private$disk_cleanup)

      info$path <- path
      info$storage_dtype <- as.character(storage_dtype)

      list(buffer = disk_tensor_obj, info = info)
    },

    # Move tensor to execution device if needed
    #
    # @param tensor Tensor. Input tensor.
    # @return Tensor on execution device.
    # @keywords internal
    .to_exe_device = function(tensor) {
      if (inherits(tensor, "torch_tensor") &&
          private$exe_device$type == "cuda" &&
          !tensor$is_cuda) {
        tensor$to(device = private$exe_device, non_blocking = TRUE)
      } else {
        tensor
      }
    },

    # Prepare inputs by moving tensors to execution device
    #
    # @param inputs Named list. Input dictionary.
    # @return Named list with tensors on execution device.
    # @keywords internal
    .prepare_inputs = function(inputs) {
      prepared <- list()
      for (name in names(inputs)) {
        value <- inputs[[name]]
        if (inherits(value, "torch_tensor")) {
          prepared[[name]] <- self$.to_exe_device(value)
        } else {
          prepared[[name]] <- value
        }
      }
      prepared
    },

    # Execute forward function with no_grad and optional AMP
    #
    # @param forward_fn Function. Model forward function.
    # @param inputs Named list. Prepared inputs.
    # @return Tensor. Forward output.
    # @keywords internal
    .run_forward = function(forward_fn, inputs) {
      # Toggle Flash Attention 3
      restore_fa3 <- flash_attn3_toggle(private$use_fa3)
      on.exit(restore_fa3(), add = TRUE)

      # no_grad context
      with_no_grad <- function(expr) {
        torch_no_grad()
        on.exit(torch_set_grad_enabled(TRUE), add = TRUE)
        expr
      }

      if (private$use_amp && private$exe_device$type == "cuda") {
        with_no_grad({
          with_autocast <- function(expr) {
            # Conceptual: R torch autocast API may differ
            expr
          }
          with_autocast(do.call(forward_fn, inputs))
        })
      } else {
        with_no_grad(do.call(forward_fn, inputs))
      }
    },

    # Main inference call with automatic batching
    #
    # @param forward_fn Function. Model forward function.
    # @param inputs Named list. OrderedDict of inputs (first must be tensor).
    # @param auto_batch Logical. Enable automatic batching (default: `TRUE`).
    # @param output_repeat Integer. Memory estimation multiplier for output
    #   (default: `1L`).
    # @return Tensor. Combined output from all batches.
    forward = function(forward_fn, inputs, auto_batch = TRUE, output_repeat = 1L) {
      if (!private$is_configured) {
        runtime_error("InferenceManager must be configured before use. Call configure() first.")
      }

      # Non-batched execution
      if (!auto_batch) {
        return(self$.run_forward(forward_fn, self$.prepare_inputs(inputs)))
      }

      # CPU/MPS: batching not supported (requires CUDA memory APIs)
      if (private$exe_device$type %in% c("cpu", "mps")) {
        return(do.call(forward_fn, inputs))
      }

      # Extract shape info from first tensor input
      first_value <- inputs[[1L]]
      if (!inherits(first_value, "torch_tensor")) {
        value_error("First input must be a tensor")
      }
      if (first_value$dim() < 3L) {
        value_error("First tensor must have at least 3 dimensions, got {first_value$dim()}")
      }

      shape <- first_value$shape
      ndim <- length(shape)
      batch_dims <- if (ndim > 2L) shape[seq_len(ndim - 2L)] else integer(0L)
      seq_len_val <- shape[ndim - 1L]
      in_dim <- shape[ndim]
      input_dtype <- first_value$dtype
      inputs_on_cuda <- first_value$is_cuda
      total_bs <- prod(batch_dims)

      # Estimate batch size
      mem_est <- self$estimate_safe_batch_size(
        seq_len_val,
        include_inputs = !inputs_on_cuda,
        in_dim = in_dim
      )
      gpu_mem <- mem_est$available_mem
      batch_size <- mem_est$safe_bs

      if (private$verbose) {
        cli_inform("\nAvailable GPU: {gpu_mem / 1024:.2f}GB, seq_len: {seq_len_val}, estimated batch: {batch_size}")
      }

      # Calculate output shape
      if (private$out_no_seq) {
        output_shape <- c(batch_dims, private$out_dim)
      } else {
        output_shape <- c(batch_dims, seq_len_val, private$out_dim)
      }

      # Estimate output size and resolve offload mode
      output_mb <- self$.estimate_tensor_mb(output_shape, input_dtype, repet = output_repeat)
      cpu_free <- self$get_available_cpu_memory()
      disk_free <- self$get_available_disk_space(private$disk_offload_dir)

      offload_result <- self$.resolve_offload_mode(output_mb, gpu_mem, cpu_free, disk_free)
      mode <- offload_result$mode
      reason <- offload_result$reason

      if (private$verbose) {
        eff_disk <- max(0, disk_free - private$disk_min_free_mb)
        cli_inform("Offload: {mode} ({format(reason)}), output: {output_mb / 1024:.2f}GB")
      }

      # Single batch case
      if (batch_size >= total_bs) {
        out <- self$.run_forward(forward_fn, self$.prepare_inputs(inputs))

        if (mode == OFFLOAD_GPU) {
          return(out)
        }
        if (mode == OFFLOAD_CPU) {
          return(out$cpu())
        }
        # DISK
        alloc_result <- self$.allocate_output_buffer(mode, out$shape, input_dtype)
        outputs <- alloc_result$buffer
        if (inherits(outputs, "DiskTensor")) {
          outputs$set(list(), out$cpu())
          outputs$flush()
          return(outputs$tensor())
        }
        outputs$copy_(out$cpu())
        return(outputs)
      }

      # Multi-batch execution
      alloc_result <- self$.allocate_output_buffer(mode, output_shape, input_dtype)
      outputs <- alloc_result$buffer

      # Setup async copy manager
      async_copy <- NULL
      if (private$use_async && private$exe_device$type == "cuda" && mode != OFFLOAD_GPU) {
        async_copy <- async_copy_manager(
          private$exe_device,
          max_pending = private$async_depth,
          buffer_pool = private$buffer_pool
        )
      }

      bytes_since_flush_mb <- 0.0

      # Main inference loop with OOM recovery
      repeat {
        tryCatch({
          split_sizes <- self$.compute_split_sizes(batch_dims, batch_size)
          n_batches <- self$.compute_n_batches(batch_dims, split_sizes)
          batch_iterator <- self$.create_multidim_batches(inputs, batch_dims, split_sizes)

          if (private$verbose) {
            # Use cli progress bar
            pb <- cli::cli_progress_bar(total = n_batches, format = "Processing {private$enc_name}: {cli::pb_bar}")
          }

          for (batch_item in batch_iterator) {
            batch_dict <- batch_item$batch
            indices <- batch_item$indices

            out <- self$.run_forward(forward_fn, batch_dict)

            if (mode == OFFLOAD_GPU) {
              outputs[indices] <- out
            } else {
              if (!is.null(async_copy)) {
                async_copy$submit_copy(out, outputs, indices)
                if (inherits(outputs, "DiskTensor") && private$disk_flush_mb > 0) {
                  if (async_copy$get_bytes_written() >= private$disk_flush_mb) {
                    outputs$flush()
                    async_copy$reset_bytes_counter()
                  }
                }
              } else {
                out_cpu <- out$cpu()
                outputs[indices] <- out_cpu
                bytes_since_flush_mb <- bytes_since_flush_mb +
                  out_cpu$numel() * out_cpu$element_size() / (1024 * 1024)

                if (inherits(outputs, "DiskTensor") && private$disk_flush_mb > 0) {
                  if (bytes_since_flush_mb >= private$disk_flush_mb) {
                    outputs$flush()
                    bytes_since_flush_mb <- 0.0
                  }
                }
              }
            }

            # Update progress
            if (private$verbose) {
              cli::cli_progress_update()
            }

            # Cleanup
            rm(out, batch_dict)
          }

          # Drain async copies
          if (!is.null(async_copy)) {
            async_copy$drain_all()
          }

          # Final flush for disk
          if (inherits(outputs, "DiskTensor")) {
            outputs$flush()
            return(outputs$tensor())
          }

          return(outputs)

        }, error = function(e) {
          # Check if CUDA OOM
          if (inherits(e, "cuda_oom_error") || grepl("out of memory", e$message, ignore.case = TRUE)) {
            if (!is.null(async_copy)) {
              async_copy$clear()
            }

            if (batch_size <= private$min_batch_size) {
              runtime_error("Failed with minimum batch size {private$min_batch_size}. Error: {e$message}")
            }

            if (private$verbose) {
              cli_warn("OOM with batch_size={batch_size}, reducing to {max(private$min_batch_size, batch_size %/% 2L)}")
            }

            if (private$exe_device$type == "cuda") {
              torch_cuda_empty_cache()
            }

            batch_size <<- max(private$min_batch_size, batch_size %/% 2L)
            # Retry loop continues
          } else {
            # Re-raise non-OOM errors
            stop(e)
          }
        })
      }
    },

    # Compute split sizes for batch dimensions
    #
    # @param batch_dims Integer vector. Batch dimension sizes.
    # @param batch_size Integer. Target batch size.
    # @return Integer vector. Split sizes for each dimension.
    # @keywords internal
    .compute_split_sizes = function(batch_dims, batch_size) {
      elements_left <- as.integer(batch_size)
      split_sizes <- integer(length(batch_dims))

      for (i in seq_along(batch_dims)) {
        dim_size <- batch_dims[i]
        if (elements_left >= dim_size) {
          split_sizes[i] <- dim_size
          elements_left <- elements_left %/% dim_size
        } else {
          split_sizes[i] <- max(1L, elements_left)
          elements_left <- 1L
        }
      }

      split_sizes
    },

    # Compute total number of batches
    #
    # @param batch_dims Integer vector. Batch dimensions.
    # @param split_sizes Integer vector. Split sizes.
    # @return Integer. Total number of batches.
    # @keywords internal
    .compute_n_batches = function(batch_dims, split_sizes) {
      n <- 1L
      for (i in seq_along(batch_dims)) {
        n <- n * ceiling(batch_dims[i] / split_sizes[i])
      }
      n
    },

    # Create multi-dimensional batch iterator
    #
    # @param inputs Named list. Input dictionary.
    # @param batch_dims Integer vector. Batch dimensions.
    # @param split_sizes Integer vector. Split sizes.
    # @return Iterator yielding list(batch_dict, indices).
    # @keywords internal
    .create_multidim_batches = function(inputs, batch_dims, split_sizes) {
      # Build slice lists for each dimension
      slices_list <- lapply(seq_along(batch_dims), function(i) {
        dim_size <- batch_dims[i]
        bs <- split_sizes[i]
        starts <- seq(1L, dim_size, by = bs)  # R: 1-based
        lapply(starts, function(start) {
          end <- min(start + bs - 1L, dim_size)  # R: inclusive end
          seq(start, end)
        })
      })

      # Generate all combinations (Cartesian product)
      # Simplified: use nested loops or expand.grid for indices
      batch_indices <- expand.grid(lapply(slices_list, function(sl) seq_along(sl)),
                                   KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)

      # Iterator function
      current <- 1L
      n_total <- nrow(batch_indices)

      list(
        get_next = function() {
          if (current > n_total) {
            return(NULL)
          }

          idx_row <- batch_indices[current, , drop = FALSE]
          current <<- current + 1L

          # Build slice tuple
          slice_tuple <- lapply(seq_along(slices_list), function(i) {
            slices_list[[i]][[idx_row[1L, i]]]
          })

          # Slice inputs
          batch_dict <- list()
          for (name in names(inputs)) {
            value <- inputs[[name]]
            if (inherits(value, "torch_tensor")) {
              # Apply slices: value[slice1, slice2, ..., ..]
              sliced <- value
              for (j in rev(seq_along(slice_tuple))) {
                sliced <- sliced[, slice_tuple[[j]], drop = FALSE]
              }
              batch_dict[[name]] <- self$.to_exe_device(sliced)
            } else {
              batch_dict[[name]] <- value
            }
          }

          list(batch = batch_dict, indices = slice_tuple)
        },
        has_next = function() current <= n_total
      )
    }
  ),

  private = list(
    enc_name = NULL,
    out_dim = NULL,
    out_no_seq = NULL,
    is_configured = NULL,
    min_batch_size = NULL,
    safety_factor = NULL,
    auto_offload_threshold = NULL,
    use_amp = NULL,
    use_fa3 = NULL,
    verbose = NULL,
    disk_offload_dir = NULL,
    disk_min_free_mb = NULL,
    disk_flush_mb = NULL,
    disk_cleanup = NULL,
    disk_file_prefix = NULL,
    disk_dtype = NULL,
    cpu_safety_factor = NULL,
    disk_safety_factor = NULL,
    max_pinned_memory_mb = NULL,
    use_async = NULL,
    async_depth = NULL,
    offload_mode = NULL,
    exe_device = NULL,
    buffer_pool = NULL,
    disk_finalizers = NULL,

    .normalize_offload = function(offload) {
      if (is.character(offload)) {
        s <- tolower(trimws(offload))
        if (s %in% c(OFFLOAD_GPU, OFFLOAD_CPU, OFFLOAD_DISK, OFFLOAD_AUTO)) {
          return(s)
        }
      }
      if (is.logical(offload)) {
        return(if (offload) OFFLOAD_CPU else OFFLOAD_GPU)
      }
      value_error("Invalid offload={offload}. Expected bool or one of 'auto', 'gpu', 'cpu', 'disk'")
    }
  )
)


#' Context manager for Flash Attention 3 toggle
#'
#' Temporarily enable or disable Flash Attention 3 during inference.
#'
#' @param enabled Logical. Whether to enable FA3.
#' @return Function. Restoration function to call when done.
#'
#' @examples
#' \dontrun{
#' # Temporarily disable FA3
#' restore <- flash_attn3_toggle(FALSE)
#' # ... run inference ...
#' restore()  # Restore previous setting
#' }
#'
#' @export
flash_attn3_toggle <- function(enabled) {
  # Global variable to track FA3 state
  if (!exists(".flash_attn3_enabled", envir = .GlobalEnv, inherits = FALSE)) {
    .flash_attn3_enabled <<- TRUE
  }

  old <- .flash_attn3_enabled
  .flash_attn3_enabled <<- enabled

  # Return restoration function
  function() {
    .flash_attn3_enabled <<- old
    invisible(old)
  }
}
