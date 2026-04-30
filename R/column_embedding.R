#' Distribution-aware Column-wise Embedding for TabICL2
#'
#' @name col_embedding
#' @keywords internal
NULL


#' Distribution-aware column-wise embedding module
#'
#' This module maps each scalar cell in a column to a high-dimensional embedding while
#' capturing statistical regularities within the column. Unlike traditional approaches
#' that use separate embedding layers per column, it employs a shared set transformer
#' to process all features.
#'
#' ColEmbedding operates in two modes depending on the `affine` parameter:
#'
#' \strong{When `affine = TRUE`:}
#' \enumerate{
#'   \item Each scalar cell is first linearly projected into the embedding dimension
#'   \item The set transformer generates distribution-aware weights and biases for each column
#'   \item The final column embeddings are computed as: \eqn{\text{column} \times W + b}
#' }
#'
#' \strong{When `affine = FALSE`:}
#' \enumerate{
#'   \item Each scalar cell is first linearly projected into the embedding dimension
#'   \item The set transformer processes the projected features
#'   \item The final column embeddings are directly the set transformer's output
#' }
#'
#' @param embed_dim Integer. Embedding dimension.
#' @param num_blocks Integer. Number of induced self-attention blocks in the set transformer.
#' @param nhead Integer. Number of attention heads in the set transformer.
#' @param dim_feedforward Integer. Dimension of the feedforward network in the set transformer.
#' @param num_inds Integer. Number of inducing points used in self-attention blocks.
#' @param dropout Numeric. Dropout probability in the set transformer (default: `0.0`).
#' @param activation Character or function. Activation function for feedforward network:
#'   `"relu"`, `"gelu"`, or a unary callable (default: `"gelu"`).
#' @param norm_first Logical. If `TRUE`, uses pre-norm architecture (default: `TRUE`).
#' @param bias_free_ln Logical. If `TRUE`, removes bias from LayerNorm layers (default: `FALSE`).
#' @param affine Logical. If `TRUE`, computes embeddings as `features * W + b` (default: `TRUE`).
#' @param feature_group Logical or character. Feature grouping mode:
#'   `FALSE` (no grouping), `TRUE` or `"same"` (circular permutation),
#'   `"valid"` (padding + reshaping) (default: `FALSE`).
#' @param feature_group_size Integer. Number of features per group when grouping is enabled (default: `3L`).
#' @param target_aware Logical. If `TRUE`, incorporates target information into embeddings (default: `FALSE`).
#' @param max_classes Integer. Number of classes for classification; `0L` for regression (default: `10L`).
#' @param reserve_cls_tokens Integer. Number of slots to reserve for CLS tokens (default: `4L`).
#' @param ssmax Logical or character. Scalable softmax type for attention:
#'   `FALSE`/`"none"`, `TRUE`/`"qassmax-mlp-elementwise"`, or custom string (default: `FALSE`).
#' @param mixed_radix_ensemble Logical. Enable mixed-radix ensembling for many-class classification
#'   (only if `target_aware = TRUE` and `max_classes > 0`) (default: `TRUE`).
#' @param recompute Logical. If `TRUE`, uses gradient checkpointing (default: `FALSE`).
#'
#' @return A `nn_module` object that can be used like any other torch module.
#'
#' @examples
#' \dontrun{
#' # Create a ColEmbedding module
#' col_emb <- col_embedding(
#'   embed_dim = 128L,
#'   num_blocks = 2L,
#'   nhead = 4L,
#'   dim_feedforward = 256L,
#'   num_inds = 32L,
#'   affine = TRUE,
#'   target_aware = FALSE
#' )
#'
#' # Forward pass during training
#' X <- torch_randn(2L, 100L, 20L)  # [batch, rows, features]
#' y_train <- torch_randint(1L, 11L, c(2L, 80L))  # [batch, train_size]
#' embeddings <- col_emb(X, y_train)
#'
#' # Forward pass during inference with caching
#' col_cache <- kv_cache()  # assuming KVCache has nn_module wrapper
#' embeddings_inf <- col_emb$forward_with_cache(
#'   X, col_cache, y_train, store_cache = TRUE
#' )
#' }
#'
#' @export
col_embedding <- nn_module(
  "ColEmbedding",

  initialize = function(
    embed_dim,
    num_blocks,
    nhead,
    dim_feedforward,
    num_inds,
    dropout = 0.0,
    activation = "gelu",
    norm_first = TRUE,
    bias_free_ln = FALSE,
    affine = TRUE,
    feature_group = FALSE,
    feature_group_size = 3L,
    target_aware = FALSE,
    max_classes = 10L,
    reserve_cls_tokens = 4L,
    ssmax = FALSE,
    mixed_radix_ensemble = TRUE,
    recompute = FALSE
  ) {
    # Store configuration as private fields (convention: prefix with .)
    self$embed_dim <- embed_dim
    self$reserve_cls_tokens <- reserve_cls_tokens
    self$feature_group <- feature_group
    self$feature_group_size <- feature_group_size
    self$target_aware <- target_aware
    self$max_classes <- max_classes
    self$affine <- affine
    self$mixed_radix_ensemble <- mixed_radix_ensemble

    # Initial linear projection: handles feature grouping dimension
    in_dim <- if (feature_group) feature_group_size else 1L
    self$in_linear <- skippable_linear(in_dim, embed_dim)

    # Shared SetTransformer for column processing
    self$tf_col <- set_transformer(
      num_blocks = num_blocks,
      d_model = embed_dim,
      nhead = nhead,
      dim_feedforward = dim_feedforward,
      num_inds = num_inds,
      dropout = dropout,
      activation = activation,
      norm_first = norm_first,
      bias_free_ln = bias_free_ln,
      ssmax = ssmax,
      recompute = recompute
    )

    # Target encoder (if target-aware)
    if (target_aware) {
      if (max_classes > 0L) {
        # Classification: one-hot + linear
        self$y_encoder <- one_hot_and_linear(max_classes, embed_dim)
      } else {
        # Regression: simple linear
        self$y_encoder <- nn_linear(1L, embed_dim)
      }
    }

    # Affine transformation modules (if affine mode)
    if (affine) {
      self$out_w <- skippable_linear(embed_dim, embed_dim)
      self$ln_w <- if (norm_first) {
        nn_layer_norm(embed_dim, elementwise_affine = !bias_free_ln)
      } else {
        nn_identity()
      }

      self$out_b <- skippable_linear(embed_dim, embed_dim)
      self$ln_b <- if (norm_first) {
        nn_layer_norm(embed_dim, elementwise_affine = !bias_free_ln)
      } else {
        nn_identity()
      }
    }

    # Inference manager for caching
    self$inference_mgr <- inference_manager$new(enc_name = "tf_col", out_dim = embed_dim)
  },

  # Map feature shuffle pattern from reference table to another table
  #
  # @param reference_pattern Integer vector. Shuffle pattern of features in reference table.
  # @param other_pattern Integer vector. Shuffle pattern of features in another table.
  # @return Integer vector. Mapping from reference table's ordering to another table's ordering.
  # @keywords internal
  map_feature_shuffle = function(reference_pattern, other_pattern) {
    # R: named vector as dictionary equivalent
    orig_to_other <- setNames(seq_along(other_pattern) - 1L, other_pattern)
    # Map each feature in reference to its index in other_pattern
    vapply(reference_pattern, function(feat) orig_to_other[as.character(feat)], integer(1L))
  },

  # Group features into fixed-size groups
  #
  # @param X Input tensor of shape `(B, TT, H)`.
  # @return Tensor of shape `(B, TT, G, feature_group_size)`.
  # @keywords internal
  feature_grouping = function(X) {
    if (!self$feature_group) {
      return(X$unsqueeze(-1L))  # (B, TT, H, 1)
    }

    shape <- X$shape
    B <- shape[1L]
    TT <- shape[2L]
    H <- shape[3L]
    size <- self$feature_group_size

    # Determine grouping mode
    mode <- if (isTRUE(self$feature_group)) "same" else self$feature_group

    if (mode == "same") {
      # Group through circular permutation
      idxs <- torch_arange(0L, H, dtype = torch_long(), device = X$device)
      # Stack shifted versions: [(idxs + 2^i) % H] for i in 0:size-1
      shifted <- lapply(seq_len(size) - 1L, function(i) {
        X[, , (idxs + (2L^i)) %% H + 1L]  # R: +1 for 1-based indexing
      })
      X <- torch_stack(shifted, dim = -1L)
    } else {
      # Group through padding and reshaping
      x_pad_cols <- (size - H %% size) %% size
      if (x_pad_cols > 0L) {
        X <- nnf_pad(X, pad = c(0L, x_pad_cols), value = 0)
      }
      # Reshape: (B, TT, H_padded) -> (B, TT, G, size)
      new_shape <- c(B, TT, -1L, size)
      X <- X$view(new_shape)
    }

    X  # (B, TT, G, size)
  },

  # Compute balanced bases for mixed-radix decomposition (internal)
  #
  # @param num_classes Integer. Total number of unique classes.
  # @return Integer vector. List of bases.
  # @keywords internal
  .compute_mixed_radix_bases = function(num_classes) {
    if (num_classes <= self$max_classes) {
      return(as.integer(num_classes))
    }

    D <- ceiling(log(num_classes) / log(self$max_classes))
    k <- ceiling(num_classes ^ (1.0 / D))
    k <- min(k, self$max_classes)

    bases <- rep(as.integer(k), D)
    product <- k ^ D
    idx <- 1L

    while (product < num_classes && idx <= D) {
      if (bases[idx] < self$max_classes) {
        product <- (product %/% bases[idx]) * (bases[idx] + 1L)
        bases[idx] <- bases[idx] + 1L
      }
      idx <- idx + 1L
    }

    as.integer(bases)
  },

  # Extract a specific digit from mixed-radix representation (internal)
  #
  # @param y Integer tensor. Original class labels.
  # @param digit_idx Integer. Digit index (0-based).
  # @param bases Integer vector. List of bases.
  # @return Integer tensor. Digit values.
  # @keywords internal
  .extract_mixed_radix_digit = function(y, digit_idx, bases) {
    divisor <- 1L
    for (i in (digit_idx + 2L):length(bases)) {
      divisor <- divisor * bases[i]
    }
    (y$to(dtype = torch_long()) %/% divisor) %% bases[digit_idx + 1L]
  },

  # Feature embedding using a shared set transformer (internal)
  #
  # @param features Input tensor `(..., TT, in_dim)`.
  # @param train_size Integer. Position to split training/test data.
  # @param y_train Tensor or NULL. Target values.
  # @param embed_with_test Logical. Whether inducing points attend to all samples.
  # @return Tensor `(..., TT, E)`.
  # @keywords internal
  .compute_embeddings = function(features, train_size, y_train = NULL, embed_with_test = FALSE) {
    src <- self$in_linear(features)

    if (!self$target_aware) {
      train_size_arg <- if (embed_with_test) NULL else train_size
      src <- self$tf_col(src, train_size = train_size_arg)
    } else {
      if(is.null(y_train)) {
        value_error("y_train must be provided when target_aware = TRUE")
      }

      num_classes <- as.integer(y_train$max()$item()) + 1L
      needs_mixed_radix <- self$max_classes > 0L && num_classes > self$max_classes

      if (!needs_mixed_radix) {
        if (self$max_classes > 0L) {
          y_emb <- self$y_encoder(y_train$to(dtype = torch_float()))
        } else {
          y_emb <- self$y_encoder(y_train$unsqueeze(-1L))
        }
        src[, , seq_len(train_size), ] <- src[, , seq_len(train_size), ] + y_emb

        train_size_arg <- if (embed_with_test) NULL else train_size
        src <- self$tf_col(src, train_size = train_size_arg)
      } else {
        if(!self$mixed_radix_ensemble) {
          value_error("Number of classes ({num_classes}) exceeds max_classes ({self$max_classes}).")
    }
        bases <- self$.compute_mixed_radix_bases(num_classes)
        num_digits <- length(bases)
        src_accum <- torch_zeros_like(src)
        src_with_y <- src$clone()

        for (digit_idx in seq_len(num_digits) - 1L) {
          y_digit <- self$.extract_mixed_radix_digit(y_train, digit_idx, bases)
          y_emb <- self$y_encoder(y_digit$to(dtype = torch_float()))
          src_with_y[, , seq_len(train_size), ] <- src[, , seq_len(train_size), ] + y_emb

          train_size_arg <- if (embed_with_test) NULL else train_size
          src_accum <- src_accum + self$tf_col(src_with_y, train_size = train_size_arg)
        }

        src <- src_accum / num_digits
      }
    }

    if (self$affine) {
      weights <- self$ln_w(self$out_w(src))
      biases <- self$ln_b(self$out_b(src))
      embeddings <- features * weights + biases
    } else {
      embeddings <- src
    }

    embeddings
  },

  # Training path when feature grouping is enabled (internal)
  # @keywords internal
  .train_forward_with_feature_group = function(X, y_train, embed_with_test) {
    train_size <- y_train$size(2L)

    X <- self$feature_grouping(X)
    if (self$reserve_cls_tokens > 0L) {
      X <- nnf_pad(X, pad = c(0L, 0L, self$reserve_cls_tokens, 0L), value = -100.0)
    }

    features <- X$transpose(1L, 2L)

    if (self$target_aware) {
      y_train <- y_train$unsqueeze(2L)$expand(-1L, features$size(2L), -1L)
    }

    embeddings <- self$.compute_embeddings(features, train_size, y_train, embed_with_test)
    embeddings$transpose(1L, 2L)
  },

  # Training path without feature grouping (internal)
  # @keywords internal
  .train_forward_without_feature_group = function(X, y_train, d, embed_with_test) {
    train_size <- y_train$size(2L)

    if (self$reserve_cls_tokens > 0L) {
      X <- nnf_pad(X, pad = c(self$reserve_cls_tokens, 0L), value = -100.0)
    }

    if (is.null(d)) {
      features <- X$transpose(1L, 2L)$unsqueeze(-1L)

      if (self$target_aware) {
        y_train <- y_train$unsqueeze(2L)$expand(-1L, features$size(2L), -1L)
      }

      embeddings <- self$.compute_embeddings(features, train_size, y_train, embed_with_test)
    } else {
      if (self$reserve_cls_tokens > 0L) {
        d <- d + self$reserve_cls_tokens
      }

      B <- X$size(1L)
      TT <- X$size(2L)
      HC <- X$size(3L)

      X <- X$transpose(1L, 2L)

      indices <- torch_arange(0L, HC, device = X$device)$unsqueeze(1L)$expand(B, HC)
      mask <- indices < d$unsqueeze(2L)

      features <- X[mask]$unsqueeze(-1L)

      if (self$target_aware) {
        y_train <- y_train$unsqueeze(2L)$expand(-1L, HC, -1L)
        y_train <- y_train[mask]
      }

      effective_embeddings <- self$.compute_embeddings(features, train_size, y_train, embed_with_test)

      embeddings <- torch_zeros(c(B, HC, TT, self$embed_dim),
                                device = X$device, dtype = effective_embeddings$dtype)
      embeddings[mask] <- effective_embeddings
    }

    embeddings$transpose(1L, 2L)
  },

  # Training forward dispatcher (internal)
  # @keywords internal
  .train_forward = function(X, y_train, d = NULL, embed_with_test = FALSE) {
    if (self$feature_group) {
      if(!is.null(d)) {
        value_error("d is not supported when feature grouping is enabled")
      }
      self$.train_forward_with_feature_group(X, y_train, embed_with_test)
    } else {
      self$.train_forward_without_feature_group(X, y_train, d, embed_with_test)
    }
  },

  # Inference path when feature grouping is enabled (internal)
  # @keywords internal
  .inference_with_feature_group = function(X, y_train, train_size, embed_with_test) {
    X <- self$feature_grouping(X)
    if (self$reserve_cls_tokens > 0L) {
      X <- nnf_pad(X, pad = c(0L, 0L, self$reserve_cls_tokens, 0L), value = -100.0)
    }

    features <- X$transpose(1L, 2L)

    if (self$target_aware) {
      y_train <- y_train$unsqueeze(2L)$expand(-1L, features$size(2L), -1L)
    } else {
      y_train <- NULL
    }

    self$inference_mgr(
      self$.compute_embeddings,
      features = features,
      train_size = train_size,
      y_train = y_train,
      embed_with_test = embed_with_test
    )
  },

  # Inference path without feature grouping (internal)
  # @keywords internal
  .inference_without_feature_group = function(X, y_train, train_size, embed_with_test,
                                               feature_shuffles) {
    if (is.null(feature_shuffles)) {
      if (self$reserve_cls_tokens > 0L) {
        X <- nnf_pad(X, pad = c(self$reserve_cls_tokens, 0L), value = -100.0)
      }

      features <- X$transpose(1L, 2L)$unsqueeze(-1L)

      if (self$target_aware) {
        y_train <- y_train$unsqueeze(2L)$expand(-1L, features$size(2L), -1L)
      } else {
        y_train <- NULL
      }

      self$inference_mgr(
        self$.compute_embeddings,
        features = features,
        train_size = train_size,
        y_train = y_train,
        embed_with_test = embed_with_test
      )
    } else {
      B <- X$size(1L)
      first_table <- X[1L, , , drop = FALSE]

      if (self$reserve_cls_tokens > 0L) {
        first_table <- nnf_pad(first_table, pad = c(self$reserve_cls_tokens, 0L), value = -100.0)
      }

      features <- first_table$transpose(1L, 2L)$unsqueeze(-1L)$squeeze(1L)

      if (self$target_aware) {
        y_first <- y_train[1L, ]$unsqueeze(1L)$expand(features$size(1L), -1L)
      } else {
        y_first <- NULL
      }

      first_embeddings <- self$inference_mgr(
        self$.compute_embeddings,
        features = features,
        train_size = train_size,
        y_train = y_first,
        embed_with_test = embed_with_test,
        output_repeat = B
      )

      embeddings <- first_embeddings$unsqueeze(1L)$`repeat`(c(B, 1L, 1L, 1L))
      first_pattern <- feature_shuffles[[1L]]

      for (i in seq(2L, B)) {
        mapping <- self$map_feature_shuffle(first_pattern, feature_shuffles[[i]])

        if (self$reserve_cls_tokens > 0L) {
          mapping <- mapping + self$reserve_cls_tokens
          mapping <- c(seq_len(self$reserve_cls_tokens) - 1L, mapping)
        }

        embeddings[i, .. ] <- first_embeddings[mapping + 1L, .. ]
      }

      embeddings
    }
  },

  # Inference forward dispatcher (internal)
  # @keywords internal
  .inference_forward = function(X, y_train, embed_with_test = FALSE,
                                 feature_shuffles = NULL, mgr_config = NULL) {
    if (is.null(mgr_config)) {
      mgr_config <- inference_config()$COL_CONFIG
    }
    self$inference_mgr$configure(!!!mgr_config)

    train_size <- y_train$size(2L)

    if (self$feature_group) {
      embeddings <- self$.inference_with_feature_group(X, y_train, train_size, embed_with_test)
    } else {
      embeddings <- self$.inference_without_feature_group(
        X, y_train, train_size, embed_with_test, feature_shuffles
      )
    }

    embeddings$transpose(1L, 2L)
  },

  # Main forward method: dispatches to train or inference path
  forward = function(X, y_train, d = NULL, embed_with_test = FALSE,
                     feature_shuffles = NULL, mgr_config = NULL) {
    if (self$training) {
      self$.train_forward(X, y_train, d, embed_with_test)
    } else {
      self$.inference_forward(X, y_train, embed_with_test, feature_shuffles, mgr_config)
    }
  },

  # Feature embedding with KV caching support (internal)
  # @keywords internal
  .compute_embeddings_with_cache = function(features, col_cache, train_size = NULL,
                                             y_train = NULL, use_cache = FALSE, store_cache = TRUE) {
    src <- self$in_linear(features)

    if (!self$target_aware) {
      src <- self$tf_col$forward_with_cache(
        src, col_cache = col_cache, train_size = train_size,
        use_cache = use_cache, store_cache = store_cache
      )
    } else {
      if (store_cache) {
        if (is.null(y_train)) {
          value_error("y_train must be provided when target_aware = TRUE and store_cache = TRUE")
        }
        if (self$max_classes > 0L) {
          y_emb <- self$y_encoder(y_train$to(dtype = torch_float()))
        } else {
          y_emb <- self$y_encoder(y_train$unsqueeze(-1L))
        }
        src[, , seq_len(train_size), ] <- src[, , seq_len(train_size), ] + y_emb
      }

      src <- self$tf_col$forward_with_cache(
        src, col_cache = col_cache, train_size = train_size,
        use_cache = use_cache, store_cache = store_cache
      )
    }

    if (self$affine) {
      weights <- self$ln_w(self$out_w(src))
      biases <- self$ln_b(self$out_b(src))
      embeddings <- features * weights + biases
    } else {
      embeddings <- src
    }

    embeddings
  },

  # Transform input table into embeddings with KV caching support
  forward_with_cache = function(X, col_cache, y_train = NULL, use_cache = FALSE,
                                store_cache = TRUE, mgr_config = NULL) {
    # Validate mutual exclusivity of cache flags
    if (!xor(use_cache, store_cache)) {
      value_error("Exactly one of use_cache or store_cache must be TRUE")
    }

    # Validate required argument when storing cache
    if (store_cache && is.null(y_train)) {
      value_error("y_train must be provided when store_cache = TRUE")
    }

    # Check mixed-radix ensemble compatibility with caching
    if (store_cache && self$target_aware && self$max_classes > 0L) {
      num_classes <- as.integer(y_train$max()$item()) + 1L
      if (num_classes > self$max_classes) {
        runtime_error("KV caching is not supported for classification with more classes ({num_classes}) than max_classes ({self$max_classes}).")
      }
    }
    if (is.null(mgr_config)) {
      mgr_config <- inference_config()$COL_CONFIG
    }
    self$inference_mgr$configure(!!!mgr_config)

    if (self$feature_group) {
      X <- self$feature_grouping(X)
      if (self$reserve_cls_tokens > 0L) {
        X <- nnf_pad(X, pad = c(0L, 0L, self$reserve_cls_tokens, 0L), value = -100.0)
      }
      features <- X$transpose(1L, 2L)
    } else {
      if (self$reserve_cls_tokens > 0L) {
        X <- nnf_pad(X, pad = c(self$reserve_cls_tokens, 0L), value = -100.0)
      }
      features <- X$transpose(1L, 2L)$unsqueeze(-1L)
    }

    if (store_cache) {
      train_size <- y_train$size(2L)
      y_train <- y_train$unsqueeze(2L)$expand(-1L, features$size(2L), -1L)
    } else {
      train_size <- NULL
      y_train <- NULL
    }

    embeddings <- self$inference_mgr(
      self$.compute_embeddings_with_cache,
      features = features,
      col_cache = col_cache,
      train_size = train_size,
      y_train = y_train,
      use_cache = use_cache,
      store_cache = store_cache
    )

    embeddings$transpose(1L, 2L)
  }
)
