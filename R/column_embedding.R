#' @importFrom torch nn_module nn_linear nn_layer_norm nn_identity
#' @importFrom torch torch_arange torch_stack torch_zeros_like torch_cat
#' @importFrom torch nnf_pad
#' @importFrom purrr map
#' @keywords internal
NULL

#' Distribution-aware Column-wise Embedding Module
#'
#' This module maps scalar cells to high-dimensional embeddings using a shared
#' Set Transformer. It supports feature grouping, target-aware encoding,
#' and affine transformation of features.
#'
#' @param embed_dim Integer. Embedding dimension.
#' @param num_blocks Integer. Number of induced self-attention blocks.
#' @param nhead Integer. Number of attention heads.
#' @param dim_feedforward Integer. Feedforward network dimension.
#' @param num_inds Integer. Number of inducing points.
#' @param dropout Numeric. Dropout probability (default: `0.0`).
#' @param activation Character. Activation function (default: `"gelu"`).
#' @param norm_first Logical. Pre-norm architecture (default: `TRUE`).
#' @param bias_free_ln Logical. Remove LayerNorm bias (default: `FALSE`).
#' @param affine Logical. Compute `features * W + b` (default: `TRUE`).
#' @param feature_group Logical or Character. Grouping mode:
#'   - `FALSE`: No grouping.
#'   - `TRUE` or `"same"`: Circular permutation.
#'   - `"valid"`: Padding and reshaping.
#' @param feature_group_size Integer. Size of feature groups (default: `3L`).
#' @param target_aware Logical. Use target info (default: `FALSE`).
#' @param max_classes Integer. Number of classes (default: `10L`).
#' @param reserve_cls_tokens Integer. CLS token slots (default: `4L`).
#' @param ssmax Logical or Character. Scalable softmax type.
#' @param mixed_radix_ensemble Logical. Mixed-radix ensembling (default: `TRUE`).
#' @param recompute Logical. Gradient checkpointing (default: `FALSE`).
#'
#' @return An `nn_module` object.
#'
#' @export
ColEmbedding <- torch::nn_module(
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
    self$embed_dim <- embed_dim
    self$reserve_cls_tokens <- reserve_cls_tokens
    self$feature_group <- feature_group
    self$feature_group_size <- feature_group_size
    self$target_aware <- target_aware
    self$max_classes <- max_classes
    self$affine <- affine
    self$mixed_radix_ensemble <- mixed_radix_ensemble

    is_grouping <- isTRUE(feature_group) || feature_group %in% c("same", "valid")
    in_dim <- if (is_grouping) feature_group_size else 1L
    self$in_linear <- skippable_linear(in_dim, embed_dim)

    self$tf_col <- set_transformer(
      num_blocks = num_blocks, d_model = embed_dim, nhead = nhead,
      dim_feedforward = dim_feedforward, num_inds = num_inds,
      dropout = dropout, activation = activation, norm_first = norm_first,
      bias_free_ln = bias_free_ln, ssmax = ssmax, recompute = recompute
    )

    if (target_aware) {
      self$y_encoder <- if (max_classes > 0L) {
        one_hot_and_linear(max_classes, embed_dim)
      } else {
        nn_linear(1L, embed_dim)
      }
    }

    if (affine) {
      self$out_w <- skippable_linear(embed_dim, embed_dim)
      self$ln_w <- if (norm_first) nn_layer_norm(embed_dim, elementwise_affine = !bias_free_ln) else nn_identity()
      self$out_b <- skippable_linear(embed_dim, embed_dim)
      self$ln_b <- if (norm_first) nn_layer_norm(embed_dim, elementwise_affine = !bias_free_ln) else nn_identity()
    }

    self$inference_mgr <- inference_manager$new(enc_name = "tf_col", out_dim = embed_dim)
  },

  map_feature_shuffle = function(reference_pattern, other_pattern) {
    orig_to_other <- setNames(seq_along(other_pattern) - 1L, other_pattern)
    vapply(reference_pattern, function(feat) orig_to_other[as.character(feat)], integer(1L))
  },

  feature_grouping = function(X) {
    if (isFALSE(self$feature_group)) {
      return(X$unsqueeze(-1L))
    }

    shape <- X$shape
    B <- shape[1L]
    TT <- shape[2L]
    H <- shape[3L]
    size <- self$feature_group_size

    mode <- if (isTRUE(self$feature_group)) "same" else self$feature_group

    if (mode == "same") {
      idxs <- torch_arange(0L, H - 1L, dtype = torch_long(), device = X$device)
      shifted <- lapply(seq_len(size) - 1L, function(i) {
        indices <- as.integer(((idxs + (2L^i)) %% H) + 1L)
        X[, , indices, drop = FALSE]
      })
      torch_stack(shifted, dim = -1L)
    } else if (mode == "valid") {
      x_pad_cols <- (size - H %% size) %% size
      if (x_pad_cols > 0L) {
        X <- nnf_pad(X, pad = c(0L, x_pad_cols), value = 0)
        H <- H + x_pad_cols
      }
      X$view(c(B, TT, -1L, size))
    } else {
      X$unsqueeze(-1L)
    }
  },

  .compute_mixed_radix_bases = function(num_classes) {
    if (num_classes <= self$max_classes) return(as.integer(num_classes))

    D <- ceiling(log(num_classes) / log(self$max_classes))
    k <- min(ceiling(num_classes ^ (1.0 / D)), self$max_classes)
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

  .extract_mixed_radix_digit = function(y, digit_idx, bases) {
    divisor <- 1L
    if (digit_idx + 2L <= length(bases)) {
      for (i in (digit_idx + 2L):length(bases)) {
        divisor <- divisor * bases[i]
      }
    }
    (y$to(dtype = torch_long()) %/% divisor) %% bases[digit_idx + 1L]
  },

  .compute_embeddings = function(features, train_size, y_train = NULL, embed_with_test = FALSE) {
    src <- self$in_linear(features)

    if (!self$target_aware) {
      train_size_arg <- if (embed_with_test) NULL else train_size
      src <- self$tf_col(src, train_size = train_size_arg)
    } else {
      if (is.null(y_train)) {
        value_error("y_train is required when target_aware = TRUE")
      }

      num_classes <- as.integer(y_train$max()$item()) + 1L
      needs_mixed_radix <- self$max_classes > 0L && num_classes > self$max_classes

      if (!needs_mixed_radix) {
        # y_train shape is now (B, 1, train_size) or (B, train_size)
        # y_encoder handles the unsqueeze/squeeze internally for classification
        # For regression, nn_linear expects (B, 1), which is already satisfied
        if (self$max_classes > 0L) {
          y_emb <- self$y_encoder(y_train$to(dtype = torch_float()))
        } else {
          y_emb <- self$y_encoder(y_train$unsqueeze(-1L))
        }
        # use narrow() to extract, modify, and reinsert
        src_train <- src$narrow(3L, 1L, train_size)
        src_train$add_(y_emb)  # modify src in place via narrow view

        train_size_arg <- if (embed_with_test) NULL else train_size
        src <- self$tf_col(src, train_size = train_size_arg)
      } else {
        if (!self$mixed_radix_ensemble) {
          value_error("Too many classes ({num_classes}) for max_classes ({self$max_classes}). Enable mixed_radix_ensemble.")
        }
        bases <- self$.compute_mixed_radix_bases(num_classes)
        num_digits <- length(bases)
        src_accum <- torch_zeros_like(src)
        src_with_y <- src$clone()

        for (digit_idx in seq_len(num_digits) - 1L) {
          y_digit <- self$.extract_mixed_radix_digit(y_train, digit_idx, bases)
          y_emb <- self$y_encoder(y_digit$to(dtype = torch_float()))
          # use narrow() to extract, modify, and reinsert
          src_with_y_train <- src_with_y$narrow(3L, 1L, train_size)
          src_with_y_train$copy_(src$narrow(3L, 1L, train_size) + y_emb)

          train_size_arg <- if (embed_with_test) NULL else train_size
          src_accum <- src_accum + self$tf_col(src_with_y, train_size = train_size_arg)
          }
        src <- src_accum / num_digits
      }
    }

    if (self$affine) {
      weights <- self$ln_w(self$out_w(src))
      biases <- self$ln_b(self$out_b(src))
      features * weights + biases
    } else {
      src
    }
  },

  .train_forward_with_feature_group = function(X, y_train, embed_with_test) {
    train_size <- y_train$size(2L)
    X <- self$feature_grouping(X)

    if (self$reserve_cls_tokens > 0L) {
      X <- nnf_pad(X, pad = c(0L, 0L, self$reserve_cls_tokens, 0L), value = -100.0)
    }

    features <- X$transpose(2L, 3L)

    # if (self$target_aware) {
    #   # y_train becomes (B, 1, train_size, 1) to broadcast with (B, G+C, train_size, E)
    #   y_train <- y_train$unsqueeze(2L)$unsqueeze(-1L)
    # }

    embeddings <- self$.compute_embeddings(features, train_size, y_train, embed_with_test)
    embeddings$transpose(2L, 3L)
  },

  .train_forward_without_feature_group = function(X, y_train, d, embed_with_test) {
    train_size <- y_train$size(2L)

    if (self$reserve_cls_tokens > 0L) {
      X <- nnf_pad(X, pad = c(self$reserve_cls_tokens, 0L), value = -100.0)
    }

    if (is.null(d)) {
      features <- X$transpose(2L, 3L)$unsqueeze(-1L)

      # if (self$target_aware) {
      #   y_train <- y_train$unsqueeze(2L)$unsqueeze(-1L)
      # }

      embeddings <- self$.compute_embeddings(features, train_size, y_train, embed_with_test)
    } else {
      if (self$reserve_cls_tokens > 0L) d <- d + self$reserve_cls_tokens

      B <- X$size(1L)
      HC <- X$size(3L)

      X_t <- X$transpose(2L, 3L)

      indices <- torch_arange(0L, HC - 1L, device = X$device)$unsqueeze(1L)$expand(B, HC)
      mask <- indices < d$unsqueeze(2L)

      features <- X_t[mask]$unsqueeze(-1L)

      # if (self$target_aware) {
      #   y_train <- y_train$unsqueeze(2L)$expand(B, HC, train_size)
      #   y_train <- y_train[mask]
      #   y_train <- y_train$unsqueeze(-1L)
      # }
      if (self$target_aware) {
        y_train <- y_train$unsqueeze(1L)$expand(B, HC, train_size)
        y_train <- y_train[mask]
      }

      eff_emb <- self$.compute_embeddings(features, train_size, y_train, embed_with_test)
      embeddings <- torch_zeros(c(B, HC, X$size(2L), self$embed_dim), device = X$device, dtype = eff_emb$dtype)
      embeddings[mask] <- eff_emb
    }
    embeddings$transpose(2L, 3L)
  },

  .train_forward = function(X, y_train, d = NULL, embed_with_test = FALSE) {
    if (!isFALSE(self$feature_group)) {
      if (!is.null(d)) value_error("d is not supported with feature grouping")
      self$.train_forward_with_feature_group(X, y_train, embed_with_test)
    } else {
      self$.train_forward_without_feature_group(X, y_train, d, embed_with_test)
    }
  },

  .inference_with_feature_group = function(X, y_train, train_size, embed_with_test) {
    X <- self$feature_grouping(X)
    if (self$reserve_cls_tokens > 0L) {
      X <- nnf_pad(X, pad = c(0L, 0L, self$reserve_cls_tokens, 0L), value = -100.0)
    }
    features <- X$transpose(2L, 3L)

    if (!self$target_aware) {
    #   y_train <- y_train$unsqueeze(2L)$unsqueeze(-1L)
    # } else {
      y_train <- NULL
    }

    self$inference_mgr$forward(
      self$.compute_embeddings,
      input = list(features = features, train_size = train_size, y_train = y_train, embed_with_test = embed_with_test)
    )
  },

  .inference_without_feature_group = function(X, y_train, train_size, embed_with_test, feature_shuffles) {
    if (is.null(feature_shuffles)) {
      if (self$reserve_cls_tokens > 0L) {
        X <- nnf_pad(X, pad = c(self$reserve_cls_tokens, 0L), value = -100.0)
      }
      features <- X$transpose(2L, 3L)$unsqueeze(-1L)

      if (!self$target_aware) {
      #   y_train <- y_train$unsqueeze(2L)$unsqueeze(-1L)
      # } else {
        y_train <- NULL
      }

      self$inference_mgr$forward(
        self$.compute_embeddings,
        input = list(features = features, train_size = train_size, y_train = y_train, embed_with_test = embed_with_test)
      )
    } else {
      B <- X$size(1L)
      first_table <- X[1L, , , drop = FALSE]
      if (self$reserve_cls_tokens > 0L) {
        first_table <- nnf_pad(first_table, pad = c(self$reserve_cls_tokens, 0L), value = -100.0)
      }
      features <- first_table$transpose(2L, 3L)$unsqueeze(-1L)$squeeze(1L)

      if (self$target_aware) {
        # y_first <- y_train[1L, ]$unsqueeze(2L)$unsqueeze(-1L)$expand(features$size(1L), train_size, 1L)
        y_first <- y_train[1L, , drop = FALSE]
      } else {
        y_first <- NULL
      }

      first_emb <- self$inference_mgr$forward(
        self$.compute_embeddings,
        input = list(features = features, train_size = train_size, y_train = y_first,
        embed_with_test = embed_with_test, output_repeat = B)
      )

      embeddings <- first_emb$unsqueeze(1L)$`repeat`(c(B, 1L, 1L, 1L))
      first_pattern <- feature_shuffles[[1L]]

      for (i in seq(2L, B)) {
        mapping <- self$map_feature_shuffle(first_pattern, feature_shuffles[[i]])
        if (self$reserve_cls_tokens > 0L) {
          mapping <- mapping + self$reserve_cls_tokens
          mapping <- c(seq_len(self$reserve_cls_tokens) - 1L, mapping)
        }
        embeddings[i, .. ] <- first_emb[mapping + 1L, .. ]
      }
      embeddings
    }
  },

  .inference_forward = function(X, y_train, embed_with_test = FALSE, feature_shuffles = NULL, mgr_config = NULL) {
    if (is.null(mgr_config)) mgr_config <- inference_config()$col
    do.call(self$inference_mgr$configure, mgr_config)

    train_size <- y_train$size(2L)
    embeddings <- if (!isFALSE(self$feature_group)) {
      self$.inference_with_feature_group(X, y_train, train_size, embed_with_test)
    } else {
      self$.inference_without_feature_group(X, y_train, train_size, embed_with_test, feature_shuffles)
    }
    embeddings$transpose(2L, 3L)
  },

  forward = function(X, y_train, d = NULL, embed_with_test = FALSE, feature_shuffles = NULL, mgr_config = NULL) {
    if (self$training) {
      self$.train_forward(X, y_train, d, embed_with_test)
    } else {
      self$.inference_forward(X, y_train, embed_with_test, feature_shuffles, mgr_config)
    }
  },

  .compute_embeddings_with_cache = function(features, col_cache, train_size = NULL, y_train = NULL, use_cache = FALSE, store_cache = TRUE) {
    src <- self$in_linear(features)

    if (!self$target_aware) {
      src <- self$tf_col$forward_with_cache(src, col_cache, train_size, use_cache, store_cache)
    } else {
      if (store_cache) {
        if (is.null(y_train)) value_error("y_train required when store_cache = TRUE")
        if (self$max_classes > 0L) {
          y_emb <- self$y_encoder(y_train$to(dtype = torch_float()))
        } else {
          y_emb <- self$y_encoder(y_train$unsqueeze(-1L))
        }
        # y_emb is (B, train_size, E)
        y_emb <- y_emb$unsqueeze(2L)  # -> (B, 1, train_size, E) will allow broadcasting

        # src in-place modification via narrow()
        src_train <- src$narrow(3L, 1L, train_size)
        src_train$add_(y_emb)
      }
      src <- self$tf_col$forward_with_cache(src, col_cache, train_size, use_cache, store_cache)
    }

    if (self$affine) {
      features * self$ln_w(self$out_w(src)) + self$ln_b(self$out_b(src))
    } else {
      src
    }
  },

  forward_with_cache = function(X, col_cache, y_train = NULL, use_cache = FALSE, store_cache = TRUE, mgr_config = NULL) {
    if (!xor(use_cache, store_cache)) value_error("Exactly one of use_cache or store_cache must be TRUE")
    if (store_cache && is.null(y_train)) value_error("y_train required when store_cache = TRUE")

    if (store_cache && self$target_aware && self$max_classes > 0L) {
      num_classes <- as.integer(y_train$max()$item()) + 1L
      if (num_classes > self$max_classes) {
        runtime_error("KV caching not supported for >{self$max_classes} classes. Use mixed_radix_ensemble=False or reduce classes.")
      }
    }

    if (is.null(mgr_config)) mgr_config <- inference_config()$col
    do.call(self$inference_mgr$configure, mgr_config)

    if (!isFALSE(self$feature_group)) {
      X <- self$feature_grouping(X)
      if (self$reserve_cls_tokens > 0L) X <- nnf_pad(X, pad = c(0L, 0L, self$reserve_cls_tokens, 0L), value = -100.0)
      features <- X$transpose(2L, 3L)
    } else {
      if (self$reserve_cls_tokens > 0L) X <- nnf_pad(X, pad = c(self$reserve_cls_tokens, 0L), value = -100.0)
      features <- X$transpose(2L, 3L)$unsqueeze(-1L)
    }

    if (store_cache) {
      train_size <- y_train$size(2L)
      # y_train <- y_train$unsqueeze(2L)$unsqueeze(-1L)
    } else {
      train_size <- NULL
      y_train <- NULL
    }

    embeddings <- self$inference_mgr$forward(
      self$.compute_embeddings_with_cache,
      input = list(features = features, col_cache = col_cache, train_size = train_size, y_train = y_train, use_cache = use_cache, store_cache = store_cache)
    )
    embeddings$transpose(2L, 3L)
  }
)
