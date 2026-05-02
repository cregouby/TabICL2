#' Get unique values from a tensor
#' @param x torch tensor
#' @return torch tensor with unique sorted values
#' @noRd
torch_unique <- function(x) {
  # Flatten and sort to make duplicates consecutive
  x_flat <- x$flatten()
  x_sorted <- x_flat$sort()[[1L]]

  # Use torch_unique_consecutive to eliminate consecutive duplicates
  # Returns list: [[1]] unique values, what we only need
  torch_unique_consecutive(x_sorted)[[1L]]
}

#' Dataset-wise in-context learning.
#'
#' @param max_classes integer. Determines the task type and output behavior.
#'   If 0: regression using quantile prediction. If > 0: classification with
#'   hierarchical support if dataset classes exceed this value.
#' @param out_dim integer. Output dimension of the model.
#' @param d_model integer. Model dimension.
#' @param num_blocks integer. Number of blocks used in the ICL encoder.
#' @param nhead integer. Number of attention heads of the ICL encoder.
#' @param dim_feedforward integer. Dimension of the feedforward network.
#' @param dropout float. Dropout probability. Default 0.0.
#' @param activation str or function. Activation function ("relu", "gelu").
#' @param norm_first logical. If TRUE, uses pre-norm architecture.
#' @param bias_free_ln logical. If TRUE, removes bias from LayerNorm layers.
#' @param ssmax logical or str. Type of scalable softmax to use.
#' @param recompute logical. If TRUE, uses gradient checkpointing.
#'
#' @importFrom torch nn_module nn_layer_norm nn_linear nn_sequential nn_gelu torch_zeros torch_tensor torch_cat nnf_softmax torch_log torch_stack torch_searchsorted
#' @export
nn_ic_learning <- nn_module(
  "nn_ic_learning",
  initialize = function(max_classes,
                        out_dim,
                        d_model,
                        num_blocks,
                        nhead,
                        dim_feedforward,
                        dropout = 0.0,
                        activation = "gelu",
                        norm_first = TRUE,
                        bias_free_ln = FALSE,
                        ssmax = FALSE,
                        recompute = FALSE) {
    self$max_classes <- max_classes
    self$norm_first <- norm_first

    self$tf_icl <- Encoder(
      num_blocks = num_blocks,
      d_model = d_model,
      nhead = nhead,
      dim_feedforward = dim_feedforward,
      dropout = dropout,
      activation = activation,
      norm_first = norm_first,
      bias_free_ln = bias_free_ln,
      ssmax = ssmax,
      recompute = recompute
    )

    if (self$norm_first) {
      self$ln <- nn_layer_norm(d_model, elementwise_affine = !bias_free_ln)
    }

    if (max_classes > 0) {
      self$y_encoder <- one_hot_and_linear(max_classes, d_model)
    } else {
      self$y_encoder <- nn_linear(1, d_model)
    }

    self$decoder <- nn_sequential(
      nn_linear(d_model, d_model * 2),
      nn_gelu(),
      nn_linear(d_model * 2, out_dim)
    )

    self$inference_mgr <- inference_manager$new(enc_name = "tf_icl", out_dim = out_dim)
  },

  .grouping = function(num_classes) {
    if (num_classes <= self$max_classes) {
      return(list(torch_zeros(num_classes, dtype = torch_int()), 1))
    }

    num_groups <- min(ceiling(num_classes / self$max_classes), self$max_classes)
    group_assignments <- torch_zeros(num_classes, dtype = torch_int())
    current_pos <- 1 # 1-indexing

    remaining_classes <- num_classes
    remaining_groups <- num_groups

    for (i in seq_len(num_groups)) {
      group_size <- ceiling(remaining_classes / remaining_groups)
      group_assignments[current_pos:(current_pos + group_size - 1)] <- i - 1
      current_pos <- current_pos + group_size
      remaining_classes <- remaining_classes - group_size
      remaining_groups <- remaining_groups - 1
    }

    list(group_assignments, num_groups)
  },

  .fit_node = function(node, R, y, current_depth) {
    unique_classes <- torch_unique(y)
    node$classes_ <- unique_classes

    # Get number of unique classes as R integer
    num_unique <- as.integer(unique_classes$shape[1])

    if (num_unique <= self$max_classes) {
      node$is_leaf <- TRUE
      node$R <- R
      node$y <- y
      return(NULL)
    }

    groups <- self$.grouping(num_unique)
    group_assignments <- groups[[1]]
    num_groups <- groups[[2]]

    class_keys <- as.character(as.integer(unique_classes))
    node$class_mapping <- setNames(as.integer(group_assignments), class_keys)

    y_vec <- as.integer(y)
    node$group_indices <- torch_tensor(unname(node$class_mapping[as.character(y_vec)]), dtype = torch_int())
    node$R <- R
    node$y <- y
    node$is_leaf <- FALSE
    node$child_nodes <- list()

    for (group in seq_len(num_groups)) {
      mask <- node$group_indices == (group - 1)
      child_node <- list(depth = current_depth + 1, child_nodes = list())
      self$.fit_node(child_node, R[mask, ..], y[mask], current_depth + 1)
      node$child_nodes[[group]] <- child_node
    }
  },

  .fit_hierarchical = function(R_train, y_train) {
    self$root <- list(depth = 0, child_nodes = list())
    self$.fit_node(self$root, R_train, y_train, 0)
  },

  .label_encoding = function(y) {
    unique_vals <- torch_unique(y)
    indices <- unique_vals$sort()[[2]]
    unique_vals_sorted <- unique_vals[indices]
    indices[torch_searchsorted(unique_vals_sorted, y)]
  },

  .icl_predictions = function(R, y_train) {
    train_size <- y_train$shape[2]
    if (self$max_classes > 0) {
      Ry_train <- self$y_encoder(y_train$to(dtype = torch_float()))
    } else {
      Ry_train <- self$y_encoder(y_train$unsqueeze(-1))
    }

    R[, 1:train_size, ..] <- R[, 1:train_size, ..] + Ry_train

    src <- self$tf_icl(R, train_size = train_size)
    if (self$norm_first) {
      src <- self$ln(src)
    }
    self$decoder(src)
  },

  .predict_standard = function(R, y_train, return_logits = FALSE, softmax_temperature = 0.9, auto_batch = TRUE) {
    out <- self$inference_mgr$forward(
      self$.icl_predictions,
      inputs = list(R = R, y_train = y_train),
      auto_batch = auto_batch
    )

    train_size <- y_train$shape[2]
    seq_len <- out$shape[2]
    if (self$max_classes == 0) {
      out <- out[, (train_size + 1):seq_len, ..]
    } else {
      num_classes <- as.integer(torch_unique(y_train[1, ..])$shape[1])
      out <- out[, (train_size + 1):seq_len, 1:num_classes]
      if (!return_logits) {
        out <- nnf_softmax(out / softmax_temperature, dim = -1)
      }
    }
    out
  },

  .predict_hierarchical = function(R_test, softmax_temperature = 0.9) {
    test_size <- R_test$shape[1]
    device <- R_test$device
    num_classes <- length(self$root$classes_)

    process_node <- function(node, R_test) {
      node_R <- torch_cat(list(node$R$to(device = device), R_test), dim = 1)

      if (node$is_leaf) {
        node_y <- self$.label_encoding(node$y$to(device = device))
        leaf_preds <- self$.predict_standard(
          R = node_R$unsqueeze(1),
          y_train = node_y$unsqueeze(1),
          softmax_temperature = softmax_temperature,
          auto_batch = FALSE
        )$squeeze(1)

        global_preds <- torch_zeros(test_size, num_classes, device = device)
        class_indices <- as.integer(node$classes_)
        for (i in seq_along(class_indices)) {
          global_preds[, class_indices[i]] <- leaf_preds[, i]
        }
        return(global_preds)
      }

      final_probs <- torch_zeros(test_size, num_classes, device = device)
      node_y <- node$group_indices$to(device = device)
      group_probs <- self$.predict_standard(
        R = node_R$unsqueeze(1),
        y_train = node_y$unsqueeze(1),
        softmax_temperature = softmax_temperature,
        auto_batch = FALSE
      )$squeeze(1)

      for (group_idx in seq_along(node$child_nodes)) {
        child_probs <- process_node(node$child_nodes[[group_idx]], R_test)
        final_probs <- final_probs + child_probs * group_probs[, group_idx, drop = FALSE]
      }
      final_probs
    }

    process_node(self$root, R_test)
  },

  .inference_forward = function(R, y_train, return_logits = TRUE, softmax_temperature = 0.9, mgr_config = NULL) {
    if (is.null(mgr_config)) {
      mgr_config <- list()
    }
    do.call(self$inference_mgr$configure, mgr_config)

    if (self$max_classes == 0) {
      out <- self$.predict_standard(R, y_train)
    } else {
      num_classes <- as.integer(torch_unique(y_train[1, ..])$shape[1])

      for (i in 1:y_train$shape[1]) {
        if (as.integer(torch_unique(y_train[i, ..])$shape[1]) != num_classes) {
          cli_abort("All tables must have the same number of classes")
        }
      }

      if (num_classes <= self$max_classes) {
        out <- self$.predict_standard(R, y_train, return_logits, softmax_temperature)
      } else {
        out_list <- list()
        train_size <- y_train$shape[2]
        for (i in 1:R$shape[1]) {
          ri <- R[i, ..]
          yi <- y_train[i, ..]

          if (!is.null(mgr_config$offload) && mgr_config$offload) {
            ri <- ri$cpu()
            yi <- yi$cpu()
          }

          self$.fit_hierarchical(ri[1:train_size, ..], yi)
          ri_len <- ri$shape[1]
          probs <- self$.predict_hierarchical(ri[(train_size + 1):ri_len, ..], softmax_temperature)
          out_list[[i]] <- probs
        }
        out <- torch_stack(out_list, dim = 1)
        if (return_logits) {
          out <- softmax_temperature * torch_log(out + 1e-6)
        }
      }
    }
    out
  },

  forward = function(R, y_train, return_logits = TRUE, softmax_temperature = 0.9, mgr_config = NULL) {
    if (self$training) {
      train_size <- y_train$shape[2]
      out <- self$.icl_predictions(R, y_train)
      seq_len <- out$shape[2]
      out <- out[, (train_size + 1):seq_len, ..]
    } else {
      out <- self$.inference_forward(R, y_train, return_logits, softmax_temperature, mgr_config)
    }
    out
  },

  prepare_repr_cache = function(R, y_train) {
    train_size <- y_train$shape[2]
    if (self$max_classes > 0) {
      Ry_train <- self$y_encoder(y_train$to(dtype = torch_float()))
    } else {
      Ry_train <- self$y_encoder(y_train$unsqueeze(-1))
    }
    R[, 1:train_size, ..] <- R[, 1:train_size, ..] + Ry_train
    R
  },

  .icl_predictions_repr_cache = function(R, train_size) {
    src <- self$tf_icl(R, train_size = train_size)
    if (self$norm_first) {
      src <- self$ln(src)
    }
    self$decoder(src)
  },

  forward_with_repr_cache = function(R, train_size, num_classes = NULL, return_logits = TRUE, softmax_temperature = 0.9, mgr_config = NULL) {
    if (is.null(mgr_config)) mgr_config <- list()
    do.call(self$inference_mgr$configure, mgr_config)

    out <- self$inference_mgr$forward(
      self$.icl_predictions_repr_cache,
      inputs = list(R = R, train_size = train_size)
    )

    seq_len <- out$shape[2]
    out <- out[, (train_size + 1):seq_len, ..]
    if (self$max_classes > 0) {
      if (is.null(num_classes)) cli_abort("num_classes must be provided for classification")
      out <- out[.., 1:num_classes]
      if (!return_logits) {
        out <- nnf_softmax(out / softmax_temperature, dim = -1)
      }
    }
    out
  },

  .icl_predictions_with_cache = function(R, icl_cache, y_train = NULL, use_cache = FALSE, store_cache = TRUE) {
    if (store_cache) {
      if (is.null(y_train)) cli_abort("y_train must be provided when store_cache=TRUE")
      train_size <- y_train$shape[2]

      if (self$max_classes > 0) {
        Ry_train <- self$y_encoder(y_train$to(dtype = torch_float()))
      } else {
        Ry_train <- self$y_encoder(y_train$unsqueeze(-1))
      }
      R[, 1:train_size, ..] <- R[, 1:train_size, ..] + Ry_train
    }

    src <- self$tf_icl$forward_with_cache(
      R,
      icl_cache = icl_cache,
      train_size = if (store_cache) train_size else NULL,
      use_cache = use_cache,
      store_cache = store_cache
    )

    if (self$norm_first) {
      src <- self$ln(src)
    }
    self$decoder(src)
  },

  forward_with_cache = function(R, icl_cache, y_train = NULL, num_classes = NULL, return_logits = TRUE, softmax_temperature = 0.9, use_cache = FALSE, store_cache = TRUE, mgr_config = NULL) {
    if (use_cache == store_cache) {
      runtime_error("Exactly one of use_cache or store_cache must be TRUE")
    }

    if (store_cache) {
      if (is.null(y_train)) cli_abort("y_train must be provided when store_cache=TRUE")
      if (self$max_classes > 0) {
        num_classes <- as.integer(torch_unique(y_train[1, ..])$shape[1])
        if (num_classes > self$max_classes) {
          value_error("KV caching is not supported for classification with more classes ({num_classes}) than max_classes ({self$max_classes}).")
        }
      }
    } else {
      if (is.null(num_classes)) cli_abort("num_classes must be provided when use_cache=TRUE")
    }

    if (is.null(mgr_config)) mgr_config <- list()
    do.call(self$inference_mgr$configure, mgr_config)

    out <- self$inference_mgr$forward(
      self$.icl_predictions_with_cache,
      inputs = list(
        R = R,
        icl_cache = icl_cache,
        y_train = y_train,
        use_cache = use_cache,
        store_cache = store_cache
      )
    )

    if (store_cache) {
      train_size <- y_train$shape[2]
      seq_len <- out$shape[2]
      out <- out[, (train_size + 1):seq_len, ..]
    }

    if (self$max_classes > 0) {
      out <- out[.., 1:num_classes]
      if (!return_logits) {
        out <- nnf_softmax(out / softmax_temperature, dim = -1)
      }
    }
    out
  }
)
