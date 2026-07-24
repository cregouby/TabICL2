# Test fixture helpers
make_entry <- function(batch = 2, heads = 4, seq = 10, dim = 64,
                        device = "cpu", dtype = torch_float32()) {
  KVCacheEntry$new(
    key = torch_randn(batch, heads, seq, dim, device = device, dtype = dtype),
    value = torch_randn(batch, heads, seq, dim, device = device, dtype = dtype)
  )
}

make_empty_entry <- function() {
  KVCacheEntry$new()
}

make_cache <- function(n_layers = 3, batch = 2, heads = 4, seq = 10, dim = 64) {
  kv <- list()
  for (i in seq_len(n_layers)) {
    kv[[as.character(i)]] <- make_entry(batch, heads, seq, dim)
  }
  KVCache$new(kv = kv)
}

make_tab_cache <- function(batch = 2, train_size = 100, n_features = 20,
                            n_col_layers = 3, n_icl_layers = 4,
                            num_classes = 10) {
  col_cache <- make_cache(n_col_layers, batch, heads = 4, seq = train_size, dim = 64)
  icl_cache <- make_cache(n_icl_layers, batch, heads = 8, seq = train_size, dim = 64)
  row_repr <- torch_randn(batch, train_size, 512)

  TabICLCache$new(
    col_cache = col_cache,
    row_repr = row_repr,
    icl_cache = icl_cache,
    train_shape = c(batch, train_size, n_features),
    num_classes = num_classes
  )
}



test_that("KVCacheEntry initialization creates empty invalid entry", {
  entry <- KVCacheEntry$new()
  expect_null(entry$key)
  expect_null(entry$value)
  expect_false(entry$is_valid())
})

test_that("KVCacheEntry with key and value tensors is valid", {
  key <- torch_randn(2, 4, 10, 64)
  value <- torch_randn(2, 4, 10, 64)
  entry <- KVCacheEntry$new(key = key, value = value)

  expect_true(entry$is_valid())
  expect_tensor(entry$key)
  expect_tensor(entry$value)
  expect_tensor_shape(entry$key, c(2, 4, 10, 64))
  expect_tensor_shape(entry$value, c(2, 4, 10, 64))
})

test_that("KVCacheEntry with only key is invalid", {
  entry <- KVCacheEntry$new(key = torch_randn(2, 4, 10, 64))
  expect_false(entry$is_valid())
})

test_that("KVCacheEntry with only value is invalid", {
  entry <- KVCacheEntry$new(value = torch_randn(2, 4, 10, 64))
  expect_false(entry$is_valid())
})

test_that("KVCacheEntry slice on invalid entry returns empty entry", {
  empty <- make_empty_entry()
  sliced <- empty$slice(1)
  expect_false(sliced$is_valid())
})

test_that("KVCacheEntry slice reduces batch dimension correctly", {
  entry <- make_entry(batch = 4, heads = 4, seq = 10, dim = 64)
  sliced <- entry$slice(1:2)

  expect_true(sliced$is_valid())
  expect_tensor_shape(sliced$key, c(2, 4, 10, 64))
  expect_tensor_shape(sliced$value, c(2, 4, 10, 64))
})

test_that("KVCacheEntry slice with single index returns one batch", {
  entry <- make_entry(batch = 4, heads = 4, seq = 10, dim = 64)
  sliced <- entry$slice(4)

  expect_true(sliced$is_valid())
  expect_tensor_shape(sliced$key, c(1, 4, 10, 64))
})

test_that("KVCacheEntry write updates batch slice in place", {
  entry <- make_entry(batch = 4, heads = 4, seq = 10, dim = 64)
  other <- make_entry(batch = 2, heads = 4, seq = 10, dim = 64)

  entry$write(1:2, other)

  expect_true(entry$is_valid())
  expect_tensor_shape(entry$key, c(4, 4, 10, 64))
})

test_that("KVCacheEntry write does nothing when other is invalid", {
  entry <- make_entry(batch = 4)
  other <- make_empty_entry()

  result <- entry$write(1:2, other)
  expect_true(entry$is_valid())
  expect_tensor_shape(entry$key, c(4, 4, 10, 64))
})

test_that("KVCacheEntry write does nothing when self is invalid", {
  entry <- make_empty_entry()
  other <- make_entry(batch = 2)

  result <- entry$write(1:2, other)
  expect_false(entry$is_valid())
})

test_that("KVCacheEntry to moves tensors to new dtype", {
  entry <- make_entry(batch = 2, dtype = torch_float32())

  moved <- entry$to(device = "cpu", dtype = torch_float64())

  expect_true(moved$is_valid())
  expect_tensor_dtype(moved$key, torch_float64())
  expect_tensor_dtype(moved$value, torch_float64())
})

test_that("KVCacheEntry to on invalid entry returns empty entry", {
  empty <- make_empty_entry()
  moved <- empty$to(device = "cpu")
  expect_false(moved$is_valid())
})

test_that("KVCacheEntry stores and retrieves correct tensor values", {
  key <- torch_tensor(array(1:24, dim = c(2, 3, 2, 2)))
  value <- torch_tensor(array(25:48, dim = c(2, 3, 2, 2)))
  entry <- KVCacheEntry$new(key = key, value = value)

  expect_equal_to_r(entry$key[1, 1, 1, 1], 1)
  expect_equal_to_r(entry$value[1, 1, 1, 1], 25)
})



# kv_cache_entry_concat Tests

test_that("kv_cache_entry_concat of empty list returns empty entry", {
  result <- kv_cache_entry_concat(list())
  expect_false(result$is_valid())
})

test_that("kv_cache_entry_concat of single entry preserves shape", {
  e1 <- make_entry(batch = 2)
  result <- kv_cache_entry_concat(list(e1), dim = 1)

  expect_true(result$is_valid())
  expect_tensor_shape(result$key, c(2, 4, 10, 64))
})

test_that("kv_cache_entry_concat of multiple entries stacks batches", {
  e1 <- make_entry(batch = 2)
  e2 <- make_entry(batch = 3)
  result <- kv_cache_entry_concat(list(e1, e2), dim = 1)

  expect_true(result$is_valid())
  expect_tensor_shape(result$key, c(5, 4, 10, 64))
  expect_tensor_shape(result$value, c(5, 4, 10, 64))
})

test_that("kv_cache_entry_concat skips invalid entries silently", {
  e1 <- make_entry(batch = 2)
  e2 <- make_empty_entry()
  e3 <- make_entry(batch = 3)

  result <- kv_cache_entry_concat(list(e1, e2, e3), dim = 1)

  expect_true(result$is_valid())
  expect_tensor_shape(result$key, c(5, 4, 10, 64))
})

test_that("kv_cache_entry_concat of all invalid entries returns empty", {
  e1 <- make_empty_entry()
  e2 <- make_empty_entry()

  result <- kv_cache_entry_concat(list(e1, e2))
  expect_false(result$is_valid())
})

test_that("kv_cache_entry_concat preserves float64 dtype", {
  e1 <- make_entry(batch = 2, dtype = torch_float64())
  e2 <- make_entry(batch = 2, dtype = torch_float64())
  result <- kv_cache_entry_concat(list(e1, e2))

  expect_tensor_dtype(result$key, torch_float64())
})



# KVCache Tests

test_that("KVCache initialization creates empty cache", {
  cache <- KVCache$new()
  expect_equal(length(cache$kv), 0)
  expect_false(cache$is_populated())
})

test_that("KVCache initialization with entries is populated", {
  kv <- list(
    layer1 = make_entry(batch = 2),
    layer2 = make_entry(batch = 2)
  )
  cache <- KVCache$new(kv = kv)

  expect_equal(length(cache$kv), 2)
  expect_true(cache$is_populated())
})

test_that("KVCache is_populated returns false for all invalid entries", {
  kv <- list(
    layer1 = make_empty_entry(),
    layer2 = make_empty_entry()
  )
  cache <- KVCache$new(kv = kv)

  expect_false(cache$is_populated())
})

test_that("KVCache slice reduces batch dimension for all entries", {
  cache <- make_cache(n_layers = 3, batch = 4)
  sliced <- cache$slice(1:2)

  expect_equal(length(sliced$kv), 3)
  expect_true(sliced$is_populated())
  expect_tensor_shape(sliced$kv[["1"]]$key, c(2, 4, 10, 64))
})

test_that("KVCache slice preserves empty entries as empty", {
  kv <- list(
    layer1 = make_entry(batch = 4),
    layer2 = make_empty_entry()
  )
  cache <- KVCache$new(kv = kv)
  sliced <- cache$slice(1:2)

  expect_true(sliced$kv[["1"]]$is_valid())
  expect_false(sliced$kv[["2"]]$is_valid())
})

test_that("KVCache write updates matching layer indices", {
  cache <- make_cache(n_layers = 2, batch = 4)
  other <- make_cache(n_layers = 2, batch = 2)

  cache$write(1:2, other)

  expect_true(cache$is_populated())
  expect_tensor_shape(cache$kv[["1"]]$key, c(4, 4, 10, 64))
})

test_that("KVCache write skips non matching layer indices", {
  cache <- make_cache(n_layers = 2, batch = 4)
  other_kv <- list(layer3 = make_entry(batch = 2))
  other <- KVCache$new(kv = other_kv)

  cache$write(1:2, other)

  expect_true(cache$kv[["1"]]$is_valid())
})

test_that("KVCache write errors on invalid target entry", {
  cache_kv <- list(layer1 = make_empty_entry())
  cache <- KVCache$new(kv = cache_kv)
  other <- make_cache(n_layers = 1, batch = 2)

  expect_error(cache$write(1:2, other), "not valid")
})

test_that("KVCache to moves all entries to new dtype", {
  cache <- make_cache(n_layers = 3, batch = 2)
  moved <- cache$to(device = "cpu", dtype = torch_float64())

  expect_equal(length(moved$kv), 3)
  expect_true(moved$is_populated())
  expect_tensor_dtype(moved$kv[["1"]]$key, torch_float64())
})

test_that("KVCache to preserves empty entries", {
  kv <- list(
    layer1 = make_entry(batch = 2),
    layer2 = make_empty_entry()
  )
  cache <- KVCache$new(kv = kv)
  moved <- cache$to(device = "cpu")

  expect_true(moved$kv[["1"]]$is_valid())
  expect_false(moved$kv[["2"]]$is_valid())
})

test_that("KVCache preallocate creates zero filled tensors", {
  reference <- make_cache(n_layers = 2, batch = 1)
  cache <- KVCache$new()
  cache$preallocate(reference, batch_shape = c(4), device = "cpu")

  expect_equal(length(cache$kv), 2)
  expect_true(cache$is_populated())
  expect_tensor_shape(cache$kv[["1"]]$key, c(4, 4, 10, 64))
  expect_equal_to_r(cache$kv[["1"]]$key[1, 1, 1, 1], 0)
})

test_that("KVCache preallocate with custom dtype sets correct dtype", {
  reference <- make_cache(n_layers = 1, batch = 1)
  cache <- KVCache$new()
  cache$preallocate(reference, batch_shape = c(2))

  expect_tensor_dtype(cache$kv[["1"]]$key, torch_int64())
})

test_that("KVCache preallocate skips invalid reference entries", {
  kv <- list(layer1 = make_empty_entry())
  reference <- KVCache$new(kv = kv)
  cache <- KVCache$new()
  cache$preallocate(reference, batch_shape = c(2))

  expect_equal(length(cache$kv), 0)
})



# kv_cache_concat Tests

test_that("kv_cache_concat of empty list returns empty cache", {
  result <- kv_cache_concat(list())
  expect_equal(length(result$kv), 0)
  expect_false(result$is_populated())
})

test_that("kv_cache_concat of single cache returns equivalent cache", {
  c1 <- make_cache(n_layers = 2, batch = 2)
  result <- kv_cache_concat(list(c1))

  expect_equal(length(result$kv), 2)
  expect_true(result$is_populated())
})

test_that("kv_cache_concat of multiple caches stacks batch dimensions", {
  c1 <- make_cache(n_layers = 2, batch = 2)
  c2 <- make_cache(n_layers = 2, batch = 3)
  result <- kv_cache_concat(list(c1, c2), dim = 1)

  expect_equal(length(result$kv), 2)
  expect_tensor_shape(result$kv[["1"]]$key, c(5, 4, 10, 64))
})

test_that("kv_cache_concat handles mismatched layer indices gracefully", {
  kv1 <- list(layer1 = make_entry(batch = 2))
  kv2 <- list(layer1 = make_entry(batch = 2), layer2 = make_entry(batch = 2))
  c1 <- KVCache$new(kv = kv1)
  c2 <- KVCache$new(kv = kv2)

  result <- kv_cache_concat(list(c1, c2))

  expect_equal(length(result$kv), 2)
  expect_true(result$kv[["layer1"]]$is_valid())
  expect_true(result$kv[["layer2"]]$is_valid())
  expect_tensor_shape(result$kv[["layer2"]]$key, c(2, 4, 10, 64))
})

test_that("kv_cache_concat preserves float64 dtype", {
  c1 <- make_cache(n_layers = 1, batch = 2, dtype = torch_float64())
  c2 <- make_cache(n_layers = 1, batch = 2, dtype = torch_float64())
  result <- kv_cache_concat(list(c1, c2))

  expect_tensor_dtype(result$kv[["1"]]$key, torch_float64())
})



# TabICLCache Tests

test_that("TabICLCache initialization with defaults creates empty cache", {
  cache <- TabICLCache$new()

  expect_false(cache$is_empty())
  expect_equal(cache$train_shape, c(0, 0, 0))
  expect_null(cache$num_classes)
  expect_equal(cache$cache_type(), "empty")
})

test_that("TabICLCache initialization with all fields sets properties correctly", {
  col_cache <- make_cache(n_layers = 2, batch = 2)
  icl_cache <- make_cache(n_layers = 3, batch = 2)
  row_repr <- torch_randn(2, 100, 512)

  cache <- TabICLCache$new(
    col_cache = col_cache,
    row_repr = row_repr,
    icl_cache = icl_cache,
    train_shape = c(2, 100, 20),
    num_classes = 10
  )

  expect_equal(cache$train_shape, c(2, 100, 20))
  expect_equal(cache$num_classes, 10)
  expect_equal(cache$cache_type(), "repr")
})

test_that("TabICLCache cache_type returns kv when only sub caches present", {
  col_cache <- make_cache(n_layers = 1, batch = 2)
  cache <- TabICLCache$new(
    col_cache = col_cache,
    row_repr = NULL,
    icl_cache = NULL
  )

  expect_equal(cache$cache_type(), "kv")
})

test_that("TabICLCache cache_type returns empty for truly empty cache", {
  cache <- TabICLCache$new(
    col_cache = KVCache$new(),
    row_repr = NULL,
    icl_cache = KVCache$new()
  )

  expect_equal(cache$cache_type(), "empty")
})

test_that("TabICLCache is_empty returns true for empty cache", {
  cache <- TabICLCache$new(
    col_cache = KVCache$new(),
    row_repr = NULL,
    icl_cache = KVCache$new()
  )

  expect_true(cache$is_empty())
})

test_that("TabICLCache is_empty returns false when row_repr present", {
  cache <- TabICLCache$new(row_repr = torch_randn(2, 100, 512))
  expect_false(cache$is_empty())
})

test_that("TabICLCache is_empty returns false when col_cache populated", {
  cache <- TabICLCache$new(col_cache = make_cache(n_layers = 1, batch = 2))
  expect_false(cache$is_empty())
})

test_that("TabICLCache cache_size_mb is zero for empty cache", {
  cache <- TabICLCache$new()
  expect_equal(cache$cache_size_mb(), 0)
})

test_that("TabICLCache cache_size_mb counts all tensors correctly", {
  cache <- make_tab_cache(batch = 2, train_size = 100, n_features = 20)
  size_mb <- cache$cache_size_mb()

  expect_true(size_mb > 0)
  expect_true(is.integer(size_mb))
})

test_that("TabICLCache slice_batch reduces batch dimension", {
  cache <- make_tab_cache(batch = 4, train_size = 100, n_features = 20)
  sliced <- cache$slice_batch(start = 1, end = 3)

  expect_equal(sliced$train_shape[1], 2)
  expect_equal(sliced$train_shape[2], 100)
  expect_equal(sliced$train_shape[3], 20)
  expect_equal(sliced$num_classes, 10)
})

test_that("TabICLCache slice_batch handles row_repr slicing", {
  cache <- make_tab_cache(batch = 4)
  sliced <- cache$slice_batch(start = 2, end = 4)

  expect_tensor_shape(sliced$row_repr, c(2, 100, 512))
})

test_that("TabICLCache slice_batch handles NULL row_repr", {
  cache <- TabICLCache$new(
    col_cache = make_cache(n_layers = 1, batch = 4),
    row_repr = NULL,
    train_shape = c(4, 100, 20)
  )
  sliced <- cache$slice_batch(start = 1, end = 3)

  expect_null(sliced$row_repr)
  expect_equal(sliced$train_shape[1], 2)
})

test_that("TabICLCache slice_batch handles single batch element", {
  cache <- make_tab_cache(batch = 4)
  sliced <- cache$slice_batch(start = 4, end = 5)

  expect_equal(sliced$train_shape[1], 1)
})

test_that("TabICLCache to moves all tensors to new dtype", {
  cache <- make_tab_cache(batch = 2)
  moved <- cache$to(device = "cpu", dtype = torch_float64())

  expect_equal(moved$train_shape, cache$train_shape)
  expect_equal(moved$num_classes, cache$num_classes)
  expect_tensor_dtype(moved$row_repr, torch_float64())
})

test_that("TabICLCache to handles NULL row_repr gracefully", {
  cache <- TabICLCache$new(
    col_cache = make_cache(n_layers = 1, batch = 2),
    row_repr = NULL,
    train_shape = c(2, 100, 20)
  )
  moved <- cache$to(device = "cpu")

  expect_null(moved$row_repr)
  expect_true(moved$col_cache$is_populated())
})

test_that("TabICLCache to preserves empty sub caches", {
  cache <- TabICLCache$new()
  moved <- cache$to(device = "cpu")

  expect_equal(moved$cache_type(), "empty")
})



# tabicl_cache_concat Tests

test_that("tabicl_cache_concat of empty list returns empty TabICLCache", {
  result <- tabicl_cache_concat(list())

  expect_equal(result$train_shape[1], 0)
  expect_equal(result$cache_type(), "empty")
})

test_that("tabicl_cache_concat of single cache preserves properties", {
  c1 <- make_tab_cache(batch = 2)
  result <- tabicl_cache_concat(list(c1))

  expect_equal(result$train_shape[1], 2)
  expect_equal(result$num_classes, 10)
})

test_that("tabicl_cache_concat sums batch sizes across caches", {
  c1 <- make_tab_cache(batch = 2, train_size = 100, n_features = 20)
  c2 <- make_tab_cache(batch = 3, train_size = 100, n_features = 20)
  result <- tabicl_cache_concat(list(c1, c2))

  expect_equal(result$train_shape[1], 5)
  expect_equal(result$train_shape[2], 100)
  expect_equal(result$train_shape[3], 20)
})

test_that("tabicl_cache_concat handles missing row_repr in some caches", {
  c1 <- make_tab_cache(batch = 2)
  c1$row_repr <- NULL
  c2 <- make_tab_cache(batch = 3)

  result <- tabicl_cache_concat(list(c1, c2))

  expect_equal(result$train_shape[1], 5)
})

test_that("tabicl_cache_concat handles missing col_cache in some caches", {
  c1 <- make_tab_cache(batch = 2)
  c1$col_cache <- NULL
  c2 <- make_tab_cache(batch = 3)

  result <- tabicl_cache_concat(list(c1, c2))

  expect_equal(result$train_shape[1], 5)
})

test_that("tabicl_cache_concat preserves num_classes from first cache", {
  c1 <- make_tab_cache(batch = 2, num_classes = 5)
  c2 <- make_tab_cache(batch = 3, num_classes = 5)
  result <- tabicl_cache_concat(list(c1, c2))

  expect_equal(result$num_classes, 5)
})

test_that("tabicl_cache_concat of fully empty caches returns empty", {
  c1 <- TabICLCache$new(train_shape = c(2, 100, 20))
  c2 <- TabICLCache$new(train_shape = c(3, 100, 20))
  result <- tabicl_cache_concat(list(c1, c2))

  expect_equal(result$train_shape[1], 5)
  expect_equal(result$cache_type(), "empty")
})



# Integration Tests

test_that("full workflow create slice concat and move cache", {
  cache1 <- make_tab_cache(batch = 2, train_size = 50, n_features = 10)
  cache2 <- make_tab_cache(batch = 3, train_size = 50, n_features = 10)

  sliced <- cache1$slice_batch(start = 1, end = 2)
  expect_equal(sliced$train_shape[1], 1)

  combined <- tabicl_cache_concat(list(cache1, cache2))
  expect_equal(combined$train_shape[1], 5)

  moved <- combined$to(device = "cpu", dtype = torch_float64())
  expect_tensor_dtype(moved$row_repr, torch_float64())

  mb <- moved$cache_size_mb()
  expect_true(mb > 0)
})

test_that("KVCacheEntry slice and write are inverse for unchanged indices", {
  entry <- make_entry(batch = 4, heads = 4, seq = 10, dim = 64)
  original_key <- torch_clone(entry$key)

  sliced <- entry$slice(1:2)
  entry$write(1:2, sliced)

  expect_true(torch_allclose(entry$key, original_key))
})

test_that("preallocate then write roundtrip produces non zero values", {
  reference <- make_cache(n_layers = 2, batch = 1)
  cache <- KVCache$new()
  cache$preallocate(reference, batch_shape = c(4))

  data <- make_cache(n_layers = 2, batch = 2)
  cache$write(1:2, data)

  expect_true(cache$is_populated())
  expect_false(torch_allclose(cache$kv[["1"]]$key, torch_zeros(4, 4, 10, 64)))
})

test_that("TabICLCache handles edge case zero batch size", {
  cache <- TabICLCache$new(train_shape = c(0, 100, 20))
  expect_equal(cache$train_shape[1], 0)
  expect_true(cache$is_empty())
})

test_that("TabICLCache handles edge case zero train size", {
  cache <- TabICLCache$new(train_shape = c(2, 0, 20))
  expect_equal(cache$train_shape[2], 0)
})
