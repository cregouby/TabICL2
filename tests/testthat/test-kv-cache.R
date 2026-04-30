test_that("KVCacheEntry: default is empty and not valid", {
  entry <- KVCacheEntry$new()
  expect_null(entry$key)
  expect_null(entry$value)
  expect_false(entry$is_valid())
})

test_that("KVCacheEntry: constructed with tensors is valid", {
  k <- torch_ones(4L, 2L, 3L, 8L)
  v <- torch_zeros(4L, 2L, 3L, 8L)
  entry <- KVCacheEntry$new(key = k, value = v)
  expect_true(entry$is_valid())
  expect_tensor(entry$key)
  expect_tensor(entry$value)
  expect_tensor_shape(entry$key, c(4L, 2L, 3L, 8L))
})

test_that("KVCacheEntry: subset slices along batch dim", {
  k <- torch_ones(6L, 2L, 3L, 8L)
  v <- torch_zeros(6L, 2L, 3L, 8L)
  entry <- KVCacheEntry$new(key = k, value = v)

  sliced <- entry$subset(2L, 4L)
  expect_tensor_shape(sliced$key,   c(3L, 2L, 3L, 8L))
  expect_tensor_shape(sliced$value, c(3L, 2L, 3L, 8L))
  expect_equal_to_r(sliced$key$sum(), 3L * 2L * 3L * 8L)
})

test_that("KVCacheEntry: subset on invalid entry returns empty", {
  entry <- KVCacheEntry$new()
  sliced <- entry$subset(1L, 3L)
  expect_false(sliced$is_valid())
})

test_that("KVCacheEntry: assign writes batch slice in-place", {
  k <- torch_zeros(6L, 2L, 3L, 8L)
  v <- torch_zeros(6L, 2L, 3L, 8L)
  target <- KVCacheEntry$new(key = k, value = v)

  src_k <- torch_ones(3L, 2L, 3L, 8L)
  src_v <- torch_ones(3L, 2L, 3L, 8L) * 2
  source <- KVCacheEntry$new(key = src_k, value = src_v)

  target$assign(2L, 4L, source)

  # positions 2-4 should now be 1.0 (key) and 2.0 (value)
  expect_equal_to_r(target$key[2L, 1L, 1L, 1L],   1)
  expect_equal_to_r(target$value[4L, 1L, 1L, 1L],  2)
  # position 1 should still be 0
  expect_equal_to_r(target$key[1L, 1L, 1L, 1L],    0)
})

test_that("KVCacheEntry: assign with invalid source does nothing", {
  k <- torch_ones(4L, 2L, 3L, 8L)
  v <- torch_ones(4L, 2L, 3L, 8L)
  target <- KVCacheEntry$new(key = k, value = v)
  empty  <- KVCacheEntry$new()
  # should not throw
  target$assign(1L, 2L, empty)
  expect_equal_to_r(target$key$sum(), 4L * 2L * 3L * 8L)
})

test_that("KVCacheEntry: to moves device", {
  k <- torch_ones(2L, 4L, 3L, 8L)
  v <- torch_zeros(2L, 4L, 3L, 8L)
  entry <- KVCacheEntry$new(key = k, value = v)

  moved <- entry$to("cpu")
  expect_tensor(moved$key)
  expect_tensor_shape(moved$key, c(2L, 4L, 3L, 8L))
})

test_that("KVCacheEntry: to on invalid returns empty", {
  entry <- KVCacheEntry$new()
  moved <- entry$to("cpu")
  expect_false(moved$is_valid())
})


test_that("kv_cache_entry_concat: empty list returns empty entry", {
  result <- kv_cache_entry_concat(list())
  expect_false(result$is_valid())
})

test_that("kv_cache_entry_concat: concatenates valid entries", {
  e1 <- KVCacheEntry$new(
    key   = torch_ones(2L, 4L, 3L, 8L),
    value = torch_zeros(2L, 4L, 3L, 8L)
  )
  e2 <- KVCacheEntry$new(
    key   = torch_ones(3L, 4L, 3L, 8L) * 2,
    value = torch_zeros(3L, 4L, 3L, 8L) * 3
  )
  result <- kv_cache_entry_concat(list(e1, e2))
  expect_tensor_shape(result$key,   c(5L, 4L, 3L, 8L))
  expect_tensor_shape(result$value, c(5L, 4L, 3L, 8L))
})

test_that("kv_cache_entry_concat: skips invalid entries", {
  e_valid <- KVCacheEntry$new(
    key   = torch_ones(2L, 4L, 3L, 8L),
    value = torch_zeros(2L, 4L, 3L, 8L)
  )
  e_empty <- KVCacheEntry$new()
  result <- kv_cache_entry_concat(list(e_empty, e_valid, e_empty))
  expect_tensor_shape(result$key, c(2L, 4L, 3L, 8L))
})


test_that("KVCache: default is empty and not populated", {
  cache <- KVCache$new()
  expect_equal(length(cache$kv), 0L)
  expect_false(cache$is_populated())
})

test_that("KVCache: is_populated returns TRUE when any entry is valid", {
  cache <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(),
    "2" = KVCacheEntry$new(
      key   = torch_ones(2L, 4L, 3L, 8L),
      value = torch_zeros(2L, 4L, 3L, 8L)
    )
  ))
  expect_true(cache$is_populated())
})

test_that("KVCache: subset slices all entries", {
  cache <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(6L, 4L, 3L, 8L),
      value = torch_zeros(6L, 4L, 3L, 8L)
    ),
    "2" = KVCacheEntry$new(
      key   = torch_ones(6L, 4L, 3L, 8L) * 2,
      value = torch_zeros(6L, 4L, 3L, 8L) * 2
    )
  ))
  sliced <- cache$subset(2L, 4L)
  expect_tensor_shape(sliced$kv[["1"]]$key, c(3L, 4L, 3L, 8L))
  expect_tensor_shape(sliced$kv[["2"]]$key, c(3L, 4L, 3L, 8L))
})

test_that("KVCache: assign writes slices into pre-allocated cache", {
  # Pre-allocate target
  k1 <- torch_zeros(6L, 4L, 3L, 8L)
  v1 <- torch_zeros(6L, 4L, 3L, 8L)
  target <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(key = k1, value = v1)
  ))

  # Source cache
  src_entry <- KVCacheEntry$new(
    key   = torch_ones(3L, 4L, 3L, 8L) * 5,
    value = torch_ones(3L, 4L, 3L, 8L) * 7
  )
  source <- KVCache$new(kv = list("1" = src_entry))

  target$assign(2L, 4L, source)

  expect_equal_to_r(target$kv[["1"]]$key[2L, 1L, 1L, 1L],   5)
  expect_equal_to_r(target$kv[["1"]]$value[4L, 1L, 1L, 1L], 7)
  expect_equal_to_r(target$kv[["1"]]$key[1L, 1L, 1L, 1L],   0)
})

test_that("KVCache: assign on invalid target raises error", {
  target <- KVCache$new(kv = list("1" = KVCacheEntry$new()))
  source <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(2L, 4L, 3L, 8L),
      value = torch_zeros(2L, 4L, 3L, 8L)
    )
  ))
  expect_error(
    target$assign(1L, 2L, source),
    "Cannot write to cache index 1 because it is not valid"
  )
})

test_that("KVCache: to moves all entries", {
  cache <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(2L, 4L, 3L, 8L),
      value = torch_zeros(2L, 4L, 3L, 8L)
    )
  ))
  moved <- cache$to("cpu")
  expect_true(moved$is_populated())
  expect_tensor_shape(moved$kv[["1"]]$key, c(2L, 4L, 3L, 8L))
})

test_that("KVCache: preallocate creates zero-filled entries", {
  reference <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(2L, 4L, 3L, 8L),
      value = torch_zeros(2L, 4L, 3L, 8L)
    ),
    "2" = KVCacheEntry$new(
      key   = torch_ones(2L, 4L, 5L, 8L),
      value = torch_zeros(2L, 4L, 5L, 8L)
    )
  ))

  target <- KVCache$new()
  target$preallocate(reference, batch_shape = c(10L), device = "cpu")

  expect_true(target$is_populated())
  # Layer 1: batch=10, num_heads=4, seq_len=3, head_dim=8
  expect_tensor_shape(target$kv[["1"]]$key, c(10L, 4L, 3L, 8L))
  # Layer 2: batch=10, num_heads=4, seq_len=5, head_dim=8
  expect_tensor_shape(target$kv[["2"]]$key, c(10L, 4L, 5L, 8L))
  # Should be all zeros
  expect_equal_to_r(target$kv[["1"]]$key$sum(), 0)
})

test_that("kv_cache_concat: empty list returns empty cache", {
  result <- kv_cache_concat(list())
  expect_false(result$is_populated())
})

test_that("kv_cache_concat: merges entries across caches", {
  c1 <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(2L, 4L, 3L, 8L),
      value = torch_zeros(2L, 4L, 3L, 8L)
    )
  ))
  c2 <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(3L, 4L, 3L, 8L) * 2,
      value = torch_zeros(3L, 4L, 3L, 8L) * 2
    )
  ))
  result <- kv_cache_concat(list(c1, c2))
  expect_tensor_shape(result$kv[["1"]]$key, c(5L, 4L, 3L, 8L))
})

test_that("kv_cache_concat: handles mismatched layer indices", {
  c1 <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(2L, 4L, 3L, 8L),
      value = torch_zeros(2L, 4L, 3L, 8L)
    )
  ))
  c2 <- KVCache$new(kv = list(
    "2" = KVCacheEntry$new(
      key   = torch_ones(3L, 4L, 5L, 8L),
      value = torch_zeros(3L, 4L, 5L, 8L)
    )
  ))
  result <- kv_cache_concat(list(c1, c2))
  # Layer 1 only from c1
  expect_tensor_shape(result$kv[["1"]]$key, c(2L, 4L, 3L, 8L))
  # Layer 2 only from c2
  expect_tensor_shape(result$kv[["2"]]$key, c(3L, 4L, 5L, 8L))
})

test_that("TabICLCache: default construction has empty sub-caches", {
  cache <- TabICLCache$new()
  expect_true(cache$is_empty())
  expect_equal(cache$cache_type(), "empty")
  expect_equal(cache$train_shape, c(0L, 0L, 0L))
})

test_that("TabICLCache: cache_type returns 'repr' when row_repr is set", {
  cache <- TabICLCache$new(row_repr = torch_ones(4L, 5L))
  expect_equal(cache$cache_type(), "repr")
  expect_false(cache$is_empty())
})

test_that("TabICLCache: cache_type returns 'kv' when col_cache is populated", {
  kv <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(2L, 4L, 3L, 8L),
      value = torch_zeros(2L, 4L, 3L, 8L)
    )
  ))
  cache <- TabICLCache$new(col_cache = kv)
  expect_equal(cache$cache_type(), "kv")
  expect_false(cache$is_empty())
})

test_that("TabICLCache: cache_type returns 'kv' when icl_cache is populated", {
  kv <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(2L, 4L, 3L, 8L),
      value = torch_zeros(2L, 4L, 3L, 8L)
    )
  ))
  cache <- TabICLCache$new(icl_cache = kv)
  expect_equal(cache$cache_type(), "kv")
})

test_that("TabICLCache: cache_size_mb returns non-negative integer", {
  k <- torch_ones(100L, 4L, 10L, 16L)  # 100*4*10*16 = 64000 float32 = ~0.24 MB
  v <- torch_zeros(100L, 4L, 10L, 16L)
  kv <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(key = k, value = v)
  ))
  cache <- TabICLCache$new(col_cache = kv)
  size <- cache$cache_size_mb()
  expect_true(is.numeric(size))
  expect_true(size >= 0)
})

test_that("TabICLCache: is_empty is TRUE for default, FALSE with data", {
  expect_true(TabICLCache$new()$is_empty())

  kv <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(2L, 4L, 3L, 8L),
      value = torch_zeros(2L, 4L, 3L, 8L)
    )
  ))
  expect_false(TabICLCache$new(col_cache = kv)$is_empty())
  expect_false(TabICLCache$new(
    row_repr = torch_ones(2L, 4L)
  )$is_empty())
})

test_that("TabICLCache: slice_batch slices all components", {
  kv <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(10L, 4L, 3L, 8L),
      value = torch_zeros(10L, 4L, 3L, 8L)
    )
  ))
  cache <- TabICLCache$new(
    col_cache   = kv,
    row_repr    = torch_ones(10L, 5L),
    icl_cache   = kv,
    train_shape = c(10L, 5L, 8L),
    num_classes = 3L
  )

  sliced <- cache$slice_batch(3L, 7L)  # 5 elements (3,4,5,6,7)

  # Batch size should be 5
  expect_equal(sliced$train_shape[1L], 5L)
  expect_equal(sliced$train_shape[2L], 5L)
  expect_equal(sliced$train_shape[3L], 8L)
  expect_equal(sliced$num_classes, 3L)

  # col_cache entries sliced
  expect_tensor_shape(sliced$col_cache$kv[["1"]]$key, c(5L, 4L, 3L, 8L))
  # row_repr sliced
  expect_tensor_shape(sliced$row_repr, c(5L, 5L))
  # icl_cache entries sliced
  expect_tensor_shape(sliced$icl_cache$kv[["1"]]$key, c(5L, 4L, 3L, 8L))
})

test_that("TabICLCache: slice_batch on empty cache returns valid empty cache", {
  cache <- TabICLCache$new()
  sliced <- cache$slice_batch(1L, 5L)
  expect_true(sliced$is_empty())
  expect_equal(sliced$train_shape[1L], 5L)
})

test_that("TabICLCache: to moves all tensors", {
  kv <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(4L, 2L, 3L, 8L),
      value = torch_zeros(4L, 2L, 3L, 8L)
    )
  ))
  cache <- TabICLCache$new(
    col_cache = kv,
    row_repr  = torch_ones(4L, 6L),
    icl_cache = kv,
    train_shape = c(4L, 5L, 8L)
  )

  moved <- cache$to("cpu")
  expect_tensor_shape(moved$col_cache$kv[["1"]]$key, c(4L, 2L, 3L, 8L))
  expect_tensor_shape(moved$row_repr, c(4L, 6L))
  expect_tensor_shape(moved$icl_cache$kv[["1"]]$key, c(4L, 2L, 3L, 8L))
  expect_equal(moved$train_shape, c(4L, 5L, 8L))
})

test_that("TabICLCache: to on empty cache returns valid empty cache", {
  cache <- TabICLCache$new()
  moved <- cache$to("cpu")
  expect_true(moved$is_empty())
})

test_that("tabicl_cache_concat: concatenates col_cache and icl_cache", {
  kv1 <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(3L, 2L, 4L, 8L),
      value = torch_zeros(3L, 2L, 4L, 8L)
    )
  ))
  kv2 <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(4L, 2L, 4L, 8L) * 2,
      value = torch_zeros(4L, 2L, 4L, 8L) * 2
    )
  ))

  c1 <- TabICLCache$new(
    col_cache   = kv1,
    icl_cache   = kv1,
    train_shape = c(3L, 10L, 5L),
    num_classes = 2L
  )
  c2 <- TabICLCache$new(
    col_cache   = kv2,
    icl_cache   = kv2,
    train_shape = c(4L, 10L, 5L),
    num_classes = 2L
  )

  result <- tabicl_cache_concat(list(c1, c2))

  # batch = 3 + 4 = 7
  expect_equal(result$train_shape[1L], 7L)
  expect_equal(result$train_shape[2L], 10L)
  expect_equal(result$train_shape[3L], 5L)
  expect_equal(result$num_classes, 2L)

  expect_tensor_shape(
    result$col_cache$kv[["1"]]$key, c(7L, 2L, 4L, 8L)
  )
  expect_tensor_shape(
    result$icl_cache$kv[["1"]]$key, c(7L, 2L, 4L, 8L)
  )
})

test_that("tabicl_cache_concat: concatenates row_repr", {
  c1 <- TabICLCache$new(
    row_repr    = torch_ones(3L, 5L),
    train_shape = c(3L, 10L, 5L)
  )
  c2 <- TabICLCache$new(
    row_repr    = torch_zeros(4L, 5L),
    train_shape = c(4L, 10L, 5L)
  )

  result <- tabicl_cache_concat(list(c1, c2))
  expect_tensor_shape(result$row_repr, c(7L, 5L))
})

test_that("tabicl_cache_concat: handles NULL components gracefully", {
  c1 <- TabICLCache$new(
    col_cache   = KVCache$new(kv = list(
      "1" = KVCacheEntry$new(
        key   = torch_ones(2L, 4L, 3L, 8L),
        value = torch_zeros(2L, 4L, 3L, 8L)
      )
    )),
    train_shape = c(2L, 10L, 5L)
  )
  c2 <- TabICLCache$new(
    row_repr    = torch_ones(3L, 5L),
    train_shape = c(3L, 10L, 5L)
  )

  # Should not error; col_cache from c2 is default empty, row_repr from c1 is NULL
  result <- tabicl_cache_concat(list(c1, c2))
  expect_equal(result$train_shape[1L], 5L)
  expect_tensor_shape(result$col_cache$kv[["1"]]$key, c(2L, 4L, 3L, 8L))
  expect_tensor_shape(result$row_repr, c(3L, 5L))
})

test_that("tabicl_cache_concat: single cache passes through", {
  kv <- KVCache$new(kv = list(
    "1" = KVCacheEntry$new(
      key   = torch_ones(4L, 2L, 3L, 8L),
      value = torch_zeros(4L, 2L, 3L, 8L)
    )
  ))
  c1 <- TabICLCache$new(
    col_cache   = kv,
    train_shape = c(4L, 10L, 5L),
    num_classes = 3L
  )
  result <- tabicl_cache_concat(list(c1))
  expect_equal(result$train_shape[1L], 4L)
  expect_equal(result$num_classes, 3L)
  expect_tensor_shape(result$col_cache$kv[["1"]]$key, c(4L, 2L, 3L, 8L))
})


test_that("tensor$element_size() returns correct byte sizes", {
  t32 <- torch_ones(2L)
  expect_equal(t32$element_size(), 4L)  # float32 = 4 bytes

  t64 <- torch_ones(2L, dtype = torch_float64())
  expect_equal(t64$element_size(), 8L)

  t16 <- torch_ones(2L, dtype = torch_float16())
  expect_equal(t16$element_size(), 2L)
})
