test_that("tab_icl2 fits with all versions", {
  skip_if_not(torch::torch_is_installed())
  for (version in names(.model_urls)) {
    mod <- tab_icl2(am ~ mpg + wt, data = mtcars, version = version, num_quantiles = 5)
    expect_s3_class(mod, "tab_icl2")
  }
})

test_that("check_data_constraints errors when too many classes", {
  x <- matrix(0, nrow = 11, ncol = 2)
  y <- factor(letters[1:11])
  expect_error(
    TabICL2:::check_data_constraints(x, y, control_tab_icl2()),
    "classes"
  )
})

test_that("sample_indicies handles numeric outcomes", {
  set.seed(1)
  molded <- list(outcomes = data.frame(outcome = rnorm(50001)))
  result <- TabICL2:::sample_indicies(molded)
  expect_length(result, 50000)
  expect_true(all(result >= 1 & result <= 50001))
})

test_that("data constraints", {
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  set.seed(418)
  orig_data <- tab_icl2(
    Class ~ .,
    data = modeldata::two_class_dat,
  )

  expect_equal(orig_data$training[1], nrow(modeldata::two_class_dat))

  set.seed(418)
  smaller_data <- tab_icl2(
    Class ~ .,
    data = modeldata::two_class_dat,
    training_set_limit = 50,
    control = control_tab_icl2()
  )

  expect_equal(smaller_data$training[1], 50)
})
