test_that("tab_icl2 fits with all versions", {
  skip_if_not(torch::torch_is_installed())
  for (version in names(.model_urls)) {
    mod <- tab_icl2(am ~ mpg + wt, data = rsample::initial_split(mtcars), version = version, num_quantiles = 5)
    expect_s3_class(mod, "tab_icl_v2")
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

test_that("classifier takes `training_set_limit` into account", {
  skip_if_not(torch::torch_is_installed())
  two_class_split <- rsample::initial_split(modeldata::two_class_dat)

  orig_data <- tab_icl2(
    Class ~ .,
    data = two_class_split
  )

  expect_equal(orig_data$training[1], nrow(training(two_class_split)))

  smaller_data <- tab_icl2(
    Class ~ .,
    data = two_class_split,
    training_set_limit = 50
  )

  expect_equal(smaller_data$training[1], 50)
})

test_that("Training regression for data.frame and formula", {

  expect_no_error(
    fit <- tab_icl2(train_val, y)
  )

  expect_no_error(
    predict(fit, train_val)
  )

  expect_no_error(
    fit <- tab_icl2(Sale_Price ~ ., data = ames_split)
  )

  expect_no_error(
    predict(fit, x)
  )
})

test_that("Training classification for data.frame", {

  expect_no_error(
    fit <- tab_icl2(attrix, attriy)
  )

  expect_no_error(
    predict(fit, attrix, type = "prob")
  )

  expect_no_error(
    predict(fit, attrix)
  )

})


test_that("can train from a recipe", {

  rec <- recipe(Attrition ~ ., data = attrition) %>%
    step_normalize(all_numeric(), -all_outcomes())

  expect_no_error(
    fit <- tab_icl2(rec, attri_split)
  )

  expect_no_error(
    predict(fit, attrition)
  )

})
