test_that("tab_icl2 fits with all versions", {
  skip_if_not(torch::torch_is_installed())

  mod <- tab_icl2(species ~ island + bill_len + bill_dep + sex, data = rsample::initial_split(penguins),
                  model_version = paste0("file://",rappdirs::user_cache_dir("torch/TabICL2/tabicl-classifier-v2-20260212.pt"))
  )
  expect_s3_class(mod, "tab_icl_v2")


  mod <- tab_icl2(am ~ mpg + wt, data = rsample::initial_split(mtcars),
                  model_version = paste0("file://",rappdirs::user_cache_dir("torch/TabICL2/tabicl-regressor-v2-20260212.pt")),
                  config = tab_icl2_config(num_quantiles = 5))
  expect_s3_class(mod, "tab_icl_v2")

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

  expect_equal(orig_data$training[1], nrow(rsample::training(two_class_split)))
})

test_that("Training regression for data.frame and formula, with different num_quantiles", {

  expect_no_error(
    fit <- tab_icl2(ames_train_val, y, config = tab_icl2_config(num_quantiles = 19))
  )
  expect_s3_class(fit$blueprint$ptypes$outcomes, "tbl_df")
  expect_named(fit$blueprint$ptypes$outcomes, ".outcome")
  expect_all_true(purrr::map_lgl(fit$blueprint$ptypes, ~nrow(.x) == 0L))

  expect_no_error(
    pred <- predict(fit, ames_val)
  )
  expect_named(pred, c(".pred"))
  expect_type(pred$.pred, "double")
  expect_length(pred$.pred, nrow(ames_val))

  expect_no_error(
    # data.frame require a `.outcome` column due to mold(x.y) not capturing y name
    augm <- augment(fit, ames_train %>% rename(.outcome = "Sale_Price"))
  )
  expect_named(augm, c(".pred", ".outcome"))
  expect_type(augm$.pred, "double")
  expect_length(augm$.pred, nrow(ames_train))


  expect_no_error(
    fit <- tab_icl2(Sale_Price ~ ., data = ames_split)
  )

  expect_no_error(
    pred <- predict(fit, rsample::testing(ames_split))
  )
  expect_named(pred, c(".pred"))
  expect_type(pred$.pred, "double")
  expect_length(pred$.pred, nrow(rsample::testing(ames_split)))

  expect_no_error(
    augm <- augment(fit, rsample::training(ames_split))
  )
  # recipe capture the y name
  expect_named(augm, c(".pred", "Sale_Price"))
  expect_type(augm$.pred, "double")
  expect_length(augm$.pred, nrow(rsample::training(ames_split)))
})

test_that("Training classification works for data.frame", {

  expect_no_error(
    fit <- tab_icl2(attrix, attriy)
  )


  expect_no_error(
    pred <- predict(fit, attrix)
  )
  expect_named(pred, c(".pred_No", ".pred_Yes", ".pred_class"))
  expect_true(is.factor(pred$.pred_class))
  expect_equal(levels(pred$.pred_class), levels(attriy))

  expect_no_error(
    # data.frame require a `.outcome` column due to mold(x.y) not capturing y name
    augm <- augment(fit, attrition[ids[1:256],] %>% rename(.outcome = "Attrition"))
  )
  expect_named(augm, c(".pred_No", ".pred_Yes", ".pred_class", ".outcome"))
  expect_s3_class(augm$.pred_class, "factor")
  expect_length(augm$.pred_class, length(attriy))

})


test_that("Training classification from a recipe", {

  rec <- recipe(Attrition ~ ., data = attrition) %>%
    step_normalize(all_numeric(), -all_outcomes())

  expect_no_error(
    fit <- tab_icl2(rec, attri_split)
  )

  expect_no_error(
    pred <- predict(fit, rsample::testing(attri_split))
  )
  expect_named(pred, c(".pred_No", ".pred_Yes", ".pred_class"))
  expect_true(is.factor(pred$.pred_class))
  expect_equal(levels(pred$.pred_class), levels(rec$ptype$Attrition))

  expect_no_error(
    augm <- augment(fit, rsample::training(attri_split))
  )
  expect_named(augm, c(".pred_No", ".pred_Yes", ".pred_class", "Attrition"))
  expect_type(augm$.pred_class, "integer")
  expect_length(augm$.pred_class, nrow(rsample::training(attri_split)))

})


test_that("Training regression for recipe works with pretrained model", {

  rec <- recipe(Sale_Price ~ ., data = ames) %>%
    step_zv(all_predictors()) %>%
    step_log(all_outcomes())  %>%
    step_normalize(all_numeric())

  expect_no_error(
    fit <- tab_icl2(rec, ames_split, model_version = "tabicl_regressor_v2")
  )

  expect_no_error(
    pred <- predict(fit, rsample::testing(ames_split))
  )
  expect_named(pred, c(".pred"))
  expect_type(pred$.pred, "double")
  expect_length(pred$.pred, nrow(rsample::testing(ames_split)))

  expect_no_error(
    augm <- augment(fit, rsample::training(ames_split))
  )
  # recipe capture the y name
  expect_named(augm, c(".pred", ".outcome"))
  expect_type(augm$.pred, "double")
  expect_length(augm$.pred, nrow(rsample::training(ames_split)))
})

