# Run before any test
suppressPackageStartupMessages(library(recipes))

# ames small data
utils::data("ames", package = "modeldata")
ids <- sample(nrow(ames), 306)
ames_split <- rsample::initial_split(ames[ids,])
ames_train_val <- ames[ids, -which(names(ames) == "Sale_Price")]
ames_train <- ames[ids[1:256],]
y <- ames_train$Sale_Price
ames_val <- ames[ids[257:306], -which(names(ames) == "Sale_Price")]

# attrition small data
utils::data("attrition", package = "modeldata")
ids <- sample(nrow(attrition), 306)

# attrition common models
attri_split <- rsample::initial_split(attrition[ids,])

attrix <- attrition[ids, -which(names(attrition) == "Attrition")]
attriy <- attrition[ids[1:256],]$Attrition
