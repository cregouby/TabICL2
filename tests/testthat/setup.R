# Run before any test
suppressPackageStartupMessages(library(recipes))

# ames small data
utils::data("ames", package = "modeldata")
ids <- sample(nrow(ames), 306)
ames_split <- rsample::initial_split(ames[ids,])
train_val <- ames[ids, -which(names(ames) == "Sale_Price")]
y <- ames[ids[1:256],]$Sale_Price

# attrition small data
utils::data("attrition", package = "modeldata")
ids <- sample(nrow(attrition), 306)

# attrition common models
attri_split <- rsample::initial_split(attrition[ids,])

attrix <- attrition[ids, -which(names(attrition) == "Attrition")]
attriy <- attrition[ids[1:256],]$Attrition
