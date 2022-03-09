# This script fulfills the assignment outlined in https://www.gastrograph.com/blogs/interviewing-data-science-interns.

# Run it with 'Rscript AFS_wine_quality_model_function.R'

# LOAD LIBRARIES
library("plsRglm") # See https://arxiv.org/abs/1810.01005 
library('plsdof') # Necessary for "plsRglm" to work
library("Metrics") # For calculating MAE


#---------------------------------------------------------------
# BUILD THE MODEL

transform_data <- function(dataX.in, order) { 
  # Remove redundant features
  df.cut <- dataX.in[c(1:5, 7, 9:12)]

  # Transform the data to remove skewness
  df.transformed <- data.frame(df.cut)
  df.transformed$fixed.acidity <- 1/(df.transformed$fixed.acidity)
  df.transformed$sulphates <- 1/(df.transformed$sulphates)
  df.transformed$volatile.acidity <- sqrt(1/(df.transformed$volatile.acidity))
  df.transformed$residual.sugar <- log10(df.transformed$residual.sugar)
  df.transformed$chlorides <- sqrt(1/(df.transformed$chlorides))
  df.transformed$citric.acid <- sqrt(df.transformed$citric.acid + 0.1)
  df.transformed$total.sulfur.dioxide <- sqrt(df.transformed$total.sulfur.dioxide)
  
  # transform the color variable into numeric factors
  df.transformed$color <- factor(df.transformed$color)
  df.transformed$color <- as.numeric(df.transformed$color)

  # Now we center and scale the data
  for (feature in names(df.transformed[1:9])) {
    df.transformed[feature] <- scale(df.transformed[,feature], center = TRUE, scale = TRUE)
  }

  # Create nonlinear terms for effective polynomial fitting
  dataX.nonlinear <- data.frame(df.transformed)
  #for (feature1 in names(dataX.nonlinear[1:9])) {
  #  for (feature2 in names(dataX.nonlinear[1:9])) {
  #    new.feature <- paste(substring(feature1,1,2), substring(feature2,1,2), sep=".")
  #    dataX.nonlinear[new.feature] <- dataX.nonlinear[feature1]*dataX.nonlinear[feature2]
  #  }
  #}
  # I don't think we really need cross-terms, we just need x^2, x^3, etc.
  for (feature in names(dataX.nonlinear[1:9])) {
    for (power in 2:order) {
      new.feature <- paste(substring(feature,1,2), as.character(power), sep=".")
      dataX.nonlinear[new.feature] <- dataX.nonlinear[feature]^power
    }
  }

  # Separate the red and white wine features
  for (feature in names(dataX.nonlinear[c(1:9, 11:length(dataX.nonlinear))])) {
    r.feature <- paste("r", feature, sep=".")
    w.feature <- paste("w", feature, sep=".")
    dataX.nonlinear[r.feature] <- with(dataX.nonlinear, ifelse(color == 1, eval(parse(text=feature)), 0.0))
    dataX.nonlinear[w.feature] <- with(dataX.nonlinear, ifelse(color == 2, eval(parse(text=feature)), 0.0))
  }

  # Remove all the columns that don't start with r. or w.
  n.mixed.color.features <- 10 + 9*(order-1)
  start.index <- n.mixed.color.features + 1
  end.index <- length(dataX.nonlinear)
  dataX.nonlinear.final <- dataX.nonlinear[c(start.index:end.index)]

  return(dataX.nonlinear.final)
}


myModel <- function(dataY.in, dataX.in, order, nt){
  # Prepare the data
  dataX.nonlinear.final <- transform_data(dataX.in, order)
  
  # Fit the plsRglm model
  # See https://cran.r-project.org/web/packages/plsRglm/vignettes/plsRglm.pdf 
  # For info about arguments: https://www.rdocumentation.org/packages/plsRglm/versions/1.3.0/topics/plsRglm
  model.plsRglm <- plsRglm(dataY.in,dataX.nonlinear.final,nt=nt,limQ2set=.0975,
                           dataPredictY=dataX.nonlinear.final,modele="pls-glm-polr",family=NULL,typeVC="none",
                           EstimXNA=FALSE,scaleX=TRUE,scaleY=NULL,pvals.expli=FALSE,
                           alpha.pvals.expli=.05,MClassed=FALSE,tol_Xi=10^(-12),
                           sparse=FALSE,sparseStop=TRUE,naive=FALSE,verbose=TRUE)

  model = structure(list(x = dataX.nonlinear.final, y = dataY.in, polyorder=order, model_plsRglm = model.plsRglm), class = "myModelClass") 
  return(model)
}

# mask predict function so that it works the way we expect
predict.myModelClass = function(modelObject, newdata) {
  transformedData <- transform_data(newdata, modelObject$polyorder)
  return(predict(modelObject$model_plsRglm, newdata = transformedData, type="class"))
} 


#---------------------------------------------------------------
# LOAD THE DATA AND TEST THE MODEL

# read csv's
df.red <- read.csv("data/winequality-red.csv", sep=';')
df.white <- read.csv("data/winequality-white.csv", sep=';')

# add categorical color variables to both sets
df.red['color'] <- 'red'
df.white['color'] <- 'white'

# merge datasets
df <- rbind(df.white, df.red)

# separate explanatory variables (X) and target variable (Y)
dataY <- factor(df$quality, ordered=TRUE)
dataX <- df[c(1:11, 13)] # everything except quality

# Split training and testing data
train.fraction <- 0.8 # use 80% of data for training and 20% for testing
N.rows <- nrow(dataX)
N.train.rows <- floor(N.rows*train.fraction)
N.test.rows <- N.rows - N.train.rows
test.indices <- sample.int(N.rows, N.test.rows) # select N.test.rows randomly

dataY.test <- dataY[c(test.indices)]
dataY.train <- dataY[-c(test.indices)]

dataX.test <- dataX[c(test.indices),]
dataX.train <- dataX[-c(test.indices),]

# fit the model with training data
set.seed(123)
my.model <- myModel(dataY.train, dataX.train, order=4, nt=10) # It usually only finds 8 significant components.

# Evaluate model performance
predictions.train <- as.numeric(predict(my.model, newdata = dataX.train))
observations.train <- as.numeric(dataY.train)
predictions.test <- as.numeric(predict(my.model, newdata = dataX.test))
observations.test <- as.numeric(dataY.test)
mae(observations.train, predictions.train)
mae(observations.test, predictions.test)
