# This script fulfills the assignment outlined in https://www.gastrograph.com/blogs/interviewing-data-science-interns.

# Run it with 'Rscript AFS_wine_quality.R'

# Using only partial least squares regression for generalized linear models (the plsRglm package in R), build an ensemble model to predict the quality score given to each wine from the Vinho Verde region of Portugal (see the data bullet in the Requirements section below to download the datasets). The data consists of chemical tests on wines and an overall assessment score averaged from at least three professional wine tasters. This is interesting data for AFS as the lack of consistent preferences among professional tasters is one of the reasons our company exists.

# The rubric for assessment is explained in the Selection Criteria section below. The prediction model should be trained and will be tested on both red and white wine after joining the two datasets. We will split the supplied data into an 80% training set and a 20% hold-out validation set before running your training script with a random seed. We will use mean absolute error as a performance metric.

#----------------------------------------------------------------

# LOAD LIBRARIES
#install.packages("mvtnorm")
#install.packages("bipartite")

# Package 'nloptr' must be installed like this: 
# Download https://github.com/stevengj/nlopt/archive/v2.7.1.tar.gz
# Extract contents, then enter the directory and type the following command:
# cmake . && make && sudo make install
# next type this command:
# sudo apt install libnlopt-dev
# Then you can do this:
#install.packages("nloptr")
# 

# Package 'nloptr' is a prerequisite for 'car'
#install.packages("car")

# Packages 'mvtnorm', 'bipartite', and 'car' are prerequisites for 'plsRglm'.
#install.packages("plsRglm")

library("plsRglm") # See https://arxiv.org/abs/1810.01005 

#install.packages("tidyverse")
library("tidyverse") # for tidying up the data set

#install.packages("ggplot2")
library("ggplot2") # for visualizations
#install.packages("GGally")
library("GGally")

#install.packages("corrplot")
library("corrplot") # for correlation plot


#---------------------------------------------------------------

# LOAD THE DATA

# read csv's
winequality_red <- read.csv("data/winequality-red.csv", sep=';')
winequality_white <- read.csv("data/winequality-white.csv", sep=';')

# add categorical color variables to both sets
winequality_red['color'] <- 'red'
winequality_white['color'] <- 'white'

# merge datasets
df <- rbind(winequality_red, winequality_white)

# The goal is to use these input variables:
#   1 - fixed acidity
#   2 - volatile acidity
#   3 - citric acid
#   4 - residual sugar
#   5 - chlorides
#   6 - free sulfur dioxide
#   7 - total sulfur dioxide
#   8 - density
#   9 - pH
#   10 - sulphates
#   11 - alcohol

#   To predict this output variable: 
#   12 - quality (score between 0 and 10)


#---------------------------------------------------------------

# EXPLORE THE DATA

str(df)

head(df)

tail(df)

summary(df)

# Let's start exploring the data one variable at a time

#   1 - fixed acidity

#   2 - volatile acidity

#   3 - citric acid

#   4 - residual sugar

#   5 - chlorides

#   6 - free sulfur dioxide

#   7 - total sulfur dioxide

#   8 - density

#   9 - pH

#   10 - sulphates

#   11 - alcohol

#   12 - quality (score between 0 and 10)

summary(df$quality)
table(df$quality)
qplot(quality, data = df, fill = color, binwidth = 1) +
    scale_x_continuous(breaks = seq(3,10,1), lim = c(3,10)) +
    scale_y_sqrt()

# Now let's look for correlations
#ggpairs(data)

#exploratory analysis, findings, and reasoning for: data splitting, feature engineering, pre-processing, model building, hyperparameter optimization, model stacking, and withholding set validation
