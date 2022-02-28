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

#install.packages('plsdof')
library('plsdof') # Necessary for "plsRglm" to work



#---------------------------------------------------------------

# LOAD THE DATA

# read csv's
df_red <- read.csv("data/winequality-red.csv", sep=';')
df_white <- read.csv("data/winequality-white.csv", sep=';')

# add categorical color variables to both sets
df_red['color'] <- 'red'
df_white['color'] <- 'white'

# merge datasets
df <- rbind(df_white, df_red)

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
#   13 - color

#   To predict this output variable: 
#   12 - quality (score between 0 and 10)


#---------------------------------------------------------------

# EXPLORE THE DATA

str(df)

head(df)

tail(df)

summary(df)

# Let's start exploring the data one variable at a time

# This function will make a plot of two overlapping transparent histograms
# It will be useful to compare the distributions for red wine and white wine
plot_multi_histogram <- function(df, feature, label_column, alpha, binwidth) {
  plt <- ggplot(df, aes(x=eval(parse(text=feature)), fill=eval(parse(text=label_column)))) +
    geom_histogram(alpha=alpha, position="identity", color="black", binwidth=binwidth) +
  #geom_density(alpha=alpha)# +
  #geom_vline(aes(xintercept=mean(eval(parse(text=feature)))), color="black", linetype="dashed", size=1)# +
  labs(x=feature, y = "Density")
  plt + guides(fill=guide_legend(title=label_column))
}

#   1 - fixed acidity
summary(df$fixed.acidity)
table(df$fixed.acidity)
sort(unique(df$fixed.acidity))
#ggplot(data = df, aes(x=fixed.acidity, fill=color)) + geom_histogram(alpha=0.2, position="identity")
plot_multi_histogram(df, "fixed.acidity", "color", alpha=0.4, binwidth=1)

#   2 - volatile acidity
summary(df$volatile.acidity)
table(df$volatile.acidity)
sort(unique(df$volatile.acidity))
plot_multi_histogram(df, "volatile.acidity", "color", alpha=0.4, binwidth=0.1)

#   3 - citric acid
summary(df$citric.acid)
table(df$citric.acid)
sort(unique(df$citric.acid))
plot_multi_histogram(df, "citric.acid", "color", alpha=0.4, binwidth=0.1)

#   4 - residual sugar
summary(df$residual.sugar)
table(df$residual.sugar)
sort(unique(df$residual.sugar))
plot_multi_histogram(df, "residual.sugar", "color", alpha=0.4, binwidth=2)

#   5 - chlorides
summary(df$chlorides)
table(df$chlorides)
sort(unique(df$chlorides))
plot_multi_histogram(df, "chlorides", "color", alpha=0.4, binwidth=0.01)

#   6 - free sulfur dioxide
summary(df$free.sulfur.dioxide)
table(df$free.sulfur.dioxide)
sort(unique(df$free.sulfur.dioxide))
plot_multi_histogram(df, "free.sulfur.dioxide", "color", alpha=0.4, binwidth=10)

#   7 - total sulfur dioxide
summary(df$total.sulfur.dioxide)
table(df$total.sulfur.dioxide)
sort(unique(df$total.sulfur.dioxide))
plot_multi_histogram(df, "total.sulfur.dioxide", "color", alpha=0.4, binwidth=10)

#   8 - density
summary(df$density)
table(df$density)
sort(unique(df$density))
plot_multi_histogram(df, "density", "color", alpha=0.4, binwidth=0.001)

#   9 - pH
summary(df$pH)
table(df$pH)
sort(unique(df$pH))
plot_multi_histogram(df, "pH", "color", alpha=0.4, binwidth=0.05)

#   10 - sulphates
summary(df$sulphates)
table(df$sulphates)
sort(unique(df$sulphates))
plot_multi_histogram(df, "sulphates", "color", alpha=0.4, binwidth=0.1)

#   11 - alcohol
summary(df$alcohol)
table(df$alcohol)
sort(unique(df$alcohol))
plot_multi_histogram(df, "alcohol", "color", alpha=0.4, binwidth=0.2)

#   12 - quality (score between 0 and 10)
summary(df$quality)
table(df$quality)
sort(unique(df$quality))
plot_multi_histogram(df, "quality", "color", alpha=0.4, binwidth=1)

# What we learned from looking at the data:
# There is more data on white wine than red wine, so we may need to rescale the data so that the model learns to classify red wine as well as white wine.
# Several of the distributions are skewed, indicating the presence of long tails or outliers.

# Now let's look for correlations between features in the data
ggpairs(df, columns=1:12, mapping=ggplot2::aes(colour = color), lower=list(continuous='points')) + scale_color_manual(values=c("red", "yellow"))
# The highest correlations we see are the following:
# 0.721 between free.sulfur.dioxide and total.sulfur.dioxide
# 0.687 between density and alcohol
# 0.553 between density and residual.sugar
# 0.495 between total.sulfur.dioxide and residual.sugar
# 0.459 between density and fixed.acidity
# 0.444 between alcohol and quality
# 0.414 between total.sulfur.dioxide and volatile.acidity
# 0.403 between free.sulfur.dioxide and residual.sugar
# There is also a correlation between pH and volatile.acidity, but it's different between red wine and white wine
# Cleary the alcohol content will be useful for our model because of its relatively high correlation with
# the wine quality rating (0.444).

# We also see some outliers. Let's take a closer look at some of the plots to 
# figure out where these outliers are and if they ought to be removed.
# The most obvious outlier is the one in residual.sugar and density.
ggplot(df, aes(x=residual.sugar, y=density, color=as.factor(quality))) +
  geom_point(size=6) + scale_color_brewer(palette = "Spectral")
# The maximum has over twice as much sugar as the next largest point.
# It is also twice as far from the minimum density as the next most dense point.
# It is clearly an outlier and I believe it is safe to remove it from the data.
# We'll do this by cutting out all points with residual.sugar > 40.
# The next largest point is also separated from most of the other points, but more so
# in density than in residual.sugar. The density is correlated with both alcohol (0.687)
# and residual.sugar (0.553). I think we can assume that much of the information about the
# density is included already in the alcohol and sugar variables, so we can remove
# density from the dataset.

# Another outlier is in free.sulfur.dioxide and total.sulfur.dioxide.
ggplot(df, aes(x=free.sulfur.dioxide, y=total.sulfur.dioxide, color=as.factor(quality))) +
  geom_point(size=6) + scale_color_brewer(palette = "Spectral")
# This point has almost twice as much free sulfur dioxide as the next largest point.
# However, it is not much of an outlier on the other axis, total.sulfur.dioxide.
# It also has the minimum quality rating, which means it could have useful predictive power.
# There is a strong correlation between these two variables (0.721), which means it may be 
# better to simply remove the variable free.sulfur.dioxide from the dataset rather than 
# remove this data point.

# Let's apply the changes (remove sugar outlier & remove free.sulfur.dioxide (6) and density (8)).
df_cut <- df[df$residual.sugar < 40, c(1:5, 7, 9:13)]
# Now we'll redraw the plots.
ggpairs(df_cut, columns=1:10, mapping=ggplot2::aes(colour = color), lower=list(continuous='points')) + scale_color_manual(values=c("red", "yellow"))
# There are still some points that tend to isolate themselves from most of the data, but
# not to such a degree as the outlier that we removed. They seem to simply follow the shape
# of a long-tailed distribution.
# This calls for a transformation of the data to make the distributions more normal,
# specifically in the following variables:
#


ggplot(df_cut, aes(x=residual.sugar, y=sulphates, color=as.factor(quality))) +
  geom_point(size=6) + scale_color_brewer(palette = "Spectral")
#exploratory analysis, findings, and reasoning for: data splitting, feature engineering, pre-processing, model building, hyperparameter optimization, model stacking, and withholding set validation


# cross-validation?

dataY <- df_cut[11]
dataY <- sapply(dataY, as.numeric)
dataX <- df_cut[1:10] # leaving out non-numeric color for now.

# might need some PCA if it doesn't go well.

# model building
set.seed(314)

# This command required me to install package 'plsdof'
plsRglm(as.numeric(dataY),dataX,nt=2,limQ2set=.0975,
             dataPredictY=dataX,modele="pls",family=NULL,typeVC="none",
             EstimXNA=FALSE,scaleX=TRUE,scaleY=NULL,pvals.expli=FALSE,
             alpha.pvals.expli=.05,MClassed=FALSE,tol_Xi=10^(-12),
             sparse=FALSE,sparseStop=TRUE,naive=FALSE,verbose=TRUE)

