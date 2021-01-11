# Install necessary packages
install.packages("RCurl")
install.packages("corrplot")
install.packages("Hmisc")
install.packages("glmnet")
install.packages('xgboost')
install.packages('zoo')
install.packages('DMwR')

# Remove scientific notation for easier viewing
options(scipen = 999)

# Import library
library('RCurl')

# Link to data
link = getURL('https://raw.githubusercontent.com/bradklassen/Predicting_NHL_Salaries/master/NHL_Salaries.csv')

# Assign data to df
df_ELC = read.csv(text = link)

# Remove unnecessary 'X' column that is an extra index
df_ELC = df_ELC[!names(df_ELC) %in% c("X")]

# Print number of rows and columns
cat('Dimensions: ', dim(df_ELC))

# Plot of NHL Salaries
hist(df_ELC$Salary/1000000, breaks = 50, col = 'red', main = 'NHL Salary Distribution', 
     xlab = 'Salary (In Millions)', ylab = 'Number of Players')

# Boxplot of Salaries and it's quantiles
summary(df_ELC$Salary)
boxplot(df_ELC$Salary/1000000, main = 'NHL Salary', xlab = 'Salary (In Millions)', 
        col = 'orange', border = 'brown', horizontal = TRUE)

# Link to data without entry-level contracts
link = getURL('https://raw.githubusercontent.com/bradklassen/Predicting_NHL_Salaries/master/NHL_Salaries_ELC_Removed.csv')

# Assign data to df
df = read.csv(text = link)

# Remove columns where Salary is null
df = df[!is.na(df$Salary), ]

# Print number of rows and columns
cat('Dimensions: ', dim(df))

# Percentage of missing data per column
percent_missing = as.data.frame(colMeans(is.na(df)) * 100)

# Count of missing data per column
count_missing = as.data.frame(colSums(is.na(df)))

# Plot of NHL Salaries
hist(df$Salary/1000000, breaks = 50, col = 'red', main = 'NHL Salary Distribution (No ELC)', 
     xlab = 'Salary (In Millions)', ylab = 'Number of Players')

# Boxplot of Salaries and it's quantiles
summary(df$Salary)
boxplot(df$Salary/1000000, main = 'NHL Salary (No ELC)', xlab = 'Salary (In Millions)', 
        col = 'orange', border = 'brown', horizontal = TRUE)

### Points Vs. Salary ###

# Plot of Points vs. Salary
plot(df$PTS, df$Salary/1000000, main = 'Points vs. Salary', col = 'red', 
     xlab = 'Points', ylab = 'Salary (In Millions)')

# Fit line to scatterplot
abline(lm(Salary/1000000 ~ PTS, data = df), col = 'blue')

### Overall Draft Pick Vs. Salary ###

# Draft Pick Overall vs. Salary plot
plot(df$Ovrl, df$Salary/1000000, main = 'Overall Draft Pick vs. Salary', 
     col = 'red', xlab = 'Overall Draft Pick', ylab = 'Salary (In Millions)')

# Fit line to scatterplot
abline(lm(Salary/1000000 ~ Ovrl, data = df), col = "blue")

# Rename 'X...' column as 'Plus_Minus'
names(df)[names(df) == 'X...'] = 'Plus_Minus'

# Load necessary library
library(corrplot)
# Create a matrix of correlations
corr_data = cor(as.matrix(df[, c('G', 'A', 'PTS', 'TOI', 'Plus_Minus', 'Salary')]))
# Plot the correlation matrix 
corrplot(corr_data, method = 'number')
# Plot a scatterplot of the pairs of inputs
pairs(df[, c('G', 'A', 'PTS', 'TOI', 'Plus_Minus')], pch = 19)
# Print correlation data
corr_data

### Create data set with all numeric inputs and output ###

# Indices of numeric columns
inds = unlist(lapply(df, is.numeric))
# Create a new data set out of the numeric inputs only
numeric_df = df[, inds]

### Calculate the correlation between all numeric inputs and output ###

# Function to flatten the correlation matrix
# cormat: matrix of the correlation coefficients
# pmat: matrix of the correlation p-values
flattenCorrMatrix = function(cormat, pmat){
  ut = upper.tri(cormat)
  # Create a data frame of the columns, correlations, and p-value 
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor = (cormat)[ut],
    p = pmat[ut]
  )
}

# Load necessary library
library(Hmisc)
# Convert to matrix
res = rcorr(as.matrix(numeric_df))
# Matrix is so large that we will flatten it for easier viewing
flat_corr = flattenCorrMatrix(res$r, res$P)

# Data set containing the correlations of inputs with the output 'Salary'
Salary_df = flat_corr[which(flat_corr$row == 'Salary'),]

# Sort the correlations in descending order
Salary_sorted = Salary_df[order(-Salary_df$cor),]

# Print the 5 inputs with the largest positive correlation to the output "Salary"
head(Salary_sorted, n = 5)

# Print the 5 inputs with the largest negative correlation to the output "Salary"
tail(Salary_sorted, n = 5)

# Load library for filling missing values
library(zoo)

# Fill missing values with mean of columns
numeric_df = na.aggregate(numeric_df)

#### Cross Validation Folds ####

# Number of rows in data set
n = dim(numeric_df)[1]

# Number of folds for K-Fold cross validation
nFolds = 10

# Number of observations per fold
n_per_fold = floor(n / nFolds)

# Set seed for split
set.seed(0)

# Create empty list for appending the folds too
folds = list()

# Shuffle index for creating folds
shuffled_index = sample(c(1:n))

# Add data to each fold
for(fold in c(1:(nFolds - 1))){
  folds[[fold]] = shuffled_index[c((1 + (fold - 1) * n_per_fold):(fold * n_per_fold))]
}

# If n does not divide evenly then lump additional observations into final fold
folds[[nFolds]] = shuffled_index[c((1 + (nFolds - 1) * n_per_fold):n)]

##########################
#### Ridge Regression ####
##########################

#### Finding Optimal Lambda ####

# Load necessary library
library(glmnet)

# Test sequence of lambda values 

# Due to computation time, various steps in the sequence have been explored

#lambda_values = seq(from = 10000, to = 750000, by = 10000) 
#lambda_values = seq(from = 500000-4*10000, to = 500000+4*10000, by = 1000) 
#lambda_values = seq(from = 492000-4*1000, to = 492000+4*1000, by = 100)
#lambda_values = seq(from = 489400-4*100, to = 489400+4*100, by = 10)
lambda_values = seq(from = 489320-4*10, to = 489320+4*10, by = 1)

# Number of lambda values to be tested
n_lambda_values = length(lambda_values)

# Initialize RMSE values for the ridge regression
Ridge_RMSE = matrix(0, nrow = n_lambda_values, ncol = nFolds)

# Iterate through folds
for(fold in c(1:nFolds)){
  
  # Create training & testing data sets
  training_data = numeric_df[-folds[[fold]],]
  testing_data = numeric_df[folds[[fold]],]
  
  # Number of observations in training and testing data sets
  n_training = dim(training_data)[1]
  n_testing = dim(testing_data)[1]
  
  # Create input matrix and output vector for training data
  trainingX = model.matrix(Salary ~ 0 + ., data = training_data)
  trainingY = matrix(training_data$Salary, nrow = n_training)
  
  # Create input matrix and output vector for testing data
  testingX = model.matrix(Salary ~ 0 + ., data = testing_data)
  testingY = matrix(testing_data$Salary, nrow = n_testing)
  
  # Number of inputs
  p = dim(trainingX)[2]
  
  # Standard deviation & mean of training X data set
  x_sd = apply(trainingX, 2, sd)
  x_mean = apply(trainingX, 2, mean)
  
  # Scaling training and testing inputs
  trainingX = t((t(trainingX) - x_mean) / x_sd)
  testingX = t((t(testingX) - x_mean) / x_sd)
  
  # Create testingX0 (with intercept) for predictions and RMSE computation 
  testingX0 = cbind(matrix(1, nrow = dim(testingX)[1]), testingX)
  
  # Perform ridge regression for a range of lambda values
  for(i in c(1:n_lambda_values)){ 
    # Set lambda to the ith element in lambda values
    lambda = lambda_values[i]
    # Fit the ridge regression model
    fit = glmnet(trainingX, trainingY - mean(trainingY), alpha = 0, 
                 lambda = lambda, intercept = FALSE)
    # Create Beta hat 
    BHat = matrix(coef(fit), nrow = (p + 1))
    # Set first element of beta hat equal to the mean of training Y data set
    BHat[1] = mean(trainingY)
    # Create prediction
    testingYhat = testingX0 %*% BHat
    # Compute RMSE value and append to the Ridge_RMSE matrix
    Ridge_RMSE[i, fold] = sqrt(sum((testingYhat - testingY)^2) / n_testing)
  }
}

# Find mean of RMSE values in rows
Ridge_RMSE = apply(Ridge_RMSE, 1, mean)

# Minimum RMSE value
Ridge_min_rmse = min(Ridge_RMSE)
# Lambda value that gives minimum RMSE value
Ridge_min_lambda = lambda_values[which.min(Ridge_RMSE)]
# Print minimum lambda value and RMSE
cat('Ridge Regression Optimal Lambda: ', Ridge_min_lambda)
cat('\nRidge Regression RMSE: ', Ridge_min_rmse)

# Refit model with minimum lambda
X = model.matrix(Salary ~ 0 + ., data = numeric_df)
Y = matrix(numeric_df$Salary, nrow = n)

# Number of inputs
p = dim(X)[2]

# Scaling training and testing inputs
X = t((t(X) - x_mean) / x_sd)

# Create testingX0 (with intercept) for predictions and RMSE computation 
X0 = cbind(matrix(1, nrow = dim(X)[1]), X)

# Fit the ridge regression model
fit_full_RIDGE = glmnet(X, Y - mean(Y), alpha = 0, lambda = Ridge_min_lambda, intercept = FALSE)
# Create Beta hat 
BHat_full_RIDGE = matrix(coef(fit_full_RIDGE), nrow = (p + 1))
# Set first element of beta hat equal to the mean of training Y data set
BHat_full_RIDGE[1] = mean(Y)
# Create prediction
Yhat = X0 %*% BHat_full_RIDGE
# Plot predicted vs actual
plot(Yhat, numeric_df$Salary, xlab = 'Predicted', ylab = 'Actual',
     main = 'Ridge Regression - Predicted vs. Actual', bty = 'n')
# Fit line to plot
abline(a = 0, b = 1, lwd = 3, col = 'red')

#### For Plotting ####

# Test sequence of lambda values 
lambda_values = seq(from = 10000, to = 750000, by = 10000) 
# Number of lambda values to be tested
n_lambda_values = length(lambda_values)

# Initialize RMSE values for the ridge regression
RMSE = matrix(0, nrow = n_lambda_values, ncol = nFolds)

# Iterate through folds
for(fold in c(1:nFolds)){
  
  # Create training & testing data sets
  training_data = numeric_df[-folds[[fold]],]
  testing_data = numeric_df[folds[[fold]],]
  
  # Number of observations in training and testing data sets
  n_training = dim(training_data)[1]
  n_testing = dim(testing_data)[1]
  
  # Create input matrix and output vector for training data
  trainingX = model.matrix(Salary ~ 0 + ., data = training_data)
  trainingY = matrix(training_data$Salary, nrow = n_training)
  
  # Create input matrix and output vector for testing data
  testingX = model.matrix(Salary ~ 0 + ., data = testing_data)
  testingY = matrix(testing_data$Salary, nrow = n_testing)
  
  # Number of inputs
  p = dim(trainingX)[2]
  
  # Standard deviation & mean of training X data set
  x_sd = apply(trainingX, 2, sd)
  x_mean = apply(trainingX, 2, mean)
  
  # Scaling training and testing inputs
  trainingX = t((t(trainingX) - x_mean) / x_sd)
  testingX = t((t(testingX) - x_mean) / x_sd)
  
  # Create testingX0 (with intercept) for predictions and RMSE computation 
  testingX0 = cbind(matrix(1, nrow = dim(testingX)[1]), testingX)
  
  # Perform ridge regression for a range of lambda values
  for(i in c(1:n_lambda_values)){ 
    # Set lambda to the ith element in lambda values
    lambda = lambda_values[i]
    # Fit the ridge regression model
    fit = glmnet(trainingX, trainingY - mean(trainingY), alpha = 0, 
                 lambda = lambda, intercept = FALSE)
    # Create Beta hat 
    BHat = matrix(coef(fit), nrow = (p + 1))
    # Set first element of beta hat equal to the mean of training Y data set
    BHat[1] = mean(trainingY)
    # Create prediction
    testingYhat = testingX0 %*% BHat
    # Compute RMSE value and append to the RMSE matrix
    RMSE[i, fold] = sqrt(sum((testingYhat - testingY)^2) / n_testing)
  }
}

# Find mean of RMSE values in rows
RMSE = apply(RMSE, 1, mean)

# Plot RMSE values as a function of lambda
plot(lambda_values, RMSE, xlab = expression(lambda), ylab = 'RMSE', 
     main = 'Ridge Regression - RMSE of testing data', bty = 'n')

###########################
#### LASSSO Regression ####
###########################

#### Finding Optimal Lambda ####

# Test sequence of lambda values 

# Due to computation time, various steps in the sequence have been explored

#lambda_values = seq(from = 1, to = 60000, by = 1000) 
#lambda_values = seq(from = 29001-4*1000, to = 29001+4*1000, by = 100)
#lambda_values = seq(from = 28601-4*100, to = 28601+4*100, by = 10)
lambda_values = seq(from = 28391-4*10, to = 28391+4*10, by = 1)

# Number of lambda values to be tested
n_lambda_values = length(lambda_values)

# Initialize RMSE values for the LASSO regression
RMSE = matrix(0, nrow = n_lambda_values, ncol = nFolds)

# Iterate through folds
for(fold in c(1:nFolds)){
  
  # Create training & testing data sets
  training_data = numeric_df[-folds[[fold]],]
  testing_data = numeric_df[folds[[fold]],]
  
  # Number of observations in training and testing data sets
  n_training = dim(training_data)[1]
  n_testing = dim(testing_data)[1]
  
  # Create input matrix and output vector for training data
  trainingX = model.matrix(Salary ~ 0 + ., data = training_data)
  trainingY = matrix(training_data$Salary, nrow = n_training)
  
  # Create input matrix and output vector for testing data
  testingX = model.matrix(Salary ~ 0 + ., data = testing_data)
  testingY = matrix(testing_data$Salary, nrow = n_testing)
  
  # Number of inputs
  p = dim(trainingX)[2]
  
  # Standard deviation & mean of training X data set
  x_sd = apply(trainingX, 2, sd)
  x_mean = apply(trainingX, 2, mean)
  
  # Scaling training and testing inputs
  trainingX = t((t(trainingX) - x_mean) / x_sd)
  testingX = t((t(testingX) - x_mean) / x_sd)
  
  # Create testingX0 (with intercept) for predictions and RMSE computation 
  testingX0 = cbind(matrix(1, nrow = dim(testingX)[1]), testingX)
  
  # Perform LASSO regression for a range of lambda values
  for(i in c(1:n_lambda_values)){ 
    # Set lambda to the ith element in lambda values
    lambda = lambda_values[i]
    # Fit the LASSO regression model
    fit = glmnet(trainingX, trainingY - mean(trainingY), alpha = 1, 
                 lambda = lambda, intercept = FALSE)
    # Create Beta hat 
    BHat = matrix(coef(fit), nrow = (p + 1))
    # Set first element of beta hat equal to the mean of training Y data set
    BHat[1] = mean(trainingY)
    # Create prediction
    testingYhat = testingX0 %*% BHat
    # Compute RMSE value and append to the RMSE matrix
    RMSE[i, fold] = sqrt(sum((testingYhat - testingY)^2) / n_testing)
  }
}

# Find mean of RMSE values in rows
RMSE = apply(RMSE, 1, mean)

# Minimum RMSE value
LASSO_min_rmse = min(RMSE)
# Lambda value that gives minimum RMSE value
LASSO_min_lambda = lambda_values[which.min(RMSE)]
# Print minimum lambda value and RMSE
cat('LASSO Regression Optimal Lambda: ', LASSO_min_lambda)
cat('\nLASSO Regression RMSE: ', LASSO_min_rmse)

# Refit model with minimum lambda
# Create input matrix and output vector for training data
X = model.matrix(Salary ~ 0 + ., data = numeric_df)
Y = matrix(numeric_df$Salary, nrow = n)

# Number of inputs
p = dim(X)[2]

# Scaling training and testing inputs
X = t((t(X) - x_mean) / x_sd)

# Create testingX0 (with intercept) for predictions and RMSE computation 
X0 = cbind(matrix(1, nrow = dim(X)[1]), X)

# Fit the LASSO regression model
fit_full_LASSO = glmnet(X, Y - mean(Y), alpha = 1, lambda = LASSO_min_lambda, intercept = FALSE)
# Create Beta hat 
BHat_full_LASSO = matrix(coef(fit_full_LASSO), nrow = (p + 1))
# Set first element of beta hat equal to the mean of training Y data set
BHat_full_LASSO[1] = mean(Y)
# Create prediction
Yhat = X0 %*% BHat_full_LASSO

# Plot predicted vs actual
plot(Yhat, numeric_df$Salary, xlab = 'Predicted', ylab = 'Actual',
     main = 'LASSO Regression - Predicted vs. Actual', bty = 'n')
# Fit line to plot
abline(a = 0, b = 1, lwd = 3, col = 'red')

#### For Plotting ####

# Test sequence of lambda values 
lambda_values = seq(from = 1, to = 60000, by = 1000) 

# Number of lambda values to be tested
n_lambda_values = length(lambda_values)

# Initialize RMSE values for the LASSO regression
RMSE = matrix(0, nrow = n_lambda_values, ncol = nFolds)

# Iterate through folds
for(fold in c(1:nFolds)){
  
  # Create training & testing data sets
  training_data = numeric_df[-folds[[fold]],]
  testing_data = numeric_df[folds[[fold]],]
  
  # Number of observations in training and testing data sets
  n_training = dim(training_data)[1]
  n_testing = dim(testing_data)[1]
  
  # Create input matrix and output vector for training data
  trainingX = model.matrix(Salary ~ 0 + ., data = training_data)
  trainingY = matrix(training_data$Salary, nrow = n_training)
  
  # Create input matrix and output vector for testing data
  testingX = model.matrix(Salary ~ 0 + ., data = testing_data)
  testingY = matrix(testing_data$Salary, nrow = n_testing)
  
  # Number of inputs
  p = dim(trainingX)[2]
  
  # Standard deviation & mean of training X data set
  x_sd = apply(trainingX, 2, sd)
  x_mean = apply(trainingX, 2, mean)
  
  # Scaling training and testing inputs
  trainingX = t((t(trainingX) - x_mean) / x_sd)
  testingX = t((t(testingX) - x_mean) / x_sd)
  
  # Create testingX0 (with intercept) for predictions and RMSE computation 
  testingX0 = cbind(matrix(1, nrow = dim(testingX)[1]), testingX)
  
  # Perform LASSO regression for a range of lambda values
  for(i in c(1:n_lambda_values)){ 
    # Set lambda to the ith element in lambda values
    lambda = lambda_values[i]
    # Fit the LASSO regression model
    fit = glmnet(trainingX, trainingY - mean(trainingY), alpha = 1, 
                 lambda = lambda, intercept = FALSE)
    # Create Beta hat 
    BHat = matrix(coef(fit), nrow = (p + 1))
    # Set first element of beta hat equal to the mean of training Y data set
    BHat[1] = mean(trainingY)
    # Create prediction
    testingYhat = testingX0 %*% BHat
    # Compute RMSE value and append to the RMSE matrix
    RMSE[i, fold] = sqrt(sum((testingYhat - testingY)^2) / n_testing)
  }
}

# Find mean of RMSE values in rows
RMSE = apply(RMSE, 1, mean)

# Plot RMSE values as a function of lambda
plot(lambda_values, RMSE, xlab = expression(lambda), ylab = 'RMSE', 
     main = 'LASSO Regression - RMSE of testing data', bty = 'n')

##################################
#### Decision Tree Regression ####
##################################

# Load required library
library(rpart)

# Build the entire tree
fit = rpart(Salary ~ ., data = numeric_df, method = 'anova', 
            control = rpart.control(cp = 0, minsplit = 1, minbucket = 1, xval = 10))

# Get complexity parameter
cp = fit$cp[which.min(matrix(fit$cp[,4])), 1]
cat('Complexity Parameter: ', cp)

# Prune the tree using complexity parameter
pruned_fit = prune.rpart(fit, cp)

# Make predictions
yHat = predict(pruned_fit, numeric_df)

# Compute RMSE value and append to the RMSE matrix
DT_RMSE = sqrt(sum((yHat - numeric_df$Salary)^2) / n)

# Plot the pruned tree with the input names and splitting values
plot(pruned_fit)
text(pruned_fit)

# Print RMSE value
cat('\nDecision Tree RMSE: ', DT_RMSE)

# Plot predicted vs actual and fit line to plot
plot(yHat, numeric_df$Salary, xlab = 'Predicted', ylab = 'Actual', 
     main = 'Decision Tree - Predicted vs. Actual', bty = 'n')
abline(a = 0, b = 1, lwd = 3, col = 'red')

############################
#### XGBoost Regression ####
############################

# Load required library
require(xgboost)

# Initialize RMSE values for the XGBoost regression
XGBoost_RMSE = matrix(0, nrow = nFolds)

# Iterate through folds
for(fold in c(1:nFolds)){
  
  # Create training & testing data sets
  training_data = numeric_df[-folds[[fold]],]
  testing_data = numeric_df[folds[[fold]],]
  
  # Number of observations in training and testing data sets
  n_training = dim(training_data)[1]
  n_testing = dim(testing_data)[1]
  
  # Create input matrix and output vector for training data
  trainingX = model.matrix(Salary ~ 0 + ., data = training_data)
  trainingY = matrix(training_data$Salary, nrow = n_training)
  
  # Create input matrix and output vector for testing data
  testingX = model.matrix(Salary ~ 0 + ., data = testing_data)
  testingY = matrix(testing_data$Salary, nrow = n_testing)
  
  # Number of inputs
  p = dim(trainingX)[2]
  
  # Standard deviation & mean of training X data set
  x_sd = apply(trainingX, 2, sd)
  x_mean = apply(trainingX, 2, mean)
  
  # Scaling training and testing inputs
  trainingX = t((t(trainingX) - x_mean) / x_sd)
  testingX = t((t(testingX) - x_mean) / x_sd)
  
  # Create testingX0 (with intercept) for predictions and RMSE computation 
  testingX0 = cbind(matrix(1, nrow = dim(testingX)[1]), testingX)
  
  # Fit XGBoost model with 10 fold cross validation, using RMSE as performance metric
  xgb_reg = xgboost(data = trainingX, label = trainingY, nrounds = 1000, 
                    nthread = 2, max_depth = 8, eta = 0.3, metrics = 'rmse', 
                    objective = 'reg:squarederror', early_stopping_rounds = 50, 
                    verbose = 0)
  
  # Make predictions
  yHat = predict(xgb_reg, testingX)
  
  # Compute RMSE value and append to the RMSE matrix
  XGBoost_RMSE[fold] = sqrt(sum((yHat - testingY)^2) / n_testing)
  
}

# Find mean of RMSE values
XGBoost_RMSE = apply(XGBoost_RMSE, 2, mean)

# Print RMSE value
cat('XGBoost Regression RMSE: ', XGBoost_RMSE)

# Check feature importance

# 10 most important features in XGBoost Regression model
importance_matrix = xgb.importance(model = xgb_reg)[1:10]

# Print the most important features
head(importance_matrix, n = 10)

# Plot the importance measured by gain
xgb.plot.importance(importance_matrix = importance_matrix)

######################################################
#### Kernel Smoothing Regression (Additive Model) ####
######################################################

# Take 5 best inputs from XGboost
kernel_df = numeric_df[,c("Salary", "FF", "DftYr", "CF", "xGF", "TOI.GP")]

# Number of rows in data set
n = dim(kernel_df)[1]

# Use epanechnikov kernel (highest relative efficency)
epanechnikov_kernel = function(t){
  return(as.integer(abs(t) <= 1) * (3/4) * (1 - t^2))
}

# 1-D Kernel smoothing function (since using additive model)
kernel_smoothing = function(x0, X, Y, K, bandwidth){
  # Inputs
  #   x0 - input to be predicted
  #   X - matrix of training inputs (n x p)
  #   Y - matrix of training outputs (n x 1)
  #   k - kernel function (function)
  #   bandwidth  - kernel bandwidth (numeric)
  #
  # Outputs
  #   predicted y0 value   
  w = K(abs(x0 - X)/bandwidth)
  if(sum(w) == 0){w = 1}
  return(sum(w*Y) / sum(w))
}

# Test sequence of lambda values 
# Due to computation time, the step has been set to 0.1 but can be lowered for finer results
lambda_values = seq(from = 0.4, to = 0.6, by = 0.1)
# Number of lambda values to be tested
n_lambda_values = length(lambda_values)

# max iterations and tolerance value for the back fitting algorithm
max_iterations = 1000
epsilon = 0.5

# Initialize MSE values for the Kernel regression
Kernel_MSE = matrix(0, nrow = n_lambda_values, ncol = nFolds)

# Iterate through folds
for(fold in c(1:nFolds)){
  
  # Create training & testing data sets
  training_data = kernel_df[-folds[[fold]],]
  testing_data = kernel_df[folds[[fold]],]
  
  # Number of observations in training and testing data sets
  n_training = dim(training_data)[1]
  n_testing = dim(testing_data)[1]
  
  # Number of inputs
  p = dim(training_data)[2] - 1
  
  # Create input matrix and output vector for training data
  trainingX = model.matrix(Salary ~ 0 + ., data=training_data)
  trainingY = training_data$Salary
  
  # Create input matrix and output vector for testing data
  testingX = model.matrix(Salary ~ 0 + ., data=testing_data)
  testingY = testing_data$Salary
  
  # Standardize input variables
  x_mean = apply(trainingX, 2, mean)
  x_sd = apply(trainingX, 2, sd)
  
  # Scale training X and testing X data sets
  trainingX = t((t(trainingX) - x_mean) / x_sd)
  testingX = t((t(testingX) - x_mean) / x_sd)
  
  # Initial estimate for intercept term (backfitting algorithm)
  hatA = mean(trainingY) 
  
  for(j in c(1:n_lambda_values)){
    # Set lambda to the ith element in lambda values
    lambda = lambda_values[j]
    
    g = matrix(0, nrow = n_training, ncol = p)
    g_old = matrix(0, nrow = n_training, ncol = p)
    
    # Perform back fitting algorithm on the training data
    for(iteration in c(1:max_iterations)){
      for(k in c(1:p)){
        # Estimate the residuals for each iteration of the algorithm
        R = trainingY - hatA - apply(g, 1, sum) + g[,k]
        for(i in c(1:n_training)){
          # Estimate of the p one-dimension kernel function for each iteration of the backing fitting algorithm
          g[i, k] = kernel_smoothing(trainingX[i, k], trainingX[,k], R, epanechnikov_kernel, lambda)
        }
        # Mean center the estimates
        g[,k] = g[,k] - mean(g[,k]) 
      }	
      
      # Stop algorithm if difference between current iteration and previous iteration is less than the tolerance
      if(sum((g - g_old)^2) < epsilon){
        break
      }else{
        # If tolerance is not met, continue to next iteration
        g_old = g
      }
    }
    
    # Prediction on the testing data
    testingYhat = matrix(hatA, nrow = n_testing)
    for(k in c(1:p)){
      # Estimate the residuals for each iteration of the algorithm
      R = trainingY - hatA - apply(g, 1, sum) + g[,k]
      for(i in c(1:n_testing)){
        # Make predictions
        testingYhat[i] = testingYhat[i] + kernel_smoothing(testingX[i, k], trainingX[,k], R, epanechnikov_kernel, lambda)
      }
    }
    # Compute MSE value and append to the MSE matrix
    Kernel_MSE[j,fold] = sum((testingY - testingYhat)^2) / n_testing
  }
}

# Find mean of MSE values
Kernel_MSE = apply(Kernel_MSE, 1, mean)

# Convert to RMSE
Kernel_RMSE = sqrt(Kernel_MSE)

# Minimum RMSE 
cat('Kernel Smoothing Regression RMSE: ', min(Kernel_RMSE))

# Lambda estimate which gives minimum RMSE
lambda = lambda_values[which.min(Kernel_RMSE)]
cat('\nKernel Smoothing Regression Lambda: ', lambda)

# Plot of possible lambda values and corresponding RSME
plot(lambda_values, Kernel_RMSE, xlab = expression(lambda), ylab = 'RMSE', main = 'Additive Model (Kernel Regression) - RMSE of testing data', bty = 'n')

### Refit model with optimal lambda ###

# Create input matrix and output vector for entire dataset
X = model.matrix(Salary ~ 0 + ., data = kernel_df)
Y = kernel_df$Salary

# Standardize input variables
x_mean = apply(X, 2, mean)
x_sd = apply(X, 2, sd)

# Initial estimate for intercept term (backfitting algorithm)
hatA = mean(Y) 

g = matrix(0, nrow = n, ncol = p)
g_old = matrix(0, nrow = n, ncol = p)

# Perform backfitting algorithm on ENTIRE dataset with minimum lambda obtianed from CV above
for(iteration in c(1:max_iterations)){
  for(k in c(1:p)){
    R = Y - hatA - apply(g, 1, sum) + g[,k]
    for(i in c(1:n_training)){
      g[i, k] = kernel_smoothing(X[i, k], X[,k], R, epanechnikov_kernel, lambda)
    }
    g[,k] = g[,k] - mean(g[,k]) 
  }	
  
  if(sum((g - g_old)^2) < epsilon){
    break
  }else{
    g_old = g
  }
}

Yhat = matrix(hatA, nrow = n)
for(k in c(1:p)){
  # Estimate the residuals for each iteration of the algorithm
  R = Y - hatA - apply(g, 1, sum) + g[,k]
  for(i in c(1:n_testing)){
    # Make predictions with fitted model
    Yhat[i] = Yhat[i] + kernel_smoothing(X[i, k], X[,k], R, epanechnikov_kernel, lambda)
  }
}

# Plot of predicted vs actual for additive model (kernel regression)
plot(Yhat, kernel_df$Salary, xlab = 'Predicted', ylab = 'Actual',
     main = 'Additive Model (Kernel Regression) - Predicted vs. Actual', bty = 'n')

# Fit line to plot
abline(a = 0, b = 1, lwd = 3, col = 'red')

###################################
#### Model Performance Summary ####
###################################

# LASSO Regression RMSE
cat('\nLASSO Regression RMSE: ', LASSO_min_rmse)

# Ridge Regression RMSE
cat('\nRidge Regression RMSE: ', Ridge_min_rmse)

# Decision Tree Regression RMSE
cat('\nDecision Tree Regression RMSE: ', DT_RMSE)

# XGBoost Regression RMSE
cat('\nXGBoost Regression RMSE: ', XGBoost_RMSE)

# Kernel Regression RMSE
cat('\nKernel Regression RMSE: ', Kernel_RMSE)