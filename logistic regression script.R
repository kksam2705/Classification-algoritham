## HEADER ####
## Who:KK
## What: logistic algoritham
## Last edited: 2023-04-08
####

## CONTENTS ####
## introduction
## Logistic Regression


## Introduction
# The dataset we are taking is inbuild in R and its about stock market.
# Weekly percentage returns for the S&P 500 stock index between 1990 and 2010.
# A data frame with 1089 observations on the following 9 variables.

# Year
# The year that the observation was recorded.

# Lag1
# Percentage return for previous week.

# Lag2
# Percentage return for 2 weeks previous

# Lag3
# Percentage return for 3 weeks previous

# Lag4
#Percentage return for 4 weeks previous

# Lag5
# Percentage return for 5 weeks previous

# Volume
# Volume of shares traded (average number of daily shares traded in billions)

# Today
# Percentage return for this week

# Direction
# A factor with levels Down and Up indicating whether the market had a positive or negative return on a given week


# Problem statement
# we have to predict the market whether it goes up or down based on the certain features provided.

# load the dataset
library(ISLR)
data(Weekly)
str(Weekly)
summary(Weekly)

# To check the missing values
library(Amelia)
missmap(Weekly)
dim(Weekly)

# Exlude the year and volume column

data <- subset(Weekly, select = c(2,3,4,5,6,8,9))
data
# I have excluded year and volumn column because of multicolinearity(Highly correlated)
# it will leading to unreliable and unstable estimates of regression coefficients.
# it is performed using the function vif(if the value is more than 5 we should exclude the values) 

## Logistic Regression
# Logistic Regression is merely regression where the dependent variable is binary (up, down; yes, no; 0, 1; etc.).

# split the data into train and test

library(caTools) #library for test and training
set.seed(1)
sample_log <- sample.split(data$Direction, SplitRatio = 0.7)
sample_log
train <- subset(data,sample_log == TRUE)
test <- subset(data, sample_log == FALSE)

# Deploy the logistic model
set.seed(123)
glm_stock <- glm(Direction ~ ., data = train, family = "binomial")
glm_stock

#Check the multicolinearity
library(car)
vif_values <- vif(glm_stock)
vif_values

#summary
summary(glm_stock)

# Result
# By performing logistic regression their is no association between direction variable
# with the predicators variables.That means p value is more than significance level we
# accept NULL Hypotheis(i.e, no evidence for whether stock market is going up or down )

# Problem Statement 

# To study a Diabetes data set and build a Machine Learning model that predicts whether or not a person has Diabetes.

# About the dataset

# pregnant	Number of times pregnant
# glucose	Plasma glucose concentration (glucose tolerance test)
# triceps	Triceps skin fold thickness (mm)
# insulin	2-Hour serum insulin (mu U/ml)
# mass	Body mass index (weight in kg/(height in m)\^2)
# pedigree	Diabetes pedigree function
# age	Age (years)
# diabetes	Class variable (test for diabetes)


library(mlbench)

data(PimaIndiansDiabetes)
levels(PimaIndiansDiabetes$diabetes)
PimaIndiansDiabetes$diabetes <- ifelse(PimaIndiansDiabetes$diabetes == "pos",1,0)
# make postive 1 and negative  0

# Check the missing values
library(Amelia)
missmap(PimaIndiansDiabetes)



# Split the dataset into train and test
set.seed(1)
sample <- sample.split(PimaIndiansDiabetes$diabetes, SplitRatio = 0.7)
train_data <-subset(PimaIndiansDiabetes, sample == TRUE)
test_data <-subset(PimaIndiansDiabetes, sample == FALSE)

# Deploy the glm() 
result <- glm(diabetes ~ ., data = train_data, family = "binomial")
summary(result)

# The results indicate that there is a statistically significant positive relationship 
# between pregnant, glucose, mass, and pedigree variables and the likelihood of having diabetes. 
# In other words, for each unit increase in these variables, there is an associated increase in the log odds of having diabetes.
# On the other hand, pressure, triceps, insulin, and age variables are not statistically 
# significant predictors of diabetes at the 5% level of significance, 
# meaning that changes in these variables do not appear to have a significant impact 
# on the log odds of having diabetes.


# Checking importance of the each variable of the model
x <- varImp(result)
x

# The variable importance measures are displayed in the "Overall" column, which is a
# measure of the reduction in the model's accuracy when each predictor variable is 
# removed one at a time. The higher the value, the more important the variable is for predicting the outcome.
# In your case, the glucose variable has the highest importance measure with a value 
# of 8.39, followed by mass with a value of 5.24. pregnant, pedigree, and age also
# have relatively high importance measures, while pressure, triceps, and insulin have lower importance measures.
# These variable importance measures can be useful for understanding which predictors
# are most influential in the model and for feature selection or dimensionality reduction in machine learning models.

# Check for multicolineraty
vif(result)

#predict the model
predict_model <- predict(result, test_data, type = "response")

# Evaluate the model

threshold <- 0.5

## The threshold variable is set to 0.5, which is the default threshold for logistic regression models. 

predicted <- ifelse(predict_model > threshold, 1,0)
cm <- table(predicted, test_data$diabetes)
cm

roc <- auc(test_data$diabetes, predicted)
roc

# Calculate accuracy, precision, and F1 score
accuracy <- mean(test_data$diabetes == predicted)
precision <- sum(test_data$diabetes & predicted) / sum(predicted)
recall <- sum(test_data$diabetes & predicted) / sum(test_data$diabetes)
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the results
cat(paste0("Accuracy: ", round(accuracy, 2), "\n"))
cat(paste0("Precision: ", round(precision, 2), "\n"))
cat(paste0("F1 Score: ", round(f1_score, 2), "\n"))
cat(paste0("recall: ", round(recall, 2), "\n"))


# Result

# Area under the curve measured between 0 to 1. whic is used to check the accuracy of the model
# in our case the value is 0.69 which is a moderate performace for prediciting 
# whether patient have a diabetes or not.the value must be close to one.

# confusion matrix is also used to check the accuracy of the model in our case
# the accuracy is 0.76 not bad.

# precision refers to the quality of positive predictions 0.71 which is also not bad.

# so overall the model performed average. so we need to change the algoritham for this model.




