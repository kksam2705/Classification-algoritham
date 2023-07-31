HEADER
Who:KK
What: Naive base algoritham
Last edited: 2023-04-08
CONTENTS
Introduction
Naive bayes
setup
Introduction
Precision agriculture is in trend nowadays. It helps the farmers to get informed decision about the farming strategy.The science of training
machines to learn and produce models for future predictions is widely used, and not for nothing. Agriculture plays a critical role in the global
economy. With the continuing expansion of the human population understanding worldwide crop yield is central addressing food security
challenges and reducing the impacts of climate change. Crop yield prediction is an important agricultural problem. The Agricultural yield primarily
depends on weather conditions (rain, temperature, etc), pesticides and accurate information about history of crop yield is an important thing for
making decisions related to agricultural risk management and future predictions.
N - ratio of Nitrogen content in soil P - ratio of Phosphorous content in soil K - ratio of Potassium content in soil temperature - temperature in
degree Celsius humidity - relative humidity in % ph - ph value of the soil rainfall - rainfall in mm label - type of crop
Dataset source: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
Naive bayes
Naive Bayes is a Supervised Machine Learning algorithm based on the Bayes Theorem that is used to solve classification problems by following a
probabilistic approach. It is based on the idea that the predictor variables in a Machine Learning model are independent of each other. Meaning
that the outcome of a model depends on a set of independent variables that have nothing to do with each other.
Problem statement
Probabilities of which type of crops is suitable for cultivation based on the criteria such as ph, rainfall,temperature, etc…
Setup
setwd("C:/Users/DELL/Downloads")
getwd()
## [1] "C:/Users/DELL/Downloads"
set working directory to import our dataset
crop <- read.csv("Crop_recommendation.csv")
Read the data into R environment
Checking Null values
library(Amelia)
## Warning: package 'Amelia' was built under R version 4.2.3
## Loading required package: Rcpp
## ##
## ## Amelia II: Multiple Imputation
## ## (Version 1.8.1, built: 2022-11-18)
## ## Copyright (C) 2005-2023 James Honaker, Gary King and Matthew Blackwell
## ## Refer to http://gking.harvard.edu/amelia/ for more information
## ##
missmap(crop)
Explore the data
summary(crop)
## N P K temperature
## Min. : 0.00 Min. : 5.00 Min. : 5.00 Min. : 8.826
## 1st Qu.: 21.00 1st Qu.: 28.00 1st Qu.: 20.00 1st Qu.:22.769
## Median : 37.00 Median : 51.00 Median : 32.00 Median :25.599
## Mean : 50.55 Mean : 53.36 Mean : 48.15 Mean :25.616
## 3rd Qu.: 84.25 3rd Qu.: 68.00 3rd Qu.: 49.00 3rd Qu.:28.562
## Max. :140.00 Max. :145.00 Max. :205.00 Max. :43.675
## humidity ph rainfall label
## Min. :14.26 Min. :3.505 Min. : 20.21 Length:2200
## 1st Qu.:60.26 1st Qu.:5.972 1st Qu.: 64.55 Class :character
## Median :80.47 Median :6.425 Median : 94.87 Mode :character
## Mean :71.48 Mean :6.469 Mean :103.46
## 3rd Qu.:89.95 3rd Qu.:6.924 3rd Qu.:124.27
## Max. :99.98 Max. :9.935 Max. :298.56
names(crop)
## [1] "N" "P" "K" "temperature" "humidity"
## [6] "ph" "rainfall" "label"
Checking the correlation for pairwise among all the
predicators in the dataset.
cor(crop[ , -8])
## N P K temperature humidity
## N 1.00000000 -0.23145958 -0.14051184 0.02650380 0.190688379
## P -0.23145958 1.00000000 0.73623222 -0.12754113 -0.118734116
## K -0.14051184 0.73623222 1.00000000 -0.16038713 0.190858861
## temperature 0.02650380 -0.12754113 -0.16038713 1.00000000 0.205319677
## humidity 0.19068838 -0.11873412 0.19085886 0.20531968 1.000000000
## ph 0.09668285 -0.13801889 -0.16950310 -0.01779502 -0.008482539
## rainfall 0.05902022 -0.06383905 -0.05346135 -0.03008378 0.094423053
## ph rainfall
## N 0.096682846 0.05902022
## P -0.138018893 -0.06383905
## K -0.169503098 -0.05346135
## temperature -0.017795017 -0.03008378
## humidity -0.008482539 0.09442305
## ph 1.000000000 -0.10906948
## rainfall -0.109069484 1.00000000
library(e1071) #for naive bayes algoritham
## Warning: package 'e1071' was built under R version 4.2.3
library(caret) #for data preprocessing
## Warning: package 'caret' was built under R version 4.2.3
## Loading required package: ggplot2
## Warning: package 'ggplot2' was built under R version 4.2.3
## Loading required package: lattice
library(caTools) # split the data for train and test
## Warning: package 'caTools' was built under R version 4.2.3
split the data for test and train
set.seed(1)
split <- sample.split(crop$label, SplitRatio = 0.7)
train_data <- subset(crop, split == "TRUE")
test_data <- subset(crop, split =="FALSE")
Deploy the naive bayes model
set.seed(123) # For reproduceability
result <- naiveBayes(label ~ ., data = train_data, family = "multinomial")
result
##
## Naive Bayes Classifier for Discrete Predictors
##
## Call:
## naiveBayes.default(x = X, y = Y, laplace = laplace, family = "multinomial")
##
## A-priori probabilities:
## Y
## apple banana blackgram chickpea coconut coffee
## 0.04545455 0.04545455 0.04545455 0.04545455 0.04545455 0.04545455
## cotton grapes jute kidneybeans lentil maize
## 0.04545455 0.04545455 0.04545455 0.04545455 0.04545455 0.04545455
## mango mothbeans mungbean muskmelon orange papaya
## 0.04545455 0.04545455 0.04545455 0.04545455 0.04545455 0.04545455
## pigeonpeas pomegranate rice watermelon
## 0.04545455 0.04545455 0.04545455 0.04545455
##
## Conditional probabilities:
## N
## Y [,1] [,2]
## apple 20.47143 11.46099
## banana 101.34286 11.11454
## blackgram 40.01429 12.97656
## chickpea 39.38571 11.69577
## coconut 22.31429 12.39862
## coffee 100.84286 11.78015
## cotton 118.01429 11.09053
## grapes 23.10000 12.72923
## jute 77.67143 10.35419
## kidneybeans 21.11429 11.36170
## lentil 19.71429 12.22630
## maize 78.50000 12.13182
## mango 18.87143 11.84798
## mothbeans 19.57143 10.83683
## mungbean 21.54286 11.31715
## muskmelon 100.25714 11.78268
## orange 19.50000 12.54413
## papaya 49.08571 12.08753
## pigeonpeas 20.71429 11.41718
## pomegranate 19.24286 12.08477
## rice 80.32857 12.09590
## watermelon 100.01429 12.39506
##
## P
## Y [,1] [,2]
## apple 133.58571 8.064042
## banana 82.22857 7.600861
## blackgram 67.55714 7.377278
## chickpea 67.78571 7.208877
## coconut 17.14286 7.947811
## coffee 28.82857 7.211045
## cotton 46.37143 7.635105
## grapes 132.58571 7.616711
## jute 46.41429 6.925050
## kidneybeans 66.47143 7.399023
## lentil 68.60000 7.157443
## maize 48.17143 7.938115
## mango 26.62857 7.061164
## mothbeans 48.11429 7.339674
## mungbean 47.57143 7.810382
## muskmelon 17.88571 7.472739
## orange 16.57143 7.871378
## papaya 58.90000 6.936879
## pigeonpeas 67.64286 7.079335
## pomegranate 18.04286 7.911532
## rice 48.50000 7.414733
## watermelon 17.00000 7.385846
##
## K
## Y [,1] [,2]
## apple 199.914286 3.322051
## banana 50.071429 3.410646
## blackgram 19.342857 3.322674
## chickpea 79.985714 3.299068
## coconut 30.585714 3.113751
## coffee 30.000000 3.088079
## cotton 19.314286 3.052881
## grapes 199.828571 3.370681
## jute 40.014286 3.511855
## kidneybeans 20.157143 3.237622
## lentil 19.214286 2.938465
## maize 19.771429 2.969270
## mango 30.300000 3.085027
## mothbeans 20.085714 3.119564
## mungbean 19.685714 3.187579
## muskmelon 50.285714 3.341192
## orange 9.957143 3.168851
## papaya 49.985714 2.985437
## pigeonpeas 20.200000 2.902273
## pomegranate 39.985714 3.019228
## rice 40.114286 2.891982
## watermelon 50.214286 3.274375
##
## temperature
## Y [,1] [,2]
## apple 22.59058 0.7946282
## banana 27.28301 1.4643216
## blackgram 30.04764 2.6667831
## chickpea 18.86736 1.1477175
## coconut 27.41991 1.3437954
## coffee 25.63910 1.4969598
## cotton 24.00747 1.1001130
## grapes 23.21029 9.8418654
## jute 25.07158 1.1576610
## kidneybeans 20.37034 2.7211267
## lentil 24.23728 3.3015803
## maize 22.41417 2.6795308
## mango 31.10622 2.6618321
## mothbeans 28.01206 2.1221696
## mungbean 28.47549 0.8327194
## muskmelon 28.65453 0.8434818
## orange 23.39035 6.8534942
## papaya 33.96815 6.2694970
## pigeonpeas 28.02477 5.9287863
## pomegranate 22.03030 2.2259221
## rice 23.66992 1.9915774
## watermelon 25.45188 0.8502784
##
## humidity
## Y [,1] [,2]
## apple 92.30501 1.540897
## banana 80.30384 2.977812
## blackgram 65.20568 2.820002
## chickpea 16.70433 1.738737
## coconut 95.00263 2.669986
## coffee 59.74336 5.661566
## cotton 79.82885 3.088613
## grapes 81.78503 1.147129
## jute 79.72386 5.599977
## kidneybeans 21.74186 2.218724
## lentil 65.23182 2.930849
## maize 64.70105 5.427350
## mango 50.19352 2.858317
## mothbeans 53.06803 7.214091
## mungbean 85.32290 2.850170
## muskmelon 92.19570 1.481342
## orange 91.98953 1.367901
## papaya 92.36651 1.311087
## pigeonpeas 47.72933 10.792016
## pomegranate 90.20402 2.767234
## rice 82.36622 1.421458
## watermelon 85.29873 2.959802
##
## ph
## Y [,1] [,2]
## apple 5.943242 0.2622972
## banana 5.983609 0.2751254
## blackgram 7.098010 0.3762074
## chickpea 7.221399 0.7620864
## coconut 5.962484 0.3002321
## coffee 6.782642 0.3948700
## cotton 6.916142 0.6170446
## grapes 6.032760 0.2971168
## jute 6.706524 0.4500477
## kidneybeans 5.742531 0.1498251
## lentil 6.958282 0.5321713
## maize 6.230769 0.4042548
## mango 5.748830 0.6874668
## mothbeans 6.683158 1.8746743
## mungbean 6.704140 0.2843508
## muskmelon 6.363593 0.2399262
## orange 6.979031 0.5731536
## papaya 6.744258 0.1581820
## pigeonpeas 5.776550 0.8716599
## pomegranate 6.403307 0.4977543
## rice 6.312567 0.8023866
## watermelon 6.479607 0.2753886
##
## rainfall
## Y [,1] [,2]
## apple 113.63155 7.054731
## banana 104.94099 9.400104
## blackgram 67.72701 4.268109
## chickpea 80.60085 8.059805
## coconut 175.09968 29.070461
## coffee 161.08023 25.796200
## cotton 81.26426 11.439471
## grapes 69.55849 3.117604
## jute 174.37404 14.559620
## kidneybeans 104.30965 25.527117
## lentil 45.13603 5.978576
## maize 84.81618 15.552570
## mango 94.66879 3.288246
## mothbeans 50.65568 13.903725
## mungbean 48.22746 6.723748
## muskmelon 24.48847 2.838499
## orange 110.27951 5.975064
## papaya 146.18454 64.691531
## pigeonpeas 149.63188 32.741976
## pomegranate 107.48176 2.701954
## rice 235.86817 34.346133
## watermelon 51.22960 5.862716
predict the model
predictions <- predict(result, newdata = test_data)
confusion matrix to check the accuracy of the model
cm <- table(test_data$label, predictions)
cm
## predictions
## apple banana blackgram chickpea coconut coffee cotton grapes jute
## apple 30 0 0 0 0 0 0 0 0
## banana 0 30 0 0 0 0 0 0 0
## blackgram 0 0 30 0 0 0 0 0 0
## chickpea 0 0 0 30 0 0 0 0 0
## coconut 0 0 0 0 30 0 0 0 0
## coffee 0 0 0 0 0 30 0 0 0
## cotton 0 0 0 0 0 0 30 0 0
## grapes 0 0 0 0 0 0 0 30 0
## jute 0 0 0 0 0 0 0 0 30
## kidneybeans 0 0 0 0 0 0 0 0 0
## lentil 0 0 0 0 0 0 0 0 0
## maize 0 0 0 0 0 0 1 0 0
## mango 0 0 0 0 0 0 0 0 0
## mothbeans 0 0 0 0 0 0 0 0 0
## mungbean 0 0 0 0 0 0 0 0 0
## muskmelon 0 0 0 0 0 0 0 0 0
## orange 0 0 0 0 0 0 0 0 0
## papaya 0 0 0 0 0 0 0 0 0
## pigeonpeas 0 0 0 0 0 0 0 0 0
## pomegranate 0 0 0 0 0 0 0 0 0
## rice 0 0 0 0 0 0 0 0 4
## watermelon 0 0 0 0 0 0 0 0 0
## predictions
## kidneybeans lentil maize mango mothbeans mungbean muskmelon
## apple 0 0 0 0 0 0 0
## banana 0 0 0 0 0 0 0
## blackgram 0 0 0 0 0 0 0
## chickpea 0 0 0 0 0 0 0
## coconut 0 0 0 0 0 0 0
## coffee 0 0 0 0 0 0 0
## cotton 0 0 0 0 0 0 0
## grapes 0 0 0 0 0 0 0
## jute 0 0 0 0 0 0 0
## kidneybeans 30 0 0 0 0 0 0
## lentil 0 30 0 0 0 0 0
## maize 0 0 29 0 0 0 0
## mango 0 0 0 30 0 0 0
## mothbeans 0 0 0 0 30 0 0
## mungbean 0 0 0 0 0 30 0
## muskmelon 0 0 0 0 0 0 30
## orange 0 0 0 0 0 0 0
## papaya 0 0 0 0 0 0 0
## pigeonpeas 0 0 0 0 0 0 0
## pomegranate 0 0 0 0 0 0 0
## rice 0 0 0 0 0 0 0
## watermelon 0 0 0 0 0 0 0
## predictions
## orange papaya pigeonpeas pomegranate rice watermelon
## apple 0 0 0 0 0 0
## banana 0 0 0 0 0 0
## blackgram 0 0 0 0 0 0
## chickpea 0 0 0 0 0 0
## coconut 0 0 0 0 0 0
## coffee 0 0 0 0 0 0
## cotton 0 0 0 0 0 0
## grapes 0 0 0 0 0 0
## jute 0 0 0 0 0 0
## kidneybeans 0 0 0 0 0 0
## lentil 0 0 0 0 0 0
## maize 0 0 0 0 0 0
## mango 0 0 0 0 0 0
## mothbeans 0 0 0 0 0 0
## mungbean 0 0 0 0 0 0
## muskmelon 0 0 0 0 0 0
## orange 30 0 0 0 0 0
## papaya 0 30 0 0 0 0
## pigeonpeas 0 0 30 0 0 0
## pomegranate 0 0 0 30 0 0
## rice 0 0 0 0 26 0
## watermelon 0 0 0 0 0 30
model evaluation
confusionMatrix(cm)
## Confusion Matrix and Statistics
##
## predictions
## apple banana blackgram chickpea coconut coffee cotton grapes jute
## apple 30 0 0 0 0 0 0 0 0
## banana 0 30 0 0 0 0 0 0 0
## blackgram 0 0 30 0 0 0 0 0 0
## chickpea 0 0 0 30 0 0 0 0 0
## coconut 0 0 0 0 30 0 0 0 0
## coffee 0 0 0 0 0 30 0 0 0
## cotton 0 0 0 0 0 0 30 0 0
## grapes 0 0 0 0 0 0 0 30 0
## jute 0 0 0 0 0 0 0 0 30
## kidneybeans 0 0 0 0 0 0 0 0 0
## lentil 0 0 0 0 0 0 0 0 0
## maize 0 0 0 0 0 0 1 0 0
## mango 0 0 0 0 0 0 0 0 0
## mothbeans 0 0 0 0 0 0 0 0 0
## mungbean 0 0 0 0 0 0 0 0 0
## muskmelon 0 0 0 0 0 0 0 0 0
## orange 0 0 0 0 0 0 0 0 0
## papaya 0 0 0 0 0 0 0 0 0
## pigeonpeas 0 0 0 0 0 0 0 0 0
## pomegranate 0 0 0 0 0 0 0 0 0
## rice 0 0 0 0 0 0 0 0 4
## watermelon 0 0 0 0 0 0 0 0 0
## predictions
## kidneybeans lentil maize mango mothbeans mungbean muskmelon
## apple 0 0 0 0 0 0 0
## banana 0 0 0 0 0 0 0
## blackgram 0 0 0 0 0 0 0
## chickpea 0 0 0 0 0 0 0
## coconut 0 0 0 0 0 0 0
## coffee 0 0 0 0 0 0 0
## cotton 0 0 0 0 0 0 0
## grapes 0 0 0 0 0 0 0
## jute 0 0 0 0 0 0 0
## kidneybeans 30 0 0 0 0 0 0
## lentil 0 30 0 0 0 0 0
## maize 0 0 29 0 0 0 0
## mango 0 0 0 30 0 0 0
## mothbeans 0 0 0 0 30 0 0
## mungbean 0 0 0 0 0 30 0
## muskmelon 0 0 0 0 0 0 30
## orange 0 0 0 0 0 0 0
## papaya 0 0 0 0 0 0 0
## pigeonpeas 0 0 0 0 0 0 0
## pomegranate 0 0 0 0 0 0 0
## rice 0 0 0 0 0 0 0
## watermelon 0 0 0 0 0 0 0
## predictions
## orange papaya pigeonpeas pomegranate rice watermelon
## apple 0 0 0 0 0 0
## banana 0 0 0 0 0 0
## blackgram 0 0 0 0 0 0
## chickpea 0 0 0 0 0 0
## coconut 0 0 0 0 0 0
## coffee 0 0 0 0 0 0
## cotton 0 0 0 0 0 0
## grapes 0 0 0 0 0 0
## jute 0 0 0 0 0 0
## kidneybeans 0 0 0 0 0 0
## lentil 0 0 0 0 0 0
## maize 0 0 0 0 0 0
## mango 0 0 0 0 0 0
## mothbeans 0 0 0 0 0 0
## mungbean 0 0 0 0 0 0
## muskmelon 0 0 0 0 0 0
## orange 30 0 0 0 0 0
## papaya 0 30 0 0 0 0
## pigeonpeas 0 0 30 0 0 0
## pomegranate 0 0 0 30 0 0
## rice 0 0 0 0 26 0
## watermelon 0 0 0 0 0 30
##
## Overall Statistics
##
## Accuracy : 0.9924
## 95% CI : (0.9824, 0.9975)
## No Information Rate : 0.0515
## P-Value [Acc > NIR] : < 2.2e-16
##
## Kappa : 0.9921
##
## Mcnemar's Test P-Value : NA
##
## Statistics by Class:
##
## Class: apple Class: banana Class: blackgram
## Sensitivity 1.00000 1.00000 1.00000
## Specificity 1.00000 1.00000 1.00000
## Pos Pred Value 1.00000 1.00000 1.00000
## Neg Pred Value 1.00000 1.00000 1.00000
## Prevalence 0.04545 0.04545 0.04545
## Detection Rate 0.04545 0.04545 0.04545
## Detection Prevalence 0.04545 0.04545 0.04545
## Balanced Accuracy 1.00000 1.00000 1.00000
## Class: chickpea Class: coconut Class: coffee Class: cotton
## Sensitivity 1.00000 1.00000 1.00000 0.96774
## Specificity 1.00000 1.00000 1.00000 1.00000
## Pos Pred Value 1.00000 1.00000 1.00000 1.00000
## Neg Pred Value 1.00000 1.00000 1.00000 0.99841
## Prevalence 0.04545 0.04545 0.04545 0.04697
## Detection Rate 0.04545 0.04545 0.04545 0.04545
## Detection Prevalence 0.04545 0.04545 0.04545 0.04545
## Balanced Accuracy 1.00000 1.00000 1.00000 0.98387
## Class: grapes Class: jute Class: kidneybeans Class: lentil
## Sensitivity 1.00000 0.88235 1.00000 1.00000
## Specificity 1.00000 1.00000 1.00000 1.00000
## Pos Pred Value 1.00000 1.00000 1.00000 1.00000
## Neg Pred Value 1.00000 0.99365 1.00000 1.00000
## Prevalence 0.04545 0.05152 0.04545 0.04545
## Detection Rate 0.04545 0.04545 0.04545 0.04545
## Detection Prevalence 0.04545 0.04545 0.04545 0.04545
## Balanced Accuracy 1.00000 0.94118 1.00000 1.00000
## Class: maize Class: mango Class: mothbeans Class: mungbean
## Sensitivity 1.00000 1.00000 1.00000 1.00000
## Specificity 0.99842 1.00000 1.00000 1.00000
## Pos Pred Value 0.96667 1.00000 1.00000 1.00000
## Neg Pred Value 1.00000 1.00000 1.00000 1.00000
## Prevalence 0.04394 0.04545 0.04545 0.04545
## Detection Rate 0.04394 0.04545 0.04545 0.04545
## Detection Prevalence 0.04545 0.04545 0.04545 0.04545
## Balanced Accuracy 0.99921 1.00000 1.00000 1.00000
## Class: muskmelon Class: orange Class: papaya
## Sensitivity 1.00000 1.00000 1.00000
## Specificity 1.00000 1.00000 1.00000
## Pos Pred Value 1.00000 1.00000 1.00000
## Neg Pred Value 1.00000 1.00000 1.00000
## Prevalence 0.04545 0.04545 0.04545
## Detection Rate 0.04545 0.04545 0.04545
## Detection Prevalence 0.04545 0.04545 0.04545
## Balanced Accuracy 1.00000 1.00000 1.00000
## Class: pigeonpeas Class: pomegranate Class: rice
## Sensitivity 1.00000 1.00000 1.00000
## Specificity 1.00000 1.00000 0.99369
## Pos Pred Value 1.00000 1.00000 0.86667
## Neg Pred Value 1.00000 1.00000 1.00000
## Prevalence 0.04545 0.04545 0.03939
## Detection Rate 0.04545 0.04545 0.03939
## Detection Prevalence 0.04545 0.04545 0.04545
## Balanced Accuracy 1.00000 1.00000 0.99685
## Class: watermelon
## Sensitivity 1.00000
## Specificity 1.00000
## Pos Pred Value 1.00000
## Neg Pred Value 1.00000
## Prevalence 0.04545
## Detection Rate 0.04545
## Detection Prevalence 0.04545
## Balanced Accuracy 1.00000
Result
The overall accuracy of the model is 99.24% which means our model correctly classified the observation of dataset.
95% confidence interval is a true population parameter. in our case the CI ranges between is 98.24% to 99.75%. which means true accuracy of the
model lies between this range.
No information rate is a statstical bench mark that is used to evaluate the classification model which means prior probabilities of the most frequent
classes in the dataset.the accuracy must be greater than no information which means making meaningful predictions which is useful for practical
information.
Kappa value is the agreement between predicted value and actutal values and it ranges from -1 to 1. in general the value below 0.4 indicates poor
agreement and and value above 0.8 indicates strong agreement.in our case 0.99 which is suggest the strong agreement.
The McNemar’s Test is a statistical test used to determine if there is a significant difference between two related proportions. It is often used in
cases where the data is paired, such as in a before-and-after study, or when two classifiers are tested on the same dataset in our case its NA
which means no paired sample tested or compared with the analysis.
p-value(accuracy > NIR) which is a null hypothesis testing.in our case value is(2.2e-16) less than alpha value(0.05) which means reject null
hypotheis which suggest that our model accuracy is significant.
The statistics are also broken down by class. For each class, we have metrics such as sensitivity (the proportion of actual positives that are
correctly identified), specificity (the proportion of actual negatives that are correctly identified), positive predictive value (the proportion of predicted
positives that are true positives), and negative predictive value (the proportion of predicted negatives that are true negatives). We also have
prevalence (the proportion of the data that belongs to each class), detection rate (the proportion of actual positives that are correctly identified),
and detection prevalence (the proportion of predicted positives).
For example we will take apple the senistivity, specificity, ppv, and npv are all 1 which means our model correctly classifies as a apple. while
prevalance and detection rate are equal prevalence proportion is 0.04545 and correctly detected 0.0454 which is a detection rate.it means it
detected all our proportions correctly.detection prevalance is the proportion pf predicted positives which is 0.0454 which is accurate.and balanced
accuracy is 1 which is perfect classification.
Overall our model classifies all the crops correctly based on the certain features for the cultivation.
