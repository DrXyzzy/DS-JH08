---
title: "Prediction Assignment Writeup"
author: "Hal Snyder"
date: "December 23, 2016"
output:
  html_document:
    keep_md: true
---


# Prediction Assignment Writeup

## Introduction

Six participants fitted with wearable sensors performed 10 sets of bicep curls in five different fashions (exercise classes `A` through `E`). The goal of this project is to construct and evaluate a machine learning model to predict which exercise classes was performed for given observations, using sensor readings as predictors.

Reference: [Human Activity Recognition, Weight-Lifting Exercises](http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises#ixzz4TjKsbTNc):  
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

## Setup and Input

Libraries


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

Load Data


```r
training <- read.csv("pml-training.csv",na.strings=c("NA","","#DIV/0!"))
```

```
## Warning in file(file, "rt"): cannot open file 'pml-training.csv': No such
## file or directory
```

```
## Error in file(file, "rt"): cannot open the connection
```

```r
dim(training)
```

```
## Error in eval(expr, envir, enclos): object 'training' not found
```

```r
testing <- read.csv("pml-testing.csv",na.strings=c("NA","","#DIV/0!"))
```

```
## Warning in file(file, "rt"): cannot open file 'pml-testing.csv': No such
## file or directory
```

```
## Error in file(file, "rt"): cannot open the connection
```

```r
dim(testing)
```

```
## Error in eval(expr, envir, enclos): object 'testing' not found
```

## Tidy the Data

There are 60 variables in the training set with no missing values. The other 100 variables each have more than 19000 missing values; drop these from training and test sets.


```r
na.counts <- colSums(is.na(training))
```

```
## Error in is.data.frame(x): object 'training' not found
```

```r
table(na.counts)
```

```
## Error in table(na.counts): object 'na.counts' not found
```

```r
hist(na.counts)
```

```
## Error in hist(na.counts): object 'na.counts' not found
```

```r
training.no.na <- training[na.counts == 0]
```

```
## Error in eval(expr, envir, enclos): object 'training' not found
```

```r
dim(training.no.na)
```

```
## Error in eval(expr, envir, enclos): object 'training.no.na' not found
```

```r
testing.no.na <- testing[na.counts == 0]
```

```
## Error in eval(expr, envir, enclos): object 'testing' not found
```

```r
dim(testing.no.na)
```

```
## Error in eval(expr, envir, enclos): object 'testing.no.na' not found
```

Drop more variables

- drop column 1, it is row number in the csv file, no predictive value
- drop timestamps (columns 3-5), see if we can get by without using them
- drop new_window (column 6) logical variable which is mostly `no` in training


```r
table(training$new_window)
```

```
## Error in table(training$new_window): object 'training' not found
```

```r
train.noX <- training.no.na[c(-1, -3, -4, -5, -6)]
```

```
## Error in eval(expr, envir, enclos): object 'training.no.na' not found
```

```r
dim(train.noX)
```

```
## Error in eval(expr, envir, enclos): object 'train.noX' not found
```

```r
test.noX <- testing.no.na[c(-1, -3, -4, -5, -6)]
```

```
## Error in eval(expr, envir, enclos): object 'testing.no.na' not found
```

```r
dim(test.noX)
```

```
## Error in eval(expr, envir, enclos): object 'test.noX' not found
```
## Explore and Preprocess Data

Variable `user_name` (column 1) appears to be significant. (Note that taking `user_name` as predictor implies the model will require calibrating to each new user.)


```r
table(training$user_name)
```

```
## Error in table(training$user_name): object 'training' not found
```

```r
plot(training[,c("user_name","classe")])
```

```
## Error in plot(training[, c("user_name", "classe")]): object 'training' not found
```

Convert to `user_name` to 6 numeric variables (dummy variables) to allow `gbm` modeling.


```r
dv <- dummyVars(~ user_name,data=test.noX)
```

```
## Error in is.data.frame(data): object 'test.noX' not found
```

```r
train.dv <- cbind(predict(dv,newdata=train.noX),train.noX[,-1])
```

```
## Error in predict(dv, newdata = train.noX): object 'dv' not found
```

```r
dim(train.dv)
```

```
## Error in eval(expr, envir, enclos): object 'train.dv' not found
```

```r
test.dv <- cbind(predict(dv,newdata=test.noX),test.noX[,-1])
```

```
## Error in predict(dv, newdata = test.noX): object 'dv' not found
```

```r
dim(test.dv)
```

```
## Error in eval(expr, envir, enclos): object 'test.dv' not found
```
Check for near-zero covariance to see if we can drop more variables. No, there are not any variables with zero variance or near-zero variance.

```r
nzc <- nearZeroVar(train.dv, saveMetrics = T)
```

```
## Error in nzv(x, freqCut = freqCut, uniqueCut = uniqueCut, saveMetrics = saveMetrics, : object 'train.dv' not found
```

```r
table(nzc$nzv)
```

```
## Error in table(nzc$nzv): object 'nzc' not found
```

```r
table(nzc$zeroVar)
```

```
## Error in table(nzc$zeroVar): object 'nzc' not found
```
Compute principal component analysis to see if some of the variables can be dropped. Gradual fall-off of curve suggests we can't reduce number of variables significantly by dropping low-importance variables, so won't use PCA for the model.

```r
p0 <- prcomp(subset(train.dv, select = -c(classe)))
```

```
## Error in subset(train.dv, select = -c(classe)): object 'train.dv' not found
```

```r
plot(p0$sdev)
```

```
## Error in plot(p0$sdev): object 'p0' not found
```

Subset the training set to reduce model run time. Use 1/10 of original training set.

```r
set.seed(107)
inTr <- createDataPartition(y=train.dv$classe,p=0.1,list=F)
```

```
## Error in createDataPartition(y = train.dv$classe, p = 0.1, list = F): object 'train.dv' not found
```

```r
train.dv.subset <- train.dv[inTr,]
```

```
## Error in eval(expr, envir, enclos): object 'train.dv' not found
```

```r
dim(train.dv.subset)
```

```
## Error in eval(expr, envir, enclos): object 'train.dv.subset' not found
```

## Buid and Evaluate the Model

Each section addresses one of the 4 parts of the project assignment.

### 1. How the Model is Built

Because outcome is a nominal variable, use gradient-boosting multinomial logistic regression (`gbm`) with k-fold cross-validation. Report on run-time of the model on the training subset used.


```r
t1 <- Sys.time()
train_control<- trainControl(method="cv", number=10, savePredictions = TRUE)
mod1 <- train(classe ~ ., data=train.dv.subset, trControl=train_control, method="gbm", verbose=F)
```

```
## Error in eval(expr, envir, enclos): object 'train.dv.subset' not found
```

```r
Sys.time() - t1
```

```
## Time difference of 0.005096674 secs
```
Compute confusion matrix for training subset.


```r
confusionMatrix.train(mod1)
```

```
## Error in match(x, table, nomatch = 0L): object 'mod1' not found
```
Compute confusion matrix obtained by applying model to entire training set.

```r
confusionMatrix(train.dv$classe,predict(mod1,newdata=train.dv))$table
```

```
## Error in confusionMatrix(train.dv$classe, predict(mod1, newdata = train.dv)): object 'train.dv' not found
```

```r
confusionMatrix(train.dv$classe,predict(mod1,newdata=train.dv))$overall[1]
```

```
## Error in confusionMatrix(train.dv$classe, predict(mod1, newdata = train.dv)): object 'train.dv' not found
```

### 2. How Cross-Validation Is Used

K-fold cross-validation is used with k = 10.

### 3. Expected Out-of-Sample Error

Because 9/10 of training set was not used to build the model, use it to estimate out-of-sample error.

```r
train.oos.subset <- train.dv[-inTr,]
```

```
## Error in eval(expr, envir, enclos): object 'train.dv' not found
```

```r
dim(train.oos.subset)
```

```
## Error in eval(expr, envir, enclos): object 'train.oos.subset' not found
```

```r
cm.oos <- confusionMatrix(train.oos.subset$classe,
                          predict(mod1,newdata=train.oos.subset))
```

```
## Error in confusionMatrix(train.oos.subset$classe, predict(mod1, newdata = train.oos.subset)): object 'train.oos.subset' not found
```

```r
cm.oos$overall[1]
```

```
## Error in eval(expr, envir, enclos): object 'cm.oos' not found
```

```r
print(paste('expected out of sample error', as.numeric(1 - cm.oos$overall[1])))
```

```
## Error in paste("expected out of sample error", as.numeric(1 - cm.oos$overall[1])): object 'cm.oos' not found
```

### 4. Why I Made the Choices I Did

- A subset of the initial training set was used to build the model because
    - build time with the full training set was > 20 minutes
    - 10% subset gave sufficient accuracy
- Gradient-boosting multinomial logistic regression (`gbm`) was used because outcome is a nominal variable and most predictors take on continuous values.
- K-fold cross-validation with value of 10 for k because that value is efficient for 60 predictors; leave-one-out cross-validation would have been too slow.

## Predict Test Set Outcomes

Here are the predictions provided by the above model for the 20 observations in the test set.

```r
predict(mod1,newdata=test.dv)
```

```
## Error in predict(mod1, newdata = test.dv): object 'mod1' not found
```

## Conclusion

Expected accuracy near 95% is obtained modeling the given data using gradient-boosted multinomial logistic regression with 10-fold cross-validation.
