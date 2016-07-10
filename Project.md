Introduction
============

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. The goal of this project
is to predict the manner in which they did the exercise. This is the
"classe" variable in the training set.

Loading required Library and datasets
=====================================

**x** represents train data while **y** represents test data.

    suppressMessages(library(caret))

    ## Warning: package 'caret' was built under R version 3.3.1

    suppressMessages(library(rpart))

    ## Warning: package 'rpart' was built under R version 3.3.1

    suppressMessages(library(randomForest))
    suppressMessages(library(rattle))
    x <- read.csv("C:/Users/shreyas768/Downloads/pml-training.csv", sep=",", na.strings=c("","NA","#DIV/0!"), stringsAsFactor=FALSE, header = TRUE)
    y <- read.csv("C:/Users/shreyas768/Downloads/pml-testing.csv", sep=",", na.strings=c("","NA","#DIV/0!), stringsAsFactors = FALSE", header =TRUE))
    dim(x)

    ## [1] 19622   160

    dim(y)

    ## [1]  20 160

Removing Unnecessary columns(Data cleaning)
===========================================

The training dataset has 19622 observations and 160 variables and
testing dataset has 20 observations and equal number of variables. The
loaded data contains some columns which has only NA values and hence
they need to be removed.

    x <- x[,colSums(!is.na(x)) > 0]
    x <- x[, colSums(is.na(x)) == 0]
    y <- y[,colSums(!is.na(y)) > 0]

Also,first seven predictors have very low predicting power for outcome
classe. Hence, we will remove them

    x <- x[,-c(1:7)]
    y <- y[,-c(1:7)]
    dim(x)

    ## [1] 19622    53

    dim(y)

    ## [1] 20 53

The training dataset now has 19622 observations and 53 variables and
testing dataset has 20 observations and equal number of variables.

Data Spliting
=============

We split the cleaned Training dataset **x** further into Training set
(**Train**, 70%) for prediction and a validation set (**Test**,30%)

    set.seed(1234)
    x$classe <- as.factor(x$classe)
    inTrain <- createDataPartition(x$classe,p=0.7,list = FALSE)
    Train <- x[inTrain,]
    Test <- x[-inTrain,]

Prediction Algorithm
====================

Let us first use Classification trees to predict the outcome.

Classification Trees
====================

    TC <- trainControl(method = "cv", number = 5)
    modFit <- train(classe ~ .,method="rpart",data=Train, trControl=TC)
    plot(modFit)

![](Project_files/figure-markdown_strict/unnamed-chunk-5-1.png)

    fancyRpartPlot(modFit$finalModel)

![](Project_files/figure-markdown_strict/unnamed-chunk-6-1.png)

Predict outcomes using validation (**test**) set

    r <- predict(modFit,Test)
    (t <- confusionMatrix(Test$classe,r))

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1530   35  105    0    4
    ##          B  486  379  274    0    0
    ##          C  493   31  502    0    0
    ##          D  452  164  348    0    0
    ##          E  168  145  302    0  467
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.489           
    ##                  95% CI : (0.4762, 0.5019)
    ##     No Information Rate : 0.5317          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.3311          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.4890   0.5027   0.3279       NA  0.99151
    ## Specificity            0.9478   0.8519   0.8797   0.8362  0.88641
    ## Pos Pred Value         0.9140   0.3327   0.4893       NA  0.43161
    ## Neg Pred Value         0.6203   0.9210   0.7882       NA  0.99917
    ## Prevalence             0.5317   0.1281   0.2602   0.0000  0.08003
    ## Detection Rate         0.2600   0.0644   0.0853   0.0000  0.07935
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638  0.18386
    ## Balanced Accuracy      0.7184   0.6773   0.6038       NA  0.93896

     t$overall[1]

    ##  Accuracy 
    ## 0.4890399

Accuracy rate is only 48.9%, i.e out-of-sample is 51.1% and hence
classification tree doesn't predict the outcome well. We will now try
Random Forrest.

Random Forests
==============

    control <- trainControl(method = "cv", number = 5,allowParallel = TRUE)
    (fit <- randomForest(classe~.,data = Train, importance = TRUE))

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = Train, importance = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 7
    ## 
    ##         OOB estimate of  error rate: 0.48%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3904    2    0    0    0 0.0005120328
    ## B   13 2641    4    0    0 0.0063957863
    ## C    0   13 2381    2    0 0.0062604341
    ## D    0    0   23 2228    1 0.0106571936
    ## E    0    0    1    7 2517 0.0031683168

    pred <- predict(fit,Test,type = "class")
    s <- confusionMatrix(Test$classe,pred)
    s$overall[1]

    ##  Accuracy 
    ## 0.9962617

The accuracy is 0.9962617 and so the out-of-sample error rate is
0.0037383. Thus, for this dataset Random Forests is way better than
Classification Tree. Now we will use Random Forests to predict the
outcome variable Classe of Testing dataset (**y**).

Prediction
==========

    predict(fit,y)

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
