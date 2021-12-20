library(tidyverse)
library(readr)
library(Matrix)
library(glmnet)
library(lubridate)
library(randomForest)
library(DMwR)
library(ranger)
library(adabag)
library(caret)
library(fastAdaboost)
library (reshape)
library(xgboost)
library(precrec)
library(tidymodels)
library(purrr)
library(modelr)
library(rlist)
library(miscTools)
library(neuralnet)
library(nnet)

# tab_fin_an is the table used
#COUNT_ANNEE is the binary variable we tried to predict

tab$COUNT_ANNEE<-as.factor(tab$COUNT_ANNEE)
n_col<- ncol(tab)
split <- sample(nrow(tab), floor(0.8*nrow(tab)))
train <-tab[split,]
test <- tab[-split,]

X_train = as.matrix(train %>% select(-COUNT_ANNEE))
y_train = train$COUNT_ANNEE
y_train <- as.numeric(y_train)-1
dtrain <- xgb.DMatrix(data = X_train,label = y_train) 

X_test = as.matrix(test %>% select(-COUNT_ANNEE))
y_test = test$COUNT_ANNEE
y_test <- as.numeric(y_test)-1
dtest <- xgb.DMatrix(data = X_test,label = y_test) 

X_2018 = as.matrix(tab_2018 %>% select(-COUNT_ANNEE))
y_2018 = tab_2018$COUNT_ANNEE
y_2018 <- as.numeric(y_2018)-1
X_2019 = as.matrix(tab_2019 %>% select(-COUNT_ANNEE))
y_2019 = tab_2019$COUNT_ANNEE
y_2019 <- as.numeric(y_2019)-1
dto_2018 <- xgb.DMatrix(data = X_2018,label = y_2018) 
dto_2019 <- xgb.DMatrix(data = X_2019,label = y_2019) 
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1,eval_metric ="aucpr")

xgb_model = xgb.train(params = params, data= dtrain,nrounds = 100)
# variable importance 
importance_xgb <- xgb.importance(model = xgb_model)
write.table(importance_xgb, file="importance_xgb.csv",col.names=TRUE, row.names=FALSE, sep=";",dec=".",fileEncoding = "UTF-8")

# View(importance_swi)

predicted.xgb.test = predict(xgb_model, dtest)

num_predicted.xgb.test <- as.numeric(predicted.xgb.test)
num_label <- as.numeric(test$COUNT_ANNEE)
xgbcurves_TOT <- evalmod(scores = num_predicted.xgb.test, labels = num_label)
autoplot(xgbcurves_TOT)
xgbcurves_TOT

tab.glmnet <- glmnet(X_train,y_train,family ="binomial",standardize = T)
predicted.glmnet.test <- predict(tab.glmnet ,X_test,type="response",s=c(0))
num_predicted.glmnet.test <- as.numeric(predicted.glmnet.test)
num_label <- as.numeric(test$COUNT_ANNEE)
glmnetcurves_TOT <- evalmod(scores = num_predicted.glmnet.test, labels = num_label)
autoplot(glmnetcurves_TOT)
glmnetcurves_TOT
cs <- as.matrix(coef(tab.glmnet, s=c(0)))
var_imp_glmnet <- as.data.frame(cs)
var_imp_glmnet <- cbind(var_imp_glmnet,names(train))

sds <- apply(X_train, 2, sd)
cs <- as.matrix(coef(tab.glmnet, s = c(0)))
std_coefs <- cs[-1, 1] * sds
var_imp_glmnet2 <- as.data.frame(std_coefs)
var_imp_glmnet2 <- cbind(var_imp_glmnet2,names(train))


write.table(var_imp_glmnet2, file="importance_glmnet2.csv",col.names=T, row.names=T, sep=";",dec=".",fileEncoding = "UTF-8")

tab.rf<-ranger(COUNT_ANNEE~.,data=train,probability = T,importance = "impurity_corrected")

predicted.rf.test <- predict(tab.rf,data=test)
num_predicted.rf.test <- as.numeric(predicted.rf.test$predictions[,2])
num_label <- as.numeric(test$COUNT_ANNEE)-1
rfcurves_TOT <- evalmod(scores = num_predicted.rf.test, labels = num_label)
autoplot(rfcurves_TOT)
rfcurves_TOT
imp_rf_2 <- importance(tab.rf)
imp_rf <-importance_pvalues(tab.rf,method = "altmann",formula=COUNT_ANNEE~.,data=train)
write.table(imp_rf_2, file="importance_rf2.csv",col.names=T, row.names=T, sep=";",dec=".",fileEncoding = "UTF-8")

library(ggfortify)
autoplot(rfcurves_TOT) 

PRED_TOT_TOT_df <-  cbind(num_predicted.xgb.test,num_predicted.rf.test,num_predicted.glmnet.test)
mean_probs_TOT_TOT<- rowMeans(PRED_TOT_TOT_df)
TOT_test <- evalmod(scores = mean_probs_TOT_TOT, labels = num_label)
autoplot(TOT_test)
TOT_test


fscore <- data.frame(seq(0,1,0.001))
fscore$FSCORE_GLMNET <- NA
fscore$FSCORE_XGB <- NA
fscore$FSCORE_RF <- NA
fscore$FSCORE_TOT <- NA
names(fscore)[1]<- "seuil"

fsc <- function (var_to_predict, predicted,seuil) {
  TP<-sum(predicted>=seuil&var_to_predict==1)
  TN<-sum(predicted<seuil&var_to_predict==0)
  FN<-sum(predicted<seuil&var_to_predict==1)
  FP<-sum(predicted>=seuil&var_to_predict==0)
  precision<-TP/(TP+FP)
  recall<-TP/(TP+FN)
  fscore<-2*(recall*precision)/(recall+precision)
  return(fscore)
  
}

for (i in 1:1000) {
  
  fscore$FSCORE_XGB[i] <-fsc(num_label,num_predicted.xgb.test,fscore$seuil[i]) 
  fscore$FSCORE_RF[i] <-fsc(num_label,num_predicted.rf.test,fscore$seuil[i]) 
  fscore$FSCORE_TOT[i] <-fsc(num_label,mean_probs_TOT_TOT,fscore$seuil[i]) 
  fscore$FSCORE_GLMNET[i] <-fsc(num_label,num_predicted.glmnet.test,fscore$seuil[i]) 
}          

max_glm <- max(fscore$FSCORE_GLMNET,na.rm = T)
max_rf <- max(fscore$FSCORE_RF,na.rm = T)
max_xgb <- max(fscore$FSCORE_XGB,na.rm = T)
max_tot <- max(fscore$FSCORE_TOT,na.rm = T)

seuil_glm <- fscore %>% filter(fscore$FSCORE_GLMNET==max_glm) %>% select(seuil)
seuil_rf <- fscore %>% filter(fscore$FSCORE_RF==max_rf)%>% select(seuil)
seuil_xgb<-fscore %>% filter(fscore$FSCORE_XGB==max_xgb)%>% select(seuil)
seuil_tot <- fscore %>% filter(fscore$FSCORE_TOT==max_tot)%>% select(seuil)

pred_num_xgb_test <- num_predicted.xgb.test
pred_num_xgb_test[pred_num_xgb_test<0.291]<-0
pred_num_xgb_test[pred_num_xgb_test>=0.291]<-1

pred_num_glmnet_test <- num_predicted.glmnet.test
pred_num_glmnet_test[pred_num_glmnet_test<0.221]<-0
pred_num_glmnet_test[pred_num_glmnet_test>=0.221]<-1

pred_num_rf_test <- num_predicted.rf.test
pred_num_rf_test[pred_num_rf_test<0.306]<-0
pred_num_rf_test[pred_num_rf_test>=0.306]<-1

pred_num_tot_test <- mean_probs_TOT_TOT
pred_num_tot_test[pred_num_tot_test<0.264]<-0
pred_num_tot_test[pred_num_tot_test>=0.264]<-1


mc_test_xgb <- confusionMatrix(data=pred_num_xgb_test,reference =  num_label)
mc_test_glment <- confusionMatrix(data=pred_num_glmnet_test,reference =  num_label)
mc_test_rf <- confusionMatrix(data=pred_num_rf_test,reference =  num_label)
mc_test_tot <- confusionMatrix(data=pred_num_tot_test,reference =  num_label)

