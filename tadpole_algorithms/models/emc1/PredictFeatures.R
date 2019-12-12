str_exp =  getwd()
str_out = paste(str_exp,"/IntermediateData/PredictionMatrix.csv",sep="")
library(nlme)

mydata = read.csv(paste(str_exp,"/IntermediateData/LongTADPOLE.csv",sep=""))
#mydata <- mydata[complete.cases(mydata[,c('Diagnosis')]),]
ds = read.csv(paste(str_exp,"/IntermediateData/PatientStages.csv",sep=""),header=FALSE)
mydata['DiseaseState']=ds

#ds = read.csv(paste(str_exp,"PatientStagesCluster.csv",sep=""),header=TRUE)
#mydata['DiseaseState']=ds$Stages

model_DiseaseState<- lme(fixed=DiseaseState~1, random=~AGE+AGE^2+AGE^3|RID ,data=mydata, control = lmeControl(opt='optim',optimMethod="L-BFGS-B",maxIter=20,returnObject=TRUE))
summary(model_DiseaseState)

lb2_subjects = read.csv(paste(str_exp,"/IntermediateData/ToPredict.csv",sep=""),header=FALSE)
lb2_subjects = data.matrix(lb2_subjects)

RID = rep(lb2_subjects, each=60)
PredictionMatrix <- data.frame(RID)
PredictionMatrix['AGE'] = -1

FM = 1:60
FMR=rep(FM,length(lb2_subjects))
PredictionMatrix['Forecast Month'] = FMR

years = 2018:2022
months = 1:12

for (rid in lb2_subjects)
{
  ForecastDate <- matrix(nrow=0, ncol=1)  
  idx1 <- which(mydata$RID == rid)
  edate_subject <- mydata[idx1,'EXAMDATE'][1]
  age_subject <- mydata[idx1,'AGE'][1]
  
  y_bl <- as.numeric(substr(edate_subject,1,4))
  m_bl <- as.numeric(substr(edate_subject,6,7))
  Age_prediction <- matrix(nrow=0, ncol=1) 
  for (y in years)
  {
    Age_prediction  <-  c(Age_prediction,age_subject + (y + (months/12.0)) - (y_bl+ (m_bl/12.0)))
    ForecastDate <- c(ForecastDate, sprintf("%04d-%02d", y,months))
  }
  idx_pred <- PredictionMatrix$RID == rid
  PredictionMatrix$AGE[idx_pred] <- Age_prediction
  PredictionMatrix$ICV_bl[idx_pred] <- mydata[idx1,'ICV_bl'][1]
  PredictionMatrix$PTGENDER[idx_pred] <- mydata[idx1,'PTGENDER'][1]
  PredictionMatrix$PTEDUCAT[idx_pred] <- mydata[idx1,'PTEDUCAT'][1]
  d <- mydata[idx1,'DiseaseState']
  PredictionMatrix[idx_pred,'DiseaseState'] = d[length(d)]
  if (nrow(mydata[idx1,])>3)
  {
    model_DS <- lm(DiseaseState~AGE+AGE^2+AGE^3, data=mydata[idx1,])
    dpred = predict(model_DS,PredictionMatrix[idx_pred,])
    PredictionMatrix[idx_pred,'DiseaseState'] = dpred
  }  else
  {
    dpred = predict(model_DiseaseState,PredictionMatrix[idx_pred,])
    PredictionMatrix[idx_pred,'DiseaseState'] = dpred
  }
}

PredictionMatrix$PTGENDER <- factor(PredictionMatrix$PTGENDER,labels=c("Female","Male"))
PredictionMatrix['Forecast Date'] = ForecastDate

write.csv(PredictionMatrix,str_out,row.names=FALSE)

library(nlme)
library(splines)

for (i in 8:ncol(mydata))
{
  idx_4 <- mydata[,i]==-4
  idx_4[is.na(idx_4)]=FALSE
  mydata[idx_4,i]=NA
}
mydata <- mydata[complete.cases(mydata[,c('ICV_bl')]),]
h <- colnames(mydata)
idx_Feats = read.csv(paste(str_exp,"/IntermediateData/FeatureIndices.csv",sep=""))
idx_Feats$x = idx_Feats$x+8

PredictionMatrix <- read.csv(paste(str_exp,"/IntermediateData/PredictionMatrix.csv",sep=""))

mydata$ICV_bl[is.na(mydata$ICV_bl)]=mean(mydata$ICV_bl,na.rm=TRUE)
mydata_ctrl <- mydata[mydata$Diagnosis==1,]

mydata_pred <- cbind(NA,NA)
mydata_pred <- cbind(mydata)


for (i in 1:250)
{
  print (i)
  options(warn=-1)
  data_featurei <- cbind(NA,NA)
  data_featurei <- cbind(mydata)
  data_featurei <- data_featurei[complete.cases(data_featurei[,h[idx_Feats$x[i]]]),]
  data_featurei['Featurei'] <- data_featurei[h[idx_Feats$x[i]]]
  model_featurei<- lme(fixed=Featurei~(ns(AGE,2)*ns(DiseaseState,4))+PTGENDER+PTEDUCAT+ICV_bl, random=~DiseaseState+DiseaseState^2|RID,data=data_featurei, control = lmeControl(opt="optim",optimMethod="L-BFGS-B",maxIter=20,returnObject=TRUE))
  PredictionMatrix[,h[idx_Feats$x[i]]] = predict(model_featurei,PredictionMatrix)
  mydata_pred[,h[idx_Feats$x[i]]] = predict(model_featurei,mydata_pred)
  
  sata_featurei = subset(mydata_ctrl, select = c("RID","AGE","PTGENDER","PTEDUCAT","ICV_bl","APOE4"))
  sata_featurei['Featurei'] <- mydata_ctrl[h[idx_Feats$x[i]]]
  sata_featurei <- sata_featurei[complete.cases(sata_featurei),]
  if (nrow(sata_featurei)>0)
  {
    smodel_featurei <- lm(Featurei ~ (ns(AGE,2))+PTEDUCAT+PTGENDER+ICV_bl,data=sata_featurei)
    data_Mean <- cbind(NA,NA)
    data_Mean <- cbind(PredictionMatrix)
    data_Mean$AGE = mean(data_featurei$AGE)
    data_Mean$ICV_bl = mean(data_featurei$ICV_bl)
    data_Mean$PTEDUCAT = mean(data_featurei$PTEDUCAT)
    data_Mean$PTGENDER = "Male"                    
    PredictionMatrix[,h[idx_Feats$x[i]]] <- PredictionMatrix[,h[idx_Feats$x[i]]] - predict(smodel_featurei,PredictionMatrix) + predict(smodel_featurei,data_Mean)
    data_Mean <- cbind(NA,NA)
    data_Mean <- cbind(mydata_pred)
    data_Mean$AGE = mean(data_featurei$AGE)
    data_Mean$ICV_bl = mean(data_featurei$ICV_bl)
    data_Mean$PTEDUCAT = mean(data_featurei$PTEDUCAT)
    data_Mean$PTGENDER = "Male"  
    mydata_pred[,h[idx_Feats$x[i]]] <- mydata_pred[,h[idx_Feats$x[i]]] - predict(smodel_featurei,mydata_pred) + predict(smodel_featurei,data_Mean)
  }
}
 
write.csv(PredictionMatrix,paste(str_exp,"/IntermediateData/PredictionMatrix.csv",sep=""),row.names = FALSE)
write.csv(mydata_pred,paste(str_exp,"/IntermediateData/PredictedLongTADPOLE.csv",sep=""),row.names = FALSE)
#write.csv(PredictionMatrix,paste(str_exp,"PredictionMatrixCluster.csv",sep=""),row.names = FALSE)
#write.csv(mydata_pred,paste(str_exp,"PredictedLongTADPOLECluster.csv",sep=""),row.names = FALSE)
