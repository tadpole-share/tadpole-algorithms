# Transfer linear mixed model values to time -> predict feature values
# Estimate Future Disease State

library(nlme)
library(splines)

str_exp =  getwd()


str_out = paste(str_exp,"/IntermediateData/TADPOLE_Submission_EMC1.csv",sep="")
mydata = read.csv(paste(str_exp,"/IntermediateData/LongTADPOLE.csv",sep=""))
#mydata <- mydata[complete.cases(mydata[,c('Diagnosis')]),]
#ds = read.csv(paste(str_exp,"PatientStagesCluster.csv",sep=""),header=TRUE)
#mydata['DiseaseState']=ds$Stages
ds = read.csv(paste(str_exp,"/IntermediateData/PatientStages.csv",sep=""),header=FALSE)

mydata['DiseaseState']=ds
#mydata['Rate']=ds$Rate
lb2_subjects = read.csv(paste(str_exp,"/IntermediateData/ToPredict.csv",sep=""),header=FALSE)
lb2_subjects = data.matrix(lb2_subjects)

# Model Fitting for ADAS
data_ADAS <- cbind(NA,NA)
data_ADAS <- cbind(mydata)
data_ADAS <- data_ADAS[complete.cases(data_ADAS[,c('ADAS13','DiseaseState')]),]
model_ADAS<- lme(fixed=ADAS13~(ns(AGE,2) * ns(DiseaseState,4))+PTEDUCAT, random=~(AGE*DiseaseState)+ (AGE^2*DiseaseState^2)|RID,data=data_ADAS, control = lmeControl(opt="optim",optimMethod="L-BFGS-B",maxIter=20,returnObject=TRUE))
summary(model_ADAS)

# Model Fitting for Vetricles
data_Ventricles <- cbind(NA,NA)
data_Ventricles <- cbind(mydata)
data_Ventricles <- data_Ventricles[complete.cases(data_Ventricles[,c('Ventricles')]),]
data_Ventricles <- data_Ventricles[complete.cases(data_Ventricles[,c('ICV_bl')]),]
model_Ventricles<- lme(fixed=Ventricles~(ns(AGE,2) * ns(DiseaseState,4))+PTGENDER+ICV_bl, random=~AGE+DiseaseState+AGE^2+DiseaseState^2|RID ,data=data_Ventricles, control = lmeControl(opt='optim',optimMethod="L-BFGS-B",maxIter=20,returnObject=TRUE))
summary(model_Ventricles)

model_DiseaseState<- lme(fixed=DiseaseState~1, random=~AGE+AGE^2+AGE^3|RID ,data=mydata, control = lmeControl(opt='optim',optimMethod="L-BFGS-B",maxIter=20,returnObject=TRUE))
summary(model_DiseaseState)

# Prediction Matrix
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
  #rate <- predict(model_DiseaseState,PredictionMatrix[idx_pred,])
  #a <-mydata[idx1,'AGE']
  #PredictionMatrix[idx_pred,'DiseaseState']<- PredictionMatrix[idx_pred,'DiseaseState']+ rate*(PredictionMatrix[idx_pred,'AGE']-a[length(a)])
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

PredictionMatrix['Forecast Date'] = ForecastDate
# Predict, Predict Confidence Interval -> ADAS13

#intervals(model_ADAS,level = 0.50)

apred = predict(model_ADAS,PredictionMatrix)
PredictionMatrix['ADAS13']=exp(apred[1:length(apred)])-1

# Predict, Predict Confidence Interval -> Ventricles
vpred <- predict(model_Ventricles,PredictionMatrix)
PredictionMatrix['Ventricles_ICV']=vpred[1:length(vpred)]
PredictionMatrix['Ventricles_ICV'] = PredictionMatrix['Ventricles_ICV']/PredictionMatrix['ICV_bl']

PredictionMatrix <- subset(PredictionMatrix, select = -AGE )
PredictionMatrix <- subset(PredictionMatrix, select = -PTGENDER )
PredictionMatrix <- subset(PredictionMatrix, select = -PTEDUCAT )
PredictionMatrix <- subset(PredictionMatrix, select = -DiseaseState )
PredictionMatrix <- subset(PredictionMatrix, select = -ICV_bl )

write.csv(PredictionMatrix,str_out,row.names=FALSE)
