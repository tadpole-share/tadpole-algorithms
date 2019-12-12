library(nlme)
library(splines)

str_exp = getwd()
mydata = read.csv(paste(str_exp,"/IntermediateData/LongTADPOLE.csv",sep=""))

mydata$ICV_bl[is.na(mydata$ICV_bl)]=mean(mydata$ICV_bl,na.rm=TRUE)
mydata$PTEDUCAT[is.na(mydata$PTEDUCAT)]=mean(mydata$PTEDUCAT,na.rm=TRUE)

mydata_ctrl <- mydata[mydata$Diagnosis==1,]
cn=colnames(mydata)

for (i in 8:ncol(mydata))
{
  idx_4 <- mydata[,i]==-4
  idx_4[is.na(idx_4)]=FALSE
  mydata[idx_4,i]=NA
}

for (i in 8:ncol(mydata))
{
  print (i)
  data_featurei = subset(mydata_ctrl, select = c("RID","AGE","PTGENDER","PTEDUCAT","ICV_bl","APOE4"))
  data_featurei['Featurei'] <- mydata_ctrl[cn[i]]
  data_featurei <- data_featurei[complete.cases(data_featurei),]
  if (nrow(data_featurei)>0)
  {
    model_featurei <- lm(Featurei ~ (ns(AGE,2))+PTEDUCAT+PTGENDER+ICV_bl,data=data_featurei)
    data_Mean <- cbind(NA,NA)
    data_Mean <- cbind(mydata)
    data_Mean$AGE = mean(data_featurei$AGE)
    data_Mean$ICV_bl = mean(data_featurei$ICV_bl)
    data_Mean$PTEDUCAT = mean(data_featurei$PTEDUCAT)
    data_Mean$PTGENDER = "Male"                    
    mydata[cn[i]] <- mydata[cn[i]] - predict(model_featurei,mydata) + predict(model_featurei,data_Mean)
  }
}

write.csv(mydata,paste(str_exp,"/IntermediateData/AgeCorrectedLongTADPOLE.csv",sep=""),row.names=FALSE)

mySubjects = read.csv(paste(str_exp,"/IntermediateData/SubjectsWithChange.csv",sep=""),header=FALSE)
ZChange <- array(0, dim=ncol(mydata))
for (i in 8:ncol(mydata))
{
  print (i)
  
  Feat <- mydata[cn[i]]
  Feat <- Feat - mean(Feat[,1],na.rm=TRUE)
  Feat <- Feat / sd(Feat[,1],na.rm=TRUE)
  CF <- array(0,dim=c(0,1))
  for (j in 1:length(mySubjects$V1))
  {
    idx = mydata$RID == mySubjects$V1[j]
    Fj = Feat[idx,cn[i]]
    Fj = Fj[complete.cases(Fj)]
    if (length(Fj)>1)
    {
      CF<-c(CF,(Fj[length(Fj)]-Fj[1]))
    }
  }
  ZChange[i] <- abs(mean(CF))
}

idx_Feats = order(ZChange,decreasing=TRUE)

write.csv(idx_Feats-8,paste(str_exp,"/IntermediateData/FeatureIndices.csv",sep=""),row.names=FALSE)