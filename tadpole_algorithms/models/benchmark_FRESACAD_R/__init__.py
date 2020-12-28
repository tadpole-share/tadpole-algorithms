# ============
# Authors:
#   Jose Tamez Peña
#   Eider Diaz
#   Rebeca Canales

# Script requires that TADPOLE_D1_D2.csv is in the parent directory. Change if
# necessary

import pandas as pd
import numpy as np
import os
import sys

from tadpole_algorithms.models.tadpole_model import TadpoleModel

import datetime as dt
from dateutil.relativedelta import relativedelta

import logging

from datetime import datetime
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)
#this import is for use R code into Python
from rpy2 import robjects
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
#to transform r to python df
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from pathlib import Path

class Benchmark_FRESACAD_R(TadpoleModel):

    def extractTrainTestDataSets_R(self,
                                    D1D2DataFileName,
                                    D3DataFilneName):
        logger.info("Extract Training and Testing sets")
        extracTrainTest_RSCRIPT = ""
        with open('R_scripts/ExtractTrainTest_tadpole_D1_D2.r', 'r') as file: 
            extracTrainTest_RSCRIPT = file.read()
        sourcePreprocess = robjects.r(extracTrainTest_RSCRIPT)
        extractTrainTest_RFUNC = robjects.globalenv['ExtractTrainTest_tadpole_D1_D2']
        outR = extractTrainTest_RFUNC(D1D2DataFileName,D3DataFilneName)
        D1Train = pd.read_csv("data/_tmp_D1TrainingSet.csv")
        D2Test = pd.read_csv("data/_tmp_D2TesingSet.csv")
        D3Train = pd.read_csv("data/_tmp_D3TrainingSet.csv")
        D3Test = pd.read_csv("data/_tmp_D3TesingSet.csv")
        del D1Train["Unnamed: 0"]
        del D2Test["Unnamed: 0"]
        del D3Train["Unnamed: 0"]
        del D3Test["Unnamed: 0"]

## The next code failed to convert Py Pandas to R Data.frame
#         with localconverter(ro.default_converter + pandas2ri.converter):
#            D1D2DataFileName_R = ro.conversion.py2rpy(D1D2DataFileName)
#        with localconverter(ro.default_converter + pandas2ri.converter):
#            D3DataFilneName_R = ro.conversion.py2rpy(D3DataFilneName)
#        D1Train,D2Test,D3Train,D3Test = extractTrainTest_RFUNC(D1D2DataFileName_R,D3DataFilneName_R)
#        with localconverter(ro.default_converter + pandas2ri.converter):
#            D1Train_df = ro.conversion.rpy2py(D1Train)
#        with localconverter(ro.default_converter + pandas2ri.converter):
#            D2Test_df = ro.conversion.rpy2py(D2Test)
#        with localconverter(ro.default_converter + pandas2ri.converter):
#            D3Train_df = ro.conversion.rpy2py(D3Train)
#        with localconverter(ro.default_converter + pandas2ri.converter):
#            D3Test_df = ro.conversion.rpy2py(D3Test)
#        return D1Train_df,D2Test_df,D3Train_df,D3Test_df

        return D1Train,D2Test,D3Train,D3Test


    def preproc_with_R(self,
                        TrainMatrix,
                        TestMatrix,
                        Dictionary,
                        MinVisit=36,
                        colImputeThreshold=0.25,
                        rowImputeThreshold=0.10,
                        includeID=True,
                        usePreProc=False):
        #using the usePreProc flag you can select between use the preprocesed data 
        logger.info("Prepocess Data Frames")
        if usePreProc == False:
            dataTADPOLEPreprocesingPy_RSCRIPT = ""
            with open('R_scripts/dataTADPOLEPreprocesingPy.r', 'r') as file: 
                dataTADPOLEPreprocesingPy_RSCRIPT = file.read()
            #replace the values on the script with the actual atributes needed (its like pasing arguments in a function)   
            sourcePreprocess = robjects.r(dataTADPOLEPreprocesingPy_RSCRIPT)
            preproc_tadpole_D1_D2_RFUNC = robjects.globalenv['dataTADPOLEPreprocesingPy']
#            print(preproc_tadpole_D1_D2_RFUNC.r_repr())
            TrainMatrix.to_csv("data/_tmp_TrainMatrix.csv")        
            TestMatrix.to_csv("data/_tmp_TestMatrix.csv") 
            Dictionary.to_csv("data/_tmp_Dictionary.csv")
            outResult = preproc_tadpole_D1_D2_RFUNC("data/_tmp_TrainMatrix.csv",
                                                    "data/_tmp_TestMatrix.csv",
                                                    "data/_tmp_Dictionary.csv",
                                                    MinVisit,
                                                    colImputeThreshold,
                                                    rowImputeThreshold,
                                                    includeID)
        
        AdjustedTrainFrame = pd.read_csv("data/_tmp_dataTadpole$AdjustedTrainFrame.csv")
        testingFrame = pd.read_csv("data/_tmp_dataTadpole$testingFrame.csv")
        Train_Imputed = pd.read_csv("data/_tmp_dataTadpole$Train_Imputed.csv")
        Test_Imputed = pd.read_csv("data/_tmp_dataTadpole$Test_Imputed.csv")
        del AdjustedTrainFrame["Unnamed: 0"]
        del testingFrame["Unnamed: 0"]
        del Test_Imputed["Unnamed: 0"]
        del Train_Imputed["Unnamed: 0"]


        return AdjustedTrainFrame,testingFrame,Train_Imputed,Test_Imputed


    def Train_Congitive(self,AdjustedTrainFrame,
                        numberOfRandomSamples=25,
                        delta=True,
                        usePreProc=False):
        logger.info("Train Cognitive Models")
        CognitiveModelsName =  "data/_CognitiveClassModels_25.RDATA.RDATA"
        if usePreProc == False :
            AdjustedTrainFileName = "data/_tmp_AdjustedTrainFrame.csv"
            AdjustedTrainFrame.to_csv(AdjustedTrainFileName)
            TrainCongitive_RFunction = ""
            with open('R_scripts/TrainCognitiveModels.r', 'r') as file: 
                TrainCongitive_RFunction = file.read()
            sourceTrain = robjects.r(TrainCongitive_RFunction)
            ContivieTrain_RFUNC = robjects.globalenv['TrainCognitiveModels']
            CognitiveModelsName = ContivieTrain_RFUNC(AdjustedTrainFileName,
                                                      numberOfRandomSamples,
                                                      delta=delta)
            
        return CognitiveModelsName

    def Train_Regression(self,AdjustedTrainFrame,
                        ImputedTrainFrame,
                        numberOfRandomSamples=50,
                        usePreProc=False):
        logger.info("Train ADAS13 and Ventricles Models")
        RegressionModelsName =  "data/_RegressionModels_50_Nolog.RDATA"
        if usePreProc == False :
            AdjustedTrainFileName = "data/_tmp_AdjustedTrainFrame.csv"
            AdjustedTrainFrame.to_csv(AdjustedTrainFileName)
            ImputedTrainFileName = "data/_tmp_ImputedTrainFrame.csv"
            ImputedTrainFrame.to_csv(ImputedTrainFileName)
            TrainRegression_RFunction = ""
            with open('R_scripts/TrainRegressionModels.r', 'r') as file: 
                TrainRegression_RFunction = file.read()
            sourceTrain = robjects.r(TrainRegression_RFunction)
            RegressionTrain_RFUNC = robjects.globalenv['TrainRegressionModels']
            RegressionModelsName = RegressionTrain_RFUNC(AdjustedTrainFileName,
                                                         ImputedTrainFileName,
                                                         numberOfRandomSamples)
            
        return RegressionModelsName

    def Forecast_All(self,
                        CognitiveModelsFileName,
                        RegressionModelsFileName,
                        AdjustedTestingFrame,
                        ImputedTestingFrame,
                        submissionTemplateFileName,
                        usePreProc=False):
        logger.info("Forecast Congitive Status, ADAS13 and Ventricles")
        forecastFilename = "data/_ForecastFRESACAD.csv"
        if usePreProc == False :
            AdjustedTestFileName = "data/_tmp_AdjustedTestFrame.csv"
            AdjustedTestingFrame.to_csv(AdjustedTestFileName)
            ImputedTestFileName = "data/_tmp_ImputedTestFrame.csv"
            ImputedTestingFrame.to_csv(ImputedTestFileName)
            Forecast_RSCRIPT = ""
            #Tadpole_D1_D2.to_csv("data/temp/train_df.csv")        
            with open('R_scripts/ForecastAll.r', 'r') as file: 
                Forecast_RSCRIPT = file.read()
            Forecast_out = robjects.r(Forecast_RSCRIPT)
            Forecast_RFUNC = robjects.globalenv['ForecastAll']
            forecastFilename = Forecast_RFUNC(CognitiveModelsFileName,
                                    RegressionModelsFileName,
                                    AdjustedTestFileName,
                                    ImputedTestFileName,
                                    submissionTemplateFileName
                                    )
            forecastFilename = forecastFilename[0]                        
            print(forecastFilename)
            print(type(forecastFilename))
#            forecastFilename = str(forecastFilename)
        data_path = Path(forecastFilename)

        Forecast = pd.read_csv(data_path)
        return Forecast

    def train(self, train_df):
        logger.info("custumtrain")

    def predict(self, test_df):
        logger.info("Predicting")

#end R functions
        
    


