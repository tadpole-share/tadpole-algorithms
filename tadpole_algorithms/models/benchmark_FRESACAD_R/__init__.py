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
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
#to transform r to python df
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

class Benchmark_FRESACAD_R(TadpoleModel):



    def preproc_tadpole_D1_D2(self,Tadpole_D1_D2,usePreProc=False):
        #using the usePreProc flag you can select between use the preprocess model results or not
        if usePreProc == False:
            preproc_tadpole_D1_D2_RSCRIPT = ""
            Tadpole_D1_D2.to_csv("data/train_df.csv")        
            with open('R_scripts/preproc_tadpole_D1_D2.r', 'r') as file: 
                preproc_tadpole_D1_D2_RSCRIPT = file.read()
            #replace the values on the script with the actual atributes needed (its like pasing arguments in a function)   
            #preproc_tadpole_D1_D2_RSCRIPT = preproc_tadpole_D1_D2_RSCRIPT.replace("_preTrain_",str(usePreProc))
            preproc_tadpole_D1_D2_RFUNC = robjects.r(preproc_tadpole_D1_D2_RSCRIPT)
            #load the result of preprocesing of Tadpole_d1_d2
        
        AdjustedTrainFrame = pd.read_csv("data/dataTadpole$AdjustedTrainFrame.csv")
        testingFrame = pd.read_csv("data/dataTadpole$testingFrame.csv")
        Train_Imputed = pd.read_csv("data/dataTadpole$Train_Imputed.csv")
        Test_Imputed = pd.read_csv("data/dataTadpole$Test_Imputed.csv")

        return AdjustedTrainFrame,testingFrame,Train_Imputed,Test_Imputed

    def preproc_tadpole_D3(self,Tadpole_D3,usePreProc=False):
        #using the usePreProc flag you can select between use the preprocess model results or not
        if usePreProc == False :
            preproc_tadpole_D3_RSCRIPT = ""
            Tadpole_D3.to_csv("data/TADPOLE_D3.csv")        
            with open('R_scripts/preproc_tadpole_D3.r', 'r') as file: 
                preproc_tadpole_D3_RSCRIPT = file.read()
            #replace the values on the script with the actual atributes needed (its like pasing arguments in a function)   
            #preproc_tadpole_D3_RSCRIPT = preproc_tadpole_D3_RSCRIPT.replace("_preTrain_",str(usePreProc))
            preproc_tadpole_D3_RFUNC = robjects.r(preproc_tadpole_D3_RSCRIPT)
            #load the result of preprocesing of Tadpole_D3

        AdjustedTrainFrame = pd.read_csv("data/dataTadpoleD3$AdjustedTrainFrame.csv")
        testingFrame = pd.read_csv("data/dataTadpoleD3$testingFrame.csv")
        Train_Imputed = pd.read_csv("data/dataTadpoleD3$Train_Imputed.csv")
        Test_Imputed = pd.read_csv("data/dataTadpoleD3$Test_Imputed.csv")

        return AdjustedTrainFrame,testingFrame,Train_Imputed,Test_Imputed



    def Forecast_D2(self,AdjustedTrainFrame,testingFrame,Train_Imputed,Test_Imputed,usePreProc=False):
        if usePreProc == False :
            Forecast_D2_RSCRIPT = ""
            #Tadpole_D1_D2.to_csv("data/temp/train_df.csv")        
            with open('R_scripts/Forecast_D2.r', 'r') as file: 
                Forecast_D2_RSCRIPT = file.read()
            Forecast_D2_RFUNC = robjects.r(Forecast_D2_RSCRIPT)
        ForecastD2_BORREGOS_TEC = pd.read_csv("data/_ForecastD2_BORREGOS_TEC.csv")
        return ForecastD2_BORREGOS_TEC

    def Forecast_D3(self,AdjustedTrainFrame,testingFrame,Train_Imputed,Test_Imputed,usePreProc=False):
        if usePreProc == False :
            Forecast_D3_RSCRIPT = ""
            with open('R_scripts/Forecast_D3.r', 'r') as file: 
                Forecast_D3_RSCRIPT = file.read()       
            Forecast_D3_RFUNC = robjects.r(Forecast_D3_RSCRIPT)
        ForecastD3_BORREGOS_TEC = pd.read_csv("data/_ForecastD3_BORREGOS_TEC.csv")
        return ForecastD3_BORREGOS_TEC



#end R functions
    def preprocess(self):
        logger.info("use preproc_tadpole_D1_D2 or preproc_tadpole_D3")


    def train(self, train_df):
        logger.info("Training phase inside R")

    def predict(self):
        logger.info("Prediction phase is inside R")

        
    


