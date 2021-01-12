import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append("../tadpole-algorithms")
import tadpole_algorithms
from tadpole_algorithms.models import Benchmark_FRESACAD_R
from tadpole_algorithms.preprocessing.split import split_test_train_tadpole
#rpy2 libs and funcs
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import r, pandas2ri 
from rpy2 import robjects
from rpy2.robjects.conversion import localconverter

import os
os.chdir('jupyter')

# Load D1_D2 train and possible test data set
data_path_train_test = Path("data/TADPOLE_D1_D2.csv")
data_df_train_test = pd.read_csv(data_path_train_test)

# Load data Dictionary
data_path_Dictionaty = Path("data/TADPOLE_D1_D2_Dict.csv")
data_Dictionaty = pd.read_csv(data_path_Dictionaty)

# Load D3 possible test set
data_path_test = Path("data/TADPOLE_D3.csv")
data_D3 = pd.read_csv(data_path_test)

# Load D4 evaluation data set 
data_path_eval = Path("data/TADPOLE_D4_corr.csv")
data_df_eval = pd.read_csv(data_path_eval)

# Split data in test, train and evaluation data
train_df, test_df, eval_df = split_test_train_tadpole(data_df_train_test, data_df_eval)

#instanciate the model to get the functions
model = Benchmark_FRESACAD_R()
#set the flag to true to use a preprocessed data
USE_PREPROC = False


#preprocess the data
D1Train,D2Test,D3Train,D3Test = model.extractTrainTestDataSets_R("data/TADPOLE_D1_D2.csv","data/TADPOLE_D3.csv")

# AdjustedTrainFrame,testingFrame,Train_Imputed,Test_Imputed = model.preproc_tadpole_D1_D2(data_df_train_test,USE_PREPROC)
AdjustedTrainFrame,testingFrame,Train_Imputed,Test_Imputed = model.preproc_with_R(D1Train,D2Test,data_Dictionaty,usePreProc=USE_PREPROC)

#Train Congitive Models
modelfilename = model.Train_Congitive(AdjustedTrainFrame,usePreProc=USE_PREPROC)

#Train ADAS/Ventricles Models
regresionModelfilename = model.Train_Regression(AdjustedTrainFrame,Train_Imputed,usePreProc=USE_PREPROC)
print(regresionModelfilename)

print(regresionModelfilename)
print(type(regresionModelfilename))

#Predict 
Forecast_D2 = model.Forecast_All(modelfilename,
                                 regresionModelfilename,
                                 testingFrame,
                                 Test_Imputed,
                                 submissionTemplateFileName="data/TADPOLE_Simple_Submission_TeamName.xlsx",
                                 usePreProc=USE_PREPROC)

#data_forecast_test = Path("data/_ForecastFRESACAD.csv")
#Forecast_D2 = pd.read_csv(data_forecast_test)

from tadpole_algorithms.evaluation import evaluate_forecast
from tadpole_algorithms.evaluation import print_metrics
# Evaluate the model 
dictionary = evaluate_forecast(eval_df, Forecast_D2)
# Print metrics
print_metrics(dictionary)

data_path_forecast_BORREGOSTECMTY = Path("Outputs/BORREGOSTECMTY/forecast_df_d2.csv")

Forecast_D2.to_csv(data_path_forecast_BORREGOSTECMTY)