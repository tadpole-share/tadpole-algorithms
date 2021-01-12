#Train model on data set D1-D2 and predict for D4
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from tadpole_algorithms.models.emc1 import train_and_predict
from tadpole_algorithms.preprocessing.split import split_test_train_tadpole

import os
os.chdir('tadpole_algorithms/tests')

"""
Train model on ADNI data set D1 / D2
Predict for subjects in the data set D2
"""

# Load D1_D2 train and possible test data set
data_path_train_test = Path("data/TADPOLE_D1_D2.csv")
data_df_train_test = pd.read_csv(data_path_train_test,low_memory=False)
idx_progress=np.logical_and(data_df_train_test['DXCHANGE']>=4, data_df_train_test['DXCHANGE']<=6)
SubC=np.unique(data_df_train_test.loc[idx_progress,'RID'])
SubC = pd.Series(SubC);
# Load D4 evaluation data set 
data_path_eval = Path("data/TADPOLE_D4_corr.csv")
data_df_eval = pd.read_csv(data_path_eval)

train_df, test_df, eval_df = split_test_train_tadpole(data_df_train_test, data_df_eval)

n_boot = 0
IntermediateFolder = 'data/EMC1_IntermediateData'
forecast_df_d2=train_and_predict.main(train_df, test_df,IntermediateFolder,n_boot)

data_path_forecast_emc1 = Path("Outputs/emc1/forecast_df_d2.csv")

os.chdir('..')
os.chdir('..')
os.chdir('tests')
forecast_df_d2.to_csv(data_path_forecast_emc1)