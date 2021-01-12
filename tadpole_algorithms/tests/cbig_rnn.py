import pandas as pd
import datetime
from pathlib import Path
from tadpole_algorithms.models import CBIG_RNN
from tadpole_algorithms.preprocessing.split import split_test_train_tadpole

"""
Train model on ADNI data set D1
Test model on ADNI data set D2
"""

# Load D1_D2 train and possible test data set
data_path_train_test = Path("data/TADPOLE_D1_D2.csv")
data_df_train_test = pd.read_csv(data_path_train_test)

# Load D4 evaluation data set
data_path_eval = Path("data/TADPOLE_D4_corr.csv")
data_df_eval = pd.read_csv(data_path_eval)

# Split data in test, train and evaluation data
train_df, test_df, eval_df = split_test_train_tadpole(data_df_train_test, data_df_eval)
test_df = test_df.fillna(0)

# Define and train model
model = CBIG_RNN(cleanup=True)
model.train(train_df)

# # Predict forecast on the test set
forecast_df_d2 = model.predict(test_df)
from tadpole_algorithms.evaluation import evaluate_forecast
from tadpole_algorithms.evaluation import print_metrics

# Evaluate the model 
dictionary = evaluate_forecast(eval_df, forecast_df_d2)

# Print metrics
print_metrics(dictionary)

data_path_forecast_cbig_rnn = Path("Outputs/cbig_rnn/forecast_df_d2.csv")

forecast_df_d2.to_csv(data_path_forecast_cbig_rnn)