import pandas as pd
import datetime
from pathlib import Path
from tadpole_algorithms.models import EMCEB
from tadpole_algorithms.preprocessing.split import split_test_train_tadpole
# Load D1_D2 evaluation data set
data_path_train_test = Path("jupyter/data/TADPOLE_D1_D2.csv")
data_df_train_test = pd.read_csv(data_path_train_test)

# Load D4 evaluation data set
data_path_eval = Path("jupyter/data/TADPOLE_D4_corr.csv")
data_df_eval = pd.read_csv(data_path_eval)

# Split data in test, train and evaluation data
train_df, test_df, eval_df = split_test_train_tadpole(data_df_train_test, data_df_eval)

# Indicate what data set is the training and testing dataset
train = "d1d2"
test = "d1d2"

# Define model
model = EMCEB()

# Preprocess and set data 
model.set_data(train_df, test_df, train, test)

# Train model
# Note to self: number of bootstraps set to 1 for computation speed. Should be 100 to compute CIs.
model.train()

# Predict forecast on the test set
forecast_df_d2 = model.predict()

from tadpole_algorithms.evaluation import evaluate_forecast
from tadpole_algorithms.evaluation import print_metrics

# Evaluate the model 
dictionary = evaluate_forecast(eval_df, forecast_df_d2)

# Print metrics
print_metrics(dictionary)

data_path_forecast_emceb = Path("jupyter/Outputs/emceb/forecast_df_d2.csv")

forecast_df_d2.to_csv(data_path_forecast_emceb)