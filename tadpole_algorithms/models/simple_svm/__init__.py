import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import logging
logger = logging.getLogger(__name__)


class SimpleSVM:
    def __init__(self):
        self.diagnosis_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVC(kernel="linear", probability=True)),
        ])
        self.adas_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVR(kernel="linear")),
        ])
        self.ventricles_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVR(kernel="linear")),
        ])

    def train_model(self, model, train_df, X_train, var_name):
        # remove rows with NaN future values
        Y_train_var = train_df[var_name]
        not_nans = np.logical_not(np.isnan(Y_train_var))
        X_train_var = X_train[not_nans]
        Y_train_var = Y_train_var[not_nans]
        model.fit(X_train_var, Y_train_var)

    def preprocess(self, train_df):
        logger.info("Pre-processing")
        train_df = train_df.copy()
        if 'Diagnosis' not in train_df.columns:
            train_df = train_df.replace({'DXCHANGE': {4: 2, 5: 3, 6: 3, 7: 1, 8: 2, 9: 1}})
            train_df = train_df.rename(columns={"DXCHANGE": "Diagnosis"})

        # Sort the dataframe based on age for each subject
        train_df = train_df.sort_values(by=['RID', 'Years_bl'])

        # Ventricles_ICV = Ventricles/ICV_bl. So make sure ICV_bl is not zero to avoid division by zero
        icv_bl_median = train_df['ICV_bl'].median()
        train_df.loc[train_df['ICV_bl'] == 0, 'ICV_bl'] = icv_bl_median

        if 'Ventricles_ICV' not in train_df.columns:
            train_df["Ventricles_ICV"] = train_df["Ventricles"].values / train_df["ICV_bl"].values

        # Select features
        train_df = train_df[
            ["RID", "Diagnosis", "ADAS13", "Ventricles_ICV", "Ventricles", "ICV_bl"]
        ]

        # Force values to numeric
        train_df = train_df.astype("float64", errors='ignore')

        return train_df

    def set_futures(self, train_df):
        # Get future value from each row's next row, e.g. shift the column one up
        for predictor in ["Diagnosis", "ADAS13", 'Ventricles_ICV']:
            train_df["Future_" + predictor] = train_df[predictor].shift(-1)

        # Drop each last row per patient
        train_df = train_df.drop(train_df.groupby('RID').tail(1).index.values)
        return train_df

    def train(self, train_set_path):
        train_df = pd.read_csv(train_set_path)
        train_df = self.preprocess(train_df)
        train_df = self.set_futures(train_df)

        # Select columns for training
        X_train = train_df[["Diagnosis", "ADAS13", "Ventricles_ICV"]]

        # fill NaNs with mean
        X_train = X_train.fillna(X_train.mean())

        logger.info("Training models")
        self.train_model(self.diagnosis_model, train_df, X_train, "Future_Diagnosis")
        self.train_model(self.adas_model, train_df, X_train, "Future_ADAS13")
        self.train_model(self.ventricles_model, train_df, X_train, "Future_Ventricles_ICV")

    def predict(self, test_df):
        logger.info("Predicting")

        # select last row per RID
        test_df = test_df.sort_values(by=['EXAMDATE'])
        test_df = test_df.groupby('RID').tail(1)

        test_df = self.preprocess(test_df)
        rids = test_df['RID']
        test_df = test_df.drop(['RID'], axis=1)

        # Select same columns as for traning for testing
        test_df = test_df[["Diagnosis", "ADAS13", "Ventricles_ICV"]]

        test_df = test_df.fillna(0)

        diag_probas = self.diagnosis_model.predict_proba(test_df)
        adas_prediction = self.adas_model.predict(test_df)

        adas_ci = np.zeros(len(adas_prediction))

        ventricles_prediction = self.adas_model.predict(test_df)
        ventricles_ci = np.zeros(len(ventricles_prediction))

        df = pd.DataFrame.from_dict({
            'rid': rids,
            'CN relative probability': diag_probas.T[0],
            'MCI relative probability': diag_probas.T[1],
            'AD relative probability': diag_probas.T[2],

            'ADAS13': adas_prediction,
            'ADAS13 50% CI lower': adas_prediction - adas_ci,
            'ADAS13 50% CI upper': adas_prediction + adas_ci,

            'Ventricles_ICV': ventricles_prediction,
            'Ventricles_ICV 50% CI lower': ventricles_prediction - ventricles_ci,
            'Ventricles_ICV 50% CI upper': ventricles_prediction + ventricles_ci,
        })

        # copy each row for each month
        df_copy = df.copy()
        new_df = df
        for i in range(0, 12*4):
            df_copy['month'] = i
            new_df = new_df.append(df_copy)

        return new_df
