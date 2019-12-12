import logging

from pathlib import Path

import pandas as pd
import scipy as sp
from scipy import stats
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from tqdm.auto import tqdm

from tadpole.models.tadpole_model import TadpoleModel


logger = logging.getLogger(__name__)


def bootstrap(model, train_df, y_df, test_df, n_bootstraps: int = 100, confidence=0.50) -> float:
    """Runs model `model` using different random sampled train & test splits.

    Returns:
        float: Confidence Interval delta for `confidence` level.
    """
    predictions = []
    for i in tqdm(range(0, n_bootstraps)):
        train_df_resampled, y_df_resampled = resample(train_df, y_df, random_state=i)
        model.fit(train_df_resampled, y_df_resampled)
        prediction = model.predict(test_df)[0]
        predictions.append(prediction)

    se = sp.stats.sem(predictions)  # Standard error
    h = se * sp.stats.t._ppf((1 + confidence) / 2., len(y_df))  # CI

    # m-h and m+h give confidence interval
    return h


class ECMEB(TadpoleModel):
    """ECM-EB method, Esther Bron - e.bron@erasmusmc.nl

    The `train_df*` attributes contain training data optimized for each variable.

    The `y_train_df*` attributes contain the labels to be used for training by each model,
    thus corresponding to the matching `train_df` DataFrame.

    Attributes:
        diagnosis_model (Pipeline): Model for predicting 'diagnosis' variable
        adas_model (Pipeline): Model for predicting 'ADAS13' variable
        ventricles_model (Pipeline): Model for predicting 'ventricles' variable

        y_diagnosis (pandas.DataFrame): 'Diagnosis' labels
        train_df_diagnosis (pandas.DataFrame): Training data used for 'diagnosis' model.
    """


    def __init__(self):
        self.diagnosis_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVC(kernel='rbf', C=0.5, gamma='auto', class_weight='balanced', probability=True)),
        ])
        self.adas_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVR(kernel='rbf', C=0.5, gamma='auto')),
        ])
        self.ventricles_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVR(kernel='rbf', C=0.5, gamma='auto')),
        ])

        self.y_diagnosis = None
        self.y_adas = None
        self.y_ventricles = None

        self.train_df_diagnosis = None
        self.train_df_adas = None
        self.train_df_ventricles = None

    @staticmethod
    def preprocess(train_df: pd.DataFrame):
        logger.info("Pre-processing")
        train_df = train_df.copy()
        if 'Diagnosis' not in train_df.columns:
            """We want to transform 'DXCHANGE' (a change in diagnosis, in contrast
            to the previous visits diagnosis) to an actual diagnosis."""
            train_df = train_df.replace({'DXCHANGE': {4: 2, 5: 3, 6: 3, 7: 1, 8: 2, 9: 1}})
            train_df = train_df.rename(columns={"DXCHANGE": "Diagnosis"})

        # Adds months to age
        train_df['AGE'] += train_df['Month_bl'] / 12.

        # Drop columns found unimportant by feature importance ranking measure.
        h = list(train_df)
        train_df: pd.DataFrame = train_df.drop(
            h[1:8] + [h[9]] + h[14:17] + h[45:47] + h[53:73] + h[74:486] + h[832:838] + h[1172:1174] + \
            h[1657:1667] + h[1895:1902] + h[1905:],
            axis=1
        )

        h = list(train_df)

        logger.info('Forcing Numeric Values')
        for i in range(5, len(h)):
            if train_df[h[i]].dtype != 'float64':
                train_df[h[i]] = pd.to_numeric(train_df[h[i]], errors='coerce')

        """Sort the DataFrame per patient on age (at time of visit). This allows using observations from
        the next row/visit to be used as a label for the previous row. (See `get_futures` method.)"""
        train_df = train_df.sort_values(by=['RID', 'AGE'])

        train_df = train_df.drop(['EXAMDATE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4'], axis=1)

        # Ventricles_ICV = Ventricles/ICV_bl. So make sure ICV_bl is not zero to avoid division by zero
        icv_bl_median = train_df['ICV_bl'].median()
        train_df.loc[train_df['ICV_bl'] == 0, 'ICV_bl'] = icv_bl_median

        if 'Ventricles_ICV' not in train_df.columns:
            train_df["Ventricles_ICV"] = train_df["Ventricles"].values / train_df["ICV_bl"].values

        """Select features based on ECMEB_features.csv file"""
        selected_features = pd.read_csv(Path(__file__).parent / 'ECMEB_features.csv')['feature'].values.tolist()
        selected_features = selected_features[0:200]
        selected_features += ['RID', 'Diagnosis', 'Ventricles_ICV']
        selected_features = set(selected_features)
        train_df = train_df.copy()[selected_features]

        train_df = ECMEB.fill_nans_by_older_values(train_df)

        return train_df

    @staticmethod
    def get_futures(train_df, features=['RID', 'Diagnosis', 'ADAS13', 'Ventricles_ICV']):
        """For each feature in `features` argument, generate a `Future_{feature}` column, that is filled
        using the next row for each patient"""

        futures_df = train_df[features].copy()

        # Get future value from each row's next row, e.g. shift the column one up
        for predictor in ["Diagnosis", "ADAS13", 'Ventricles_ICV']:
            futures_df["Future_" + predictor] = futures_df[predictor].shift(-1)

        # Drop each last row per patient
        futures_df = futures_df.drop(futures_df.groupby('RID').tail(1).index.values)
        return futures_df

    @staticmethod
    def fill_nans_by_older_values(train_df):
        """Fill nans in feature matrix by older values (ffill), then by newer (bfill)"""

        df_filled_nans = train_df.groupby('RID').fillna(method='ffill').fillna(method='bfill')
        train_df[df_filled_nans.columns] = df_filled_nans
        return train_df

    def train(self, train_set_path):
        train_df = pd.read_csv(train_set_path)
        train_df = self.preprocess(train_df)
        futures = self.get_futures(train_df)

        # Not part of `preprocess` because it's needed for the futures.
        train_df = train_df.drop(['RID'], axis=1)

        # Fill left over nans with mean
        train_df = train_df.fillna(train_df.mean())
        train_df = train_df.fillna(0)

        def non_nan_y(_train_df, _y_df):
            """Drops all rows with a `y` value that is NaN

            Returns:
                Tuple containing (`train_df`, `y_df`), without NaNs for `y_df`.
            """

            # indices where the y value is not nan
            not_nan_idx = _y_df[_y_df.notna()].index

            # return from both the train dataframe and y the records with these indices
            return _train_df.loc[not_nan_idx], _y_df[not_nan_idx]

        self.train_df_diagnosis, self.y_diagnosis = non_nan_y(train_df, futures['Future_Diagnosis'])
        self.train_df_adas, self.y_adas = non_nan_y(train_df, futures['Future_ADAS13'])
        self.train_df_ventricles, self.y_ventricles = non_nan_y(train_df, futures['Future_Ventricles_ICV'])

        logger.info("Training models")
        self.diagnosis_model.fit(self.train_df_diagnosis, self.y_diagnosis)
        self.adas_model.fit(self.train_df_adas, self.y_adas)
        self.ventricles_model.fit(self.train_df_ventricles, self.y_ventricles)

    def predict(self, test_df):
        logger.info("Predicting")
        # test_df = self.preprocess(test_series.to_frame().T)

        # select last row per RID
        test_df = test_df.sort_values(by=['EXAMDATE'])
        test_df = test_df.groupby('RID').tail(1)

        test_df = self.preprocess(test_df)
        rids = test_df['RID']
        test_df = test_df.drop(['RID'], axis=1)
        test_df = test_df.fillna(0)

        diag_probas = self.diagnosis_model.predict_proba(test_df)
        adas_prediction = self.adas_model.predict(test_df)

        logger.info("Bootstrap adas")
        adas_ci = bootstrap(
            self.adas_model,
            self.train_df_adas,
            self.y_adas,
            test_df
        )

        ventricles_prediction = self.adas_model.predict(test_df)
        logger.info("Bootstrap ventricles")
        ventricles_ci = bootstrap(
            self.ventricles_model,
            self.train_df_ventricles,
            self.y_ventricles,
            test_df
        )


        df = pd.DataFrame.from_dict({
            'rids': rids,
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
        for i in range(0, 12 * 4):
            df_copy['month'] = i
            new_df = new_df.append(df_copy)

        return new_df
