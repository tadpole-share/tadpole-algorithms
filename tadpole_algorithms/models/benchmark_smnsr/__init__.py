import pandas as pd
import numpy as np
from tadpole_algorithms.models.tadpole_model import TadpoleModel
from datetime import datetime
from dateutil.relativedelta import relativedelta
from smnsr.models import SMNSR, Y_HAT
from smnsr.patients import TADPOLEData, AugmentedTADPOLEData
from smnsr.patients.timeseries_creation import create_features


class BenchmarkSMNSR(TadpoleModel):
    """Wrapper for SMNSR for replication of TADPOLE-Challenge results

    The default parameter settings replicate the configuration used in the challenge.

    Args:
        modality_k (int, optional): The maximum size of modality k-combinations.
            All combinations up to k will be included.
        challenge_modalities (bool, optional): Description of `param2`. Activates inclusion of k-combinations which
            mimick the combinations utilized in the challenge
        mode (str, optional): Prediction mode. Use 'xgb' for gradient boosting, 'bypass_knnsr' to use only the lower level
            KNNSR regression, or 'linear' for simple linear regression using baseline and time.
        max_modalities (int, optional): The number of top scoring modalities to include. If more than 1, the average
            of the best performing modalities will be utilized.
        training_cv_folds (int, optional): The number of cross-validation folds during training.
        forecast_min (str, optional): The method for capping the forecast to a minimum. Use "baseline" for
            predicting no less than the baseline value, "future" for the no less than the lowest value in
            KNNSR neighbourhood, and None for no capping.
        pretrained (bool, optional): Utilize pretrained KNNSR data.
        tmp_dir (str, optional): Specify the temporary directory for download of pre-trained models from Google Drive
        verbosity (int, optinal): Console output verbosity. Set to 2 for maximum verbosity.
        n_cpus (int,optional): Limit the number of CPUs utilized in training. Set to None for all available CPUs
    """

    TMP_FILE = "smnsr_tmp.p"
    PRE_CALCULATED_KNNSR = (
        "https://drive.google.com/uc?id=18l4FBWEU0gNotvFnxvhk6_mEAdxPhpxT"
    )

    def __init__(
        self,
        modality_k: int = 8,
        challenge_modalities: bool = True,
        mode: str = "xgb",
        max_modalities: int = 1,
        training_cv_folds: int = 5,
        forecast_min: str = "baseline",
        pretrained: bool = True,
        tmp_dir: str = None,
        verbosity: int = 1,
        n_cpus: int = None,
    ):

        self._modality_k = modality_k
        self._challenge_modalities = challenge_modalities
        self._mode = mode
        self.max_modalities = max_modalities
        self._training_cv_folds = training_cv_folds
        self._forecast_min = forecast_min
        self._pretrained = pretrained
        self._verbosity = verbosity
        self._n_cpus = n_cpus
        self._tmp_dir = tmp_dir

    def train(self, train_df: pd.DataFrame):

        data = TADPOLEData(
            data=train_df,
            modality_k=self._modality_k,
            challenge_filter=self._challenge_modalities,
        )

        if self._pretrained:
            ts_features = self.PRE_CALCULATED_KNNSR
        else:
            ts_features = create_features(
                data, data.get_ptids(), [], num_cpus=self._n_cpus
            )

        augmented_data = AugmentedTADPOLEData(
            data, ts_features, data.get_ptids(), self._verbosity
        )

        self.adas_smnsr = SMNSR(
            augmented_data,
            training_cv_folds=self._training_cv_folds,
            verbosity=2,
            mode=self._mode,
            forecast_min=self._forecast_min,
            max_modalities=self.max_modalities,
            forecast=True,
        )
        if self._verbosity > 0:
            print("Fitting model")
        self.adas_smnsr.fit(data.get_ptids())
        if self._verbosity > 0:
            print("Model fitted")

    def predict(self, test_df):
        """Performs forecast from 2018-01-01 to 2022-12-01 for every PTID, starting from the last available measurement.
        Args:
            test_df (DataFrame): Any DataFrame with PTID column. The matching last time-points will be retrieved from
            the TADPOLE_D1_D2.csv automatically.
        """
        if self._verbosity == 2:
            print("Prediction started")
        predictions = self.adas_smnsr.predict(
            test_df,
            target="ADAS13",
            forecast_start="2018-01-01",
            forecast_end="2022-12-01",
        )
        predictions = predictions.sort_values(TADPOLEData.RID)

        rdelta = relativedelta(
            datetime.strptime("2022-12-01", "%Y-%m-%d"),
            datetime.strptime("2018-01-01", "%Y-%m-%d"),
        )
        n_months = rdelta.years * 12 + rdelta.months

        # Count backwards from 2022-12-01
        predictions = predictions.groupby(TADPOLEData.RID).tail(n_months)

        n_rid = predictions[TADPOLEData.RID].unique().shape[0]
        months = []
        for i in range(0, n_rid):
            months += np.arange(1, n_months + 1).astype(int).tolist()

        def add_months_to_str_date(months, strdate):
            return (
                datetime.strptime(strdate, "%Y-%m-%d") + relativedelta(months=months)
            ).strftime("%Y-%m-%d")

        exam_dates = list(
            map(lambda x: add_months_to_str_date(x, "2018-01-01"), months)
        )
        df = pd.DataFrame.from_dict(
            {
                "RID": predictions[TADPOLEData.RID],
                "month": months,
                "Forecast Date": exam_dates,
                "ADAS13": predictions[Y_HAT],
                "ADAS13 50% CI lower": predictions[Y_HAT]
                - 0,  # CI estimation is currently missing
                "ADAS13 50% CI upper": predictions[Y_HAT]
                + 0,  # CI estimation is currently missing
            }
        )
        df["CN relative probability"] = np.nan
        df["MCI relative probability"] = np.nan
        df["AD relative probability"] = np.nan
        df["Ventricles_ICV 50% CI lower"] = np.nan
        df["Ventricles_ICV"] = np.nan
        df["Ventricles_ICV 50% CI upper"] = np.nan
        return df
