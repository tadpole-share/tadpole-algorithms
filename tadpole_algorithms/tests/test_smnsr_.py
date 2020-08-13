from tadpole_algorithms.models import BenchmarkSMNSR
from smnsr.patients import TADPOLEData
import os

TADPOLE_DATA = "data/TADPOLE_D1_D2.csv"

assert os.path.exists(TADPOLE_DATA), "TADPOLE_D1_D2.csv must be provided for testing"


def test_train():
    data = TADPOLEData(
        data=TADPOLE_DATA, modality_path=None, modality_k=8, challenge_filter=True
    )
    model = BenchmarkSMNSR(mode="bypass_knnsr", training_cv_folds=2)
    model.train(data.df_raw)
    assert model.adas_smnsr.is_fitted()


def test_predict():
    data = TADPOLEData(
        data=TADPOLE_DATA, modality_path=None, modality_k=8, challenge_filter=True
    )
    model = BenchmarkSMNSR(modality_k=8, training_cv_folds=2, mode="bypass_knnsr")
    model.train(data.df_raw)
    prediction = model.predict(data.df_raw)
    assert prediction.shape[0] > 0
