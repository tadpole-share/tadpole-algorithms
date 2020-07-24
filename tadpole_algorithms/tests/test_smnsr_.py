from tadpole_algorithms.models import BenchmarkSMNSR
from smnsr.patients import TADPOLEData

def test_train():
    data = TADPOLEData(modality_path=None,modality_k=8,challenge_filter=True)
    model = BenchmarkSMNSR(mode="bypass_knnsr",training_cv_folds=2)
    model.train(data.df_raw)
    assert model.adas_smnsr.is_fitted()

def test_predict():
    data = TADPOLEData(modality_path=None,modality_k=8,challenge_filter=True)
    model = BenchmarkSMNSR(modality_k=2,training_cv_folds=2)
    model.train(data.df_raw)
    prediction = model.predict(data.df_raw)
    assert prediction.shape[0] > 0

