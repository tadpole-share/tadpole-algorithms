from tadpole.models.tadpole_model import TadpoleModel


class ImaUCL(TadpoleModel):
    def __init__(self):
        pass

    def train(self, train_set_path):
        raise NotImplementedError('not implemented')

    def predict(self, test_series, predict_datetime):
        raise NotImplementedError('not implemented')

