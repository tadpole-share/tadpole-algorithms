from abc import ABC, abstractmethod


class TadpoleModel(ABC):
    @abstractmethod
    def train(self, train_df):
        pass

    @abstractmethod
    def predict(self, test_df):
        pass

    def save(self, path):
        raise NotImplementedError(f'Save not implemented for model {self.__class__.__name__}.')

    def load(self, path):
        raise NotImplementedError(f'Load not implemented for model {self.__class__.__name__}.')
