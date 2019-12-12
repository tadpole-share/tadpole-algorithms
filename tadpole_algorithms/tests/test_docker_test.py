from pathlib import Path
import datetime
import pandas as pd

from tadpole.models import SimpleSVM
from tadpole.models.docker_test import DockerTestModel


def test_docker_test():
    model = DockerTestModel()
    model.train(Path('data/train_short.csv'))
    # test_set_path = Path('data/tadpole_test_set.csv')
    # test_set_df = pd.read_csv(test_set_path)
    #
    # # select last row per RID
    # test_set_df = test_set_df.sort_values(by=['EXAMDATE'])
    # test_set_df = test_set_df.groupby('RID').tail(1)
    #
    # test_set_df = test_set_df.fillna(0)
    #
    # print(model.predict(test_set_df.iloc[0], datetime.datetime.now()))
