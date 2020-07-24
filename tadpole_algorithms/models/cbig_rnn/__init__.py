#!/usr/bin/env python
"""
Description:
RNN model implementation for Tadpole-share project
Date: 7/19/2020
Email: anlijuncn@gmail.com
Written by Lijun AN and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""
import os
import shutil
import logging
import pandas as pd

from tadpole_algorithms.models.tadpole_model import TadpoleModel
from tadpole_algorithms.models.cbig_rnn.cbig.Nguyen2020 import gen_cv_fold
from tadpole_algorithms.models.cbig_rnn.cbig.Nguyen2020 import gen_cv_pickle
from tadpole_algorithms.models.cbig_rnn.cbig.Nguyen2020 import cbig_train
from tadpole_algorithms.models.cbig_rnn.cbig.Nguyen2020 import cbig_predict

logger = logging.getLogger(__name__)


class CBIG_RNN(TadpoleModel):
    """
    CBIG_RNN@Tadpole-Share Implementation
    """

    def __init__(self,
                 verobose='verbose',
                 clean_up=True,
                 **kwargs):
        # default model parameters for MinimalRNN model
        self.lr = 1e-3  # learning rate
        self.h_drop = 0.4  # hidden dropout rate
        self.i_drop = 0.1  # input dropout rate
        self.h_size = 128  # hidden size
        self.nb_layers = 2  # nb_layers
        self.epochs = 100  # epoch
        self.model = 'MinRNN'  # model type
        self.weight_decay = 1e-5  # weight decay
        self.features = 'longitudinal_features'
        self.isD3 = False
        # check if user set different model parameters
        for k, w in kwargs.items():
            setattr(self, k, w)
        # get the work directory for saving and loading model
        self.work_directory = os.path.abspath(os.path.dirname(__file__))
        # other parameters such as verbose
        self.verbose = verobose
        # clean up the output directory if clean_up is True
        if clean_up:
            if os.path.isdir(os.path.join(self.work_directory, 'out')):
                shutil.rmtree(os.path.join(self.work_directory, 'out'))
            # create an empty out folder
            os.makedirs(os.path.join(self.work_directory, 'out'))

    def preprocess(self, df: pd.DataFrame, isD3=False, isTrain=True):
        # preprocesing
        logger.info("Pre-processing")
        # save it as a csv file
        if isTrain:
            name = 'train'
        else:
            name = 'test'
        # for D3 (cross-sectional set)
        if isD3 and ~isTrain:
            df['DX'] = df['Diagnosis'].values
            month_bl = df['VISCODE'].values
            for i in range(len(month_bl)):
                month_bl[i] = float(month_bl[i][1:])
            df['Month_bl'] = month_bl
            df = df.rename(columns={'Diagnosis': 'DXCHANGE', 'ICV_bl': 'ICV'})
            df = df.replace({'DXCHANGE': {0: 7,
                                          1: 4,
                                          2: 3}})
        if ~isTrain and ~isD3:
            df.loc[(df.RID == 1195) & (df.VISCODE == 'm42'), ['Month_bl']] = 42
            df.loc[(df.RID == 4960) & (df.VISCODE == 'm48'), ['Month_bl']] = 48
        df.to_csv(os.path.join(self.work_directory, 'out', name + '.csv'))
        # args
        gen_cv_fold_args = gen_cv_fold.get_args()
        gen_cv_fold_args.spreadsheet = \
            os.path.join(self.work_directory, 'out', name + '.csv')
        gen_cv_fold_args.features = \
            os.path.join(self.work_directory, 'data', self.features)
        gen_cv_fold_args.outdir = os.path.join(self.work_directory, 'out')
        gen_cv_fold_args.isTrain = isTrain
        # gen_cv_fold.py
        gen_cv_fold.main(gen_cv_fold_args, name)
        logger.info("Pre-processing...Filling missing value")
        # args
        gen_cv_pickle_args = gen_cv_pickle.get_args()
        gen_cv_pickle_args.mask = os.path.join(self.work_directory, 'out',
                                               name + 'fold0_mask.csv')
        gen_cv_pickle_args.isTrain = isTrain

        gen_cv_pickle_args.spreadsheet = \
            os.path.join(self.work_directory, 'out', name + '.csv')
        gen_cv_pickle_args.features = \
            os.path.join(self.work_directory, 'data', self.features)
        gen_cv_pickle_args.strategy = 'model'
        gen_cv_pickle_args.out = os.path.join(self.work_directory, 'out',
                                              name + 'test.f0.pkl')
        # gen_cv_pickle.py
        gen_cv_pickle.main(gen_cv_pickle_args)

    def train(self, train_df):
        logger.info("Training models")
        # run preprocess()
        self.preprocess(train_df, isTrain=True)
        # train model
        train_args = cbig_train.get_args()
        train_args.data = os.path.join(self.work_directory, 'out',
                                       'train' + 'test.f0.pkl')
        train_args.lr = self.lr  # learning rate
        train_args.h_drop = self.h_drop  # hidden dropout rate
        train_args.i_drop = self.i_drop  # input dropout rate
        train_args.h_size = self.h_size # hidden size
        train_args.nb_layers = self.nb_layers  # nb_layers
        train_args.epochs = self.epochs  # epoch
        train_args.model = self.model  # model type
        train_args.weight_decay = self.weight_decay  # weight decay
        train_args.out = os.path.join(self.work_directory, 'out',
                                      'model.f0.pt')
        train_args.verbose = self.verbose
        cbig_train.train(train_args)

    def predict(self, test_df):
        logger.info("Predicting")
        # preprocessing
        self.preprocess(test_df, self.isD3, isTrain=False)
        # make prediction
        predict_args = cbig_predict.get_args()
        predict_args.checkpoint = os.path.join(self.work_directory, 'out',
                                               'model.f0.pt')
        predict_args.data = os.path.join(self.work_directory, 'out',
                                         'test' + 'test.f0.pkl')
        predict_args.out = os.path.join(self.work_directory, 'out',
                                        'prediction_test.f0.csv')
        # predict
        cbig_predict.main(predict_args)

        pred_df = pd.read_csv(os.path.join(self.work_directory, 'out',
                                           'prediction_test.f0.csv'))

        return pred_df

    def save(self, path):
        logger.info("Saving model")

    def load(self, path):
        logger.info("Loading model")



