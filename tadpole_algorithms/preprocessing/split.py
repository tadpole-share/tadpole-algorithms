import numpy as np
import pandas as pd 

#def split_test_train_random


def split_test_train_tadpole(df_train_test, df_eval, random_seed=0):
    """
    Split dataframes into three parts: train, test & evaluation
    These are the sets as used in challenge evaluation for the paper Marinescu et al, 2020, ArXiv
    Train: patients (RIDs) from D1,D2 ADNI Data sets
    Test: roll-over patients (RIDs) from D1,D2 ADNI Data sets that are in D4
    Eval: D4 ADNI Data set
    """

    # get only patient IDs with at least 2 rows per patient (required for test/eval set)
    ids = df_train_test.groupby('RID').filter(lambda x: len(x) > 1)['RID'].unique()
    
    train_df = df_train_test[df_train_test['RID'].isin(ids)]
        
    # get last row per RID
    df_train_test = df_train_test.groupby('RID').tail(1)

    # select all records where RID is in d4.
    test_df = df_train_test[
        df_train_test['RID'].isin(df_eval['RID'].unique())
    ]

    eval_df = df_eval

    return train_df, test_df, eval_df


def split_test_train_d3(df_train, df_test, df_eval, random_seed=0):
    """
    Split a dataframe into three parts: train, test and evaluation
    Data set used for train is D1D2, for test D3 and D4 for evalution
    Cross sectional dataset is used as test set

    """
    # get only patient IDs with at least 2 rows per patient (required for test/eval set)
    ids = df_train.groupby('RID').filter(lambda x: len(x) > 1)['RID'].unique()
    
    train_df = df_train[df_train['RID'].isin(ids)]

    # get D1 rows only
    train_df = train_df.loc[train_df['D2'] == 1]

    # select all records where RID is in d4.
    test_df = df_test[
        df_test['RID'].isin(df_eval['RID'])
    ]

    eval_df = df_eval

    return train_df, test_df, eval_df

# def split_test_train_parelsnoer(df_train_test, df_eval, random_seed=0):
#     #### Will be written soon



    