import numpy as np

def split_test_train(df_train_test, df_eval, random_seed=0):
    """
    Split a dataframe into three parts: train, test & evaluation
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

