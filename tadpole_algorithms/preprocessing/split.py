import numpy as np


def split_test_train(df, fraction, random_seed=0):
    """
    Split a dataframe into three parts: train, test & evaluation
    Splits by random sampling patients (RIDs) from df;
    n_patients_in_test_set = n_patients * fraction
    """

    # ids = df['RID'].unique()
    # get only patient IDs with at least 2 rows per patient (required for test/eval set)
    ids = df.groupby('RID').filter(lambda x: len(x) > 1)['RID'].unique()

    # how many patients for test / eval data
    len_subsample = int(len(ids) * fraction)

    np.random.seed(random_seed)
    subsample_rids = np.random.choice(ids, len_subsample, replace=False)

    df = df.sort_values(by=['RID', 'Years_bl'])

    test_eval_df = df[df['RID'].isin(subsample_rids)]
    train_df = df[~df['RID'].isin(subsample_rids)]

    last_row_indices = test_eval_df.groupby('RID').tail(1).index.values

    eval_df = test_eval_df[test_eval_df.index.isin(last_row_indices)]
    test_df = test_eval_df[~test_eval_df.index.isin(last_row_indices)]

    return train_df, test_df, eval_df

