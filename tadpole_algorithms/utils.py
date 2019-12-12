import random
import numpy as np

def test_train_split(df, test_fraction=0.1):
    # remove all rows without 'Ventricles"
    # df_clean = df[df["Ventricles"].notnull()]
    patient_ids = set(df["RID"].unique().tolist())

    # split 90/10 train/test
    train_patient_ids = set(random.sample(patient_ids, int(len(patient_ids) * test_fraction)))
    test_patient_ids = patient_ids - train_patient_ids
    train_df = df[df['RID'].isin(train_patient_ids)]

    # combined test & ground truth
    test_ground_truth_df = df[df['RID'].isin(test_patient_ids)]

    # drop all patients with only one record
    test_ground_truth_df = test_ground_truth_df.groupby('RID').filter(lambda x: len(x) > 1)

    # ground truth is last row per patient
    ground_truth_df = test_ground_truth_df.groupby('RID').tail(1)

    # test set is the rest of the rows
    test_df = test_ground_truth_df.drop(ground_truth_df.index)

    assert len(ground_truth_df.groupby('RID')) == len(test_df.groupby('RID'))

    return train_df.reset_index(drop=True),\
           test_df.reset_index(drop=True),\
           ground_truth_df.reset_index(drop=True)
