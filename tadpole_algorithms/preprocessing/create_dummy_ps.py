import pandas as pd
from pathlib import Path

def create_parelsnoer_dummy(df):
    """
    This function creates a dummy parelsnoer data set (used for the TADPOLE SHARE Hackathon on 23-24 of July)
    The ADNI D3 data set is constricted in terms of variables that are present in the Parelsnoer dataset
    It serves as a Parelsnoer dummy dataset
    """

    # Drop columns which are not present in parelsnoer
    h = list(df)
    df = df.drop(h[9:12] + [h[22]] + h[24:37], axis=1)

    # Force values to numeric
    df = df.astype("float64", errors='ignore')

    return df
