import pandas as pd
import numpy as np

def rewrite_d3(tadpoleD3_df):
    """
    Rewrite Data frame sothat it has colomns: RID, Diagnosis, ADAS13, Ventricles_ICV, Ventricles, ICV_bl
    """

    # Recode Diagnosis
    tadpoleD3_df = tadpoleD3_df.replace({'DX': {'MCI': 2, 'NL to MCI': 2, 'Dementia to MCI': 2, 'Dementia': 3, 'MCI to Dementia': 3, 'NL to Dementia': 3, 'NL': 1, 'MCI to NL': 1, 'Dementia to NL': 1}})
    tadpoleD3_df = tadpoleD3_df.rename(columns={'DX': 'Diagnosis', 'ICV': 'ICV_bl'})

    return tadpoleD3_df
