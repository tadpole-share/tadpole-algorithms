"""Tadpole data transformations"""
import pandas as pd

from datetime import datetime


def get_diagnosis(df):
    """Map DXCHANGE column to Diagnosis

    Convert diagnoses such as 'MCI to Dementia' to 'Dementia', etc ...
    ctlDxchange = [1, 7, 9] mciDxchange = [2, 4, 8] adDxChange = [3, 5, 6]
    """
    mapping = {1:'CN',
               7:'CN',
               9:'CN',
               2:'MCI',
               4:'MCI',
               8:'MCI',
               3:'AD',
               5:'AD',
               6:'AD'}
    return df['DXCHANGE'].replace(mapping)


def convert_to_year_month(dates):
    # considers every month estimate to be the actual first day 2017-01

    if pd.api.types.is_string_dtype(dates):
        # string to datetime object
        return dates.apply(lambda x: datetime.strptime(x, '%Y-%m'))
    else:
        # datetime object
        # ? (convert to first of the month)
        return dates.apply(lambda x: datetime(x.year, x.month, 1, 0, 0, 0))


def convert_to_year_month_day(dates):
    if pd.api.types.is_string_dtype(dates):
        # string to datetime object
        return dates.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    else:
        return dates


def map_string_diagnosis(diagnoses):
    mapping = {'CN' : 0, 'MCI' : 1, 'AD' : 2}
    if pd.api.types.is_string_dtype(diagnoses):
        return diagnoses.replace(mapping)
    return diagnoses