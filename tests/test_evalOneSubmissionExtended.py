import pytest
from datetime import datetime

import pandas as pd

from tadpole_algorithms.transformations import convert_to_year_month, \
    convert_to_year_month_day, map_string_diagnosis


def test_forecastDf_date_conversion():
    forecastDf = pd.DataFrame([{'Forecast Date': '2019-07'}])

    assert pd.api.types.is_string_dtype(forecastDf.dtypes)

    # original conversion code
    forecastDf['Forecast Date'] = [datetime.strptime(x, '%Y-%m') for x in forecastDf['Forecast Date']] # considers every month estimate to be the actual first day 2017-01

    print(forecastDf.dtypes)
    assert pd.api.types.is_datetime64_ns_dtype(forecastDf['Forecast Date'])

    # new conversion code

    # from string
    forecastDf_new1 = pd.DataFrame([{'Forecast Date': '2019-07'}])
    forecastDf_new1['Forecast Date'] = convert_to_year_month(forecastDf_new1['Forecast Date'])

    assert pd.api.types.is_datetime64_ns_dtype(forecastDf_new1['Forecast Date'])

    # from date object
    forecastDf_new2 = pd.DataFrame([{'Forecast Date': datetime(2019, 7, 1, 0, 0, 0, 0)}])
    forecastDf_new2['Forecast Date'] = convert_to_year_month(forecastDf_new2['Forecast Date'])

    assert pd.api.types.is_datetime64_ns_dtype(forecastDf_new2['Forecast Date'])

    assert forecastDf['Forecast Date'].equals(forecastDf_new1['Forecast Date'])
    assert forecastDf_new1['Forecast Date'].equals(forecastDf_new2['Forecast Date'])


def test_d4Df_date_conversions():
    d4Df = pd.DataFrame([{'ScanDate': '2019-07-10'}])

    assert pd.api.types.is_string_dtype(d4Df.dtypes)

    # original code:
    d4Df['ScanDate'] = [datetime.strptime(x, '%Y-%m-%d') for x in d4Df['ScanDate']]

    assert pd.api.types.is_datetime64_ns_dtype(d4Df['ScanDate'])

    # new conversion code

    # from string
    d4Df_new1 = pd.DataFrame([{'ScanDate': '2019-07-10'}])
    d4Df_new1['ScanDate'] = convert_to_year_month_day(d4Df_new1['ScanDate'])

    assert pd.api.types.is_datetime64_ns_dtype(d4Df_new1['ScanDate'])

    # from date object
    d4Df_new2 = pd.DataFrame([{'ScanDate': datetime(2019, 7, 10, 0, 0, 0, 0)}])
    d4Df_new2['ScanDate'] = convert_to_year_month_day(d4Df_new2['ScanDate'])

    assert pd.api.types.is_datetime64_ns_dtype(d4Df_new2['ScanDate'])

    assert d4Df['ScanDate'].equals(d4Df_new1['ScanDate'])
    assert d4Df_new1['ScanDate'].equals(d4Df_new2['ScanDate'])


def test_map_string_diagnoses():
    d4Df = pd.DataFrame([{'Diagnosis': 'CN'}])

    # original conversion code
    mapping = {'CN' : 0, 'MCI' : 1, 'AD' : 2}
    d4Df.replace({'Diagnosis': mapping}, inplace=True)

    assert d4Df['Diagnosis'][0] == 0

    # new conversion code

    # for strings
    d4Df_new1 = pd.DataFrame([{'Diagnosis': 'CN'}])
    d4Df_new1['Diagnosis'] = map_string_diagnosis(d4Df_new1['Diagnosis'])

    assert d4Df_new1['Diagnosis'][0] == 0

    # for ints
    d4Df_new2 = pd.DataFrame([{'Diagnosis': 0}])
    d4Df_new2['Diagnosis'] = map_string_diagnosis(d4Df_new2['Diagnosis'])

    assert d4Df_new2['Diagnosis'][0] == 0

    assert d4Df['Diagnosis'].equals(d4Df_new1['Diagnosis'])
    assert d4Df_new1['Diagnosis'][0] == d4Df_new2['Diagnosis'][0]
