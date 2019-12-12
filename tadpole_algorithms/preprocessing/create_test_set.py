# Generate tadpole_test_set given TADPOLE_D1_D2.csv & TADPOLE_D4_corr.csv
# This data set is the test set used for Tadpole-Share

import pandas

from pathlib import Path

d1d2_path = Path('../data/TADPOLE_D1_D2.csv')
d4_path = Path('../data/TADPOLE_D4_corr.csv')
test_set_path = Path('../data/tadpole_test_set.csv')


d1d2_df = pandas.read_csv(d1d2_path)
d4_df = pandas.read_csv(d4_path)

d1d2_df = d1d2_df.sort_values(by=['EXAMDATE'])

# get last row per RID
d1d2 = d1d2_df.groupby('RID').tail(1)

# select all records where RID is in d4.
test_df = d1d2_df[
    d1d2_df['RID'].isin(d4_df['RID'].unique())
]

test_df.to_csv(test_set_path, index=False)
