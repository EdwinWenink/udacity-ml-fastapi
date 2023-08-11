"""
Generate data report on census data.

Observations:

- The original data set had white spaces in the CSV. This was preprocessed manually once.
- Missing values are registered as '?' and should be converted to NaN for easier analysis
- The marital status factor has some categories that are very similar and can be merged.
- Capital gain and loss can be combined into a netto capital value that is positive or negative
- There are 23 duplicate rows
- Correlation:
    * `education_num` and `education` are highly correlated.
      `education_num` seems to be a label-encoding of `education`. Drop either of them.
    * `sex` and `relationship` are highly correlated
      This makes sense since relationship 'Wife' implies sex 'Female' etc.
- Imbalance:
    * `race`
    * `native-country`
    * `capital-gain`
    * `capital-loss`
- Missing values:
    * `workclass`: 1836 (5.6%)
    * `occupation`: 1843 (5.7%)
    * `native-country`: 483 (1.7%)
- Zeros:
    * `capital-gain` 29849 (91.7%)
    * `capital-loss` 31042 (95.3%)
"""

import numpy as np
from scipy import stats
from ydata_profiling import ProfileReport

from src.ml.data import load_csv

CAT_COLS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "salary"
]

NUM_COLS = [
    "capital-loss",
    "fnlgt",
    "age",
    "education-num",
    "capital-gain",
    "hours-per-week"
]

# Load census data
df = load_csv('data/census.csv')

# isna() suggests there are no missing values
print(df.isna().sum())

# However, missing values are encoded as '?' in this data set
# We convert them to NaN such that the report picks these up as missing
# Now we see:
#   - `workclass` has 1836 missing values
#   - `occupation` has 1843 missing values
#   - `native-country` has 583 missing values
df.replace('?', np.nan, inplace=True)
print(df.isna().sum())

# Which of the numerical columns contain outliers?
# We use a standard definition of outlier, with a z-score outside [-3, 3]
# N outliers per column
#
# capital-loss      1470
# fnlgt              347
# age                121
# education-num      219
# capital-gain       215
# hours-per-week     440
print((np.abs(stats.zscore(df[NUM_COLS])) > 3).sum(axis=0))

# Further inspection:
# capital-gain and capital-loss have many zeros
# Let's combine them so at least the zeros are combined into a single column
df['capital-change'] = df['capital-gain'] - df['capital-loss']

# Updated stats. Most values are zero so let's define outliers using non-zero values only
print((np.abs(stats.zscore(df['capital-change'])) > 3).sum(axis=0))  # 215
df_non_zero_change = df[df['capital-change'] > 0]
print((np.abs(stats.zscore(df_non_zero_change['capital-change'])) > 3).sum(axis=0))  # 159

print(df['capital-change'].min())  # -4356
print(df['capital-change'].max())  # 99999

# It turns out, when dropping zeros all outliers are 99999
# So it's a good idea to drop these
outliers = df_non_zero_change[np.abs(stats.zscore(df_non_zero_change['capital-change'])) > 3]
print(outliers['capital-change'].unique())

# These are not impossible. Let's leave them
df['hours-per-week'].min()  # 1
df['hours-per-week'].max()  # 99

# Idem
df['age'].min()  # 17
df['age'].max()  # 90

# There are only a few duplicates.
# There is no strong indication these are actual duplicates rather than
# individuals with the same characteristics, so I'll leave them in.
print(sum(df.duplicated()))
# df.drop_duplicates(inplace=True)

# Generate pandas profile of data set
profile = ProfileReport(df, title="Census data report.")

# Export as file
profile.to_file("census_report.html")
