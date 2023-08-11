from typing import List

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def load_csv(uri: str) -> pd.DataFrame:
    return pd.read_csv(uri)


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocess the data set.
    See src/eda.py for insights underlying these decisions.
    '''
    # Register missing values ('?') as NaN
    df = df.replace('?', np.nan)

    # Drop rows with missing observations
    df = df.dropna()

    # `capital-gain` and `capital-loss` have many zeros and outliers
    # Let's combine them so at least the zeros are combined into a single column
    df['capital-change'] = df['capital-gain'] - df['capital-loss']

    # Drop the old columns
    df = df.drop(['capital-gain', 'capital-loss'], axis='columns')

    # Drop outliers defined on non-zero capital change
    df_non_zero_change = df[df['capital-change'] > 0]
    outliers = df_non_zero_change[np.abs(stats.zscore(df_non_zero_change['capital-change'])) > 3]
    df = df.drop(outliers.index, axis='index')

    return df


def feature_engineering(
    X: pd.DataFrame, categorical_features: List[str] = [],
    label: str = '', training: bool = True,
    encoder: OneHotEncoder = None,
    lb: LabelBinarizer = None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label:
        y = X.pop(label)
    else:
        y = pd.Series([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
