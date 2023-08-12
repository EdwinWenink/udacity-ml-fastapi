"""
Configuration file for pytest.
"""

import logging
from typing import Callable, List, Tuple

import pytest
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from src.ml.data import load_csv, preprocessing, feature_engineering


def pytest_configure(config):
    """Setup pytest namespace variables"""
    _ = config
    pytest.df = None
    pytest.df_preprocessed = None
    pytest.X = None
    pytest.y = None
    pytest.expected_input_cols = set(
        ['age', 'workclass', 'fnlgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
         'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'])
    pytest.cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    pytest.target_label = 'salary'
    # TODO add cols expected by the model after feature engineering


def pytest_runtest_setup(item):
    """Initialize logger per test item."""
    # Get the logger for the current test
    logger = logging.getLogger(item.nodeid)

    # Log a message for the start of the test
    logger.info("Starting test: %s", item.name)


@pytest.fixture(scope='session')
def valid_input_path():
    """Defines a valid data input path."""
    return "./data/census.csv"


@pytest.fixture(scope="session")
def target_label() -> str:
    """Returns the target label of the classification problem."""
    return 'salary'


# The following fixtures form a chain
@pytest.fixture(scope="session")
def import_data() -> Callable:
    """Returns data importing function."""
    return load_csv


@pytest.fixture(scope='session')
def input_data(import_data, valid_input_path):
    pytest.df = import_data(valid_input_path)
    return pytest.df


@pytest.fixture(scope='session')
def preprocessed_data(input_data) -> pd.DataFrame:
    # NOTE but what if this function itself fails?
    pytest.df_preprocessed = preprocessing(input_data)
    return pytest.df_preprocessed


@pytest.fixture(scope='session')
def get_Xy(preprocessed_data) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pytest.X, pytest.y, _, _ = feature_engineering(
        preprocessed_data,
        categorical_features=pytest.cat_features,
        label=pytest.target_label
    )
    return pytest.X, pytest.y


@pytest.fixture(scope='function')
def unfitted_model() -> BaseEstimator:
    # Decision tree
    return RandomForestClassifier()
