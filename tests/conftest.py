"""
Configuration file for pytest.
"""

import logging
from typing import Callable, List

import pytest
import pandas as pd

from src.ml.data import load_csv, preprocessing


def pytest_configure(config):
    """Setup pytest namespace variables"""
    _ = config
    pytest.df = None
    # TODO how do I make sure these are filled in order?
    pytest.df_preprocessed = None
    pytest.X_train = None
    pytest.y_train = None
    pytest.X_test = None
    pytest.y_test = None
    pytest.expected_input_cols = set(
        ['age', 'workclass', 'fnlgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
         'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'])
    # TODO add cols expected by the model after feature engineering


def pytest_runtest_setup(item):
    """Initialize logger per test item."""
    # Get the logger for the current test
    logger = logging.getLogger(item.nodeid)

    # Log a message for the start of the test
    logger.info("Starting test: %s", item.name)


@pytest.fixture
def valid_input_path():
    """Defines a valid data input path."""
    return "./data/census.csv"


@pytest.fixture(scope="module")
def import_data() -> Callable:
    """Returns data importing function."""
    return load_csv


@pytest.fixture(scope="session")
def target_label() -> str:
    """Returns the target label of the classification problem."""
    return 'salary'


@pytest.fixture(scope='session')
def input_data(import_data, valid_input_path):
    pytest.df = import_data(valid_input_path)
    return pytest.df


@pytest.fixture(scope='session')
def preprocessed_data() -> pd.DataFrame:
    # NOTE but what if this function itself fails?
    pytest.df_preprocessed = preprocessing(pytest.df)
    return pytest.df_preprocessed
