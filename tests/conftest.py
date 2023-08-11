"""
Configuration file for pytest.
"""

import logging
from typing import Callable, List

import pytest

from src.ml.data import load_csv


def pytest_configure(config):
    """Setup pytest namespace variables"""
    _ = config
    pytest.df = None
    pytest.X_train = None
    pytest.y_train = None
    pytest.X_test = None
    pytest.y_test = None
    pytest.expected_cols = set(
        ['age', 'workclass', 'fnlgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
         'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'])


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
