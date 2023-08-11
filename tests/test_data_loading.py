import logging
from typing import Callable

import pytest
import pandas as pd


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def test_import(import_data: Callable, valid_input_path):
    '''
    Test data loading when provided with a valid input path.
    '''
    try:
        # Stores the dataframe in the pytest namespace for later use
        pytest.df = import_data(valid_input_path)
        logger.info("Data successfully imported from %s", valid_input_path)
    except FileNotFoundError as err:
        logger.error("Input data wasn't found at expected path %s", valid_input_path)
        raise err


def test_input_size():
    try:
        assert pytest.df.shape[0] > 0
        assert pytest.df.shape[1] > 0
        logger.info("Input dataframe has non-zero dimensions.")
    except AssertionError as err:
        logger.error(
            "The input dataframe doesn't appear to have rows and columns")
        raise err


def test_df_keys():
    '''
    Assert the input data has the expected keys.
    '''
    try:
        logger.info("Dataframe keys: %s", pytest.df.keys())
        logger.info("Expected columns: %s", pytest.expected_input_cols)
        assert set(pytest.df.keys()) == set(pytest.expected_input_cols)
    except AssertionError:
        logger.error("Input data does not have the expected columns.")
