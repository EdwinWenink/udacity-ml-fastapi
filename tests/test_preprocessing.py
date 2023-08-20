import logging

import pytest
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def test_capital_change_col(preprocessed_data: pd.DataFrame):
    """
    The preprocessing introduces a column called 'capital-change'.
    Check if it is present and the old columns are dropped.
    """
    try:
        assert 'capital-change' in preprocessed_data.columns
        logger.info("Column `capital-change` present as expected.")
    except AssertionError:
        logger.error("Column `capital-change` missing.")

    try:
        assert 'capital-gain' not in preprocessed_data.columns
        assert 'capital-loss' not in preprocessed_data.columns
        logger.info("Columns `capital-gain` and `capital-loss` were dropped.")
    except AssertionError:
        logger.error("Columns `capital-gain` and `capital-loss` were not dropped.")


def test_preprocessed_size():
    try:
        assert pytest.df_preprocessed.shape[0] > 0
        assert pytest.df_preprocessed.shape[1] > 0
        logger.info("Preprocessed dataframe has non-zero dimensions.")
    except AssertionError as err:
        logger.error(
            "The preprocessed dataframe doesn't appear to have rows and columns")
        raise err
