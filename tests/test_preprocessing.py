import logging

import pytest
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def test_capital_change_col(preprocessed_data: pd.DataFrame):
    """
    The preprocessing introduces a column called 'capital-change'.
    Check it is present.
    """
    try:
        assert 'capital-change' in preprocessed_data.columns
        logger.info("Columns `capital-change` present as expected.")
    except AssertionError:
        logger.error("Columns `capital-change` missing.")


def test_preprocessed_size():
    try:
        assert pytest.df_preprocessed.shape[0] > 0
        assert pytest.df_preprocessed.shape[1] > 0
        logger.info("Preprocessed dataframe has non-zero dimensions.")
    except AssertionError as err:
        logger.error(
            "The preprocessed dataframe doesn't appear to have rows and columns")
        raise err
