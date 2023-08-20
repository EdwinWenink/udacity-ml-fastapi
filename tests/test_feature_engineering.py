import logging
from typing import Callable

import pytest
import pandas as pd


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def test_binary_label(target_label):
    """
    Since this is a binary classification problem we
    assert that indeed we have two target label values
    """
    try:
        assert len(pytest.df[target_label].unique()) == 2
        logger.info("Target label is binary.")
    except AssertionError:
        logger.error("Target label is not binary.")

# TODO test expected labels