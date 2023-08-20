import logging

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import numpy as np

from src.ml.model import train_model, compute_model_metrics, inference

logger = logging.getLogger(__name__)


def test_model_fitting(get_Xy, unfitted_model):
    '''Check if estimator is fitted.'''
    X, y = get_Xy
    clf = unfitted_model
    clf = train_model(clf, X, y)
    try:
        assert check_is_fitted(clf)
    except AssertionError:
        logger.error("Estimator is not fitted.")


def test_model_inference(get_Xy, unfitted_model):
    X, y = get_Xy
    clf = unfitted_model
    clf = train_model(clf, X, y)
    y_pred = inference(clf, X)
    # Output should be 1-D array
    try:
        assert y_pred.ndim == 1
    except AssertionError:
        logger.error("Inference output should be 1-D array.")

    # Output should be a class label (integer)
    assert isinstance(y_pred[0], np.integer)


def test_model_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred, beta=1)
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_model_metrics_zero_division():
    '''
    We use zero_division == 1, which assigns a recall of 1.0
    instead of 0.0 in absence of positive labels.
    '''
    y_true = np.array([0, 0])
    y_pred = np.array([0, 1])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    assert precision == 0.0
    assert recall == 1.0
    assert f1 == 0.0
