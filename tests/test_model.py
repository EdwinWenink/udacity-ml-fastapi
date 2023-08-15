import logging

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.ml.model import train_model

logger = logging.getLogger(__name__)


def test_model_fitting(get_Xy, unfitted_model):
    '''Check if estimator is fitted.'''
    X, y = get_Xy
    clf = unfitted_model
    clf = train_model(clf, X, y)
    try:
        check_is_fitted(clf)
    except NotFittedError:
        logger.error("Estimator is not fitted.")
