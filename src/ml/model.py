import pickle

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.base import BaseEstimator


# Optional: implement hyperparameter tuning.
def train_model(clf, X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model: sklearn BaseEstimator
        Trained machine learning model.
    """
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(clf, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn BaseEstimator
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y = clf.predict(X)
    return y


def load_model(model_uri: str):
    """
    Load a picked trained model.
    """
    with open(model_uri, "rb") as fhandle:
        model = pickle.load(fhandle)
    return model
