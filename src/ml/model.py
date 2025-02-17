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


def compute_model_metrics(y, preds, beta=1):
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
    # `zero_division` determines the behavior of e.g. recall in absence of positive labels.
    # When there are no positive labels, this function will return a recall of 1.0
    positive_label = 1
    zero_division = 1
    fbeta = float(fbeta_score(y, preds, beta=beta, zero_division=zero_division,
                              pos_label=positive_label))
    precision = float(precision_score(y, preds, zero_division=zero_division,
                                      pos_label=positive_label))
    recall = float(recall_score(y, preds, zero_division=zero_division,
                                pos_label=positive_label))
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
    NOTE: this should include required data transforms.
    """
    with open(model_uri, "rb") as fhandle:
        model = pickle.load(fhandle)
    return model
