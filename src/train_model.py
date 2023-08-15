'''
Script to train a simple machine learning model.
'''
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.ml.model import train_model, compute_model_metrics, inference
from src.ml.data import load_csv, preprocessing, feature_engineering


# Data preparation
# ----------------

# Load census data
df = load_csv('data/census.csv')

# Preprocess data
df = preprocessing(df)

# TODO Optional enhancement, use K-fold cross validation instead of a train-test split.
df_train, df_test = train_test_split(df, test_size=0.20)

# Categorical variables
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Target label
target_label = "salary"

# Feature engineering
X_train, y_train, encoder, lb = feature_engineering(
    df_train, categorical_features=cat_features, label=target_label, training=True
)

# Process the test data using the fitted OneHotEncoder and label binarizer
X_test, y_test, _, _ = feature_engineering(
    df_test, categorical_features=cat_features, label=target_label,
    training=False, encoder=encoder, lb=lb
)

# Training
# --------

# TODO I didn't notice but use the functions from src.ml.model
#
# def train_model(X_train, y_train):
#
# def compute_model_metrics(y, preds):
#
# def inference(model, X):

# Model parameters
params = {
    'n_estimators': 200
}

# Decision tree
clf = RandomForestClassifier(**params)

# Train model.
clf = train_model(clf, X_train, y_train)

# Save model
model_path = 'model/random_forest.pkl'
with open(model_path, 'wb') as fhandle:
    pickle.dump(clf, file=fhandle)

print("Saved fitted model to", model_path)


# Evaluation
# ----------

# Predict
y_pred = inference(clf, X_test)

# Model metrics overall
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("Fbeta:", fbeta)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Classification_report
report = classification_report(y_test, y_pred)
print("\nClassification report:\n", report)

# TODO
# - Write a function that outputs the performance of the model on slices of the data.
#   Suggestion: for simplicity, the function can just output the performance on slices
#   of just the categorical features.
# - Write a model card
# - Write unit tests for at least 3 functions in the model code