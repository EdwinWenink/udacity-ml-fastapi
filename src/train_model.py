'''
Script to train a simple machine learning model.
'''

import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml.data import load_csv, process_data


# Load census data
data = load_csv('data/census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# Variables
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

# Preprocessing
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=target_label, training=True
)

# Process the test data using the fitted OneHotEncoder and label binarizer
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label=target_label,
    training=False, encoder=encoder, lb=lb
)

# Train and save a model.
