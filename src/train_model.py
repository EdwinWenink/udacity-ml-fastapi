'''
Script to train a simple machine learning model.
'''
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.ml.data import load_csv, preprocessing, feature_engineering


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

# Preprocessing
X_train, y_train, encoder, lb = feature_engineering(
    df_train, categorical_features=cat_features, label=target_label, training=True
)

# Process the test data using the fitted OneHotEncoder and label binarizer
X_test, y_test, _, _ = feature_engineering(
    df_test, categorical_features=cat_features, label=target_label,
    training=False, encoder=encoder, lb=lb
)

# Model parameters
params = {
    'n_estimators': 200
}

# Decision tree
clf = RandomForestClassifier(**params)

# Train and save a model.
clf.fit(X_train, y_train)
joblib.dump(clf, filename='model/random_forest.pkl')

# Predict
y_pred = clf.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Classification_report
report = classification_report(y_test, y_pred)
print("\nClassification report:\n", report)
