'''
Script to train a simple machine learning model.
'''
import pickle
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer


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
X_train, y_train, column_transformer, lb = feature_engineering(
    df_train, categorical_features=cat_features, label=target_label, training=True
)

# Print encoding of target labels for interpretation
assert len(lb.classes_) == 2
print("Label interpretation:")
for true_label, encoded_label in zip(lb.classes_, lb.transform(lb.classes_)):
    print(f"{true_label} -> {int(encoded_label)}")

# Process the test data using the fitted OneHotEncoder and label binarizer
X_test, y_test, _, _ = feature_engineering(
    df_test, categorical_features=cat_features, label=target_label,
    training=False, column_transformer=column_transformer, lb=lb
)

# Training
# --------

# Model parameters
# For this demo, we don't bother with hyper parameter optimization
params = {
    'n_estimators': 200
}

# Decision tree
clf = RandomForestClassifier(**params)

# Train model.
clf = train_model(clf, X_train, y_train)

# Save model including data transforms
pipe = make_pipeline(column_transformer, clf)

model_path = 'model/random_forest.pkl'
with open(model_path, 'wb') as fhandle:
    pickle.dump(pipe, file=fhandle)

print("Saved fitted model incl. data transforms to", model_path)


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
report = classification_report(y_test, y_pred, zero_division=1)
print("\nClassification report:\n", report)

# Slices
# ------

# Compute metrics on slices of the data
# For simplicity we slice only on categorical features
results = dict()
for cat_feature in cat_features:
    feature_values = dict()
    print("Computing slices for feature:", cat_feature)
    print(df_test[cat_feature].unique())
    for value in list(df_test[cat_feature].unique()):
        filter = df_test[cat_feature] == value

        # Apply encoding
        X_slice, y_slice, _, _ = feature_engineering(
            df_test[filter], categorical_features=cat_features, label=target_label,
            training=False, column_transformer=column_transformer, lb=lb)

        # Predict labels for the slice and compute metrics
        y_slice_pred = inference(clf, X_slice)
        precision, recall, f1 = compute_model_metrics(y_slice, y_slice_pred)

        # Store slice results and support
        feature_values[value] = {'precision': precision,
                                 'recall': recall,
                                 'f1': f1,
                                 'support': len(X_slice)}

    results[cat_feature] = feature_values

slice_eval_file = 'slice_evaluation.yaml'
print(f"Writing evaluation of data slices to file {slice_eval_file}.")
with open(slice_eval_file, 'w') as fhandle:
    yaml.dump(results, fhandle)

# TODO
# - Write a model card
# - Write unit tests for at least 3 functions in the model code