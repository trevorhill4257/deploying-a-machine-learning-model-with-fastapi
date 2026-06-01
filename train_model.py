"""
train_model.py

Project Step 4: ML Pipeline
Script that takes the data, processes it, trains the model, saves the model
and encoder, and computes performance on data slices.

Pipeline steps (per project instructions):
1. Load the census.csv data.
2. Split into a training dataset and a test dataset.
3. Use process_data to process both the training and test data.
4. Use train_model to train the model on the training dataset.
5. Use inference to run model inferences on the test dataset.
6. Compute performance on data slices using performance_on_categorical_slice
   and save the output to slice_output.txt.

Run with:
    > python train_model.py
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# TODO (Step 4): Load the census.csv data with pandas. Set the project path
# and the data path.
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)

# TODO (Step 4): Split the provided data into a train and a test dataset.
# Optional enhancement: use K-fold cross validation instead of a train-test
# split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

# Categorical features for the Census Income dataset.
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

# TODO (Step 4): Use the process_data function to process the training data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# TODO (Step 4): Use the process_data function to process the test data.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO (Step 4): Use the train_model function to train the model on the
# training dataset.
model = train_model(X_train, y_train)

# Save the model and the encoder.
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
print(f"Model saved to {model_path}")
print(f"Encoder saved to {encoder_path}")

# Load the model back (demonstrates load_model).
model = load_model(model_path)
print(f"Loading model from {model_path}")

# TODO (Step 4): Use the inference function to run model inferences on the
# test dataset.
preds = inference(model, X_test)

# Calculate and print the overall metrics.
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO (Step 4): Compute the performance on model slices using the
# performance_on_categorical_slice function. Iterate through the categorical
# features and the unique values of each, and write the results to
# slice_output.txt.
slice_output_path = os.path.join(project_path, "slice_output.txt")
with open(slice_output_path, "w") as f:
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model,
            )
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
