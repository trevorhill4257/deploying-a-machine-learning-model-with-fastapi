"""
ml/model.py

Model utilities for the Census Income classification project.

Project Step 3: Model
- Implement the stubbed functions to train, run inference, save, and load
  the model, and to compute classification metrics.
- Implement performance_on_categorical_slice to compute metrics on data
  slices.

Rubric (Model building):
- Functions to train, save and load the model and any categorical encoders.
- Function for model inference.
- Function to determine the classification metrics.
- performance_on_categorical_slice computes performance on model slices.
"""

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train, y_train):
    """
    Train a machine learning model and return it.

    Parameters
    ----------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """
    # TODO (Step 3): Instantiate a classifier (e.g. RandomForestClassifier),
    # fit it on the training data, and return the trained model.
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validate the trained machine learning model using precision, recall,
    and F1.

    Parameters
    ----------
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
    # TODO (Step 3): Compute and return precision, recall, and F1 (fbeta).
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Parameters
    ----------
    model
        Trained machine learning model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # TODO (Step 3): Run model.predict on X and return the predictions.
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """Serialize a model or encoder to a file using pickle."""
    # TODO (Step 3): Save the model/encoder to `path` (e.g. with pickle).
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """Load a pickle file from `path` and return it."""
    # TODO (Step 3): Load and return the model/encoder from `path`.
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Compute the model metrics on a slice of the data specified by a column
    name and a value for that column.

    Processes the data using one hot encoding for the categorical features
    and a label binarizer for the labels, holding the given column fixed at
    `slice_value`.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the features and label, already preprocessed
        only in the sense of being the raw dataframe.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature held fixed.
    categorical_features : list
        Categorical feature column names.
    label : str
        Name of the label column in `data`.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained OneHotEncoder.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer.
    model
        Trained machine learning model.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    # TODO (Step 3/4): Filter the dataframe to rows where
    # data[column_name] == slice_value, process the slice with process_data
    # (training=False, reusing encoder and lb), run inference, and compute
    # the metrics.
    slice_data = data[data[column_name] == slice_value]

    X_slice, y_slice, _, _ = process_data(
        slice_data,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
