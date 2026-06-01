"""
ml/data.py

Data processing utilities for the Census Income classification project.

Project Step 2: Data
- Inspect the data (census.csv) and understand the preprocessing scripts.
- This module provides the `process_data` function used to process the
  data before training and during inference.

Rubric (Model building):
- The data should be split (train/test) or use cross-validation.
- process_data is used to process both the training and test data.
"""

import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X,
    categorical_features=[],
    label=None,
    training=True,
    encoder=None,
    lb=None,
):
    """
    Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features
    and a label binarizer for the labels. This can be used in either
    training or inference/validation.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`.
    categorical_features : list[str]
        List containing the names of the categorical features (default=[]).
    label : str
        Name of the label column in `X`. If None, then an empty array will
        be returned for y (default=None).
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the
        encoder passed in.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the
        binarizer passed in.
    """
    # TODO (Step 2): Separate the label column from the features when a
    # label is provided.
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    # TODO (Step 2): Split the categorical and continuous features.
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        # TODO (Step 2): Fit the OneHotEncoder and LabelBinarizer on the
        # training data.
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        # TODO (Step 2): Transform using the already-fitted encoder/lb.
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def apply_label(inference):
    """Convert the binary label in a single inference sample into string output."""
    # TODO: Map the model's binary output back to the salary string label.
    if inference[0] == 1:
        return ">50K"
    elif inference[0] == 0:
        return "<=50K"
