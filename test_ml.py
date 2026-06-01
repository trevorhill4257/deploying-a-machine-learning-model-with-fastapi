"""
test_ml.py

Project Step 4: Unit Test
At least 3 unit tests on the ML functions and/or the data.

Run with:
    > pytest test_ml.py -v

Rubric (Model building):
- Write at least 3 unit tests on ML or the data.
- Include a screenshot showing all tests passed, named unit_test.png.

Test ideas implemented below (you may replace with your own):
1. test_train_model_returns_expected_type - train_model returns the expected
   model type.
2. test_inference_returns_expected_type - inference returns a numpy array of
   predictions.
3. test_compute_model_metrics_values - compute_model_metrics returns floats
   in the valid [0, 1] range (or the expected value).
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def dummy_train_data():
    """Provide a small, deterministic dataset for testing."""
    # TODO (Step 4): Build or load a small dataset to use across tests.
    X = np.random.rand(20, 4)
    y = np.random.randint(0, 2, size=20)
    return X, y


def test_train_model_returns_expected_type(dummy_train_data):
    """train_model should return a fitted RandomForestClassifier."""
    # TODO (Step 4): Assert the returned object is the expected algorithm.
    X, y = dummy_train_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_expected_type(dummy_train_data):
    """inference should return a numpy array with one prediction per row."""
    # TODO (Step 4): Assert the type and shape of the predictions.
    X, y = dummy_train_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)


def test_compute_model_metrics_values():
    """compute_model_metrics should return three floats within [0, 1]."""
    # TODO (Step 4): Assert metrics are computed correctly / in range.
    y = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    for metric in (precision, recall, fbeta):
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0
