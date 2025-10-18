"""Test cases for the F1 metric implementation in model evaluation."""
import numpy as np
import pytest
from keras import backend as K
from tensorflow import convert_to_tensor

from src.utils.model_evaluation import f1


@pytest.mark.parametrize(
    "y_true, y_pred, expected_f1",
    [
        # Perfect prediction
        ([1, 0, 1, 0], [1, 0, 1, 0], 1.0),
        # All incorrect
        ([1, 1, 0, 0], [0, 0, 1, 1], 0.0),
        # Half correct
        ([1, 0, 1, 0], [1, 1, 0, 0], 0.5),
        # All positive
        ([1, 1, 1, 1], [1, 1, 1, 1], 1.0),
        # All negative
        ([0, 0, 0, 0], [0, 0, 0, 0], 0.0),
        # Mixed, partial match
        ([1, 0, 1, 0], [1, 1, 1, 0], 0.8),
    ],
)
def test_f1_basic_cases(y_true, y_pred, expected_f1):
    """Test the f1 metric across a variety of prediction scenarios."""
    y_true_tensor = convert_to_tensor(np.array(y_true, dtype=np.float32))
    y_pred_tensor = convert_to_tensor(np.array(y_pred, dtype=np.float32))
    result = f1(y_true_tensor, y_pred_tensor)
    result_value = K.get_value(result)
    assert np.isclose(result_value, expected_f1, atol=1e-2)


def test_f1_handles_epsilon():
    """Test that f1 handles a zero-denominator case gracefully using epsilon."""
    y_true = convert_to_tensor(np.array([0, 0, 0, 0], dtype=np.float32))
    y_pred = convert_to_tensor(np.array([0, 0, 0, 0], dtype=np.float32))
    result = f1(y_true, y_pred)
    result_value = K.get_value(result)
    assert np.isclose(result_value, 0.0, atol=1e-6)


def test_f1_partial_overlap():
    """Test the f1 score for partial overlap of prediction and ground truth."""
    y_true = convert_to_tensor(np.array([1, 1, 0, 0], dtype=np.float32))
    y_pred = convert_to_tensor(np.array([1, 0, 1, 0], dtype=np.float32))
    result = f1(y_true, y_pred)
    result_value = K.get_value(result)
    # Precision: 1/(1+1)=0.5, Recall: 1/(1+1)=0.5, F1=0.5
    assert np.isclose(result_value, 0.5, atol=1e-2)
