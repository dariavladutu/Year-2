"""Unit tests for postprocessing utilities in the CV6 project."""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from src.utils import postprocessing_utils as pp


def test_threshold_mask() -> None:
    """Test the threshold_mask function from the postprocessing_utils module."""
    raw = np.array([[0.05, 0.1, 0.15]])
    result = pp.threshold_mask(raw, threshold=0.1)
    expected = np.array([[0, 0, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_morphological_closing() -> None:
    """Test the morphological_closing function from the postprocessing_utils module."""
    binary = np.zeros((10, 10), dtype=np.uint8)
    binary[4:6, 4:6] = 1
    result = pp.morphological_closing(binary)
    assert result.shape == binary.shape
    assert result.dtype == np.float32


def test_crop_top_and_dish() -> None:
    """Test the crop_top_and_dish function from the postprocessing_utils module."""
    binary = np.ones((100, 100), dtype=np.uint8)
    crop_info = {"x_start": 10, "crop_size": 50}
    cropped, params = pp.crop_top_and_dish(binary, crop_info)
    assert cropped.shape[0] == int(100 * (1 - 0.15))
    assert cropped.shape[1] == 50
    assert params["x_start"] == 10
