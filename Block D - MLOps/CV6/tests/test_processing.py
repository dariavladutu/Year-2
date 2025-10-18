"""Post-processing utilities for image segmentation masks."""

import numpy as np
import pytest
from cv2 import MORPH_CLOSE, morphologyEx
from skimage.morphology import remove_small_objects

from src.utils.processing import refine_mask


def test_refine_mask_basic_closing():
    """Test that refine_mask closes small holes using morphological closing."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    mask[9:11, 9:11] = 0  # small hole

    refined = refine_mask(mask, kernel_size=3, iterations=2)

    assert refined[10, 10] == 255  # hole should be filled
    assert refined.shape == mask.shape
    assert set(np.unique(refined)).issubset({0, 255})


def test_refine_mask_removes_small_objects():
    """Test that refine_mask removes objects smaller than the default threshold."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:8, 5:8] = 255       # small object (9 px)
    mask[20:45, 20:45] = 255   # large object (625 px)

    refined = refine_mask(mask, kernel_size=3, iterations=1)

    assert np.all(refined[5:8, 5:8] == 0)       # small object removed
    assert np.all(refined[30, 30] == 255)       # large object retained


def test_refine_mask_accepts_binary_0_1():
    """Test that refine_mask converts 0/1 input masks to 0/255 output."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1

    refined = refine_mask(mask)

    assert set(np.unique(refined)).issubset({0, 255})


def test_refine_mask_invalid_input_raises():
    """Test that refine_mask raises ValueError on invalid input shape."""
    mask = np.array([0, 1, 1, 0], dtype=np.uint8)

    with pytest.raises(ValueError):
        refine_mask(mask)


def test_refine_mask_all_zeros():
    """Test that a fully black mask stays unchanged."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    refined = refine_mask(mask)

    assert np.all(refined == 0)


def test_refine_mask_all_ones():
    """Test that a fully white mask remains white after processing."""
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    refined = refine_mask(mask)

    assert np.all(refined == 255)
