"""Unit tests for preprocessing utilities in the CV6 project."""

import numpy as np

from src.utils import preprocessing_utils as pre


def test_uncropper_with_no_crop() -> None:
    """Test the uncropper function with no cropping applied."""
    img = np.zeros((100, 100), dtype=np.uint8)
    info = {"used_crop": False}
    result = pre.uncropper(img, info)
    np.testing.assert_array_equal(result, img)


def test_padder_and_unpadder() -> None:
    """Test the padder and unpadder functions."""
    img = np.zeros((250, 250), dtype=np.uint8)
    padded, pads = pre.padder(img, patch_size=128)
    unpadded = pre.unpadder(padded, pads)
    np.testing.assert_array_equal(unpadded, img)


def test_cropper_and_uncropper() -> None:
    """Test the cropper and uncropper functions."""
    img = np.zeros((256, 256), dtype=np.uint8)
    img[50:200, 50:200] = 255
    cropped, info = pre.cropper(img)
    uncropped = pre.uncropper(cropped, info)
    assert uncropped.shape == img.shape
