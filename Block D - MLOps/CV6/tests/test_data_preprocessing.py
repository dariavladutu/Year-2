"""# CV6/src/utils/test_data_preprocessing.py"""
import numpy as np
import cv2
import pytest

from src.utils.data_preprocessing import cropper, padder


def make_circle_image_mask(shape=(100, 100), radius=30, center=None):
    """Create a synthetic grayscale image and binary mask with a white circle."""
    if center is None:
        center = (int(shape[1] // 2), int(shape[0] // 2))
    center = (int(center[0]), int(center[1]))
    image = np.zeros(shape, dtype=np.uint8)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(image, center, radius, 255, -1)
    cv2.circle(mask, center, radius, 1, -1)
    return image, mask


def test_cropper_with_circle():
    """Test cropper on an image containing a circular shape."""
    image, mask = make_circle_image_mask()
    cropped_image, cropped_mask, crop_info = cropper(image, mask)

    # The cropped output should be square and include the circle
    assert cropped_image.shape == cropped_mask.shape
    assert cropped_image.shape[0] == cropped_image.shape[1]
    assert crop_info["used_crop"] is True

    # Cropped content should retain the white circle
    assert np.max(cropped_image) == 255
    assert np.max(cropped_mask) == 1


def test_cropper_no_contour():
    """Test cropper behavior when there is no content to crop."""
    image = np.zeros((50, 60), dtype=np.uint8)
    mask = np.zeros((50, 60), dtype=np.uint8)
    cropped_image, cropped_mask, crop_info = cropper(image, mask)

    # Should return original image and mask
    assert np.array_equal(cropped_image, image)
    assert np.array_equal(cropped_mask, mask)
    assert crop_info["used_crop"] is False
    assert crop_info["original_shape"] == (50, 60)


@pytest.mark.parametrize("orig_shape,patch_size", [
    ((100, 100), 32),
    ((45, 60), 16),
    ((128, 128), 64),
    ((31, 31), 8),
])
def test_padder_shapes(orig_shape, patch_size):
    """Test padder output shapes are multiples of the given patch size."""
    image = np.ones(orig_shape, dtype=np.uint8) * 100
    mask = np.ones(orig_shape, dtype=np.uint8)
    padded_image, padded_mask, pad_info = padder(image, mask, patch_size)

    # Output height and width must be divisible by patch size
    h, w = padded_image.shape[:2]
    assert h % patch_size == 0
    assert w % patch_size == 0

    # Image and mask shapes must match
    assert padded_image.shape == padded_mask.shape

    # Padding metadata should be consistent with output shape
    assert pad_info["top"] + pad_info["bottom"] + orig_shape[0] == h
    assert pad_info["left"] + pad_info["right"] + orig_shape[1] == w


def test_padder_no_padding_needed():
    """Test padder when image is already divisible by the patch size."""
    image = np.zeros((32, 32), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    padded_image, padded_mask, pad_info = padder(image, mask, 32)

    # Expecting a new padded size that fits the next patch multiple
    assert padded_image.shape == (64, 64)
    assert padded_mask.shape == (64, 64)

    # Check individual padding amounts
    assert pad_info["top"] == 16
    assert pad_info["bottom"] == 16
    assert pad_info["left"] == 16
    assert pad_info["right"] == 16
