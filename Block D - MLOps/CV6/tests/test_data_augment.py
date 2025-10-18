"""Unit tests for data augmentation functions in utils.data_augment."""

import numpy as np
import pytest
from PIL import Image
from src.utils import data_augment


def make_patch(val_img, val_mask, shape=(8, 8)):
    """Create a dummy image and mask patch with constant values."""
    img = np.full(shape, val_img, dtype=np.uint8)
    mask = np.full(shape, val_mask, dtype=np.uint8)
    return img, mask


def test_augment_patches_basic():
    """Test basic 3x augmentation: original, flipped, enhanced."""
    img, mask = make_patch(100, 1)
    patches = [(img, mask)]
    augmented = data_augment.augment_patches(patches)
    assert len(augmented) == 3
    np.testing.assert_array_equal(augmented[0][0], img)
    np.testing.assert_array_equal(augmented[0][1], mask)
    np.testing.assert_array_equal(augmented[1][0], np.fliplr(img))
    np.testing.assert_array_equal(augmented[1][1], np.fliplr(mask))
    assert not np.array_equal(augmented[2][0], img)
    np.testing.assert_array_equal(augmented[2][1], mask)


def test_augment_patches_multiple():
    """Test augmentation on multiple image-mask pairs."""
    patches = [make_patch(50, 0), make_patch(200, 2)]
    augmented = data_augment.augment_patches(patches)
    assert len(augmented) == 6
    np.testing.assert_array_equal(augmented[0][0], patches[0][0])
    np.testing.assert_array_equal(augmented[3][0], patches[1][0])


def test_sample_and_merge_background_counts():
    """Test correct counts of clean and noisy samples in merged output."""
    clean = [make_patch(10, 0) for _ in range(10)]
    noisy = [make_patch(20, 1) for _ in range(20)]
    result = data_augment.sample_and_merge_background(
        clean, noisy, clean_ratio=0.4, noisy_ratio=0.6, target_size=10
    )
    assert len(result) == 10
    clean_imgs = [img for img, _ in result if np.all(img == 10)]
    noisy_imgs = [img for img, _ in result if np.all(img == 20)]
    assert len(clean_imgs) == 4
    assert len(noisy_imgs) == 6


def test_sample_and_merge_background_not_enough():
    """Test fallback behavior when not enough patches are available."""
    clean = [make_patch(10, 0) for _ in range(2)]
    noisy = [make_patch(20, 1) for _ in range(2)]
    result = data_augment.sample_and_merge_background(
        clean, noisy, clean_ratio=0.7, noisy_ratio=0.7, target_size=3
    )
    assert len(result) == 4
