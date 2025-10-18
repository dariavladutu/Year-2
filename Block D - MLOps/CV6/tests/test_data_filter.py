"""Unit tests for filtering and classifying root image patches."""

import numpy as np
import pytest
from src.utils.data_filter import (
    filter_patches_by_mask_content,
    classify_background_patches,
)


def make_patch(img_value, mask_value, shape=(16, 16)):
    """Create a dummy image and mask patch with constant values."""
    img_patch = np.full(shape, img_value, dtype=np.uint8)
    mask_patch = np.full(shape, mask_value, dtype=np.uint8)
    return img_patch, mask_patch


def test_filter_patches_by_mask_content_basic():
    """Test that patches are correctly split based on mask content threshold."""
    patch1 = (np.zeros((16, 16)), np.ones((16, 16)))
    mask2 = np.zeros((16, 16))
    mask2[:10, :] = 1
    mask2 = mask2.astype(np.uint8).flatten()
    mask2[:100] = 1
    mask2[100:] = 0
    mask2 = mask2.reshape((16, 16))
    patch2 = (np.zeros((16, 16)), mask2)
    patches = [patch1, patch2]

    root_rich, background = filter_patches_by_mask_content(patches, threshold=150)
    assert len(root_rich) == 1
    assert len(background) == 1
    assert np.array_equal(root_rich[0][1], patch1[1])
    assert np.array_equal(background[0][1], patch2[1])


def test_filter_patches_by_mask_content_threshold_edge():
    """Test filtering when a patch has exactly threshold white pixels."""
    mask = np.zeros((10, 15), dtype=np.uint8)
    mask.flat[:150] = 1
    patch = (np.zeros((10, 15)), mask)
    root_rich, background = filter_patches_by_mask_content([patch], threshold=150)
    assert len(root_rich) == 1
    assert len(background) == 0


def test_filter_patches_by_mask_content_empty():
    """Test filtering behavior when patch list is empty."""
    root_rich, background = filter_patches_by_mask_content([], threshold=10)
    assert root_rich == []
    assert background == []


def test_classify_background_patches_basic():
    """Test clean vs noisy patch classification by standard deviation."""
    clean_patch = make_patch(10, 0)
    noisy_img = np.arange(16 * 16).reshape((16, 16)).astype(np.uint8)
    noisy_patch = (noisy_img, np.zeros((16, 16), dtype=np.uint8))
    patches = [clean_patch, noisy_patch]
    clean, noisy = classify_background_patches(patches, std_threshold=4.5)
    assert clean == [clean_patch]
    assert noisy == [noisy_patch]


def test_classify_background_patches_threshold_edge():
    """Test patch with std exactly at threshold is classified as noisy."""
    img = np.array([0, 9], dtype=np.float32).repeat(128).reshape((16, 16))
    patch = (img, np.zeros((16, 16)))
    clean, noisy = classify_background_patches([patch], std_threshold=np.std(img))
    assert clean == []
    assert noisy == [patch]


def test_classify_background_patches_empty():
    """Test classification behavior when input patch list is empty."""
    clean, noisy = classify_background_patches([], std_threshold=1.0)
    assert clean == []
    assert noisy == []
