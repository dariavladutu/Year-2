"""Unit tests for patchify_pair function in data_patchify module."""
import numpy as np
import pytest
from patchify import patchify
from src.utils.data_patchify import patchify_pair


def test_patchify_pair_basic():
    """Test basic functionality of patchify_pair."""
    # Create a simple 4x4 image and mask, patch_size=2, step=2
    image = np.arange(16).reshape(4, 4)
    mask = np.arange(16).reshape(4, 4)
    patches = patchify_pair(image, mask, patch_size=2, step=2)
    # Should get 4 patches (2x2 grid)
    assert len(patches) == 4
    for img_patch, mask_patch in patches:
        assert img_patch.shape == (2, 2)
        assert mask_patch.shape == (2, 2)
        # Patches should be aligned
        np.testing.assert_array_equal(img_patch, mask_patch)


def test_patchify_pair_step_smaller_than_patch():
    """Test when step is smaller than patch size."""
    # 5x5 image, patch_size=3, step=1
    image = np.ones((5, 5), dtype=np.uint8)
    mask = np.ones((5, 5), dtype=np.uint8) * 2
    patches = patchify_pair(image, mask, patch_size=3, step=1)
    # (5-3)//1 + 1 = 3, so 3x3 = 9 patches
    assert len(patches) == 9
    for img_patch, mask_patch in patches:
        assert img_patch.shape == (3, 3)
        assert mask_patch.shape == (3, 3)
        assert np.all(img_patch == 1)
        assert np.all(mask_patch == 2)


def test_patchify_pair_assert_shape():
    """Test assertion of shape compatibility."""
    image = np.zeros((8, 8))
    mask = np.zeros((8, 7))
    with pytest.raises(AssertionError):
        patchify_pair(image, mask, patch_size=2, step=2)


def test_patchify_pair_non_square():
    """Test non-square image and mask."""
    image = np.arange(24).reshape(4, 6)
    mask = np.arange(24).reshape(4, 6)
    patches = patchify_pair(image, mask, patch_size=2, step=2)
    # (4-2)//2+1 = 2, (6-2)//2+1 = 3, so 2x3 = 6 patches
    assert len(patches) == 6
    for img_patch, mask_patch in patches:
        assert img_patch.shape == (2, 2)
        assert mask_patch.shape == (2, 2)


def test_patchify_pair_patch_size_equals_image():
    """Test when patch size equals image size."""
    image = np.random.rand(8, 8)
    mask = np.random.rand(8, 8)
    patches = patchify_pair(image, mask, patch_size=8, step=8)
    assert len(patches) == 1
    img_patch, mask_patch = patches[0]
    np.testing.assert_array_equal(img_patch, image)
    np.testing.assert_array_equal(mask_patch, mask)

