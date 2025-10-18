"""Sample test suite for save_patch_dataset function."""
import os
import shutil

import cv2
import numpy as np
import pytest

from src.utils.data_save_output import save_patch_dataset


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary image and mask output directories for testing."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    return str(img_dir), str(mask_dir)


def create_dummy_patch(shape=(8, 8, 3), mask_shape=(8, 8)):
    """Generate a dummy image and binary mask pair for testing."""
    img = np.random.randint(0, 256, shape, dtype=np.uint8)
    mask = np.random.randint(0, 2, mask_shape, dtype=np.uint8) * 255
    return img, mask


def test_save_patch_dataset_creates_files(temp_dirs):
    """Test that save_patch_dataset writes images and masks to disk."""
    img_dir, mask_dir = temp_dirs
    patches = [create_dummy_patch() for _ in range(3)]
    patch_list = [(img, mask) for img, mask in patches]

    save_patch_dataset(patch_list, img_dir, mask_dir, base_prefix="testpatch")

    for idx in range(3):
        img_path = os.path.join(img_dir, f"testpatch_{idx}.png")
        mask_path = os.path.join(mask_dir, f"testpatch_{idx}.tif")

        # Assert files exist
        assert os.path.isfile(img_path)
        assert os.path.isfile(mask_path)

        # Assert files are valid images
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        assert img is not None
        assert mask is not None


def test_save_patch_dataset_empty_list(temp_dirs):
    """Test save_patch_dataset when given an empty patch list."""
    img_dir, mask_dir = temp_dirs
    save_patch_dataset([], img_dir, mask_dir)

    # Output directories should be created and remain empty
    assert os.path.isdir(img_dir)
    assert os.path.isdir(mask_dir)
    assert len(os.listdir(img_dir)) == 0
    assert len(os.listdir(mask_dir)) == 0


def test_save_patch_dataset_custom_prefix(temp_dirs):
    """Test custom base_prefix naming for saved patch outputs."""
    img_dir, mask_dir = temp_dirs
    patch = [create_dummy_patch()]
    save_patch_dataset(patch, img_dir, mask_dir, base_prefix="custom")

    img_path = os.path.join(img_dir, "custom_0.png")
    mask_path = os.path.join(mask_dir, "custom_0.tif")

    # Assert files were saved with custom prefix
    assert os.path.isfile(img_path)
    assert os.path.isfile(mask_path)
