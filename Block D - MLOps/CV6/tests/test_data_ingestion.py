"""Test cases for data ingestion utilities in data_ingestion module."""
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from src.utils.data_ingestion import (
    get_valid_image_mask_pairs,
    load_image_file,
    load_image_from_bytes,
    load_mask_tif,
)


def create_test_image_array(shape=(10, 10), value=128):
    """Create a test grayscale image array filled with a constant value."""
    return np.full(shape, value, dtype=np.uint8)


def create_test_image_bytes(shape=(10, 10), value=128):
    """Generate PNG image bytes from a grayscale array."""
    arr = create_test_image_array(shape, value)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_test_tif_bytes(shape=(10, 10), value=1, metadata=None):
    """Create a TIFF image with optional JSON-encoded metadata."""
    arr = np.full(shape, value, dtype=np.uint8)
    buf = io.BytesIO()
    desc = json.dumps(metadata) if metadata else None
    tifffile.imwrite(buf, arr, description=desc)
    buf.seek(0)
    return buf


def test_load_image_from_bytes():
    """Test loading an image from PNG bytes using load_image_from_bytes()."""
    arr = create_test_image_array()
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    loaded = load_image_from_bytes(image_bytes)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == arr.shape
    assert np.array_equal(loaded, arr)


def test_load_image_file(tmp_path):
    """Test loading an image from a file path using load_image_file()."""
    arr = create_test_image_array()
    img = Image.fromarray(arr, mode="L")
    img_path = tmp_path / "test.png"
    img.save(img_path)
    loaded = load_image_file(img_path)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == arr.shape
    assert np.array_equal(loaded, arr)


def test_load_mask_tif_with_metadata():
    """Test loading a TIFF mask with embedded metadata using load_mask_tif()."""
    arr = np.ones((5, 5), dtype=np.uint8)
    metadata = {"foo": "bar"}
    buf = create_test_tif_bytes(arr.shape, 1, metadata)
    mask, meta = load_mask_tif(buf)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == arr.shape
    assert np.array_equal(mask, arr)
    assert meta == metadata


def test_load_mask_tif_without_metadata():
    """Test loading a TIFF mask without metadata using load_mask_tif()."""
    arr = np.ones((5, 5), dtype=np.uint8)
    buf = create_test_tif_bytes(arr.shape, 1, None)
    mask, meta = load_mask_tif(buf)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == arr.shape
    assert np.array_equal(mask, arr)
    assert meta == {"shape": [5, 5]}


def test_get_valid_image_mask_pairs(tmp_path):
    """Test retrieving valid image/mask filename pairs from two folders."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    # Create matching image-mask pairs
    for i in range(3):
        img_name = f"img00{i+1}.png"
        mask_name = f"img00{i+1}_root_mask.tif"
        Image.fromarray(create_test_image_array()).save(images_dir / img_name)
        tifffile.imwrite(masks_dir / mask_name, create_test_image_array())

    # Add unmatched image and mask
    Image.fromarray(create_test_image_array()).save(images_dir / "img999.png")
    tifffile.imwrite(masks_dir / "img888_root_mask.tif", create_test_image_array())

    pairs = get_valid_image_mask_pairs(images_dir, masks_dir)
    assert pairs == ["img001", "img002", "img003"]


def test_get_valid_image_mask_pairs_no_pairs(tmp_path):
    """Test get_valid_image_mask_pairs raises an error if no matches found."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    # Add unmatched image and mask
    Image.fromarray(create_test_image_array()).save(images_dir / "img001.png")
    tifffile.imwrite(masks_dir / "img002_root_mask.tif", create_test_image_array())

    with pytest.raises(ValueError):
        get_valid_image_mask_pairs(images_dir, masks_dir)
