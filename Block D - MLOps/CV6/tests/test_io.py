"""Unit tests for the io_utils module in the CV6 project.

This module tests the functionality of loading images from bytes,
loading TIFF masks with metadata, and encoding masks to TIFF in base64 format.
"""

import json
import os
import sys

import tifffile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
import io as io_lib

import numpy as np
from PIL import Image

from src.utils import io_utils as io


def test_load_image_from_bytes() -> None:
    """Test.

    Test the `load_image_from_bytes` function from the
    corresponding module.
    """
    img = Image.fromarray(np.ones((50, 50), dtype=np.uint8) * 255)
    buf = io_lib.BytesIO()
    img.save(buf, format="PNG")
    result = io.load_image_from_bytes(buf.getvalue())
    assert result.shape == (50, 50)


def test_load_mask_tif() -> None:
    """Test.

    Test the `load_mask_tif` function from the corresponding module.
    """
    # Create a sample TIFF file with metadata
    mask = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    metadata = {"info": "test"}
    buf = io_lib.BytesIO()
    tifffile.imwrite(buf, mask, description=json.dumps(metadata))
    buf.seek(0)

    # Load the mask and metadata
    loaded_mask, loaded_metadata = io.load_mask_tif(buf)
    assert loaded_mask.shape == (10, 10)
    assert loaded_metadata == metadata


def test_encode_mask_to_tiff_base64() -> None:
    """Test.

    Test the `encode_mask_to_tiff_base64` function from
    the corresponding module.
    """
    mask = np.ones((10, 10), dtype=np.float32)
    metadata = {"info": "test"}
    tiff_bytes, base64_str, data_uri = io.encode_mask_to_tiff_base64(mask, metadata)
    assert isinstance(tiff_bytes, bytes)
    assert base64_str.startswith("S") or len(base64_str) > 10
    assert data_uri.startswith("data:")


def test_encode_mask_to_tiff_base64_extra() -> None:
    """Test.

    Test the `encode_mask_to_tiff_base64` function with a larger mask.
    """
    mask = np.random.rand(20, 20).astype(np.float32)
    metadata = {"info": "extra test"}
    tiff_bytes, base64_str, data_uri = io.encode_mask_to_tiff_base64(mask, metadata)
    assert len(tiff_bytes) > 100
    assert len(base64_str) > 50
    assert data_uri.startswith("data:")
