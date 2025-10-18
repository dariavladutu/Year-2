"""Unit tests for the analysis module in the CV6 project."""

import io
import logging
import os
import sys

import matplotlib
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
matplotlib.use("Agg")  # prevent Tk errors in headless test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

from src.utils import analysis_utils as au


def test_measure_primary_root_and_tip_with_graph() -> None:
    """Test.

    Test the `measure_primary_root_and_tip_with_graph` function
    from the corresponding module.
    """
    # Create a vertical line to simulate a root structure
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv = 50
    mask[10:90, cv] = 1
    result = au.measure_primary_root_and_tip(mask)
    length, bot_tip, top_tip, smooth, angle, depth, span, path = result
    assert length > 0
    assert bot_tip is not None
    assert top_tip is not None
    assert 0 <= smooth <= 1
    assert isinstance(path, list)


def test_render_full_mask_with_roots_tiff() -> None:
    """Test.

    Test the `render_full_mask_with_roots_tiff` function from the
    corresponding module.
    """
    # Create a dummy mask and root data
    mask = np.zeros((100, 100), dtype=np.uint8)
    lengths = {"a": 12.3}
    tips = {"a": (30, 40)}
    paths = {"a": [(30, 40), (31, 41)]}
    result, tiff_bytes = au.render_full_mask_with_roots_tiff(mask, lengths, tips, paths)
    assert "a" in result
    assert isinstance(tiff_bytes, bytes)


def test_overlay_roots_on_image() -> None:
    """Test.

    Test the `overlay_roots_on_image` function from the
    corresponding module.
    """
    # Create a dummy image
    img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    bytes_data = buf.getvalue()

    tips = {"a": (30, 40)}
    paths = {"a": [(30, 40), (31, 41)]}
    lengths = {"a": 12.3}
    img_out, meas_strs = au.overlay_roots_on_image(bytes_data, tips, paths, lengths)
    assert isinstance(img_out, Image.Image)
    assert "a" in meas_strs


def test_measure_primary_root_and_tip_empty() -> None:
    """Test.

    Test the `measure_primary_root_and_tip_empty` function from
    the corresponding module.
    """
    # Create an empty mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = au.measure_primary_root_and_tip(mask)
    assert isinstance(result, tuple)
    assert result[0] == 0.0


def test_adjust_measurements_to_full() -> None:
    """Test.

    Test the `adjust_measurements_to_full` function from
    the corresponding module.
    """
    # Test with a simple case
    tips = {"a": (10, 10)}
    paths = {"a": [(10, 10), (11, 10)]}
    params = {"x_start": 5, "top_crop": 2}
    t_full, p_full = au.adjust_measurements_to_full(tips, paths, params)
    assert t_full["a"] == (12, 15)
