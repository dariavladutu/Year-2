"""RSA Analysis Tests"""
import numpy as np
import pytest

from src.utils.rsa_analysis import extract_rsa, measure_root


def make_test_mask(shape=(100, 100), fill_rects=None):
    """Create a binary mask with optional filled rectangles for testing."""
    mask = np.zeros(shape, dtype=np.uint8)
    if fill_rects:
        for (x1, y1, x2, y2) in fill_rects:
            mask[y1:y2, x1:x2] = 255
    return mask


def test_extract_rsa_returns_five_parts():
    """Test that extract_rsa splits the image into exactly five vertical parts."""
    mask = make_test_mask((100, 100), fill_rects=[(10, 10, 90, 90)])
    parts = extract_rsa(mask)

    assert isinstance(parts, list)
    assert len(parts) == 5

    for part in parts:
        assert isinstance(part, np.ndarray)
        assert part.shape[0] == 100

    # Total width of all parts should equal original width
    total_width = sum(part.shape[1] for part in parts)
    assert total_width == 100


def test_extract_rsa_filters_small_objects():
    """Test that extract_rsa removes small noise and retains large objects."""
    mask = make_test_mask((50, 50), fill_rects=[(0, 0, 10, 50)])  # large object in part 0
    mask[10:12, 15:17] = 255  # small noise
    mask[20:22, 30:32] = 255  # small noise

    parts = extract_rsa(mask)

    assert np.count_nonzero(parts[0]) > 0
    for i in range(1, 5):
        assert np.count_nonzero(parts[i]) == 0


def test_extract_rsa_empty_mask():
    """Test that extract_rsa returns all-zero parts for an empty input mask."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    parts = extract_rsa(mask)

    assert len(parts) == 5
    for part in parts:
        assert np.count_nonzero(part) == 0


def test_measure_root_empty_mask():
    """Test measure_root returns defaults for an empty mask."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    branch_data, G, longest_path, longest_path_len = measure_root(mask)

    assert branch_data is None
    assert G is None
    assert longest_path is None
    assert longest_path_len == 0


def test_measure_root_invalid_input():
    """Test measure_root handles non-2D input gracefully."""
    mask = np.zeros((50, 50, 3), dtype=np.uint8)
    branch_data, G, longest_path, longest_path_len = measure_root(mask)

    assert branch_data is None
    assert G is None
    assert longest_path is None
    assert longest_path_len == 0


def test_measure_root_simple_line():
    """Test measure_root detects and analyzes a vertical line path."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:18, 10] = 255  # vertical line

    branch_data, G, longest_path, longest_path_len = measure_root(mask)

    assert longest_path_len > 0
    assert longest_path is not None
    assert G is not None
    assert branch_data is not None


def test_measure_root_no_skeleton():
    """Test measure_root handles a single-pixel mask with no skeleton path."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 255  # single point, no skeleton

    branch_data, G, longest_path, longest_path_len = measure_root(mask)

    assert branch_data is None
    assert G is None
    assert longest_path is None
    assert longest_path_len == 0
