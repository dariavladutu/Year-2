"""Unit tests for SinglePairGenerator and get_generators in data_generate."""

import pytest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
from src.utils.data_generate import SinglePairGenerator, get_generators


def create_dummy_image(path, size=(64, 64), value=128):
    """Create a dummy grayscale image and save to disk."""
    img = Image.fromarray(np.full(size, value, dtype=np.uint8))
    img.save(path)


def create_dummy_mask(path, size=(64, 64), threshold=127):
    """Create a dummy mask image and save to disk."""
    arr = np.full(size, threshold + 10, dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path)


def test_single_pair_generator_basic(tmp_path):
    """Test generator returns batches with expected shape and content."""
    img_paths = []
    mask_paths = []
    for i in range(4):
        img = tmp_path / f"img_{i}.png"
        mask = tmp_path / f"img_{i}_root_mask.tif"
        create_dummy_image(img)
        create_dummy_mask(mask)
        img_paths.append(img)
        mask_paths.append(mask)

    gen = SinglePairGenerator(
        img_paths, mask_paths, batch_size=2, patch_size=32, shuffle=False
    )
    assert len(gen) == 2

    X, y = gen[0]
    assert X.shape == (2, 32, 32, 1)
    assert y.shape == (2, 32, 32, 1)
    assert np.allclose(X, 128 / 255.0)
    assert np.all(y == 1)


def test_single_pair_generator_shuffle(tmp_path):
    """Test that shuffling image paths does not alter data integrity."""
    img_paths = []
    mask_paths = []
    for i in range(3):
        img = tmp_path / f"img_{i}.png"
        mask = tmp_path / f"img_{i}_root_mask.tif"
        create_dummy_image(img, value=100 + i)
        create_dummy_mask(mask, threshold=120 + i)
        img_paths.append(img)
        mask_paths.append(mask)

    gen = SinglePairGenerator(
        img_paths, mask_paths, batch_size=1, patch_size=16, shuffle=True
    )
    before = list(gen.image_paths)
    gen.on_epoch_end()
    after = list(gen.image_paths)
    assert set(before) == set(after)


def test_get_generators(tmp_path):
    """Test get_generators returns correct generator instances and shapes."""
    root = tmp_path / "dataset"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "test").mkdir(parents=True)
    (root / "masks" / "train").mkdir(parents=True)
    (root / "masks" / "test").mkdir(parents=True)

    # Train set
    for i in range(2):
        img = root / "images" / "train" / f"sample{i}.png"
        mask = root / "masks" / "train" / f"sample{i}_root_mask.tif"
        create_dummy_image(img)
        create_dummy_mask(mask)

    # Test set
    for i in range(1):
        img = root / "images" / "test" / f"sampleX{i}.png"
        mask = root / "masks" / "test" / f"sampleX{i}_root_mask.tif"
        create_dummy_image(img)
        create_dummy_mask(mask)

    train_gen, val_gen, n_train, n_val = get_generators(
        [root], patch_size=16, batch_size=1
    )
    assert isinstance(train_gen, SinglePairGenerator)
    assert isinstance(val_gen, SinglePairGenerator)
    assert n_train == 2
    assert n_val == 1

    X, y = train_gen[0]
    assert X.shape == (1, 16, 16, 1)
    assert y.shape == (1, 16, 16, 1)
