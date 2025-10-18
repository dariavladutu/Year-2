"""Inference module test suite for run_pipeline function."""
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest import mock
from utils import inference
from PIL import Image
import cv2

from src.utils.inference import run_pipeline
from src.utils.processing import refine_mask


class DummyModel:
    def predict(self, img_patches):
        # Return a dummy mask (all ones) for each patch
        return np.ones_like(img_patches)

@pytest.fixture
def temp_image_dir():
    temp_dir = tempfile.mkdtemp()
    # Create a dummy image file (as npy for easy loading)
    img = np.ones((512, 512), dtype=np.uint8) * 255
    img_path = os.path.join(temp_dir, "test_image.png")
    # Save as PNG using OpenCV or PIL
    try:
        Image.fromarray(img).save(img_path)
    except ImportError:
        cv2.imwrite(img_path, img)
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def dummy_model():
    return DummyModel()

@pytest.fixture
def patch_dependencies(monkeypatch):
    # Patch cropper, padder, refine_mask, extract_rsa, measure_root, patchify, unpatchify
    monkeypatch.setattr("utils.inference.cropper", lambda path: np.ones((512, 512), dtype=np.uint8))
    monkeypatch.setattr("utils.inference.padder", lambda img, size: img)
    monkeypatch.setattr("utils.inference.refine_mask", lambda mask: mask)
    monkeypatch.setattr("utils.inference.extract_rsa", lambda mask: [np.ones((512, 512), dtype=np.uint8)])
    monkeypatch.setattr("utils.inference.measure_root", lambda root: (None, None, None, 123.0))
    monkeypatch.setattr("utils.inference.patchify", lambda img, shape, step: np.ones((2, 2, shape[0], shape[1]), dtype=np.uint8))
    monkeypatch.setattr("utils.inference.unpatchify", lambda patches, shape: np.ones(shape, dtype=np.uint8))

def test_run_pipeline_creates_csv(temp_image_dir, dummy_model, patch_dependencies):
    output_csv = os.path.join(temp_image_dir, "output.csv")
    inference.run_pipeline(
        input_dir=temp_image_dir,
        model=dummy_model,
        output_csv_path=output_csv,
        patch_size=256
    )
    assert os.path.exists(output_csv)
    df = pd.read_csv(output_csv, index_col=0)
    assert not df.empty
    assert "Length (px)" in df.columns
    # The dummy measure_root returns 123.0
    assert (df["Length (px)"] == 123.0).all()

def test_run_pipeline_raises_for_missing_dir():
    with pytest.raises(FileNotFoundError):
        inference.run_pipeline(
            input_dir="nonexistent_dir",
            model=DummyModel(),
            output_csv_path="dummy.csv",
            patch_size=256
        )

def test_run_pipeline_handles_non_image_files(temp_image_dir, dummy_model, patch_dependencies):
    # Add a non-image file
    with open(os.path.join(temp_image_dir, "not_an_image.txt"), "w") as f:
        f.write("hello")
    output_csv = os.path.join(temp_image_dir, "output2.csv")
    inference.run_pipeline(
        input_dir=temp_image_dir,
        model=dummy_model,
        output_csv_path=output_csv,
        patch_size=256
    )
    assert os.path.exists(output_csv)
    df = pd.read_csv(output_csv, index_col=0)
    assert not df.empty