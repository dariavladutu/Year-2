"""Test cases for the model loading functionality in model_loader."""
import os
import pytest
from unittest import mock

from utils import model_loader


@pytest.fixture
def dummy_model_path(tmp_path):
    """Fixture that creates a temporary dummy model file."""
    model_file = tmp_path / "dummy_model.h5"
    model_file.write_text("dummy content")
    return str(model_file)


def test_load_unet_model_file_not_found():
    """Test that loading a nonexistent model path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as excinfo:
        model_loader.load_unet_model("non_existent_model.h5")

    assert "Model file not found" in str(excinfo.value)


@mock.patch("utils.model_loader.load_model")
def test_load_unet_model_success(mock_load_model, dummy_model_path):
    """Test successful model loading with a mocked load_model and custom f1."""
    dummy_model = mock.Mock()
    mock_load_model.return_value = dummy_model

    with mock.patch("utils.model_loader.f1", autospec=True):
        model = model_loader.load_unet_model(dummy_model_path)

    mock_load_model.assert_called_once_with(
        dummy_model_path, custom_objects={"f1": mock.ANY}
    )
    assert model == dummy_model


@mock.patch("utils.model_loader.load_model", side_effect=RuntimeError("load failed"))
def test_load_unet_model_load_failure(mock_load_model, dummy_model_path):
    """Test that model loading failure raises an OSError with message context."""
    with mock.patch("utils.model_loader.f1", autospec=True):
        with pytest.raises(OSError) as excinfo:
            model_loader.load_unet_model(dummy_model_path)

    assert "Failed to load model from" in str(excinfo.value)
