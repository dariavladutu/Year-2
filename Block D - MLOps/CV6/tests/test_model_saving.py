"""Test cases for saving and loading Keras models."""
import os
import tempfile

import pytest
from keras import Sequential
from keras.layers import Dense
from keras.models import Model

from src.utils.model_saving import save_model, load_trained_model


def create_simple_model() -> Model:
    """Create a small Keras model for testing save/load functionality."""
    model = Sequential([
        Dense(4, activation='relu', input_shape=(2,)),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def test_save_and_load_model(tmp_path):
    """Test saving and loading a model successfully."""
    model = create_simple_model()
    save_path = tmp_path / "test_model.keras"

    save_model(model, str(save_path))
    assert os.path.exists(save_path)

    loaded_model = load_trained_model(str(save_path))
    assert isinstance(loaded_model, Model)

    # Check that loaded model has the same number of layers
    assert len(model.layers) == len(loaded_model.layers)


def test_save_model_invalid_path():
    """Test save_model raises a RuntimeError for an invalid directory path."""
    model = create_simple_model()
    invalid_path = "/invalid_dir/test_model.keras"

    with pytest.raises(RuntimeError) as excinfo:
        save_model(model, invalid_path)

    assert "Failed to save model" in str(excinfo.value)


def test_load_trained_model_invalid_path():
    """Test load_trained_model raises a RuntimeError for a missing file path."""
    invalid_path = "/invalid_dir/nonexistent_model.keras"

    with pytest.raises(RuntimeError) as excinfo:
        load_trained_model(invalid_path)

    assert "Failed to load model" in str(excinfo.value)
