import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from keras import Model, layers

from src.utils.model_training import train_model


@pytest.fixture
def simple_model():
    """Create a small Keras model for binary classification."""
    inputs = layers.Input(shape=(4,))
    x = layers.Dense(8, activation='relu')(inputs)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model


@pytest.fixture
def dummy_data():
    """Generate a dummy train/validation dataset as tf.data.Dataset."""
    x_train = np.random.rand(32, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, size=(32, 1)).astype(np.float32)
    x_val = np.random.rand(8, 4).astype(np.float32)
    y_val = np.random.randint(0, 2, size=(8, 1)).astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(8)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(8)
    return train_ds, val_ds


def test_train_model_success(simple_model, dummy_data):
    """Test successful training and saving of a model."""
    train_ds, val_ds = dummy_data
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "best_model.keras")
        model, history = train_model(
            simple_model, train_ds, val_ds, output_path, epochs=2, batch_size=8
        )

        assert os.path.exists(output_path)
        assert hasattr(history, "history")
        assert "loss" in history.history
        assert "val_loss" in history.history


def test_train_model_raises_on_invalid_model(dummy_data):
    """Test that training raises RuntimeError when model is not a valid Keras model."""
    train_ds, val_ds = dummy_data

    class DummyModel:
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "best_model.keras")
        with pytest.raises(RuntimeError):
            train_model(
                DummyModel(), train_ds, val_ds, output_path, epochs=1, batch_size=8
            )


def test_train_model_raises_on_invalid_data(simple_model):
    """Test that training raises RuntimeError when training data is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "best_model.keras")
        with pytest.raises(RuntimeError):
            train_model(simple_model, None, None, output_path, epochs=1, batch_size=8)
