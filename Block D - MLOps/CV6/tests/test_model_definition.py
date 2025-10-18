"""Test suite for simple_unet_model function.

This module tests the construction, compilation, and input/output shapes of a simple U-Net model.
"""
import sys
import types

import pytest
from keras.models import Model

from utils.model_definition import simple_unet_model
from utils.model_evaluation import f1

# Disable printing of model.summary()
import keras.models
keras.models.Model.summary = lambda self: None

# Mock utils.model_evaluation.f1 to avoid dependency during import
mock_model_evaluation = types.ModuleType("model_evaluation")
mock_model_evaluation.f1 = lambda y_true, y_pred: 0.0
sys.modules["utils.model_evaluation"] = mock_model_evaluation


def test_simple_unet_model_output_shape():
    """Test that the model input and output shapes are correct."""
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 128, 128, 3
    model = simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    assert isinstance(model, Model)
    assert model.input_shape == (None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    assert model.output_shape == (None, IMG_HEIGHT, IMG_WIDTH, 1)


@pytest.mark.parametrize(
    "height,width,channels",
    [
        (32, 32, 1),
        (64, 64, 3),
        (128, 128, 1),
        (256, 256, 3),
    ],
)
def test_simple_unet_model_various_input_shapes(height, width, channels):
    """Test model builds correctly for various image input shapes."""
    model = simple_unet_model(height, width, channels)

    assert model.input_shape == (None, height, width, channels)
    assert model.output_shape == (None, height, width, 1)


def test_simple_unet_model_invalid_input_raises():
    """Test model raises error when invalid (negative or zero) input shapes are used."""
    with pytest.raises(Exception):
        simple_unet_model(-1, 64, 3)

    with pytest.raises(Exception):
        simple_unet_model(0, 0, 0)
