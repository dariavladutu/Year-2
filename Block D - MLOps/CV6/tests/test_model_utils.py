"""Model utilities for building, training, and evaluating models."""
import os

import numpy as np
import pytest
import tensorflow as tf
from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras import backend as K

import src.utils.model_utils as model_utils



# ----------------------------- F1 METRIC TESTS -----------------------------


def test_f1_perfect_score():
    """Test that F1 score returns 1.0 for perfect predictions."""
    y_true = tf.constant([[1, 0, 1, 0]], dtype=tf.float32)
    y_pred = tf.constant([[1, 0, 1, 0]], dtype=tf.float32)
    f1_score = model_utils.f1(y_true, y_pred)
    np.testing.assert_allclose(f1_score.numpy(), 1.0, atol=1e-5)


def test_f1_zero_score():
    """Test that F1 score returns 0.0 when predictions are completely incorrect."""
    y_true = tf.constant([[1, 1, 0, 0]], dtype=tf.float32)
    y_pred = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)
    f1_score = model_utils.f1(y_true, y_pred)
    np.testing.assert_allclose(f1_score.numpy(), 0.0, atol=1e-5)


def test_f1_partial_score():
    """Test F1 score computation for partial matches."""
    y_true = tf.constant([[1, 0, 1, 0]], dtype=tf.float32)
    y_pred = tf.constant([[1, 1, 0, 0]], dtype=tf.float32)
    # TP=1, FP=1, FN=1 => precision=0.5, recall=0.5, F1=0.5
    f1_score = model_utils.f1(y_true, y_pred)
    np.testing.assert_allclose(f1_score.numpy(), 0.5, atol=1e-5)


# ----------------------------- UNET TESTS -----------------------------


def test_build_unet_output_shape():
    """Test that build_unet returns model with correct output shape."""
    model = model_utils.build_unet(input_shape=(64, 64, 1))
    assert isinstance(model, Model)
    assert model.output_shape == (None, 64, 64, 1)


# ------------------------ MOBILENET UNET TESTS ------------------------


@pytest.mark.parametrize("input_shape", [(64, 64, 3), (128, 128, 3)])
def test_build_mobilenet_unet_output_shape(input_shape):
    """Test that MobileNet UNet outputs correct shape for various inputs."""
    model = model_utils.build_mobilenet_unet(input_shape=input_shape)
    assert isinstance(model, Model)
    assert model.output_shape == (None, input_shape[0], input_shape[1], 1)


# ------------------------- CALLBACK UTILITY TEST -------------------------


def test_get_early_stopping_returns_callback():
    """Test that get_early_stopping returns a properly configured callback."""
    cb = model_utils.get_early_stopping()
    assert isinstance(cb, EarlyStopping)
    assert cb.monitor == "val_f1"
    assert cb.patience == 5
    assert cb.restore_best_weights is True
    assert isinstance(cb, EarlyStopping)


# ------------------ MODEL LISTING / PATH / LOAD UTILS ------------------


def test_list_local_models_and_get_model_path_local(tmp_path):
    """Test listing local models and resolving their paths."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "foo.h5").write_bytes(b"dummy")
    (models_dir / "bar.h5").write_bytes(b"dummy")

    # Check model listing
    models = model_utils.list_local_models(str(models_dir))
    ids = {m["id"] for m in models}
    assert ids == {"foo", "bar"}

    # Monkeypatch get_model_path_local to use our temp dir
    orig_path = model_utils.get_model_path_local

    def patched_get_model_path_local(model_id):
        path = models_dir / f"{model_id}.h5"
        if not path.exists():
            raise FileNotFoundError
        return str(path)

    try:
        model_utils.get_model_path_local = patched_get_model_path_local
        assert os.path.exists(model_utils.get_model_path_local("foo"))
        with pytest.raises(FileNotFoundError):
            model_utils.get_model_path_local("not_exist")
    finally:
        model_utils.get_model_path_local = orig_path


