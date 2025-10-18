"""Utilities for building and training models."""

# ─── Imports ─────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import Tensor
from keras import backend as K
import keras
import os
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Dropout, Input, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.applications import MobileNetV2
from keras.layers import Concatenate, concatenate
from typing import Dict , List

# ─── Model functions ─────────────────────────────────────────────────


def f1(y_true: tf.Tensor, y_pred: tf.Tensor) -> Tensor:
    """Batch-wise F1 score for binary segmentation/classification.
    
    The metric is computed from precision and recall calculated on the
    current batch. It is written with TensorFlow graph ops so it can be
    safely used in a compiled `tf.keras` model (e.g. as a metric or in a
    `load_model` call).

    Args:
        y_true: `tf.Tensor`
            Ground-truth binary labels with any shape.
        y_pred: `tf.Tensor`
            Predicted probabilities or logits with the same shape as
            `y_true`.

    Returns:
        tf.Tensor
            A scalar tensor in the range ``[0, 1]`` representing the F1
            score for the batch.
    """

    def recall_m(y_true: tf.Tensor, y_pred: tf.Tensor) -> None:
        """Compute batch recall (sensitivity).

        Args:
            y_true: See parent docstring.
            y_pred: See parent docstring.

        Returns:
            tf.Tensor: Batch recall in ``[0, 1]``.
        """
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return TP / (positives + K.epsilon())

    def precision_m(y_true: tf.Tensor, y_pred: tf.Tensor) -> None:
        """Compute batch precision (positive-predictive value).

        Args:
            y_true: See parent docstring.
            y_pred: See parent docstring.

        Returns:
            tf.Tensor: Batch precision in ``[0, 1]``.
        """
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return TP / (predicted_positives + K.epsilon())

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1_score


f1.__name__ = "f1"


def load_model(model_path: str) -> keras.Model:
    """Load a serialized Keras model that uses the custom ``f1`` metric.

    Args:
        model_path:
            Path to a `.h5` file or SavedModel directory produced by
            `model.save`. The model must have been trained/compiled with the
            same custom F1 function defined above.
    
    Returns:
         keras.Model:
            The deserialized model, ready for inference or further training,
            with the ``f1`` metric registered.

    Raises:
        OSError: If ``model_path`` does not point to a valid Keras model.
    """
    return keras.models.load_model(
        model_path,
        custom_objects={"f1": f1},
    )


def list_local_models(model_dir: str = "models") -> List[Dict[str, str]]:
    """List all local models in the specified directory.
    
    Args:
        model_dir: Directory to search for model files.
        
    Returns:
        List[Dict[str, str]]: A list of dictionaries with model metadata.
    """
    models = []
    for fname in os.listdir(model_dir):
        if fname.endswith(".h5"):
            model_id = fname.replace(".h5", "")
            models.append({
                "id": model_id,
                "version": "local",
                "description": f"Local model: {model_id}"
            })
    return models


def get_model_path_local(model_id: str) -> str:
    """Get the local path to a Keras model file by its ID.
    
    Args:
        model_id: The ID of the model (without file extension).
    """ 
    path = f"./models/{model_id}.h5"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{model_id}' not found locally.")
    return path


def build_unet(input_shape: tuple = (256, 256, 1)) -> keras.Model:
    """Build a U-Net model for binary segmentation tasks.
    
    This function constructs a U-Net architecture suitable for 
    tasks like medical image segmentation.
    
    Args:
        input_shape: Shape of the input images (height, width, channels).
    
    """
    inputs = Input(input_shape)

    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=[f1, 'accuracy']
    )
    print("Compiled metrics:", model.metrics)
    return model


def build_mobilenet_unet(input_shape: tuple = (256, 256, 3)) -> keras.Model:
    """Build a U-Net model using MobileNetV2 as the encoder.
        
    This function constructs a U-Net architecture with MobileNetV2 as the backbone.
    The model is designed for binary segmentation tasks, such as plant root detection.
    
    Args:
        input_shape: Shape of the input images (height, width, channels).
        
    Returns:
        tf.keras.Model: Compiled U-Net model with MobileNetV2 encoder.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Use these encoder layers for skip connections
    skip_names = [
        "block_1_expand_relu", 
        "block_3_expand_relu",
        "block_6_expand_relu",
        "block_13_expand_relu"
    ]
    skips = [base_model.get_layer(name).output for name in skip_names]
    encoder_output = base_model.output

    # Decoder
    x = encoder_output
    for skip in reversed(skips):
        x = Conv2DTranspose(skip.shape[-1], (3, 3), strides=2, padding="same")(x)
        x = Concatenate()([x, skip])
        x = Conv2D(skip.shape[-1], (3, 3), activation="relu", padding="same")(x)

    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same")(x)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[f1]
    )
    return model


# ─── Callback Setup ─────────────────────────────
def get_early_stopping() -> EarlyStopping:
    """Create an EarlyStopping callback for model training.
    
    This callback monitors the validation F1 score and stops training if it
    doesn't improve for a specified number of epochs.
    """
    return EarlyStopping(
        monitor='val_f1', 
        patience=5, 
        restore_best_weights=True,
        mode='max', 
        verbose=1
    )
