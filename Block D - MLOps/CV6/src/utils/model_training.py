"""Model training module for Keras models."""

from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.models import Model
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_model(
    model: Model, 
    train_data: tf.data.Dataset, 
    val_data: tf.data.Dataset, 
    output_path: str,
    epochs: int = 20, 
    batch_size: int = 16
) -> tuple:
    """Trains a Keras model using the provided training and validation data.

    Compiles the model with Adam optimizer and binary cross-entropy loss,
    and uses ModelCheckpoint and EarlyStopping callbacks.

    Args:
        model (keras.Model): The compiled Keras model to train.
        train_data (tf.data.Dataset or generator): Training dataset.
        val_data (tf.data.Dataset or generator): Validation dataset.
        output_path (str): File path to save the best model.
        epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Size of the training batches. 
        Defaults to 16.

    Returns:
        tuple: A tuple containing:
            - model (keras.Model): The trained Keras model.
            - history (keras.callbacks.History): Training history object.
    """
    try:
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = [
            ModelCheckpoint(
                output_path,
                save_best_only=True,
                monitor="val_loss"
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=5
            ),
        ]

        logging.info("Starting model training...")
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        logging.info("Training completed successfully.")
        return model, history

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise RuntimeError(f"Error during training: {e}") from e
