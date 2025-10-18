"""Model saving module for Keras models."""

from keras.models import Model, load_model


def save_model(model: Model, path: str) -> None:
    """Save a trained Keras model to the specified path.

    Args:
        model (keras.Model): The Keras model to save.
        path (str): Destination path where the model will be saved.
    """
    try:
        model.save(path)
        print(f"[✓] Model saved to: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model to '{path}': {e}") from e


def load_trained_model(path: str) -> Model:
    """Load a trained Keras model from a specified file path.

    Args:
        path (str): Path to the saved Keras model file.

    Returns:
        tensorflow.keras.Model: The loaded Keras model.
    """
    try:
        model = load_model(path)
        print(f"[✓] Model loaded from: {path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{path}': {e}") from e
