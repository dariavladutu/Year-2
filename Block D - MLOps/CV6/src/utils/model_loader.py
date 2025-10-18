"""Model Loader for U-Net in Keras."""

import logging
import os
import keras
from keras.models import load_model
from utils.model_evaluation import f1  # Ensure custom metric is available

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_unet_model(model_path: str) -> keras.Model:
    """Load a pre-trained U-Net model.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        keras.Model: Loaded U-Net model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        OSError: If model loading fails.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = load_model(model_path, custom_objects={"f1": f1})
        logging.info(f"Loaded model from: {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise OSError(f"Failed to load model from {model_path}: {e}")
