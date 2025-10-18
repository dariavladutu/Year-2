import shutil
import logging
from pathlib import Path

def merge_datasets():
    processed_data = Path("/data/processed")
    original_data = Path("/data/training")

    for folder in ["images", "masks"]:
        dest_dir = original_data / folder / "train"
        src_dir = processed_data / folder
        for file in src_dir.glob("*.png"):
            shutil.copy(file, dest_dir / file.name)

    logging.info("✅ Dataset merge completed.")

def get_train_val_datasets():
    # Placeholder: implement tf.data or generator loading logic
    raise NotImplementedError("Define how to load your train and validation data")
