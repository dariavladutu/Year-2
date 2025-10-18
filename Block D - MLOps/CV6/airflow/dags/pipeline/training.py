import logging
from model_definition import simple_unet_model
from model_training import train_model

def train_model_pipeline():
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 256, 256, 1
    model = simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    from pipeline.dataset_utils import get_train_val_datasets
    train_data, val_data = get_train_val_datasets()

    model, history = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        output_path="/models/latest_model.h5"
    )

    logging.info("✅ Model retrained.")
