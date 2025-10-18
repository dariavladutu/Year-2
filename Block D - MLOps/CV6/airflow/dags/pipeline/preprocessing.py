""" Preprocessing script for new data.
This script patchifies new images and masks, saving them to the specified directories.
"""
import logging
from pathlib import Path
from utils.data_patchify import patchify_and_save

def preprocess_new_data():
    input_dir = Path("/data/new_images")
    output_img_dir = Path("/data/processed/images")
    output_mask_dir = Path("/data/processed/masks")
    patchify_and_save(
        images_dir=input_dir,
        masks_dir=input_dir,
        output_images_dir=output_img_dir,
        output_masks_dir=output_mask_dir
    )
    logging.info("✅ Preprocessing completed.")
