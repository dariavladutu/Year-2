"""Inference module for root length measurement pipeline."""

import logging
import os

import numpy as np
import pandas as pd
from patchify import patchify, unpatchify

from utils.processing import refine_mask
from utils.rsa_analysis import extract_rsa, measure_root
from utils.preprocessing_utils import cropper, padder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_pipeline(input_dir: str, model: object, 
                 output_csv_path: str, patch_size: int) -> None:
    """Image processing pipeline to extract root length measurements.

    This function reads images from a directory, preprocesses them using 
    padding and patching,
    applies a segmentation model, refines the output masks, 
    extracts root structures, and measures
    the longest path (root length) for each plant. 
    The results are saved as a CSV.

    Args:
        input_dir (str): Directory containing input image files.
        model (keras.Model or compatible): Trained model used for 
        root segmentation.
        output_csv_path (str): Path where the output CSV file with 
        measurements will be saved.
        patch_size (int, optional): Size of the square patches 
        used for model prediction. Defaults to 256.
        
    Raises:
        FileNotFoundError: If the input directory does not exist.
        RuntimeError: If any step of the pipeline fails.
        
    Returns:
        None

    Saves:
        A CSV file with plant IDs and their corresponding root lengths in pixels.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    results = []
    try:
        for image_filename in os.listdir(input_dir):
            if not image_filename.lower().endswith((
                '.png',
                '.jpg',
                '.jpeg',
                '.bmp',
                '.tif',
                '.tiff'
            )):
                continue
            image_path = os.path.join(input_dir, image_filename)
            cropped_image = cropper(image_path)
            padded_image = padder(cropped_image, patch_size)

            img_patches = patchify(
                padded_image,
                (patch_size, patch_size),
                step=patch_size
            )
            expected_shape = (
                padded_image.shape[0] // patch_size,
                padded_image.shape[1] // patch_size,
            )
            img_patches = img_patches.reshape(-1, patch_size, patch_size)

            root_mask = model.predict(img_patches)
            root_mask = root_mask.reshape(
                expected_shape[0], expected_shape[1], patch_size, patch_size
            )
            root_mask = unpatchify(root_mask, padded_image.shape[:2])

            # Mask cleaning
            root_mask[:, :300] = 0
            root_mask[:, -250:] = 0
            root_mask[:400, :] = 0
            root_mask[-400:, :] = 0

            root_mask = refine_mask(root_mask)
            roots = extract_rsa(root_mask)

            for i, root in enumerate(roots):
                root = np.array(root, dtype=np.uint8)
                branch_data, G, longest_path, longest_path_len = measure_root(root)

                base_filename = os.path.splitext(image_filename)[0]
                plant_id = f"{base_filename}_plant_{i+1}"

                logging.info(f"Plant ID: {plant_id}, Length: {longest_path_len}")

                results.append({"Plant ID": plant_id, "Length (px)": longest_path_len})
    except Exception as e:
        logging.error(f"Error during pipeline execution: {e}")
        raise

    df = pd.DataFrame(results)
    df.set_index("Plant ID", inplace=True)
    df.to_csv(output_csv_path, index=True)
    logging.info(f"Pipeline completed. Results saved to {output_csv_path}")
    logging.info(df.describe())
