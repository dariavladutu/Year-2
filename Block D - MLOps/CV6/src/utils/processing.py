"""Refine a binary mask.

This module provides a function to refine a binary 
mask by applying morphological closing
and removing small objects, resulting in a cleaner and 
more usable mask for further analysis.
"""

import logging
import cv2
import numpy as np
from skimage.morphology import remove_small_objects

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def refine_mask(
    mask: np.ndarray, 
    kernel_size: int = 7,
    iterations: int = 5
) -> np.ndarray:
    """Refines a binary mask using morphological closing small objects.

    Args:
        mask (np.ndarray): Input binary mask as a 2D NumPy array 
        (0 and 255 or 0 and 1).
        kernel_size (int, optional): Size of the square structuring element used 
        in closing. 
        Defaults to 7.
        iterations (int, optional): Number of iterations for morphological closing. 
        Defaults to 5.

    Returns:
        np.ndarray: Refined binary mask as a 2D uint8 NumPy array 
        with values 0 and 255.
    """
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array.")
    try:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed_mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=iterations
        )
        cleaned_mask = remove_small_objects(
            closed_mask > 0, min_size=30, connectivity=2
        ).astype(np.uint8)
        return (cleaned_mask * 255).astype(np.uint8)
    except Exception as e:
        logging.error(f"Failed to refine mask: {e}")
        raise ValueError(f"Refine mask failed due to: {e}") from e
