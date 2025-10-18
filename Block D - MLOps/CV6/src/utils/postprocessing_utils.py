"""Post-processing utilities for image segmentation masks."""

# ─── Imports ──────────────────────────────────────────────────────────
from typing import Any, Dict, Tuple

import cv2
import numpy as np


# ─── Post-processing functions ────────────────────────────────────────
def threshold_mask(raw_mask: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize a floating-point mask.

    Args:
        raw_mask (np.ndarray): Input mask with values in the continuous
            range ``[0, 1]``.
        threshold (float, optional): Cut-off above which pixels are set to
            1. Pixels ``<= threshold`` become 0. Defaults to ``0.1``.

    Returns:
        np.ndarray: Binary mask (dtype ``uint8``) containing only 0 or 1.
    """
    return (raw_mask > threshold).astype(np.uint8)


def morphological_closing(
    binary: np.ndarray,
    kernel_size: tuple = (3, 3),
    dilate_iter: int = 5,
    erode_iter: int = 3,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
) -> np.ndarray:
    """Refine a binary mask with morphological closing.

    The operation performs **dilation** followed by **erosion** to fill
    small holes and bridge tiny gaps.

    Args:
        binary (np.ndarray): Input binary mask (values 0 or 1 or 0 or 255).
        kernel_size (tuple, optional): Structuring-element size
            ``(height, width)``. Defaults to ``(3, 3)``.
        dilate_iter (int, optional): Number of dilation iterations.
            Defaults to ``5``.
        erode_iter (int, optional): Number of erosion iterations.
            Defaults to ``3``.
        kernel_shape (int, optional): OpenCV constant specifying the
            structuring-element shape (e.g. ``cv2.MORPH_ELLIPSE`` or
            ``cv2.MORPH_RECT``). Defaults to ``cv2.MORPH_ELLIPSE``.

    Returns:
        np.ndarray: Closed mask as ``float32`` array with values 0.0 or 1.0.
    """
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    dilated = cv2.dilate(binary, kernel, iterations=dilate_iter)
    closed = cv2.erode(dilated, kernel, iterations=erode_iter)
    return closed.astype(np.float32)  # 0/1 → 0.0/1.0


def crop_top_and_dish(
    binary: np.ndarray,
    crop_info: Dict[str, Any],
    top_crop_ratio: float = 0.15,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Crop a single-plant mask to the Petri-dish area and remove the noisy top.

    Args:
        binary (np.ndarray): Single-plant binary mask (2-D array).
        crop_info (Dict[str, Any]): Information describing the horizontal
            crop. Must contain:

            * **x_start** (*int*) – Leftmost column index of the dish crop.
            * **crop_size** (*int*) – Width of the square crop.

        top_crop_ratio (float, optional): Fraction of the dish height that
            should be removed from the top (to discard legends, labels,
            etc.). Defaults to ``0.15`` (15 %).

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: A two-item tuple:

        * **cropped** (*np.ndarray*) – Cropped binary mask.
        * **params** (*dict*) – Metadata needed to reconstruct the original
          layout with keys:

          * ``x_start`` – Left index used for horizontal crop.
          * ``x_end`` – Right index used for horizontal crop.
          * ``top_crop`` – Number of top rows removed.
          * ``orig_shape`` – Original mask shape ``(height, width)``.
    """
    x_start = crop_info["x_start"]
    crop_size = crop_info["crop_size"]
    x_end = min(x_start + crop_size, binary.shape[1])

    # Horizontal crop to dish width
    dish = binary[:, x_start:x_end]

    # Vertical crop to remove top noise region
    top_crop = int(dish.shape[0] * top_crop_ratio)
    cropped = dish[top_crop:, :]

    # Store reconstruction parameters
    params = {
        "x_start": x_start,
        "x_end": x_end,
        "top_crop": top_crop,
        "orig_shape": binary.shape,
    }
    return cropped, params
