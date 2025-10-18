"""Auto-generated module docstring."""

# ─── Imports ──────────────────────────────────────────────────────────

from typing import Any, Dict, List

import cv2
import numpy as np
import tensorflow as tf
from patchify import patchify, unpatchify

from .io_utils import load_image_from_bytes
from .preprocessing_utils import cropper, padder, uncropper, unpadder

# ─── Segmentation utilities ───────────────────────────────────────────


def segment_image(
    model: tf.keras.Model,
    image_bytes: bytes,
    patch_size: int = 256,
    step: int = 128,
) -> Dict[str, Any]:
    """Run a sliding-window U-Net on raw image bytes and return the mask.

    The helper decodes the bytes, isolates the Petri dish, tiles the image,
    feeds every tile through ``model.predict`` and finally stitches the
    probabilities back together so the output mask aligns with the original
    resolution.

    Workflow
    --------
    1. ``load_image_from_bytes`` → grayscale ``np.ndarray``
    2. ``cropper``               → crop dish region
    3. ``padder``                → pad to multiple of ``patch_size``
    4. ``patchify``              → sliding-window tiles (normalised to [0, 1])
    5. ``model.predict``         → per-tile probabilities
    6. ``unpatchify``            → reassemble padded mask
    7. ``unpadder``              → remove padding
    8. ``uncropper``             → restore full-frame coordinates

    Args:
        model: Loaded *tf.keras* model that was trained on patches of size
            ``patch_size``.
        image_bytes: Raw bytes from an image file (e.g. ``UploadFile.read()``).
        patch_size: Edge length of the square tiles used during training and
            inference. Defaults to ``256``.
        step: Sliding-window stride in pixels. A value smaller than
            ``patch_size`` produces overlapping tiles. Defaults to ``128``.

    Returns:
        Dict[str, Any]: A dictionary with three keys:

        * ``mask`` – ``float32`` array of shape ``(H, W)`` with probabilities
          in the range [0, 1].
        * ``crop_info`` – Metadata required to undo the crop.
        * ``pad_info`` – ``Tuple[int, int, int, int]`` describing
          ``(top, bottom, left, right)`` padding that was applied.
    """
    # 1) Decode bytes → 2-D uint8 grayscale
    img = load_image_from_bytes(image_bytes)

    # 2) Crop to Petri dish
    cropped, crop_info = cropper(img)

    # 3) Pad so H and W are divisible by `patch_size`
    padded, pad_info = padder(cropped, patch_size)

    # 4) Patchify & normalise
    patches = patchify(padded, (patch_size, patch_size), step=step)
    n_h, n_w, _, _ = patches.shape
    batch = patches.reshape(-1, patch_size, patch_size, 1).astype("float32") / 255.0

    # 5) Predict all tiles
    preds = model.predict(batch, verbose=0).squeeze()  # (n_h·n_w, H, W)
    preds = preds.reshape(n_h, n_w, patch_size, patch_size)

    # 6) Reassemble padded mask
    full_padded = unpatchify(preds, padded.shape)

    # 7–8) Remove padding and undo crop
    unpadded = unpadder(full_padded, pad_info)
    mask = uncropper(unpadded, crop_info)

    return {
        "mask": mask.astype("float32"),
        "crop_info": crop_info,
        "pad_info": pad_info,
    }


def segment_plants_from_dish(
    mask: np.ndarray,
    num_plants: int = 5,
    min_area: int = 100,
    aspect_ratio_threshold: float = 1.5,
    vertical_start_thresh_ratio: float = 0.3,
    angle_filter_area_thresh: int = 1200,
    min_angle_degrees: float = 65,
    edge_margin_px: int = 250,
) -> List[np.ndarray]:
    """Split a dish-level mask into individual plant masks.

    The function assumes the Petri dish contains *num_plants* evenly spaced
    vertical bands. Connected components are filtered by heuristic thresholds
    and the largest valid component in each band is retained.

    Args:
        mask: Single-channel binary mask of the cropped dish (dtype ``uint8``).
        num_plants: Expected number of plants (vertical bands) in the image.
        min_area: Minimum connected-component area to keep.
        aspect_ratio_threshold: Minimum height-to-width ratio for a component
            to be considered predominantly vertical.
        vertical_start_thresh_ratio: Reject components that start below this
            fraction of the image height.
        angle_filter_area_thresh: Below this area an additional angle filter
            is applied.
        min_angle_degrees: Minimum angle (degrees) between a component’s major
            axis and the horizontal axis.
        edge_margin_px: Horizontal margin (pixels) from left and right edges
            inside which components are discarded.

    Returns:
        list[np.ndarray]: Length-``num_plants`` list in which each element is a
        ``uint8`` binary mask (0/255) for the respective plant band. Empty
        bands contain an all-zero mask.
    """
    h, w = mask.shape
    band_width = w / num_plants

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    plant_masks: List[np.ndarray] = [
        np.zeros_like(mask, dtype=np.uint8) for _ in range(num_plants)
    ]

    for i in range(1, num_labels):  # skip background
        x, y, cw, ch, area = stats[i]
        cx, cy = centroids[i]

        # Area & shape filters
        if area < min_area:
            continue
        aspect_ratio = (ch / cw) if cw else 0
        if aspect_ratio < aspect_ratio_threshold or y > int(
            h * vertical_start_thresh_ratio
        ):
            continue

        # Angle-based filter for small components
        if area < angle_filter_area_thresh:
            angle = np.degrees(np.arctan2(ch, cw))
            too_small_angle = angle < min_angle_degrees
            too_close_left = x < edge_margin_px
            too_close_right = (x + cw) > (w - edge_margin_px)
            if too_small_angle or too_close_left or too_close_right:
                continue

        # Determine plant band
        band_idx = min(int(cx // band_width), num_plants - 1)

        component_mask = (labels == i).astype(np.uint8)
        if np.count_nonzero(component_mask) > np.count_nonzero(plant_masks[band_idx]):
            plant_masks[band_idx] = component_mask

    return [(m * 255).astype(np.uint8) for m in plant_masks]


def merge_segmented_masks(
    segmented: List[np.ndarray],
    crop_params: Dict[str, Any],
) -> np.ndarray:
    """Merge individual plant masks into a single dish-cropped mask.

    Args:
        segmented: List of binary ``uint8`` masks (0/255), all sharing the same
            height and width.
        crop_params: Dictionary with preprocessing metadata (e.g. ``x_start``,
            ``x_end``, ``top_crop``, ``orig_shape``). Only ``orig_shape`` is
            used here; the rest are stored for consistency.

    Returns:
        np.ndarray: Combined ``uint8`` mask (0/255) of identical shape to the
        input masks.
    """
    h_crop, w_crop = segmented[0].shape
    combined = np.zeros((h_crop, w_crop), dtype=np.uint8)
    for m in segmented:
        combined = np.maximum(combined, m)
    return combined


def reconstruct_full_mask(
    cropped_mask: np.ndarray,
    crop_params: Dict[str, Any],
) -> np.ndarray:
    """Paste a cropped mask back into a blank canvas of the original size.

    Args:
        cropped_mask: 2-D ``uint8`` array produced in the dish-cropped
            coordinate space.
        crop_params: Dictionary returned by :func:`cropper` containing at
            least the following keys:

            * ``x_start`` (int): Column index where the crop begins.
            * ``top_crop`` (int): Number of rows removed from the top edge.
            * ``orig_shape`` (tuple[int, int]): ``(H_full, W_full)`` of the
              original image.

    Returns:
        np.ndarray: ``uint8`` mask of shape ``orig_shape`` with
        ``cropped_mask`` inserted at the correct location.
    """
    h_full, w_full = crop_params["orig_shape"]
    x_start = crop_params["x_start"]
    top_crop = crop_params["top_crop"]

    full_mask = np.zeros((h_full, w_full), dtype=np.uint8)
    full_mask[
        top_crop : top_crop + cropped_mask.shape[0],
        x_start : x_start + cropped_mask.shape[1],
    ] = cropped_mask
    return full_mask
