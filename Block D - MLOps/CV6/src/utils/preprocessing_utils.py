"""Pre-processing utilities for computer vision tasks."""

# ─── Imports ─────────────────────────────────────────────────────────
from typing import Dict, Tuple
import cv2
import numpy as np

# ─── Pre-processing helpers ──────────────────────────────────────────


def padder(
    image: np.ndarray, 
    patch_size: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad an image to become exact multiples of *patch_size*.

    Args:
        image (np.ndarray): Input grayscale (H × W) or color (H × W × C) image.
        patch_size (int): Desired tile size to be used later, e.g. for
            ``patchify`` or CNN inference.

    Returns:
        Tuple[np.ndarray, Tuple[int, int, int, int]]:
            * **padded_image** – The zero-padded image.
            * **pads** – A 4-tuple ``(top, bottom, left, right)`` describing
              how many pixels were added to each border (useful for `unpadder`).

    Notes:
        *Padding is symmetric*—the extra pixels are split as evenly as possible
        between opposing borders.
    """
    h, w = image.shape[:2]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top = height_padding // 2
    bottom = height_padding - top
    left = width_padding // 2
    right = width_padding - left

    padded = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    return padded, (top, bottom, left, right)


def unpadder(padded: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
    """Remove padding that was previously applied with :func:`padder`.

    Args:
        padded (np.ndarray): Image returned from ``padder``.
        pads (Tuple[int, int, int, int]): The ``(top, bottom, left, right)``
            tuple returned alongside the padded image.

    Returns:
        np.ndarray: Image cropped back to its original spatial resolution.
    """
    top, bottom, left, right = pads
    h, w = padded.shape[:2]
    return padded[top : h - bottom, left : w - right]


def cropper(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Isolate the Petri dish (largest external contour) and square-crop it.

    The routine looks for the largest external contour in *image*.
    If found, it creates a square crop centred on that contour and returns
    metadata that lets you undo the crop.

    Args:
        image (np.ndarray): Input grayscale image.

    Returns:
        Tuple[np.ndarray, Dict]:
            * **cropped** – The cropped square (or the original image if
              nothing was detected).
            * **info** – A dictionary with:
              ``{
                  "original_shape": Tuple[int, int],
                  "used_crop": bool,
                  "x_start": int,
                  "y_start": int,
                  "crop_size": int
              }``

              If ``used_crop`` is *False*, only ``original_shape`` is valid.

    Raises:
        ValueError: If *image* is not 2-D (grayscale) or 3-D (BGR/RGB).

    Notes:
        The crop is **square**, so some background may be included to keep
        aspect ratio 1:1—a common requirement for many CNNs.
    """
    if image.ndim not in {2, 3}:
        raise ValueError("`image` must be 2-D or 3-D (H×W or H×W×C)")

    orig_shape = image.shape
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image, {"original_shape": orig_shape, "used_crop": False}

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    size = max(w, h)  # square side
    cx, cy = x + w // 2, y + h // 2
    xs = max(cx - size // 2, 0)
    ys = max(cy - size // 2, 0)

    cropped = image[ys : ys + size, xs : xs + size]
    return cropped, {
        "original_shape": orig_shape,
        "used_crop": True,
        "x_start": xs,
        "y_start": ys,
        "crop_size": size,
    }


def uncropper(cropped: np.ndarray, info: Dict) -> np.ndarray:
    """Re-insert a square crop back into its original image canvas.

    Args:
        cropped (np.ndarray): The image returned by :func:`cropper`.
        info (Dict): The accompanying metadata dict returned by
            :func:`cropper`.

    Returns:
        np.ndarray: A canvas with shape ``info["original_shape"]`` where
        *cropped* has been pasted at the original location. If no crop was
        actually performed (``info["used_crop"]`` is *False*), the function
        simply returns *cropped* unchanged.
    """
    if not info.get("used_crop", False):
        return cropped

    h0, w0 = info["original_shape"]
    canvas = np.zeros((h0, w0), dtype=cropped.dtype)
    xs, ys, sz = info["x_start"], info["y_start"], info["crop_size"]
    canvas[ys : ys + sz, xs : xs + sz] = cropped
    return canvas
