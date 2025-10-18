"""Utility functions for reading and writing images and masks."""

# ─── Imports ─────────────────────────────────────────────────────────
import base64
import io
import json
from typing import Dict, Tuple, Union

import numpy as np
import tifffile
from PIL import Image


# ─── Input functions ─────────────────────────────────────────────────
def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a 2-D grayscale image.

    Args:
        image_bytes (bytes): In-memory bytes representing any image format
            readable by Pillow (e.g. PNG, JPEG, TIFF).

    Returns:
        np.ndarray: A **H × W** array of type ``uint8`` containing the
        single-channel (grayscale) pixel data. The original resolution is
        preserved so downstream code can tile or patch the image without
        resampling.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    return np.array(img)


def load_mask_tif(
    mask_src: Union[str, io.BytesIO]
) -> Tuple[np.ndarray, Dict]:
    """Load a TIFF mask and its embedded JSON metadata.

    The metadata is expected to be stored in the first page’s
    ``ImageDescription`` tag and convertible from JSON.

    Args:
        mask_src (str | io.BytesIO): Path to the ``.tif/.tiff`` file **or**
            an in-memory buffer containing the TIFF data.

    Returns:
        Tuple[np.ndarray, Dict]:
            * **mask** – The mask image as a NumPy array (dtype is preserved
              from file).
            * **metadata** – A dictionary parsed from the JSON string found in
              the ``ImageDescription`` tag. An empty dict is returned when the
              tag is absent or empty.
    """
    with tifffile.TiffFile(mask_src) as tif:
        mask = tif.asarray()
        desc = tif.pages[0].tags["ImageDescription"].value

    metadata = json.loads(desc) if desc else {}
    return mask, metadata


# ─── Output functions ────────────────────────────────────────────────
def encode_mask_to_tiff_base64(
    mask: np.ndarray,
    metadata: Dict
) -> Tuple[bytes, str, str]:
    """Serialize a float mask to TIFF and Base64.

    The mask is first converted from ``float32`` in ``[0, 1]`` to ``uint8`` in
    ``[0, 255]``. It is then written to an in-memory TIFF with the supplied
    metadata stored in the ``ImageDescription`` tag. Finally, the TIFF bytes
    are Base64-encoded and returned both as a bare string and as a data-URI.

    Args:
        mask (np.ndarray): 2-D array of floats in the range ``0–1``.
        metadata (dict): Arbitrary JSON-serialisable metadata to embed inside
            the TIFF.

    Returns:
        Tuple[bytes, str, str]:
            * **tiff_bytes** – Raw TIFF bytes.
            * **base64_str** – Base64 text of the TIFF bytes.
            * **data_uri** – The Base64 text prefixed with
              ``"data:application/octet-stream;base64,"`` for direct embedding
              in HTML or JavaScript.

    """
    # 1) Convert float mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # 2) Write to TIFF in memory with metadata
    buf = io.BytesIO()
    tifffile.imwrite(buf, mask_uint8, description=json.dumps(metadata))
    tiff_bytes = buf.getvalue()

    # 3) Base64 encode
    b64 = base64.b64encode(tiff_bytes).decode("utf-8")
    return tiff_bytes, b64, "data:application/octet-stream;base64," + b64


def save_mask_tif(
    image_path: str,
    mask_uint8: np.ndarray,
    metadata: Dict
) -> None:
    """Save a mask as a TIFF file with metadata.

    This function writes a 2-D mask image to a TIFF file, embedding the
    provided metadata as JSON in the ``ImageDescription`` tag.

    Args:
        image_path (str): Path where the TIFF file will be saved.
        mask_uint8 (np.ndarray): 2-D array of type ``uint8`` representing the
            mask image. Values should be in the range ``0–255``.
        metadata (Dict): Arbitrary JSON-serialisable metadata to embed inside
            the TIFF file.

    Raises:
        ValueError: If `mask_uint8` is not a 2-D NumPy array.
    """
    if not isinstance(mask_uint8, np.ndarray) or mask_uint8.ndim != 2:
        raise ValueError("mask_uint8 must be a 2-D NumPy array.")
    tifffile.imwrite(image_path, mask_uint8, description=json.dumps(metadata))
