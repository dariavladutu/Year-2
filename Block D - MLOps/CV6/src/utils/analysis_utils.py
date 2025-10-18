"""Utilities for analyzing plant root images."""

# ─── Imports ──────────────────────────────────────────────────────────

import io
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tifffile
from PIL import Image
from skimage.morphology import skeletonize

# ─── Analysis functions ───────────────────────────────────────────────


def measure_primary_root_and_tip(
    mask: np.ndarray,
) -> Tuple[
    float,
    Optional[Tuple[int, int]],
    Optional[Tuple[int, int]],
    float,
    float,
    float,
    float,
    List[Tuple[int, int]],
]:
    """Re-map tip coordinates and skeleton paths back to the full image frame.

    Args:
        root_tips: Mapping ``{plant_id: (row, col)}`` in **cropped**
            coordinates. Tips that were not detected should have value
            ``None``.
        root_paths: Mapping ``{plant_id: [(row, col), ...]}`` giving each
            primary-root skeleton in cropped coordinates.
        crop_params: Dictionary returned by :pyfunc:`cropper` that contains at
            least ``'x_start'`` and ``'top_crop'`` (offsets of the crop within
            the original image).

    Returns:
        tips_full: Same structure as *root_tips*, but in original
            full-image coordinates.
        paths_full: Same structure as *root_paths*, but each point translated
            to full-image coordinates.
    """
    binary = (mask > 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if labels.max() == 0:
        return 0.0, None, None, 0.0, 0.0, 0.0, 0.0, []

    i = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x, y, w, h, _ = stats[i]
    comp = (labels == i).astype(np.uint8)
    ske = skeletonize(comp[y : y + h, x : x + w].astype(bool)).astype(np.uint8)
    pts = set(map(tuple, np.argwhere(ske)))

    G = nx.Graph()
    for ry, rx in pts:
        for dy, dx in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            nb = (ry + dy, rx + dx)
            if nb in pts:
                G.add_edge((ry, rx), nb, weight=np.hypot(dy, dx))
    if G.number_of_nodes() == 0:
        return 0.0, None, None, 0.0, 0.0, 0.0, 0.0, []

    top = min(G.nodes, key=lambda n: n[0])
    bot = max(G.nodes, key=lambda n: n[0])
    try:
        path = nx.dijkstra_path(G, top, bot, "weight")
        length = nx.dijkstra_path_length(G, top, bot, "weight")
    except nx.NetworkXNoPath:
        return 0.0, None, None, 0.0, 0.0, 0.0, 0.0, []

    full_path = [(ry + y, rx + x) for (ry, rx) in path]
    top_full = full_path[0]
    bot_full = full_path[-1]

    direct_dist = np.linalg.norm(np.array(bot_full) - np.array(top_full))
    smoothness = round(direct_dist / length, 3) if length > 0 else 0.0

    dy = bot_full[0] - top_full[0]
    dx = bot_full[1] - top_full[1]
    angle_deg = round(math.degrees(math.atan2(dy, dx)), 2)

    depth = abs(dy)
    span = abs(max(p[1] for p in full_path) - min(p[1] for p in full_path))

    return (
        round(length, 2),
        bot_full,
        top_full,
        smoothness,
        angle_deg,
        float(depth),
        float(span),
        full_path,
    )


def adjust_measurements_to_full(
    root_tips: Dict[str, Tuple[int, int]],
    root_paths: Dict[str, List[Tuple[int, int]]],
    crop_params: Dict[str, Any],
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[Tuple[int, int]]]]:
    """Re-map tip coordinates and skeleton paths back to the full image frame.

    Args:
        root_tips: Mapping ``{plant_id: (row, col)}`` in **cropped**
            coordinates. Tips that were not detected should have value
            ``None``.
        root_paths: Mapping ``{plant_id: [(row, col), ...]}`` giving each
            primary-root skeleton in cropped coordinates.
        crop_params: Dictionary returned by :pyfunc:`cropper` that contains at
            least ``'x_start'`` and ``'top_crop'`` (offsets of the crop within
            the original image).

    Returns:
        tips_full: Same structure as *root_tips*, but in original
            full-image coordinates.
        paths_full: Same structure as *root_paths*, but each point translated
            to full-image coordinates.
    """
    x0 = crop_params["x_start"]
    y0 = crop_params["top_crop"]

    tips_full = {}
    paths_full = {}

    for key, tip in root_tips.items():
        if tip is None:
            tips_full[key] = None
            paths_full[key] = []
        else:
            r, c = tip
            tips_full[key] = (r + y0, c + x0)

            path = root_paths.get(key, [])
            paths_full[key] = [(r0 + y0, c0 + x0) for r0, c0 in path]

    return tips_full, paths_full


def render_full_mask_with_roots_tiff(
    full_mask: np.ndarray,
    root_lengths: Dict[str, float],
    tips_full: Dict[str, Tuple[int, int]],
    paths_full: Dict[str, List[Tuple[int, int]]],
) -> Tuple[Dict[str, Dict], bytes]:
    """Render a labelled overlay on the full-resolution mask and export as TIFF.

    The function draws each primary-root path and tip on *full_mask*, encodes
    per-plant measurements in the TIFF *ImageDescription* tag, and returns both
    the measurement dictionary and the raw TIFF bytes.

    Args:
        full_mask: Grayscale or binary mask (``H×W``) onto which drawings will
            be overlaid.
        root_lengths: Mapping ``{plant_id: length_px}``.
        tips_full: Mapping ``{plant_id: (row, col)}`` in full-image coords.
        paths_full: Mapping ``{plant_id: [(row, col), ...]}`` in full coords.

    Returns:
        measurements: Nested dict ``{plant_id: {'length_px': float,
        'tip_coord': (row, col) or None}}`` identical to the JSON stored in the
        TIFF.
        tiff_bytes: Byte string that can be saved directly (e.g. with
        ``open('out.tif','wb').write(tiff_bytes)``).
    """
    # 1) build measurements JSON
    measurements = {
        key: {
            "length_px": float(root_lengths[key]),
            "tip_coord": (
                None
                if tips_full[key] is None
                else (int(tips_full[key][0]), int(tips_full[key][1]))
            ),
        }
        for key in root_lengths
    }
    desc = json.dumps(measurements)

    # 2) draw onto a Matplotlib figure at full resolution
    h, w = full_mask.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(full_mask, cmap="gray", vmin=0, vmax=255)
    cmap = plt.get_cmap("tab10")

    for idx, key in enumerate(root_lengths):
        # draw the primary path thicker
        path = paths_full.get(key, [])
        if path:
            y_path, x_path = zip(*path)
            ax.plot(x_path, y_path, color=cmap(idx), linewidth=6)

        # draw a big yellow tip and a bold label
        tip = tips_full.get(key)
        if tip:
            ty, tx = tip
            ax.scatter(
                tx,
                ty,
                s=150,  # bigger marker
                c="yellow",
                edgecolors="black",
                linewidths=2,
                zorder=3,
            )
            ax.text(
                tx + 15,
                ty + 15,
                f"{root_lengths[key]:.1f}px",
                color=cmap(idx),
                fontsize=20,  # large font
                fontweight="bold",
                zorder=4,
            )

    ax.axis("off")
    plt.tight_layout(pad=0)

    # 3) grab RGBA buffer
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape((H, W, 4))
    img = buf[:, :, [1, 2, 3, 0]]  # ARGB→RGBA
    plt.close(fig)

    # 4) write TIFF with ImageDescription
    tiff_buf = io.BytesIO()
    tifffile.imwrite(tiff_buf, img, photometric="rgb", description=desc)
    return measurements, tiff_buf.getvalue()


def overlay_roots_on_image(
    image_bytes: bytes,
    tips_full: Dict[str, Tuple[int, int]],
    paths_full: Dict[str, List[Tuple[int, int]]],
    root_lengths: Dict[str, float],
) -> Tuple[Image.Image, Dict[str, str]]:
    """Overlay root skeletons and tip labels on the original RGB image.

    Args:
        image_bytes: Raw bytes of the original image (any format readable by
            :pyclass:`PIL.Image`).
        tips_full: Mapping ``{plant_id: (row, col)}`` in full-image coords.
        paths_full: Mapping ``{plant_id: [(row, col), ...]}``.
        root_lengths: Mapping ``{plant_id: length_px}``.

    Returns:
        overlay_img: A PIL Image object containing the rendered overlay.
        measurement_strs: Mapping ``{plant_id: '<length>px at (row, col)'}``,
            convenient for APIs that expect plain strings.
    """
    # 1) Load image
    img_arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    h, w, _ = img_arr.shape

    # 2) Create a figure exactly the same size in inches
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(img_arr)
    ax.axis("off")

    # 3) Draw each plant
    measurement_strs: Dict[str, str] = {}
    for __, plant in enumerate(root_lengths):
        path = paths_full.get(plant, [])
        # plot skeleton path
        if path:
            y_coords, x_coords = zip(*path)
            ax.plot(x_coords, y_coords, linewidth=3, label=plant)

        # plot tip and label
        tip = tips_full.get(plant)
        if tip:
            ty, tx = tip
            # bigger yellow tip marker
            ax.scatter(
                tx, ty, s=150, c="yellow", edgecolors="black", linewidths=2, zorder=3
            )
            # larger font
            text = f"{root_lengths[plant]:.1f}px"
            ax.text(tx + 15, ty + 15, text, fontsize=20, fontweight="bold")

            # build the measurement string
            measurement_strs[plant] = f"{root_lengths[plant]:.1f}px at ({ty}, {tx})"

    plt.tight_layout(pad=0)

    # 4) Extract RGB buffer and close figure
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape((h, w, 3))
    plt.close(fig)

    return Image.fromarray(buf), measurement_strs
