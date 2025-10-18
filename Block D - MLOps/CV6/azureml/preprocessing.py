# ─── Imports ─────────────────────────────────────────────────────────
import os
import io
import cv2
import json
import random
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any
from patchify import patchify
import tifffile

# ─── Loading & Pair Matching ─────────────────────────────────────────

def load_image_file(path: Union[str, Path]) -> np.ndarray:
    with open(path, "rb") as f:
        return np.array(Image.open(io.BytesIO(f.read())).convert("L"))


def load_mask_file(path: Union[str, Path]) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def load_mask_tif(mask_src: Union[str, io.BytesIO]) -> Tuple[np.ndarray, Dict]:
    with tifffile.TiffFile(mask_src) as tif:
        mask = tif.asarray()
        try:
            desc = tif.pages[0].tags["ImageDescription"].value
            metadata = json.loads(desc)
        except (KeyError, json.JSONDecodeError, AttributeError):
            metadata = {}
    return mask, metadata


def get_valid_image_mask_pairs(images_dir: Union[str, Path], masks_dir: Union[str, Path]) -> List[str]:
    image_dir = Path(images_dir)
    mask_dir = Path(masks_dir)

    image_basenames = {f.stem for f in image_dir.glob("*.png")}
    mask_basenames = {f.stem.replace("_root_mask", "") for f in mask_dir.glob("*_root_mask.tif")}

    valid_pairs = sorted(image_basenames & mask_basenames)
    if not valid_pairs:
        raise ValueError("No matching image-mask pairs found.")
    return valid_pairs

# ─── Preprocessing ───────────────────────────────────────────────────
def cropper(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    orig_shape = image.shape
    blurred_image = cv2.GaussianBlur(image, (11, 11), 0)
    _, otsu_thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, mask, {"original_shape": orig_shape, "used_crop": False}
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    size = max(w, h)
    x_center = x + w // 2
    y_center = y + h // 2
    x_start = max(0, x_center - size // 2)
    y_start = max(0, y_center - size // 2)
    cropped_img = image[y_start:y_start + size, x_start:x_start + size]
    cropped_mask = mask[y_start:y_start + size, x_start:x_start + size]
    return cropped_img, cropped_mask, {
        "original_shape": orig_shape,
        "used_crop": True,
        "x_start": x_start,
        "y_start": y_start,
        "crop_size": size
    }


def padder(image: np.ndarray, mask: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    h, w = image.shape[:2]
    height_pad = ((h // patch_size) + 1) * patch_size - h
    width_pad  = ((w // patch_size) + 1) * patch_size - w
    top = height_pad // 2
    bottom = height_pad - top
    left = width_pad // 2
    right = width_pad - left
    padded_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    padded_mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_img, padded_mask, {"top": top, "bottom": bottom, "left": left, "right": right}

# ─── Patching ────────────────────────────────────────────────────────
def patchify_pair(image: np.ndarray, mask: np.ndarray, patch_size: int = 256, step: int = 128) -> List[Tuple[np.ndarray, np.ndarray]]:
    assert image.shape == mask.shape, "Image and mask shapes must match."
    patches = []
    img_patches = patchify(image, (patch_size, patch_size), step=step)
    mask_patches = patchify(mask, (patch_size, patch_size), step=step)
    for i in range(img_patches.shape[0]):
        for j in range(img_patches.shape[1]):
            patches.append((img_patches[i, j], mask_patches[i, j]))
    return patches


# ─── Filtering & Augmentation ────────────────────────────────────────
def filter_patches_by_mask_content(patches: List[Tuple[np.ndarray, np.ndarray]], threshold: int = 150) -> Tuple[List, List]:
    root_rich = []
    background = []
    for img_patch, mask_patch in patches:
        if np.sum(mask_patch >= 1) >= threshold:
            root_rich.append((img_patch, mask_patch))
        else:
            background.append((img_patch, mask_patch))
    return root_rich, background


def classify_background_patches(background: List[Tuple[np.ndarray, np.ndarray]], std_threshold: float = 4.5) -> Tuple[List, List]:
    clean, noisy = [], []
    for img_patch, mask_patch in background:
        if np.std(img_patch) < std_threshold:
            clean.append((img_patch, mask_patch))
        else:
            noisy.append((img_patch, mask_patch))
    return clean, noisy


def augment_patches(root_rich: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    augmented = []
    for img, mask in root_rich:
        augmented.append((img, mask))
        augmented.append((cv2.flip(img, 1), cv2.flip(mask, 1)))
        pil_img = Image.fromarray(img).convert("L")
        pil_img = ImageEnhance.Brightness(pil_img).enhance(1.2)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
        augmented.append((np.array(pil_img), mask))
    return augmented


def sample_and_merge_background(clean: List, noisy: List, clean_ratio: float, noisy_ratio: float, target_size: int) -> List:
    clean_count = int(target_size * clean_ratio)
    noisy_count = int(target_size * noisy_ratio)
    random.shuffle(clean)
    random.shuffle(noisy)
    return clean[:clean_count] + noisy[:noisy_count]


# ─── Saving ──────────────────────────────────────────────────────────
def save_patch_dataset(patch_list: List[Tuple[np.ndarray, np.ndarray]], output_img_dir: str, output_mask_dir: str, base_prefix: str = "patch") -> None:
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    for idx, (img, mask) in enumerate(patch_list):
        cv2.imwrite(os.path.join(output_img_dir, f"{base_prefix}_{idx}.png"), img)
        cv2.imwrite(os.path.join(output_mask_dir, f"{base_prefix}_{idx}.tif"), mask)


def split_dataset(data: List, ratios: Tuple[float, float, float]) -> Tuple[List, List, List]:
    random.shuffle(data)
    n = len(data)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test

# ─── Post-Processing ─────────────────────────────────────────────────
def threshold_mask(raw_mask: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    return (raw_mask > threshold).astype(np.uint8)

def morphological_closing(binary: np.ndarray, kernel_size=(3, 3), dilate_iter=5, erode_iter=3, kernel_shape=cv2.MORPH_ELLIPSE) -> np.ndarray:
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    closed = cv2.erode(cv2.dilate(binary, kernel, dilate_iter), kernel, erode_iter)
    return closed.astype(np.float32)

def crop_top_and_dish(binary: np.ndarray, crop_info: Dict[str, Any], top_crop_ratio: float = 0.15) -> Tuple[np.ndarray, Dict[str, Any]]:
    x_start = crop_info["x_start"]
    crop_size = crop_info["crop_size"]
    x_end = min(x_start + crop_size, binary.shape[1])
    dish = binary[:, x_start:x_end]
    top_crop = int(dish.shape[0] * top_crop_ratio)
    cropped = dish[top_crop:, :]
    return cropped, {
        "x_start": x_start,
        "x_end": x_end,
        "top_crop": top_crop,
        "orig_shape": binary.shape,
    }