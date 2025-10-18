"""Data generator for image-mask pairs for training and validation."""
# ─── Imports ─────────────────────────────────────────────────────────
import numpy as np
import random
from pathlib import Path
from PIL import Image
from keras.utils import Sequence
from typing import List, Tuple
 

class SinglePairGenerator(Sequence):
    """Data generator for image-mask pairs.
    
    This generator yields batches of image-mask pairs for training or validation.
    """
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        batch_size: int = 32,
        patch_size: int = 256,
        shuffle: bool = True
    ) -> None:
        """Data generator for image-mask pairs.
        
        Args:
            image_paths (List[Path]): List of image file paths.
            mask_paths (List[Path]): List of mask file paths.
            batch_size (int): Number of pairs per batch.
            patch_size (int): Size of the patches to extract.
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        batch_img_paths = self.image_paths[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        batch_mask_paths = self.mask_paths[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]

        images = []
        masks = []

        for img_path, mask_path in zip(
            batch_img_paths,
            batch_mask_paths
        ):
            img = Image.open(img_path).convert("L").resize(
                (self.patch_size, self.patch_size)
            )
            mask = Image.open(mask_path).convert("L").resize(
                (self.patch_size, self.patch_size)
            )

            img_arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=-1)
            mask_np = np.array(mask, dtype=np.uint8)
            mask_arr = np.expand_dims((mask_np > 127).astype(np.uint8), axis=-1)

            images.append(img_arr)
            masks.append(mask_arr)

        return np.array(images), np.array(masks)

    def on_epoch_end(self) -> None:
        """Shuffle the dataset after each epoch."""
        if self.shuffle:
            temp = list(zip(self.image_paths, self.mask_paths))
            random.shuffle(temp)
            self.image_paths, self.mask_paths = zip(*temp)
            self.image_paths = list(self.image_paths)
            self.mask_paths = list(self.mask_paths)
            self.mask_paths = list(self.mask_paths)


def get_generators(
    data_roots: List[Path],
    patch_size: int = 256,
    batch_size: int = 32
) -> Tuple[SinglePairGenerator, SinglePairGenerator, int, int]:
    """Create training and validation data generators from multiple root datasets.
    
    Args:
        data_roots (List[Path]): List of root directories 
        containing 'images' and 'masks'.
        patch_size (int): Size of the patches to extract.
        batch_size (int): Number of image-mask pairs per batch.
        
    Returns:
        Tuple[SinglePairGenerator, SinglePairGenerator, int, int]:
            - Training generator
            - Validation generator
            - Number of training batches
            - Number of validation batches
    """
    def collect_pairs(root_dir: Path, split: str) -> Tuple[List[Path], List[Path]]:
        img_dir = Path(root_dir) / "images" / split
        mask_dir = Path(root_dir) / "masks" / split

        images = sorted(list(img_dir.glob("*.png")))
        masks = sorted([mask for mask in mask_dir.glob("*_root_mask.tif")
                        if mask.stem.replace("_root_mask", "")  
                        + ".png" in [img.name for img in images]])

        matched_images = []
        matched_masks = []

        for mask in masks:
            base = mask.stem.replace("_root_mask", "")
            img = img_dir / f"{base}.png"
            if img.exists():
                matched_images.append(img)
                matched_masks.append(mask)

        return matched_images, matched_masks

    # Aggregate across all root datasets
    train_imgs, train_masks = [], []
    val_imgs, val_masks = [], []

    for root in data_roots:
        imgs, masks = collect_pairs(root, "train")
        train_imgs += imgs
        train_masks += masks

        imgs, masks = collect_pairs(root, "test")
        val_imgs += imgs
        val_masks += masks

    train_gen = SinglePairGenerator(train_imgs, train_masks, batch_size, patch_size)
    val_gen = SinglePairGenerator(val_imgs, val_masks, batch_size, patch_size)

    return train_gen, val_gen, len(train_gen), len(val_gen)
