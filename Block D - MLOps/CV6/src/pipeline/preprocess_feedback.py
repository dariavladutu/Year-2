"""This script processes feedback data for training a model."""
from datetime import datetime
import shutil
import subprocess
from flask import json
from pathlib import Path

# Timestamp-based version ID
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
version_id = f"version_{timestamp}"

# Setup versioned output dir
version_root = Path(f"./feedback_patches/{version_id}")
raw_images = version_root / "raw/images"
raw_masks = version_root / "raw/masks"
patch_images = version_root / "train/images"
patch_masks = version_root / "train/masks"

# Create folders
for p in [raw_images, raw_masks, patch_images, patch_masks]:
    p.mkdir(parents=True, exist_ok=True)

# Copy feedback data → versioned raw folder
feedback_dir = Path("./feedback_data")
session_ids = set()

for img in feedback_dir.glob("*.png"):
    shutil.copy(img, raw_images / img.name)

for mask in feedback_dir.glob("*.tif"):
    base = mask.stem.replace("_mask", "")
    new_name = f"{base}_root_mask.tif"
    shutil.copy(mask, raw_masks / new_name)
    session_ids.add(base.split("_")[0])  # crude assumption

# Clean feedback_data folder
shutil.rmtree(feedback_dir)

# Run patchify pipeline into versioned output
subprocess.run([
    "python", "main.py",
    "--images_dir", str(raw_images),
    "--masks_dir", str(raw_masks),
    "--output_images_dir", str(patch_images),
    "--output_masks_dir", str(patch_masks),
    "--patch_size", "256",
    "--step", "128"
], check=True)

# Save metadata
metadata = {
    "version": version_id,
    "generated_at": timestamp,
    "source_sessions": list(session_ids),
    "num_images": len(list(patch_images.glob("*.png")))
}
with open(version_root / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"[✔] Feedback patch version saved → {version_id}")
