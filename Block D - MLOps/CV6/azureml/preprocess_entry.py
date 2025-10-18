import argparse
import shutil
from pathlib import Path
import sys
import os

from preprocessing import (
    load_image_file,
    load_mask_file,
    get_valid_image_mask_pairs,
    cropper,
    padder,
    patchify_pair,
    filter_patches_by_mask_content,
    classify_background_patches,
    augment_patches,
    sample_and_merge_background,
    save_patch_dataset,
    split_dataset,
)


def run_pipeline(args):
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)

    valid_ids = get_valid_image_mask_pairs(image_dir, mask_dir)
    all_augmented = []
    all_background = []

    for sample_id in valid_ids:
        img = load_image_file(image_dir / f"{sample_id}.png")
        mask = load_mask_file(mask_dir / f"{sample_id}_root_mask.tif")
        cropped_img, cropped_mask, _ = cropper(img, mask)
        padded_img, padded_mask, _ = padder(cropped_img, cropped_mask, args.patch_size)
        patches = patchify_pair(
            padded_img, padded_mask, args.patch_size, args.step_size
        )
        root_rich, background = filter_patches_by_mask_content(
            patches, threshold=args.mask_threshold
        )
        all_augmented.extend(augment_patches(root_rich))
        all_background.extend(background)

    clean, noisy = classify_background_patches(all_background)
    background_sample = sample_and_merge_background(
        clean, noisy, args.clean_ratio, args.noisy_ratio, target_size=len(all_augmented)
    )
    full_dataset = all_augmented + background_sample
    train, val, test = split_dataset(full_dataset, tuple(args.split_ratio))

    for split_name, split_data in zip(["train", "val", "test"], [train, val, test]):
        img_out = output_dir / split_name / "images"
        mask_out = output_dir / split_name / "masks"
        save_patch_dataset(split_data, str(img_out), str(mask_out), base_prefix="patch")

    print(f"Done. Total patches: {len(full_dataset)}. Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="AzureML preprocessing entry")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--train_output_dir", type=str, required=True)
    parser.add_argument("--val_output_dir", type=str, required=True)
    parser.add_argument("--test_output_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--step_size", type=int, default=128)
    parser.add_argument("--mask_threshold", type=int, default=150)
    parser.add_argument("--clean_ratio", type=float, default=0.6)
    parser.add_argument("--noisy_ratio", type=float, default=0.4)
    parser.add_argument("--split_ratio", nargs=3, type=float, default=[0.7, 0.2, 0.1])

    args = parser.parse_args()

    # Intermediate output directory
    temp_output_root = Path("outputs")
    temp_output_root.mkdir(parents=True, exist_ok=True)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    pipeline_args = Struct(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=str(temp_output_root),
        patch_size=args.patch_size,
        step_size=args.step_size,
        mask_threshold=args.mask_threshold,
        clean_ratio=args.clean_ratio,
        noisy_ratio=args.noisy_ratio,
        split_ratio=args.split_ratio,
    )

    run_pipeline(pipeline_args)

    shutil.move(temp_output_root / "train", args.train_output_dir)
    shutil.move(temp_output_root / "val", args.val_output_dir)
    shutil.move(temp_output_root / "test", args.test_output_dir)

    print("Saved patches to AzureML outputs.")


if __name__ == "__main__":
    main()
