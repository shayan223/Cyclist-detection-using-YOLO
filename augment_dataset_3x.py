#!/usr/bin/env python3
"""
Pre-generate augmented training samples to triple the training set size.

Reads a YOLO-format dataset (data.yaml + train/images, train/labels) and creates
a new dataset with 3x training samples:
  - Original images (copied as-is)
  - Horizontal flip (image + labels transformed)
  - HSV color jitter (image only; labels unchanged)

Val and test splits are copied as-is. Use the output dataset with fine_tune_rtdetr.py
by setting CONFIG_FILE_PATH to the new data.yaml (e.g. pdx_cyclist_dataset_3x/data.yaml).

Usage:
    python augment_dataset_3x.py --dataset-dir pdx_cyclist_dataset --output-dir pdx_cyclist_dataset_3x
    # Then in fine_tune_rtdetr.py: CONFIG_FILE_PATH = 'pdx_cyclist_dataset_3x/data.yaml'
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

# Match fine_tune_rtdetr.py augmentation ranges for HSV
HSV_H = 0.015   # Hue (± fraction of 180)
HSV_S = 0.7     # Saturation (± fraction of 255)
HSV_V = 0.4     # Value (± fraction of 255)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(images_dir: Path):
    """Return sorted list of image paths in directory."""
    if not images_dir.exists():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_yolo_labels(label_path: Path):
    """Read YOLO format labels: one line per object = class_id x_center y_center width height (normalized)."""
    if not label_path.exists():
        return []
    lines = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                lines.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return lines


def write_yolo_labels(label_path: Path, rows):
    """Write YOLO format labels."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for row in rows:
            f.write(f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")


def labels_hflip(rows):
    """Transform YOLO labels for horizontal flip: x_center -> 1 - x_center."""
    out = []
    for cls, xc, yc, w, h in rows:
        out.append([cls, 1.0 - xc, yc, w, h])
    return out


def image_hflip(img):
    """Horizontal flip image."""
    return cv2.flip(img, 1)


def image_hsv_jitter(img, h=HSV_H, s=HSV_S, v=HSV_V, rng=None):
    """Apply random HSV jitter; labels unchanged. h,s,v are max absolute fractions."""
    if rng is None:
        rng = random
    img = img.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    H = np.clip(H + rng.uniform(-h * 180, h * 180), 0, 180).astype(np.uint8)
    S = np.clip(S + rng.uniform(-s * 255, s * 255), 0, 255).astype(np.uint8)
    V = np.clip(V + rng.uniform(-v * 255, v * 255), 0, 255).astype(np.uint8)
    hsv = np.stack([H, S, V], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def augment_train_split(
    dataset_dir: Path,
    output_dir: Path,
    data_cfg: dict,
    seed: int = 42,
):
    """Create 3x training samples: original, hflip, HSV jitter. Copy val/test as-is."""
    rng = random.Random(seed)
    train_images_rel = data_cfg.get("train")
    if not train_images_rel:
        raise ValueError("data.yaml has no 'train' key")
    train_images_dir = dataset_dir / train_images_rel
    train_labels_dir = dataset_dir / train_images_rel.replace("/images", "/labels")

    out_train_images = output_dir / train_images_rel
    out_train_labels = output_dir / train_images_rel.replace("/images", "/labels")
    out_train_images.mkdir(parents=True, exist_ok=True)
    out_train_labels.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(train_images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images in {train_images_dir}")

    count_orig = count_hflip = count_hsv = 0
    for img_path in image_paths:
        stem = img_path.stem
        ext = img_path.suffix
        label_path = train_labels_dir / f"{stem}.txt"
        labels = read_yolo_labels(label_path)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 1) Original
        out_img = out_train_images / f"{stem}{ext}"
        out_lbl = out_train_labels / f"{stem}.txt"
        shutil.copy2(img_path, out_img)
        if labels:
            write_yolo_labels(out_lbl, labels)
        count_orig += 1

        # 2) Horizontal flip
        stem2 = f"{stem}_hflip"
        out_img2 = out_train_images / f"{stem2}{ext}"
        out_lbl2 = out_train_labels / f"{stem2}.txt"
        cv2.imwrite(str(out_img2), image_hflip(img))
        if labels:
            write_yolo_labels(out_lbl2, labels_hflip(labels))
        count_hflip += 1

        # 3) HSV jitter (labels unchanged)
        stem3 = f"{stem}_hsv"
        out_img3 = out_train_images / f"{stem3}{ext}"
        out_lbl3 = out_train_labels / f"{stem3}.txt"
        cv2.imwrite(str(out_img3), image_hsv_jitter(img, rng=rng))
        if labels:
            write_yolo_labels(out_lbl3, labels)
        count_hsv += 1

    total_train = count_orig + count_hflip + count_hsv
    print(f"Train: {len(image_paths)} originals -> {total_train} samples (3x)")
    return total_train


def copy_split_as_is(dataset_dir: Path, output_dir: Path, split_key: str, images_rel: str):
    """Copy a split (e.g. val, test) without augmentation."""
    if not images_rel:
        return 0
    src_images = dataset_dir / images_rel
    src_labels = dataset_dir / images_rel.replace("/images", "/labels")
    dst_images = output_dir / images_rel
    dst_labels = output_dir / images_rel.replace("/images", "/labels")
    if not src_images.exists():
        return 0
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(src_images)
    for img_path in image_paths:
        shutil.copy2(img_path, dst_images / img_path.name)
        label_path = src_labels / img_path.with_suffix(".txt").name
        if label_path.exists():
            shutil.copy2(label_path, dst_labels / label_path.name)
    print(f"{split_key}: {len(image_paths)} images copied")
    return len(image_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Triple training set by pre-generating augmented samples (original + hflip + HSV)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="pdx_cyclist_dataset",
        help="Path to YOLO dataset (must contain data.yaml and train/images, train/labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pdx_cyclist_dataset_3x",
        help="Output dataset path (will contain 3x train images, same val/test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for HSV jitter.",
    )
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    n_train = augment_train_split(dataset_dir, output_dir, data_cfg, seed=args.seed)
    n_val = copy_split_as_is(dataset_dir, output_dir, "Val", data_cfg.get("val", ""))
    n_test = copy_split_as_is(dataset_dir, output_dir, "Test", data_cfg.get("test", ""))

    out_yaml = {
        "names": data_cfg.get("names", ["cyclist", "pedestrian"]),
        "nc": data_cfg.get("nc", 2),
        "train": data_cfg["train"],
        "val": data_cfg.get("val", "valid/images"),
        "test": data_cfg.get("test", "test/images"),
    }
    out_yaml_path = output_dir / "data.yaml"
    with open(out_yaml_path, "w") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False, default_flow_style=False)

    print(f"\nWrote {out_yaml_path}")
    print(f"Training samples: {n_train} (3x original train size)")
    print("In fine_tune_rtdetr.py set: CONFIG_FILE_PATH = '%s'" % (output_dir / "data.yaml"))


if __name__ == "__main__":
    main()
