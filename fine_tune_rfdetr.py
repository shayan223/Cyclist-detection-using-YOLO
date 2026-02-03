"""
Fine-tune RF-DETR on a cyclist/pedestrian dataset with data augmentation.

RF-DETR (https://github.com/roboflow/rf-detr) expects COCO-format data:
  dataset_dir/
    train/   -> _annotations.coco.json + images
    valid/   -> _annotations.coco.json + images
    test/    -> _annotations.coco.json + images

This script can convert a YOLO-format dataset (data.yaml + train/images, train/labels, etc.)
to COCO format, then run RF-DETR training. Augmentation is applied by the RF-DETR
training pipeline (built-in); conceptually equivalent to the RT-DETR script's
HSV, geometric, flip, mosaic/mixup, and auto-augment settings.
"""

import os
import json
import shutil
from pathlib import Path

import torch
import yaml
from PIL import Image

# --- Configuration ---
# YOLO dataset (used when converting to COCO). Leave CONFIG_FILE_PATH set if you want auto-conversion.
CONFIG_FILE_PATH = "pdx_cyclist_dataset/data.yaml"
# COCO dataset directory. If None and CONFIG_FILE_PATH is set, YOLO will be converted to COCO here.
DATASET_DIR = None  # e.g. "pdx_cyclist_dataset_coco" or set after converting
# Pre-trained checkpoint: None = COCO base weights; path = resume or further fine-tune
MODEL_PATH = None  # e.g. "./pdx_rfdetr/pdx_rfdetr_finetune/checkpoint_best_total.pth"
# Model size: RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge (see rfdetr docs)
MODEL_SIZE = "RFDETRMedium"

EPOCHS = 10
BATCH = 8
GRAD_ACCUM_STEPS = 2  # effective batch = BATCH * GRAD_ACCUM_STEPS (aim ~16)
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "pdx_rfdetr"
RUN_NAME = "pdx_rfdetr_finetune"

# --- Training behavior (aligned with fine_tune_rtdetr.py) ---
PATIENCE = 5
CHECKPOINT_INTERVAL = 5
EARLY_STOPPING = True
WANDB_DISABLED = True

# --- Data augmentation (conceptual; RF-DETR applies its own augmentation) ---
# Documented here for parity with fine_tune_rtdetr.py. RF-DETR uses built-in
# augmentation (e.g. color, geometry, flips) during training.
# HSV_H, HSV_S, HSV_V, DEGREES, TRANSLATE, SCALE, FLIPLR, MOSAIC, MIXUP, AUTO_AUGMENT, etc.
# are not exposed as train() args; the library handles them internally.


def yolo_to_coco_bbox(yolo_bbox, img_w, img_h):
    """Convert YOLO normalized (class, x_center, y_center, w, h) to COCO (x_min, y_min, w, h) in pixels."""
    _, x_c, y_c, w_n, h_n = yolo_bbox
    w = w_n * img_w
    h = h_n * img_h
    x_min = (x_c - w_n / 2) * img_w
    y_min = (y_c - h_n / 2) * img_h
    return [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)]


def convert_yolo_to_coco(yolo_root: str, data_yaml_path: str, output_dir: str) -> str:
    """
    Convert a YOLO-format dataset (data.yaml + train/images, train/labels, etc.)
    to COCO format expected by RF-DETR. Creates output_dir/train, output_dir/valid,
    output_dir/test with _annotations.coco.json and copied images in each.

    Returns the absolute path to output_dir (the COCO dataset_dir).
    """
    yolo_root = Path(yolo_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg["names"]
    nc = data_cfg["nc"]

    categories = [{"id": i + 1, "name": names[i]} for i in range(nc)]
    name_to_id = {names[i]: i + 1 for i in range(nc)}

    for split, yolo_split_key in [("train", "train"), ("valid", "val"), ("test", "test")]:
        if yolo_split_key not in data_cfg:
            continue
        images_rel = data_cfg[yolo_split_key]  # e.g. "train/images"
        images_dir = yolo_root / images_rel
        labels_rel = images_rel.replace("/images", "/labels")
        labels_dir = yolo_root / labels_rel
        if not images_dir.exists():
            continue

        split_out = output_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        images_list = []
        annotations_list = []
        ann_id = 1

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue
            stem = img_path.stem
            label_path = labels_dir / f"{stem}.txt"
            images_list.append({
                "id": len(images_list) + 1,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            })
            image_id = len(images_list)
            if label_path.exists():
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                        bbox = yolo_to_coco_bbox([cls_id] + coords, w, h)
                        annotations_list.append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": cls_id + 1,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                        })
                        ann_id += 1

        coco = {
            "images": images_list,
            "annotations": annotations_list,
            "categories": categories,
        }
        anno_path = split_out / "_annotations.coco.json"
        with open(anno_path, "w") as f:
            json.dump(coco, f, indent=2)
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                shutil.copy2(img_path, split_out / img_path.name)
    return str(output_dir.resolve())


def fine_tune_rfdetr(
    dataset_dir: str,
    output_dir: str = OUTPUT_DIR,
    run_name: str = RUN_NAME,
    model_size: str = MODEL_SIZE,
    pretrain_weights: str | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
    lr: float = LR,
    device: str = DEVICE,
    patience: int = PATIENCE,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    early_stopping: bool = EARLY_STOPPING,
    wandb_disabled: bool = WANDB_DISABLED,
):
    """Fine-tunes an RF-DETR model on a COCO-format dataset (with built-in augmentation)."""
    if wandb_disabled:
        os.environ["WANDB_DISABLED"] = "true"

    from rfdetr import RFDETRMedium, RFDETRLarge, RFDETRNano, RFDETRSmall

    size_map = {
        "RFDETRNano": RFDETRNano,
        "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRLarge": RFDETRLarge,
    }
    if model_size not in size_map:
        raise ValueError(f"model_size must be one of {list(size_map.keys())}")
    model_class = size_map[model_size]
    model = model_class(pretrain_weights=pretrain_weights) if pretrain_weights else model_class()

    out_path = os.path.join(output_dir, run_name)
    print(f"Training on device: {device}")
    print(f"Dataset (COCO): {dataset_dir}")
    print(f"Output: {out_path}")
    print(f"Epochs: {epochs}, batch_size: {batch_size}, grad_accum_steps: {grad_accum_steps}, lr: {lr}")
    print("Data augmentation: applied by RF-DETR training pipeline (built-in).")

    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        output_dir=out_path,
        device=device,
        early_stopping=early_stopping,
        early_stopping_patience=patience,
        checkpoint_interval=checkpoint_interval,
        resume=None,
    )


if __name__ == "__main__":
    if WANDB_DISABLED:
        os.environ["WANDB_DISABLED"] = "true"

    dataset_dir = DATASET_DIR
    if not dataset_dir and CONFIG_FILE_PATH:
        yolo_root = str(Path(CONFIG_FILE_PATH).parent)
        default_coco = yolo_root + "_coco"
        dataset_dir = default_coco
        print(f"Converting YOLO dataset to COCO at: {dataset_dir}")
        dataset_dir = convert_yolo_to_coco(yolo_root, CONFIG_FILE_PATH, dataset_dir)
        print(f"COCO dataset ready: {dataset_dir}")
    if not dataset_dir:
        raise ValueError(
            "Set DATASET_DIR to your COCO dataset path, or set CONFIG_FILE_PATH to a YOLO data.yaml to auto-convert."
        )

    fine_tune_rfdetr(
        dataset_dir=dataset_dir,
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        model_size=MODEL_SIZE,
        pretrain_weights=MODEL_PATH,
        epochs=EPOCHS,
        batch_size=BATCH,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LR,
        device=DEVICE,
        patience=PATIENCE,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        early_stopping=EARLY_STOPPING,
        wandb_disabled=WANDB_DISABLED,
    )
