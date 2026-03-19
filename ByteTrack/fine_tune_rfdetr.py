from __future__ import annotations

"""
Fine-tune RF-DETR on the PDX cyclist/pedestrian dataset.

────────────────────────────────────────────────────────────────────────────────
STANDARD vs. CLASS-PRESERVING fine-tuning
────────────────────────────────────────────────────────────────────────────────

Standard fine-tune (default)
  Replaces the 80-class COCO head with a 2-class head.
  Fast, best cyclist/pedestrian accuracy, but loses all other COCO classes.
  Use this if you only care about cyclist + pedestrian detection.

Class-preserving fine-tune (--preserve-classes)
  Goal: add cyclist detection on top of the base model while keeping cars,
  buses, people, etc. working as well as possible.

  The fundamental problem: without COCO training images in the mix, the model
  has no examples of the 78 other classes and will gradually unlearn them.
  The levers we use to minimise this:

  1. FREEZE THE ENCODER (lr_encoder ≈ 0)
     The backbone (feature extractor) stays frozen. Only the transformer
     decoder head adapts. This is the single most important lever — it
     prevents the backbone representations from drifting away from what the
     full COCO head was built on.

  2. CLASS REMAPPING onto COCO IDs
     Instead of creating new class IDs 0/1, we remap:
       "cyclist"    → COCO bicycle  (ID 1)
       "pedestrian" → COCO person   (ID 0)
     The classification head starts from weights already tuned for these
     semantically similar classes, so less adaptation is needed.
     NOTE: this still produces a 2-class output head — it does not preserve
     the other 78 classes automatically.

  3. CONSERVATIVE HYPERPARAMETERS
     Low LR (1e-5), few epochs (40), gentle augmentation, reduced mosaic.
     Smaller weight updates = less forgetting.

  4. COCO DATA MIXING (--coco-data, most effective)
     If you supply a COCO-format directory, images from it are merged into
     every training epoch alongside your cyclist images. This directly
     prevents forgetting by keeping all 80 classes active in every batch.
     Without this, you will still lose the other 78 classes — the above
     techniques slow the decay but do not stop it entirely.

  Recommended workflow:
    a) Try --preserve-classes without --coco-data first (fast, good cyclist
       accuracy, some degradation on other classes).
    b) If other-class performance matters, add --coco-data pointing to a
       small COCO subset (even 2 000 images makes a big difference).

────────────────────────────────────────────────────────────────────────────────
Segmentation note
────────────────────────────────────────────────────────────────────────────────
rfdetr-seg-medium requires polygon mask annotations.
Our dataset has bounding boxes only → use --bbox (default).
To fine-tune the seg model, export your dataset from Roboflow as
"COCO Segmentation JSON" and pass the folder with --data.

────────────────────────────────────────────────────────────────────────────────
Usage examples
────────────────────────────────────────────────────────────────────────────────
# Standard 2-class fine-tune (fastest, best cyclist accuracy)
python ByteTrack/fine_tune_rfdetr.py

# Class-preserving (freeze encoder, remap to COCO IDs, conservative settings)
python ByteTrack/fine_tune_rfdetr.py --preserve-classes

# Class-preserving + COCO data mixing (best of both worlds)
python ByteTrack/fine_tune_rfdetr.py --preserve-classes --coco-data ../coco_subset

# Resume an interrupted run
python ByteTrack/fine_tune_rfdetr.py --resume runs/rfdetr-medium_res560/checkpoint.pth
"""

import argparse
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

import torch

# ── rfdetr ────────────────────────────────────────────────────────────────────
try:
    from rfdetr import RFDETRMedium, RFDETRSmall, RFDETRLarge
    try:
        from rfdetr import RFDETRSegMedium, RFDETRSegSmall, RFDETRSegLarge
        _SEG_AVAILABLE = True
    except ImportError:
        _SEG_AVAILABLE = False
except ImportError as e:
    raise SystemExit("rfdetr not installed:  pip install rfdetr") from e

# ── YOLO→COCO converter from RF_DETR/ ─────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT / "RF_DETR"))
try:
    from convert_yolo_to_rfdetr import convert_yolo_to_rfdetr
    _CONVERTER_AVAILABLE = True
except ImportError:
    _CONVERTER_AVAILABLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model maps ────────────────────────────────────────────────────────────────
BBOX_MODELS = {"small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}
SEG_MODELS  = {
    "small":  RFDETRSegSmall  if _SEG_AVAILABLE else None,
    "medium": RFDETRSegMedium if _SEG_AVAILABLE else None,
    "large":  RFDETRSegLarge  if _SEG_AVAILABLE else None,
}

# ── COCO class IDs for our two classes ────────────────────────────────────────
# Used when --preserve-classes remaps our labels onto the COCO taxonomy so the
# classification head inherits pretrained weights for these semantically
# similar classes rather than starting from a randomly-initialised neuron.
COCO_REMAP = {
    "cyclist":    {"id": 1,  "name": "bicycle",  "supercategory": "vehicle"},
    "pedestrian": {"id": 0,  "name": "person",   "supercategory": "person"},
}

# ── Augmentation profiles ──────────────────────────────────────────────────────
_STANDARD_AUGMENT = {
    "mosaic":       1.0,
    "mixup":        0.2,
    "perspective":  0.0008,
    "close_mosaic": 10,
    "degrees":      2.0,
    "translate":    0.1,
    "scale":        0.6,
    "fliplr":       0.5,
    "flipud":       0.0,
}

# Conservative: reduced mosaic/mixup to limit distribution shift when
# the encoder is frozen and only the head is adapting.
_PRESERVE_AUGMENT = {
    "mosaic":       0.5,
    "mixup":        0.0,
    "perspective":  0.0,
    "close_mosaic": 5,
    "degrees":      1.0,
    "translate":    0.1,
    "scale":        0.4,
    "fliplr":       0.5,
    "flipud":       0.0,
}


def _supported_kwargs(model, recipe: dict) -> dict:
    accepted    = set(inspect.signature(model.train).parameters.keys())
    supported   = {k: v for k, v in recipe.items() if k in accepted}
    unsupported = [k for k in recipe if k not in accepted]
    if unsupported:
        print(f"  Note: augmentation keys not supported by this rfdetr build (skipped): {unsupported}")
    return supported


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _find_yaml(path: Path) -> Path:
    if path.is_file():
        return path
    yaml_path = path / "data.yaml"
    if yaml_path.exists():
        return yaml_path
    candidates = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"No unique data.yaml found in {path}. Pass the yaml file directly.")


def _is_coco_dir(path: Path) -> bool:
    return (path / "train" / "_annotations.coco.json").exists()


def _convert_yolo(data_arg: str, output_dir: Path, seg: bool) -> Path:
    """Convert YOLO dataset to COCO if needed. Returns COCO dir path."""
    path = Path(data_arg).resolve()

    if _is_coco_dir(path):
        print(f"Using existing COCO dataset: {path}")
        return path

    if seg:
        raise SystemExit(
            "\n[ERROR] Segmentation fine-tuning requires COCO JSON with polygon masks.\n"
            "Your dataset has bounding boxes only.\n\n"
            "  → Use --bbox (default) for detection fine-tuning.\n"
            "  → Or annotate with SAM in Roboflow, export as 'COCO Segmentation JSON'.\n"
        )

    if not _CONVERTER_AVAILABLE:
        raise SystemExit(
            "YOLO→COCO converter not found (expected RF_DETR/convert_yolo_to_rfdetr.py).\n"
            "Pass an already-converted COCO directory with --data instead."
        )

    yaml_path = _find_yaml(path)
    print(f"Converting YOLO dataset → COCO: {yaml_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_yolo_to_rfdetr(yaml_path, output_dir)
    return output_dir


def _remap_coco_ids(coco_dir: Path, remapped_dir: Path) -> Path:
    """
    Rewrite _annotations.coco.json files so that our class IDs match the COCO
    taxonomy (cyclist→bicycle/1, pedestrian→person/0). This allows the
    classification head to start from COCO-pretrained weights for these neurons
    rather than a randomly initialised new class.
    """
    if _is_coco_dir(remapped_dir):
        print(f"Using cached remapped dataset: {remapped_dir}")
        return remapped_dir

    print("Remapping class IDs to COCO taxonomy (cyclist→bicycle/1, pedestrian→person/0)...")
    remapped_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        src_split = coco_dir / split
        if not src_split.exists():
            continue
        dst_split = remapped_dir / split
        dst_split.mkdir(parents=True, exist_ok=True)

        ann_path = src_split / "_annotations.coco.json"
        if not ann_path.exists():
            continue

        with ann_path.open() as f:
            coco = json.load(f)

        # Build old_id → new COCO id mapping from original category names
        old_to_new: dict[int, int] = {}
        new_categories = []
        for cat in coco.get("categories", []):
            name_lower = cat["name"].lower()
            if name_lower in COCO_REMAP:
                new_cat = COCO_REMAP[name_lower].copy()
                old_to_new[cat["id"]] = new_cat["id"]
                if not any(c["id"] == new_cat["id"] for c in new_categories):
                    new_categories.append(new_cat)
            else:
                # Keep unmapped classes as-is (offset to avoid collision)
                new_id = cat["id"] + 100
                old_to_new[cat["id"]] = new_id
                new_categories.append({**cat, "id": new_id})

        # Rewrite annotation category_ids
        new_annotations = []
        for ann in coco.get("annotations", []):
            old_cat = ann.get("category_id")
            if old_cat not in old_to_new:
                continue
            new_annotations.append({**ann, "category_id": old_to_new[old_cat]})

        new_coco = {**coco, "categories": new_categories, "annotations": new_annotations}
        with (dst_split / "_annotations.coco.json").open("w") as f:
            json.dump(new_coco, f, indent=2)

        # Copy images (symlink if possible to save disk)
        for img in src_split.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                dst = dst_split / img.name
                if not dst.exists():
                    try:
                        dst.symlink_to(img)
                    except (OSError, NotImplementedError):
                        shutil.copy2(img, dst)

        print(f"  {split}: {len(new_annotations)} annotations, categories: "
              f"{[c['name'] for c in new_categories]}")

    return remapped_dir


def _merge_coco_datasets(primary: Path, secondary: Path, merged: Path) -> Path:
    """
    Merge a secondary COCO dataset (e.g. a COCO subset) into the primary one.
    Images from both appear in the combined _annotations.coco.json so the
    model sees all classes every epoch.
    """
    if _is_coco_dir(merged):
        print(f"Using cached merged dataset: {merged}")
        return merged

    print(f"Merging COCO data from {secondary} into training set...")
    merged.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid"):
        src1 = primary   / split
        src2 = secondary / split
        dst  = merged    / split
        dst.mkdir(parents=True, exist_ok=True)

        if not src1.exists():
            continue

        with (src1 / "_annotations.coco.json").open() as f:
            coco1 = json.load(f)

        if src2.exists() and (src2 / "_annotations.coco.json").exists():
            with (src2 / "_annotations.coco.json").open() as f:
                coco2 = json.load(f)
        else:
            coco2 = {"images": [], "annotations": [], "categories": []}

        # Offset IDs in coco2 to avoid collision with coco1
        max_img_id = max((i["id"] for i in coco1["images"]), default=0)
        max_ann_id = max((a["id"] for a in coco1["annotations"]), default=0)

        new_images, new_anns = [], []
        img_id_map: dict[int, int] = {}
        for img in coco2["images"]:
            new_id = img["id"] + max_img_id
            img_id_map[img["id"]] = new_id
            new_images.append({**img, "id": new_id})
        for ann in coco2["annotations"]:
            new_anns.append({
                **ann,
                "id":       ann["id"] + max_ann_id,
                "image_id": img_id_map[ann["image_id"]],
            })

        # Merge categories (union by name)
        all_cats = {c["name"]: c for c in coco1["categories"]}
        for c in coco2.get("categories", []):
            all_cats.setdefault(c["name"], c)

        merged_coco = {
            "images":      coco1["images"] + new_images,
            "annotations": coco1["annotations"] + new_anns,
            "categories":  list(all_cats.values()),
        }
        with (dst / "_annotations.coco.json").open("w") as f:
            json.dump(merged_coco, f, indent=2)

        # Copy / symlink all images
        for src_dir in (src1, src2 if src2.exists() else src1):
            for img in src_dir.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    dst_img = dst / img.name
                    if not dst_img.exists():
                        try:
                            dst_img.symlink_to(img)
                        except (OSError, NotImplementedError):
                            shutil.copy2(img, dst_img)

        total_imgs = len(merged_coco["images"])
        total_anns = len(merged_coco["annotations"])
        print(f"  {split}: {total_imgs} images, {total_anns} annotations after merge")

    return merged


# ── Main training function ────────────────────────────────────────────────────

def train(
    data: str,
    seg: bool,
    size: str,
    pretrain_weights: str | None,
    preserve_classes: bool,
    coco_data: str | None,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    lr_encoder: float | None,
    resolution: int,
    output_dir: Path,
    checkpoint_interval: int,
    early_stopping: bool,
    early_stopping_patience: int,
    no_augment: bool,
    resume: str | None,
):
    # ── Dataset pipeline ──────────────────────────────────────────────────────
    coco_base = output_dir / "coco_dataset"
    dataset_dir = _convert_yolo(data, coco_base, seg)

    if preserve_classes:
        remapped = output_dir / "coco_remapped"
        dataset_dir = _remap_coco_ids(dataset_dir, remapped)

    if coco_data:
        coco_secondary = Path(coco_data).resolve()
        if not coco_secondary.exists():
            raise FileNotFoundError(f"--coco-data path not found: {coco_secondary}")
        if not _is_coco_dir(coco_secondary):
            # Try converting if it's a YOLO dataset
            coco_secondary = _convert_yolo(str(coco_secondary), output_dir / "coco_secondary", seg)
        merged = output_dir / "coco_merged"
        dataset_dir = _merge_coco_datasets(dataset_dir, coco_secondary, merged)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_map = SEG_MODELS if seg else BBOX_MODELS
    if size not in model_map or model_map[size] is None:
        raise SystemExit(
            f"Model size '{size}' not available for {'seg' if seg else 'bbox'}.\n"
            f"Available: {[k for k, v in model_map.items() if v is not None]}"
        )
    model_cls = model_map[size]

    if pretrain_weights:
        if not os.path.exists(pretrain_weights):
            raise FileNotFoundError(f"Pretrain weights not found: {pretrain_weights}")
        print(f"Loading checkpoint: {pretrain_weights}")
        model = model_cls(pretrain_weights=pretrain_weights)
    else:
        model = model_cls()

    # ── Resolve effective hyperparameters ─────────────────────────────────────
    if preserve_classes:
        # Freeze encoder: near-zero LR so backbone weights don't move
        effective_lr_encoder = lr_encoder if lr_encoder is not None else 1e-6
        effective_lr         = lr  # head LR (can be higher — head starts from COCO prior)
        augment_recipe       = _PRESERVE_AUGMENT
        print("\n── Class-preserving mode ─────────────────────────────────────────")
        print("  Encoder LR:    frozen (~0)  →  backbone features preserved")
        print("  Class remap:   cyclist→bicycle/1, pedestrian→person/0")
        print("  Augmentation:  conservative (reduced mosaic/mixup)")
        if coco_data:
            print("  COCO mixing:   enabled  →  other classes stay active in every batch")
        else:
            print("  COCO mixing:   disabled  →  other 78 classes will still degrade over time")
            print("                 Add --coco-data <path> for best multi-class preservation")
        print("──────────────────────────────────────────────────────────────────")
    else:
        effective_lr_encoder = lr_encoder
        effective_lr         = lr
        augment_recipe       = _STANDARD_AUGMENT

    model_label = f"rfdetr-{'seg-' if seg else ''}{size}"
    print(f"\nModel:      {model_label}  (preserve_classes={preserve_classes})")
    print(f"Dataset:    {dataset_dir}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {DEVICE}")
    print(f"Resolution: {resolution}  (÷56 ok: {resolution % 56 == 0})")
    print(f"Epochs:     {epochs}  |  batch={batch_size}  |  "
          f"grad_accum={grad_accum_steps}  |  lr={effective_lr}  |  lr_encoder={effective_lr_encoder}")

    # ── Build train() kwargs ──────────────────────────────────────────────────
    train_kwargs: dict = {
        "dataset_dir":             str(dataset_dir),
        "output_dir":              str(output_dir),
        "epochs":                  epochs,
        "batch_size":              batch_size,
        "grad_accum_steps":        grad_accum_steps,
        "lr":                      effective_lr,
        "resolution":              resolution,
        "device":                  DEVICE,
        "checkpoint_interval":     checkpoint_interval,
        "early_stopping":          early_stopping,
        "early_stopping_patience": early_stopping_patience,
    }
    if effective_lr_encoder is not None:
        train_kwargs["lr_encoder"] = effective_lr_encoder
    if resume:
        train_kwargs["resume"] = resume
    if not no_augment:
        train_kwargs.update(_supported_kwargs(model, augment_recipe))

    # ── Train ─────────────────────────────────────────────────────────────────
    os.environ["WANDB_DISABLED"] = "true"
    model.train(**train_kwargs)

    best = output_dir / "checkpoint_best_total.pth"
    print(f"\nTraining complete.  Best checkpoint: {best}")
    print(f"\nTo use in bytetrack_rfdetr.py (ultralytics backend):")
    print(f"  python ByteTrack/bytetrack_rfdetr.py --model {best} --bbox")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Dataset
    parser.add_argument(
        "--data", default="../v5_pdx_cyclist_dataset/data.yaml",
        help="YOLO data.yaml, YOLO dataset dir, or already-converted COCO dir.",
    )

    # Mode
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--bbox", dest="seg", action="store_false", default=False,
                   help="[default] Fine-tune rfdetr-medium (bounding box).")
    g.add_argument("--seg", dest="seg", action="store_true",
                   help="Fine-tune rfdetr-seg-medium. Requires COCO segmentation masks.")

    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--pretrain-weights", default="",
                        help="Checkpoint (.pth) to continue fine-tuning from.")

    # Class-preserving options
    parser.add_argument(
        "--preserve-classes", action="store_true",
        help="Minimise impact on existing COCO class performance. "
             "Enables encoder freeze, COCO class ID remapping, and conservative augmentation. "
             "See module docstring for full explanation.",
    )
    parser.add_argument(
        "--coco-data", default="",
        help="Path to a COCO-format dataset (or YOLO dir) to mix into training. "
             "Used with --preserve-classes to keep all 80 COCO classes active. "
             "Even a small subset (~2 000 images) helps significantly.",
    )

    # Training hyperparameters
    parser.add_argument("--epochs",                  type=int,   default=100,
                        help="Default 100; --preserve-classes mode recommends 30-50.")
    parser.add_argument("--batch-size",              type=int,   default=4)
    parser.add_argument("--grad-accum-steps",        type=int,   default=4)
    parser.add_argument("--lr",                      type=float, default=1e-4,
                        help="Head LR. --preserve-classes mode recommends 1e-5.")
    parser.add_argument("--lr-encoder",              type=float, default=None,
                        help="Encoder LR override. --preserve-classes sets this to ~0 automatically.")
    parser.add_argument("--resolution",              type=int,   default=560)
    parser.add_argument("--checkpoint-interval",     type=int,   default=10)
    parser.add_argument("--no-early-stopping",       action="store_true")
    parser.add_argument("--early-stopping-patience", type=int,   default=15)
    parser.add_argument("--no-augment",              action="store_true")
    parser.add_argument("--resume",                  default="")
    parser.add_argument("--output-dir",              default="")

    args = parser.parse_args()

    if args.resolution % 56 != 0:
        print(f"WARNING: {args.resolution} not divisible by 56 "
              f"(try: 448, 504, 560, 616, 672).")

    # Encourage conservative defaults when in preserve-classes mode
    if args.preserve_classes:
        if args.lr == 1e-4:
            print("NOTE: --preserve-classes mode — lowering default LR from 1e-4 to 1e-5. "
                  "Pass --lr explicitly to override.")
            args.lr = 1e-5
        if args.epochs == 100:
            print("NOTE: --preserve-classes mode — lowering default epochs from 100 to 40. "
                  "Pass --epochs explicitly to override.")
            args.epochs = 40

    model_label = f"rfdetr{'-seg' if args.seg else ''}-{args.size}"
    suffix      = "_preserve" if args.preserve_classes else ""
    output_dir  = (
        Path(args.output_dir).resolve() if args.output_dir
        else _HERE / "runs" / f"{model_label}_res{args.resolution}{suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train(
        data=args.data,
        seg=args.seg,
        size=args.size,
        pretrain_weights=args.pretrain_weights or None,
        preserve_classes=args.preserve_classes,
        coco_data=args.coco_data or None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        resolution=args.resolution,
        output_dir=output_dir,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        no_augment=args.no_augment,
        resume=args.resume or None,
    )


if __name__ == "__main__":
    main()
