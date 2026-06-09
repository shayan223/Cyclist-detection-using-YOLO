#!/usr/bin/env python3
"""
convert_annotations_to_csv.py
Convert interpolated_annotate.py YOLO labels → inference-style CSV.

Output format matches deepSORT_rtdetr.py --csv output:
  frame, predictions_json

Usage:
  python convert_annotations_to_csv.py \
      --run-dir annotation/run_001 \
      --video   my_video.mp4 \
      --output  annotation_run001.csv \
      [--total-frames 1800]   # optional: pad with empty rows up to N frames
"""

import argparse
import csv
import json
import re
from pathlib import Path

import cv2

# Must match CLASS_NAMES in interpolated_annotate.py
CLASS_NAMES = {0: "Cyclist", 1: "Pedestrian"}


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert normalised YOLO centre format → pixel [x1, y1, x2, y2]."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [x1, y1, x2, y2]


def get_image_size(run_dir: Path):
    """Read width/height from the first image found in the run folder."""
    img_dir = run_dir / "images"
    for img_path in sorted(img_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    raise RuntimeError(f"No readable images found in {img_dir}")


def parse_frame_num(filename: str):
    """Extract frame number from 'frame_XXXXXXXX.jpg'."""
    m = re.search(r"frame_(\d+)", filename)
    return int(m.group(1)) if m else None


def load_annotations(run_dir: Path, img_w: int, img_h: int) -> dict:
    """
    Returns {frame_num: [prediction_dict, ...]} for every label file.
    confidence is set to 1.0 (ground-truth annotations have no score).
    """
    lbl_dir = run_dir / "labels"
    annotations = {}

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        frame_num = parse_frame_num(lbl_path.stem)
        if frame_num is None:
            continue

        preds = []
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            bbox = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            preds.append({
                "class_id":   cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "confidence": 1.0,   # ground-truth — no detector score
                "bbox":       bbox,
            })

        annotations[frame_num] = preds

    return annotations


def write_csv(annotations: dict, output_path: Path, total_frames: int):
    """Write frame-by-frame CSV, padding with empty rows for unannotated frames."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "predictions_json"])

        for frame_num in range(total_frames):
            preds = annotations.get(frame_num, [])
            writer.writerow([frame_num, json.dumps(preds)])

    print(f"Wrote {total_frames} rows → {output_path}")
    annotated = len(annotations)
    print(f"  Annotated frames : {annotated}")
    print(f"  Empty frames     : {total_frames - annotated}")


def get_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="Path to annotation run folder (e.g. annotation/run_001)")
    p.add_argument("--video",   default=None,
                   help="Source video — used to get total frame count and image size")
    p.add_argument("--output",  default=None,
                   help="Output CSV path (default: <run-dir>/annotations.csv)")
    p.add_argument("--total-frames", type=int, default=None,
                   help="Override total frame count (skip if --video is provided)")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    # Image dimensions
    if args.video:
        cap = cv2.VideoCapture(args.video)
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    else:
        print("No --video provided; reading image size from run folder...")
        img_w, img_h = get_image_size(run_dir)
    print(f"Image size: {img_w}×{img_h}")

    # Total frames
    if args.total_frames:
        total_frames = args.total_frames
    elif args.video:
        total_frames = get_total_frames(args.video)
    else:
        # Fall back to max annotated frame + 1
        lbl_dir = run_dir / "labels"
        frame_nums = [
            parse_frame_num(p.stem)
            for p in lbl_dir.glob("*.txt")
            if parse_frame_num(p.stem) is not None
        ]
        total_frames = max(frame_nums) + 1 if frame_nums else 0
        print(f"No --video/--total-frames; using max annotated frame+1 = {total_frames}")
    print(f"Total frames: {total_frames}")

    annotations = load_annotations(run_dir, img_w, img_h)
    print(f"Loaded {len(annotations)} annotated frames from {run_dir}")

    output_path = Path(args.output) if args.output else run_dir / "annotations.csv"
    write_csv(annotations, output_path, total_frames)


if __name__ == "__main__":
    main()