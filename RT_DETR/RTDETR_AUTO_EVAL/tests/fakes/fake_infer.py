#!/usr/bin/env python3
"""
Fake inference stub for the e2e test ("random model with stat function").

Mirrors the CLI of deepSORT_rtdetr.py (--config, -i/--input, --csv) but uses no
ML deps. Detection quality is a polynomial of the swept hyperparameters
(scene.quality); detections are emitted against the synthetic scene GT so the
real run_evaluation produces a score that tracks that quality.

Writes {video_stem}_rtdetr.csv next to the input video.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene  # noqa: E402  (local fakes module)


def _extract_params(cfg: dict) -> dict:
    classes = cfg.get("classes", [])
    ped = next((c for c in classes if c.get("id") == 1), {})
    return {
        "inference.confidence": cfg.get("inference", {}).get("confidence", 0.60),
        "nms.hard_iou": cfg.get("nms", {}).get("hard_iou", 0.25),
        "classes.1.min_confidence": ped.get("min_confidence", 0.60),
    }


def _jitter_box(box, rng, max_shift):
    """Shift a box; larger shift -> lower IoU. Capped at 8px so kept boxes stay TPs."""
    s = max(1, min(8, max_shift))
    dx = rng.randint(-s, s)
    dy = rng.randint(-s, s)
    x1, y1, x2, y2 = box
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


def _predictions_for_frame(frame: int, q: float, rng: random.Random) -> list[dict]:
    gt = scene.gt_for_frame(frame)
    preds: list[dict] = []

    # Worse quality -> bigger jitter -> lower IoU (drags mean IoU toward the 0.65
    # gate in trial_score), but capped so kept boxes remain above the 0.5 match.
    max_shift = 2 + int(round((1.0 - q) * 6))

    # True positives: per class, recall scales with quality (graded objective).
    by_class: dict[int, list] = {}
    for box in gt:
        by_class.setdefault(box["class_id"], []).append(box)
    for cid, boxes in by_class.items():
        # Floor (not round): even near-optimal trials miss a few boxes, so no
        # trial trivially reaches AP=1.0 and the running-best climbs gradually.
        keep = int(q * len(boxes))
        for box in boxes[:keep]:
            preds.append({
                "class_id": cid,
                "bbox": _jitter_box(box["bbox"], rng, max_shift),
                "confidence": round(q, 4),
            })

    # False positives: more as quality drops, and ranked ABOVE the true positives
    # (high confidence) so they actually depress precision / AP.
    n_fp = int(round((1.0 - q) * scene.FP_MAX))
    for i in range(n_fp):
        x = 5000 + 120 * i
        y = 4000 + 60 * i
        preds.append({
            "class_id": i % 2,
            "bbox": [x, y, x + 40, y + 40],
            "confidence": 1.0,  # outranks every TP -> precision = k/(k+m), continuous in q
        })
    return preds


def main():
    p = argparse.ArgumentParser(description="Fake RT-DETR inference stub.")
    p.add_argument("--config", required=True)
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--csv", action="store_true")
    p.add_argument("--output", "-o", default=None)
    args, _unknown = p.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    params = _extract_params(cfg)
    q = scene.quality(params)

    # Deterministic per trial: seed jitter/FP placement by the config stem.
    stem = os.path.splitext(os.path.basename(args.config))[0]

    video_stem = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(os.path.dirname(os.path.abspath(args.input)), f"{video_stem}_rtdetr.csv")

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "predictions_json"])
        for frame in range(scene.N_FRAMES):
            rng = random.Random(f"{stem}:{frame}")
            preds = _predictions_for_frame(frame, q, rng)
            writer.writerow([frame, json.dumps(preds)])

    print(f"[fake_infer] {stem}: quality={q:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
