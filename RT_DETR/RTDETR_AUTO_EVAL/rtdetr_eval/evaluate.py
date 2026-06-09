#!/usr/bin/env python3
"""
Compare GT annotations CSV vs inference CSV.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "Cyclist", 1: "Pedestrian"}
COLORS = {0: "#E24B4A", 1: "#378ADD"}


def compute_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recalls, precisions):
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at = [p for r, p in zip(recalls, precisions) if r >= thr]
        ap += max(prec_at) if prec_at else 0.0
    return ap / 11.0


def evaluate(gt_csv, pred_csv, iou_thresh):
    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)

    gt_map = {r["frame"]: json.loads(r["predictions_json"]) for _, r in gt_df.iterrows()}
    pred_map = {r["frame"]: json.loads(r["predictions_json"]) for _, r in pred_df.iterrows()}
    all_frames = sorted(set(gt_map) | set(pred_map))

    pr_data = {cid: [] for cid in CLASS_NAMES}
    matched_ious = []
    total_gt = defaultdict(int)

    for frame in all_frames:
        gts = gt_map.get(frame, [])
        def _confidence(p) -> float:
            try:
                c = p.get("confidence")  # type: ignore[attribute-defined-outside-init]
                return float(c) if c is not None else 0.0
            except Exception:
                return 0.0

        preds = sorted(pred_map.get(frame, []), key=_confidence, reverse=True)

        for gt in gts:
            total_gt[gt["class_id"]] += 1

        matched_gts = set()
        for pred in preds:
            pb = pred["bbox"]
            pcls = pred["class_id"]
            conf = _confidence(pred)
            best_iou, best_idx = 0.0, -1
            for gi, gt in enumerate(gts):
                if gi in matched_gts or gt["class_id"] != pcls:
                    continue
                iou = compute_iou(pb, gt["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, gi

            is_tp = best_iou >= iou_thresh
            if pcls in pr_data:
                pr_data[pcls].append((conf, int(is_tp)))
            if is_tp:
                matched_gts.add(best_idx)
                matched_ious.append(best_iou)

    results = {}
    for cid in CLASS_NAMES:
        data = sorted(pr_data[cid], key=lambda x: x[0], reverse=True)
        cum_tp = cum_fp = 0
        n_gt = total_gt[cid]
        precs, recs = [], []
        for _conf, is_tp in data:
            if is_tp:
                cum_tp += 1
            else:
                cum_fp += 1
            precs.append(cum_tp / (cum_tp + cum_fp))
            recs.append(cum_tp / n_gt if n_gt > 0 else 0)

        ap = compute_ap(recs, precs) if recs else 0.0
        results[cid] = {"name": CLASS_NAMES[cid], "precs": precs, "recs": recs, "ap": ap}

    valid_aps = [results[c]["ap"] for c in CLASS_NAMES if results[c]["ap"] > 0.0]
    mAP = np.mean(valid_aps) if valid_aps else 0.0
    skipped = [CLASS_NAMES[c] for c in CLASS_NAMES if results[c]["ap"] == 0.0]
    if skipped:
        print(f"Note: mAP excludes zero-AP class(es): {', '.join(skipped)}")
    return results, mAP, matched_ious


def plot_pr_curve(results, mAP, iou_thresh, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for cid, r in results.items():
        if r["recs"]:
            ax.plot(
                r["recs"],
                r["precs"],
                color=COLORS[cid],
                linewidth=2,
                label=f"{r['name']}  AP={r['ap']:.3f}",
            )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve  (IoU >= {iou_thresh})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98,
        0.06,
        f"mAP = {mAP:.3f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_iou_distribution(ious, iou_thresh, out_path):
    if not ious:
        print("No matched boxes - skipping IoU histogram.")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ious, bins=20, range=(0, 1), color="#378ADD", edgecolor="white", linewidth=0.5)
    ax.axvline(iou_thresh, color="red", linestyle="--", label=f"Threshold ({iou_thresh})")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title("IoU Distribution of Matched Boxes")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.text(
        0.98,
        0.95,
        f"Mean={np.mean(ious):.3f}  Median={np.median(ious):.3f}  n={len(ious)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def build_metrics_dict(results, mAP, ious, iou_thresh: float) -> dict[str, Any]:
    ious_arr = np.array(ious) if ious else np.array([])
    return {
        "iou_threshold": iou_thresh,
        "mAP": round(float(mAP), 4),
        "classes": {
            r["name"]: {
                "ap": round(r["ap"], 4),
                "precision": round(r["precs"][-1], 4) if r["precs"] else 0.0,
                "recall": round(r["recs"][-1], 4) if r["recs"] else 0.0,
            }
            for r in results.values()
        },
        "iou_stats": {
            "n_matched": len(ious),
            "mean": round(float(ious_arr.mean()), 4) if len(ious) else 0.0,
            "median": round(float(np.median(ious_arr)), 4) if len(ious) else 0.0,
            "min": round(float(ious_arr.min()), 4) if len(ious) else 0.0,
            "max": round(float(ious_arr.max()), 4) if len(ious) else 0.0,
            "pct_above_75": round(float((ious_arr >= 0.75).mean()), 4) if len(ious) else 0.0,
        },
    }


def run_evaluation(
    gt_csv: Path | str,
    pred_csv: Path | str,
    out_dir: Path,
    iou_thresh: float,
    *,
    plots: bool = True,
    write_json: bool = True,
) -> dict[str, Any]:
    """
    Run GT vs predictions evaluation; write plots + evaluation_results.json under out_dir.
    Returns the same structure written to evaluation_results.json.
    """
    gt_csv = Path(gt_csv)
    pred_csv = Path(pred_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, mAP, ious = evaluate(str(gt_csv), str(pred_csv), iou_thresh)

    print(f"\n{'Class':<14} {'AP':>6}")
    print("-" * 22)
    for cid, r in results.items():
        print(f"{r['name']:<14} {r['ap']:>6.3f}")
    print("-" * 22)
    print(f"{'mAP':<14} {mAP:>6.3f}\n")

    if plots:
        plot_pr_curve(results, mAP, iou_thresh, out_dir / "pr_curve.png")
        plot_iou_distribution(ious, iou_thresh, out_dir / "iou_distribution.png")

    output = build_metrics_dict(results, mAP, ious, iou_thresh)
    if write_json:
        json_path = out_dir / "evaluation_results.json"
        json_path.write_text(json.dumps(output, indent=2))
        print(f"Saved: {json_path}")
    return output


def main():
    p = argparse.ArgumentParser(description="Evaluate detections CSV vs ground truth.")
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--out", default=".")
    args = p.parse_args()
    run_evaluation(args.gt, args.pred, Path(args.out), args.iou, plots=True, write_json=True)


if __name__ == "__main__":
    main()
