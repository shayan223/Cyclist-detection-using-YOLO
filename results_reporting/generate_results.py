#!/usr/bin/env python3
"""Generate reproducible tables and figures for the RT-DETR tuning suite."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "RT_DETR"
    / "RTDETR_AUTO_EVAL"
    / "runs"
    / "camera_1_trim5"
    / "20260608_220235"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs"
EXPECTED_HEADLINES = {
    "manual_trim5": {
        "mAP": 0.5851,
        "cyclist_ap": 0.4545,
        "cyclist_recall": 0.4457,
    },
    "tuned_best_config": {
        "mAP": 0.7673,
        "cyclist_ap": 0.8182,
        "cyclist_recall": 0.8035,
    },
}

EXPERIMENTS = ["bare_bones", "manual_trim5", "tuned_best_config"]
REPORT_CONFIGS = ["bare_bones.yaml", "manual_trim5.yaml", "tuned_best_config.yaml"]
REPORT_VIDEOS = ["bare_bones.mp4", "manual_trim5.mp4", "tuned_best_config.mp4"]
EXPERIMENT_LABELS = {
    "bare_bones": "Bare bones",
    "manual_trim5": "Manual Trim 5",
    "tuned_best_config": "Auto-Tuned",
}
COLORS = {
    "bare_bones": "#6C757D",
    "manual_trim5": "#D95F02",
    "tuned_best_config": "#1B9E77",
    "explore": "#7570B3",
    "exploit": "#E7298A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible results tables and plots for the completed RT-DETR tuning suite."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Suite run directory (default: {DEFAULT_RUN_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of ranked trials to include in top-trial tables and heatmap.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_sources(run_dir: Path) -> dict[str, Path]:
    summary_path = require_file(run_dir / "summary.csv", "summary CSV")
    leaderboard_path = require_file(
        run_dir / "trials" / "trial_leaderboard.json", "trial leaderboard JSON"
    )
    top_params_path = require_file(run_dir / "trials" / "top_params.json", "top params JSON")
    trial_manifest_path = require_file(
        run_dir / "trials" / "manifest.json", "trial parameter manifest JSON"
    )
    manual_pred_path = require_file(
        run_dir / "predictions" / "manual_trim5_rtdetr.csv", "manual prediction CSV"
    )
    tuned_pred_path = require_file(
        run_dir / "predictions" / "tuned_best_config_rtdetr.csv", "tuned prediction CSV"
    )
    manual_eval_path = require_file(
        run_dir / "eval" / "manual_trim5" / "evaluation_results.json",
        "manual evaluation JSON",
    )
    tuned_eval_path = require_file(
        run_dir / "eval" / "tuned_best_config" / "evaluation_results.json",
        "tuned evaluation JSON",
    )
    config_sources = {
        f"config_{Path(name).stem}": require_file(
            run_dir / "configs" / name,
            f"report comparison config {name}",
        )
        for name in REPORT_CONFIGS
    }
    video_sources = {
        f"video_{Path(name).stem}": require_file(
            run_dir / "videos" / name,
            f"report inference video {name}",
        )
        for name in REPORT_VIDEOS
    }

    summary = pd.read_csv(summary_path)
    if "gt_path" not in summary.columns:
        raise ValueError(f"summary.csv does not include a gt_path column: {summary_path}")
    gt_path = Path(str(summary["gt_path"].dropna().iloc[0]))
    if not gt_path.is_file():
        fallback = REPO_ROOT / "RT_DETR" / "RTDETR_AUTO_EVAL" / "data" / "camera_1" / "ground_truth.csv"
        gt_path = fallback
    require_file(gt_path, "ground-truth CSV")

    sources = {
        "summary": summary_path,
        "leaderboard": leaderboard_path,
        "top_params": top_params_path,
        "trial_manifest": trial_manifest_path,
        "ground_truth": gt_path,
        "manual_predictions": manual_pred_path,
        "tuned_predictions": tuned_pred_path,
        "manual_eval": manual_eval_path,
        "tuned_eval": tuned_eval_path,
    }
    sources.update(config_sources)
    sources.update(video_sources)
    return sources


def ensure_output_dirs(out_dir: Path) -> dict[str, Path]:
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    configs_dir = out_dir / "configs"
    videos_dir = out_dir / "videos"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    return {
        "tables": tables_dir,
        "figures": figures_dir,
        "configs": configs_dir,
        "videos": videos_dir,
    }


def numeric_summary(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    numeric_cols = [
        "score",
        "mAP",
        "cyclist_ap",
        "cyclist_precision",
        "cyclist_recall",
        "pedestrian_ap",
        "pedestrian_precision",
        "pedestrian_recall",
        "mean_iou",
        "median_iou",
        "iou_pct_above_75",
        "n_matched",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def row_for(summary: pd.DataFrame, experiment: str) -> pd.Series:
    rows = summary.loc[summary["experiment_name"] == experiment]
    if rows.empty:
        raise ValueError(f"Expected experiment not found in summary.csv: {experiment}")
    return rows.iloc[0]


def validate_headline_metrics(summary: pd.DataFrame) -> None:
    for experiment, metrics in EXPECTED_HEADLINES.items():
        row = row_for(summary, experiment)
        for metric, expected in metrics.items():
            actual = float(row[metric])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-4):
                raise ValueError(
                    f"Headline metric mismatch for {experiment}.{metric}: "
                    f"expected {expected:.4f}, got {actual:.4f}"
                )


def make_main_results(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment in EXPERIMENTS:
        row = row_for(summary, experiment)
        rows.append(
            {
                "Experiment": EXPERIMENT_LABELS[experiment],
                "Experiment ID": experiment,
                "Type": row["experiment_type"],
                "Score": row["score"],
                "mAP": row["mAP"],
                "Cyclist AP": row["cyclist_ap"],
                "Cyclist Precision": row["cyclist_precision"],
                "Cyclist Recall": row["cyclist_recall"],
                "Pedestrian AP": row["pedestrian_ap"],
                "Pedestrian Precision": row["pedestrian_precision"],
                "Pedestrian Recall": row["pedestrian_recall"],
                "Mean IoU": row["mean_iou"],
                "Median IoU": row["median_iou"],
                "IoU >= 0.75": row["iou_pct_above_75"],
                "Matched Boxes": int(row["n_matched"]),
            }
        )
    return pd.DataFrame(rows)


def make_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "mAP": "mAP",
        "Cyclist AP": "cyclist_ap",
        "Cyclist Recall": "cyclist_recall",
        "Matched Boxes": "n_matched",
        "Mean IoU": "mean_iou",
        "IoU >= 0.75": "iou_pct_above_75",
    }
    manual = row_for(summary, "manual_trim5")
    tuned = row_for(summary, "tuned_best_config")
    rows = []
    for label, col in metric_map.items():
        manual_value = float(manual[col])
        tuned_value = float(tuned[col])
        delta = tuned_value - manual_value
        percent = (delta / manual_value * 100.0) if manual_value else np.nan
        rows.append(
            {
                "Metric": label,
                "Manual Trim 5": manual_value,
                "Auto-Tuned": tuned_value,
                "Absolute Delta": delta,
                "Percent Delta": percent,
            }
        )
    return pd.DataFrame(rows)


def phase_for_trial(trial: str) -> str:
    return trial.split("_", 1)[0] if "_" in trial else "unknown"


def trial_order_key(trial: str) -> tuple[int, int, str]:
    phase = phase_for_trial(trial)
    phase_order = {"explore": 0, "exploit": 1}.get(phase, 2)
    match = re.search(r"_(\d+)$", trial)
    number = int(match.group(1)) if match else 0
    return phase_order, number, trial


def make_top_trials(leaderboard_path: Path, top_n: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    leaderboard = json.loads(leaderboard_path.read_text())
    rows = []
    for rank, item in enumerate(leaderboard[:top_n], start=1):
        classes = item["metrics"]["classes"]
        iou = item["metrics"]["iou_stats"]
        rows.append(
            {
                "Rank": rank,
                "Trial": item["trial"],
                "Phase": phase_for_trial(item["trial"]),
                "Score": item["score"],
                "mAP": item["metrics"]["mAP"],
                "Cyclist AP": classes["Cyclist"]["ap"],
                "Cyclist Recall": classes["Cyclist"]["recall"],
                "Pedestrian AP": classes["Pedestrian"]["ap"],
                "Pedestrian Recall": classes["Pedestrian"]["recall"],
                "Mean IoU": iou["mean"],
                "Matched Boxes": iou["n_matched"],
            }
        )
    return pd.DataFrame(rows), leaderboard


def make_best_config_params(top_params_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    top_params = json.loads(top_params_path.read_text())
    if not top_params:
        raise ValueError(f"No entries found in {top_params_path}")
    best = top_params[0]
    rows = [
        {
            "Trial": best["trial"],
            "Score": best["score"],
            "Parameter": name,
            "Value": value,
        }
        for name, value in best["params"].items()
    ]
    return pd.DataFrame(rows), best


def write_table_outputs(df: pd.DataFrame, csv_path: Path, tex_path: Path | None = None) -> list[Path]:
    df.to_csv(csv_path, index=False)
    written = [csv_path]
    if tex_path is not None:
        tex_path.write_text(df.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
        written.append(tex_path)
    return written


def copy_report_configs(run_dir: Path, configs_dir: Path) -> list[Path]:
    written = []
    for config_name in REPORT_CONFIGS:
        src = require_file(
            run_dir / "configs" / config_name,
            f"report comparison config {config_name}",
        )
        dst = configs_dir / config_name
        shutil.copy2(src, dst)
        written.append(dst)

    manifest_path = configs_dir / "config_manifest.json"
    manifest = {
        "copied_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": relative(run_dir / "configs"),
        "configs": [
            {
                "name": name,
                "source": relative(run_dir / "configs" / name),
                "report_copy": relative(configs_dir / name),
            }
            for name in REPORT_CONFIGS
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(manifest_path)
    return written


def copy_report_videos(run_dir: Path, videos_dir: Path) -> list[Path]:
    written = []
    for video_name in REPORT_VIDEOS:
        src = require_file(
            run_dir / "videos" / video_name,
            f"report inference video {video_name}",
        )
        dst = videos_dir / video_name
        shutil.copy2(src, dst)
        written.append(dst)

    manifest_path = videos_dir / "video_manifest.json"
    manifest = {
        "copied_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": relative(run_dir / "videos"),
        "videos": [
            {
                "name": name,
                "source": relative(run_dir / "videos" / name),
                "report_copy": relative(videos_dir / name),
                "bytes": int((videos_dir / name).stat().st_size),
            }
            for name in REPORT_VIDEOS
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(manifest_path)
    return written


def load_prediction_map(csv_path: Path) -> dict[int, list[dict[str, Any]]]:
    df = pd.read_csv(csv_path)
    records: dict[int, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        records[int(row["frame"])] = json.loads(row["predictions_json"])
    return records


def confidence(prediction: dict[str, Any]) -> float:
    try:
        value = prediction.get("confidence", 0.0)
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def compute_iou(a: list[float], b: list[float]) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def matched_ious(gt_csv: Path, pred_csv: Path, iou_threshold: float = 0.5) -> list[float]:
    gt_map = load_prediction_map(gt_csv)
    pred_map = load_prediction_map(pred_csv)
    all_frames = sorted(set(gt_map) | set(pred_map))
    ious: list[float] = []

    for frame in all_frames:
        gts = gt_map.get(frame, [])
        preds = sorted(pred_map.get(frame, []), key=confidence, reverse=True)
        matched_gt_indexes: set[int] = set()

        for pred in preds:
            pred_class = pred.get("class_id")
            best_iou = 0.0
            best_index = -1
            for gt_index, gt in enumerate(gts):
                if gt_index in matched_gt_indexes or gt.get("class_id") != pred_class:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_index = gt_index
            if best_iou >= iou_threshold and best_index >= 0:
                matched_gt_indexes.add(best_index)
                ious.append(best_iou)

    return ious


def plot_aggregate_metrics(main_results: pd.DataFrame, out_path: Path) -> None:
    metrics = ["mAP", "Cyclist AP", "Cyclist Recall", "Pedestrian AP", "Pedestrian Recall"]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, experiment_id in enumerate(EXPERIMENTS):
        row = main_results.loc[main_results["Experiment ID"] == experiment_id].iloc[0]
        values = [float(row[m]) for m in metrics]
        ax.bar(
            x + (offset - 1) * width,
            values,
            width=width,
            label=EXPERIMENT_LABELS[experiment_id],
            color=COLORS[experiment_id],
        )
    ax.set_ylabel("Metric value")
    ax.set_title("Aggregate Detection Metrics by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left", ncols=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_manual_vs_tuned_deltas(summary: pd.DataFrame, out_path: Path) -> None:
    manual = row_for(summary, "manual_trim5")
    tuned = row_for(summary, "tuned_best_config")
    metrics = [
        ("Cyclist AP", "cyclist_ap", COLORS["tuned_best_config"], 0.008, 0.016),
        ("Cyclist Precision", "cyclist_precision", COLORS["tuned_best_config"], 0.0, 0.0),
        ("Cyclist Recall", "cyclist_recall", COLORS["tuned_best_config"], -0.008, -0.006),
        ("Pedestrian AP", "pedestrian_ap", "#377EB8", 0.0, -0.002),
        ("Pedestrian Precision", "pedestrian_precision", "#377EB8", 0.0, 0.010),
        ("Pedestrian Recall", "pedestrian_recall", "#377EB8", 0.0, -0.018),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = [0, 1]
    for label, col, color, left_offset, right_offset in metrics:
        values = [float(manual[col]), float(tuned[col])]
        ax.plot(x, values, marker="o", linewidth=2.0, color=color, alpha=0.78)
        ax.text(
            1.03,
            values[1] + right_offset,
            f"{label} ({values[1]:.3f})",
            va="center",
            fontsize=9,
            color=color,
        )
        ax.text(
            -0.04,
            values[0] + left_offset,
            f"{values[0]:.3f}",
            va="center",
            ha="right",
            fontsize=8,
        )

    ax.set_xlim(-0.18, 1.55)
    ax.set_ylim(0.35, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(["Manual Trim 5", "Auto-Tuned"])
    ax.set_ylabel("Metric value")
    ax.set_title("Manual vs Auto-Tuned Class Metric Changes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_iou_comparison(manual_ious: list[float], tuned_ious: list[float], out_path: Path) -> None:
    bins = np.linspace(0.5, 1.0, 21)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(
        manual_ious,
        bins=bins,
        alpha=0.55,
        color=COLORS["manual_trim5"],
        label=f"Manual Trim 5 (n={len(manual_ious)})",
    )
    ax.hist(
        tuned_ious,
        bins=bins,
        alpha=0.55,
        color=COLORS["tuned_best_config"],
        label=f"Auto-Tuned (n={len(tuned_ious)})",
    )
    for values, color, label in [
        (manual_ious, COLORS["manual_trim5"], "Manual median"),
        (tuned_ious, COLORS["tuned_best_config"], "Auto-Tuned median"),
    ]:
        ax.axvline(np.median(values), color=color, linestyle="--", linewidth=2, label=label)
    ax.axvline(0.75, color="#333333", linestyle=":", linewidth=2, label="IoU 0.75")
    ax.set_xlabel("Matched-box IoU")
    ax.set_ylabel("Count")
    ax.set_title("Manual vs Auto-Tuned IoU Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_search_scores(leaderboard: list[dict[str, Any]], out_path: Path) -> None:
    ordered = sorted(leaderboard, key=lambda item: trial_order_key(item["trial"]))
    x = np.arange(1, len(ordered) + 1)
    scores = np.array([float(item["score"]) for item in ordered])
    phases = [phase_for_trial(item["trial"]) for item in ordered]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, scores, color="#666666", linewidth=1.0, alpha=0.45)
    for phase in ["explore", "exploit"]:
        indexes = [i for i, p in enumerate(phases) if p == phase]
        ax.scatter(
            x[indexes],
            scores[indexes],
            s=32,
            color=COLORS[phase],
            label=phase.capitalize(),
            zorder=3,
        )
    best_index = int(np.argmax(scores))
    ax.scatter(
        x[best_index],
        scores[best_index],
        s=90,
        facecolor="none",
        edgecolor="#111111",
        linewidth=2,
        zorder=4,
    )
    ax.annotate(
        f"{ordered[best_index]['trial']} ({scores[best_index]:.4f})",
        xy=(x[best_index], scores[best_index]),
        xytext=(x[best_index] + 2, min(0.98, scores[best_index] + 0.055)),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1},
        fontsize=9,
    )
    ax.set_xlabel("Trial order")
    ax.set_ylabel("Score")
    ax.set_title("Two-Phase Configuration Search Scores")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_top_trial_params_heatmap(
    top_trials: pd.DataFrame,
    trial_manifest_path: Path,
    out_path: Path,
) -> None:
    manifest = json.loads(trial_manifest_path.read_text())
    trial_names = top_trials["Trial"].tolist()
    rows = []
    for trial in trial_names:
        params = manifest.get(trial)
        if not params:
            raise ValueError(f"Missing parameters for top trial {trial} in {trial_manifest_path}")
        rows.append(params)
    params_df = pd.DataFrame(rows, index=trial_names)
    numeric = params_df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="any")
    if numeric.empty:
        raise ValueError("No numeric trial parameters available for heatmap")

    normalized = numeric.copy()
    for col in normalized.columns:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        normalized[col] = 0.5 if col_max == col_min else (normalized[col] - col_min) / (col_max - col_min)

    fig_width = max(10, 0.55 * len(normalized.columns))
    fig_height = max(5.5, 0.42 * len(normalized.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(normalized.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(normalized.columns)))
    ax.set_xticklabels(normalized.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(normalized.index)))
    ax.set_yticklabels(normalized.index, fontsize=9)
    ax.set_title("Normalized Hyperparameters for Top Trials")
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalized value within top trials")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_eval_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_iou_recompute(label: str, ious: list[float], eval_metrics: dict[str, Any]) -> None:
    expected_count = int(eval_metrics["iou_stats"]["n_matched"])
    if len(ious) != expected_count:
        raise ValueError(
            f"Recomputed {label} IoU count does not match evaluation JSON: "
            f"{len(ious)} vs {expected_count}"
        )
    if ious:
        expected_mean = float(eval_metrics["iou_stats"]["mean"])
        actual_mean = float(np.mean(ious))
        if not math.isclose(actual_mean, expected_mean, rel_tol=0.0, abs_tol=5e-4):
            raise ValueError(
                f"Recomputed {label} mean IoU does not match evaluation JSON: "
                f"{actual_mean:.4f} vs {expected_mean:.4f}"
            )


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    top_n = max(1, int(args.top_n))

    sources = load_sources(run_dir)
    output_dirs = ensure_output_dirs(out_dir)
    tables_dir = output_dirs["tables"]
    figures_dir = output_dirs["figures"]
    configs_dir = output_dirs["configs"]
    videos_dir = output_dirs["videos"]

    summary = numeric_summary(sources["summary"])
    validate_headline_metrics(summary)

    main_results = make_main_results(summary)
    delta_table = make_delta_table(summary)
    top_trials, leaderboard = make_top_trials(sources["leaderboard"], top_n)
    best_params, best_seed = make_best_config_params(sources["top_params"])

    written: list[Path] = []
    written += write_table_outputs(
        main_results,
        tables_dir / "main_results.csv",
        tables_dir / "main_results.tex",
    )
    written += write_table_outputs(delta_table, tables_dir / "main_results_deltas.csv")
    written += write_table_outputs(
        top_trials,
        tables_dir / "top_trials.csv",
        tables_dir / "top_trials.tex",
    )
    written += write_table_outputs(
        best_params,
        tables_dir / "best_config_params.csv",
        tables_dir / "best_config_params.tex",
    )
    written += copy_report_configs(run_dir, configs_dir)
    written += copy_report_videos(run_dir, videos_dir)

    aggregate_path = figures_dir / "aggregate_metrics.png"
    deltas_path = figures_dir / "manual_vs_tuned_deltas.png"
    iou_path = figures_dir / "iou_comparison.png"
    search_path = figures_dir / "search_scores.png"
    heatmap_path = figures_dir / "top_trial_params_heatmap.png"

    plot_aggregate_metrics(main_results, aggregate_path)
    plot_manual_vs_tuned_deltas(summary, deltas_path)
    manual_eval = load_eval_json(sources["manual_eval"])
    tuned_eval = load_eval_json(sources["tuned_eval"])
    if manual_eval["iou_threshold"] != tuned_eval["iou_threshold"]:
        raise ValueError(
            "Manual and tuned evaluation JSON files use different IoU thresholds: "
            f"{manual_eval['iou_threshold']} vs {tuned_eval['iou_threshold']}"
        )
    iou_threshold = float(manual_eval["iou_threshold"])
    manual_ious = matched_ious(sources["ground_truth"], sources["manual_predictions"], iou_threshold)
    tuned_ious = matched_ious(sources["ground_truth"], sources["tuned_predictions"], iou_threshold)
    validate_iou_recompute("manual", manual_ious, manual_eval)
    validate_iou_recompute("tuned", tuned_ious, tuned_eval)
    plot_iou_comparison(manual_ious, tuned_ious, iou_path)
    plot_search_scores(leaderboard, search_path)
    plot_top_trial_params_heatmap(top_trials, sources["trial_manifest"], heatmap_path)
    written += [aggregate_path, deltas_path, iou_path, search_path, heatmap_path]

    manual = row_for(summary, "manual_trim5")
    tuned = row_for(summary, "tuned_best_config")
    manifest_path = out_dir / "manifest.json"
    written_with_manifest = written + [manifest_path]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": relative(run_dir),
        "source_files": {name: relative(path) for name, path in sources.items()},
        "generated_files": [relative(path) for path in written_with_manifest],
        "headline_metrics": {
            "manual_mAP": float(manual["mAP"]),
            "tuned_mAP": float(tuned["mAP"]),
            "mAP_absolute_delta": float(tuned["mAP"] - manual["mAP"]),
            "mAP_percent_delta": float((tuned["mAP"] - manual["mAP"]) / manual["mAP"] * 100.0),
            "manual_cyclist_ap": float(manual["cyclist_ap"]),
            "tuned_cyclist_ap": float(tuned["cyclist_ap"]),
            "manual_cyclist_recall": float(manual["cyclist_recall"]),
            "tuned_cyclist_recall": float(tuned["cyclist_recall"]),
            "best_trial": best_seed["trial"],
            "best_trial_score": float(best_seed["score"]),
            "sweep_best_reproduced": bool(
                math.isclose(
                    float(row_for(summary, "sweep_best")["score"]),
                    float(tuned["score"]),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            )
            if "sweep_best" in set(summary["experiment_name"])
            else None,
        },
        "notes": [
            "All figures are generated programmatically from suite CSV/JSON artifacts.",
            "The detector checkpoint is unchanged; this report evaluates YAML/inference-parameter tuning.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(manifest_path)

    print(f"Generated {len(written)} artifacts under {out_dir}")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
