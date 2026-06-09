"""Experiment suite runner for RT-DETR AUTO_EVAL."""

from __future__ import annotations

import csv
import copy
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from rtdetr_eval.evaluate import run_evaluation
from rtdetr_eval.search import run_exploitation, run_exploration
from rtdetr_eval.trials import predictions_path, run_trials, trial_score


SUMMARY_FIELDS = [
    "experiment_name",
    "experiment_type",
    "model_path",
    "source_video_path",
    "config_path",
    "video_path",
    "gt_path",
    "score",
    "mAP",
    "cyclist_ap",
    "cyclist_precision",
    "cyclist_recall",
    "pedestrian_ap",
    "pedestrian_precision",
    "pedestrian_recall",
    "iou_threshold",
    "mean_iou",
    "median_iou",
    "iou_min",
    "iou_max",
    "iou_pct_above_75",
    "n_matched",
    "seed",
    "n_explore",
    "n_refine",
    "best_trial",
    "predictions_csv",
    "eval_json",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _write_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)


def _normalise_base_config(cfg: dict[str, Any], model: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg["model"] = str(model)
    cfg["input"] = ""
    cfg["output"] = ""
    cfg.setdefault("inference", {})["half"] = False
    cfg.setdefault("debug", {})["log_detections"] = False
    cfg.setdefault("debug", {})["inference_only"] = False
    return cfg


def _bare_bones_config(cfg: dict[str, Any], output_video: Path) -> dict[str, Any]:
    cfg = _normalise_base_config(cfg, Path(cfg["model"]))
    cfg["output"] = str(output_video)
    cfg.setdefault("debug", {})["inference_only"] = True
    passes = cfg.setdefault("passes", {})
    for name in ("top_region", "sahi", "warp"):
        passes.setdefault(name, {})["enabled"] = False
    return cfg


def _manual_config(cfg: dict[str, Any], model: Path, output_video: Path) -> dict[str, Any]:
    cfg = _normalise_base_config(cfg, model)
    cfg["output"] = str(output_video)
    return cfg


def _run_inference(
    infer_script: Path,
    config: Path,
    video: Path,
    *,
    cwd: Path,
    inference_only: bool = False,
) -> None:
    cmd = [
        sys.executable,
        str(infer_script),
        "--config",
        str(config.resolve()),
        "-i",
        str(video.resolve()),
        "--csv",
    ]
    if inference_only:
        cmd.append("--inference-only")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _copy_predictions(video: Path, dest: Path) -> Path:
    pred = predictions_path(video)
    if not pred.is_file():
        raise FileNotFoundError(f"Expected predictions CSV after inference: {pred}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pred, dest)
    return dest


def _flatten_metrics(
    *,
    name: str,
    experiment_type: str,
    model: Path,
    config: Path,
    video: Path,
    source_video: Path,
    gt: Path,
    metrics: dict[str, Any],
    seed: int | None,
    n_explore: int | None,
    n_refine: int | None,
    best_trial: str = "",
    predictions_csv: Path | None = None,
    eval_json: Path | None = None,
) -> dict[str, Any]:
    cyclist = metrics["classes"].get("Cyclist", {})
    pedestrian = metrics["classes"].get("Pedestrian", {})
    iou = metrics.get("iou_stats", {})
    return {
        "experiment_name": name,
        "experiment_type": experiment_type,
        "model_path": str(model),
        "source_video_path": str(source_video),
        "config_path": str(config),
        "video_path": str(video),
        "gt_path": str(gt),
        "score": round(trial_score(metrics), 6),
        "mAP": metrics.get("mAP", 0.0),
        "cyclist_ap": cyclist.get("ap", 0.0),
        "cyclist_precision": cyclist.get("precision", 0.0),
        "cyclist_recall": cyclist.get("recall", 0.0),
        "pedestrian_ap": pedestrian.get("ap", 0.0),
        "pedestrian_precision": pedestrian.get("precision", 0.0),
        "pedestrian_recall": pedestrian.get("recall", 0.0),
        "iou_threshold": metrics.get("iou_threshold", ""),
        "mean_iou": iou.get("mean", 0.0),
        "median_iou": iou.get("median", 0.0),
        "iou_min": iou.get("min", 0.0),
        "iou_max": iou.get("max", 0.0),
        "iou_pct_above_75": iou.get("pct_above_75", 0.0),
        "n_matched": iou.get("n_matched", 0),
        "seed": "" if seed is None else seed,
        "n_explore": "" if n_explore is None else n_explore,
        "n_refine": "" if n_refine is None else n_refine,
        "best_trial": best_trial,
        "predictions_csv": "" if predictions_csv is None else str(predictions_csv),
        "eval_json": "" if eval_json is None else str(eval_json),
    }


def _evaluate_named_run(
    *,
    name: str,
    experiment_type: str,
    infer_script: Path,
    inference_cwd: Path,
    config: Path,
    video: Path,
    source_video: Path,
    gt: Path,
    model: Path,
    run_dir: Path,
    iou: float,
    seed: int | None,
    n_explore: int | None = None,
    n_refine: int | None = None,
    inference_only: bool = False,
) -> dict[str, Any]:
    _run_inference(
        infer_script,
        config,
        video,
        cwd=inference_cwd,
        inference_only=inference_only,
    )
    pred_copy = _copy_predictions(video, run_dir / "predictions" / f"{name}_rtdetr.csv")
    eval_dir = run_dir / "eval" / name
    metrics = run_evaluation(gt, pred_copy, eval_dir, iou, plots=True, write_json=True)
    return _flatten_metrics(
        name=name,
        experiment_type=experiment_type,
        model=model,
        config=config,
        video=video,
        source_video=source_video,
        gt=gt,
        metrics=metrics,
        seed=seed,
        n_explore=n_explore,
        n_refine=n_refine,
        predictions_csv=pred_copy,
        eval_json=eval_dir / "evaluation_results.json",
    )


def _latest_leaderboard_row(leaderboard: Path) -> dict[str, Any]:
    data = json.loads(leaderboard.read_text())
    if not data:
        raise ValueError(f"Leaderboard is empty: {leaderboard}")
    return data[0]


def run_suite(
    *,
    model: Path,
    video: Path,
    gt: Path,
    manual_config: Path,
    infer_script: Path,
    inference_cwd: Path,
    out_dir: Path | None = None,
    n: int = 50,
    refine_n: int = 30,
    seed: int | None = 42,
    iou: float = 0.5,
    sigma: float = 0.15,
    top_k: int = 3,
    eval_plots: bool = False,
) -> Path:
    model = model.resolve()
    video = video.resolve()
    gt = gt.resolve()
    manual_config = manual_config.resolve()
    infer_script = infer_script.resolve()
    inference_cwd = inference_cwd.resolve()

    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / "camera_1_trim5" / stamp
    out_dir = out_dir.resolve()
    configs_dir = out_dir / "configs"
    input_dir = out_dir / "input"
    videos_dir = out_dir / "videos"
    trials_dir = out_dir / "trials"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    source_video = video
    eval_video = input_dir / video.name
    if source_video != eval_video:
        shutil.copyfile(source_video, eval_video)
    video = eval_video.resolve()

    base_cfg = _normalise_base_config(_load_yaml(manual_config), model)

    rows: list[dict[str, Any]] = []

    bare_cfg_path = configs_dir / "bare_bones.yaml"
    _write_yaml(
        bare_cfg_path,
        _bare_bones_config({**base_cfg, "model": str(model)}, videos_dir / "bare_bones.mp4"),
    )
    rows.append(
        _evaluate_named_run(
            name="bare_bones",
            experiment_type="detector_only",
            infer_script=infer_script,
            inference_cwd=inference_cwd,
            config=bare_cfg_path,
            video=video,
            source_video=source_video,
            gt=gt,
            model=model,
            run_dir=out_dir,
            iou=iou,
            seed=seed,
            inference_only=True,
        )
    )

    manual_cfg_path = configs_dir / "manual_trim5.yaml"
    _write_yaml(manual_cfg_path, _manual_config(base_cfg, model, videos_dir / "manual_trim5.mp4"))
    rows.append(
        _evaluate_named_run(
            name="manual_trim5",
            experiment_type="manual_config",
            infer_script=infer_script,
            inference_cwd=inference_cwd,
            config=manual_cfg_path,
            video=video,
            source_video=source_video,
            gt=gt,
            model=model,
            run_dir=out_dir,
            iou=iou,
            seed=seed,
        )
    )

    tuning_base_path = configs_dir / "tuning_base.yaml"
    _write_yaml(tuning_base_path, _normalise_base_config(base_cfg, model))

    print("\n=== Phase 1: exploration ===")
    run_exploration(tuning_base_path, trials_dir, n, seed)
    run_trials(
        trials_dir,
        video,
        gt,
        infer_script=infer_script,
        inference_cwd=inference_cwd,
        iou_thresh=iou,
        eval_plots=eval_plots,
        top_params_k=top_k,
    )

    print("\n=== Phase 2: exploitation (warm-start) ===")
    run_exploitation(
        tuning_base_path,
        trials_dir / "top_params.json",
        trials_dir,
        refine_n,
        sigma_frac=sigma,
        top_k=top_k,
        seed=seed,
    )
    run_trials(
        trials_dir,
        video,
        gt,
        infer_script=infer_script,
        inference_cwd=inference_cwd,
        iou_thresh=iou,
        eval_plots=eval_plots,
        top_params_k=top_k,
    )

    best = _latest_leaderboard_row(trials_dir / "trial_leaderboard.json")
    best_trial = best["trial"]
    best_trial_cfg = trials_dir / f"{best_trial}.yaml"
    rows.append(
        _flatten_metrics(
            name="sweep_best",
            experiment_type="search_leaderboard",
            model=model,
            config=best_trial_cfg,
            video=video,
            source_video=source_video,
            gt=gt,
            metrics=best["metrics"],
            seed=seed,
            n_explore=n,
            n_refine=refine_n,
            best_trial=best_trial,
        )
    )

    tuned_cfg = _load_yaml(trials_dir / "best_config.yaml")
    tuned_cfg["model"] = str(model)
    tuned_cfg["input"] = ""
    tuned_cfg["output"] = str(videos_dir / "tuned_best_config.mp4")
    tuned_cfg.setdefault("inference", {})["half"] = False
    tuned_cfg_path = configs_dir / "tuned_best_config.yaml"
    _write_yaml(tuned_cfg_path, tuned_cfg)
    rows.append(
        _evaluate_named_run(
            name="tuned_best_config",
            experiment_type="best_config_rerun",
            infer_script=infer_script,
            inference_cwd=inference_cwd,
            config=tuned_cfg_path,
            video=video,
            source_video=source_video,
            gt=gt,
            model=model,
            run_dir=out_dir,
            iou=iou,
            seed=seed,
            n_explore=n,
            n_refine=refine_n,
        )
    )

    summary = out_dir / "summary.csv"
    with open(summary, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary CSV: {summary}")
    return summary
