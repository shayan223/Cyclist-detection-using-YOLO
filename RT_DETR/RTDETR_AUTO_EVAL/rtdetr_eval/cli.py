"""Unified CLI: explore configs, run trials, refine (warm-start), evaluate, or full pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from rtdetr_eval.evaluate import run_evaluation
from rtdetr_eval.search import run_exploitation, run_exploration
from rtdetr_eval.suite import run_suite
from rtdetr_eval.paths import (
    default_eval_video,
    default_ground_truth,
    default_trials_dir,
    inference_dir,
    repo_root,
    resolve_eval_video,
    resolve_ground_truth,
)
from rtdetr_eval.trials import run_trials


def _add_run_args(sp) -> None:
    """Shared inference/eval args for commands that run trials."""
    sp.add_argument("--video", type=Path, default=None, help="Input video (default: resolved per paths.py)")
    sp.add_argument("--gt", type=Path, default=None, help="Ground-truth CSV (default: resolved per paths.py)")
    sp.add_argument("--iou", type=float, default=0.5)
    sp.add_argument("--infer-script", type=Path, default=None)
    sp.add_argument("--inference-cwd", type=Path, default=None)
    sp.add_argument("--best-out", type=Path, default=None)
    sp.add_argument("--eval-plots", action="store_true")
    sp.add_argument("--top-k", type=int, default=3, help="How many top seeds to record for warm-start (default: 3)")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rtdetr_eval",
        description="RT-DETR hyperparameter search: explore, run, refine (warm-start), evaluate vs GT.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # Phase 1: exploration ("generate" kept as a hidden alias for back-compat).
    explore_kwargs = [("explore", {"help": "Sample random trial YAMLs (Phase 1)"}), ("generate", {})]
    for name, kw in explore_kwargs:
        g = sub.add_parser(name, **kw)
        g.add_argument("--config", type=Path, required=True, help="Base YAML template")
        g.add_argument("--n", type=int, default=50, help="Number of new trials")
        g.add_argument(
            "--out-dir",
            type=Path,
            default=None,
            help=f"Directory for explore_*.yaml + manifest.json (default: {default_trials_dir()})",
        )
        g.add_argument("--seed", type=int, default=None)

    r = sub.add_parser("run-trials", help="Run each trial on a video and write best_config.yaml")
    r.add_argument(
        "--trials-dir",
        type=Path,
        default=None,
        help=f"Trial YAML directory (default: {default_trials_dir()})",
    )
    _add_run_args(r)

    # Phase 2: exploitation.
    rf = sub.add_parser("refine", help="Warm-start: perturb around the top-K seeds (Phase 2)")
    rf.add_argument("--config", type=Path, required=True, help="Base YAML template")
    rf.add_argument(
        "--trials-dir",
        type=Path,
        default=None,
        help=f"Directory holding top_params.json + exploit output (default: {default_trials_dir()})",
    )
    rf.add_argument(
        "--top",
        type=Path,
        default=None,
        help="top_params.json from a previous run-trials (default: <trials-dir>/top_params.json)",
    )
    rf.add_argument("--n", type=int, default=30, help="Total perturbations to generate")
    rf.add_argument("--sigma", type=float, default=0.15, help="Gaussian std as fraction of each param range")
    rf.add_argument("--top-k", type=int, default=3, help="Number of top seeds to perturb around")
    rf.add_argument("--out-dir", type=Path, default=None, help="Where to write exploit_*.yaml (default: --trials-dir)")
    rf.add_argument("--seed", type=int, default=None)

    e = sub.add_parser("eval", help="Single GT vs predictions evaluation (+ plots)")
    e.add_argument("--gt", type=Path, required=True)
    e.add_argument("--pred", type=Path, required=True)
    e.add_argument("--out", type=Path, default=Path("."))
    e.add_argument("--iou", type=float, default=0.5)

    f = sub.add_parser("full", help="explore then run-trials (one command)")
    f.add_argument("--config", type=Path, required=True, help="Base YAML template")
    f.add_argument("--trials-dir", type=Path, default=None, help=f"Trial YAML directory (default: {default_trials_dir()})")
    f.add_argument("--n", type=int, default=50)
    f.add_argument("--seed", type=int, default=None)
    _add_run_args(f)

    # One-command warm-start: explore -> run -> refine -> run.
    s = sub.add_parser("search", help="Two-phase warm-start: explore -> run -> refine -> run")
    s.add_argument("--config", type=Path, required=True, help="Base YAML template")
    s.add_argument("--trials-dir", type=Path, default=None, help=f"Trial YAML directory (default: {default_trials_dir()})")
    s.add_argument("--n", type=int, default=50, help="Phase 1 exploration trials")
    s.add_argument("--refine-n", type=int, default=30, help="Phase 2 exploitation trials")
    s.add_argument("--sigma", type=float, default=0.15, help="Phase 2 Gaussian std as fraction of range")
    s.add_argument("--seed", type=int, default=None)
    _add_run_args(s)

    suite = sub.add_parser("suite", help="Run baseline, manual config, and tuned search experiments")
    suite.add_argument("--model", type=Path, required=True, help="RT-DETR model checkpoint")
    suite.add_argument("--video", type=Path, required=True, help="Evaluation video")
    suite.add_argument("--gt", type=Path, required=True, help="Ground-truth CSV")
    suite.add_argument("--manual-config", type=Path, required=True, help="Hand-picked YAML config")
    suite.add_argument("--infer-script", type=Path, required=True, help="deepSORT_rtdetr.py to use")
    suite.add_argument("--inference-cwd", type=Path, required=True, help="Working directory for inference")
    suite.add_argument("--out-dir", type=Path, default=None, help="Output directory for suite artifacts")
    suite.add_argument("--n", type=int, default=50, help="Phase 1 exploration trials")
    suite.add_argument("--refine-n", type=int, default=30, help="Phase 2 exploitation trials")
    suite.add_argument("--seed", type=int, default=42)
    suite.add_argument("--iou", type=float, default=0.5)
    suite.add_argument("--sigma", type=float, default=0.15)
    suite.add_argument("--top-k", type=int, default=3)
    suite.add_argument("--eval-plots", action="store_true", help="Write plots for every search trial")

    sub.add_parser("print-paths", help="Show resolved repo defaults")

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "print-paths":
        root = repo_root()
        print(f"repo_root:          {root}")
        print(f"inference_parameter: {inference_dir()}")
        print(f"default --gt:         {default_ground_truth()}")
        print(f"default trials dir:   {default_trials_dir()}")
        dv = default_eval_video()
        print(f"default --video:      {dv if dv else '(none - add trim5.mp4 under data/camera_1/videos/ or pass --video)'}")
        print(f"deepSORT_rtdetr:      {root / 'deepSORT_rtdetr.py'}")
        return

    if args.command in ("explore", "generate"):
        out_dir = args.out_dir if args.out_dir is not None else default_trials_dir()
        run_exploration(args.config.resolve(), out_dir.resolve(), args.n, args.seed)
        return

    if args.command == "eval":
        run_evaluation(args.gt, args.pred, args.out.resolve(), args.iou, plots=True, write_json=True)
        return

    if args.command == "refine":
        trials_dir = (args.trials_dir if args.trials_dir is not None else default_trials_dir()).resolve()
        out_dir = (args.out_dir if args.out_dir is not None else trials_dir).resolve()
        top = (args.top if args.top is not None else trials_dir / "top_params.json").resolve()
        run_exploitation(
            args.config.resolve(), top, out_dir, args.n,
            sigma_frac=args.sigma, top_k=args.top_k, seed=args.seed,
        )
        return

    if args.command == "run-trials":
        if args.trials_dir is None:
            args.trials_dir = default_trials_dir()
        gt = resolve_ground_truth(args.gt)
        video = resolve_eval_video(args.video)
        run_trials(
            args.trials_dir.resolve(),
            video,
            gt,
            infer_script=args.infer_script,
            inference_cwd=args.inference_cwd,
            iou_thresh=args.iou,
            best_out=args.best_out,
            eval_plots=args.eval_plots,
            top_params_k=args.top_k,
        )
        return

    if args.command == "full":
        if args.trials_dir is None:
            args.trials_dir = default_trials_dir()
        gt = resolve_ground_truth(args.gt)
        video = resolve_eval_video(args.video)
        trials_dir = args.trials_dir.resolve()
        run_exploration(args.config.resolve(), trials_dir, args.n, args.seed)
        run_trials(
            trials_dir, video, gt,
            infer_script=args.infer_script,
            inference_cwd=args.inference_cwd,
            iou_thresh=args.iou,
            best_out=args.best_out,
            eval_plots=args.eval_plots,
            top_params_k=args.top_k,
        )
        return

    if args.command == "search":
        if args.trials_dir is None:
            args.trials_dir = default_trials_dir()
        gt = resolve_ground_truth(args.gt)
        video = resolve_eval_video(args.video)
        trials_dir = args.trials_dir.resolve()
        config = args.config.resolve()

        common = dict(
            infer_script=args.infer_script,
            inference_cwd=args.inference_cwd,
            iou_thresh=args.iou,
            best_out=args.best_out,
            eval_plots=args.eval_plots,
            top_params_k=args.top_k,
        )

        print("\n=== Phase 1: exploration ===")
        run_exploration(config, trials_dir, args.n, args.seed)
        run_trials(trials_dir, video, gt, **common)

        print("\n=== Phase 2: exploitation (warm-start) ===")
        top = trials_dir / "top_params.json"
        run_exploitation(
            config, top, trials_dir, args.refine_n,
            sigma_frac=args.sigma, top_k=args.top_k, seed=args.seed,
        )
        run_trials(trials_dir, video, gt, **common)
        return

    if args.command == "suite":
        summary = run_suite(
            model=args.model,
            video=args.video,
            gt=args.gt,
            manual_config=args.manual_config,
            infer_script=args.infer_script,
            inference_cwd=args.inference_cwd,
            out_dir=args.out_dir,
            n=args.n,
            refine_n=args.refine_n,
            seed=args.seed,
            iou=args.iou,
            sigma=args.sigma,
            top_k=args.top_k,
            eval_plots=args.eval_plots,
        )
        print(f"Suite complete: {summary}")
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
