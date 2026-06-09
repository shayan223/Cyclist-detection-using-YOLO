#!/usr/bin/env python3
"""
Visualize the two-phase warm-start search e2e workflow + optimization result.

Runs the real two-phase pipeline (explore -> run-trials -> refine -> run-trials)
against the fake polynomial model (tests/fakes), then renders three artifacts
into viz/artifacts/:

  1. e2e_workflow.png      - the pipeline stages / data flow
  2. loss_landscape_3d.png - 3D loss surface (1 - quality) with trials overlaid
  3. convergence.png       - running-best score across explore -> exploit trials

Usage:
    python3 viz/visualize_e2e.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from rtdetr_eval.search import run_exploitation, run_exploration
from rtdetr_eval.trials import run_trials
from tests.fakes import scene

FAKES = REPO_ROOT / "tests" / "fakes"
BASE_YAML = FAKES / "base.yaml"
FAKE_INFER = FAKES / "fake_infer.py"
ART = Path(__file__).resolve().parent / "artifacts"

SEED = 50
N_EXPLORE = 24
N_EXPLOIT = 16
TOP_K = 3

CONF = "inference.confidence"
PMIN = "classes.1.min_confidence"
HIOU = "nms.hard_iou"


# ---------------------------------------------------------------------------
# Run the real workflow against the fake model
# ---------------------------------------------------------------------------

def run_workflow(work: Path) -> tuple[dict, dict]:
    trials = work / "trials"
    base = work / "base.yaml"
    shutil.copyfile(BASE_YAML, base)
    video = work / "clip.mp4"
    video.touch()
    gt = work / "ground_truth.csv"
    scene.write_gt_csv(gt, scene.N_FRAMES)

    def _trials():
        run_trials(
            trials, video, gt,
            infer_script=FAKE_INFER, inference_cwd=work,
            iou_thresh=0.5, eval_plots=False, top_params_k=TOP_K,
        )

    run_exploration(base, trials, n=N_EXPLORE, seed=SEED)
    _trials()
    run_exploitation(base, trials / "top_params.json", trials,
                     n=N_EXPLOIT, sigma_frac=0.15, top_k=TOP_K, seed=SEED)
    _trials()

    manifest = json.loads((trials / "manifest.json").read_text())
    leaderboard = json.loads((trials / "trial_leaderboard.json").read_text())
    return manifest, leaderboard


# ---------------------------------------------------------------------------
# Figure 1: workflow diagram
# ---------------------------------------------------------------------------

def plot_workflow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Two-phase warm-start search - e2e workflow", fontsize=14, weight="bold")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    linewidth=1.4, edgecolor="#333", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.3, color="#555"))

    explore_c = "#dbeafe"
    run_c = "#fde68a"
    out_c = "#bbf7d0"
    refine_c = "#fbcfe8"

    # Phase 1 row (top)
    box(0.3, 4.2, 2.0, 1.0, "base.yaml\n(template)", "#e5e7eb")
    box(2.8, 4.2, 2.0, 1.0, "explore\nsample_random", explore_c)
    box(5.3, 4.2, 2.2, 1.0, "explore_*.yaml\n+ manifest", explore_c)
    box(8.0, 4.2, 2.2, 1.0, "run-trials\nfake_infer + REAL eval", run_c)
    box(10.5, 4.2, 1.3, 1.0, "best_config\ntop_params", out_c)
    for (x1, x2) in [(2.3, 2.8), (4.8, 5.3), (7.5, 8.0), (10.2, 10.5)]:
        arrow(x1, 4.7, x2, 4.7)

    # down to phase 2
    arrow(11.15, 4.2, 11.15, 2.4)

    # Phase 2 row (bottom, right-to-left)
    box(9.9, 1.4, 2.2, 1.0, "refine\nGaussian top-3\n(score-weighted)", refine_c)
    box(6.8, 1.4, 2.2, 1.0, "exploit_*.yaml", refine_c)
    box(3.8, 1.4, 2.2, 1.0, "run-trials\n(explore + exploit)", run_c)
    box(1.0, 1.4, 2.2, 1.0, "final best_config\n+ leaderboard", out_c)
    for (x1, x2) in [(9.9, 9.0), (6.8, 6.0), (3.8, 3.2)]:
        arrow(x1, 1.9, x2, 1.9)

    ax.text(6, 3.25, "Phase 2 re-globs explore_ + exploit_  ->  leaderboard is a superset (best can only improve)",
            ha="center", va="center", fontsize=8.5, style="italic", color="#444")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: 3D loss landscape
# ---------------------------------------------------------------------------

def plot_landscape(path: Path, manifest: dict, score_by_trial: dict) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    c_lo, c_hi = scene.BOUNDS[CONF]
    p_lo, p_hi = scene.BOUNDS[PMIN]
    cc = np.linspace(c_lo, c_hi, 60)
    pp = np.linspace(p_lo, p_hi, 60)
    C, P = np.meshgrid(cc, pp)

    # Loss slice with hard_iou pinned at its optimum.
    Z = np.zeros_like(C)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            q = scene.quality({CONF: C[i, j], PMIN: P[i, j], HIOU: scene.OPTIMUM[HIOU]})
            Z[i, j] = 1.0 - q

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(C, P, Z, cmap="viridis", alpha=0.65, linewidth=0, antialiased=True)
    ax.contour(C, P, Z, zdir="z", offset=Z.min(), levels=12, cmap="viridis", alpha=0.5)

    def scatter(prefix, color, marker, label):
        xs, ys, zs = [], [], []
        for name, params in manifest.items():
            if not name.startswith(prefix):
                continue
            xs.append(params[CONF])
            ys.append(params[PMIN])
            zs.append(1.0 - scene.quality(params))
        ax.scatter(xs, ys, zs, c=color, marker=marker, s=42, depthshade=False,
                   edgecolors="k", linewidths=0.4, label=f"{label} (n={len(xs)})")

    scatter("explore_", "#3b82f6", "o", "explore")
    scatter("exploit_", "#ef4444", "^", "exploit")

    # True optimum.
    opt_loss = 1.0 - scene.quality(scene.OPTIMUM)
    ax.scatter([scene.OPTIMUM[CONF]], [scene.OPTIMUM[PMIN]], [opt_loss],
               c="gold", marker="*", s=420, edgecolors="k", linewidths=0.8, label="true optimum")

    # Best found.
    best_name = max(score_by_trial, key=score_by_trial.get)
    bp = manifest[best_name]
    ax.scatter([bp[CONF]], [bp[PMIN]], [1.0 - scene.quality(bp)],
               c="lime", marker="X", s=200, edgecolors="k", linewidths=0.8, label=f"best found ({best_name})")

    ax.set_xlabel("inference.confidence")
    ax.set_ylabel("classes.1.min_confidence")
    ax.set_zlabel("loss = 1 - quality")
    ax.set_title("Loss landscape (hard_iou fixed at optimum) with search trials", fontsize=13, weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=28, azim=-52)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: convergence
# ---------------------------------------------------------------------------

def plot_convergence(path: Path, manifest: dict, score_by_trial: dict) -> None:
    explore = sorted(n for n in manifest if n.startswith("explore_"))
    exploit = sorted(n for n in manifest if n.startswith("exploit_"))
    order = explore + exploit
    scores = [score_by_trial.get(n, 0.0) for n in order]

    running_best, best = [], -1.0
    for s in scores:
        best = max(best, s)
        running_best.append(best)

    fig, ax = plt.subplots(figsize=(11, 5))
    idx = np.arange(len(order))
    split = len(explore)

    ax.scatter(idx[:split], scores[:split], c="#3b82f6", s=30, label="explore trial score")
    ax.scatter(idx[split:], scores[split:], c="#ef4444", marker="^", s=34, label="exploit trial score")
    ax.plot(idx, running_best, color="#16a34a", linewidth=2.2, label="running best")
    ax.axvline(split - 0.5, color="#888", linestyle="--", linewidth=1)
    ax.text(split - 0.5, ax.get_ylim()[0], "  phase 1 | phase 2", va="bottom", ha="left", fontsize=9, color="#555")

    ax.set_xlabel("trial order (explore -> exploit)")
    ax.set_ylabel("trial score")
    ax.set_title("Optimization convergence: warm-start exploitation tightens around the best", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        manifest, leaderboard = run_workflow(Path(td))

    score_by_trial = {row["trial"]: row["score"] for row in leaderboard}

    plot_workflow(ART / "e2e_workflow.png")
    plot_landscape(ART / "loss_landscape_3d.png", manifest, score_by_trial)
    plot_convergence(ART / "convergence.png", manifest, score_by_trial)

    best_name = max(score_by_trial, key=score_by_trial.get)
    print("\nArtifacts written to", ART)
    for f in ("e2e_workflow.png", "loss_landscape_3d.png", "convergence.png"):
        print("  -", ART / f)
    print(f"\nBest trial: {best_name}  score={score_by_trial[best_name]:.4f}  "
          f"distance_to_optimum={scene.param_distance(manifest[best_name]):.4f}")
    mean_explore = np.mean([scene.param_distance(p) for n, p in manifest.items() if n.startswith("explore_")])
    mean_exploit = np.mean([scene.param_distance(p) for n, p in manifest.items() if n.startswith("exploit_")])
    print(f"Mean distance to optimum:  explore={mean_explore:.4f}  exploit={mean_exploit:.4f}")


if __name__ == "__main__":
    main()
