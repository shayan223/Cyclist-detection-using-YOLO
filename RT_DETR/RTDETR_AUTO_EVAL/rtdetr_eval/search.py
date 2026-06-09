"""
Hyperparameter search for RT-DETR + DeepSORT inference configs.

Two-phase warm-start local search:

  Phase 1 (exploration)  -- run_exploration(): broad random sampling across the
                            full search space -> explore_NNN.yaml.
  Phase 2 (exploitation) -- run_exploitation(): Gaussian perturbation around the
                            top-K configs from Phase 1 (score-weighted budget)
                            -> exploit_NNN.yaml.

Tuning only changes hyperparameters in YAML; it never retrains the detector.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import yaml

from rtdetr_eval.paths import default_trials_dir

SEARCH_SPACE = [
    {"path": "inference.confidence", "type": "float", "low": 0.40, "high": 0.85, "dp": 2},
    {"path": "inference.iou", "type": "float", "low": 0.50, "high": 0.85, "dp": 2},
    {"path": "inference.downscale_width", "type": "choice", "choices": [640, 800, 960, 1280]},
    {"path": "passes.top_region.ratio", "type": "float", "low": 0.40, "high": 0.70, "dp": 2},
    {"path": "nms.hard_iou", "type": "float", "low": 0.15, "high": 0.40, "dp": 2},
    {"path": "nms.containment_fraction", "type": "float", "low": 0.35, "high": 0.65, "dp": 2},
    {"path": "nms.soft_nms_iou", "type": "float", "low": 0.15, "high": 0.35, "dp": 2},
    {"path": "nms.soft_nms_sigma", "type": "float", "low": 0.10, "high": 0.40, "dp": 2},
    {"path": "classes.0.min_confidence", "type": "float", "low": 0.45, "high": 0.80, "dp": 2},
    {"path": "classes.1.min_confidence", "type": "float", "low": 0.45, "high": 0.75, "dp": 2},
    {"path": "classes.0.tracker.max_age", "type": "int", "low": 5, "high": 25},
    {"path": "classes.0.tracker.n_init", "type": "int", "low": 1, "high": 5},
    {"path": "classes.0.tracker.max_iou_distance", "type": "float", "low": 0.60, "high": 0.95, "dp": 2},
    {"path": "classes.0.tracker.max_cosine_distance", "type": "float", "low": 0.40, "high": 0.80, "dp": 2},
    {"path": "classes.1.tracker.max_age", "type": "int", "low": 8, "high": 40},
    {"path": "classes.1.tracker.n_init", "type": "int", "low": 2, "high": 6},
    {"path": "classes.1.tracker.max_iou_distance", "type": "float", "low": 0.30, "high": 0.65, "dp": 2},
    {"path": "classes.1.tracker.max_cosine_distance", "type": "float", "low": 0.20, "high": 1.00, "dp": 2},
]

RESOLUTION_PAIRS = {640: 360, 800: 450, 960: 540, 1280: 720}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _apply_constraints(sample: dict) -> dict:
    """Enforce cross-param constraints shared by random and local sampling."""
    floor = sample.get("inference.confidence", 0.60)
    for key in ("classes.0.min_confidence", "classes.1.min_confidence"):
        if key in sample and sample[key] < floor:
            sample[key] = round(floor, 2)

    if "inference.downscale_width" in sample:
        sample["inference.downscale_height"] = RESOLUTION_PAIRS[sample["inference.downscale_width"]]
    return sample


def sample_random() -> dict:
    """Phase 1: draw a config uniformly across the full search space."""
    sample = {}
    for param in SEARCH_SPACE:
        p = param["path"]
        if param["type"] == "float":
            sample[p] = round(random.uniform(param["low"], param["high"]), param.get("dp", 3))
        elif param["type"] == "int":
            sample[p] = random.randint(param["low"], param["high"])
        elif param["type"] == "choice":
            sample[p] = random.choice(param["choices"])
    return _apply_constraints(sample)


def sample_local(seed_values: dict, sigma_frac: float = 0.15) -> dict:
    """Phase 2: Gaussian perturbation around a seed config.

    For each continuous/integer param, draw N(seed, sigma_frac * (high - low)),
    clip to bounds, and round. Categorical params (resolution) are frozen to the
    seed value so the optimizer only moves the continuous knobs.
    """
    sample = {}
    for param in SEARCH_SPACE:
        p = param["path"]
        base = seed_values.get(p)

        if param["type"] == "choice":
            sample[p] = base if base is not None else random.choice(param["choices"])
            continue

        low, high = param["low"], param["high"]
        if base is None:
            base = random.uniform(low, high) if param["type"] == "float" else random.randint(low, high)

        sigma = sigma_frac * (high - low)
        val = max(low, min(high, random.gauss(base, sigma)))
        if param["type"] == "int":
            sample[p] = int(round(val))
        else:
            sample[p] = round(val, param.get("dp", 3))

    return _apply_constraints(sample)


# ---------------------------------------------------------------------------
# Nested dict helpers
# ---------------------------------------------------------------------------

def set_nested(d: dict, dotpath: str, value):
    keys = dotpath.split(".")
    obj = d
    for key in keys[:-1]:
        obj = obj[int(key)] if key.isdigit() else obj[key]
    last = keys[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        obj[last] = value


def get_nested(d: dict, dotpath: str):
    keys = dotpath.split(".")
    obj = d
    for key in keys:
        obj = obj[int(key)] if key.isdigit() else obj[key]
    return obj


def apply_sample(base_cfg: dict, sample: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for path, value in sample.items():
        try:
            set_nested(cfg, path, value)
        except (KeyError, IndexError, TypeError) as exc:
            print(f"  [warn] Cannot set {path}: {exc}")
    return cfg


# ---------------------------------------------------------------------------
# Trial writing
# ---------------------------------------------------------------------------

def _write_trials(base_cfg: dict, samples: list[dict], out_dir: Path, prefix: str) -> None:
    """Write a list of sampled configs as `{prefix}_NNN.yaml` and update manifest."""
    manifest_path = out_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    existing = sorted(out_dir.glob(f"{prefix}_*.yaml"))
    start_n = len(existing) + 1
    if existing:
        print(f"Found {len(existing)} existing {prefix} configs - starting from {prefix}_{start_n:03d}")

    print(f"Writing {len(samples)} {prefix} configs -> {out_dir}/")
    print("-" * 60)
    for offset, sample in enumerate(samples):
        trial_id = f"{prefix}_{start_n + offset:03d}"
        cfg = apply_sample(base_cfg, sample)
        cfg["input"] = ""
        cfg["output"] = ""

        with open(out_dir / f"{trial_id}.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        manifest[trial_id] = sample
        conf = sample.get("inference.confidence", "?")
        p_min = sample.get("classes.1.min_confidence", "?")
        p_age = sample.get("classes.1.tracker.max_age", "?")
        res = sample.get("inference.downscale_width", "?")
        print(f"  {trial_id}   conf={conf}  ped_min={p_min}  ped_age={p_age}  res={res}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")


# ---------------------------------------------------------------------------
# Phase 1: exploration
# ---------------------------------------------------------------------------

def run_exploration(base_config: Path, out_dir: Path, n: int, seed: int | None) -> None:
    """Phase 1: broad random sweep -> explore_NNN.yaml."""
    if seed is not None:
        random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config) as f:
        base_cfg = yaml.safe_load(f)

    samples = [sample_random() for _ in range(n)]
    _write_trials(base_cfg, samples, out_dir, prefix="explore")

    print("\nNext step:")
    print(
        f"  python -m rtdetr_eval run-trials \\\n"
        f"      --trials-dir {out_dir} \\\n"
        f"      --video <path/to/video.mp4>"
    )


# Backwards-compatible alias.
def generate_trials(base_config: Path, out_dir: Path, n: int, seed: int | None) -> None:
    run_exploration(base_config, out_dir, n, seed)


# ---------------------------------------------------------------------------
# Phase 2: exploitation
# ---------------------------------------------------------------------------

def allocate_budget(n: int, scores: list[float], top_k: int) -> list[int]:
    """Split `n` perturbations across the top-K seeds, weighted by score.

    Every included seed gets at least 1; the remainder is distributed by score
    using the largest-remainder method so the counts sum exactly to `n`.
    """
    k = min(top_k, len(scores))
    if k <= 0:
        return []
    scores = list(scores[:k])

    if n <= k:
        # Not enough budget for one each - fund the best `n` seeds.
        return [1 if i < n else 0 for i in range(k)]

    counts = [1] * k
    remaining = n - k

    total = sum(s for s in scores if s > 0)
    weights = [(s / total if s > 0 else 0.0) for s in scores] if total > 0 else [1.0 / k] * k

    raw = [w * remaining for w in weights]
    floors = [int(math.floor(r)) for r in raw]
    for i in range(k):
        counts[i] += floors[i]

    leftover = remaining - sum(floors)
    order = sorted(range(k), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    for i in range(leftover):
        counts[order[i % k]] += 1
    return counts


def _load_top_seeds(top_params: Path, top_k: int) -> list[dict]:
    """Read top_params.json -> list of {trial, score, params} (best first)."""
    data = json.loads(Path(top_params).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{top_params} is empty or not a list of seeds.")
    return data[:top_k]


def run_exploitation(
    base_config: Path,
    top_params: Path,
    out_dir: Path,
    n: int,
    sigma_frac: float = 0.15,
    top_k: int = 3,
    seed: int | None = None,
) -> None:
    """Phase 2: score-weighted Gaussian perturbation around the top-K seeds."""
    if seed is not None:
        random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config) as f:
        base_cfg = yaml.safe_load(f)

    seeds = _load_top_seeds(top_params, top_k)
    scores = [float(s.get("score", 0.0)) for s in seeds]
    counts = allocate_budget(n, scores, top_k)

    print(f"Exploitation: {n} perturbations around top-{len(seeds)} (sigma={sigma_frac})")
    for s, c in zip(seeds, counts):
        print(f"  seed {s.get('trial', '?')}  score={s.get('score', 0.0):.4f}  -> {c} configs")

    samples: list[dict] = []
    for s, c in zip(seeds, counts):
        params = s.get("params", {})
        samples.extend(sample_local(params, sigma_frac) for _ in range(c))

    _write_trials(base_cfg, samples, out_dir, prefix="exploit")

    print("\nNext step:")
    print(
        f"  python -m rtdetr_eval run-trials \\\n"
        f"      --trials-dir {out_dir} \\\n"
        f"      --video <path/to/video.mp4>"
    )


# ---------------------------------------------------------------------------
# CLI (standalone use; the unified CLI lives in rtdetr_eval.cli)
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate randomised (explore) trial configs.")
    p.add_argument("--config", required=True, help="Base YAML template")
    p.add_argument("--n", type=int, default=50, help="Number of trials (default: 50)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {default_trials_dir()})",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    args = p.parse_args()
    out = args.out_dir if args.out_dir is not None else default_trials_dir()
    run_exploration(Path(args.config).resolve(), out.resolve(), args.n, args.seed)


if __name__ == "__main__":
    main()
