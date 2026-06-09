"""Repository layout helpers."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Directory that contains `deepSORT_rtdetr.py` and `inference_parameter/`."""
    return Path(__file__).resolve().parent.parent


def inference_dir() -> Path:
    return repo_root() / "inference_parameter"


def data_dir() -> Path:
    """Dataset storage (videos, ground truth, etc.)."""
    return repo_root() / "data"


def runs_dir() -> Path:
    """Run outputs (trial sweeps, leaderboards, etc.)."""
    return repo_root() / "runs"


def camera_data_dir(camera: str = "camera_1") -> Path:
    return data_dir() / camera


def camera_videos_dir(camera: str = "camera_1") -> Path:
    return camera_data_dir(camera) / "videos"


def default_trials_dir() -> Path:
    """Default sweep output directory."""
    return inference_dir() / "trials" / "camera_1"


def default_ground_truth() -> Path:
    """First existing candidate under `inference_parameter/camera_1/`."""
    cam = inference_dir() / "camera_1"
    for name in ("ground_truth.csv", "gt.csv"):
        p = cam / name
        if p.is_file():
            return p
    return cam / "ground_truth.csv"


def default_eval_video() -> Path | None:
    """
    Default clip for sweeps: `camera_1/trim5.mp4`, or the only `.mp4` in that folder,
    or `<stem>.mp4` matching a single `*_rtdetr.csv` predictions file.
    Returns None if no candidate file exists on disk.
    """
    cam = inference_dir() / "camera_1"
    trim5 = cam / "trim5.mp4"
    if trim5.is_file():
        return trim5
    mp4s = sorted(cam.glob("*.mp4"))
    if len(mp4s) == 1:
        return mp4s[0]
    for pred in sorted(cam.glob("*_rtdetr.csv")):
        base = pred.name
        if not base.endswith("_rtdetr.csv"):
            continue
        stem = base[: -len("_rtdetr.csv")]
        if not stem:
            continue
        cand = cam / f"{stem}.mp4"
        if cand.is_file():
            return cand
    return None


def resolve_ground_truth(path: Path | None) -> Path:
    if path is not None:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Ground truth CSV not found: {p}")
        return p
    d = default_ground_truth()
    if not d.is_file():
        raise FileNotFoundError(f"No default GT CSV at {d}. Pass --gt.")
    return d


def resolve_eval_video(path: Path | None) -> Path:
    if path is not None:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Video not found: {p}")
        return p
    dv = default_eval_video()
    if dv is None or not dv.is_file():
        cam = inference_dir() / "camera_1"
        raise FileNotFoundError(
            f"No default video in {cam} (e.g. trim5.mp4). Pass --video."
        )
    return dv.resolve()
