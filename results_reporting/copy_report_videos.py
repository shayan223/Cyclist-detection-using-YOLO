#!/usr/bin/env python3
"""Copy the three suite inference videos into the report outputs folder."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "RT_DETR"
    / "RTDETR_AUTO_EVAL"
    / "runs"
    / "camera_1_trim5"
    / "20260608_220235"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs" / "videos"
REPORT_VIDEOS = ["bare_bones.mp4", "manual_trim5.mp4", "tuned_best_config.mp4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy bare-bones, manual, and tuned inference videos into the report folder."
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
        help=f"Video output directory (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def copy_videos(run_dir: Path, out_dir: Path) -> list[Path]:
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for video_name in REPORT_VIDEOS:
        src = require_file(
            run_dir / "videos" / video_name,
            f"report inference video {video_name}",
        )
        dst = out_dir / video_name
        shutil.copy2(src, dst)
        written.append(dst)

    manifest_path = out_dir / "video_manifest.json"
    manifest = {
        "copied_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": relative(run_dir / "videos"),
        "videos": [
            {
                "name": name,
                "source": relative(run_dir / "videos" / name),
                "report_copy": relative(out_dir / name),
                "bytes": int((out_dir / name).stat().st_size),
            }
            for name in REPORT_VIDEOS
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(manifest_path)
    return written


def main() -> None:
    args = parse_args()
    written = copy_videos(args.run_dir, args.out_dir)
    print(f"Copied {len(REPORT_VIDEOS)} inference videos into {args.out_dir.resolve()}")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
