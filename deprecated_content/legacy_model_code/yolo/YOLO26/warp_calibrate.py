"""
warp_calibrate.py — Interactive perspective calibration for the warp detection pass.

Usage:
    # Use the same video as configured in config.yaml (top-level `input:`)
    python warp_calibrate.py

    # Explicitly override the input video and frame
    python warp_calibrate.py --input ../trim4.mp4 --frame 60 --downscale-width 960 --downscale-height 540

Click 4 points that form a rectangle in REAL-WORLD space (e.g. the four corners
of the bike lane, or a section of road markings you know are parallel).

Click in ORDER: top-left → top-right → bottom-right → bottom-left  (clockwise).

Keys:
    r      — reset points and start again
    u      — undo last point
    p      — preview the warp with current points (requires 4 points)
    Enter  — confirm and print config values
    q      — quit without saving
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------

POINT_COLOUR  = (0, 255, 0)
LINE_COLOUR   = (0, 200, 0)
LABEL_COLOUR  = (0, 255, 255)
FONT          = cv2.FONT_HERSHEY_SIMPLEX
CORNER_LABELS = ["TL", "TR", "BR", "BL"]

state: dict = {"points": [], "base_frame": None, "display_frame": None}


def _redraw():
    frame = state["base_frame"].copy()
    pts   = state["points"]
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (x, y), 6, POINT_COLOUR, -1)
        cv2.putText(frame, CORNER_LABELS[i], (x + 8, y - 8), FONT, 0.6, LABEL_COLOUR, 2)
    if len(pts) >= 2:
        for i in range(1, len(pts)):
            cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), LINE_COLOUR, 2)
    if len(pts) == 4:
        cv2.line(frame, tuple(pts[3]), tuple(pts[0]), LINE_COLOUR, 2)
        cv2.putText(frame, "Press Enter to confirm, 'p' to preview, 'r' to reset",
                    (10, frame.shape[0] - 10), FONT, 0.5, (255, 255, 255), 1)
    else:
        remaining = 4 - len(pts)
        cv2.putText(frame, f"Click {remaining} more point(s) — {CORNER_LABELS[len(pts)]} next",
                    (10, frame.shape[0] - 10), FONT, 0.55, (200, 200, 200), 1)
    state["display_frame"] = frame
    cv2.imshow("warp_calibrate", frame)


def _on_mouse(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < 4:
        state["points"].append([x, y])
        _redraw()


def _preview_warp(dst_w, dst_h):
    pts = state["points"]
    if len(pts) != 4:
        print("Need 4 points to preview.")
        return
    src = np.array(pts, dtype=np.float32)
    dst = np.array([[0, 0], [dst_w - 1, 0],
                    [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
    H       = cv2.getPerspectiveTransform(src, dst)
    warped  = cv2.warpPerspective(state["base_frame"], H, (dst_w, dst_h))
    cv2.imshow("Warp preview (press any key to close)", warped)
    cv2.waitKey(0)
    cv2.destroyWindow("Warp preview (press any key to close)")


def main():
    parser = argparse.ArgumentParser(description="Perspective warp calibration helper.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to RT-DETR config.yaml (used to pick default input video).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input video path. If omitted, uses the `input` field from config.yaml.",
    )
    parser.add_argument("--frame",            type=int, default=30, help="Frame index to use.")
    parser.add_argument("--downscale-width",  type=int, default=960)
    parser.add_argument("--downscale-height", type=int, default=540)
    parser.add_argument("--dst-width",  type=int, default=960,  help="Warp output width.")
    parser.add_argument("--dst-height", type=int, default=480,  help="Warp output height.")
    args = parser.parse_args()

    # Resolve input video from CLI or config.yaml
    input_path = args.input
    if input_path is None:
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            sys.exit(f"Config file not found: {cfg_path}")
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:  # pragma: no cover - defensive
            sys.exit(f"Failed to load YAML config '{cfg_path}': {exc}")

        input_from_cfg = cfg.get("input")
        if not input_from_cfg:
            sys.exit(f"'input' not found in config: {cfg_path}")

        # Interpret the path relative to the config file location, mirroring runtime
        input_path = str((cfg_path.parent / input_from_cfg).resolve())
        print(f"[warp_calibrate] Using input from config.yaml: {input_from_cfg} -> {input_path}")
    else:
        input_path = str(Path(input_path).resolve())

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {input_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = min(args.frame, max(0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("Could not read frame.")

    # Match the downscale used during inference
    if args.downscale_width > 0 and args.downscale_height > 0:
        fh, fw = frame.shape[:2]
        if fw > args.downscale_width or fh > args.downscale_height:
            frame = cv2.resize(frame, (args.downscale_width, args.downscale_height))

    state["base_frame"]    = frame.copy()
    state["display_frame"] = frame.copy()

    cv2.namedWindow("warp_calibrate")
    cv2.setMouseCallback("warp_calibrate", _on_mouse)

    print(__doc__)
    print(f"Frame size: {frame.shape[1]}×{frame.shape[0]}  |  "
          f"Using frame #{frame_idx} of {total}")
    _redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            state["points"] = []
            _redraw()
        elif key == ord("u") and state["points"]:
            state["points"].pop()
            _redraw()
        elif key == ord("p"):
            _preview_warp(args.dst_width, args.dst_height)
        elif key in (13, ord("\r"), ord("\n")) and len(state["points"]) == 4:
            break
        elif key == ord("q"):
            print("Cancelled.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    pts = state["points"]
    print("\n" + "=" * 60)
    print("Add / update the following in config.yaml under passes.warp:")
    print("=" * 60)
    print("  warp:")
    print("    enabled: true")
    print("    src_points:")
    for p in pts:
        print(f"      - [{p[0]}, {p[1]}]")
    print(f"    dst_size: [{args.dst_width}, {args.dst_height}]")
    print("    confidence: null")
    print("    imgsz: 0")
    print("=" * 60)
    print("\nTip: press 'p' before confirming to visually verify the warp looks correct.")
    print("     Aim for parallel road lines becoming parallel in the preview.")


if __name__ == "__main__":
    main()
