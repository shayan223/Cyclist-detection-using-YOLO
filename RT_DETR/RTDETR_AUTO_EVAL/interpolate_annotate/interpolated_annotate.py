#!/usr/bin/env python3
"""
Video Annotation Tool — Cyclist & Pedestrian Dataset Creator
============================================================
Refactored for clarity and extended with keyframe interpolation.

New features vs. original
--------------------------
* Interpolator class: saves automatically generated labels for every frame
  between two user-annotated keyframes (gap ≤ --interpolate-frames).
* --interpolate-frames  0 | 5 | 10   (0 = disabled, default 5)
* Visual interpolation preview overlay on the paused frame.
* Clean class-based architecture: BBox · Interpolator · DatasetWriter ·
  MotionEngine · PolygonSelector · VideoAnnotator

Controls (unchanged from original)
------------------------------------
  Space          Pause / Resume
  Click box      Select / deselect detected box
  Click + drag   Draw a manual bounding box
  t              Toggle class  (cyclist ↔ pedestrian)
  s              Save keyframe (+ auto-generate interpolated frames)
  c              Clear selections
  f / b          Step ±10 frames while paused
  n / p          Next / previous CSV timestamp
  e              Toggle edge refinement
  + / -          Increase / decrease playback speed
  r              Reset to beginning
  q              Quit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: install tqdm for progress bars  (pip install tqdm)")


# ══════════════════════════════════════════════════════════════════════════════
# Visual constants (BGR)
# ══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES   = ["cyclist", "pedestrian"]
# Colors for selected auto-detected boxes
SEL_COLORS    = {0: (0, 255, 0),   1: (255, 0, 255)}
# Colors for manually drawn boxes
MAN_COLORS    = {0: (255, 255, 0), 1: (0, 255, 255)}
UNSEL_COLOR   = (255, 0, 0)   # detected but not yet selected
INTERP_COLOR  = (0, 200, 255) # interpolation preview


# ══════════════════════════════════════════════════════════════════════════════
# BBox
# ══════════════════════════════════════════════════════════════════════════════
class BBox:
    """Immutable bounding box with class label.  All coords are integers."""

    __slots__ = ("x", "y", "w", "h", "cls")

    def __init__(self, x, y, w, h, cls: int = 0):
        self.x   = int(round(x))
        self.y   = int(round(y))
        self.w   = int(round(w))
        self.h   = int(round(h))
        self.cls = int(cls)

    # ── properties ──────────────────────────────────────────────────────────
    @property
    def xywh(self)   -> Tuple: return (self.x, self.y, self.w, self.h)
    @property
    def center(self) -> Tuple: return (self.x + self.w / 2, self.y + self.h / 2)
    @property
    def area(self)   -> int:   return self.w * self.h

    # ── conversions ──────────────────────────────────────────────────────────
    def to_yolo(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        cx = (self.x + self.w / 2) / img_w
        cy = (self.y + self.h / 2) / img_h
        return cx, cy, self.w / img_w, self.h / img_h

    # ── interpolation ────────────────────────────────────────────────────────
    def lerp(self, other: "BBox", t: float) -> "BBox":
        """Linear interpolation. t=0 → self, t=1 → other."""
        return BBox(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.w + (other.w - self.w) * t,
            self.h + (other.h - self.h) * t,
            self.cls,
        )

    def __repr__(self) -> str:
        return f"BBox({self.x},{self.y},{self.w},{self.h},cls={self.cls})"

    # ── helper: build from raw tuple ────────────────────────────────────────
    @classmethod
    def from_tuple(cls, t) -> "BBox":
        """Accept (x,y,w,h) or (x,y,w,h,cls)."""
        if len(t) >= 5:
            return cls(t[0], t[1], t[2], t[3], t[4])
        return cls(t[0], t[1], t[2], t[3])


# ══════════════════════════════════════════════════════════════════════════════
# Interpolator
# ══════════════════════════════════════════════════════════════════════════════
class Interpolator:
    """
    Keyframe-based interpolation.

    Workflow
    --------
    1. User saves frame A  →  add(frame_A, image_A, boxes_A)  →  []
    2. User saves frame B  →  add(frame_B, image_B, boxes_B)
       If gap(A, B) ≤ n+1: returns [(frame, boxes), …] for A+1 … B-1
       Caller must read those video frames and hand them to DatasetWriter.

    Parameters
    ----------
    n : int
        Maximum gap (exclusive) between two keyframes to auto-interpolate.
        0 disables interpolation.  Typical: 5 or 10.
    """

    def __init__(self, n: int = 5):
        self.n   = n
        self._kf: dict[int, dict] = {}   # frame_num → {image, boxes}

    # ── public ───────────────────────────────────────────────────────────────
    def add(self, frame_num: int, image, boxes: List[BBox]) -> List[Tuple[int, List[BBox]]]:
        """Register a keyframe; return list of (frame_num, boxes) to interpolate."""
        self._kf[frame_num] = {"image": image, "boxes": boxes}
        if self.n == 0 or len(self._kf) < 2:
            return []

        # Find nearest previous keyframe within the window
        prev_candidates = sorted(fn for fn in self._kf if fn < frame_num)
        if not prev_candidates:
            return []
        prev_fn = prev_candidates[-1]
        gap = frame_num - prev_fn

        # Only interpolate if the gap is ≥2 and ≤ n
        if gap < 2 or gap > self.n:
            return []

        return self._gen(prev_fn, frame_num)

    def preview_boxes(self, frame_num: int) -> Optional[List[BBox]]:
        """
        If frame_num falls between two keyframes, return interpolated boxes
        for live overlay preview (does NOT register a new keyframe).
        """
        if self.n == 0 or len(self._kf) < 2:
            return None
        sorted_kf = sorted(self._kf)
        for i in range(len(sorted_kf) - 1):
            a, b = sorted_kf[i], sorted_kf[i + 1]
            if a < frame_num < b and (b - a) <= self.n:
                t = (frame_num - a) / (b - a)
                pairs = self._match(self._kf[a]["boxes"], self._kf[b]["boxes"])
                return self._lerp_pairs(pairs, t)
        return None

    # ── private ───────────────────────────────────────────────────────────────
    def _gen(self, fn_a: int, fn_b: int) -> List[Tuple[int, List[BBox]]]:
        gap   = fn_b - fn_a
        pairs = self._match(self._kf[fn_a]["boxes"], self._kf[fn_b]["boxes"])
        out   = []
        for step in range(1, gap):
            t = step / gap
            out.append((fn_a + step, self._lerp_pairs(pairs, t)))
        return out

    @staticmethod
    def _match(a: List[BBox], b: List[BBox], max_dist_ratio: float = 2.0) -> List[Tuple]:
        """Greedy nearest-neighbour matching by class + centre distance."""
        if not a: return [(None, bx) for bx in b]
        if not b: return [(ax, None) for ax in a]
        
        used, pairs = set(), []
        for ax in a:
            best_i, best_d = None, float("inf")
            # FIX 1: threshold based on the object's own size
            max_dist = max(ax.w, ax.h) * max_dist_ratio
            for i, bx in enumerate(b):
                if i in used or ax.cls != bx.cls:
                    continue
                d = ((ax.center[0] - bx.center[0]) ** 2 +
                    (ax.center[1] - bx.center[1]) ** 2) ** 0.5
                if d < max_dist and d < best_d:   # FIX 1: reject far candidates
                    best_d, best_i = d, i
            if best_i is not None:
                pairs.append((ax, b[best_i]))
                used.add(best_i)                  # FIX 2: mark slot as consumed
            else:
                pairs.append((ax, None))          # object left scene — no match
        for i, bx in enumerate(b):
            if i not in used:
                pairs.append((None, bx))          # new object entering scene
        return pairs


    @staticmethod
    def _lerp_pairs(pairs, t: float) -> List[BBox]:
        result = []
        for ba, bb in pairs:
            if ba and bb:
                result.append(ba.lerp(bb, t))
            elif ba:
                # Object left scene: ghost it briefly then let it disappear
                if t < 0.35:
                    result.append(ba)
            elif bb:
                # New object entering: only materialise near end of gap
                if t > 0.65:
                    result.append(bb)
        return result

# ══════════════════════════════════════════════════════════════════════════════
# Dataset Writer
# ══════════════════════════════════════════════════════════════════════════════
class DatasetWriter:
    """Saves images + YOLO labels into a fresh run_NNN folder in cwd each session."""

    def __init__(self, dataset_dir: str):
        # Always create a new run_NNN folder in the current working directory.
        # dataset_dir is ignored — the run folder is the source of truth.
        base          = self._next_run_dir()
        self.run_dir  = base
        self.img_dir  = base / "images"
        self.lbl_dir  = base / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.saved    = 0
        self.next_num = 0          # always start from 0 inside a fresh run
        self._write_yaml(base)
        print(f"  Run folder: {base.resolve()}")

    @staticmethod
    def _next_run_dir() -> Path:
        """Find the next available run_NNN directory in cwd."""
        cwd = Path.cwd()
        n   = 1
        while True:
            candidate = cwd / f"run_{n:03d}"
            if not candidate.exists():
                return candidate
            n += 1

    def save(self, image, frame_num: int, boxes: List[BBox], tag: str = "") -> str:
        name = f"frame_{frame_num:08d}.jpg"
        cv2.imwrite(str(self.img_dir / name), image)
        h, w = image.shape[:2]
        with open(self.lbl_dir / name.replace(".jpg", ".txt"), "w") as f:
            for b in boxes:
                cx, cy, bw, bh = b.to_yolo(w, h)
                f.write(f"{b.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        self.saved    += 1
        self.next_num += 1
        counts = {c: sum(1 for b in boxes if b.cls == c) for c in range(len(CLASS_NAMES))}
        label  = ", ".join(f"{v} {CLASS_NAMES[k]}" for k, v in counts.items() if v)
        prefix = f"[{tag}] " if tag else ""
        print(f"  {prefix}frame {frame_num:6d} → {name}  ({label or '0 boxes'})")
        return name

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _write_yaml(base: Path):
        (base / "data.yaml").write_text(
            "names:\n- cyclist\n- pedestrian\nnc: 2\n"
            f"train: {(base / 'images').resolve()}\n"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Motion Engine
# ══════════════════════════════════════════════════════════════════════════════
class MotionEngine:
    """
    MOG2 background subtraction + object tracking + geometric filters.
    Returns only detections that survive temporal and spatial consistency checks.
    """

    def __init__(self, cfg):
        self.cfg      = cfg
        self.bg       = cv2.createBackgroundSubtractorMOG2(
                            history=500, varThreshold=50, detectShadows=False)
        self.history  = deque(maxlen=cfg.temporal_frames)
        self.tracks   = {}
        self.track_id = 0
        self._fnum    = 0
        self._k5      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._k7      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def reset(self):
        self.history.clear()
        self.tracks.clear()
        self.track_id = 0
        self._fnum    = 0

    def detect(self, frame, mask, poly_pts, use_edges: bool = False
               ) -> Tuple[bool, List[Tuple]]:
        """
        Returns (persistent_motion: bool, boxes: list of (x,y,w,h) tuples).
        boxes is non-empty only when persistent_motion is True.
        """
        self._fnum += 1
        fg    = self._foreground(frame, mask, use_edges)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = self._cnts_to_boxes(cnts, mask, poly_pts, frame.shape[:2])
        boxes = self._track_filter(boxes)
        boxes = self._adjacent_filter(boxes)
        boxes = self._cluster_filter(boxes)
        motion = bool(boxes)
        self.history.append(motion)
        persistent = (len(self.history) >= self.cfg.temporal_frames
                      and all(self.history))
        return persistent, (boxes if persistent else [])

    # ── foreground mask ───────────────────────────────────────────────────────
    def _foreground(self, frame, mask, use_edges: bool):
        fg = self.bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._k5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k7)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k7)
        if use_edges:
            gray  = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                     if frame.ndim == 3 else frame)
            blur  = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 100, 200)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self._k5)
            dil   = cv2.dilate(cv2.bitwise_and(fg, mask), self._k5, iterations=1)
            fg    = cv2.bitwise_or(fg, cv2.bitwise_and(edges, dil))
            fg    = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k7)
        return fg

    # ── contour → box candidates ──────────────────────────────────────────────
    def _cnts_to_boxes(self, cnts, mask, poly_pts, shape: Tuple) -> List[Tuple]:
        cfg = self.cfg
        pts = np.array(poly_pts, dtype=np.int32)
        H, W = shape
        out  = []
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w / 2, y + h / 2
            corners = [(x, y), (x+w, y), (x, y+h), (x+w, y+h), (cx, cy)]
            if not any(cv2.pointPolygonTest(pts, c, False) >= 0 for c in corners):
                continue
            if w < 5 or h < 5:
                continue
            ar = max(w / h, h / w)
            if not (cfg.min_aspect_ratio <= ar <= cfg.max_aspect_ratio):
                continue
            # Area inside polygon
            cm = np.zeros((H, W), np.uint8)
            cv2.drawContours(cm, [cnt], -1, 255, -1)
            area = cv2.countNonZero(cv2.bitwise_and(cm, mask))
            if area < cfg.min_area:
                continue
            # Solidity
            fa = cv2.contourArea(cnt)
            ha = cv2.contourArea(cv2.convexHull(cnt))
            if ha > 0 and fa / ha < cfg.min_solidity:
                continue
            # Compactness
            pe = cv2.arcLength(cnt, True)
            if pe > 0 and 4 * np.pi * fa / (pe * pe) < cfg.min_compactness:
                continue
            out.append((x, y, w, h, area))
        return out

    # ── temporal object tracking ───────────────────────────────────────────────
    def _track_filter(self, boxes: List[Tuple]) -> List[Tuple]:
        fnum = self._fnum
        if not boxes:
            self.tracks = {tid: t for tid, t in self.tracks.items()
                           if fnum - t[-1]["fn"] < self.cfg.temporal_frames * 2}
            return []
        objs = [{"bb": (x, y, w, h),
                 "cx": x + w / 2, "cy": y + h / 2,
                 "area": a, "fn": fnum}
                for x, y, w, h, a in boxes]
        matched, result = set(), []
        for obj in objs:
            best_id, best_d = None, float("inf")
            for tid, hist in self.tracks.items():
                last = hist[-1]
                dx, dy = obj["cx"] - last["cx"], obj["cy"] - last["cy"]
                d    = (dx*dx + dy*dy) ** 0.5
                x, y, w, h = obj["bb"]
                ratio = max(obj["area"], last["area"]) / max(min(obj["area"], last["area"]), 1)
                if d < max(w, h) * 2 and ratio < 4 and d < best_d:
                    best_d, best_id = d, tid
            if best_id is not None:
                self.tracks[best_id].append(obj)
                matched.add(best_id)
                if len(self.tracks[best_id]) >= 2:
                    result.append(obj["bb"])
            else:
                self.tracks[self.track_id] = [obj]
                self.track_id += 1
        self.tracks = {tid: t for tid, t in self.tracks.items()
                       if tid in matched
                       or fnum - t[-1]["fn"] < self.cfg.temporal_frames}
        return result

    # ── spatial filters ───────────────────────────────────────────────────────
    def _adjacent_filter(self, boxes: List[Tuple]) -> List[Tuple]:
        """Drop small boxes that are nearby much larger ones (artifact suppression)."""
        if len(boxes) <= 1:
            return boxes
        objs = sorted(
            [{"bb": b, "area": b[2]*b[3], "sz": max(b[2], b[3]),
              "cx": b[0]+b[2]/2, "cy": b[1]+b[3]/2} for b in boxes],
            key=lambda o: o["area"], reverse=True,
        )
        result = []
        for obj in objs:
            too_close = any(
                ((obj["cx"]-k["cx"])**2 + (obj["cy"]-k["cy"])**2)**0.5 < k["sz"]*1.5
                and obj["sz"] / k["sz"] < 0.3
                for k in result
            )
            if not too_close:
                result.append(obj)
        return [o["bb"] for o in result]

    def _cluster_filter(self, boxes: List[Tuple]) -> List[Tuple]:
        """Keep only the cfg.max_objects largest well-separated objects."""
        if len(boxes) <= 1:
            return boxes
        cfg  = self.cfg
        objs = sorted(
            [{"bb": b, "area": b[2]*b[3], "sz": max(b[2], b[3]),
              "cx": b[0]+b[2]/2, "cy": b[1]+b[3]/2} for b in boxes],
            key=lambda o: o["area"], reverse=True,
        )
        min_d  = np.mean([o["sz"] for o in objs]) * cfg.min_distance_ratio
        result = []
        for obj in objs:
            if len(result) >= cfg.max_objects:
                break
            close = any(
                ((obj["cx"]-k["cx"])**2 + (obj["cy"]-k["cy"])**2)**0.5 < min_d
                for k in result
            )
            if not close:
                result.append(obj)
        return [o["bb"] for o in result]


# ══════════════════════════════════════════════════════════════════════════════
# Polygon Selector  (first-frame interactive ROI picker)
# ══════════════════════════════════════════════════════════════════════════════
class PolygonSelector:
    """Draws a polygon ROI on the first video frame."""

    def __init__(self):
        self.points:   List[List[int]] = []
        self.complete: bool            = False
        self.mask:     Optional[np.ndarray] = None
        self._shape:   Optional[Tuple] = None

    def _mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.complete:
            self.points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) >= 3:
            self.complete = True
            self._build_mask()

    def _build_mask(self):
        H, W = self._shape
        self.mask = np.zeros((H, W), dtype=np.uint8)
        pts = np.array(self.points, dtype=np.int32)
        cv2.fillPoly(self.mask, [pts], 255)

    def run(self, frame) -> bool:
        """Show interactive polygon picker. Returns True if confirmed."""
        self._shape = frame.shape[:2]
        WIN = "Select ROI — left-click: add point | right-click: close | c: confirm | r: reset | q: quit"
        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, self._mouse)
        print("\n=== ROI Selection ===")
        print("Left-click to add points, right-click to close polygon, c to confirm.")
        while True:
            disp = frame.copy()
            if self.points:
                pts = np.array(self.points, dtype=np.int32)
                if self.complete:
                    overlay = disp.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.35, disp, 0.65, 0, disp)
                    cv2.polylines(disp, [pts], True, (0, 255, 255), 2)
                else:
                    for p in self.points:
                        cv2.circle(disp, tuple(p), 5, (0, 0, 255), -1)
                    for i in range(len(self.points) - 1):
                        cv2.line(disp, tuple(self.points[i]),
                                 tuple(self.points[i+1]), (0, 255, 0), 2)
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return False
            if key == ord("r"):
                self.points, self.complete, self.mask = [], False, None
            if key == ord("c") and self.complete:
                cv2.destroyAllWindows()
                return True
        cv2.destroyAllWindows()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Annotation State  (thin container passed into mouse callbacks)
# ══════════════════════════════════════════════════════════════════════════════
class AnnotationState:
    """Mutable annotation state for a single paused frame."""

    def __init__(self):
        self.selected: List[BBox] = []   # auto-detected boxes the user clicked
        self.manual:   List[BBox] = []   # hand-drawn boxes
        self.cur_cls:  int         = 0   # 0=cyclist 1=pedestrian
        # drawing state
        self.drawing:  bool             = False
        self.drag_start: Optional[Tuple] = None
        self.drag_box:   Optional[Tuple] = None   # live (x,y,w,h)

    def clear(self):
        self.selected.clear()
        self.manual.clear()
        self.drawing    = False
        self.drag_start = None
        self.drag_box   = None

    def all_boxes(self) -> List[BBox]:
        return self.selected + self.manual

    def toggle_class(self):
        self.cur_cls = 1 - self.cur_cls
        print(f"Class → {CLASS_NAMES[self.cur_cls]}")

    def mouse_callback(self, event, sx, sy, _flags, param):
        """param dict: {'detected': list[tuple4], 'scale': float}"""
        scale = param.get("scale", 1.0)
        x, y  = int(sx / scale), int(sy / scale)
        detected: List[Tuple] = param.get("detected", [])

        if event == cv2.EVENT_LBUTTONDOWN:
            hit = next(
                (i for i, (bx, by, bw, bh) in enumerate(detected)
                 if bx <= x <= bx+bw and by <= y <= by+bh),
                None,
            )
            if hit is not None:
                bbox = BBox.from_tuple(detected[hit])
                already = next((b for b in self.selected
                                if b.xywh == bbox.xywh), None)
                if already:
                    self.selected.remove(already)
                    print(f"Deselected box {hit+1}")
                else:
                    bbox.cls = self.cur_cls          # type: ignore[attr-defined]
                    # BBox is "immutable" by convention; just rebuild
                    self.selected.append(BBox(*bbox.xywh, self.cur_cls))
                    print(f"Selected box {hit+1} as {CLASS_NAMES[self.cur_cls]}")
            else:
                self.drawing    = True
                self.drag_start = (x, y)
                self.drag_box   = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.drag_start:
                x1, y1 = self.drag_start
                self.drag_box = (min(x1, x), min(y1, y),
                                 abs(x - x1), abs(y - y1))

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            if self.drag_start:
                x1, y1 = self.drag_start
                bw, bh = abs(x - x1), abs(y - y1)
                if bw > 10 and bh > 10:
                    nb = BBox(min(x1, x), min(y1, y), bw, bh, self.cur_cls)
                    if not any(b.xywh == nb.xywh for b in self.manual):
                        self.manual.append(nb)
                        print(f"Manual box → {CLASS_NAMES[self.cur_cls]}: {nb.xywh}")
            self.drawing    = False
            self.drag_start = None
            self.drag_box   = None


# ══════════════════════════════════════════════════════════════════════════════
# CSV timestamp loader (standalone helper)
# ══════════════════════════════════════════════════════════════════════════════
def parse_ts(value_str: str, total_frames: int, fps: float) -> Tuple[int, float]:
    """Parse a timestamp string → (frame_num, seconds)."""
    s = str(value_str).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            sec = int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        else:
            sec = int(parts[0])*60 + float(parts[1])
        fn = int(sec * fps)
    else:
        v = float(s)
        if v > total_frames:
            sec, fn = v, int(v * fps)
        else:
            fn, sec = int(v), v / fps
    fn = max(0, min(fn, total_frames - 1))
    return fn, fn / fps


def load_csv_timestamps(csv_path: str, total_frames: int, fps: float
                        ) -> List[Tuple[int, float]]:
    """Load and sort timestamps from a CSV file."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        sample = f.read(2048); f.seek(0)
        try:
            delim = csv.Sniffer().sniff(sample).delimiter
        except csv.Error:
            delim = ","
        if delim == ":":
            delim = "," if "," in sample else "\t"

        has_header = any(kw in sample.lower()
                         for kw in ("timestamp","time","frame","date"))
        reader = (csv.DictReader(f, delimiter=delim) if has_header
                  else csv.reader(f, delimiter=delim))

        for row in reader:
            try:
                val = (row.get("timestamp") or row.get("time") or
                       row.get("frame") or row.get("t") or
                       (row[0] if isinstance(row, list) else None))
                if val:
                    rows.append(parse_ts(val, total_frames, fps))
            except Exception:
                continue

    rows.sort()
    print(f"Loaded {len(rows)} timestamps from {csv_path}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Video Annotator  (main application)
# ══════════════════════════════════════════════════════════════════════════════
class VideoAnnotator:
    """
    Ties together: MotionEngine · AnnotationState · Interpolator ·
    DatasetWriter · PolygonSelector.

    Parameters mirror the argparse namespace so the class can also be
    instantiated directly in tests or notebooks.
    """

    SPEED_STEPS        = [1.0, 2.0, 4.0, 8.0, 16.0]
    RESUME_COOLDOWN    = 100     # frames before auto-pause re-arms after resume
    DISPLAY_SCALE      = 1.5
    FRAME_STEP         = 10     # frames jumped by f / b keys

    def __init__(self, cfg):
        self.cfg = cfg

        # ── video ─────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(cfg.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open: {cfg.video_path}")
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.W      = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H      = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.N      = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── sub-systems ───────────────────────────────────────────────────
        self.motion  = MotionEngine(cfg)
        self.writer  = DatasetWriter(cfg.dataset_dir)
        self.interp  = Interpolator(n=cfg.interpolate_frames)
        self.ann     = AnnotationState()
        self.poly    = PolygonSelector()

        # ── playback state ────────────────────────────────────────────────
        self.frame_num    = 0
        self.speed_idx    = 0
        self.cooldown     = 0    # resume cooldown counter
        self.timestamps:  List[Tuple[int, float]] = []
        self.ts_idx:      int = -1

        # ── runtime ──────────────────────────────────────────────────────
        self._events:     List[dict] = []

    # ── public entry point ────────────────────────────────────────────────────
    def run(self, timestamp_list: Optional[List] = None):
        """Full processing loop (region selection + annotation)."""
        # Select ROI
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame.")
        if not self.poly.run(first):
            print("ROI selection cancelled.")
            return []

        # Optional CSV timestamps
        if timestamp_list:
            self.timestamps = timestamp_list
            print(f"Timestamp navigation: {len(self.timestamps)} entries  (n/p to jump)")

        # Rewind + apply start offset
        self._seek(int(self.N * self.cfg.start_percent / 100))
        self.motion.reset()
        self.ann.clear()

        print("\n=== Dataset Creation ===")
        print(f"  Video      : {self.cfg.video_path}")
        print(f"  Resolution : {self.W}×{self.H}  @ {self.fps:.2f} fps")
        print(f"  Frames     : {self.N}  ({self.N/self.fps:.1f} s)")
        print(f"  Dataset    : {self.cfg.dataset_dir}")
        print(f"  Interp     : {self.cfg.interpolate_frames} frames"
              f"  ({'OFF' if self.cfg.interpolate_frames == 0 else 'ON'})")
        print(f"  Auto-pause : {'ON' if not self.cfg.no_auto_pause else 'OFF'}")
        print("========================\n")

        self._loop()
        print(f"\nDone. Saved {self.writer.saved} samples → {self.cfg.dataset_dir}")
        return self._events

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────
    def _loop(self):
        WIN = "Annotator"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        scaled_w = int(self.W * self.DISPLAY_SCALE)
        scaled_h = int(self.H * self.DISPLAY_SCALE)
        cv2.resizeWindow(WIN, scaled_w, scaled_h)

        cb_params: dict = {"detected": [], "scale": self.DISPLAY_SCALE}
        cv2.setMouseCallback(WIN, self.ann.mouse_callback, cb_params)

        paused      = False
        use_edges   = not self.cfg.no_edge_refinement
        cur_frame   = None   # last decoded frame (for display when paused)
        cur_boxes:  List[Tuple] = []
        cur_motion  = False
        last_save_fn: Optional[int] = None  # frame_num of last keyframe save

        # Jump to first timestamp if provided
        if self.timestamps:
            self._jump_ts(0)
            paused = True

        pbar = (tqdm(total=max(0, self.N - self.frame_num), desc="Scanning",
                     unit="frame", file=sys.stdout)
                if TQDM_AVAILABLE else None)

        while True:
            # ── advance or hold ───────────────────────────────────────────
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame_num += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(saved=self.writer.saved)

                motion, boxes = self.motion.detect(
                    frame, self.poly.mask, self.poly.points, use_edges)
                cur_frame, cur_boxes, cur_motion = frame, boxes, motion
                self.cooldown += 1

                # Auto-pause
                if (not self.cfg.no_auto_pause
                        and motion
                        and self.cooldown >= self.RESUME_COOLDOWN):
                    paused = True
                    self.ann.clear()
                    print(f"\n[AUTO-PAUSE] motion @ frame {self.frame_num}")

                if motion:
                    self._record_event(boxes)
            else:
                # Paused: keep displaying stored frame
                if cur_frame is None:
                    break
                frame = cur_frame

            # ── build display ─────────────────────────────────────────────
            cb_params["detected"] = cur_boxes
            disp = self._render(frame, cur_boxes, cur_motion, paused, use_edges)
            scaled = cv2.resize(disp, None,
                                fx=self.DISPLAY_SCALE, fy=self.DISPLAY_SCALE,
                                interpolation=cv2.INTER_LINEAR)
            cv2.imshow(WIN, scaled)

            wait = 30 if paused else max(1, int(1000 / self.fps / self._speed()))
            key  = cv2.waitKey(wait) & 0xFF

            # ── keyboard ─────────────────────────────────────────────────
            if key == ord("q"):
                break

            elif key == ord(" "):
                paused = not paused
                if paused:
                    self.ann.clear()
                    print(f"Paused @ frame {self.frame_num}")
                else:
                    self.ann.clear()
                    self.cooldown = 0
                    print(f"Resumed (cooldown reset)")

            elif key == ord("s") and paused:
                saved_at = self.frame_num
                self._save_keyframe(cur_frame, self.frame_num, last_save_fn)
                last_save_fn = saved_at
                self.ann.clear()
                # Auto-jump: seek from saved_at so cap position is clean after interp seeks
                if self.cfg.interpolate_frames > 0:
                    jump   = self.cfg.interpolate_frames
                    target = max(0, min(saved_at + jump, self.N - 1))
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    # Read frame_num from cap after seek (OpenCV seek is not always exact)
                    self.frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    ret, jumped_frame = self.cap.read()
                    if ret:
                        cur_frame  = jumped_frame
                        _, cur_boxes = self.motion.detect(
                            cur_frame, self.poly.mask, self.poly.points, use_edges)
                        cur_motion = bool(cur_boxes)
                        print(f"Auto-jumped +{jump} -> frame {self.frame_num} (target={target})")

            elif key == ord("c") and paused:
                self.ann.clear()
                cur_boxes, cur_motion = [], False
                print("Cleared")

            elif key == ord("t"):
                self.ann.toggle_class()

            elif key == ord("f") and paused:
                cur_frame, cur_boxes, cur_motion = self._step(+self.FRAME_STEP, use_edges)

            elif key == ord("b") and paused:
                cur_frame, cur_boxes, cur_motion = self._step(-self.FRAME_STEP, use_edges)

            elif key == ord("r"):
                self._seek(0); self.motion.reset(); self.ann.clear()
                paused, cur_frame, cur_boxes, cur_motion = False, None, [], False
                self.cooldown = 0; last_save_fn = None
                print("Reset")

            elif key == ord("e"):
                use_edges = not use_edges
                print(f"Edge refinement: {'ON' if use_edges else 'OFF'}")

            elif key in (ord("+"), ord("=")):
                self.speed_idx = min(self.speed_idx + 1, len(self.SPEED_STEPS) - 1)
                print(f"Speed: {self._speed():.1f}×")

            elif key in (ord("-"), ord("_")):
                self.speed_idx = max(self.speed_idx - 1, 0)
                print(f"Speed: {self._speed():.1f}×")

            elif key == ord("n") and self.timestamps:
                result = self._jump_ts(self.ts_idx + 1)
                if result:
                    cur_frame, cur_boxes, cur_motion = result
                    self.ann.clear(); paused = True

            elif key == ord("p") and self.timestamps:
                result = self._jump_ts(self.ts_idx - 1)
                if result:
                    cur_frame, cur_boxes, cur_motion = result
                    self.ann.clear(); paused = True

        if pbar:
            pbar.close()
        cv2.destroyAllWindows()

    # ─────────────────────────────────────────────────────────────────────────
    # Save keyframe + interpolation
    # ─────────────────────────────────────────────────────────────────────────
    def _save_keyframe(self, frame, frame_num: int,
                       last_save_fn: Optional[int]):
        """Save current annotation; generate + save interpolated frames if applicable."""
        all_boxes = self.ann.all_boxes()
        if not all_boxes:
            print("Nothing to save — select or draw boxes first.")
            return

        # Register with interpolator and get intermediate frames to generate
        interp_frames = self.interp.add(frame_num, frame, all_boxes)

        # ── Save the annotated keyframe ────────────────────────────────
        self.writer.save(frame, frame_num, all_boxes, tag="keyframe")

        # ── Save interpolated frames ───────────────────────────────────────
        if interp_frames:
            gap = frame_num - (last_save_fn or frame_num)
            print(f"  Interpolating {len(interp_frames)} frames "
                  f"(gap={gap}, n={self.cfg.interpolate_frames}) ...")
            # Seek to each interp frame and save. Do NOT restore position here —
            # the auto-jump in _loop always seeks from saved_at explicitly,
            # so a restore here just causes a second imprecise seek (the time-jump bug).
            for fn, boxes in interp_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ret, img = self.cap.read()
                if ret:
                    self.writer.save(img, fn, boxes, tag="interp")
        else:
            if self.cfg.interpolate_frames > 0 and last_save_fn is not None:
                gap = frame_num - last_save_fn
                print(f"  No interpolation (gap={gap}, max={self.cfg.interpolate_frames})")

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────────────
    def _render(self, frame, boxes: List[Tuple], motion: bool,
                paused: bool, use_edges: bool) -> np.ndarray:
        disp = frame.copy()

        # ROI overlay
        pts = np.array(self.poly.points, dtype=np.int32)
        overlay = disp.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, disp, 0.8, 0, disp)
        cv2.polylines(disp, [pts], True, (0, 255, 255), 2)

        # Auto-detected boxes
        if motion and boxes:
            for i, (x, y, w, h) in enumerate(boxes):
                sel = next((b for b in self.ann.selected if b.xywh == (x, y, w, h)), None)
                if sel:
                    color = SEL_COLORS[sel.cls]
                    label = f"#{i+1} [{CLASS_NAMES[sel.cls].upper()}]"
                    thick = 3
                else:
                    color, label, thick = UNSEL_COLOR, f"#{i+1}", 2
                cv2.rectangle(disp, (x, y), (x+w, y+h), color, thick)
                self._put_label(disp, label, x, y, color)

        # Manual boxes
        for b in self.ann.manual:
            x, y, w, h = b.xywh
            color = MAN_COLORS[b.cls]
            cv2.rectangle(disp, (x, y), (x+w, y+h), color, 2)
            self._put_label(disp, f"MANUAL {CLASS_NAMES[b.cls].upper()}", x, y, color)

        # Live drag box
        if self.ann.drag_box:
            x, y, w, h = self.ann.drag_box
            color = MAN_COLORS[self.ann.cur_cls]
            cv2.rectangle(disp, (x, y), (x+w, y+h), color, 1)

        # Interpolation preview overlay
        preview = self.interp.preview_boxes(self.frame_num)
        if preview:
            for b in preview:
                x, y, w, h = b.xywh
                cv2.rectangle(disp, (x, y), (x+w, y+h), INTERP_COLOR, 1)
                cv2.putText(disp, "~interp~", (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, INTERP_COLOR, 1)

        # HUD
        self._draw_hud(disp, motion, paused, use_edges)
        return disp

    def _draw_hud(self, disp, motion: bool, paused: bool, use_edges: bool):
        lines = []
        if paused:
            cls_name = CLASS_NAMES[self.ann.cur_cls].upper()
            lines.append(("PAUSED", (0, 255, 255)))
            lines.append((f"Class: {cls_name}  [t=toggle]", SEL_COLORS[self.ann.cur_cls]))
            c_sel  = sum(1 for b in self.ann.selected if b.cls == 0)
            p_sel  = sum(1 for b in self.ann.selected if b.cls == 1)
            c_man  = sum(1 for b in self.ann.manual   if b.cls == 0)
            p_man  = sum(1 for b in self.ann.manual   if b.cls == 1)
            lines.append((f"Sel {c_sel}C/{p_sel}P  Man {c_man}C/{p_man}P", (255, 255, 255)))
        else:
            status = "MOTION!" if motion else "scanning"
            lines.append((status, (0, 165, 255) if motion else (0, 200, 0)))
            lines.append((f"Speed: {self._speed():.1f}×  Edge: {'ON' if use_edges else 'OFF'}",
                          (200, 200, 200)))
            cd = max(0, self.RESUME_COOLDOWN - self.cooldown)
            if cd:
                lines.append((f"Cooldown: {cd} frames", (0, 255, 255)))

        lines.append((f"Frame {self.frame_num}/{self.N} | "
                      f"{self.frame_num/self.fps:.1f}s | "
                      f"saved={self.writer.saved}", (200, 200, 200)))

        if self.cfg.interpolate_frames > 0:
            lines.append((f"Interp: ±{self.cfg.interpolate_frames} frames", INTERP_COLOR))

        if self.timestamps and self.ts_idx >= 0:
            lines.append((f"TS {self.ts_idx+1}/{len(self.timestamps)} "
                          f"@ {self.timestamps[self.ts_idx][1]:.1f}s", (0, 255, 255)))

        for i, (text, color) in enumerate(lines):
            cv2.putText(disp, text, (8, 18 + i*17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    @staticmethod
    def _put_label(img, text: str, x: int, y: int, color):
        sz, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        ly = max(y - 4, sz[1] + 4)
        cv2.rectangle(img, (x, ly-sz[1]-2), (x+sz[0]+3, ly+2), color, -1)
        cv2.putText(img, text, (x+1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _speed(self) -> float:
        return self.SPEED_STEPS[self.speed_idx]

    def _seek(self, target_fn: int):
        target_fn = max(0, min(target_fn, self.N - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_fn)
        self.frame_num = target_fn

    def _step(self, delta: int, use_edges: bool
              ) -> Tuple[np.ndarray, List[Tuple], bool]:
        """Step ±N frames while paused; returns (frame, boxes, motion)."""
        target = max(0, min(self.frame_num + delta, self.N - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.frame_num = target
        ret, frame = self.cap.read()
        if not ret:
            return None, [], False
        motion, boxes = self.motion.detect(
            frame, self.poly.mask, self.poly.points, use_edges)
        self.ann.clear()
        print(f"Stepped to frame {self.frame_num}")
        return frame, boxes, motion

    def _jump_ts(self, idx: int):
        """Jump to timestamps[idx]. Returns (frame, boxes, motion) or None."""
        if not self.timestamps or idx < 0 or idx >= len(self.timestamps):
            return None
        self.ts_idx = idx
        fn, ts_sec  = self.timestamps[idx]
        self._seek(fn)
        ret, frame  = self.cap.read()
        if not ret:
            return None
        motion, boxes = self.motion.detect(
            frame, self.poly.mask, self.poly.points)
        print(f"Timestamp {idx+1}/{len(self.timestamps)} "
              f"→ frame {fn} ({ts_sec:.1f}s)")
        return frame, boxes, motion

    def _record_event(self, boxes: List[Tuple]):
        """Record a motion event (for JSON export)."""
        fn  = self.frame_num
        if self._events and fn - self._events[-1]["frame"] < self.fps * 0.5:
            return  # cooldown
        self._events.append({
            "frame":               fn,
            "timestamp":           fn / self.fps,
            "timestamp_formatted": str(timedelta(seconds=int(fn / self.fps))),
            "bounding_boxes":      [list(b) for b in boxes],
        })

    def save_events_json(self, path: str):
        data = {
            "video":    self.cfg.video_path,
            "fps":      self.fps,
            "frames":   self.N,
            "polygon":  self.poly.points,
            "events":   self._events,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Events saved -> {path}")

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cyclist / pedestrian video annotation tool with keyframe interpolation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("video_path")

    # Dataset
    p.add_argument("--dataset-dir",   default="pdx_cyclist_dataset",
                   help="Output dataset directory")
    p.add_argument("--output",        default=None,
                   help="Save motion events JSON to this path")

    # Interpolation  ← NEW
    p.add_argument("--interpolate-frames", type=int, default=5,
                   choices=[0, 5, 10],
                   help="Frames to interpolate between keyframes: 0=off, 5, or 10")

    # Motion detection
    p.add_argument("--motion-threshold",  type=int,   default=100)
    p.add_argument("--min-area",          type=int,   default=50)
    p.add_argument("--temporal-frames",   type=int,   default=8,
                   help="Frames motion must persist before triggering")
    p.add_argument("--min-aspect-ratio",  type=float, default=0.5)
    p.add_argument("--max-aspect-ratio",  type=float, default=5.5)
    p.add_argument("--min-solidity",      type=float, default=0.3)
    p.add_argument("--min-compactness",   type=float, default=0.2)
    p.add_argument("--min-distance-ratio",type=float, default=0.4)
    p.add_argument("--max-objects",       type=int,   default=3)

    # Behaviour
    p.add_argument("--no-auto-pause",       action="store_true")
    p.add_argument("--no-edge-refinement",  action="store_true")
    p.add_argument("--start-percent",       type=float, default=0.0)
    p.add_argument("--timestamps-csv",      default=None,
                   help="CSV of timestamps / frame numbers to navigate")

    return p


def main():
    cfg = build_parser().parse_args()

    if not (0.0 <= cfg.start_percent <= 100.0):
        raise SystemExit("--start-percent must be 0–100")

    annotator = VideoAnnotator(cfg)

    # Load CSV timestamps if provided
    ts_list = None
    if cfg.timestamps_csv:
        ts_list = load_csv_timestamps(
            cfg.timestamps_csv, annotator.N, annotator.fps)

    events = annotator.run(timestamp_list=ts_list)

    # Print summary
    if events:
        print(f"\n{len(events)} motion events detected.")
    if cfg.output:
        annotator.save_events_json(cfg.output)


if __name__ == "__main__":
    main()