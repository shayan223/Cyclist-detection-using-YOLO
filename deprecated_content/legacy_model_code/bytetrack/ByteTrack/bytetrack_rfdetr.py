from __future__ import annotations

"""
ByteTrack video tracker — model-agnostic.

Two backends
------------
inference  (default)
    Base pretrained model via the Roboflow inference package.
    Default model: rfdetr-seg-medium
    Tracks ALL classes; HUD shows per-class totals for person, bicycle, car, bus.
    Class names follow COCO conventions (e.g. "person", "bicycle", "car", "bus").

ultralytics
    Local .pt model (RT-DETR, YOLO, etc.).
    Use with: --model path/to/model.pt
    Filters to cyclist / pedestrian classes with separate trackers per class.
"""

import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    import supervision as sv
except ImportError as e:
    raise SystemExit("supervision is required:  pip install supervision") from e

try:
    from inference import get_model as _inference_get_model
    _INFERENCE_AVAILABLE = True
except ImportError:
    _INFERENCE_AVAILABLE = False

try:
    from ultralytics import RTDETR as _RTDETR, YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

try:
    from trackers import ByteTrackTracker as _ByteTrackImpl
    _TRACKER_MODE = "trackers"
except ImportError:
    _TRACKER_MODE = "supervision"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Street-scene classes to keep — everything else is discarded after inference.
# These are lowercase COCO class names as returned by rfdetr-seg-medium / rfdetr-medium.
STREET_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
}

# Which of those classes get their own HUD counter and unique-ID tracker.
# Display label → COCO class name
TRACKED_CLASS_NAMES = {
    "People":      "person",
    "Cyclists":    "bicycle",
    "Motorcycles": "motorcycle",
    "Cars":        "car",
    "Trucks":      "truck",
    "Buses":       "bus",
}

_PALETTE_HEX = [
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
]


def _print_gpu_info():
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / 1024 ** 3
        print(f"GPU: {props.name}  ({mem_gb:.1f} GB VRAM)")
        print(f"CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")
    else:
        print("WARNING: CUDA not available — running on CPU. Inference will be slow.")
        print("  Check that PyTorch is installed with CUDA support:  torch.cuda.is_available()")


# ─────────────────────────────────────────────────────────────────────────────
# ByteTrack wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _ByteTrackWrapper:
    def __init__(self, frame_rate=30, track_activation_threshold=0.25,
                 lost_track_buffer=30, minimum_matching_threshold=0.8,
                 minimum_consecutive_frames=1):
        kwargs = dict(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        if _TRACKER_MODE == "trackers":
            self._backend = _ByteTrackImpl(**kwargs)
            self._update = self._backend.update
        else:
            print("WARNING: roboflow/trackers not found — using supervision.ByteTrack")
            self._backend = sv.ByteTrack(**kwargs)
            self._update = self._backend.update_with_detections

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self._update(detections)

    def reset(self):
        if hasattr(self._backend, "reset"):
            self._backend.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, backend: str, model_id: str, imgsz: int = 0):
    """Return (predict_fn, backend_name).
    predict_fn(frame_bgr, conf, nms_thresh) -> sv.Detections
    Detections are always in the coordinate space of the frame passed in.
    """
    if backend == "auto":
        backend = "ultralytics" if model_path.lower().endswith((".pt", ".onnx")) else "inference"

    if backend == "ultralytics":
        if not _ULTRALYTICS_AVAILABLE:
            raise SystemExit("ultralytics not installed:  pip install ultralytics")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        use_half = DEVICE == "cuda"
        _imgsz   = imgsz if imgsz > 0 else 640
        print(f"Loading ultralytics model: {model_path}  (device={DEVICE}, fp16={use_half}, imgsz={_imgsz})")
        name = os.path.basename(model_path).lower()
        model = (_RTDETR(model_path) if any(x in name for x in ("rtdetr", "rt-detr", "rt_detr"))
                 else _YOLO(model_path))

        print("Warming up model...")
        dummy = np.zeros((_imgsz, _imgsz, 3), dtype=np.uint8)
        model.predict(dummy, imgsz=_imgsz, device=DEVICE, verbose=False)
        print("Warmup done.")

        def predict(frame_bgr, conf, nms_thresh):
            # Ultralytics handles letterboxing internally; boxes returned in original frame coords
            results = model.predict(frame_bgr, imgsz=_imgsz, conf=conf, iou=nms_thresh,
                                    device=DEVICE, half=use_half, verbose=False)[0]
            return sv.Detections.from_ultralytics(results)

        return predict, "ultralytics"

    else:
        if not _INFERENCE_AVAILABLE:
            raise SystemExit(
                "inference package not installed:  pip install inference\n"
                "For GPU support:                  pip install inference-gpu\n"
                "Or pass a local .pt model with --model."
            )
        if DEVICE == "cuda":
            print("TIP: Make sure you have installed 'inference-gpu' for GPU acceleration.")
        print(f"Loading model via Roboflow inference: {model_id}")
        raw = _inference_get_model(model_id)

        def predict(frame_bgr, conf, nms_thresh):
            result = raw.infer(frame_bgr, confidence=conf)[0]
            return sv.Detections.from_inference(result).with_nms(threshold=nms_thresh)

        return predict, "inference"


# ─────────────────────────────────────────────────────────────────────────────
# Inference resize helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resize_for_inference(frame: np.ndarray, imgsz: int):
    """Resize frame so its longest side == imgsz, preserving aspect ratio.
    Returns (resized_frame, scale_x, scale_y) where scale_* maps inference
    coordinates back to original frame coordinates.
    """
    h, w = frame.shape[:2]
    scale = imgsz / max(h, w)
    if scale >= 1.0:           # frame already smaller than imgsz — no resize needed
        return frame, 1.0, 1.0
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, w / new_w, h / new_h


def _scale_detections(dets: sv.Detections, sx: float, sy: float) -> sv.Detections:
    """Scale xyxy boxes from inference space back to original frame space."""
    if sx == 1.0 and sy == 1.0:
        return dets
    scaled = dets.xyxy.copy()
    scaled[:, 0] *= sx; scaled[:, 2] *= sx
    scaled[:, 1] *= sy; scaled[:, 3] *= sy
    return sv.Detections(
        xyxy=scaled,
        mask=dets.mask,
        confidence=dets.confidence,
        class_id=dets.class_id,
        tracker_id=dets.tracker_id,
        data=dets.data,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Class filter
# ─────────────────────────────────────────────────────────────────────────────

def _filter_street_classes(dets: sv.Detections, class_names_override=None) -> sv.Detections:
    """Keep only detections whose class name is in STREET_CLASSES."""
    if len(dets) == 0:
        return dets
    keep = []
    for i in range(len(dets)):
        name = _get_class_name(dets, i, class_names_override)
        keep.append(name in STREET_CLASSES)
    mask = np.array(keep, dtype=bool)
    return dets[mask] if mask.any() else sv.Detections.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Deadzone helpers  (ported from RT_DETR/deepSORT_rtdetr.py)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_deadzones(dets: sv.Detections, deadzones: list) -> sv.Detections:
    """Remove detections whose centre point falls inside any deadzone polygon."""
    if not deadzones or len(dets) == 0:
        return dets
    keep = []
    for i in range(len(dets)):
        x1, y1, x2, y2 = dets.xyxy[i]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        in_zone = any(
            cv2.pointPolygonTest(np.array(z, dtype=np.float32), (float(cx), float(cy)), False) >= 0
            for z in deadzones
        )
        keep.append(not in_zone)
    return dets[np.array(keep, dtype=bool)]


def _draw_deadzones(frame, deadzones: list, show: bool) -> np.ndarray:
    if not show or not deadzones:
        return frame
    for zone in deadzones:
        pts = np.array(zone, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 180))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 0, 200), 1)
    return frame


def draw_deadzones_interactive(frame) -> list:
    """
    Show the first video frame and let the user draw rectangular deadzones.

    Left-click + drag : draw a rectangle
    U                 : undo last zone
    C                 : clear all zones
    Enter / Space     : confirm
    Esc               : skip (no deadzones)
    """
    zones    = []
    drawing  = False
    start_pt = None
    curr_pt  = None

    def _mouse(event, x, y, flags, param):
        nonlocal drawing, start_pt, curr_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True; start_pt = (x, y); curr_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            curr_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if start_pt and (abs(x - start_pt[0]) > 5 or abs(y - start_pt[1]) > 5):
                x1, y1 = min(start_pt[0], x), min(start_pt[1], y)
                x2, y2 = max(start_pt[0], x), max(start_pt[1], y)
                zones.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            start_pt = curr_pt = None

    instructions = [
        "Draw deadzones: left-click + drag to add a zone",
        "U: undo last   C: clear all   Enter/Space: confirm   Esc: skip",
    ]

    try:
        cv2.namedWindow("Draw Deadzones")
        cv2.setMouseCallback("Draw Deadzones", _mouse)
        while True:
            img = frame.copy()
            for zone in zones:
                pts = np.array(zone, dtype=np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 180))
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                cv2.polylines(img, [pts], True, (0, 0, 255), 2)
            if drawing and start_pt and curr_pt:
                cv2.rectangle(img, start_pt, curr_pt, (0, 200, 255), 1)
            for j, txt in enumerate(instructions):
                y_pos = 30 + j * 25
                cv2.putText(img, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Draw Deadzones", img)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):   break          # Enter / Space
            elif key == 27:       zones = []; break  # Esc
            elif key == ord("u") and zones: zones.pop()
            elif key == ord("c"):               zones = []
        cv2.destroyWindow("Draw Deadzones")
    except (cv2.error, Exception) as e:
        print(f"Deadzone editor unavailable: {e}")
    return zones


# ─────────────────────────────────────────────────────────────────────────────
# Annotators
# ─────────────────────────────────────────────────────────────────────────────

def _build_annotators(use_segmentation: bool = True):
    color = sv.ColorPalette.from_hex(_PALETTE_HEX)
    return (
        sv.BoxAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK),
        sv.MaskAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK) if use_segmentation else None,
        sv.LabelAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK,
                          text_color=sv.Color.BLACK, text_scale=0.4, text_thickness=1),
        sv.TraceAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK,
                          thickness=1, trace_length=100),
    )


def _annotate(frame, tracked, labels, box_ann, mask_ann, label_ann, trace_ann):
    if tracked is None or len(tracked) == 0:
        return frame
    if mask_ann and tracked.mask is not None:
        frame = mask_ann.annotate(frame, tracked)
    frame = box_ann.annotate(frame, tracked)
    frame = trace_ann.annotate(frame, tracked)
    frame = label_ann.annotate(frame, tracked, labels)
    return frame


def _get_class_name(det: sv.Detections, i: int, class_names: list[str] | None) -> str:
    if "class_name" in (det.data or {}):
        return str(det.data["class_name"][i]).lower()
    if class_names and det.class_id is not None:
        return class_names[int(det.class_id[i])].lower()
    return f"class_{int(det.class_id[i])}" if det.class_id is not None else "obj"


def _get_labels(tracked: sv.Detections, class_names: list[str] | None) -> list[str]:
    labels = []
    for i in range(len(tracked)):
        tid = int(tracked.tracker_id[i])
        cls = _get_class_name(tracked, i, class_names)
        labels.append(f"{cls} #{tid}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# HUD helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_hud_geometry(width, height, n_lines: int):
    """Return (tx, line_ys, bg_rect) for an n-line HUD in the bottom-right corner."""
    fs, ft, pad, ls = 0.45, 1, 8, 3
    sample = "People: 999 (Total: 9999)"
    tw = max(cv2.getTextSize(sample, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0][0], 280)
    th = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0][1]
    tx = width - tw - pad
    # Lines from bottom up
    ys = [height - pad - i * (th + ls) for i in range(n_lines)]
    bg = (max(0, tx - pad),
          max(0, ys[-1] - th - pad),
          min(width, tx + tw + pad),
          min(height, ys[0] + pad))
    return fs, ft, tx, ys, bg


def _draw_hud(frame, lines: list[tuple[str, tuple]], fs, ft, tx, ys, bg):
    """lines = [(text, bgr_color), ...]  ordered bottom→top."""
    cv2.rectangle(frame, (bg[0], bg[1]), (bg[2], bg[3]), (0, 0, 0), -1)
    for (text, color), y in zip(lines, ys):
        cv2.putText(frame, text, (tx, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, ft)


# ─────────────────────────────────────────────────────────────────────────────
# Main video processing loop
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    input_path, output_path, predict_fn, backend,
    confidence=0.2, nms_threshold=0.3,
    track_activation_threshold=0.25, lost_track_buffer=30,
    minimum_matching_threshold=0.8, minimum_consecutive_frames=1,
    disable_display=True,
    cyclist_class_id=0, pedestrian_class_id=1,
    class_names=None,
    use_segmentation=True,
    imgsz=0,
    deadzones=None,
    show_deadzones=False,
    inference_only=False,
):
    deadzones = deadzones or []
    all_classes_mode = (backend == "inference")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Inference resize: only applies to the inference backend.
    # For ultralytics the imgsz is already passed to model.predict — no pre-resize needed.
    _inf_imgsz = imgsz if (imgsz > 0 and backend == "inference") else 0
    if _inf_imgsz:
        _sample_h = int(round(height * _inf_imgsz / max(width, height)))
        _sample_w = int(round(width  * _inf_imgsz / max(width, height)))
        print(f"Video: {width}x{height}  →  inference at {_sample_w}x{_sample_h}  (--imgsz {_inf_imgsz})")
    else:
        print(f"Video: {width}x{height}  (full resolution inference)")

    tracker_kwargs = dict(
        frame_rate=fps,
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold,
        minimum_consecutive_frames=minimum_consecutive_frames,
    )

    if not inference_only:
        if all_classes_mode:
            tracker = _ByteTrackWrapper(**tracker_kwargs)
            tracker.reset()
        else:
            cyclist_tracker    = _ByteTrackWrapper(**tracker_kwargs)
            pedestrian_tracker = _ByteTrackWrapper(**tracker_kwargs)
            cyclist_tracker.reset()
            pedestrian_tracker.reset()

    # Per-class unique ID sets (inference / all-classes mode)
    class_ids_seen: dict[str, set] = {k: set() for k in TRACKED_CLASS_NAMES}
    # Custom model mode
    cyclist_ids_seen:    set = set()
    pedestrian_ids_seen: set = set()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    box_ann, mask_ann, label_ann, trace_ann = _build_annotators(use_segmentation)

    # HUD geometry
    if all_classes_mode and not inference_only:
        n_hud = len(TRACKED_CLASS_NAMES)   # one line per street class
    else:
        n_hud = 2                          # cyclist + pedestrian
    fs_h, ft_h, tx_h, ys_h, bg_h = _build_hud_geometry(width, height, n_hud)

    frame_count     = 0
    paused          = False
    speed           = 1.0
    annotated_frame = None
    display_available = not disable_display

    progress = tqdm(total=total if total > 0 else None, desc="Processing", unit="frame")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                if _inf_imgsz:
                    inf_frame, sx, sy = _resize_for_inference(frame, _inf_imgsz)
                    all_dets = predict_fn(inf_frame, confidence, nms_threshold)
                    all_dets = _scale_detections(all_dets, sx, sy)
                else:
                    all_dets = predict_fn(frame, confidence, nms_threshold)

                # Drop non-street classes (inference backend only — custom model already filtered)
                if all_classes_mode:
                    all_dets = _filter_street_classes(all_dets)

                # Apply deadzones
                all_dets = _filter_deadzones(all_dets, deadzones)

                annotated_frame = frame.copy()
                annotated_frame = _draw_deadzones(annotated_frame, deadzones, show_deadzones)

                if inference_only:
                    labels = []
                    for i in range(len(all_dets)):
                        cls = _get_class_name(all_dets, i, class_names)
                        conf = float(all_dets.confidence[i]) if all_dets.confidence is not None else 0.
                        labels.append(f"{cls} {conf:.2f}")
                    color = sv.ColorPalette.from_hex(_PALETTE_HEX)
                    annotated_frame = sv.BoxAnnotator(color=color).annotate(annotated_frame, all_dets)
                    annotated_frame = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK,
                                                        text_scale=0.4, text_thickness=1).annotate(annotated_frame, all_dets, labels)

                elif all_classes_mode:
                    # ── Inference backend: track every class ──────────────────
                    tracked = tracker.update(all_dets)

                    # Count current frame + accumulate unique IDs per class
                    current_counts: dict[str, int] = {k: 0 for k in TRACKED_CLASS_NAMES}
                    if tracked.tracker_id is not None:
                        for i in range(len(tracked)):
                            cls = _get_class_name(tracked, i, class_names)
                            tid = int(tracked.tracker_id[i])
                            for display_name, coco_name in TRACKED_CLASS_NAMES.items():
                                if cls == coco_name:
                                    current_counts[display_name] += 1
                                    class_ids_seen[display_name].add(tid)

                    labels = _get_labels(tracked, class_names)
                    annotated_frame = _annotate(annotated_frame, tracked, labels,
                                                box_ann, mask_ann, label_ann, trace_ann)

                    # HUD: one line per class, bottom-up order
                    hud_lines = [
                        (f"{name}: {current_counts[name]} (Total: {len(class_ids_seen[name])})", (0, 255, 0))
                        for name in TRACKED_CLASS_NAMES
                    ]
                    _draw_hud(annotated_frame, hud_lines, fs_h, ft_h, tx_h, ys_h, bg_h)

                    if frame_count % 10 == 0:
                        counts_str = "  ".join(f"{n}={current_counts[n]}" for n in TRACKED_CLASS_NAMES)
                        totals_str = "  ".join(f"{n}={len(class_ids_seen[n])}" for n in TRACKED_CLASS_NAMES)
                        print(f"Frame {frame_count}: {counts_str} | Total: {totals_str}")

                else:
                    # ── Ultralytics backend: cyclist / pedestrian split ────────
                    cyc_dets = all_dets[all_dets.class_id == cyclist_class_id]
                    ped_dets = all_dets[all_dets.class_id == pedestrian_class_id]

                    cyc_tracked = cyclist_tracker.update(cyc_dets)
                    ped_tracked = pedestrian_tracker.update(ped_dets)

                    if cyc_tracked.tracker_id is not None:
                        cyclist_ids_seen.update(cyc_tracked.tracker_id.tolist())
                    if ped_tracked.tracker_id is not None:
                        pedestrian_ids_seen.update(ped_tracked.tracker_id.tolist())

                    cyc_labels = [f"Cyclist #{int(t)}" for t in (cyc_tracked.tracker_id or [])]
                    ped_labels = [f"Pedestrian #{int(t)}" for t in (ped_tracked.tracker_id or [])]

                    annotated_frame = _annotate(annotated_frame, cyc_tracked, cyc_labels,
                                                box_ann, mask_ann, label_ann, trace_ann)
                    annotated_frame = _annotate(annotated_frame, ped_tracked, ped_labels,
                                                box_ann, mask_ann, label_ann, trace_ann)

                    hud_lines = [
                        (f"Cyclists: {len(cyc_tracked)} (Total: {len(cyclist_ids_seen)})",    (0, 255, 0)),
                        (f"Pedestrians: {len(ped_tracked)} (Total: {len(pedestrian_ids_seen)})", (255, 0, 0)),
                    ]
                    _draw_hud(annotated_frame, hud_lines, fs_h, ft_h, tx_h, ys_h, bg_h)

                    if frame_count % 10 == 0:
                        print(f"Frame {frame_count}: {len(cyc_tracked)} cyclist(s), {len(ped_tracked)} pedestrian(s)"
                              f" | Total unique: {len(cyclist_ids_seen)} C, {len(pedestrian_ids_seen)} P")

                writer.write(annotated_frame)
                frame_count += 1
                progress.update(1)

            if display_available and annotated_frame is not None:
                try:
                    cv2.imshow("ByteTrack", annotated_frame)
                except (cv2.error, Exception):
                    display_available = False

            key = cv2.waitKey(max(1, int(1000 / fps / speed)) if display_available else 1) & 0xFF
            if   key == ord("q"): break
            elif key == ord("p"):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key in (ord("+"), ord("=")):
                speed = min(5., speed + 0.5); print(f"Speed: {speed:.1f}x")
            elif key == ord("-"):
                speed = max(0.1, speed - 0.5); print(f"Speed: {speed:.1f}x")

    finally:
        progress.close()
        cap.release()
        writer.release()
        if display_available:
            try: cv2.destroyAllWindows()
            except: pass
        print(f"\nDone: {output_path}")
        if not inference_only:
            if all_classes_mode:
                for name in TRACKED_CLASS_NAMES:
                    print(f"Total unique {name.lower()}: {len(class_ids_seen[name])}")
            else:
                print(f"Total unique cyclists:    {len(cyclist_ids_seen)}")
                print(f"Total unique pedestrians: {len(pedestrian_ids_seen)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input",  "-i", default="../trim_3.mp4")
    parser.add_argument("--output", "-o", default="")
    parser.add_argument(
        "--model", "-m", default="",
        help="Local model path (.pt / .onnx). Leave empty to use Roboflow inference backend.",
    )
    parser.add_argument("--backend", choices=["auto", "inference", "ultralytics"], default="auto")
    parser.add_argument(
        "--bbox", action="store_true",
        help="Use bounding-box-only model (rfdetr-medium). Disables mask annotation.",
    )
    parser.add_argument(
        "--model-id", default="",
        help="Roboflow inference model ID. Defaults to rfdetr-seg-medium, or rfdetr-medium with --bbox.",
    )
    # Ultralytics (custom model) only
    parser.add_argument("--cyclist-class-id",    type=int, default=0)
    parser.add_argument("--pedestrian-class-id", type=int, default=1)

    parser.add_argument("--confidence",  "-c", type=float, default=0.2)
    parser.add_argument("--nms-threshold",      type=float, default=0.3)
    parser.add_argument(
        "--imgsz", type=int, default=0,
        help="Resize longest edge to this size before inference (e.g. 640, 480, 320). "
             "0 = use full resolution. For ultralytics backend this also sets the model imgsz. "
             "Smaller = faster, with some accuracy trade-off.",
    )

    # ByteTrack
    parser.add_argument("--track-activation-threshold", type=float, default=0.25)
    parser.add_argument("--lost-track-buffer",           type=int,   default=30)
    parser.add_argument("--minimum-matching-threshold",  type=float, default=0.8)
    parser.add_argument("--minimum-consecutive-frames",  type=int,   default=1)

    # Deadzones
    parser.add_argument("--deadzone", action="store_true",
                        help="Interactively draw rectangular deadzones on the first frame. "
                             "Detections whose centre falls inside a deadzone are suppressed.")
    parser.add_argument("--show-deadzones", action="store_true",
                        help="Render deadzone overlays on the output video.")

    parser.add_argument("--no-display",     action="store_true", default=True)
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")

    use_segmentation = not args.bbox
    model_id = args.model_id or ("rfdetr-medium" if args.bbox else "rfdetr-seg-medium")

    if not args.output:
        base = os.path.splitext(args.input)[0]
        suffix = "bytetrack_inference" if args.inference_only else "bytetrack"
        args.output = f"{base}_{suffix}.mp4"

    _print_gpu_info()
    predict_fn, backend = load_model(args.model.strip() or "", args.backend, model_id, imgsz=args.imgsz)
    print(f"Backend: {backend}  |  Segmentation: {use_segmentation}")

    # Deadzone setup — needs the first frame
    deadzones = []
    if args.deadzone:
        cap = cv2.VideoCapture(args.input)
        ret, first_frame = cap.read()
        cap.release()
        if ret:
            print("Deadzone setup: left-click + drag to draw exclusion rectangles.")
            deadzones = draw_deadzones_interactive(first_frame)
            print(f"{len(deadzones)} deadzone(s) configured.")
        else:
            print("WARNING: Could not read first frame for deadzone setup.")

    process_video(
        input_path=args.input,
        output_path=args.output,
        predict_fn=predict_fn,
        backend=backend,
        confidence=args.confidence,
        nms_threshold=args.nms_threshold,
        track_activation_threshold=args.track_activation_threshold,
        lost_track_buffer=args.lost_track_buffer,
        minimum_matching_threshold=args.minimum_matching_threshold,
        minimum_consecutive_frames=args.minimum_consecutive_frames,
        disable_display=args.no_display,
        cyclist_class_id=args.cyclist_class_id,
        pedestrian_class_id=args.pedestrian_class_id,
        use_segmentation=use_segmentation,
        imgsz=args.imgsz,
        deadzones=deadzones,
        show_deadzones=args.show_deadzones,
        inference_only=args.inference_only,
    )


if __name__ == "__main__":
    main()
