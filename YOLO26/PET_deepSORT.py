"""
Post Encroachment Time (PET) conflict zone analysis using YOLO26 + DeepSORT.

Detection is driven by config.yaml (same config as deepSORT_yolo26.py), giving
access to the full multi-pass detection pipeline (full-frame, top-region, SAHI,
perspective warp), per-class tracker settings, and containment-aware NMS.

PET definition (standard): PET(A1,A2,CA) = t_entry(A2,CA) - t_exit(A1,CA), the
time gap between one actor leaving and the other entering the conflict area;
scale [0, inf). PET is undefined when both occupy the conflict area before either
leaves (overlap).
Reference: https://criticality-metrics.readthedocs.io/en/latest/time-scale/PET.html
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import yaml

import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MODEL_PATH = './yolo26l_100epoch_v5.pt'


# ---------------------------------------------------------------------------
# Model / config loading
# ---------------------------------------------------------------------------

def load_model(model_path, device, use_compile=False):
    """Load a YOLO26 model checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Loading YOLO26 model: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    if use_compile and device == 'cuda':
        try:
            model.model = torch.compile(model.model)
            print("torch.compile enabled (first inference will be slower while compiling).")
        except Exception as e:
            print(f"torch.compile unavailable, skipping: {e}")
    print(f"Model loaded on device: {device}")
    return model


def load_config(config_path):
    """Load YAML config file and return as a dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Detection helpers  (identical to deepSORT_yolo26.py)
# ---------------------------------------------------------------------------

def _run_detector(model, image, conf_threshold, iou_threshold, imgsz=None, half=False, device=None):
    """Run one Ultralytics YOLO detector pass and return raw results."""
    kwargs = {"conf": conf_threshold, "iou": iou_threshold, "agnostic_nms": False, "verbose": False}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    if half:
        kwargs["half"] = True
    if device is not None:
        kwargs["device"] = device
    return model(image, **kwargs)


def _extract_boxes(results, x_offset=0, y_offset=0, class_filter=None):
    """Extract [x1, y1, x2, y2, conf, cls] from an Ultralytics result list."""
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for (x1, y1, x2, y2), conf, cls in zip(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
        ):
            cls_int = int(cls)
            if class_filter is not None and cls_int not in class_filter:
                continue
            detections.append([
                float(x1 + x_offset), float(y1 + y_offset),
                float(x2 + x_offset), float(y2 + y_offset),
                float(conf), cls_int,
            ])
    return detections


def _clip_detection(det, frame_w, frame_h):
    """Clip a detection to frame bounds; returns None if the box collapses."""
    x1, y1, x2, y2, conf, cls_int = det
    x1 = max(0.0, min(float(frame_w - 1), x1))
    y1 = max(0.0, min(float(frame_h - 1), y1))
    x2 = max(0.0, min(float(frame_w - 1), x2))
    y2 = max(0.0, min(float(frame_h - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2, conf, cls_int]


# ---------------------------------------------------------------------------
# IoU / NMS helpers  (identical to deepSORT_yolo26.py)
# ---------------------------------------------------------------------------

def _compute_iou_xyxy(a, b):
    """Compute IoU for two [x1, y1, x2, y2, ...] detections."""
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    b_area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = a_area + b_area - inter
    return inter / denom if denom > 0.0 else 0.0


def _is_contained(inner, outer, min_overlap_fraction=0.8):
    """
    Return True if `inner` is substantially contained within `outer`.
    Checks whether >= min_overlap_fraction of inner's area overlaps outer.
    Catches the box-inside-box case that IoU-based NMS misses.
    """
    inter_x1 = max(inner[0], outer[0])
    inter_y1 = max(inner[1], outer[1])
    inter_x2 = min(inner[2], outer[2])
    inter_y2 = min(inner[3], outer[3])
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0.0:
        return False
    inner_area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return (inter / inner_area) >= min_overlap_fraction if inner_area > 0.0 else False


def _hard_nms_per_class(detections, iou_threshold=0.45, containment_fraction=0.8):
    """
    Two-phase greedy hard NMS per class.
    Phase 1: confidence-sorted IoU + containment suppression.
    Phase 2: size-based containment cleanup regardless of confidence.
    """
    by_class = defaultdict(list)
    for det in detections:
        by_class[int(det[5])].append(det)
    kept = []
    for class_dets in by_class.values():
        dets = sorted(class_dets, key=lambda d: d[4], reverse=True)
        survivors = []
        while dets:
            best = dets.pop(0)
            survivors.append(best)
            dets = [
                d for d in dets
                if _compute_iou_xyxy(best, d) <= iou_threshold
                and not _is_contained(d, best, containment_fraction)
            ]

        def _box_area(d):
            return max(0.0, d[2] - d[0]) * max(0.0, d[3] - d[1])

        final = [
            det for i, det in enumerate(survivors)
            if not any(
                _box_area(other) > _box_area(det)
                and _is_contained(det, other, containment_fraction)
                for j, other in enumerate(survivors) if j != i
            )
        ]
        kept.extend(final)
    return kept


def _soft_nms_per_class(detections, iou_threshold=0.5, sigma=0.5, score_threshold=1e-3):
    """Gaussian Soft-NMS on a single-class detection list."""
    dets = [d.copy() for d in detections]
    kept = []
    while dets:
        dets.sort(key=lambda d: d[4], reverse=True)
        best = dets.pop(0)
        kept.append(best)
        survivors = []
        for det in dets:
            iou = _compute_iou_xyxy(best, det)
            if iou > iou_threshold:
                det[4] *= np.exp(-(iou * iou) / max(sigma, 1e-6))
            if det[4] >= score_threshold:
                survivors.append(det)
        dets = survivors
    return kept


def _apply_crowd_postprocess(detections, crowd_mode, soft_nms_iou, soft_nms_sigma, score_threshold):
    """Optional soft-NMS crowd suppression, applied per class."""
    if crowd_mode != "soft-nms":
        return detections
    by_class = defaultdict(list)
    for det in detections:
        by_class[int(det[5])].append(det)
    merged = []
    for class_dets in by_class.values():
        merged.extend(
            _soft_nms_per_class(class_dets, iou_threshold=soft_nms_iou,
                                 sigma=soft_nms_sigma, score_threshold=score_threshold)
        )
    return merged


# ---------------------------------------------------------------------------
# Multi-pass detection helpers  (identical to deepSORT_yolo26.py)
# ---------------------------------------------------------------------------

def _run_top_region_pass(frame, model, iou_threshold, top_region_ratio, conf_threshold,
                         imgsz, class_filter, half=False, device=None):
    """Run detection on the upper region of the frame."""
    h, _ = frame.shape[:2]
    top_h = int(max(1, min(h, round(h * top_region_ratio))))
    roi = frame[:top_h, :]
    results = _run_detector(model, roi, conf_threshold, iou_threshold, imgsz=imgsz, half=half, device=device)
    return _extract_boxes(results, x_offset=0, y_offset=0, class_filter=class_filter)


def _build_homography(src_points, dst_size):
    """Compute H (src→dst) and H_inv (dst→src) from 4 calibration points."""
    src = np.array(src_points, dtype=np.float32)
    dw, dh = dst_size
    dst = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], dtype=np.float32)
    H     = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    return H, H_inv


def _warp_boxes_to_original(boxes, H_inv):
    """Map axis-aligned boxes from warped space back to original frame coordinates."""
    if not boxes:
        return []
    mapped = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        ch  = np.column_stack([corners, np.ones(4)])
        t   = (H_inv @ ch.T).T
        t   = t[:, :2] / t[:, 2:3]
        mapped.append([
            float(t[:, 0].min()), float(t[:, 1].min()),
            float(t[:, 0].max()), float(t[:, 1].max()),
            box[4], box[5],
        ])
    return mapped


def _run_warp_pass(frame, model, iou_threshold, conf_threshold, H, H_inv, dst_size,
                   imgsz, class_filter, half=False, device=None):
    """Perspective-corrected detection pass via inverse homography."""
    dw, dh = dst_size
    warped       = cv2.warpPerspective(frame, H, (dw, dh))
    results      = _run_detector(model, warped, conf_threshold, iou_threshold,
                                 imgsz=imgsz, half=half, device=device)
    warped_boxes = _extract_boxes(results, class_filter=class_filter)
    return _warp_boxes_to_original(warped_boxes, H_inv)


def _run_tiled_pass(frame, model, iou_threshold, conf_threshold, tile_size, tile_overlap,
                    imgsz, class_filter, half=False, device=None,
                    y_max_fraction=1.0, prescale=1.0):
    """SAHI-style tiled inference restricted to top fraction with optional prescale."""
    h, w = frame.shape[:2]
    y_limit = max(1, int(round(h * min(1.0, max(0.0, y_max_fraction)))))
    region  = frame[:y_limit, :]
    prescale = max(1.0, float(prescale))
    if prescale != 1.0:
        new_w = int(round(w * prescale))
        new_h = int(round(y_limit * prescale))
        region = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rh, rw = region.shape[:2]
    tile = max(64, int(tile_size))
    overlap = max(0.0, min(0.8, float(tile_overlap)))
    stride = max(32, int(round(tile * (1.0 - overlap))))
    y_starts = list(range(0, max(1, rh - tile + 1), stride))
    x_starts = list(range(0, max(1, rw - tile + 1), stride))
    if not y_starts or y_starts[-1] != max(0, rh - tile):
        y_starts.append(max(0, rh - tile))
    if not x_starts or x_starts[-1] != max(0, rw - tile):
        x_starts.append(max(0, rw - tile))
    tiles, offsets = [], []
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(rh, y0 + tile)
            x1 = min(rw, x0 + tile)
            tile_img = region[y0:y1, x0:x1]
            if tile_img.size == 0:
                continue
            tiles.append(tile_img)
            offsets.append((x0, y0))
    if not tiles:
        return []
    kwargs = {"conf": conf_threshold, "iou": iou_threshold, "agnostic_nms": False, "verbose": False}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    if half:
        kwargs["half"] = True
    if device is not None:
        kwargs["device"] = device
    all_results = model(tiles, **kwargs)
    inv_scale = 1.0 / prescale
    detections = []
    for result, (x_offset, y_offset) in zip(all_results, offsets):
        boxes = _extract_boxes([result], x_offset=x_offset, y_offset=y_offset, class_filter=class_filter)
        if prescale != 1.0:
            boxes = [
                [b[0] * inv_scale, b[1] * inv_scale,
                 b[2] * inv_scale, b[3] * inv_scale, b[4], b[5]]
                for b in boxes
            ]
        detections.extend(boxes)
    return detections


def _split_detections_by_class(detections, class_ids, min_confidences=None):
    """Split flat detection list into per-class DeepSort input tuples."""
    min_confidences = min_confidences or {}
    by_class = {cls_id: [] for cls_id in class_ids}
    for x1, y1, x2, y2, conf, cls_int in detections:
        if cls_int not in by_class:
            continue
        if float(conf) < min_confidences.get(cls_int, 0.0):
            continue
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        if bbox[2] > 0 and bbox[3] > 0:
            by_class[cls_int].append((bbox, float(conf), cls_int))
    return by_class


# ---------------------------------------------------------------------------
# Deadzone helpers
# ---------------------------------------------------------------------------

def _is_in_deadzone(det, deadzones):
    """Return True if the detection's centre point falls inside any deadzone polygon."""
    if not deadzones:
        return False
    cx = (det[0] + det[2]) / 2.0
    cy = (det[1] + det[3]) / 2.0
    for zone in deadzones:
        pts = np.array(zone, dtype=np.float32)
        if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
            return True
    return False


def _draw_deadzones_interactive(frame, window_name="Draw Deadzones"):
    """
    Interactive deadzone editor shown on the first video frame.

    Controls
    --------
    Left-click + drag : draw a rectangular deadzone
    U                 : undo last zone
    C                 : clear all zones
    Enter / Space     : confirm and continue
    Esc               : skip (return empty list — no deadzones)
    """
    zones    = []
    drawing  = False
    start_pt = None
    curr_pt  = None

    def _mouse(event, x, y, flags, param):
        nonlocal drawing, start_pt, curr_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing  = True
            start_pt = (x, y)
            curr_pt  = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            curr_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
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
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, _mouse)

        while True:
            img = frame.copy()
            # Overlay existing zones (semi-transparent dark red)
            for zone in zones:
                pts  = np.array(zone, dtype=np.int32)
                _ov  = img.copy()
                cv2.fillPoly(_ov, [pts], (0, 0, 180))
                cv2.addWeighted(_ov, 0.35, img, 0.65, 0, img)
                cv2.polylines(img, [pts], True, (0, 0, 220), 2)
            # Rubber-band rectangle while drawing
            if drawing and start_pt and curr_pt:
                cv2.rectangle(img, start_pt, curr_pt, (0, 220, 220), 2)
            # Instructions (black outline then white fill for readability on any background)
            for i, txt in enumerate(instructions):
                y_pos = 22 + i * 24
                cv2.putText(img, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, img)

            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):        # Enter or Space → confirm
                break
            elif key == 27:            # Esc → skip (no deadzones)
                zones = []
                break
            elif key == ord('u') and zones:
                zones.pop()
            elif key == ord('c'):
                zones = []

        cv2.destroyWindow(window_name)
    except (cv2.error, Exception) as e:
        print(f"Deadzone drawing unavailable: {e}")
        zones = []

    return zones


# ---------------------------------------------------------------------------
# PET helpers  (identical to root PET_deepSORT.py)
# ---------------------------------------------------------------------------

def _bbox_overlap_cells(bbox_xyxy, width, height, grid_rows, grid_cols):
    """Return set of (row, col) grid cell indices that the bbox overlaps."""
    x1, y1, x2, y2 = bbox_xyxy
    cell_w = width / grid_cols
    cell_h = height / grid_rows
    cells = set()
    col_start = max(0, int(x1 / cell_w))
    col_end   = min(grid_cols - 1, int(x2 / cell_w))
    row_start = max(0, int(y1 / cell_h))
    row_end   = min(grid_rows - 1, int(y2 / cell_h))
    for r in range(row_start, row_end + 1):
        for c in range(col_start, col_end + 1):
            cells.add((r, c))
    return cells


def _get_pet_output_dir(input_video_path, root="PET_Analysis"):
    """
    Return (output_dir_path, run_number) for this video.
    Creates root if needed; does not create the run subdir (caller creates when writing).
    """
    video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
    if not video_basename:
        video_basename = "video"
    os.makedirs(root, exist_ok=True)
    prefix = video_basename + "_"
    run_number = 1
    for name in os.listdir(root):
        if name.startswith(prefix) and os.path.isdir(os.path.join(root, name)):
            try:
                n = int(name[len(prefix):])
                if n >= run_number:
                    run_number = n + 1
            except ValueError:
                continue
    run_dir_name = f"{video_basename}_{run_number}"
    output_dir = os.path.join(root, run_dir_name)
    return output_dir, run_number


def _neighbor_cells(r, c, grid_rows, grid_cols, include_self=True):
    """Return set of (row, col) for cell (r,c) and its 3x3 neighbors."""
    out = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not include_self and dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
                out.add((nr, nc))
    return out


def _select_grid_cell_for_pet(frame, grid_rows, grid_cols, window_name="Select PET grid cell"):
    """
    Let the user select a single grid cell by clicking on the frame.
    Returns (row, col) or None if selection fails or is cancelled.
    """
    height, width = frame.shape[:2]
    cell_w = width / grid_cols
    cell_h = height / grid_rows
    selected = {'cell': None}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            c = int(x / cell_w)
            r = int(y / cell_h)
            if 0 <= r < grid_rows and 0 <= c < grid_cols:
                selected['cell'] = (r, c)

    try:
        display = frame.copy()
        for i in range(1, grid_rows):
            y = int(i * cell_h)
            cv2.line(display, (0, y), (width, y), (60, 60, 60), 1)
        for j in range(1, grid_cols):
            x = int(j * cell_w)
            cv2.line(display, (x, 0), (x, height), (60, 60, 60), 1)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, _on_mouse)

        while True:
            img = display.copy()
            if selected['cell'] is not None:
                r, c = selected['cell']
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.imshow(window_name, img)
            key = cv2.waitKey(20) & 0xFF
            if selected['cell'] is not None and key in (13, 32, ord('q'), 27):
                break
        cv2.destroyWindow(window_name)
        return selected['cell']
    except (cv2.error, Exception):
        try:
            cv2.destroyWindow(window_name)
        except (cv2.error, Exception):
            pass
        return None


def _build_heatmap_image(cell_pet_values, grid_rows, grid_cols, width, height, fps, max_pet_time, first_frame):
    """
    Build a heatmap: average standard PET per cell.
    Low PET (critical) = red; high PET (safe) = blue.
    Returns BGR image blended over first_frame.
    """
    max_sec = max_pet_time / fps if fps > 0 else 1.0
    heat = np.full((grid_rows, grid_cols), np.nan, dtype=np.float64)
    for (r, c), values in cell_pet_values.items():
        if 0 <= r < grid_rows and 0 <= c < grid_cols and values:
            heat[r, c] = np.mean([abs(v) for v in values])

    heat_uint8 = np.full((grid_rows, grid_cols), 128, dtype=np.uint8)
    valid = ~np.isnan(heat)
    if np.any(valid):
        v = np.clip(heat[valid], 0.0, max_sec)
        norm = v / max_sec
        heat_uint8[valid] = (255 * (1.0 - norm)).clip(0, 255).astype(np.uint8)

    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.resize(heat_color, (width, height), interpolation=cv2.INTER_NEAREST)

    mask = (heat_uint8 != 128)
    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    blended = cv2.addWeighted(heat_color, 0.45, first_frame, 0.55, 0)
    overlay = np.where(mask[:, :, np.newaxis], blended, first_frame).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# Main PET video processing
# ---------------------------------------------------------------------------

def process_video(
    input_video_path,
    output_video_path,
    model,
    cfg,
    device='cpu',
    disable_display=True,
    # PET-specific parameters (not in config)
    grid_size=10,
    max_pet_time=30,
    use_neighbors=True,
    output_csv_path=None,
    output_heatmap_path=None,
    show_grid=False,
    single_cell_mode=False,
    deadzones=None,
    show_deadzones=False,
):
    deadzones = deadzones or []
    """
    Process video for PET conflict analysis.

    Detection uses the same multi-pass pipeline as deepSORT_yolo26.py (driven by
    config.yaml). PET grid logic, CSV, heatmap, and plot outputs are unchanged
    from the original PET_deepSORT.py.
    """
    # --- Unpack config ---
    inf_cfg  = cfg.get('inference', {})
    pass_cfg = cfg.get('passes', {})
    nms_cfg  = cfg.get('nms', {})
    dbg_cfg  = cfg.get('debug', {})
    classes  = cfg.get('classes', [])

    confidence_threshold = inf_cfg.get('confidence', 0.65)
    iou_threshold        = inf_cfg.get('iou', 0.7)
    half                 = inf_cfg.get('half', False)
    imgsz                = inf_cfg.get('imgsz', 0) or None
    downscale_width      = inf_cfg.get('downscale_width', 0)
    downscale_height     = inf_cfg.get('downscale_height', 0)

    top_cfg          = pass_cfg.get('top_region', {})
    top_region_pass  = top_cfg.get('enabled', False)
    top_region_ratio = top_cfg.get('ratio', 0.45)
    top_region_imgsz = top_cfg.get('imgsz', 0) or None
    top_region_conf  = top_cfg.get('confidence') or None

    sahi_cfg      = pass_cfg.get('sahi', {})
    tile_mode     = 'sahi' if sahi_cfg.get('enabled', False) else 'off'
    tile_size     = sahi_cfg.get('tile_size', 480)
    tile_overlap  = sahi_cfg.get('tile_overlap', 0.4)
    tile_interval = sahi_cfg.get('tile_interval', 1)
    tile_imgsz    = sahi_cfg.get('imgsz', 0) or None
    tile_conf     = sahi_cfg.get('confidence') or None
    tile_y_max    = sahi_cfg.get('y_max_fraction', 1.0)
    tile_prescale = sahi_cfg.get('prescale', 1.0)

    warp_cfg     = pass_cfg.get('warp', {})
    warp_enabled = warp_cfg.get('enabled', False)
    warp_H       = warp_H_inv = warp_dst_size = None
    warp_conf    = warp_imgsz = None
    if warp_enabled:
        warp_src      = warp_cfg.get('src_points')
        warp_dst_size = warp_cfg.get('dst_size')
        warp_conf     = warp_cfg.get('confidence') or None
        warp_imgsz    = warp_cfg.get('imgsz', 0) or None
        if warp_src and warp_dst_size and len(warp_src) == 4:
            warp_H, warp_H_inv = _build_homography(warp_src, warp_dst_size)
            print(f"Warp pass enabled: {warp_src} → {warp_dst_size}")
        else:
            print("WARNING: warp pass enabled but src_points/dst_size not configured. "
                  "Run warp_calibrate.py to generate them. Warp pass disabled.")
            warp_enabled = False

    nms_iou         = nms_cfg.get('hard_iou', 0.45)
    nms_containment = nms_cfg.get('containment_fraction', 0.8)
    crowd_mode      = nms_cfg.get('crowd_mode', 'off')
    soft_nms_iou    = nms_cfg.get('soft_nms_iou', 0.25)
    soft_nms_sigma  = nms_cfg.get('soft_nms_sigma', 0.2)

    debug_detections = dbg_cfg.get('log_detections', False)
    class_ids        = {c['id'] for c in classes}
    class_map        = {c['id']: c for c in classes}
    min_confidences  = {c['id']: c['min_confidence'] for c in classes if 'min_confidence' in c}

    # --- Open video (and optionally show first frame for single-cell selection) ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {input_video_path}")

    grid_rows = grid_cols = grid_size
    selected_cell = None

    if single_cell_mode:
        ret_sel, frame_sel = cap.read()
        if ret_sel and frame_sel is not None:
            tqdm.write(
                "Select a grid cell by clicking on the frame; "
                "press Enter/Space or q/ESC to confirm."
            )
            chosen = _select_grid_cell_for_pet(frame_sel, grid_rows, grid_cols)
            if chosen is not None:
                selected_cell = chosen
                tqdm.write(
                    f"Using single grid cell (row={selected_cell[0]}, col={selected_cell[1]}) "
                    "for PET computation."
                )
            else:
                tqdm.write("No grid cell selected; falling back to full-grid PET computation.")
        else:
            tqdm.write("Could not read first frame for grid selection; falling back to full-grid PET computation.")
        cap.release()
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not re-open video file {input_video_path}")

    fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cell_w = width / grid_cols
    cell_h = height / grid_rows

    # --- Per-class DeepSORT trackers (params from config) ---
    if DeepSort is None:
        raise RuntimeError(
            "deep_sort_realtime is not installed. "
            "Install it with: pip install deep-sort-realtime"
        )
    trackers = {}
    for cls_cfg in classes:
        cid   = cls_cfg['id']
        t_cfg = cls_cfg.get('tracker', {})
        trackers[cid] = DeepSort(
            max_age=t_cfg.get('max_age', 15),
            max_iou_distance=t_cfg.get('max_iou_distance', 0.5),
            n_init=t_cfg.get('n_init', 3),
            nms_max_overlap=t_cfg.get('nms_max_overlap', 0.3),
            max_cosine_distance=t_cfg.get('max_cosine_distance', 0.2),
            embedder=t_cfg.get('embedder', 'mobilenet'),
        )

    # --- Video writer ---
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
    ]
    out = None
    for _codec_name, fourcc in codecs_to_try:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if out.isOpened():
            break
        out.release()
        out = None
    if out is None:
        raise RuntimeError("Could not create VideoWriter with any tried codec.")

    # --- PET state ---
    # grid[(r,c)][class_id] = [(frame, track_id), ...]
    grid_occupancy = defaultdict(lambda: defaultdict(list))

    conflict_events           = []
    cell_pet_values           = defaultdict(list)
    frame_to_pets             = defaultdict(list)
    conflict_cells_prev_frame = set()

    # --- Single-cell PET timer state (active in single_cell_mode with a selected_cell) ---
    _sc_timer_state         = 'idle'  # 'idle' | 'running' | 'locked'
    _sc_timer_start_frame   = None    # frame_count when first class entered the cell
    _sc_timer_locked_secs   = None    # PET seconds captured at the moment of lock
    _sc_last_pet_secs       = None    # most recent pet_seconds computed for selected_cell
    _sc_exit_frame          = None    # frame when the cell last became empty (gap-phase tracking)
    _sc_lock_hold_frames    = max(1, int(fps * 3))  # hold red "locked" display for ~3 s
    _sc_lock_hold_remaining = 0

    frame_count       = 0
    _cached_tile_dets = []
    display_available = not disable_display

    pbar = tqdm(total=total_frames if total_frames > 0 else None, unit="frame", desc="PET analysis")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]

            # Optional downscale for faster inference
            if (downscale_width > 0 and downscale_height > 0
                    and (frame_w > downscale_width or frame_h > downscale_height)):
                proc_frame = cv2.resize(frame, (downscale_width, downscale_height))
                scale_x = frame_w / downscale_width
                scale_y = frame_h / downscale_height
            else:
                proc_frame = frame
                scale_x = scale_y = 1.0
            proc_h, proc_w = proc_frame.shape[:2]

            # --- Multi-pass detection ---
            full_dets = _extract_boxes(
                _run_detector(model, proc_frame, confidence_threshold, iou_threshold,
                              imgsz=imgsz, half=half, device=device),
                class_filter=class_ids,
            )

            top_dets = []
            if top_region_pass:
                top_dets = _run_top_region_pass(
                    proc_frame, model, iou_threshold, top_region_ratio,
                    top_region_conf or confidence_threshold,
                    top_region_imgsz, class_ids, half=half, device=device,
                )

            run_tiles = tile_mode == "sahi" and (frame_count % max(1, tile_interval) == 0)
            if run_tiles:
                tile_dets = _run_tiled_pass(
                    proc_frame, model, iou_threshold,
                    tile_conf or confidence_threshold,
                    tile_size, tile_overlap, tile_imgsz, class_ids, half=half, device=device,
                    y_max_fraction=tile_y_max, prescale=tile_prescale,
                )
                _cached_tile_dets = tile_dets
            elif tile_mode == "sahi":
                tile_dets = _cached_tile_dets
            else:
                tile_dets = []

            warp_dets = []
            if warp_enabled:
                warp_dets = _run_warp_pass(
                    proc_frame, model, iou_threshold,
                    warp_conf or confidence_threshold,
                    warp_H, warp_H_inv, warp_dst_size,
                    warp_imgsz, class_ids, half=half, device=device,
                )

            # Merge, clip, scale back to original frame coords, NMS
            merged_dets = []
            for det in full_dets + top_dets + tile_dets + warp_dets:
                clipped = _clip_detection(det, frame_w=proc_w, frame_h=proc_h)
                if clipped is not None:
                    merged_dets.append(clipped)
            if scale_x != 1.0 or scale_y != 1.0:
                merged_dets = [
                    [d[0]*scale_x, d[1]*scale_y, d[2]*scale_x, d[3]*scale_y, d[4], d[5]]
                    for d in merged_dets
                ]
            merged_dets    = _hard_nms_per_class(merged_dets, iou_threshold=nms_iou,
                                                 containment_fraction=nms_containment)
            processed_dets = _apply_crowd_postprocess(
                merged_dets, crowd_mode=crowd_mode,
                soft_nms_iou=soft_nms_iou, soft_nms_sigma=soft_nms_sigma,
                score_threshold=confidence_threshold * 0.5,
            )

            # Deadzone filter: suppress any detection whose centre falls inside
            # a user-defined exclusion zone (applied in original frame coordinates).
            if deadzones:
                processed_dets = [d for d in processed_dets
                                  if not _is_in_deadzone(d, deadzones)]

            if debug_detections and frame_count % 10 == 0:
                print(f"Frame {frame_count}: full={len(full_dets)} top={len(top_dets)} "
                      f"tile={len(tile_dets)} final={len(processed_dets)}")

            # --- Update trackers ---
            by_class = _split_detections_by_class(processed_dets, class_ids, min_confidences)
            confirmed_by_class = {}
            for cid in class_ids:
                dets   = by_class.get(cid, [])
                tracks = trackers[cid].update_tracks(dets, frame=frame)

                # Post-tracker containment check (same as deepSORT_yolo26.py)
                confirmed = [t for t in tracks if t.is_confirmed()]

                def _tlbr(t):
                    return list(map(int, t.to_tlbr()))

                def _tarea(t):
                    b = _tlbr(t)
                    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

                visible = [
                    t for t in confirmed
                    if not any(
                        _tarea(other) > _tarea(t)
                        and _is_contained(_tlbr(t), _tlbr(other), nms_containment)
                        for other in confirmed if other is not t
                    )
                ]
                confirmed_by_class[cid] = visible

            # --- Update PET grid occupancy ---
            for cid, tracks in confirmed_by_class.items():
                for track in tracks:
                    tlbr  = track.to_tlbr()
                    cells = _bbox_overlap_cells(
                        (tlbr[0], tlbr[1], tlbr[2], tlbr[3]),
                        width, height, grid_rows, grid_cols,
                    )
                    for (r, c) in cells:
                        grid_occupancy[(r, c)][cid].append((frame_count, track.track_id))

            # Prune occupancy older than the conflict window
            cutoff = frame_count - max_pet_time
            for key in list(grid_occupancy.keys()):
                for cid in list(grid_occupancy[key].keys()):
                    grid_occupancy[key][cid] = [
                        (f, tid) for (f, tid) in grid_occupancy[key][cid] if f > cutoff
                    ]
                empty = all(len(v) == 0 for v in grid_occupancy[key].values())
                if empty:
                    del grid_occupancy[key]

            # --- PET conflict detection ---
            sorted_cids = sorted(class_ids)
            cid_a, cid_b = (sorted_cids[0], sorted_cids[1]) if len(sorted_cids) >= 2 else (None, None)

            conflict_cells_this_frame = set()

            if cid_a is not None and cid_b is not None:
                if selected_cell is not None:
                    cells_to_check = {selected_cell} if selected_cell in grid_occupancy else set()
                else:
                    cells_to_check = set(grid_occupancy.keys())

                for (r, c) in list(cells_to_check):
                    if (r, c) in conflict_cells_this_frame:
                        continue

                    list_a, list_b = [], []
                    if use_neighbors and not single_cell_mode:
                        cells_region = _neighbor_cells(r, c, grid_rows, grid_cols, include_self=True)
                    else:
                        cells_region = {(r, c)}
                    for (nr, nc) in cells_region:
                        if (nr, nc) not in grid_occupancy:
                            continue
                        list_a.extend(grid_occupancy[(nr, nc)].get(cid_a, []))
                        list_b.extend(grid_occupancy[(nr, nc)].get(cid_b, []))

                    if not list_a or not list_b:
                        continue

                    pet_frames        = None
                    signed_pet_frames = None
                    best_id_a = best_id_b = None
                    for (fa, ida) in list_a:
                        for (fb, idb) in list_b:
                            d = abs(fa - fb)
                            if pet_frames is None or d < pet_frames:
                                pet_frames        = d
                                signed_pet_frames = fa - fb
                                best_id_a         = ida
                                best_id_b         = idb

                    if pet_frames is None:
                        continue

                    conflict_cells_this_frame.add((r, c))
                    pet_seconds        = (pet_frames / fps) if fps > 0 else 0.0
                    signed_pet_seconds = (signed_pet_frames / fps) if fps > 0 else 0.0
                    time_sec           = frame_count / fps if fps > 0 else 0.0
                    overlap            = pet_frames == 0
                    frame_to_pets[frame_count].append(signed_pet_seconds)
                    # Capture PET value for the single-cell timer overlay
                    if selected_cell is not None and (r, c) == selected_cell:
                        _sc_last_pet_secs = pet_seconds

                    if (r, c) not in conflict_cells_prev_frame:
                        name_a = class_map[cid_a]['name']
                        name_b = class_map[cid_b]['name']
                        conflict_events.append({
                            'frame':              frame_count,
                            'time_sec':           round(time_sec, 3),
                            'cell_row':           r,
                            'cell_col':           c,
                            'pet_frames':         pet_frames,
                            'pet_seconds':        round(pet_seconds, 3),
                            'pet_undefined_overlap': overlap,
                            'signed_pet_frames':  signed_pet_frames,
                            'signed_pet_seconds': round(signed_pet_seconds, 3),
                            f'{name_a.lower()}_id': best_id_a,
                            f'{name_b.lower()}_id': best_id_b,
                        })
                        cell_pet_values[(r, c)].append(signed_pet_seconds)

            # --- Single-cell PET timer state machine ---
            if single_cell_mode and selected_cell is not None:
                _classes_in_cell_now = set()
                for _cid, _trks in confirmed_by_class.items():
                    for _trk in _trks:
                        _b = _trk.to_tlbr()
                        if selected_cell in _bbox_overlap_cells(
                            (_b[0], _b[1], _b[2], _b[3]),
                            width, height, grid_rows, grid_cols,
                        ):
                            _classes_in_cell_now.add(_cid)
                            break

                if _sc_timer_state == 'idle':
                    if _classes_in_cell_now:
                        _sc_timer_state       = 'running'
                        _sc_timer_start_frame = frame_count
                        _sc_exit_frame        = None
                elif _sc_timer_state == 'running':
                    if selected_cell in conflict_cells_this_frame:
                        # PET event fired → lock and display actual PET seconds
                        _sc_timer_state         = 'locked'
                        _sc_timer_locked_secs   = (
                            _sc_last_pet_secs if _sc_last_pet_secs is not None
                            else (frame_count - (_sc_exit_frame or _sc_timer_start_frame)) / fps
                                 if fps > 0 else 0.0
                        )
                        _sc_lock_hold_remaining = _sc_lock_hold_frames
                        _sc_exit_frame          = None
                    elif _classes_in_cell_now:
                        # An entity is present (could be re-entry) — clear gap tracking
                        _sc_exit_frame = None
                    else:
                        # Cell is now empty: record exit frame and keep timer running.
                        # The timer shows the growing gap until max_pet_time expires.
                        if _sc_exit_frame is None:
                            _sc_exit_frame = frame_count
                        if (frame_count - _sc_exit_frame) > max_pet_time:
                            # Occupancy window expired — no PET possible from this sequence
                            _sc_timer_state       = 'idle'
                            _sc_timer_start_frame = None
                            _sc_exit_frame        = None
                            _sc_last_pet_secs     = None
                elif _sc_timer_state == 'locked':
                    _sc_lock_hold_remaining -= 1
                    if _sc_lock_hold_remaining <= 0:
                        _sc_timer_state       = 'idle'
                        _sc_timer_start_frame = None
                        _sc_timer_locked_secs = None
                        _sc_last_pet_secs     = None
                        _sc_exit_frame        = None

            # --- Annotate frame ---
            annotated_frame = frame.copy()
            # Draw deadzone overlays (only when --show-deadzones is enabled)
            if show_deadzones:
                for _dz in deadzones:
                    _pts = np.array(_dz, dtype=np.int32)
                    _ov  = annotated_frame.copy()
                    cv2.fillPoly(_ov, [_pts], (0, 0, 180))
                    cv2.addWeighted(_ov, 0.25, annotated_frame, 0.75, 0, annotated_frame)
                    cv2.polylines(annotated_frame, [_pts], True, (0, 0, 200), 1)

            if show_grid:
                for i in range(1, grid_rows):
                    y = int(i * cell_h)
                    cv2.line(annotated_frame, (0, y), (width, y), (60, 60, 60), 1)
                for j in range(1, grid_cols):
                    x = int(j * cell_w)
                    cv2.line(annotated_frame, (x, 0), (x, height), (60, 60, 60), 1)

            for (r, c) in conflict_cells_this_frame:
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Selected-cell PET timer overlay (single_cell_mode only)
            if single_cell_mode and selected_cell is not None:
                _sr, _sc_col = selected_cell
                _sx1 = int(_sc_col * cell_w);       _sx2 = int((_sc_col + 1) * cell_w)
                _sy1 = int(_sr     * cell_h);       _sy2 = int((_sr     + 1) * cell_h)
                _cell_px = _sx2 - _sx1
                _fscale  = max(0.45, min(1.4, _cell_px / 120))
                _thick   = max(1, round(_fscale * 2))
                if _sc_timer_state == 'idle':
                    # Faint border marks the selected cell even when idle
                    cv2.rectangle(annotated_frame, (_sx1, _sy1), (_sx2, _sy2), (160, 160, 160), 1)
                elif _sc_timer_state == 'running':
                    if _sc_exit_frame is None:
                        # Entity currently in cell — show presence duration
                        _elapsed = (frame_count - _sc_timer_start_frame) / fps if fps > 0 else 0.0
                        _lbl = f"{_elapsed:.2f}s"
                    else:
                        # Gap phase — show time elapsed since entity left
                        _gap = (frame_count - _sc_exit_frame) / fps if fps > 0 else 0.0
                        _lbl = f"GAP {_gap:.2f}s"
                    _clr     = (0, 210, 0)
                    cv2.rectangle(annotated_frame, (_sx1, _sy1), (_sx2, _sy2), _clr, 2)
                    (_tw, _th), _bl = cv2.getTextSize(_lbl, cv2.FONT_HERSHEY_SIMPLEX, _fscale, _thick)
                    _tx = (_sx1 + _sx2) // 2 - _tw // 2
                    _ty = (_sy1 + _sy2) // 2 + _th // 2
                    cv2.rectangle(annotated_frame, (_tx - 3, _ty - _th - _bl - 2),
                                  (_tx + _tw + 3, _ty + _bl + 2), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, _lbl, (_tx, _ty),
                                cv2.FONT_HERSHEY_SIMPLEX, _fscale, _clr, _thick)
                elif _sc_timer_state == 'locked':
                    _lbl = f"PET {_sc_timer_locked_secs:.2f}s"
                    _clr = (40, 40, 220)
                    cv2.rectangle(annotated_frame, (_sx1, _sy1), (_sx2, _sy2), _clr, 2)
                    (_tw, _th), _bl = cv2.getTextSize(_lbl, cv2.FONT_HERSHEY_SIMPLEX, _fscale, _thick)
                    _tx = (_sx1 + _sx2) // 2 - _tw // 2
                    _ty = (_sy1 + _sy2) // 2 + _th // 2
                    cv2.rectangle(annotated_frame, (_tx - 3, _ty - _th - _bl - 2),
                                  (_tx + _tw + 3, _ty + _bl + 2), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, _lbl, (_tx, _ty),
                                cv2.FONT_HERSHEY_SIMPLEX, _fscale, _clr, _thick)

            for cid, tracks in confirmed_by_class.items():
                cls_cfg    = class_map[cid]
                color      = tuple(cls_cfg['color'])
                text_color = tuple(cls_cfg['text_color'])
                prefix     = cls_cfg['name'][0]
                for track in tracks:
                    tlbr = track.to_tlbr()
                    x1, y1, x2, y2 = int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    conf_val = track.det_conf
                    conf_str = f" {conf_val:.2f}" if conf_val is not None else ""
                    label    = f"{prefix}#{track.track_id}{conf_str}"
                    lsz      = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - lsz[1] - 8),
                                  (x1 + lsz[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            if conflict_cells_this_frame:
                cv2.putText(
                    annotated_frame,
                    f"CONFLICT ZONES: {len(conflict_cells_this_frame)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )

            # Single-cell PET timer HUD (top-left, stacks below conflict-zones line if present)
            if single_cell_mode and selected_cell is not None and _sc_timer_state != 'idle':
                _hud_y = 65 if conflict_cells_this_frame else 30
                if _sc_timer_state == 'running':
                    if _sc_exit_frame is None:
                        _hud_elapsed = (frame_count - _sc_timer_start_frame) / fps if fps > 0 else 0.0
                        _hud_txt   = f"TIMER  {_hud_elapsed:.2f}s"
                    else:
                        _hud_gap   = (frame_count - _sc_exit_frame) / fps if fps > 0 else 0.0
                        _hud_txt   = f"GAP  {_hud_gap:.2f}s"
                    _hud_color   = (0, 210, 0)
                else:
                    _hud_txt     = f"PET LOCKED  {_sc_timer_locked_secs:.2f}s"
                    _hud_color   = (40, 40, 220)
                (_hw, _hh), _ = cv2.getTextSize(_hud_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated_frame, (8, _hud_y - _hh - 4),
                              (14 + _hw, _hud_y + 4), (0, 0, 0), -1)
                cv2.putText(annotated_frame, _hud_txt,
                            (10, _hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _hud_color, 2)

            out.write(annotated_frame)
            conflict_cells_prev_frame = conflict_cells_this_frame
            frame_count += 1
            pbar.update(1)

            if display_available and annotated_frame is not None:
                try:
                    cv2.imshow('YOLO26 PET Analysis', annotated_frame)
                except (cv2.error, Exception):
                    display_available = False
            if display_available:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        tqdm.write("Interrupted by user")
    finally:
        pbar.close()
        cap.release()
        out.release()
        if display_available:
            try:
                cv2.destroyAllWindows()
            except (cv2.error, Exception):
                pass

    # --- CSV output ---
    base = os.path.splitext(output_video_path)[0]
    if output_csv_path is None:
        output_csv_path = base + "_PET_conflicts.csv"

    df = pd.DataFrame(conflict_events)
    if not df.empty:
        df.to_csv(output_csv_path, index=False)

    # --- Heatmap ---
    output_heatmap_path = output_heatmap_path or (base + "_PET_heatmap.png")
    cap_heat = cv2.VideoCapture(input_video_path)
    if cap_heat.isOpened():
        ret, first_frame = cap_heat.read()
        cap_heat.release()
        if ret and first_frame is not None and cell_pet_values:
            heatmap_img = _build_heatmap_image(
                cell_pet_values, grid_rows, grid_cols, width, height, fps, max_pet_time, first_frame
            )
            cv2.imwrite(output_heatmap_path, heatmap_img)

    # --- Average PET over time plot ---
    output_plot_path = base + "_PET_over_time.png"
    if frame_to_pets:
        frames_all   = np.arange(frame_count, dtype=int)
        time_sec_arr = frames_all.astype(float) / fps if fps > 0 else frames_all.astype(float)
        avg_pet_arr  = np.full_like(time_sec_arr, np.nan, dtype=float)
        for f, vals in frame_to_pets.items():
            if 0 <= f < frame_count and vals:
                avg_pet_arr[f] = np.mean([abs(p) for p in vals])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_sec_arr, avg_pet_arr, color="steelblue", linewidth=1)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, None)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Average PET (s)")
        ax.set_title("Average PET over time (lower = more critical)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_plot_path, dpi=150)
        plt.close(fig)

    # --- Risk 1/(1+PET) over time plot ---
    output_risk_plot_path = base + "_Risk_PET_over_time.png"
    if frame_to_pets:
        frames_sorted  = sorted(frame_to_pets.keys())
        time_sec_arr   = np.array(frames_sorted, dtype=float) / fps if fps > 0 else np.array(frames_sorted)
        risk_per_frame = np.array([
            np.mean([1.0 / (1.0 + abs(p)) for p in frame_to_pets[f]]) for f in frames_sorted
        ])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_sec_arr, risk_per_frame, color="crimson", linewidth=1)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Risk 1/(1+PET)")
        ax.set_title("Risk over time (higher = lower PET = more critical)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_risk_plot_path, dpi=150)
        plt.close(fig)

    tqdm.write(f"Video:   {os.path.abspath(output_video_path)}")
    tqdm.write(f"CSV:     {os.path.abspath(output_csv_path)}")
    tqdm.write(f"Heatmap: {os.path.abspath(output_heatmap_path)}")
    if frame_to_pets:
        tqdm.write(f"Plot:    {os.path.abspath(output_plot_path)}")
        tqdm.write(f"Risk:    {os.path.abspath(output_risk_plot_path)}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YOLO26 PET conflict zone detection with config-driven multi-pass detection."
    )
    # Config / model / IO (same pattern as deepSORT_yolo26.py)
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config file.")
    parser.add_argument("--input",  "-i", default=None, help="Override input video path from config.")
    parser.add_argument("--output", "-o", default=None, help="Override output video path.")
    parser.add_argument("--model",  "-m", default=None, help="Override model path from config.")

    # PET-specific parameters
    parser.add_argument("--grid-size",    type=int,   default=100,  help="N for NxN conflict grid (default: 100).")
    parser.add_argument("--max-pet-time", type=int,   default=10,   help="Conflict window in frames (default: 10).")
    parser.add_argument("--no-neighbors", action="store_true",      help="Disable 3x3 neighbor cell aggregation.")
    parser.add_argument("--show-grid",    action="store_true",      help="Draw faint grid lines on output video.")
    parser.add_argument("--no-grid",      action="store_true",
                        help="Single-cell mode: user clicks one cell; PET computed only in that cell.")
    parser.add_argument("--csv",     metavar="PATH",                help="Output CSV path (default: auto).")
    parser.add_argument("--heatmap", metavar="PATH",                help="Output heatmap image path (default: auto).")
    parser.add_argument("--display", action="store_true",           help="Show live preview window.")
    parser.add_argument("--deadzone", action="store_true",
                        help="Interactively draw rectangular deadzones on the first frame before "
                             "processing. Detections whose centre falls inside a deadzone are "
                             "suppressed. Left-click + drag to draw; U undo; C clear; Enter confirm.")
    parser.add_argument("--show-deadzones", action="store_true",
                        help="Render deadzone overlays on the output video / live preview "
                             "(hidden by default; requires --deadzone to have any effect).")

    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.input:
        cfg['input'] = args.input
    if args.output:
        cfg['output'] = args.output
    if args.model:
        cfg['model'] = args.model

    input_path = cfg['input']
    model_path = cfg['model']

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Auto-generate output path in PET_Analysis/ directory
    if not cfg.get('output') and not args.output:
        output_dir, run_number = _get_pet_output_dir(input_path)
        os.makedirs(output_dir, exist_ok=True)
        video_basename = os.path.splitext(os.path.basename(input_path))[0] or "video"
        cfg['output'] = os.path.join(output_dir, f"{video_basename}.mp4")
        tqdm.write(f"Results directory: {os.path.abspath(output_dir)} (run {run_number})")

    use_compile = cfg.get('inference', {}).get('compile', False)
    model = load_model(model_path, DEVICE, use_compile=use_compile)

    deadzones = []
    if args.deadzone:
        _cap = cv2.VideoCapture(cfg['input'])
        _ret, _first = _cap.read()
        _cap.release()
        if _ret and _first is not None:
            print("Deadzone setup: left-click + drag to draw exclusion rectangles on the first frame.")
            deadzones = _draw_deadzones_interactive(_first)
            print(f"{len(deadzones)} deadzone(s) configured.")
        else:
            print("WARNING: Could not read first frame for deadzone setup.")

    process_video(
        input_video_path=cfg['input'],
        output_video_path=cfg['output'],
        model=model,
        cfg=cfg,
        device=DEVICE,
        disable_display=not args.display,
        grid_size=args.grid_size,
        max_pet_time=args.max_pet_time,
        use_neighbors=not args.no_neighbors,
        output_csv_path=args.csv,
        output_heatmap_path=args.heatmap,
        show_grid=args.show_grid,
        single_cell_mode=args.no_grid,
        deadzones=deadzones,
        show_deadzones=args.show_deadzones,
    )


if __name__ == "__main__":
    main()
