"""
Post Encroachment Time (PET) conflict zone analysis using RT-DETR + DeepSORT.

Detection is driven by config.yaml (same config as deepSORT_rtdetr.py), giving
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
import math
import os
from collections import defaultdict, deque

import yaml

import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import RTDETR

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MODEL_PATH = './pdx_finetuned_rtdetr.onnx'
DEFAULT_NON_PET_SECONDS = 10.0
ANCHOR_FOOTPRINT_WIDTH_FRACTION = 0.40
ANCHOR_FOOTPRINT_HEIGHT_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Model / config loading
# ---------------------------------------------------------------------------

def load_model(model_path, device, use_compile=False):
    """Load an RT-DETR model checkpoint (.pt or .onnx)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Loading RT-DETR model: {model_path}")
    model = RTDETR(model_path)
    is_pt = model_path.endswith(".pt")
    if is_pt:
        model.to(device)
        if use_compile and device == 'cuda':
            try:
                model.model = torch.compile(model.model)
                print("torch.compile enabled (first inference will be slower while compiling).")
            except Exception as e:
                print(f"torch.compile unavailable, skipping: {e}")
    else:
        print(f"Non-PyTorch model ({model_path.rsplit('.', 1)[-1]}): device will be set per inference call.")
    print(f"Model loaded on device: {device}")
    return model


def load_config(config_path):
    """Load YAML config file and return as a dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Detection helpers  (identical to deepSORT_rtdetr.py)
# ---------------------------------------------------------------------------

def _run_detector(model, image, conf_threshold, iou_threshold, imgsz=None, half=False, device=None):
    """Run one Ultralytics RT-DETR detector pass and return raw results."""
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
# IoU / NMS helpers  (identical to deepSORT_rtdetr.py)
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
# Multi-pass detection helpers  (identical to deepSORT_rtdetr.py)
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


def _draw_polygon_interactive(frame, window_name="Draw Conflict Zone"):
    """
    Interactive polygon editor for selecting a PET conflict zone.

    Controls
    --------
    Left-click          : add polygon point
    U                   : undo last point
    C                   : clear polygon
    Enter / Space       : confirm polygon with at least 3 points
    Esc                 : skip (return None)
    """
    points = []
    mouse_pos = None

    def _mouse(event, x, y, flags, param):
        nonlocal mouse_pos
        mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            mouse_pos = (x, y)

    instructions = [
        "Draw conflict zone: left-click points to build polygon",
        "U: undo   C: clear   Enter/Space: confirm   Esc: skip",
    ]

    try:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, _mouse)

        while True:
            img = frame.copy()
            if points:
                pts = np.array(points, dtype=np.int32)
                for idx, pt in enumerate(points):
                    cv2.circle(img, pt, 4, (0, 255, 255), -1)
                    cv2.putText(img, str(idx + 1), (pt[0] + 6, pt[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                if len(points) >= 2:
                    cv2.polylines(img, [pts], False, (0, 255, 255), 2)
                if len(points) >= 3:
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [pts], (0, 180, 180))
                    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
                    cv2.polylines(img, [pts], True, (0, 255, 255), 2)
                elif len(points) >= 1 and mouse_pos is not None:
                    cv2.line(img, points[-1], mouse_pos, (0, 220, 220), 1)

            for i, txt in enumerate(instructions):
                y_pos = 22 + i * 24
                cv2.putText(img, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, img)

            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32) and len(points) >= 3:
                break
            if key == 27:
                points = []
                break
            if key == ord('u') and points:
                points.pop()
            if key == ord('c'):
                points = []

        cv2.destroyWindow(window_name)
    except (cv2.error, Exception) as e:
        print(f"Conflict zone drawing unavailable: {e}")
        points = []

    return points if len(points) >= 3 else None


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


def _cell_rect(cell, width, height, grid_rows, grid_cols):
    """Return the pixel rectangle for one grid cell."""
    r, c = cell
    cell_w = width / grid_cols
    cell_h = height / grid_rows
    return (
        float(c * cell_w),
        float(r * cell_h),
        float((c + 1) * cell_w),
        float((r + 1) * cell_h),
    )


def _point_in_polygon(point, polygon):
    """Return True when point is inside or on a polygon boundary."""
    x, y = float(point[0]), float(point[1])
    pts = [(float(px), float(py)) for px, py in polygon]
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if _on_segment((x1, y1), (x2, y2), (x, y), eps=1e-6):
            return True
        intersects = ((y1 > y) != (y2 > y))
        if intersects:
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x_cross >= x:
                inside = not inside
    return inside


def _rect_overlaps_polygon(rect, polygon):
    """Inclusive rectangle/polygon overlap test."""
    x1, y1, x2, y2 = rect
    rect_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if any(_point_in_polygon(pt, polygon) for pt in rect_pts):
        return True
    if any(x1 <= px <= x2 and y1 <= py <= y2 for px, py in polygon):
        return True
    rect_edges = list(zip(rect_pts, rect_pts[1:] + rect_pts[:1]))
    poly_edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    for a1, a2 in rect_edges:
        for b1, b2 in poly_edges:
            if _segment_intersection_point(a1, a2, b1, b2) is not None:
                return True
    return False


def _conflict_zone_cells_from_polygon(polygon, width, height, grid_rows, grid_cols):
    """Return grid cells whose rectangles overlap the conflict-zone polygon."""
    if not polygon or len(polygon) < 3:
        return None
    cells = set()
    for r in range(grid_rows):
        for c in range(grid_cols):
            if _rect_overlaps_polygon(_cell_rect((r, c), width, height, grid_rows, grid_cols), polygon):
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


def _pet_bin_label(pet_seconds):
    """Map a PET value in seconds to the requested reporting bin."""
    if pet_seconds is None:
        return None
    if pet_seconds < 1.5:
        return "0-1.5s"
    if pet_seconds < 3.0:
        return "1.5-3s"
    if pet_seconds < 5.0:
        return "3-5s"
    return "5+s"


def _pet_bin_risk(pet_seconds):
    """Map PET seconds to a discrete risk level for binned step plots."""
    if pet_seconds is None:
        return None
    if pet_seconds < 1.5:
        return 1.0
    if pet_seconds < 3.0:
        return 0.75
    if pet_seconds < 5.0:
        return 0.5
    return 0.25


def _pet_bin_plot_level(pet_seconds):
    """Map PET seconds to a discrete plot level; 0 is reserved for no PET."""
    if pet_seconds is None:
        return None
    if pet_seconds < 1.5:
        return 4
    if pet_seconds < 3.0:
        return 3
    if pet_seconds < 5.0:
        return 2
    return 1


def _track_anchor_point(tlbr):
    """Use bottom-center as the ground-contact proxy for trajectory and speed."""
    x1, y1, x2, y2 = tlbr
    return (float((x1 + x2) / 2.0), float(y2))


def _track_anchor_footprint(
    tlbr,
    width_fraction=ANCHOR_FOOTPRINT_WIDTH_FRACTION,
    height_fraction=ANCHOR_FOOTPRINT_HEIGHT_FRACTION,
):
    """Return a bbox-clipped rectangle around the track anchor."""
    x1, y1, x2, y2 = map(float, tlbr)
    anchor_x, anchor_y = _track_anchor_point((x1, y1, x2, y2))
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    half_w = box_w * max(0.0, float(width_fraction)) / 2.0
    height = box_h * max(0.0, float(height_fraction))
    return (
        max(x1, anchor_x - half_w),
        max(y1, anchor_y - height),
        min(x2, anchor_x + half_w),
        min(y2, anchor_y),
    )


def _append_point_if_new(points, frame_idx, point):
    """Append a visit/history point only when it materially changes the path."""
    point = (float(point[0]), float(point[1]))
    if not points or points[-1][0] != frame_idx or points[-1][1] != point:
        points.append((int(frame_idx), point))


def _append_rect_if_new(rects, frame_idx, rect):
    """Append a frame-indexed rectangle only when it materially changes."""
    rect = tuple(float(v) for v in rect)
    if not rects or rects[-1][0] != frame_idx or rects[-1][1] != rect:
        rects.append((int(frame_idx), rect))


def _prune_points(points, min_frame):
    """Drop trajectory points older than `min_frame` while keeping the newest sample."""
    while len(points) > 1 and points[0][0] < min_frame:
        points.popleft()


def _region_cells_for_primary(cell, grid_rows, grid_cols, use_neighbors):
    """Return the region cells considered part of a PET region for a primary cell."""
    if use_neighbors:
        return _neighbor_cells(cell[0], cell[1], grid_rows, grid_cols, include_self=True)
    return {cell}


def _region_rect_from_cells(cells, width, height, grid_rows, grid_cols):
    """Return the pixel-space rectangle that bounds a region made of grid cells."""
    cell_w = width / grid_cols
    cell_h = height / grid_rows
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    x1 = min(cols) * cell_w
    y1 = min(rows) * cell_h
    x2 = (max(cols) + 1) * cell_w
    y2 = (max(rows) + 1) * cell_h
    return (float(x1), float(y1), float(x2), float(y2))


def _pet_activation_display_cells(primary_cell, region_cells, conflict_zone_cells=None):
    """Return grid cells to highlight for a PET activation."""
    if conflict_zone_cells is None:
        return {primary_cell}
    return set(region_cells)


def _point_in_rect(point, rect):
    """Return True if a point lies inside a rectangle."""
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _orientation(a, b, c):
    """Orientation helper for segment intersection."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p, eps=1e-6):
    """Return True if `p` lies on segment a-b."""
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segment_intersection_point(a1, a2, b1, b2, eps=1e-6):
    """Return the intersection point of two segments, or None if they do not intersect."""
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < eps:
        for candidate in (a1, a2, b1, b2):
            if _on_segment(a1, a2, candidate, eps=eps) and _on_segment(b1, b2, candidate, eps=eps):
                return (float(candidate[0]), float(candidate[1]))
        return None

    det_a = x1 * y2 - y1 * x2
    det_b = x3 * y4 - y3 * x4
    px = (det_a * (x3 - x4) - (x1 - x2) * det_b) / denom
    py = (det_a * (y3 - y4) - (y1 - y2) * det_b) / denom
    point = (float(px), float(py))
    if _on_segment(a1, a2, point, eps=eps) and _on_segment(b1, b2, point, eps=eps):
        return point
    return None


def _rect_intersection(rect_a, rect_b, eps=1e-6):
    """Return the overlapping rectangle for two rects, or None."""
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    x1 = max(float(ax1), float(bx1))
    y1 = max(float(ay1), float(by1))
    x2 = min(float(ax2), float(bx2))
    y2 = min(float(ay2), float(by2))
    if x2 + eps < x1 or y2 + eps < y1:
        return None
    return (x1, y1, x2, y2)


def _rect_center(rect):
    """Return the center point of a rectangle."""
    x1, y1, x2, y2 = rect
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def _segment_intersects_rect(segment, rect):
    """Return an intersection point when a segment touches a rectangle."""
    p1, p2 = segment
    if _point_in_rect(p1, rect):
        return (float(p1[0]), float(p1[1]))
    if _point_in_rect(p2, rect):
        return (float(p2[0]), float(p2[1]))
    x1, y1, x2, y2 = rect
    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for edge_a, edge_b in edges:
        point = _segment_intersection_point(p1, p2, edge_a, edge_b)
        if point is not None:
            return point
    return None


def _segments_from_points(points):
    """Build consecutive polyline segments from `(frame, point)` samples."""
    out = []
    for idx in range(len(points) - 1):
        p1 = points[idx][1]
        p2 = points[idx + 1][1]
        if p1 != p2:
            out.append((p1, p2))
    return out


def _polyline_intersects_in_rect(points_a, points_b, rect, footprints_a=None, footprints_b=None):
    """Return `(True, point)` when two trajectories intersect inside the region rectangle."""
    segs_a = _segments_from_points(points_a)
    segs_b = _segments_from_points(points_b)
    for a1, a2 in segs_a:
        for b1, b2 in segs_b:
            pt = _segment_intersection_point(a1, a2, b1, b2)
            if pt is not None and _point_in_rect(pt, rect):
                return True, pt
    if footprints_a and footprints_b:
        rect_segments_a = [
            ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
            for p1, p2 in segs_a
        ]
        rect_segments_b = [
            ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
            for p1, p2 in segs_b
        ]
        for _, footprint_a in footprints_a:
            footprint_a_in_region = _rect_intersection(footprint_a, rect)
            if footprint_a_in_region is None:
                continue
            for segment_b in rect_segments_b:
                point = _segment_intersects_rect(segment_b, footprint_a_in_region)
                if point is not None:
                    return True, point
            for _, footprint_b in footprints_b:
                overlap = _rect_intersection(footprint_a_in_region, footprint_b)
                if overlap is not None:
                    overlap_in_region = _rect_intersection(overlap, rect)
                    if overlap_in_region is not None:
                        return True, _rect_center(overlap_in_region)
        for _, footprint_b in footprints_b:
            footprint_b_in_region = _rect_intersection(footprint_b, rect)
            if footprint_b_in_region is None:
                continue
            for segment_a in rect_segments_a:
                point = _segment_intersects_rect(segment_a, footprint_b_in_region)
                if point is not None:
                    return True, point
    return False, None


def _heading_vector_from_points(points):
    """Estimate a movement vector from the oldest/newest distinct points."""
    if len(points) < 2:
        return None
    start = points[0][1]
    end = points[-1][1]
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    return (dx, dy)


def _recent_heading_vector_from_points(points):
    """Estimate frame-to-frame movement from the newest distinct point pair."""
    if len(points) < 2:
        return None
    newest = points[-1]
    for older in reversed(list(points)[:-1]):
        frame_delta = newest[0] - older[0]
        if frame_delta <= 0:
            continue
        dx = float(newest[1][0] - older[1][0])
        dy = float(newest[1][1] - older[1][1])
        norm = math.hypot(dx, dy)
        if norm >= 1e-6:
            return (dx, dy)
    return None


def _buffered_heading_vector_from_points(points, frame_window=None):
    """Estimate movement over the newest buffered span of trajectory points."""
    points_list = list(points)
    if len(points_list) < 2:
        return None
    newest_frame, newest_point = points_list[-1]
    min_frame = None if frame_window is None else newest_frame - max(1, int(frame_window))
    candidates = points_list[:-1]
    if min_frame is not None:
        candidates = [p for p in candidates if p[0] >= min_frame]
    if not candidates:
        candidates = points_list[:-1]
    for older_frame, older_point in candidates:
        frame_delta = newest_frame - older_frame
        if frame_delta <= 0:
            continue
        dx = float(newest_point[0] - older_point[0])
        dy = float(newest_point[1] - older_point[1])
        if math.hypot(dx, dy) >= 1e-6:
            return (dx, dy)
    return _recent_heading_vector_from_points(points_list)


def _angle_delta_degrees(vec_a, vec_b):
    """Return the unsigned angle between two vectors in degrees."""
    if vec_a is None or vec_b is None:
        return None
    ax, ay = vec_a
    bx, by = vec_b
    norm_a = math.hypot(ax, ay)
    norm_b = math.hypot(bx, by)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return None
    cosine = (ax * bx + ay * by) / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    return float(math.degrees(math.acos(cosine)))


def _is_opposing_direction(angle_delta_degrees, tolerance_degrees):
    """Return True when headings are opposite within the configured tolerance."""
    if angle_delta_degrees is None:
        return False
    return abs(180.0 - angle_delta_degrees) <= tolerance_degrees


def _is_collinear_direction(angle_delta_degrees, tolerance_degrees):
    """Return True for same- or opposite-direction headings within tolerance."""
    if angle_delta_degrees is None:
        return True
    return (
        angle_delta_degrees <= tolerance_degrees
        or abs(180.0 - angle_delta_degrees) <= tolerance_degrees
    )


def _opposing_motion_relation(points_a, points_b, vec_a, vec_b):
    """Classify opposite-direction motion as toward, away, or ambiguous."""
    if len(points_a) < 1 or len(points_b) < 1 or vec_a is None or vec_b is None:
        return "unknown"
    ax, ay = points_a[-1][1]
    bx, by = points_b[-1][1]
    rel_ab = (float(bx - ax), float(by - ay))
    rel_len = math.hypot(rel_ab[0], rel_ab[1])
    if rel_len < 1e-6:
        return "unknown"
    a_toward_b = vec_a[0] * rel_ab[0] + vec_a[1] * rel_ab[1]
    b_toward_a = vec_b[0] * -rel_ab[0] + vec_b[1] * -rel_ab[1]
    if a_toward_b > 0 and b_toward_a > 0:
        return "toward"
    if a_toward_b < 0 and b_toward_a < 0:
        return "away"
    return "opposing"


def _speed_to_unit(speed_ft_per_sec, speed_unit):
    """Convert speed from ft/s to the requested display unit."""
    if speed_ft_per_sec is None:
        return None
    if speed_unit == "mph":
        return float(speed_ft_per_sec) * 0.681818
    return float(speed_ft_per_sec)


def _speed_unit_suffix(speed_unit):
    """Human-readable speed unit suffix for overlays and CSV."""
    return "mph" if speed_unit == "mph" else "ft/s"



def _smooth_speed_ft_per_sec(raw_speed_ft_per_sec, previous_speed_ft_per_sec, alpha):
    """Optionally smooth speed with EMA; alpha=0 disables EMA."""
    if raw_speed_ft_per_sec is None:
        return previous_speed_ft_per_sec
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0.0 or previous_speed_ft_per_sec is None:
        return float(raw_speed_ft_per_sec)
    return alpha * float(raw_speed_ft_per_sec) + (1.0 - alpha) * float(previous_speed_ft_per_sec)


def _estimate_speed_ft_per_sec(points, fps):
    """Estimate speed from the last two trajectory points."""
    if fps <= 0 or len(points) < 2:
        return None
    (frame_a, point_a), (frame_b, point_b) = points[-2], points[-1]
    frame_delta = frame_b - frame_a
    if frame_delta <= 0:
        return None
    dist_pixels = math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
    return (dist_pixels / frame_delta) * fps


def _transform_point_homography(point, H):
    """Map a single image-space point through a homography matrix."""
    if point is None or H is None:
        return None
    try:
        x, y = float(point[0]), float(point[1])
        H_arr = np.asarray(H, dtype=np.float64)
        vec = H_arr @ np.array([x, y, 1.0], dtype=np.float64)
        if abs(vec[2]) < 1e-9:
            return None
        return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))
    except Exception:
        return None


def _point_distance(a, b):
    """Return Euclidean distance between two points, or None for invalid points."""
    if a is None or b is None:
        return None
    return float(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))



def _coerce_point(point):
    """Return a numeric 2D point tuple, or None when malformed."""
    if point is None:
        return None
    try:
        if len(point) != 2:
            return None
        return (float(point[0]), float(point[1]))
    except (TypeError, ValueError):
        return None


def _line_length_pixels(line):
    start = _coerce_point(line.get("start")) if line else None
    end = _coerce_point(line.get("end")) if line else None
    return _point_distance(start, end)


def _normalize_speed_lines(calibration, known_length_ft=52.0):
    """
    Normalize either the legacy single-line calibration or the new config block.

    Endpoint order for four-point calibration is:
    primary.start, primary.end, secondary.start, secondary.end.
    """
    calibration = calibration or {}
    speed_lines = calibration.get("speed_lines", calibration)

    primary = speed_lines.get("primary")
    if primary is None and calibration.get("start") and calibration.get("end"):
        primary = {
            "start": calibration.get("start"),
            "end": calibration.get("end"),
            "length_ft": calibration.get("length_ft", known_length_ft),
            "length_pixels": calibration.get("length_pixels"),
            "feet_per_pixel": calibration.get("feet_per_pixel"),
        }

    if primary:
        primary = dict(primary)
        primary.setdefault("length_ft", known_length_ft)
        length_pixels = primary.get("length_pixels") or _line_length_pixels(primary)
        primary["length_pixels"] = float(length_pixels) if length_pixels else None
        if primary["length_pixels"]:
            primary["feet_per_pixel"] = float(primary.get("length_ft", known_length_ft)) / primary["length_pixels"]

    secondary = speed_lines.get("secondary")
    if secondary:
        secondary = dict(secondary)
        secondary.setdefault("length_ft", 72.0)
        length_pixels = secondary.get("length_pixels") or _line_length_pixels(secondary)
        secondary["length_pixels"] = float(length_pixels) if length_pixels else None
        if secondary["length_pixels"]:
            secondary["feet_per_pixel"] = float(secondary.get("length_ft", 72.0)) / secondary["length_pixels"]

    enabled_for_speed = speed_lines.get("enabled_for_speed")
    if enabled_for_speed is None:
        enabled_for_speed = bool(secondary)

    return {
        "primary": primary,
        "secondary": secondary,
        "enabled_for_speed": bool(enabled_for_speed),
        "enabled_for_warp": bool(speed_lines.get("enabled_for_warp", False)),
        "world_points_ft": speed_lines.get("world_points_ft"),
    }


def _speed_line_endpoints(speed_lines):
    primary = speed_lines.get("primary") or {}
    secondary = speed_lines.get("secondary") or {}
    points = [
        _coerce_point(primary.get("start")),
        _coerce_point(primary.get("end")),
        _coerce_point(secondary.get("start")),
        _coerce_point(secondary.get("end")),
    ]
    return points if all(p is not None for p in points) else None


def _build_original_to_processing_transform(frame_width, frame_height, downscale_width, downscale_height):
    """Map original-frame coordinates into the resized inference frame."""
    if (
        downscale_width > 0
        and downscale_height > 0
        and (frame_width > downscale_width or frame_height > downscale_height)
    ):
        sx = float(downscale_width) / float(frame_width)
        sy = float(downscale_height) / float(frame_height)
        return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64)


def _build_two_line_metric_transform(speed_lines, base_transform=None):
    """
    Build an affine metric transform from two measured image directions.

    The returned matrix maps points into a local feet coordinate system. This is
    intentionally conservative: it improves directional scale without replacing
    a true projective ground-plane calibration.
    """
    endpoints = _speed_line_endpoints(speed_lines)
    if not endpoints:
        return None, "secondary speed line is incomplete."

    primary = speed_lines.get("primary") or {}
    secondary = speed_lines.get("secondary") or {}
    primary_length_ft = float(primary.get("length_ft", 52.0))
    secondary_length_ft = float(secondary.get("length_ft", 72.0))
    if primary_length_ft <= 0 or secondary_length_ft <= 0:
        return None, "speed line lengths must be positive."

    transformed = []
    for point in endpoints:
        mapped = _transform_point_homography(point, base_transform) if base_transform is not None else point
        if mapped is None:
            return None, "could not transform speed line endpoints."
        transformed.append(mapped)

    primary_start, primary_end, secondary_start, secondary_end = transformed
    v_primary = np.array(
        [primary_end[0] - primary_start[0], primary_end[1] - primary_start[1]],
        dtype=np.float64,
    )
    v_secondary = np.array(
        [secondary_end[0] - secondary_start[0], secondary_end[1] - secondary_start[1]],
        dtype=np.float64,
    )
    basis = np.column_stack([v_primary, v_secondary])
    if abs(float(np.linalg.det(basis))) < 1e-6:
        return None, "speed lines are parallel or too close to parallel."

    scale = np.diag([primary_length_ft, secondary_length_ft])
    metric_2x2 = scale @ np.linalg.inv(basis)
    origin = np.array(primary_start, dtype=np.float64)
    offset = -metric_2x2 @ origin
    metric_H = np.array(
        [
            [metric_2x2[0, 0], metric_2x2[0, 1], offset[0]],
            [metric_2x2[1, 0], metric_2x2[1, 1], offset[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if base_transform is not None:
        metric_H = metric_H @ np.asarray(base_transform, dtype=np.float64)
    return metric_H, None


def _world_points_to_dst_points(world_points_ft, dst_size):
    """Map arbitrary world-foot coordinates into the configured warp output rectangle."""
    points = [_coerce_point(p) for p in (world_points_ft or [])]
    if len(points) != 4 or any(p is None for p in points):
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if abs(max_x - min_x) < 1e-6 or abs(max_y - min_y) < 1e-6:
        return None
    dw, dh = dst_size
    return [
        (
            (x - min_x) / (max_x - min_x) * float(dw - 1),
            (y - min_y) / (max_y - min_y) * float(dh - 1),
        )
        for x, y in points
    ]


def _build_line_warp_homography(speed_lines, dst_size, image_to_processing_H=None):
    """
    Build a detection-warp homography from the two speed-line endpoint pairs.

    Requires four source endpoints and four matching world_points_ft values.
    """
    endpoints = _speed_line_endpoints(speed_lines)
    if not endpoints:
        return None, None, "line warp requested but primary/secondary endpoints are incomplete."
    if not dst_size or len(dst_size) != 2:
        return None, None, "line warp requested but dst_size is missing."

    src_points = []
    for point in endpoints:
        mapped = _transform_point_homography(point, image_to_processing_H) if image_to_processing_H is not None else point
        if mapped is None:
            return None, None, "line warp could not map endpoints into processing coordinates."
        src_points.append(mapped)

    dst_points = _world_points_to_dst_points(speed_lines.get("world_points_ft"), dst_size)
    if dst_points is None:
        return None, None, "line warp requested but four valid world_points_ft are required."

    try:
        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        H_inv = cv2.getPerspectiveTransform(dst, src)
        return H, H_inv, None
    except Exception as exc:
        return None, None, f"line warp calibration failed: {exc}"


def _build_speed_context(calibration, requested_speed_space, warp_H, known_length_ft=52.0):
    """
    Resolve speed estimation space and scale.

    Returns (context, warning). context is None when speed should be disabled.
    """
    if not calibration:
        return None, "speed tracking enabled but calibration unavailable. Continuing without speed output."

    requested = (requested_speed_space or "auto").lower()
    if requested not in {"auto", "warp", "image"}:
        requested = "auto"

    speed_lines = _normalize_speed_lines(calibration, known_length_ft=known_length_ft)
    primary = speed_lines.get("primary") or {}

    if requested != "image" and speed_lines.get("secondary") and speed_lines.get("enabled_for_speed"):
        base_transform = None
        base_label = "image"
        if requested == "warp":
            if warp_H is None:
                return None, "speed-space=warp requested but homography is unavailable. Continuing without speed output."
            base_transform = warp_H
            base_label = "warp"
        elif requested == "auto" and warp_H is not None:
            base_transform = warp_H
            base_label = "warp"

        metric_H, metric_warning = _build_two_line_metric_transform(speed_lines, base_transform)
        if metric_H is not None:
            return {
                "space": f"ground_plane_{base_label}",
                "feet_per_pixel": None,
                "speed_scale": 1.0,
                "calibration_length_pixels": float(primary.get("length_pixels") or 0.0),
                "transform": metric_H,
                "calibration": speed_lines,
            }, None
        if requested == "warp":
            return None, f"{metric_warning} Continuing without speed output."
        print(f"WARNING: {metric_warning} Falling back to single-line speed calibration.")

    image_length = primary.get("length_pixels") or calibration.get("length_pixels")
    image_fpp = primary.get("feet_per_pixel") or calibration.get("feet_per_pixel")
    if not image_length or image_length < 1e-6 or not image_fpp:
        return None, "speed calibration is invalid. Continuing without speed output."

    def _image_context():
        return {
            "space": "image",
            "feet_per_pixel": float(image_fpp),
            "speed_scale": float(image_fpp),
            "calibration_length_pixels": float(image_length),
            "transform": None,
        }

    if requested == "image":
        return _image_context(), None

    if warp_H is None:
        if requested == "warp":
            return None, "speed-space=warp requested but homography is unavailable. Continuing without speed output."
        return _image_context(), "speed-space=auto could not find a valid homography; falling back to image-space speed."

    start_warp = _transform_point_homography(primary.get("start") or calibration.get("start"), warp_H)
    end_warp = _transform_point_homography(primary.get("end") or calibration.get("end"), warp_H)
    warp_length = _point_distance(start_warp, end_warp)
    if warp_length is None or warp_length < 1e-6:
        if requested == "warp":
            return None, "speed-space=warp could not transform the calibration line. Continuing without speed output."
        return _image_context(), "speed-space=auto could not transform the calibration line; falling back to image-space speed."

    return {
        "space": "warp",
        "feet_per_pixel": float(known_length_ft) / float(warp_length),
        "speed_scale": float(known_length_ft) / float(warp_length),
        "calibration_length_pixels": float(warp_length),
        "transform": warp_H,
        "calibration_start": start_warp,
        "calibration_end": end_warp,
    }, None

def _format_speed_label(speed_value, speed_unit):
    """Format a speed label for overlays."""
    if speed_value is None:
        return None
    decimals = 1 if speed_unit == "mph" else 2
    return f"{speed_value:.{decimals}f} { _speed_unit_suffix(speed_unit) }"



def _pet_speed_event_fields(name_a, speed_a_ft, name_b, speed_b_ft, speed_unit):
    """Build explicit per-actor speed fields and speed deltas for PET rows."""
    speed_key = speed_unit.replace("/", "_")
    speeds_by_name = {
        str(name_a).lower(): speed_a_ft,
        str(name_b).lower(): speed_b_ft,
    }
    cyclist_speed_ft = speeds_by_name.get("cyclist")
    pedestrian_speed_ft = speeds_by_name.get("pedestrian")
    speed_diff_ft = None
    if cyclist_speed_ft is not None and pedestrian_speed_ft is not None:
        speed_diff_ft = abs(float(cyclist_speed_ft) - float(pedestrian_speed_ft))

    return {
        "cyclist_speed_ft_per_s": round(cyclist_speed_ft, 3) if cyclist_speed_ft is not None else None,
        "pedestrian_speed_ft_per_s": round(pedestrian_speed_ft, 3) if pedestrian_speed_ft is not None else None,
        "speed_difference_ft_per_s": round(speed_diff_ft, 3) if speed_diff_ft is not None else None,
        f"cyclist_speed_{speed_key}": round(_speed_to_unit(cyclist_speed_ft, speed_unit), 3) if cyclist_speed_ft is not None else None,
        f"pedestrian_speed_{speed_key}": round(_speed_to_unit(pedestrian_speed_ft, speed_unit), 3) if pedestrian_speed_ft is not None else None,
        f"speed_difference_{speed_key}": round(_speed_to_unit(speed_diff_ft, speed_unit), 3) if speed_diff_ft is not None else None,
    }


def _draw_direction_arrow(frame, tlbr, vector, color):
    """Draw a short direction arrow from the bbox center."""
    if vector is None:
        return
    x1, y1, x2, y2 = map(float, tlbr)
    dx, dy = vector
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    box_diag = math.hypot(x2 - x1, y2 - y1)
    arrow_len = max(18.0, min(60.0, box_diag * 0.35))
    end = (
        int(round(cx + (dx / norm) * arrow_len)),
        int(round(cy + (dy / norm) * arrow_len)),
    )
    cv2.arrowedLine(frame, (cx, cy), end, color, 2, tipLength=0.35)


def _draw_measure_line_interactive(frame, window_name="Draw 52 ft calibration line", length_ft=52.0, label=None):
    """
    Prompt the user to draw a single calibration line.

    Controls
    --------
    Left-click + drag : draw/update the calibration line
    C                 : clear line
    Enter / Space     : confirm and continue
    Esc               : cancel speed calibration
    """
    state = {"start": None, "end": None, "drawing": False}
    instructions = [
        f"Draw the full {length_ft:g} ft {label or 'calibration'} line: left-click + drag",
        "C: clear   Enter/Space: confirm   Esc: cancel",
    ]

    def _mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["end"] = (x, y)
            state["drawing"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["end"] = (x, y)
            state["drawing"] = False

    try:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, _mouse)
        while True:
            img = frame.copy()
            if state["start"] is not None and state["end"] is not None:
                cv2.line(img, state["start"], state["end"], (0, 255, 255), 2)
            for idx, txt in enumerate(instructions):
                y_pos = 22 + idx * 24
                cv2.putText(img, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, img)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32) and state["start"] is not None and state["end"] is not None:
                break
            if key == 27:
                state["start"] = None
                state["end"] = None
                break
            if key == ord('c'):
                state["start"] = None
                state["end"] = None
        cv2.destroyWindow(window_name)
    except (cv2.error, Exception) as e:
        print(f"Speed calibration drawing unavailable: {e}")
        state["start"] = None
        state["end"] = None

    if state["start"] is None or state["end"] is None:
        return None
    length_pixels = math.hypot(
        state["end"][0] - state["start"][0],
        state["end"][1] - state["start"][1],
    )
    if length_pixels < 1e-6:
        return None
    return {
        "start": state["start"],
        "end": state["end"],
        "length_pixels": float(length_pixels),
        "length_ft": float(length_ft),
        "feet_per_pixel": float(length_ft) / float(length_pixels),
    }


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

    Detection uses the same multi-pass pipeline as deepSORT_rtdetr.py (driven by
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

    # True only when --no-grid AND user picked a cell. When --no-grid but selection fails,
    # we fall back to full-grid PET; `_sc_active` is False so neighbor aggregation and PET
    # math match normal full-grid mode (class-separated timer/HUD stay off).
    _sc_active = bool(single_cell_mode and selected_cell is not None)

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
    # grid[(r,c)] = {'class_id': [(frame, track_id), ...], ...}
    grid_occupancy = defaultdict(lambda: defaultdict(list))

    conflict_events           = []
    cell_pet_values           = defaultdict(list)
    frame_to_pets             = defaultdict(list)
    conflict_cells_prev_frame = set()

    # --- Single-cell PET timer state (only read/written when _sc_active) ---
    _sc_timer_state         = 'idle'  # 'idle' | 'running' | 'locked'
    _sc_timer_start_frame   = None    # frame_count baseline for the conflict-sequence (independent of per-class entry timers)
    _sc_timer_locked_secs   = None    # PET seconds captured at the moment of lock
    _sc_last_pet_secs       = None    # most recent pet_seconds computed for selected_cell
    _sc_exit_frame          = None    # frame when the cell last became empty (gap-phase tracking)
    _sc_lock_hold_frames    = max(1, int(fps * 3))  # hold red "locked" display for ~3 s
    _sc_lock_hold_remaining = 0
    _sc_prev_classes_in_cell = set()  # for detecting per-class entry events
    _sc_last_entry_frame_by_class = {}  # cid -> frame_count (independent per-class entry timestamps)

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

                # Post-tracker containment check (same as deepSORT_rtdetr.py)
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
            # We need exactly two classes: a "first actor" and a "second actor".
            # Sort class IDs so the pairing is deterministic regardless of config order.
            sorted_cids = sorted(class_ids)
            cid_a, cid_b = (sorted_cids[0], sorted_cids[1]) if len(sorted_cids) >= 2 else (None, None)

            conflict_cells_this_frame = set()

            if cid_a is not None and cid_b is not None:
                if _sc_active:
                    cells_to_check = {selected_cell} if selected_cell in grid_occupancy else set()
                else:
                    cells_to_check = set(grid_occupancy.keys())

                for (r, c) in list(cells_to_check):
                    if (r, c) in conflict_cells_this_frame:
                        continue

                    # Aggregate timestamps from this cell and optionally its 3x3 neighbors
                    list_a, list_b = [], []
                    if use_neighbors and not _sc_active:
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

                    # Find minimum-gap pair → standard PET
                    pet_frames        = None
                    signed_pet_frames = None
                    best_id_a = best_id_b = None
                    for (fa, ida) in list_a:
                        for (fb, idb) in list_b:
                            d = abs(fa - fb)
                            if pet_frames is None or d < pet_frames:
                                pet_frames        = d
                                signed_pet_frames = fa - fb  # positive → a exited before b entered
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
                    if _sc_active and (r, c) == selected_cell:
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
            if _sc_active:
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

                _entered_classes = _classes_in_cell_now.difference(_sc_prev_classes_in_cell)
                _sc_prev_classes_in_cell = set(_classes_in_cell_now)
                # Ensure per-class timers update for classes present at sequence start
                # and any newly-entered classes thereafter.
                if _sc_timer_state == 'idle' and _classes_in_cell_now:
                    for _cid0 in _classes_in_cell_now:
                        _sc_last_entry_frame_by_class[_cid0] = frame_count
                else:
                    for _ec in _entered_classes:
                        _sc_last_entry_frame_by_class[_ec] = frame_count

                if _sc_timer_state == 'idle':
                    if _classes_in_cell_now:
                        _sc_timer_state       = 'running'
                        # Start a new conflict sequence as soon as any class appears in the cell.
                        # Per-class "last entry" timers update independently below.
                        _sc_timer_start_frame  = frame_count
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
                        _sc_prev_classes_in_cell = set()
                        _sc_last_entry_frame_by_class = {}
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
                            _sc_prev_classes_in_cell = set()
                            _sc_last_entry_frame_by_class = {}
                elif _sc_timer_state == 'locked':
                    _sc_lock_hold_remaining -= 1
                    if _sc_lock_hold_remaining <= 0:
                        _sc_timer_state       = 'idle'
                        _sc_timer_start_frame = None
                        _sc_timer_locked_secs = None
                        _sc_last_pet_secs     = None
                        _sc_exit_frame        = None
                        _sc_prev_classes_in_cell = set()
                        _sc_last_entry_frame_by_class = {}

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

            # Faint grid lines
            if show_grid:
                for i in range(1, grid_rows):
                    y = int(i * cell_h)
                    cv2.line(annotated_frame, (0, y), (width, y), (60, 60, 60), 1)
                for j in range(1, grid_cols):
                    x = int(j * cell_w)
                    cv2.line(annotated_frame, (x, 0), (x, height), (60, 60, 60), 1)

            # Conflict zones (semi-transparent red)
            for (r, c) in conflict_cells_this_frame:
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Selected-cell PET timer overlay (only when --no-grid and a cell was selected)
            if _sc_active:
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

            # Bounding boxes per class (colors from config)
            for cid, tracks in confirmed_by_class.items():
                cls_cfg    = class_map[cid]
                color      = tuple(cls_cfg['color'])
                text_color = tuple(cls_cfg['text_color'])
                prefix     = cls_cfg['name'][0]  # 'C' for Cyclist, 'P' for Pedestrian
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

            # Conflict count overlay (full-grid mode only; no-grid stacks HUD in block below)
            if conflict_cells_this_frame and not _sc_active:
                cv2.putText(
                    annotated_frame,
                    f"CONFLICT ZONES: {len(conflict_cells_this_frame)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )

            # Single-cell HUD: top-left stack — conflict line, per-class entry times, then sequence/PET
            if _sc_active:
                _hud_y = 10
                if conflict_cells_this_frame:
                    _cz = f"CONFLICT ZONES: {len(conflict_cells_this_frame)}"
                    (_cw, _ch), _ = cv2.getTextSize(_cz, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated_frame, (8, _hud_y - _ch - 4),
                                  (14 + _cw, _hud_y + 4), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, _cz, (10, _hud_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    _hud_y += 32
                if _sc_last_entry_frame_by_class:
                    for _cid_k in sorted(_sc_last_entry_frame_by_class.keys()):
                        _last_f = _sc_last_entry_frame_by_class.get(_cid_k)
                        if _last_f is None:
                            continue
                        _dt = (frame_count - _last_f) / fps if fps > 0 else 0.0
                        _nm = class_map.get(_cid_k, {}).get('name', str(_cid_k))
                        _prefix = _nm[0].upper() if _nm else str(_cid_k)
                        _txt = f"{_prefix} ENTRY  {_dt:.2f}s ago"
                        (_w2, _h2), _ = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                        cv2.rectangle(annotated_frame, (8, _hud_y - _h2 - 4),
                                      (14 + _w2, _hud_y + 4), (0, 0, 0), -1)
                        cv2.putText(annotated_frame, _txt,
                                    (10, _hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
                        _hud_y += 24
                if _sc_timer_state != 'idle':
                    if _sc_timer_state == 'running':
                        if _sc_exit_frame is None:
                            _hud_elapsed = (frame_count - _sc_timer_start_frame) / fps if fps > 0 else 0.0
                            _hud_txt = f"TIMER  {_hud_elapsed:.2f}s"
                        else:
                            _hud_gap = (frame_count - _sc_exit_frame) / fps if fps > 0 else 0.0
                            _hud_txt = f"GAP  {_hud_gap:.2f}s"
                        _hud_color = (0, 210, 0)
                    else:
                        _hud_txt = f"PET LOCKED  {_sc_timer_locked_secs:.2f}s"
                        _hud_color = (40, 40, 220)
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
                    cv2.imshow('RT-DETR PET Analysis', annotated_frame)
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
        description="RT-DETR PET conflict zone detection with config-driven multi-pass detection."
    )
    # Config / model / IO (same pattern as deepSORT_rtdetr.py)
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config file.")
    parser.add_argument("--input",  "-i", default=None, help="Override input video path from config.")
    parser.add_argument("--output", "-o", default=None, help="Override output video path.")
    parser.add_argument("--model",  "-m", default=None, help="Override model path from config.")

    # PET-specific parameters
    parser.add_argument("--grid-size",    type=int,   default=100,  help="N for NxN conflict grid (default: 100).")
    parser.add_argument("--max-pet-time", type=int,   default=300,   help="Conflict window in frames (default: 10).")
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

    # Model fallback: try .onnx → .pt for RT-DETR
    if not os.path.exists(model_path):
        base = os.path.splitext(model_path)[0]
        for fallback in (base + ".onnx", base + ".pt"):
            if os.path.exists(fallback):
                print(f"Model not found at {model_path}, falling back to: {fallback}")
                model_path = fallback
                cfg['model'] = model_path
                break
        else:
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


def process_video(
    input_video_path,
    output_video_path,
    model,
    cfg,
    device='cpu',
    disable_display=True,
    grid_size=10,
    max_pet_time=30,
    use_neighbors=True,
    output_csv_path=None,
    output_heatmap_path=None,
    show_grid=False,
    single_cell_mode=False,
    deadzones=None,
    show_deadzones=False,
    show_indicators=False,
    conflict_zone_polygon=None,
    parallel_angle_tolerance=15.0,
    trajectory_history=15,
    track_speed=False,
    speed_unit="mph",
    hide_speed_overlay=False,
    speed_smoothing_window=5,
    speed_smoothing_alpha=0.0,
    speed_space="auto",
    line_warp=False,
    calibration=None,
    show_direction_lines=False,
):
    """Process video for trajectory-aware PET conflict analysis."""
    deadzones = deadzones or []
    cfg_calibration = cfg.get('calibration', {}) or {}
    calibration = calibration or cfg_calibration or {}
    if calibration and cfg_calibration.get("speed_lines") and not calibration.get("speed_lines"):
        merged_calibration = dict(cfg_calibration)
        merged_calibration.update(calibration)
        calibration = merged_calibration

    inf_cfg = cfg.get('inference', {})
    pass_cfg = cfg.get('passes', {})
    nms_cfg = cfg.get('nms', {})
    dbg_cfg = cfg.get('debug', {})
    classes = cfg.get('classes', [])

    confidence_threshold = inf_cfg.get('confidence', 0.65)
    iou_threshold = inf_cfg.get('iou', 0.7)
    half = inf_cfg.get('half', False)
    imgsz = inf_cfg.get('imgsz', 0) or None
    downscale_width = inf_cfg.get('downscale_width', 0)
    downscale_height = inf_cfg.get('downscale_height', 0)

    top_cfg = pass_cfg.get('top_region', {})
    top_region_pass = top_cfg.get('enabled', False)
    top_region_ratio = top_cfg.get('ratio', 0.45)
    top_region_imgsz = top_cfg.get('imgsz', 0) or None
    top_region_conf = top_cfg.get('confidence') or None

    sahi_cfg = pass_cfg.get('sahi', {})
    tile_mode = 'sahi' if sahi_cfg.get('enabled', False) else 'off'
    tile_size = sahi_cfg.get('tile_size', 480)
    tile_overlap = sahi_cfg.get('tile_overlap', 0.4)
    tile_interval = sahi_cfg.get('tile_interval', 1)
    tile_imgsz = sahi_cfg.get('imgsz', 0) or None
    tile_conf = sahi_cfg.get('confidence') or None
    tile_y_max = sahi_cfg.get('y_max_fraction', 1.0)
    tile_prescale = sahi_cfg.get('prescale', 1.0)

    warp_cfg = pass_cfg.get('warp', {})
    warp_enabled = warp_cfg.get('enabled', False)
    warp_H = warp_H_inv = warp_dst_size = None
    warp_conf = warp_cfg.get('confidence') or None
    warp_imgsz = warp_cfg.get('imgsz', 0) or None
    warp_src = warp_cfg.get('src_points')
    warp_dst_size = warp_cfg.get('dst_size')

    nms_iou = nms_cfg.get('hard_iou', 0.45)
    nms_containment = nms_cfg.get('containment_fraction', 0.8)
    crowd_mode = nms_cfg.get('crowd_mode', 'off')
    soft_nms_iou = nms_cfg.get('soft_nms_iou', 0.25)
    soft_nms_sigma = nms_cfg.get('soft_nms_sigma', 0.2)

    debug_detections = dbg_cfg.get('log_detections', False)
    class_ids = {c['id'] for c in classes}
    class_map = {c['id']: c for c in classes}
    min_confidences = {c['id']: c['min_confidence'] for c in classes if 'min_confidence' in c}

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {input_video_path}")

    grid_rows = grid_cols = grid_size
    selected_cell = None
    if single_cell_mode:
        ret_sel, frame_sel = cap.read()
        if ret_sel and frame_sel is not None:
            tqdm.write("Select a grid cell by clicking on the frame; press Enter/Space or q/ESC to confirm.")
            selected_cell = _select_grid_cell_for_pet(frame_sel, grid_rows, grid_cols)
            if selected_cell is not None:
                tqdm.write(f"Using single grid cell (row={selected_cell[0]}, col={selected_cell[1]}) for PET computation.")
            else:
                tqdm.write("No grid cell selected; falling back to full-grid PET computation.")
        else:
            tqdm.write("Could not read first frame for grid selection; falling back to full-grid PET computation.")
        cap.release()
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not re-open video file {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    original_to_processing_H = _build_original_to_processing_transform(
        width, height, downscale_width, downscale_height
    )
    speed_lines = _normalize_speed_lines(calibration)
    line_warp_requested = bool(line_warp or speed_lines.get("enabled_for_warp"))
    if line_warp_requested:
        warp_H, warp_H_inv, line_warp_warning = _build_line_warp_homography(
            speed_lines, warp_dst_size, original_to_processing_H
        )
        if line_warp_warning:
            print(f"WARNING: {line_warp_warning} Falling back to passes.warp.src_points.")
            warp_H = warp_H_inv = None
        elif warp_enabled:
            print("Warp pass enabled from calibration.speed_lines endpoints.")

    if warp_H is None and warp_src and warp_dst_size and len(warp_src) == 4:
        warp_H, warp_H_inv = _build_homography(warp_src, warp_dst_size)
        if warp_enabled:
            print(f"Warp pass enabled: {warp_src} -> {warp_dst_size}")
    elif warp_H is None and warp_enabled:
        print("WARNING: warp pass enabled but src_points/dst_size not configured. Warp pass disabled.")
        warp_enabled = False

    speed_warp_H = warp_H @ original_to_processing_H if warp_H is not None else None

    cell_w = width / grid_cols
    cell_h = height / grid_rows
    _sc_active = bool(single_cell_mode and selected_cell is not None)
    region_neighbors_enabled = bool(use_neighbors and not _sc_active)
    conflict_zone_cells = _conflict_zone_cells_from_polygon(
        conflict_zone_polygon, width, height, grid_rows, grid_cols
    ) if conflict_zone_polygon else None
    if conflict_zone_polygon and not conflict_zone_cells:
        print("WARNING: conflict zone did not overlap any grid cells. Falling back to full-grid PET.")
        conflict_zone_polygon = None
        conflict_zone_cells = None
    elif conflict_zone_cells:
        print(f"Conflict zone active: {len(conflict_zone_cells)} grid cell(s) included for PET.")
    if _sc_active and conflict_zone_cells is not None and selected_cell not in conflict_zone_cells:
        print("WARNING: selected single PET cell is outside the conflict zone. Disabling single-cell mode.")
        _sc_active = False
    history_frames = max(int(max_pet_time), int(trajectory_history))
    speed_context = None
    speed_feet_per_pixel = None
    active_speed_space = None

    if track_speed:
        speed_context, speed_warning = _build_speed_context(calibration, speed_space, speed_warp_H)
        if speed_warning:
            print(f"WARNING: {speed_warning}")
        if speed_context:
            active_speed_space = speed_context["space"]
            speed_feet_per_pixel = speed_context["feet_per_pixel"]
            if active_speed_space == "warp":
                print(
                    "Speed calibration active (warp): homography-corrected 52 ft line "
                    f"/ {speed_context['calibration_length_pixels']:.1f}px in rectified space"
                )
            elif str(active_speed_space).startswith("ground_plane"):
                print(f"Speed calibration active ({active_speed_space}): two-line metric ground-plane transform")
            else:
                print(
                    f"Speed calibration active ({active_speed_space}): "
                    f"52 ft / {speed_context['calibration_length_pixels']:.1f}px"
                )
        else:
            track_speed = False

    if DeepSort is None:
        raise RuntimeError("deep_sort_realtime is not installed. Install it with: pip install deep-sort-realtime")

    trackers = {}
    for cls_cfg in classes:
        cid = cls_cfg['id']
        t_cfg = cls_cfg.get('tracker', {})
        trackers[cid] = DeepSort(
            max_age=t_cfg.get('max_age', 15),
            max_iou_distance=t_cfg.get('max_iou_distance', 0.5),
            n_init=t_cfg.get('n_init', 3),
            nms_max_overlap=t_cfg.get('nms_max_overlap', 0.3),
            max_cosine_distance=t_cfg.get('max_cosine_distance', 0.2),
            embedder=t_cfg.get('embedder', 'mobilenet'),
        )

    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
    ]
    out = None
    for _, fourcc in codecs_to_try:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if out.isOpened():
            break
        out.release()
        out = None
    if out is None:
        raise RuntimeError("Could not create VideoWriter with any tried codec.")

    track_histories = defaultdict(deque)
    track_speed_histories = defaultdict(lambda: deque(maxlen=max(1, int(speed_smoothing_window))))
    track_speeds_ft = {}
    speed_smoothing_alpha = max(0.0, min(1.0, float(speed_smoothing_alpha)))
    active_region_visits = defaultdict(lambda: defaultdict(dict))
    closed_region_visits = defaultdict(lambda: defaultdict(list))
    event_signatures = set()
    conflict_events = []
    cell_pet_values = defaultdict(list)
    frame_to_pets = defaultdict(list)
    frame_to_pet_bin_levels = defaultdict(list)
    frame_to_pet_bin_risks = defaultdict(list)
    display_available = not disable_display
    _cached_tile_dets = []

    _sc_timer_state = 'idle'
    _sc_timer_start_frame = None
    _sc_timer_locked_secs = None
    _sc_timer_locked_bin = None
    _sc_last_pet_secs = None
    _sc_last_pet_bin = None
    _sc_exit_frame = None
    _sc_lock_hold_frames = max(1, int(fps * 3))
    _sc_lock_hold_remaining = 0
    _sc_prev_classes_in_cell = set()
    _sc_last_entry_frame_by_class = {}
    _sc_last_event_frame = -1

    sorted_cids = sorted(class_ids)
    cid_a, cid_b = (sorted_cids[0], sorted_cids[1]) if len(sorted_cids) >= 2 else (None, None)
    pbar = tqdm(total=total_frames if total_frames > 0 else None, unit="frame", desc="PET analysis")
    frame_count = 0
    speed_point_histories = defaultdict(deque)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            if (downscale_width > 0 and downscale_height > 0
                    and (frame_w > downscale_width or frame_h > downscale_height)):
                proc_frame = cv2.resize(frame, (downscale_width, downscale_height))
                scale_x = frame_w / downscale_width
                scale_y = frame_h / downscale_height
            else:
                proc_frame = frame
                scale_x = scale_y = 1.0
            proc_h, proc_w = proc_frame.shape[:2]

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

            merged_dets = []
            for det in full_dets + top_dets + tile_dets + warp_dets:
                clipped = _clip_detection(det, frame_w=proc_w, frame_h=proc_h)
                if clipped is not None:
                    merged_dets.append(clipped)
            if scale_x != 1.0 or scale_y != 1.0:
                merged_dets = [
                    [d[0] * scale_x, d[1] * scale_y, d[2] * scale_x, d[3] * scale_y, d[4], d[5]]
                    for d in merged_dets
                ]
            merged_dets = _hard_nms_per_class(merged_dets, iou_threshold=nms_iou, containment_fraction=nms_containment)
            processed_dets = _apply_crowd_postprocess(
                merged_dets, crowd_mode=crowd_mode,
                soft_nms_iou=soft_nms_iou, soft_nms_sigma=soft_nms_sigma,
                score_threshold=confidence_threshold * 0.5,
            )
            if deadzones:
                processed_dets = [d for d in processed_dets if not _is_in_deadzone(d, deadzones)]

            if debug_detections and frame_count % 10 == 0:
                print(f"Frame {frame_count}: full={len(full_dets)} top={len(top_dets)} tile={len(tile_dets)} final={len(processed_dets)}")

            by_class = _split_detections_by_class(processed_dets, class_ids, min_confidences)
            confirmed_by_class = {}
            track_snapshots = defaultdict(dict)
            occupied_cells_all = set()

            for cid in class_ids:
                dets = by_class.get(cid, [])
                tracks = trackers[cid].update_tracks(dets, frame=frame)
                confirmed = [t for t in tracks if t.is_confirmed()]

                def _tlbr(track_obj):
                    return list(map(int, track_obj.to_tlbr()))

                def _tarea(track_obj):
                    box = _tlbr(track_obj)
                    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

                visible = [
                    t for t in confirmed
                    if not any(
                        _tarea(other) > _tarea(t)
                        and _is_contained(_tlbr(t), _tlbr(other), nms_containment)
                        for other in confirmed if other is not t
                    )
                ]
                confirmed_by_class[cid] = visible

                for track in visible:
                    track_id = track.track_id
                    tlbr = track.to_tlbr()
                    anchor = _track_anchor_point(tlbr)
                    anchor_footprint = _track_anchor_footprint(tlbr)
                    cells = _bbox_overlap_cells((tlbr[0], tlbr[1], tlbr[2], tlbr[3]), width, height, grid_rows, grid_cols)
                    occupied_cells_all.update(cells)
                    history = track_histories[(cid, track_id)]
                    _append_point_if_new(history, frame_count, anchor)
                    _prune_points(history, frame_count - history_frames)

                    speed_ft_per_sec = None
                    if track_speed and speed_context:
                        speed_point = anchor
                        if speed_context.get("transform") is not None:
                            speed_point = _transform_point_homography(anchor, speed_context.get("transform"))
                        speed_history = speed_point_histories[(cid, track_id)]
                        if speed_point is not None:
                            _append_point_if_new(speed_history, frame_count, speed_point)
                            _prune_points(speed_history, frame_count - history_frames)
                        pixel_speed = _estimate_speed_ft_per_sec(speed_history, fps)
                        if pixel_speed is not None:
                            speed_ft_per_sec = pixel_speed * float(speed_context.get("speed_scale", speed_feet_per_pixel or 1.0))
                            speed_ft_per_sec = _smooth_speed_ft_per_sec(
                                speed_ft_per_sec,
                                track_speeds_ft.get((cid, track_id)),
                                speed_smoothing_alpha,
                            )
                            track_speed_histories[(cid, track_id)].append(speed_ft_per_sec)
                        if track_speed_histories[(cid, track_id)]:
                            if speed_smoothing_alpha > 0.0:
                                speed_ft_per_sec = float(track_speed_histories[(cid, track_id)][-1])
                            else:
                                speed_ft_per_sec = float(np.mean(track_speed_histories[(cid, track_id)]))
                            track_speeds_ft[(cid, track_id)] = speed_ft_per_sec

                    track_snapshots[cid][track_id] = {
                        "track": track,
                        "tlbr": tlbr,
                        "point": anchor,
                        "footprint": anchor_footprint,
                        "cells": set(cells),
                        "speed_ft_per_sec": track_speeds_ft.get((cid, track_id)),
                    }

            cutoff_frame = frame_count - max_pet_time
            if _sc_active:
                cells_with_presence = {selected_cell}
            else:
                cells_with_presence = set(occupied_cells_all)
                if region_neighbors_enabled:
                    for occ_cell in list(occupied_cells_all):
                        cells_with_presence.update(_neighbor_cells(occ_cell[0], occ_cell[1], grid_rows, grid_cols, include_self=True))
            cells_to_update = set(cells_with_presence) | set(active_region_visits.keys()) | set(closed_region_visits.keys())
            if conflict_zone_cells is not None:
                cells_to_update &= conflict_zone_cells

            for cell in list(cells_to_update):
                region_cells = _region_cells_for_primary(cell, grid_rows, grid_cols, region_neighbors_enabled)
                present_by_class = defaultdict(dict)
                for cid, snapshots in track_snapshots.items():
                    for track_id, snapshot in snapshots.items():
                        if snapshot["cells"] & region_cells:
                            present_by_class[cid][track_id] = {
                                "point": snapshot["point"],
                                "footprint": snapshot["footprint"],
                            }

                active_by_class = active_region_visits[cell]
                for cid in class_ids:
                    current_tracks = present_by_class.get(cid, {})
                    current_active = active_by_class[cid]
                    for track_id, sample in current_tracks.items():
                        point = sample["point"]
                        footprint = sample["footprint"]
                        visit = current_active.get(track_id)
                        if visit is None:
                            visit = {
                                "class_id": cid,
                                "track_id": track_id,
                                "entry_frame": frame_count,
                                "last_frame": frame_count,
                                "points": [(frame_count, point)],
                                "footprints": [(frame_count, footprint)],
                            }
                            current_active[track_id] = visit
                        else:
                            visit["last_frame"] = frame_count
                            _append_point_if_new(visit["points"], frame_count, point)
                            _append_rect_if_new(visit["footprints"], frame_count, footprint)

                    for track_id in list(current_active.keys()):
                        if track_id in current_tracks:
                            continue
                        visit = current_active.pop(track_id)
                        visit["exit_frame"] = visit["last_frame"]
                        closed_region_visits[cell][cid].append(visit)

                    if closed_region_visits[cell][cid]:
                        closed_region_visits[cell][cid] = [
                            visit for visit in closed_region_visits[cell][cid]
                            if visit.get("exit_frame", visit["last_frame"]) >= cutoff_frame
                        ]

                if all(not active_by_class[cid] and not closed_region_visits[cell].get(cid) for cid in class_ids):
                    active_region_visits.pop(cell, None)
                    closed_region_visits.pop(cell, None)

            conflict_cells_this_frame = set()
            events_this_frame = []

            if cid_a is not None and cid_b is not None:
                cells_to_check = {selected_cell} if _sc_active else (set(active_region_visits.keys()) | set(closed_region_visits.keys()))
                if conflict_zone_cells is not None:
                    cells_to_check &= conflict_zone_cells
                for cell in sorted(cells_to_check):
                    region_cells = _region_cells_for_primary(cell, grid_rows, grid_cols, region_neighbors_enabled)
                    region_rect = _region_rect_from_cells(region_cells, width, height, grid_rows, grid_cols)
                    visits_a = list(closed_region_visits.get(cell, {}).get(cid_a, [])) + list(active_region_visits.get(cell, {}).get(cid_a, {}).values())
                    visits_b = list(closed_region_visits.get(cell, {}).get(cid_b, [])) + list(active_region_visits.get(cell, {}).get(cid_b, {}).values())
                    if not visits_a or not visits_b:
                        continue

                    for visit_a in visits_a:
                        for visit_b in visits_b:
                            sig = (
                                cell,
                                visit_a["class_id"], visit_a["track_id"], visit_a["entry_frame"], visit_a.get("exit_frame", -1),
                                visit_b["class_id"], visit_b["track_id"], visit_b["entry_frame"], visit_b.get("exit_frame", -1),
                            )

                            end_a = visit_a.get("exit_frame", visit_a["last_frame"])
                            end_b = visit_b.get("exit_frame", visit_b["last_frame"])
                            latest_start = max(visit_a["entry_frame"], visit_b["entry_frame"])
                            earliest_end = min(end_a, end_b)

                            if latest_start <= earliest_end:
                                overlap = True
                                pet_frames = 0
                                pet_seconds = 0.0
                                signed_pet_frames = 0
                                signed_pet_seconds = 0.0
                            elif end_a < visit_b["entry_frame"]:
                                overlap = False
                                pet_frames = visit_b["entry_frame"] - end_a
                                signed_pet_frames = pet_frames
                                pet_seconds = pet_frames / fps if fps > 0 else 0.0
                                signed_pet_seconds = signed_pet_frames / fps if fps > 0 else 0.0
                            elif end_b < visit_a["entry_frame"]:
                                overlap = False
                                pet_frames = visit_a["entry_frame"] - end_b
                                signed_pet_frames = -pet_frames
                                pet_seconds = pet_frames / fps if fps > 0 else 0.0
                                signed_pet_seconds = signed_pet_frames / fps if fps > 0 else 0.0
                            else:
                                continue

                            if pet_frames > max_pet_time:
                                continue

                            intersects, intersection_point = _polyline_intersects_in_rect(
                                visit_a["points"],
                                visit_b["points"],
                                region_rect,
                                visit_a.get("footprints"),
                                visit_b.get("footprints"),
                            )
                            if not intersects:
                                continue

                            heading_a = _buffered_heading_vector_from_points(visit_a["points"], trajectory_history)
                            heading_b = _buffered_heading_vector_from_points(visit_b["points"], trajectory_history)
                            angle_delta = _angle_delta_degrees(heading_a, heading_b)
                            if _is_collinear_direction(angle_delta, parallel_angle_tolerance):
                                continue
                            motion_relation = _opposing_motion_relation(
                                visit_a["points"], visit_b["points"], heading_a, heading_b
                            )

                            conflict_cells_this_frame.update(
                                _pet_activation_display_cells(cell, region_cells, conflict_zone_cells)
                            )
                            if sig in event_signatures:
                                continue

                            pet_bin = None if overlap else _pet_bin_label(pet_seconds)
                            pet_bin_risk = None if overlap else _pet_bin_risk(pet_seconds)
                            pet_bin_level = None if overlap else _pet_bin_plot_level(pet_seconds)
                            name_a = class_map[cid_a]['name']
                            name_b = class_map[cid_b]['name']
                            speed_a_ft = track_speeds_ft.get((cid_a, visit_a["track_id"])) if track_speed else None
                            speed_b_ft = track_speeds_ft.get((cid_b, visit_b["track_id"])) if track_speed else None
                            speed_key = speed_unit.replace("/", "_")
                            event_row = {
                                'frame': frame_count,
                                'time_sec': round(frame_count / fps if fps > 0 else 0.0, 3),
                                'cell_row': cell[0],
                                'cell_col': cell[1],
                                'pet_frames': int(pet_frames),
                                'pet_seconds': round(pet_seconds, 3),
                                'pet_bin': pet_bin,
                                'pet_bin_risk': pet_bin_risk,
                                'pet_bin_level': pet_bin_level,
                                'pet_undefined_overlap': overlap,
                                'signed_pet_frames': int(signed_pet_frames),
                                'signed_pet_seconds': round(signed_pet_seconds, 3),
                                f'{name_a.lower()}_id': visit_a["track_id"],
                                f'{name_b.lower()}_id': visit_b["track_id"],
                                f'{name_a.lower()}_entry_frame': visit_a["entry_frame"],
                                f'{name_a.lower()}_exit_frame': end_a,
                                f'{name_b.lower()}_entry_frame': visit_b["entry_frame"],
                                f'{name_b.lower()}_exit_frame': end_b,
                                'trajectory_intersects': True,
                                'intersection_x': round(intersection_point[0], 3) if intersection_point else None,
                                'intersection_y': round(intersection_point[1], 3) if intersection_point else None,
                                'trajectory_angle_delta_degrees': round(angle_delta, 3) if angle_delta is not None else None,
                                'collinear_angle_tolerance_degrees': float(parallel_angle_tolerance),
                                'opposing_motion_relation': motion_relation,
                                'direction_filter_passed': True,
                                'speed_tracking_enabled': bool(track_speed),
                                'speed_space': active_speed_space if track_speed else None,
                                'speed_feet_per_pixel': round(speed_feet_per_pixel, 8) if speed_feet_per_pixel is not None else None,
                                f'{name_a.lower()}_speed_ft_per_s': round(speed_a_ft, 3) if speed_a_ft is not None else None,
                                f'{name_b.lower()}_speed_ft_per_s': round(speed_b_ft, 3) if speed_b_ft is not None else None,
                                f'{name_a.lower()}_speed_{speed_key}': round(_speed_to_unit(speed_a_ft, speed_unit), 3) if speed_a_ft is not None else None,
                                f'{name_b.lower()}_speed_{speed_key}': round(_speed_to_unit(speed_b_ft, speed_unit), 3) if speed_b_ft is not None else None,
                            }
                            event_row.update(speed_event_fields)
                            conflict_events.append(event_row)
                            events_this_frame.append(event_row)
                            event_signatures.add(sig)

                            if not overlap:
                                frame_to_pets[frame_count].append(pet_seconds)
                                frame_to_pet_bin_levels[frame_count].append(pet_bin_level)
                                frame_to_pet_bin_risks[frame_count].append(pet_bin_risk)
                                cell_pet_values[cell].append(pet_seconds)
                                if _sc_active and cell == selected_cell:
                                    _sc_last_pet_secs = pet_seconds
                                    _sc_last_pet_bin = pet_bin

            if _sc_active:
                classes_in_cell_now = set()
                selected_region = {selected_cell}
                for cid, snapshots in track_snapshots.items():
                    for snapshot in snapshots.values():
                        if snapshot["cells"] & selected_region:
                            classes_in_cell_now.add(cid)
                            break

                entered_classes = classes_in_cell_now.difference(_sc_prev_classes_in_cell)
                _sc_prev_classes_in_cell = set(classes_in_cell_now)
                if _sc_timer_state == 'idle' and classes_in_cell_now:
                    for cid_now in classes_in_cell_now:
                        _sc_last_entry_frame_by_class[cid_now] = frame_count
                else:
                    for cid_now in entered_classes:
                        _sc_last_entry_frame_by_class[cid_now] = frame_count

                selected_events = [ev for ev in events_this_frame if (ev['cell_row'], ev['cell_col']) == selected_cell]
                if _sc_timer_state == 'idle':
                    if classes_in_cell_now:
                        _sc_timer_state = 'running'
                        _sc_timer_start_frame = frame_count
                        _sc_exit_frame = None
                elif _sc_timer_state == 'running':
                    if selected_events and _sc_last_event_frame != frame_count:
                        locked_event = min(selected_events, key=lambda ev: (ev['pet_undefined_overlap'], ev['pet_seconds']))
                        _sc_timer_state = 'locked'
                        _sc_timer_locked_secs = locked_event['pet_seconds']
                        _sc_timer_locked_bin = locked_event['pet_bin'] if not locked_event['pet_undefined_overlap'] else "OVERLAP"
                        _sc_lock_hold_remaining = _sc_lock_hold_frames
                        _sc_exit_frame = None
                        _sc_prev_classes_in_cell = set()
                        _sc_last_entry_frame_by_class = {}
                        _sc_last_event_frame = frame_count
                    elif classes_in_cell_now:
                        _sc_exit_frame = None
                    else:
                        if _sc_exit_frame is None:
                            _sc_exit_frame = frame_count
                        if (frame_count - _sc_exit_frame) > max_pet_time:
                            _sc_timer_state = 'idle'
                            _sc_timer_start_frame = None
                            _sc_exit_frame = None
                            _sc_last_pet_secs = None
                            _sc_last_pet_bin = None
                            _sc_prev_classes_in_cell = set()
                            _sc_last_entry_frame_by_class = {}
                elif _sc_timer_state == 'locked':
                    _sc_lock_hold_remaining -= 1
                    if _sc_lock_hold_remaining <= 0:
                        _sc_timer_state = 'idle'
                        _sc_timer_start_frame = None
                        _sc_timer_locked_secs = None
                        _sc_timer_locked_bin = None
                        _sc_last_pet_secs = None
                        _sc_last_pet_bin = None
                        _sc_exit_frame = None
                        _sc_prev_classes_in_cell = set()
                        _sc_last_entry_frame_by_class = {}

            annotated_frame = frame.copy()
            if show_deadzones:
                for zone in deadzones:
                    pts = np.array(zone, dtype=np.int32)
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 0, 180))
                    cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)
                    cv2.polylines(annotated_frame, [pts], True, (0, 0, 200), 1)

            if conflict_zone_polygon:
                pts = np.array(conflict_zone_polygon, dtype=np.int32)
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 180, 180))
                cv2.addWeighted(overlay, 0.18, annotated_frame, 0.82, 0, annotated_frame)
                cv2.polylines(annotated_frame, [pts], True, (0, 255, 255), 2)
                if show_grid and conflict_zone_cells:
                    for zr, zc in conflict_zone_cells:
                        zx1 = int(zc * cell_w)
                        zy1 = int(zr * cell_h)
                        zx2 = int((zc + 1) * cell_w)
                        zy2 = int((zr + 1) * cell_h)
                        cv2.rectangle(annotated_frame, (zx1, zy1), (zx2, zy2), (0, 180, 180), 1)

            if show_grid:
                for i in range(1, grid_rows):
                    y = int(i * cell_h)
                    cv2.line(annotated_frame, (0, y), (width, y), (60, 60, 60), 1)
                for j in range(1, grid_cols):
                    x = int(j * cell_w)
                    cv2.line(annotated_frame, (x, 0), (x, height), (60, 60, 60), 1)

            for row, col in conflict_cells_this_frame:
                x1 = int(col * cell_w)
                y1 = int(row * cell_h)
                x2 = int((col + 1) * cell_w)
                y2 = int((row + 1) * cell_h)
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if show_indicators:
                display_speed_lines = _normalize_speed_lines(calibration)
                primary_line = display_speed_lines.get("primary") or {}
                secondary_line = display_speed_lines.get("secondary") or {}
                if primary_line.get("start") and primary_line.get("end"):
                    cv2.line(annotated_frame, tuple(map(int, primary_line["start"])), tuple(map(int, primary_line["end"])), (0, 255, 255), 2)
                if secondary_line.get("start") and secondary_line.get("end"):
                    cv2.line(annotated_frame, tuple(map(int, secondary_line["start"])), tuple(map(int, secondary_line["end"])), (255, 255, 0), 2)

            for cid, tracks in confirmed_by_class.items():
                cls_cfg = class_map[cid]
                color = tuple(cls_cfg['color'])
                text_color = tuple(cls_cfg['text_color'])
                prefix = cls_cfg['name'][0]
                for track in tracks:
                    track_id = track.track_id
                    tlbr = track.to_tlbr()
                    x1, y1, x2, y2 = map(int, tlbr)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    if show_direction_lines:
                        _draw_direction_arrow(
                            annotated_frame,
                            tlbr,
                            _buffered_heading_vector_from_points(track_histories[(cid, track_id)], trajectory_history),
                            color,
                        )
                    conf_val = track.det_conf
                    conf_str = f" {conf_val:.2f}" if conf_val is not None else ""
                    label = f"{prefix}#{track_id}{conf_str}"
                    if track_speed and not hide_speed_overlay:
                        speed_value = _speed_to_unit(track_speeds_ft.get((cid, track_id)), speed_unit)
                        speed_label = _format_speed_label(speed_value, speed_unit)
                        if speed_label:
                            label = f"{label} {speed_label}"
                    lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - lsz[1] - 8), (x1 + lsz[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            if conflict_cells_this_frame and not _sc_active:
                cv2.putText(annotated_frame, f"CONFLICT ZONES: {len(conflict_cells_this_frame)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if _sc_active:
                sr, sc_col = selected_cell
                sx1 = int(sc_col * cell_w)
                sx2 = int((sc_col + 1) * cell_w)
                sy1 = int(sr * cell_h)
                sy2 = int((sr + 1) * cell_h)
                cell_px = sx2 - sx1
                fscale = max(0.45, min(1.4, cell_px / 120))
                thick = max(1, round(fscale * 2))
                if _sc_timer_state == 'idle':
                    cv2.rectangle(annotated_frame, (sx1, sy1), (sx2, sy2), (160, 160, 160), 1)
                elif _sc_timer_state == 'running':
                    if _sc_exit_frame is None:
                        elapsed = (frame_count - _sc_timer_start_frame) / fps if fps > 0 else 0.0
                        lbl = f"{elapsed:.2f}s"
                    else:
                        gap = (frame_count - _sc_exit_frame) / fps if fps > 0 else 0.0
                        lbl = f"GAP {gap:.2f}s"
                    clr = (0, 210, 0)
                    cv2.rectangle(annotated_frame, (sx1, sy1), (sx2, sy2), clr, 2)
                    (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
                    tx = (sx1 + sx2) // 2 - tw // 2
                    ty = (sy1 + sy2) // 2 + th // 2
                    cv2.rectangle(annotated_frame, (tx - 3, ty - th - bl - 2), (tx + tw + 3, ty + bl + 2), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fscale, clr, thick)
                elif _sc_timer_state == 'locked':
                    bin_text = f" [{_sc_timer_locked_bin}]" if _sc_timer_locked_bin else ""
                    lbl = f"PET {_sc_timer_locked_secs:.2f}s{bin_text}"
                    clr = (40, 40, 220)
                    cv2.rectangle(annotated_frame, (sx1, sy1), (sx2, sy2), clr, 2)
                    (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
                    tx = (sx1 + sx2) // 2 - tw // 2
                    ty = (sy1 + sy2) // 2 + th // 2
                    cv2.rectangle(annotated_frame, (tx - 3, ty - th - bl - 2), (tx + tw + 3, ty + bl + 2), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fscale, clr, thick)

                hud_y = 10
                if conflict_cells_this_frame:
                    cz = f"CONFLICT ZONES: {len(conflict_cells_this_frame)}"
                    (cw, ch), _ = cv2.getTextSize(cz, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated_frame, (8, hud_y - ch - 4), (14 + cw, hud_y + 4), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, cz, (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    hud_y += 32
                if _sc_last_entry_frame_by_class:
                    for cid_key in sorted(_sc_last_entry_frame_by_class.keys()):
                        last_f = _sc_last_entry_frame_by_class.get(cid_key)
                        if last_f is None:
                            continue
                        dt = (frame_count - last_f) / fps if fps > 0 else 0.0
                        name = class_map.get(cid_key, {}).get('name', str(cid_key))
                        prefix = name[0].upper() if name else str(cid_key)
                        txt = f"{prefix} ENTRY  {dt:.2f}s ago"
                        (w2, h2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                        cv2.rectangle(annotated_frame, (8, hud_y - h2 - 4), (14 + w2, hud_y + 4), (0, 0, 0), -1)
                        cv2.putText(annotated_frame, txt, (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
                        hud_y += 24
                if _sc_timer_state != 'idle':
                    if _sc_timer_state == 'running':
                        if _sc_exit_frame is None:
                            hud_elapsed = (frame_count - _sc_timer_start_frame) / fps if fps > 0 else 0.0
                            hud_txt = f"TIMER  {hud_elapsed:.2f}s"
                        else:
                            hud_gap = (frame_count - _sc_exit_frame) / fps if fps > 0 else 0.0
                            hud_txt = f"GAP  {hud_gap:.2f}s"
                        hud_color = (0, 210, 0)
                    else:
                        extra_bin = f" [{_sc_timer_locked_bin}]" if _sc_timer_locked_bin else ""
                        hud_txt = f"PET LOCKED  {_sc_timer_locked_secs:.2f}s{extra_bin}"
                        hud_color = (40, 40, 220)
                    (hw, hh), _ = cv2.getTextSize(hud_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated_frame, (8, hud_y - hh - 4), (14 + hw, hud_y + 4), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, hud_txt, (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)

            out.write(annotated_frame)
            frame_count += 1
            pbar.update(1)

            if display_available:
                try:
                    cv2.imshow('RT-DETR PET Analysis', annotated_frame)
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

    base = os.path.splitext(output_video_path)[0]
    if output_csv_path is None:
        output_csv_path = base + "_PET_conflicts.csv"

    df = pd.DataFrame(conflict_events)
    if not df.empty:
        df.to_csv(output_csv_path, index=False)

    output_heatmap_path = output_heatmap_path or (base + "_PET_heatmap.png")
    cap_heat = cv2.VideoCapture(input_video_path)
    if cap_heat.isOpened():
        ret, first_frame = cap_heat.read()
        cap_heat.release()
        if ret and first_frame is not None and cell_pet_values:
            heatmap_img = _build_heatmap_image(cell_pet_values, grid_rows, grid_cols, width, height, fps, max_pet_time, first_frame)
            cv2.imwrite(output_heatmap_path, heatmap_img)

    output_plot_path = base + "_PET_over_time.png"
    if frame_count > 0:
        frames_all = np.arange(frame_count, dtype=int)
        time_sec_arr = frames_all.astype(float) / fps if fps > 0 else frames_all.astype(float)
        pet_seconds_arr = np.full_like(time_sec_arr, DEFAULT_NON_PET_SECONDS, dtype=float)
        for frame_idx, vals in frame_to_pets.items():
            if 0 <= frame_idx < frame_count and vals:
                pet_seconds_arr[frame_idx] = min(DEFAULT_NON_PET_SECONDS, min(vals))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.step(time_sec_arr, pet_seconds_arr, where="post", color="steelblue", linewidth=1.5)
        event_mask = pet_seconds_arr < DEFAULT_NON_PET_SECONDS
        if np.any(event_mask):
            ax.scatter(time_sec_arr[event_mask], pet_seconds_arr[event_mask], color="steelblue", s=12, zorder=3)
        ax.axhline(DEFAULT_NON_PET_SECONDS, color="gray", linestyle="--", linewidth=0.8)
        ax.axhspan(0, 1.5, color="crimson", alpha=0.08)
        ax.axhspan(1.5, 3.0, color="orange", alpha=0.08)
        ax.axhspan(3.0, 5.0, color="gold", alpha=0.08)
        ax.axhspan(5.0, DEFAULT_NON_PET_SECONDS, color="green", alpha=0.05)
        ax.set_ylim(0, DEFAULT_NON_PET_SECONDS + 0.5)
        ax.set_yticks([0, 1.5, 3.0, 5.0, DEFAULT_NON_PET_SECONDS])
        ax.set_yticklabels(["0", "1.5", "3", "5", "Normal/10"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("PET (s)")
        ax.set_title("PET over time (normal/no PET = 10s; lower = more critical)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_plot_path, dpi=150)
        plt.close(fig)

    output_risk_plot_path = base + "_Risk_PET_over_time.png"
    if frame_count > 0:
        frames_all = np.arange(frame_count, dtype=int)
        time_sec_arr = frames_all.astype(float) / fps if fps > 0 else frames_all.astype(float)
        risk_per_frame = np.zeros_like(time_sec_arr, dtype=float)
        for frame_idx, vals in frame_to_pet_bin_risks.items():
            if 0 <= frame_idx < frame_count and vals:
                risk_per_frame[frame_idx] = max(r for r in vals if r is not None)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.step(time_sec_arr, risk_per_frame, where="post", color="crimson", linewidth=1.5)
        event_mask = risk_per_frame > 0
        if np.any(event_mask):
            ax.scatter(time_sec_arr[event_mask], risk_per_frame[event_mask], color="crimson", s=12, zorder=3)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["Normal", "5+s", "3-5s", "1.5-3s", "0-1.5s"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Binned PET risk")
        ax.set_title("Binned PET risk over time (higher = lower PET = more critical)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_risk_plot_path, dpi=150)
        plt.close(fig)

    tqdm.write(f"Video:   {os.path.abspath(output_video_path)}")
    tqdm.write(f"CSV:     {os.path.abspath(output_csv_path)}")
    tqdm.write(f"Heatmap: {os.path.abspath(output_heatmap_path)}")
    if frame_count > 0:
        tqdm.write(f"Plot:    {os.path.abspath(output_plot_path)}")
        tqdm.write(f"Risk:    {os.path.abspath(output_risk_plot_path)}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="RT-DETR PET conflict zone detection with config-driven multi-pass detection."
    )
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config file.")
    parser.add_argument("--input", "-i", default=None, help="Override input video path from config.")
    parser.add_argument("--output", "-o", default=None, help="Override output video path.")
    parser.add_argument("--model", "-m", default=None, help="Override model path from config.")
    parser.add_argument("--grid-size", type=int, default=100, help="N for NxN conflict grid (default: 100).")
    parser.add_argument("--max-pet-time", type=int, default=300, help="Conflict window in frames (default: 300).")
    parser.add_argument("--no-neighbors", action="store_true", help="Disable 3x3 neighbor cell aggregation.")
    parser.add_argument("--show-grid", action="store_true", help="Draw faint grid lines on output video.")
    parser.add_argument("--no-grid", action="store_true", help="Single-cell mode: user clicks one cell; PET computed only in that cell.")
    parser.add_argument("--csv", metavar="PATH", help="Output CSV path (default: auto).")
    parser.add_argument("--heatmap", metavar="PATH", help="Output heatmap image path (default: auto).")
    parser.add_argument("--display", action="store_true", help="Show live preview window.")
    parser.add_argument("--deadzone", action="store_true", help="Interactively draw rectangular deadzones on the first frame before processing.")
    parser.add_argument("--show-deadzones", action="store_true", help="Render deadzone overlays on the output video / live preview.")
    parser.add_argument("--conflict-zone", action="store_true",
                        help="Interactively draw a polygon; PET is computed only for grid cells overlapping it.")
    parser.add_argument("--parallel-angle-tolerance", type=float, default=15.0,
                        help="Exclude PET pairs whose buffered headings are within this many degrees of 0 or 180.")
    parser.add_argument("--trajectory-history", type=int, default=15,
                        help="Recent frame history used for trajectory heading/intersection checks.")
    parser.add_argument("--track-speed", action="store_true",
                        help="Prompt for a 52 ft calibration line on the first frame and estimate object speed.")
    parser.add_argument("--second-speed-line", action="store_true",
                        help="After the 52 ft speed line, optionally prompt for a 72 ft second reference line.")
    parser.add_argument("--line-warp", action="store_true",
                        help="Use calibration.speed_lines endpoints for warp calibration when valid; falls back to passes.warp.src_points.")
    parser.add_argument("--speed-unit", choices=["mph", "ft/s"], default="mph",
                        help="Display/export unit for speed outputs.")
    parser.add_argument("--speed-space", choices=["auto", "warp", "image"], default="auto",
                        help="Coordinate space for speed estimation: auto uses homography when available.")
    parser.add_argument("--speed-smoothing-window", type=int, default=5,
                        help="Rolling speed smoothing window in samples when EMA is disabled.")
    parser.add_argument("--speed-smoothing-alpha", type=float, default=0.0,
                        help="Optional EMA smoothing alpha for speed, 0 disables EMA; lower values are smoother.")
    parser.add_argument("--hide-speed-overlay", action="store_true",
                        help="Keep speed in CSV outputs but hide speed labels on the video.")
    parser.add_argument("--show-direction-lines", action="store_true",
                        help="Draw short movement direction arrows from each bounding-box center.")
    parser.add_argument("--show-indicators", action="store_true",
                        help="Draw calibration speed-line indicators on the output video / live preview.")

    args = parser.parse_args()
    cfg = load_config(args.config)
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
        base = os.path.splitext(model_path)[0]
        for fallback in (base + ".onnx", base + ".pt"):
            if os.path.exists(fallback):
                print(f"Model not found at {model_path}, falling back to: {fallback}")
                model_path = fallback
                cfg['model'] = model_path
                break
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    if not cfg.get('output') and not args.output:
        output_dir, run_number = _get_pet_output_dir(input_path)
        os.makedirs(output_dir, exist_ok=True)
        video_basename = os.path.splitext(os.path.basename(input_path))[0] or "video"
        cfg['output'] = os.path.join(output_dir, f"{video_basename}.mp4")
        tqdm.write(f"Results directory: {os.path.abspath(output_dir)} (run {run_number})")

    use_compile = cfg.get('inference', {}).get('compile', False)
    model = load_model(model_path, DEVICE, use_compile=use_compile)

    first_frame_for_setup = None
    if args.deadzone or args.track_speed or args.second_speed_line or args.conflict_zone:
        setup_cap = cv2.VideoCapture(cfg['input'])
        ret_first, first_frame_for_setup = setup_cap.read()
        setup_cap.release()
        if not ret_first or first_frame_for_setup is None:
            first_frame_for_setup = None
            print("WARNING: Could not read first frame for interactive setup.")

    deadzones = []
    if args.deadzone:
        if first_frame_for_setup is not None:
            print("Deadzone setup: left-click + drag to draw exclusion rectangles on the first frame.")
            deadzones = _draw_deadzones_interactive(first_frame_for_setup.copy())
            print(f"{len(deadzones)} deadzone(s) configured.")
        else:
            print("WARNING: Could not read first frame for deadzone setup.")

    conflict_zone_polygon = None
    if args.conflict_zone:
        if first_frame_for_setup is not None:
            print("Conflict zone setup: left-click polygon points on the first frame.")
            conflict_zone_polygon = _draw_polygon_interactive(first_frame_for_setup.copy())
            if conflict_zone_polygon:
                print(f"Conflict zone configured with {len(conflict_zone_polygon)} point(s).")
            else:
                print("WARNING: Conflict zone cancelled or invalid. Using full-grid PET.")
        else:
            print("WARNING: Could not read first frame for conflict zone setup.")

    calibration = dict(cfg.get('calibration', {}) or {})
    if args.track_speed:
        speed_lines = dict((calibration.get("speed_lines") or {}))
        if first_frame_for_setup is not None:
            print("Speed setup: draw the full 52 ft bus-stop stretch on the first frame.")
            primary_line = _draw_measure_line_interactive(
                first_frame_for_setup.copy(), length_ft=52.0, label="primary"
            ) or {}
            if primary_line:
                speed_lines["primary"] = primary_line
                calibration.update(primary_line)
                print(f"Speed calibration configured at {primary_line['feet_per_pixel']:.6f} ft/pixel.")
                if args.second_speed_line:
                    print("Speed setup: draw the optional 72 ft perpendicular reference line.")
                    secondary_line = _draw_measure_line_interactive(
                        first_frame_for_setup.copy(),
                        window_name="Draw 72 ft calibration line",
                        length_ft=72.0,
                        label="secondary",
                    ) or {}
                    if secondary_line:
                        speed_lines["secondary"] = secondary_line
                        speed_lines.setdefault("enabled_for_speed", True)
                        print(f"Second speed calibration configured at {secondary_line['feet_per_pixel']:.6f} ft/pixel.")
                    else:
                        print("WARNING: Second speed line cancelled or invalid. Using the 52 ft line only.")
                calibration["speed_lines"] = speed_lines
            elif not speed_lines.get("primary"):
                print("WARNING: Speed calibration cancelled or invalid. Speed tracking disabled.")
        elif not speed_lines.get("primary"):
            print("WARNING: Could not read first frame for speed setup. Speed tracking disabled.")
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
        show_indicators=args.show_indicators,
        conflict_zone_polygon=conflict_zone_polygon,
        parallel_angle_tolerance=args.parallel_angle_tolerance,
        trajectory_history=args.trajectory_history,
        track_speed=args.track_speed and bool(_normalize_speed_lines(calibration).get("primary")),
        speed_unit=args.speed_unit,
        speed_space=args.speed_space,
        speed_smoothing_window=args.speed_smoothing_window,
        speed_smoothing_alpha=args.speed_smoothing_alpha,
        line_warp=args.line_warp,
        hide_speed_overlay=args.hide_speed_overlay,
        calibration=calibration,
        show_direction_lines=args.show_direction_lines,
    )


if __name__ == "__main__":
    main()
