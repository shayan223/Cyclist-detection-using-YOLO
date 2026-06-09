from __future__ import annotations

import argparse
import os
from collections import defaultdict

import yaml

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MODEL_PATH = './yolo26l_100epoch_v5.pt'


# ---------------------------------------------------------------------------
# Model loading
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
# Detection helpers
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
# IoU / NMS helpers
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
    Checks whether >= min_overlap_fraction of inner's area overlaps outer,
    regardless of how small inner is relative to outer.  This catches the
    box-inside-box case that IoU-based NMS misses: a small torso crop fully
    inside a full-body box has IoU ~0.2 (low union penalty) but overlap
    fraction ~1.0 (inner is entirely inside outer).
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
    Greedy hard NMS per class. Always applied first to remove cross-pass/tile
    duplicate boxes before soft-NMS and the tracker see them.

    Two phases:
      Phase 1 — confidence-sorted greedy IoU NMS + containment (low-conf inner
                 inside high-conf outer). Handles the standard duplicate case.
      Phase 2 — size-sorted containment cleanup: removes any box whose area is
                 >= containment_fraction contained within a *larger* box,
                 regardless of which has higher confidence. Fixes the reverse
                 case where a high-confidence partial crop (torso) sits inside a
                 lower-confidence full-body box and survives Phase 1 because it
                 becomes `best` first in the confidence-sorted pass.
    """
    by_class = defaultdict(list)
    for det in detections:
        by_class[int(det[5])].append(det)
    kept = []
    for class_dets in by_class.values():
        # Phase 1: standard confidence-sorted greedy IoU NMS
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

        # Phase 2: remove any survivor substantially contained within a larger
        # survivor, regardless of confidence ordering.
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
# Multi-pass detection helpers
# ---------------------------------------------------------------------------

def _run_top_region_pass(frame, model, iou_threshold, top_region_ratio, conf_threshold, imgsz, class_filter, half=False, device=None):
    """Run detection on the upper region of the frame (where distant objects appear)."""
    h, _ = frame.shape[:2]
    top_h = int(max(1, min(h, round(h * top_region_ratio))))
    roi = frame[:top_h, :]
    results = _run_detector(model, roi, conf_threshold, iou_threshold, imgsz=imgsz, half=half, device=device)
    return _extract_boxes(results, x_offset=0, y_offset=0, class_filter=class_filter)


def _build_homography(src_points, dst_size):
    """Compute H (src→dst) and H_inv (dst→src) from 4 calibration points.

    src_points: [[x,y], ...] 4 points in original frame, clockwise from top-left,
                that form a rectangle in real-world space (e.g. lane corners).
    dst_size:   [width, height] of the rectified output image.
    """
    src = np.array(src_points, dtype=np.float32)
    dw, dh = dst_size
    dst = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], dtype=np.float32)
    H     = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    return H, H_inv


def _warp_boxes_to_original(boxes, H_inv):
    """Map axis-aligned boxes from warped space back to original frame coordinates.

    Because the warp is a projective transform, a rectangle in warped space maps
    to a quadrilateral in original space.  We transform all 4 corners and take
    the axis-aligned bounding box of the result.
    """
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
    """Perspective-corrected detection pass.

    Applies the calibrated homography to straighten the camera's perspective
    distortion, runs detection on the rectified image, then maps boxes back to
    original frame coordinates via the inverse homography.  This makes distant
    objects (which shrink toward the top of an overhead camera) appear at a more
    consistent scale, improving recall without needing very small SAHI tiles.

    Calibrate H/H_inv with warp_calibrate.py and store the points in config.yaml.
    """
    dw, dh = dst_size
    warped       = cv2.warpPerspective(frame, H, (dw, dh))
    results      = _run_detector(model, warped, conf_threshold, iou_threshold,
                                 imgsz=imgsz, half=half, device=device)
    warped_boxes = _extract_boxes(results, class_filter=class_filter)
    return _warp_boxes_to_original(warped_boxes, H_inv)


def _run_tiled_pass(frame, model, iou_threshold, conf_threshold, tile_size, tile_overlap,
                    imgsz, class_filter, half=False, device=None,
                    y_max_fraction=1.0, prescale=1.0):
    """SAHI-style tiled inference pass; batches all tiles into a single model call.

    Args:
        y_max_fraction: Restrict tiling to the top fraction of the frame (0.0–1.0).
                        e.g. 0.5 tiles only the top half, saving ~50% compute and
                        focusing where perspective makes objects small.
        prescale:       Upscale the tiling region by this factor before slicing tiles.
                        e.g. 2.0 doubles the apparent size of objects for the model,
                        equivalent to a simple image-pyramid step. Detections are
                        scaled back to original coordinates after inference.
                        Costs proportionally more memory; use 1.5–2.0 for top region.
    """
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
    """Split flat detection list into per-class DeepSort input tuples.

    min_confidences: optional {class_id: float} applied as a per-class
    post-filter after the global inference threshold.  Detections below a
    class's floor are dropped before reaching the tracker.
    """
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
# Main video processing
# ---------------------------------------------------------------------------

def process_video(
    input_video_path,
    output_video_path,
    model,
    cfg,
    device='cpu',
    disable_display=True,
    inference_only=False,
    deadzones=None,
    show_deadzones=False,
):
    deadzones = deadzones or []
    # --- Unpack config sections ---
    inf_cfg   = cfg.get('inference', {})
    pass_cfg  = cfg.get('passes', {})
    nms_cfg   = cfg.get('nms', {})
    dbg_cfg   = cfg.get('debug', {})
    classes   = cfg.get('classes', [])

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

    # --- Warp pass ---
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

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = int(1000 / fps) if fps > 0 else 33

    print(f"Video: {width}x{height} @ {fps}fps  ({total_frames} frames)")

    # --- Per-class trackers ---
    trackers = {}
    ids_seen = {}
    if not inference_only:
        if DeepSort is None:
            raise RuntimeError(
                "deep_sort_realtime is not installed. Install it or run with --inference-only."
            )
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
            ids_seen[cid] = set()

    # --- Video writer with codec fallbacks ---
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    requested_ext  = os.path.splitext(output_video_path)[1].lower()
    requested_base = os.path.splitext(output_video_path)[0] if requested_ext else output_video_path

    writer_attempts = []
    if requested_ext in ("", ".mp4"):
        writer_attempts += [(f"{requested_base}.mp4", "mp4v"), (f"{requested_base}.mp4", "avc1"),
                            (f"{requested_base}.avi", "XVID"), (f"{requested_base}.avi", "MJPG")]
    elif requested_ext == ".avi":
        writer_attempts += [(output_video_path, "XVID"), (output_video_path, "MJPG"),
                            (f"{requested_base}.mp4", "mp4v")]
    else:
        writer_attempts += [(output_video_path, "mp4v"), (f"{requested_base}.mp4", "mp4v"),
                            (f"{requested_base}.avi", "XVID")]

    out = None
    final_output_path = output_video_path
    for candidate_path, codec_name in writer_attempts:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        out = cv2.VideoWriter(candidate_path, fourcc, fps, (width, height))
        if out.isOpened():
            final_output_path = candidate_path
            print(f"VideoWriter: {codec_name} -> {candidate_path}")
            break
        out.release()
        out = None

    if out is None:
        raise RuntimeError("Could not create VideoWriter with any tried codec.")

    # --- Overlay geometry (dynamic: one row per class) ---
    font_scale, font_thickness, padding, line_spacing = 0.8, 2, 15, 5
    sample_size, _ = cv2.getTextSize("Pedestrians: 999 (Total: 9999)",
                                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_height    = sample_size[1]
    max_text_width = max(sample_size[0], 350)
    text_x         = width - max_text_width - padding
    line_height    = text_height + line_spacing
    n_classes      = len(classes)
    bg_x1 = max(0, text_x - padding)
    bg_y1 = max(0, height - padding - n_classes * line_height - padding)
    bg_x2 = min(width, text_x + max_text_width + padding)
    bg_y2 = min(height, height)

    frame_count       = 0
    _cached_tile_dets = []
    paused            = False
    speed_multiplier  = 1.0
    annotated_frame   = None
    display_available = not disable_display

    progress_bar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing video", unit="frame")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_h, frame_w = frame.shape[:2]

                # Optional frame downscale for faster inference
                if downscale_width > 0 and downscale_height > 0 and (frame_w > downscale_width or frame_h > downscale_height):
                    proc_frame = cv2.resize(frame, (downscale_width, downscale_height))
                    scale_x = frame_w / downscale_width
                    scale_y = frame_h / downscale_height
                else:
                    proc_frame = frame
                    scale_x = scale_y = 1.0
                proc_h, proc_w = proc_frame.shape[:2]

                # Pass 1: full-frame detection
                full_dets = _extract_boxes(
                    _run_detector(model, proc_frame, confidence_threshold, iou_threshold,
                                  imgsz=imgsz, half=half, device=device),
                    class_filter=class_ids,
                )

                # Pass 2: optional top-region pass
                top_dets = []
                if top_region_pass:
                    top_dets = _run_top_region_pass(
                        proc_frame, model, iou_threshold, top_region_ratio,
                        top_region_conf or confidence_threshold,
                        top_region_imgsz, class_ids, half=half, device=device,
                    )

                # Pass 3: optional SAHI tiled pass (batched; runs every tile_interval frames)
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

                # Pass 4: optional perspective warp pass
                warp_dets = []
                if warp_enabled:
                    warp_dets = _run_warp_pass(
                        proc_frame, model, iou_threshold,
                        warp_conf or confidence_threshold,
                        warp_H, warp_H_inv, warp_dst_size,
                        warp_imgsz, class_ids, half=half, device=device,
                    )

                # Merge, clip, scale back, NMS
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

                annotated_frame = frame.copy()
                # Draw deadzone overlays (only when --show-deadzones is enabled)
                if show_deadzones:
                    for _dz in deadzones:
                        _pts = np.array(_dz, dtype=np.int32)
                        _ov  = annotated_frame.copy()
                        cv2.fillPoly(_ov, [_pts], (0, 0, 180))
                        cv2.addWeighted(_ov, 0.25, annotated_frame, 0.75, 0, annotated_frame)
                        cv2.polylines(annotated_frame, [_pts], True, (0, 0, 200), 1)
                current_counts  = {cid: 0 for cid in class_ids}

                if inference_only:
                    for x1, y1, x2, y2, conf, cls_int in processed_dets:
                        if cls_int not in class_map:
                            continue
                        if float(conf) < min_confidences.get(cls_int, 0.0):
                            continue
                        cls_cfg    = class_map[cls_int]
                        color      = tuple(cls_cfg['color'])
                        text_color = tuple(cls_cfg['text_color'])
                        label      = f"{cls_cfg['name']} {conf:.2f}"
                        x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                        current_counts[cls_int] += 1
                        cv2.rectangle(annotated_frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                        lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1_i, y1_i - lsz[1] - 10),
                                      (x1_i + lsz[0], y1_i), color, -1)
                        cv2.putText(annotated_frame, label, (x1_i, y1_i - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                else:
                    by_class = _split_detections_by_class(processed_dets, class_ids, min_confidences)
                    for cid, cls_cfg in class_map.items():
                        dets   = by_class.get(cid, [])
                        tracks = trackers[cid].update_tracks(dets, frame=frame)
                        color      = tuple(cls_cfg['color'])
                        text_color = tuple(cls_cfg['text_color'])

                        # Post-tracker containment check on Kalman track boxes.
                        # Pre-tracker NMS only sees raw detections — it cannot suppress
                        # a confirmed track whose Kalman box has drifted into another
                        # track, or a track that was confirmed before NMS was tightened.
                        confirmed = [t for t in tracks if t.is_confirmed()]
                        def _tlbr(t):
                            return list(map(int, t.to_tlbr()))
                        def _tarea(t):
                            b = _tlbr(t)
                            return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

                        visible_tracks = [
                            t for t in confirmed
                            if not any(
                                _tarea(other) > _tarea(t)
                                and _is_contained(_tlbr(t), _tlbr(other), nms_containment)
                                for other in confirmed if other is not t
                            )
                        ]

                        for track in visible_tracks:
                            track_id = track.track_id
                            x1, y1, x2, y2 = map(int, track.to_tlbr())
                            ids_seen[cid].add(track_id)
                            current_counts[cid] += 1
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            conf_val = track.det_conf
                            conf_str = f" {conf_val:.2f}" if conf_val is not None else ""
                            label = f"{cls_cfg['name']} #{track_id}{conf_str}"
                            lsz   = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(annotated_frame, (x1, y1 - lsz[1] - 10),
                                          (x1 + lsz[0], y1), color, -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                if frame_count % 10 == 0:
                    counts_str = ", ".join(
                        f"{class_map[cid]['name']}: {current_counts[cid]}" for cid in sorted(class_ids)
                    )
                    totals_str = ", ".join(
                        f"{class_map[cid]['name']}: {len(ids_seen[cid])}" for cid in sorted(class_ids)
                    ) if not inference_only else "n/a"
                    msg = f"Frame {frame_count}: [{counts_str}] | Total unique: [{totals_str}]"
                    if debug_detections:
                        msg += (f" | full={len(full_dets)} top={len(top_dets)} "
                                f"tile={len(tile_dets)} merged={len(merged_dets)} final={len(processed_dets)}")
                    print(msg)

                # --- Draw overlay counter (bottom-right, one row per class) ---
                cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                for i, cls_cfg in enumerate(reversed(classes)):
                    cid   = cls_cfg['id']
                    color = tuple(cls_cfg['color'])
                    cnt   = current_counts.get(cid, 0)
                    if inference_only:
                        text = f"{cls_cfg['name']}s: {cnt}"
                    else:
                        text = f"{cls_cfg['name']}s: {cnt} (Total: {len(ids_seen.get(cid, set()))})"
                    text_y = height - padding - i * line_height
                    cv2.putText(annotated_frame, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

                out.write(annotated_frame)
                frame_count += 1
                progress_bar.update(1)

            if display_available and annotated_frame is not None:
                try:
                    cv2.imshow("YOLO26 + DeepSORT", annotated_frame)
                except (cv2.error, Exception):
                    display_available = False
                    print("Display not available. Continuing without live view.")

            actual_delay = max(1, int(frame_delay / speed_multiplier)) if display_available else 1
            key = cv2.waitKey(actual_delay) & 0xFF if display_available else 0
            if key == ord("q"):
                break
            elif key == ord("s") and annotated_frame is not None:
                frame_filename = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(frame_filename, annotated_frame)
                print(f"Frame saved: {frame_filename}")
            elif key == ord("p"):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key in (ord("+"), ord("=")):
                speed_multiplier = min(5.0, speed_multiplier + 0.5)
                print(f"Speed: {speed_multiplier:.1f}x")
            elif key == ord("-"):
                speed_multiplier = max(0.1, speed_multiplier - 0.5)
                print(f"Speed: {speed_multiplier:.1f}x")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")

    finally:
        progress_bar.close()
        cap.release()
        out.release()
        if display_available:
            try:
                cv2.destroyAllWindows()
            except (cv2.error, Exception):
                pass
        print(f"\nDone. Output: {final_output_path}")
        if inference_only:
            print("Inference-only mode (tracking disabled).")
        else:
            for cls_cfg in classes:
                cid = cls_cfg['id']
                print(f"Total unique {cls_cfg['name']}s tracked: {len(ids_seen.get(cid, set()))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLO26 + DeepSORT tracking.")
    parser.add_argument("--config", default="./config_trim4.yaml", help="Path to YAML config file.")
    parser.add_argument("--input",  "-i", default=None, help="Override input video path from config.")
    parser.add_argument("--output", "-o", default=None, help="Override output video path.")
    parser.add_argument("--model",  "-m", default=None, help="Override model path from config.")
    parser.add_argument("--no-display",     action="store_true", default=True, help="Disable live preview window.")
    parser.add_argument("--inference-only", action="store_true", help="Detector-only mode (no DeepSORT).")
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

    if not cfg.get('output'):
        base     = os.path.splitext(input_path)[0]
        inf_only = args.inference_only or cfg.get('debug', {}).get('inference_only', False)
        cfg['output'] = base + ("_yolo26_inference.mp4" if inf_only else "_yolo26_deepsort.mp4")

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
        disable_display=args.no_display,
        inference_only=args.inference_only or cfg.get('debug', {}).get('inference_only', False),
        deadzones=deadzones,
        show_deadzones=args.show_deadzones,
    )


if __name__ == "__main__":
    main()
