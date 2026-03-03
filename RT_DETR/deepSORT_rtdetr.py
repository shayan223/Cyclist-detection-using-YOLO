from __future__ import annotations

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import RTDETR

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MODEL_PATH = './runs/detect/cyclist_detection_rtdetr/rtdetr_finetune_imgsz640_mos1.0_mix0.15_fliplr0.52/weights/best.pt'


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path, device):
    """Load an RT-DETR model checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Loading RT-DETR model: {model_path}")
    model = RTDETR(model_path)
    model.to(device)
    print(f"Model loaded on device: {device}")
    return model


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _run_detector(model, image, conf_threshold, iou_threshold, imgsz=None):
    """Run one Ultralytics RT-DETR detector pass and return raw results."""
    kwargs = {"conf": conf_threshold, "iou": iou_threshold, "agnostic_nms": False, "verbose": False}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
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


def _hard_nms_per_class(detections, iou_threshold=0.45):
    """
    Greedy hard NMS per class. Always applied first to remove cross-pass/tile
    duplicate boxes before soft-NMS and the tracker see them.
    """
    by_class = defaultdict(list)
    for det in detections:
        by_class[int(det[5])].append(det)
    kept = []
    for class_dets in by_class.values():
        dets = sorted(class_dets, key=lambda d: d[4], reverse=True)
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets if _compute_iou_xyxy(best, d) <= iou_threshold]
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

def _run_top_region_pass(frame, model, iou_threshold, top_region_ratio, conf_threshold, imgsz, class_filter):
    """Run detection on the upper region of the frame (where distant objects appear)."""
    h, _ = frame.shape[:2]
    top_h = int(max(1, min(h, round(h * top_region_ratio))))
    roi = frame[:top_h, :]
    results = _run_detector(model, roi, conf_threshold, iou_threshold, imgsz=imgsz)
    return _extract_boxes(results, x_offset=0, y_offset=0, class_filter=class_filter)


def _run_tiled_pass(frame, model, iou_threshold, conf_threshold, tile_size, tile_overlap, imgsz, class_filter):
    """SAHI-style tiled inference pass; maps all tiles back to full-frame coordinates."""
    h, w = frame.shape[:2]
    tile = max(64, int(tile_size))
    overlap = max(0.0, min(0.8, float(tile_overlap)))
    stride = max(32, int(round(tile * (1.0 - overlap))))
    detections = []

    y_starts = list(range(0, max(1, h - tile + 1), stride))
    x_starts = list(range(0, max(1, w - tile + 1), stride))
    if not y_starts or y_starts[-1] != max(0, h - tile):
        y_starts.append(max(0, h - tile))
    if not x_starts or x_starts[-1] != max(0, w - tile):
        x_starts.append(max(0, w - tile))

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(h, y0 + tile)
            x1 = min(w, x0 + tile)
            tile_img = frame[y0:y1, x0:x1]
            if tile_img.size == 0:
                continue
            results = _run_detector(model, tile_img, conf_threshold, iou_threshold, imgsz=imgsz)
            detections.extend(_extract_boxes(results, x_offset=x0, y_offset=y0, class_filter=class_filter))
    return detections


def _detections_to_tracker_inputs(detections, cyclist_class_id=0, pedestrian_class_id=1):
    """Split flat detection list into per-class DeepSort input tuples."""
    cyclist_dets, pedestrian_dets = [], []
    for x1, y1, x2, y2, conf, cls_int in detections:
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        if cls_int == cyclist_class_id:
            cyclist_dets.append((bbox, float(conf), cls_int))
        elif cls_int == pedestrian_class_id:
            pedestrian_dets.append((bbox, float(conf), cls_int))
    return cyclist_dets, pedestrian_dets


# ---------------------------------------------------------------------------
# Main video processing
# ---------------------------------------------------------------------------

def process_video(
    input_video_path,
    output_video_path,
    model,
    confidence_threshold=0.65,
    max_age=15,
    max_iou_distance=0.6,
    iou_threshold=0.7,
    disable_display=True,
    imgsz=None,
    crowd_mode="soft-nms",
    soft_nms_iou=0.25,
    soft_nms_sigma=0.2,
    top_region_pass=True,
    top_region_ratio=0.45,
    top_region_imgsz=None,
    top_region_confidence=None,
    tile_mode="sahi",
    tile_size=480,
    tile_overlap=0.6,
    tile_imgsz=None,
    tile_confidence=None,
    nms_iou=0.45,
    nms_max_overlap=0.7,
    downscale_width=640,
    downscale_height=480,
    debug_detections=False,
    inference_only=False,
):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = int(1000 / fps) if fps > 0 else 33

    print(f"Video: {width}x{height} @ {fps}fps  ({total_frames} frames)")

    # --- Trackers ---
    cyclist_tracker = pedestrian_tracker = None
    if not inference_only:
        if DeepSort is None:
            raise RuntimeError(
                "deep_sort_realtime is not installed. Install it or run with --inference-only."
            )
        cyclist_tracker = DeepSort(
            max_age=max_age, max_iou_distance=max_iou_distance,
            n_init=3, nms_max_overlap=nms_max_overlap, embedder="mobilenet",
        )
        pedestrian_tracker = DeepSort(
            max_age=max_age, max_iou_distance=max_iou_distance,
            n_init=3, nms_max_overlap=nms_max_overlap, embedder="mobilenet",
        )

    cyclist_ids_seen = set()
    pedestrian_ids_seen = set()

    # --- Video writer with codec fallbacks ---
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        raise RuntimeError(f"Output directory does not exist: {output_dir}")

    requested_ext = os.path.splitext(output_video_path)[1].lower()
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

    # --- Fixed overlay geometry ---
    font_scale, font_thickness, padding, line_spacing = 0.8, 2, 15, 5
    max_text_size, _ = cv2.getTextSize("Pedestrians: 999 (Total: 9999)", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    max_text_width = max(max_text_size[0], 350)
    sample_size, _ = cv2.getTextSize("Cyclists: 0 (Total: 0)", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_height = sample_size[1]
    text_x = width - max_text_width - padding
    text_y_pedestrian = height - padding
    text_y_cyclist = text_y_pedestrian - text_height - line_spacing
    bg_x1 = max(0, text_x - padding)
    bg_y1 = max(0, text_y_cyclist - text_height - padding)
    bg_x2 = min(width, text_x + max_text_width + padding)
    bg_y2 = min(height, text_y_pedestrian + padding)

    frame_count = 0
    paused = False
    speed_multiplier = 1.0
    annotated_frame = None
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
                class_filter = {0, 1}

                # Pass 1: full-frame detection
                full_dets = _extract_boxes(
                    _run_detector(model, proc_frame, confidence_threshold, iou_threshold, imgsz=imgsz),
                    class_filter=class_filter,
                )

                # Pass 2: optional top-region pass (catches distant objects)
                top_dets = []
                if top_region_pass:
                    top_conf = confidence_threshold if top_region_confidence is None else top_region_confidence
                    top_dets = _run_top_region_pass(
                        proc_frame, model, iou_threshold, top_region_ratio, top_conf, top_region_imgsz, class_filter
                    )

                # Pass 3: optional SAHI-style tiled pass
                tile_dets = []
                if tile_mode == "sahi":
                    tile_conf = confidence_threshold if tile_confidence is None else tile_confidence
                    tile_dets = _run_tiled_pass(
                        proc_frame, model, iou_threshold, tile_conf, tile_size, tile_overlap, tile_imgsz, class_filter
                    )

                # Merge and clip to proc_frame bounds
                merged_dets = []
                for det in full_dets + top_dets + tile_dets:
                    clipped = _clip_detection(det, frame_w=proc_w, frame_h=proc_h)
                    if clipped is not None:
                        merged_dets.append(clipped)

                # Scale detections back to original frame coordinates
                if scale_x != 1.0 or scale_y != 1.0:
                    merged_dets = [
                        [d[0] * scale_x, d[1] * scale_y, d[2] * scale_x, d[3] * scale_y, d[4], d[5]]
                        for d in merged_dets
                    ]

                # Hard NMS (always on): eliminates cross-pass/tile duplicate boxes
                merged_dets = _hard_nms_per_class(merged_dets, iou_threshold=nms_iou)

                # Optional soft-NMS crowd suppression
                processed_dets = _apply_crowd_postprocess(
                    merged_dets,
                    crowd_mode=crowd_mode,
                    soft_nms_iou=soft_nms_iou,
                    soft_nms_sigma=soft_nms_sigma,
                    score_threshold=confidence_threshold * 0.5,
                )

                annotated_frame = frame.copy()
                cyclist_count = 0
                pedestrian_count = 0

                if inference_only:
                    for x1, y1, x2, y2, conf, cls_int in processed_dets:
                        x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                        if cls_int == 0:
                            cyclist_count += 1
                            color, label, text_color = (0, 255, 0), f"Cyclist {conf:.2f}", (0, 0, 0)
                        elif cls_int == 1:
                            pedestrian_count += 1
                            color, label, text_color = (255, 0, 0), f"Pedestrian {conf:.2f}", (255, 255, 255)
                        else:
                            continue
                        cv2.rectangle(annotated_frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1_i, y1_i - label_size[1] - 10),
                                      (x1_i + label_size[0], y1_i), color, -1)
                        cv2.putText(annotated_frame, label, (x1_i, y1_i - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                else:
                    cyclist_dets, pedestrian_dets = _detections_to_tracker_inputs(processed_dets)
                    cyclist_tracks = cyclist_tracker.update_tracks(cyclist_dets, frame=frame) if cyclist_dets else []
                    pedestrian_tracks = pedestrian_tracker.update_tracks(pedestrian_dets, frame=frame) if pedestrian_dets else []

                    for track in [t for t in cyclist_tracks if t.is_confirmed()]:
                        track_id = track.track_id
                        x1, y1, x2, y2 = map(int, track.to_tlbr())
                        cyclist_ids_seen.add(track_id)
                        cyclist_count += 1
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Cyclist #{track_id}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                                      (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    for track in [t for t in pedestrian_tracks if t.is_confirmed()]:
                        track_id = track.track_id
                        x1, y1, x2, y2 = map(int, track.to_tlbr())
                        pedestrian_ids_seen.add(track_id)
                        pedestrian_count += 1
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"Pedestrian #{track_id}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                                      (x1 + label_size[0], y1), (255, 0, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if frame_count % 10 == 0:
                    msg = (
                        f"Frame {frame_count}: {cyclist_count} cyclist(s), {pedestrian_count} pedestrian(s) | "
                        f"Total unique: {len(cyclist_ids_seen)} cyclists, {len(pedestrian_ids_seen)} pedestrians"
                    )
                    if debug_detections:
                        msg += (
                            f" | full={len(full_dets)} top={len(top_dets)} "
                            f"tile={len(tile_dets)} merged={len(merged_dets)} final={len(processed_dets)}"
                        )
                    print(msg)

                if inference_only:
                    cyclist_text = f"Cyclists: {cyclist_count}"
                    pedestrian_text = f"Pedestrians: {pedestrian_count}"
                else:
                    cyclist_text = f"Cyclists: {cyclist_count} (Total: {len(cyclist_ids_seen)})"
                    pedestrian_text = f"Pedestrians: {pedestrian_count} (Total: {len(pedestrian_ids_seen)})"

                cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.putText(annotated_frame, cyclist_text, (text_x, text_y_cyclist),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                cv2.putText(annotated_frame, pedestrian_text, (text_x, text_y_pedestrian),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)
                out.write(annotated_frame)
                frame_count += 1
                progress_bar.update(1)

            if display_available and annotated_frame is not None:
                try:
                    cv2.imshow("RT-DETR + DeepSORT", annotated_frame)
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
            print(f"Total unique cyclists tracked: {len(cyclist_ids_seen)}")
            print(f"Total unique pedestrians tracked: {len(pedestrian_ids_seen)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RT-DETR + DeepSORT cyclist/pedestrian tracking.")
    parser.add_argument("--input", "-i", default="../trim6.mp4", help="Input video path.")
    parser.add_argument("--output", "-o", default="", help="Output video path.")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_PATH, help="RT-DETR .pt checkpoint path.")
    parser.add_argument("--confidence", "-c", type=float, default=0.65, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="Ultralytics NMS IoU threshold.")
    parser.add_argument("--max-age", type=int, default=15, help="DeepSort max frames to keep an unmatched track.")
    parser.add_argument("--max-iou-distance", type=float, default=0.6, help="DeepSort max IoU distance for track matching.")
    parser.add_argument("--no-display", action="store_true", default=True, help="Disable live preview window.")
    parser.add_argument("--imgsz", type=int, default=0, help="Inference resolution override. 0 = model default.")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="Hard NMS IoU threshold (pre-tracker duplicate removal).")
    parser.add_argument("--nms-max-overlap", type=float, default=0.7, help="DeepSort internal NMS overlap threshold.")
    parser.add_argument("--crowd-mode", choices=["off", "soft-nms"], default="soft-nms", help="Crowd suppression mode.")
    parser.add_argument("--soft-nms-iou", type=float, default=0.25, help="Soft-NMS IoU trigger threshold.")
    parser.add_argument("--soft-nms-sigma", type=float, default=0.2, help="Soft-NMS Gaussian penalty sigma.")
    parser.add_argument("--top-region-pass", action="store_true", default=True, help="Extra inference pass on upper frame region.")
    parser.add_argument("--top-region-ratio", type=float, default=0.45, help="Fraction of frame height used for top-region pass.")
    parser.add_argument("--top-region-imgsz", type=int, default=0, help="Top-region inference resolution. 0 = model default.")
    parser.add_argument("--top-region-confidence", type=float, default=-1.0, help="Top-region confidence. Negative = use --confidence.")
    parser.add_argument("--tile-mode", choices=["off", "sahi"], default="sahi", help="Tiled SAHI-style inference mode.")
    parser.add_argument("--tile-size", type=int, default=480, help="Tile size in pixels.")
    parser.add_argument("--tile-overlap", type=float, default=0.6, help="Tile overlap fraction (0.0-0.8).")
    parser.add_argument("--tile-imgsz", type=int, default=0, help="Tile inference resolution. 0 = model default.")
    parser.add_argument("--tile-confidence", type=float, default=-1.0, help="Tile confidence. Negative = use --confidence.")
    parser.add_argument("--downscale-width", type=int, default=640, help="Resize frame width before inference. 0 to disable.")
    parser.add_argument("--downscale-height", type=int, default=480, help="Resize frame height before inference. 0 to disable.")
    parser.add_argument("--debug-detections", action="store_true", help="Print per-pass detection counts every 10 frames.")
    parser.add_argument("--inference-only", action="store_true", help="Detector-only mode (no DeepSORT tracking).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not (0.0 <= args.confidence <= 1.0):
        raise ValueError("--confidence must be in [0.0, 1.0]")
    if not (0.0 < args.top_region_ratio <= 1.0):
        raise ValueError("--top-region-ratio must be in (0.0, 1.0]")
    if not (0.0 <= args.tile_overlap <= 0.8):
        raise ValueError("--tile-overlap must be in [0.0, 0.8]")

    if not args.output:
        base = os.path.splitext(args.input)[0]
        args.output = (
            f"{base}_rtdetr_inference.mp4" if args.inference_only else f"{base}_rtdetr_deepsort.mp4"
        )

    model = load_model(args.model, DEVICE)

    process_video(
        input_video_path=args.input,
        output_video_path=args.output,
        model=model,
        confidence_threshold=args.confidence,
        max_age=args.max_age,
        max_iou_distance=args.max_iou_distance,
        iou_threshold=args.iou,
        disable_display=args.no_display,
        imgsz=(args.imgsz if args.imgsz > 0 else None),
        crowd_mode=args.crowd_mode,
        soft_nms_iou=args.soft_nms_iou,
        soft_nms_sigma=args.soft_nms_sigma,
        top_region_pass=args.top_region_pass,
        top_region_ratio=args.top_region_ratio,
        top_region_imgsz=(args.top_region_imgsz if args.top_region_imgsz > 0 else None),
        top_region_confidence=(args.top_region_confidence if args.top_region_confidence >= 0 else None),
        tile_mode=args.tile_mode,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_imgsz=(args.tile_imgsz if args.tile_imgsz > 0 else None),
        tile_confidence=(args.tile_confidence if args.tile_confidence >= 0 else None),
        nms_iou=args.nms_iou,
        nms_max_overlap=args.nms_max_overlap,
        downscale_width=args.downscale_width,
        downscale_height=args.downscale_height,
        debug_detections=args.debug_detections,
        inference_only=args.inference_only,
    )


if __name__ == "__main__":
    main()
