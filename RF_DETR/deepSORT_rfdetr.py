from __future__ import annotations

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None

MODEL_MAP = {
    "RFDETRNano": RFDETRNano,
    "RFDETRSmall": RFDETRSmall,
    "RFDETRMedium": RFDETRMedium,
    "RFDETRLarge": RFDETRLarge,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_size, model_path):
    model_cls = MODEL_MAP[model_size]
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        print(f"Loading fine-tuned RF-DETR checkpoint: {model_path}")
        model = model_cls(pretrain_weights=model_path)
    else:
        print(f"Loading base RF-DETR model ({model_size})")
        model = model_cls()
    class_names = getattr(model, "class_names", None)
    if class_names:
        print(f"Model class names: {class_names}")
    print(f"Inference device: {DEVICE}")
    return model, class_names


def _run_detector(model, image_bgr, conf_threshold, imgsz=None):
    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    kwargs = {"threshold": conf_threshold}
    if imgsz:
        kwargs["resolution"] = imgsz
    return model.predict(pil_image, **kwargs)


def _extract_boxes(detections, x_offset=0, y_offset=0, class_filter=None):
    output = []
    if detections is None or not hasattr(detections, "xyxy"):
        return output
    xyxy = np.asarray(detections.xyxy)
    if xyxy.size == 0:
        return output
    class_ids = np.asarray(detections.class_id).flatten() if hasattr(detections, "class_id") else np.zeros(len(xyxy))
    confs = np.asarray(detections.confidence).flatten() if hasattr(detections, "confidence") else np.ones(len(xyxy))
    if xyxy.ndim == 1:
        xyxy = xyxy.reshape(1, -1)

    for i in range(len(xyxy)):
        cls_int = int(class_ids[i]) if i < len(class_ids) else 0
        if class_filter is not None and cls_int not in class_filter:
            continue
        x1, y1, x2, y2 = xyxy[i][:4]
        conf = float(confs[i]) if i < len(confs) else 1.0
        output.append(
            [
                float(x1 + x_offset),
                float(y1 + y_offset),
                float(x2 + x_offset),
                float(y2 + y_offset),
                conf,
                cls_int,
            ]
        )
    return output


def _clip_detection(det, frame_w, frame_h):
    x1, y1, x2, y2, conf, cls_int = det
    x1 = max(0.0, min(float(frame_w - 1), x1))
    y1 = max(0.0, min(float(frame_h - 1), y1))
    x2 = max(0.0, min(float(frame_w - 1), x2))
    y2 = max(0.0, min(float(frame_h - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2, conf, cls_int]


def _compute_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _soft_nms_per_class(detections, iou_threshold=0.5, sigma=0.5, score_threshold=1e-3):
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


def _hard_nms_per_class(detections, iou_threshold=0.45):
    """Greedy hard NMS per class. Always applied to remove cross-pass/tile duplicate boxes."""
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


def _apply_crowd_postprocess(detections, crowd_mode, soft_nms_iou, soft_nms_sigma, score_threshold):
    if crowd_mode != "soft-nms":
        return detections
    by_class = defaultdict(list)
    for det in detections:
        by_class[int(det[5])].append(det)
    merged = []
    for class_dets in by_class.values():
        merged.extend(
            _soft_nms_per_class(
                class_dets,
                iou_threshold=soft_nms_iou,
                sigma=soft_nms_sigma,
                score_threshold=score_threshold,
            )
        )
    return merged


def _run_top_region_pass(frame, model, top_region_ratio, conf_threshold, imgsz, class_filter):
    h, _ = frame.shape[:2]
    top_h = int(max(1, min(h, round(h * top_region_ratio))))
    roi = frame[:top_h, :]
    detections = _run_detector(model, roi, conf_threshold, imgsz=imgsz)
    return _extract_boxes(detections, x_offset=0, y_offset=0, class_filter=class_filter)


def _run_tiled_pass(frame, model, conf_threshold, tile_size, tile_overlap, imgsz, class_filter):
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
            dets = _run_detector(model, tile_img, conf_threshold, imgsz=imgsz)
            detections.extend(_extract_boxes(dets, x_offset=x0, y_offset=y0, class_filter=class_filter))
    return detections


def _detections_to_tracker_inputs(detections, cyclist_class_id, pedestrian_class_id):
    cyclist_detections = []
    pedestrian_detections = []
    for x1, y1, x2, y2, conf, cls_int in detections:
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        if cls_int == cyclist_class_id:
            cyclist_detections.append((bbox, float(conf), cls_int))
        elif cls_int == pedestrian_class_id:
            pedestrian_detections.append((bbox, float(conf), cls_int))
    return cyclist_detections, pedestrian_detections


def _resolve_base_coco_class_ids(class_names, cyclist_class_id, pedestrian_class_id):
    # COCO defaults are typically person=0, bicycle=1 in index-based APIs.
    if class_names:
        mapping = {str(name).lower(): idx for idx, name in enumerate(class_names)}
        bicycle_idx = mapping.get("bicycle", 1)
        person_idx = mapping.get("person", 0)
        return bicycle_idx, person_idx
    return cyclist_class_id, pedestrian_class_id


def process_video(
    input_video_path,
    output_video_path,
    model,
    class_names,
    confidence_threshold=0.65,
    max_age=30,
    max_iou_distance=0.7,
    disable_display=True,
    imgsz=None,
    crowd_mode="off",
    soft_nms_iou=0.4,
    soft_nms_sigma=0.3,
    top_region_pass=True,
    top_region_ratio=0.35,
    top_region_imgsz=None,
    top_region_confidence=None,
    tile_mode="off",
    tile_size=480,
    tile_overlap=0.5,
    tile_imgsz=None,
    tile_confidence=None,
    nms_iou=0.45,
    nms_max_overlap=0.7,
    downscale_width=640,
    downscale_height=480,
    debug_detections=False,
    cyclist_class_id=0,
    pedestrian_class_id=1,
    inference_only=False,
):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = int(1000 / fps) if fps > 0 else 33

    cyclist_tracker = None
    pedestrian_tracker = None
    if not inference_only:
        if DeepSort is None:
            raise RuntimeError(
                "deep_sort_realtime is not installed. Install it or run with --inference-only."
            )
        cyclist_tracker = DeepSort(max_age=max_age, max_iou_distance=max_iou_distance, n_init=3, nms_max_overlap=nms_max_overlap, embedder="mobilenet")
        pedestrian_tracker = DeepSort(max_age=max_age, max_iou_distance=max_iou_distance, n_init=3, nms_max_overlap=nms_max_overlap, embedder="mobilenet")

    cyclist_ids_seen = set()
    pedestrian_ids_seen = set()

    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_video_path}")

    frame_count = 0
    paused = False
    speed_multiplier = 1.0
    annotated_frame = None
    display_available = not disable_display

    font_scale = 0.8
    font_thickness = 2
    padding = 15
    line_spacing = 5
    max_possible_text = "Pedestrians: 999 (Total: 9999)"
    max_text_size, _ = cv2.getTextSize(max_possible_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    max_text_width = max(max_text_size[0], 350)
    sample_text_size, _ = cv2.getTextSize("Cyclists: 0 (Total: 0)", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_height = sample_text_size[1]
    text_x = width - max_text_width - padding
    text_y_pedestrian = height - padding
    text_y_cyclist = text_y_pedestrian - text_height - line_spacing
    bg_x1 = max(0, text_x - padding)
    bg_y1 = max(0, text_y_cyclist - text_height - padding)
    bg_x2 = min(width, text_x + max_text_width + padding)
    bg_y2 = min(height, text_y_pedestrian + padding)

    progress_bar_total = total_frames if total_frames > 0 else None
    progress_bar = tqdm(total=progress_bar_total, desc="Processing video", unit="frame")

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
                class_filter = {cyclist_class_id, pedestrian_class_id}

                full_dets = _extract_boxes(
                    _run_detector(model, proc_frame, confidence_threshold, imgsz=imgsz),
                    class_filter=class_filter,
                )

                top_dets = []
                if top_region_pass:
                    top_conf = confidence_threshold if top_region_confidence is None else top_region_confidence
                    top_dets = _run_top_region_pass(
                        proc_frame, model, top_region_ratio, top_conf, top_region_imgsz, class_filter
                    )

                tile_dets = []
                if tile_mode == "sahi":
                    tile_conf = confidence_threshold if tile_confidence is None else tile_confidence
                    tile_dets = _run_tiled_pass(
                        proc_frame, model, tile_conf, tile_size, tile_overlap, tile_imgsz, class_filter
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

                # Hard NMS always applied first to eliminate cross-pass/tile duplicate boxes
                merged_dets = _hard_nms_per_class(merged_dets, iou_threshold=nms_iou)

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
                        if cls_int == cyclist_class_id:
                            cyclist_count += 1
                            color = (0, 255, 0)
                            label = f"Cyclist {conf:.2f}"
                            text_color = (0, 0, 0)
                        elif cls_int == pedestrian_class_id:
                            pedestrian_count += 1
                            color = (255, 0, 0)
                            label = f"Pedestrian {conf:.2f}"
                            text_color = (255, 255, 255)
                        else:
                            continue
                        cv2.rectangle(annotated_frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(
                            annotated_frame,
                            (x1_i, y1_i - label_size[1] - 10),
                            (x1_i + label_size[0], y1_i),
                            color,
                            -1,
                        )
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1_i, y1_i - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            text_color,
                            2,
                        )
                else:
                    cyclist_detections, pedestrian_detections = _detections_to_tracker_inputs(
                        processed_dets,
                        cyclist_class_id=cyclist_class_id,
                        pedestrian_class_id=pedestrian_class_id,
                    )
                    cyclist_tracks = (
                        cyclist_tracker.update_tracks(cyclist_detections, frame=frame) if cyclist_detections else []
                    )
                    pedestrian_tracks = (
                        pedestrian_tracker.update_tracks(pedestrian_detections, frame=frame) if pedestrian_detections else []
                    )

                    for track in [t for t in cyclist_tracks if t.is_confirmed()]:
                        track_id = track.track_id
                        x1, y1, x2, y2 = map(int, track.to_tlbr())
                        cyclist_ids_seen.add(track_id)
                        cyclist_count += 1
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Cyclist #{track_id}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    for track in [t for t in pedestrian_tracks if t.is_confirmed()]:
                        track_id = track.track_id
                        x1, y1, x2, y2 = map(int, track.to_tlbr())
                        pedestrian_ids_seen.add(track_id)
                        pedestrian_count += 1
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"Pedestrian #{track_id}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (255, 0, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if frame_count % 10 == 0:
                    msg = (
                        f"Frame {frame_count}: {cyclist_count} cyclist(s), {pedestrian_count} pedestrian(s) | "
                        f"Total unique: {len(cyclist_ids_seen)} cyclists, {len(pedestrian_ids_seen)} pedestrians"
                    )
                    if debug_detections:
                        msg += (
                            f" | dets full={len(full_dets)} top={len(top_dets)} "
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
                cv2.putText(
                    annotated_frame, cyclist_text, (text_x, text_y_cyclist),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness
                )
                cv2.putText(
                    annotated_frame, pedestrian_text, (text_x, text_y_pedestrian),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness
                )
                out.write(annotated_frame)
                frame_count += 1
                progress_bar.update(1)

            if display_available and annotated_frame is not None:
                try:
                    cv2.imshow("RF-DETR + DeepSORT", annotated_frame)
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
                print(f"Frame saved as: {frame_filename}")
            elif key == ord("p"):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key in (ord("+"), ord("=")):
                speed_multiplier = min(5.0, speed_multiplier + 0.5)
                print(f"Speed increased to {speed_multiplier:.1f}x")
            elif key == ord("-"):
                speed_multiplier = max(0.1, speed_multiplier - 0.5)
                print(f"Speed decreased to {speed_multiplier:.1f}x")
    finally:
        progress_bar.close()
        cap.release()
        out.release()
        if display_available:
            try:
                cv2.destroyAllWindows()
            except (cv2.error, Exception):
                pass
        print(f"\nVideo processing completed: {output_video_path}")
        if inference_only:
            print("Inference-only mode complete (tracking disabled).")
        else:
            print(f"Total unique cyclists tracked: {len(cyclist_ids_seen)}")
            print(f"Total unique pedestrians tracked: {len(pedestrian_ids_seen)}")


def main():
    parser = argparse.ArgumentParser(description="Video tracking with RF-DETR + DeepSORT (cyclist/pedestrian).")
    parser.add_argument("--input", "-i", default="../trim_3.mp4", help="Input video path")
    parser.add_argument("--output", "-o", default="", help="Output video path")
    parser.add_argument("--model", "-m", default="RF_DETR/runs/rfdetr_finetune_res576_mos1.0_mix0.2_persp0.0008/checkpoint_best_total.pth", help="Fine-tuned RF-DETR checkpoint path. Empty uses base model.")
    parser.add_argument("--model-size", choices=list(MODEL_MAP.keys()), default="RFDETRSmall")
    parser.add_argument("--base-coco", action="store_true", help="Use COCO class mapping (bicycle/person) instead of custom ids.")
    parser.add_argument("--cyclist-class-id", type=int, default=0, help="Cyclist class id for fine-tuned model.")
    parser.add_argument("--pedestrian-class-id", type=int, default=1, help="Pedestrian class id for fine-tuned model.")
    parser.add_argument("--confidence", "-c", type=float, default=0.65)
    parser.add_argument("--max-age", type=int, default=15)
    parser.add_argument("--max-iou-distance", type=float, default=0.6)
    parser.add_argument("--no-display", action="store_true", default=True)
    parser.add_argument("--imgsz", type=int, default=0, help="Inference resolution override. 0 uses model default.")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="Hard NMS IoU threshold applied to all merged detections before tracking.")
    parser.add_argument("--nms-max-overlap", type=float, default=0.7, help="DeepSort internal NMS overlap threshold. Removes duplicate detections inside the tracker.")
    parser.add_argument("--crowd-mode", choices=["off", "soft-nms"], default="soft-nms")
    parser.add_argument("--soft-nms-iou", type=float, default=0.25)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.2)
    parser.add_argument("--top-region-pass", action="store_true", default=True)
    parser.add_argument("--top-region-ratio", type=float, default=0.45)
    parser.add_argument("--top-region-imgsz", type=int, default=0)
    parser.add_argument("--top-region-confidence", type=float, default=-1.0)
    parser.add_argument("--tile-mode", choices=["off", "sahi"], default="sahi")
    parser.add_argument("--tile-size", type=int, default=480)
    parser.add_argument("--tile-overlap", type=float, default=0.6)
    parser.add_argument("--tile-imgsz", type=int, default=0)
    parser.add_argument("--tile-confidence", type=float, default=-1.0)
    parser.add_argument("--downscale-width", type=int, default=640, help="Resize frame width before inference. 0 to disable.")
    parser.add_argument("--downscale-height", type=int, default=480, help="Resize frame height before inference. 0 to disable.")
    parser.add_argument("--debug-detections", action="store_true")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run detector-only mode (no DeepSORT tracking).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not (0.0 <= args.confidence <= 1.0):
        raise ValueError("--confidence must be in [0.0, 1.0]")
    if not (0.0 < args.top_region_ratio <= 1.0):
        raise ValueError("--top-region-ratio must be in (0.0, 1.0]")
    if not (0.0 <= args.tile_overlap <= 0.8):
        raise ValueError("--tile-overlap must be in [0.0, 0.8]")

    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = (
            f"{base_name}_rfdetr_inference.mp4"
            if args.inference_only
            else f"{base_name}_rfdetr_deepsort.mp4"
        )

    model, class_names = load_model(args.model_size, args.model.strip() or None)

    cyclist_class_id = args.cyclist_class_id
    pedestrian_class_id = args.pedestrian_class_id
    if args.base_coco:
        cyclist_class_id, pedestrian_class_id = _resolve_base_coco_class_ids(
            class_names, cyclist_class_id, pedestrian_class_id
        )
        print(
            f"COCO class mapping active -> cyclist(bicycle)={cyclist_class_id}, "
            f"pedestrian(person)={pedestrian_class_id}"
        )

    process_video(
        input_video_path=args.input,
        output_video_path=args.output,
        model=model,
        class_names=class_names,
        confidence_threshold=args.confidence,
        max_age=args.max_age,
        max_iou_distance=args.max_iou_distance,
        disable_display=args.no_display,
        imgsz=(args.imgsz if args.imgsz > 0 else None),
        nms_iou=args.nms_iou,
        nms_max_overlap=args.nms_max_overlap,
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
        downscale_width=args.downscale_width,
        downscale_height=args.downscale_height,
        debug_detections=args.debug_detections,
        cyclist_class_id=cyclist_class_id,
        pedestrian_class_id=pedestrian_class_id,
        inference_only=args.inference_only,
    )


if __name__ == "__main__":
    main()
