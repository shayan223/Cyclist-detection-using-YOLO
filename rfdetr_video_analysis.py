"""
RF-DETR video analysis: run inference on video using either the base (COCO) model
or a fine-tuned checkpoint (e.g. cyclist/pedestrian). Optional Deep SORT tracking.

Usage:
  Base model:      python rfdetr_video_analysis.py -i video.mp4
  Fine-tuned:      python rfdetr_video_analysis.py -i video.mp4 -m pdx_rfdetr/.../checkpoint_best_total.pth
  With tracking:   python rfdetr_video_analysis.py -i video.mp4 -m ... --track
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False

# --- Configuration ---
MODEL_SIZE = "RFDETRMedium"  # RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
DEFAULT_FINETUNED_PATH = "./pdx_rfdetr/pdx_rfdetr_finetune/checkpoint_best_total.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fine-tuned: data.yaml has names [cyclist, pedestrian] so 0=cyclist, 1=pedestrian.
# If your checkpoint uses the opposite order (0=pedestrian, 1=cyclist), use --swap-classes.
FINETUNED_CLASS_NAMES = ["cyclist", "pedestrian"]

# Base COCO: person=0, bicycle=1 (we map to pedestrian / cyclist)
COCO_PERSON_ID = 1
COCO_BICYCLE_ID = 0


def get_model_class(size: str):
    size_map = {
        "RFDETRNano": RFDETRNano,
        "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRLarge": RFDETRLarge,
    }
    if size not in size_map:
        raise ValueError(f"model_size must be one of {list(size_map.keys())}")
    return size_map[size]


def load_model(model_path: str | None, model_size: str = MODEL_SIZE, device: str = DEVICE):
    """Load RF-DETR: base (COCO) if model_path is None, else fine-tuned from checkpoint."""
    model_class = get_model_class(model_size)
    if model_path is None or model_path.strip() == "":
        print("Loading base RF-DETR model (COCO pretrained)...")
        model = model_class()
        use_finetuned = False
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Loading fine-tuned RF-DETR from: {model_path}")
        model = model_class(pretrain_weights=model_path)
        use_finetuned = True
    print(f"Model loaded on device: {device}")
    # Use model's class names if available (e.g. from checkpoint)
    class_names = None
    if hasattr(model, "class_names") and model.class_names:
        class_names = model.class_names
        print(f"Using model class names: {class_names}")
    return model, use_finetuned, class_names


def predict_frame(model, frame_bgr: np.ndarray, threshold: float = 0.5, use_finetuned: bool = True):
    """Run RF-DETR on a BGR frame; return detections as list of (class_id, confidence, (x1,y1,x2,y2))."""
    # RF-DETR expects RGB (PIL or numpy)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    detections = model.predict(pil_image, threshold=threshold)

    # supervision-style: detections.xyxy, .class_id, .confidence (numpy arrays)
    out = []
    if detections is None or (not hasattr(detections, "xyxy")) or len(detections.xyxy) == 0:
        return out
    xyxy = np.asarray(detections.xyxy)
    class_ids = np.asarray(detections.class_id).flatten() if hasattr(detections, "class_id") else np.zeros(len(xyxy), dtype=int)
    confidences = np.asarray(detections.confidence).flatten() if hasattr(detections, "confidence") else np.ones(len(xyxy))
    if xyxy.ndim == 1:
        xyxy = xyxy.reshape(1, -1)
    n = len(xyxy)
    if len(class_ids) < n:
        class_ids = np.resize(class_ids, n)
    if len(confidences) < n:
        confidences = np.resize(confidences, n)
    for i in range(n):
        x1, y1, x2, y2 = xyxy[i][:4]
        out.append((int(class_ids[i]), float(confidences[i]), (int(x1), int(y1), int(x2), int(y2))))
    return out


def _detections_to_cyclist_pedestrian(
    detections, use_finetuned, swap_classes
):
    """Split raw detections into (cyclist_detections, pedestrian_detections) for Deep SORT.
    Each list contains ([left, top, width, height], confidence, class_id) for Deep SORT.
    """
    cyclist_dets = []
    pedestrian_dets = []
    for cls, conf, (x1, y1, x2, y2) in detections:
        w, h = x2 - x1, y2 - y1
        bbox_ltwh = [x1, y1, w, h]
        if use_finetuned:
            idx = (cls - 1) if cls in (1, 2) else cls
            if idx not in (0, 1):
                continue
            is_cyclist = (idx == 1) if swap_classes else (idx == 0)
            is_pedestrian = (idx == 0) if swap_classes else (idx == 1)
        else:
            is_cyclist = cls == COCO_BICYCLE_ID
            is_pedestrian = cls == COCO_PERSON_ID
        if is_cyclist:
            cyclist_dets.append((bbox_ltwh, float(conf), cls))
        elif is_pedestrian:
            pedestrian_dets.append((bbox_ltwh, float(conf), cls))
    return cyclist_dets, pedestrian_dets


def process_video(
    input_video_path: str,
    output_video_path: str,
    model,
    use_finetuned: bool,
    confidence_threshold: float = 0.5,
    swap_classes: bool = False,
    class_names: list | None = None,
    debug: bool = False,
    use_track: bool = False,
    max_age: int = 30,
    max_iou_distance: float = 0.7,
    disable_display: bool = False,
):
    """Process video: run RF-DETR per frame; optionally Deep SORT tracking. Overlay boxes and live display."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = max(1, int(1000 / fps))

    print(f"Video: {width}x{height} @ {fps} fps, {total_frames} frames")
    print("Controls: q=quit, s=save frame, p=pause, +/- speed")

    # Deep SORT trackers (separate per class like deepSORT_yolo.py)
    cyclist_tracker = None
    pedestrian_tracker = None
    cyclist_ids_seen = set()
    pedestrian_ids_seen = set()
    if use_track:
        if not DEEPSORT_AVAILABLE:
            raise RuntimeError("Deep SORT requested but deep_sort_realtime not installed. pip install deep-sort-realtime")
        try:
            cyclist_tracker = DeepSort(max_age=max_age, max_iou_distance=max_iou_distance, n_init=2, embedder="mobilenet")
            pedestrian_tracker = DeepSort(max_age=max_age, max_iou_distance=max_iou_distance, n_init=2, embedder="mobilenet")
        except TypeError:
            cyclist_tracker = DeepSort(max_age=max_age, max_iou_distance=max_iou_distance, n_init=2)
            pedestrian_tracker = DeepSort(max_age=max_age, max_iou_distance=max_iou_distance, n_init=2)
        print("Deep SORT tracking enabled (separate trackers for cyclist/pedestrian)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    paused = False
    speed_multiplier = 1.0
    annotated_frame = np.zeros((height, width, 3), dtype=np.uint8)
    font_scale = 0.8
    font_thickness = 2
    text_x = width - 200
    bg_x1, bg_y1 = max(0, text_x - 15), max(0, height - 65)
    bg_x2, bg_y2 = min(width, text_x + 205), min(height, height + 15)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = predict_frame(model, frame, threshold=confidence_threshold, use_finetuned=use_finetuned)

                if debug and detections and frame_count < 5:
                    raw = [(c, conf) for c, conf, _ in detections[:5]]
                    print(f"  [debug] frame {frame_count} raw (class_id, conf): {raw}")

                annotated_frame = frame.copy()
                cyclist_count = 0
                pedestrian_count = 0

                if use_track and cyclist_tracker is not None:
                    cyclist_dets, pedestrian_dets = _detections_to_cyclist_pedestrian(
                        detections, use_finetuned, swap_classes
                    )
                    cyclist_tracks = cyclist_tracker.update_tracks(cyclist_dets, frame=frame) if cyclist_dets else []
                    pedestrian_tracks = pedestrian_tracker.update_tracks(pedestrian_dets, frame=frame) if pedestrian_dets else []
                    confirmed_cyclist = [t for t in cyclist_tracks if t.is_confirmed()]
                    confirmed_pedestrian = [t for t in pedestrian_tracks if t.is_confirmed()]

                    for track in confirmed_cyclist:
                        track_id = track.track_id
                        cyclist_ids_seen.add(track_id)
                        cyclist_count += 1
                        x1, y1, x2, y2 = map(int, track.to_tlbr())
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Cyclist #{track_id}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    for track in confirmed_pedestrian:
                        track_id = track.track_id
                        pedestrian_ids_seen.add(track_id)
                        pedestrian_count += 1
                        x1, y1, x2, y2 = map(int, track.to_tlbr())
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"Pedestrian #{track_id}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw, y1), (255, 0, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if frame_count % 10 == 0:
                        print(f"Frame {frame_count}: {cyclist_count} cyclist(s), {pedestrian_count} pedestrian(s) | "
                              f"Total unique: {len(cyclist_ids_seen)} cyclists, {len(pedestrian_ids_seen)} pedestrians")
                    cyclist_text = f"Cyclists: {cyclist_count} (Total: {len(cyclist_ids_seen)})"
                    pedestrian_text = f"Pedestrians: {pedestrian_count} (Total: {len(pedestrian_ids_seen)})"
                else:
                    # No tracking: draw detections directly
                    for cls, conf, (x1, y1, x2, y2) in detections:
                        if use_finetuned:
                            idx = (cls - 1) if cls in (1, 2) else cls
                            if idx not in (0, 1):
                                continue
                            is_cyclist = (idx == 1) if swap_classes else (idx == 0)
                            is_pedestrian = (idx == 0) if swap_classes else (idx == 1)
                        else:
                            is_cyclist = cls == COCO_BICYCLE_ID
                            is_pedestrian = cls == COCO_PERSON_ID
                        if not (is_cyclist or is_pedestrian):
                            continue
                        if is_cyclist:
                            cyclist_count += 1
                            color = (0, 255, 0)
                            label = f"cyclist: {conf:.2f}"
                        else:
                            pedestrian_count += 1
                            color = (255, 0, 0)
                            label = f"pedestrian: {conf:.2f}"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    print(f"Frame {frame_count}: {cyclist_count} cyclist(s), {pedestrian_count} pedestrian(s)")
                    cyclist_text = f"Cyclists: {cyclist_count}"
                    pedestrian_text = f"Pedestrians: {pedestrian_count}"

                cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.putText(annotated_frame, cyclist_text, (text_x, height - 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                cv2.putText(annotated_frame, pedestrian_text, (text_x, height - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)

                writer.write(annotated_frame)
                frame_count += 1
                if frame_count % 30 == 0 and total_frames > 0:
                    print(f"Progress: {frame_count}/{total_frames} ({100 * frame_count / total_frames:.1f}%)")

            if not disable_display:
                cv2.imshow("RF-DETR: Cyclist & Pedestrian Detection", annotated_frame)
            actual_delay = max(1, int(frame_delay / speed_multiplier)) if not disable_display else 1
            key = (cv2.waitKey(actual_delay) & 0xFF) if not disable_display else 0
            if key == ord("q"):
                break
            elif key == ord("s"):
                path = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(path, annotated_frame)
                print(f"Saved {path}")
            elif key == ord("p"):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key in (ord("+"), ord("=")):
                speed_multiplier = min(5.0, speed_multiplier + 0.5)
                print(f"Speed {speed_multiplier:.1f}x")
            elif key == ord("-"):
                speed_multiplier = max(0.1, speed_multiplier - 0.5)
                print(f"Speed {speed_multiplier:.1f}x")
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"Done. Output: {output_video_path}")
    if use_track:
        print(f"Total unique cyclists tracked: {len(cyclist_ids_seen)}, pedestrians: {len(pedestrian_ids_seen)}")


def main():
    parser = argparse.ArgumentParser(description="RF-DETR video analysis: base or fine-tuned model")
    parser.add_argument("--input", "-i", default="test.avi", help="Input video path")
    parser.add_argument("--output", "-o", help="Output video path (default: <input>_rfdetr.mp4)")
    parser.add_argument("--model", "-m", default=None, help="Fine-tuned checkpoint path (omit for base COCO model)")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="Confidence threshold (0–1)")
    parser.add_argument("--size", "-s", default=MODEL_SIZE, choices=["RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"], help="Model size (for base or checkpoint)")
    parser.add_argument("--swap-classes", action="store_true", help="Swap cyclist/pedestrian labels (use if fine-tuned model uses 0=pedestrian, 1=cyclist)")
    parser.add_argument("--debug", action="store_true", help="Print raw class_id for first few frames to verify indexing")
    parser.add_argument("--track", "-t", action="store_true", help="Enable Deep SORT tracking (persistent IDs per cyclist/pedestrian)")
    parser.add_argument("--max-age", type=int, default=30, help="Deep SORT: max frames to keep track without detection (default: 30)")
    parser.add_argument("--max-iou-distance", type=float, default=0.7, help="Deep SORT: max IOU distance for association (default: 0.7)")
    parser.add_argument("--no-display", action="store_true", help="Disable live display (faster when processing only)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input video not found: {args.input}")
        return

    out = args.output
    if out is None:
        base = Path(args.input).stem
        out = f"{base}_rfdetr_tracked.mp4" if args.track else f"{base}_rfdetr.mp4"

    model_path = args.model.strip() if args.model else None
    model, use_finetuned, class_names = load_model(model_path, model_size=args.size, device=DEVICE)
    print(f"Starting video analysis ({'fine-tuned' if use_finetuned else 'base COCO'})")
    if args.swap_classes and use_finetuned:
        print("Swap-classes enabled: 0=pedestrian, 1=cyclist")
    if args.track:
        print("Deep SORT tracking enabled (separate trackers for cyclist/pedestrian)")
    print(f"Input: {args.input}  Output: {out}  Confidence: {args.confidence}")
    process_video(
        args.input,
        out,
        model,
        use_finetuned,
        confidence_threshold=args.confidence,
        swap_classes=args.swap_classes,
        class_names=class_names,
        debug=args.debug,
        use_track=args.track,
        max_age=args.max_age,
        max_iou_distance=args.max_iou_distance,
        disable_display=args.no_display,
    )


if __name__ == "__main__":
    main()
