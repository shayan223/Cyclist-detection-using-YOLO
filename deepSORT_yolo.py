import cv2
import argparse
import os
from ultralytics import YOLO, RTDETR
import torch
import numpy as np
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# --- Configuration ---
EXPERIMENT_NAME = 'pdx_rtdetr_finetune3'#'rtdetr_finetune4'#'yolo_finetune'
CONFIG_FILE_PATH = './training_data/dataset.yaml'#'./training_data/config.yaml'
#MODEL_PATH = './cyclist_detection_yolo8/'+EXPERIMENT_NAME+'/weights/best.pt' #'yolov8l.pt'  # Base YOLO model
BATCH = 8
DEFAULT_MODEL_PATH = './pdx_rtdetr/'+EXPERIMENT_NAME+'/weights/best.pt' #'yolov8l.pt'  # Base YOLO model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
# Class names for COCO dataset (YOLO pretrained models)
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
'''

def _is_rtdetr_path(model_path):
    """Return True if path suggests an RT-DETR model (e.g. pdx_rtdetr, rtdetr_finetune)."""
    path_lower = model_path.replace('\\', '/').lower()
    return 'rtdetr' in path_lower


def load_model(model_path, device, use_rtdetr=None):
    """Load YOLO or RT-DETR model. If use_rtdetr is None, auto-detect from path."""
    if use_rtdetr is None:
        use_rtdetr = _is_rtdetr_path(model_path)
    if use_rtdetr:
        print(f"Loading RT-DETR model from: {model_path}")
        model = RTDETR(model_path)
    else:
        print(f"Loading YOLO model from: {model_path}")
        model = YOLO(model_path)
    model.to(device)
    print(f"Model loaded successfully on device: {device}")
    return model


def _run_detector(model, image, conf_threshold, iou_threshold, imgsz=None):
    """Run one detector pass and return raw Ultralytics results."""
    kwargs = {
        "conf": conf_threshold,
        "iou": iou_threshold,
        "agnostic_nms": False,
        "verbose": False,
    }
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    return model(image, **kwargs)


def _extract_boxes(results, x_offset=0, y_offset=0, class_filter=None):
    """Extract [x1, y1, x2, y2, conf, cls] from Ultralytics result list."""
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        boxes_conf = boxes.conf.cpu().numpy()
        boxes_cls = boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, boxes_conf, boxes_cls):
            cls_int = int(cls)
            if class_filter is not None and cls_int not in class_filter:
                continue
            detections.append([
                float(x1 + x_offset),
                float(y1 + y_offset),
                float(x2 + x_offset),
                float(y2 + y_offset),
                float(conf),
                cls_int,
            ])
    return detections


def _clip_detection(det, frame_w, frame_h):
    """Clip detection to frame bounds; return None if invalid."""
    x1, y1, x2, y2, conf, cls_int = det
    x1 = max(0.0, min(float(frame_w - 1), x1))
    y1 = max(0.0, min(float(frame_h - 1), y1))
    x2 = max(0.0, min(float(frame_w - 1), x2))
    y2 = max(0.0, min(float(frame_h - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2, conf, cls_int]


def _compute_iou_xyxy(a, b):
    """Compute IoU for two [x1,y1,x2,y2,...] detections."""
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
    """
    Linear+Gaussian Soft-NMS on one class list.
    Input/output format: [x1, y1, x2, y2, conf, cls].
    """
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
                # Strongly-overlapping boxes receive heavier confidence decay.
                det[4] *= np.exp(-(iou * iou) / max(sigma, 1e-6))
            if det[4] >= score_threshold:
                survivors.append(det)
        dets = survivors
    return kept


def _apply_crowd_postprocess(detections, crowd_mode, soft_nms_iou, soft_nms_sigma, score_threshold):
    """Apply crowd-scene suppression strategy."""
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


def _run_top_region_pass(frame, model, iou_threshold, top_region_ratio, conf_threshold, imgsz, class_filter):
    """Run detection on upper region and map boxes back to full-frame coordinates."""
    h, _ = frame.shape[:2]
    top_h = int(max(1, min(h, round(h * top_region_ratio))))
    roi = frame[:top_h, :]
    results = _run_detector(model, roi, conf_threshold, iou_threshold, imgsz=imgsz)
    return _extract_boxes(results, x_offset=0, y_offset=0, class_filter=class_filter)


def _run_tiled_pass(frame, model, iou_threshold, conf_threshold, tile_size, tile_overlap, imgsz, class_filter):
    """Run a SAHI-style tiled inference pass and merge tiles in full-frame coords."""
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


def _detections_to_tracker_inputs(detections):
    """Split [xyxy, conf, cls] detections into per-class DeepSORT input format."""
    cyclist_detections = []
    pedestrian_detections = []
    for x1, y1, x2, y2, conf, cls_int in detections:
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        if cls_int == 0:
            cyclist_detections.append((bbox, float(conf), cls_int))
        elif cls_int == 1:
            pedestrian_detections.append((bbox, float(conf), cls_int))
    return cyclist_detections, pedestrian_detections


def process_video(
    input_video_path,
    output_video_path,
    model,
    confidence_threshold=0.9,
    max_age=30,
    max_iou_distance=0.7,
    iou_threshold=0.1,
    disable_display=False,
    imgsz=None,
    crowd_mode="off",
    soft_nms_iou=0.5,
    soft_nms_sigma=0.5,
    top_region_pass=False,
    top_region_ratio=0.35,
    top_region_imgsz=None,
    top_region_confidence=None,
    tile_mode="off",
    tile_size=960,
    tile_overlap=0.2,
    tile_imgsz=None,
    tile_confidence=None,
    debug_detections=False,
):
    """Process video: detect cyclists/pedestrians (YOLO or RT-DETR), track with DeepSORT, overlay boxes."""
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {input_video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame delay for natural playback
    frame_delay = int(1000 / fps) if fps > 0 else 33  # Convert FPS to milliseconds per frame
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Frame delay: {frame_delay}ms")
    print(f"\nControls:")
    print(f"  Press 'q' to quit")
    print(f"  Press 's' to save current frame")
    print(f"  Press 'p' to pause/resume")
    print(f"  Press '+' to increase speed")
    print(f"  Press '-' to decrease speed")
    
    # Initialize DeepSORT tracker
    # Create separate trackers for cyclists and pedestrians
    # n_init=2 reduces computation (fewer frames needed to confirm track)
    # embedder='mobilenet' uses lighter feature extractor (faster than default)
    try:
        # Try with mobilenet embedder for better performance
        cyclist_tracker = DeepSort(
            max_age=max_age, 
            max_iou_distance=max_iou_distance, 
            n_init=2,  # Reduced from 3 for faster track confirmation
            embedder='mobilenet'  # Lighter feature extractor for better performance
        )
        pedestrian_tracker = DeepSort(
            max_age=max_age, 
            max_iou_distance=max_iou_distance, 
            n_init=2,
            embedder='mobilenet'
        )
    except TypeError:
        # Fallback if embedder parameter not supported in this version
        cyclist_tracker = DeepSort(
            max_age=max_age, 
            max_iou_distance=max_iou_distance, 
            n_init=2
        )
        pedestrian_tracker = DeepSort(
            max_age=max_age, 
            max_iou_distance=max_iou_distance, 
            n_init=2
        )
    
    # Track unique IDs seen across the video
    cyclist_ids_seen = set()
    pedestrian_ids_seen = set()
    
    # Setup video writer with extension+codec fallbacks (Windows OpenCV can fail with no extension/H264).
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    requested_ext = os.path.splitext(output_video_path)[1].lower()
    requested_base = os.path.splitext(output_video_path)[0] if requested_ext else output_video_path

    writer_attempts = []
    if requested_ext in ("", ".mp4"):
        writer_attempts.extend([
            (f"{requested_base}.mp4", "mp4v"),
            (f"{requested_base}.mp4", "avc1"),
            (f"{requested_base}.mp4", "H264"),
            (f"{requested_base}.avi", "XVID"),
            (f"{requested_base}.avi", "MJPG"),
        ])
    elif requested_ext == ".avi":
        writer_attempts.extend([
            (output_video_path, "XVID"),
            (output_video_path, "MJPG"),
            (f"{requested_base}.mp4", "mp4v"),
        ])
    else:
        # Unknown extension: try requested path first, then common containers.
        writer_attempts.extend([
            (output_video_path, "mp4v"),
            (f"{requested_base}.mp4", "mp4v"),
            (f"{requested_base}.avi", "XVID"),
        ])

    out = None
    final_output_video_path = output_video_path
    for candidate_path, codec_name in writer_attempts:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        out = cv2.VideoWriter(candidate_path, fourcc, fps, (width, height))
        if out.isOpened():
            final_output_video_path = candidate_path
            print(f"VideoWriter using codec: {codec_name}")
            if final_output_video_path != output_video_path:
                print(f"Output path adjusted to: {final_output_video_path}")
            break
        out.release()
        out = None

    if out is None:
        attempted = ", ".join([f"{path} ({codec})" for path, codec in writer_attempts])
        raise RuntimeError(
            f"Could not create VideoWriter. Tried: {attempted}. "
            "Try installing OpenH264 (https://github.com/cisco/openh264/releases) or use a different output path."
        )
    
    frame_count = 0
    paused = False
    speed_multiplier = 1.0  # Speed control (1.0 = normal speed)
    annotated_frame = None  # Initialize to avoid undefined variable
    display_available = not disable_display  # May be set False if imshow fails (no GUI in OpenCV)
    
    # Calculate fixed text overlay dimensions ONCE to prevent shifting
    font_scale = 0.8
    font_thickness = 2
    padding = 15
    line_spacing = 5
    
    # Calculate max width for worst-case scenario text
    max_possible_text = "Pedestrians: 999 (Total: 9999)"
    max_text_size, _ = cv2.getTextSize(max_possible_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    max_text_width = max(max_text_size[0], 350)  # Ensure minimum width
    
    # Calculate fixed text height (use a sample text to get consistent height)
    sample_text = "Cyclists: 0 (Total: 0)"
    sample_text_size, _ = cv2.getTextSize(sample_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_height = sample_text_size[1]  # Fixed height for all text
    
    # Fixed X position (from right edge)
    text_x = width - max_text_width - padding
    
    # Fixed Y positions (from bottom)
    text_y_pedestrian = height - padding  # Bottom line
    text_y_cyclist = text_y_pedestrian - text_height - line_spacing  # Top line
    
    # Fixed background rectangle bounds
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
                class_filter = {0, 1}

                # Pass 1: full-frame detection.
                full_results = _run_detector(model, frame, confidence_threshold, iou_threshold, imgsz=imgsz)
                full_dets = _extract_boxes(full_results, class_filter=class_filter)

                # Pass 2: optional top-region high-resolution pass.
                top_dets = []
                if top_region_pass:
                    top_conf = confidence_threshold if top_region_confidence is None else top_region_confidence
                    top_dets = _run_top_region_pass(
                        frame,
                        model,
                        iou_threshold=iou_threshold,
                        top_region_ratio=top_region_ratio,
                        conf_threshold=top_conf,
                        imgsz=top_region_imgsz,
                        class_filter=class_filter,
                    )

                # Pass 3: optional SAHI-style tiled pass.
                tile_dets = []
                if tile_mode == "sahi":
                    tile_conf = confidence_threshold if tile_confidence is None else tile_confidence
                    tile_dets = _run_tiled_pass(
                        frame,
                        model,
                        iou_threshold=iou_threshold,
                        conf_threshold=tile_conf,
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                        imgsz=tile_imgsz,
                        class_filter=class_filter,
                    )

                merged_dets = []
                for det in full_dets + top_dets + tile_dets:
                    clipped = _clip_detection(det, frame_w=frame_w, frame_h=frame_h)
                    if clipped is not None:
                        merged_dets.append(clipped)

                processed_dets = _apply_crowd_postprocess(
                    merged_dets,
                    crowd_mode=crowd_mode,
                    soft_nms_iou=soft_nms_iou,
                    soft_nms_sigma=soft_nms_sigma,
                    score_threshold=confidence_threshold * 0.25,
                )

                cyclist_detections, pedestrian_detections = _detections_to_tracker_inputs(processed_dets)
                
                # Update trackers only if there are detections (saves computation)
                cyclist_tracks = cyclist_tracker.update_tracks(cyclist_detections, frame=frame) if cyclist_detections else []
                pedestrian_tracks = pedestrian_tracker.update_tracks(pedestrian_detections, frame=frame) if pedestrian_detections else []
                
                # Process and draw tracked objects
                # Copy frame only when needed (for video writing, we need a copy to avoid modifying original)
                annotated_frame = frame.copy()  # Need copy for video writer
                cyclist_count = 0  # Counter for cyclists in current frame
                pedestrian_count = 0  # Counter for pedestrians in current frame
                
                # Pre-filter confirmed tracks to avoid repeated is_confirmed() calls
                confirmed_cyclist_tracks = [t for t in cyclist_tracks if t.is_confirmed()]
                confirmed_pedestrian_tracks = [t for t in pedestrian_tracks if t.is_confirmed()]
                
                # Draw cyclist tracks - batch operations
                for track in confirmed_cyclist_tracks:
                    track_id = track.track_id
                    # Use to_tlbr() which returns [x1, y1, x2, y2] format
                    tlbr = track.to_tlbr()
                    x1, y1, x2, y2 = int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3])
                    
                    cyclist_ids_seen.add(track_id)
                    cyclist_count += 1
                    
                    # Draw bounding box (green for cyclists)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with track ID - cache label string
                    label = f"Cyclist #{track_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    label_y = y1 - 5
                    label_bg_y1 = y1 - label_size[1] - 10
                    cv2.rectangle(annotated_frame, (x1, label_bg_y1), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, label_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Draw pedestrian tracks - batch operations
                for track in confirmed_pedestrian_tracks:
                    track_id = track.track_id
                    # Use to_tlbr() which returns [x1, y1, x2, y2] format
                    tlbr = track.to_tlbr()
                    x1, y1, x2, y2 = int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3])
                    
                    pedestrian_ids_seen.add(track_id)
                    pedestrian_count += 1
                    
                    # Draw bounding box (blue for pedestrians)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Draw label with track ID - cache label string
                    label = f"Pedestrian #{track_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    label_y = y1 - 5
                    label_bg_y1 = y1 - label_size[1] - 10
                    cv2.rectangle(annotated_frame, (x1, label_bg_y1), 
                                (x1 + label_size[0], y1), (255, 0, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, label_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Print counts to console (less frequently for better performance)
                if frame_count % 10 == 0:  # Print every 10 frames instead of every frame
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
                
                # Draw counts on video (lower right corner)
                cyclist_text = f"Cyclists: {cyclist_count} (Total: {len(cyclist_ids_seen)})"
                pedestrian_text = f"Pedestrians: {pedestrian_count} (Total: {len(pedestrian_ids_seen)})"
                
                # Draw background rectangle for better visibility (using pre-calculated fixed bounds)
                cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                
                # Draw cyclist count (green)
                cv2.putText(annotated_frame, cyclist_text, (text_x, text_y_cyclist), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                
                # Draw pedestrian count (blue)
                cv2.putText(annotated_frame, pedestrian_text, (text_x, text_y_pedestrian), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                frame_count += 1
                progress_bar.update(1)
            
            # Display the frame (only if we have a frame to display and display is enabled)
            if display_available and annotated_frame is not None:
                try:
                    cv2.imshow('Cyclist & Pedestrian Detection - Live View', annotated_frame)
                except (cv2.error, Exception):
                    display_available = False
                    print("Display not available (OpenCV built without GUI). Use --no-display to avoid this. Continuing without live view.")
            
            # Calculate actual delay based on speed multiplier
            actual_delay = max(1, int(frame_delay / speed_multiplier)) if display_available else 1
            
            # Handle keyboard input (only if display is enabled)
            key = cv2.waitKey(actual_delay) & 0xFF if display_available else 0
            if key == ord('q'):
                print("\nQuit requested by user")
                break
            elif key == ord('s'):
                # Save current frame
                frame_filename = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(frame_filename, annotated_frame)
                print(f"Frame saved as: {frame_filename}")
            elif key == ord('p'):
                paused = not paused
                status = "Paused" if paused else "Resumed"
                print(f"Video {status}")
            elif key == ord('+') or key == ord('='):
                speed_multiplier = min(5.0, speed_multiplier + 0.5)
                print(f"Speed increased to {speed_multiplier:.1f}x")
            elif key == ord('-'):
                speed_multiplier = max(0.1, speed_multiplier - 0.5)
                print(f"Speed decreased to {speed_multiplier:.1f}x")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        # Clean up
        progress_bar.close()
        cap.release()
        out.release()
        if display_available:
            try:
                cv2.destroyAllWindows()
            except (cv2.error, Exception):
                pass
        print(f"\nVideo processing completed!")
        print(f"Output saved to: {final_output_video_path}")
        print(f"Total unique cyclists tracked: {len(cyclist_ids_seen)}")
        print(f"Total unique pedestrians tracked: {len(pedestrian_ids_seen)}")

def main():
    parser = argparse.ArgumentParser(description='Analyze video for cyclist and pedestrian tracking using YOLO or RT-DETR + DeepSORT')
    parser.add_argument('--input', '-i', required=False, help='Input video file path', default='short_test_2.mp4')
    parser.add_argument('--output', '-o', help='Output video file path (default: input_tracked.mp4)')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL_PATH, help='Model path (YOLO or RT-DETR .pt)')
    parser.add_argument('--yolo', action='store_true', help='Force YOLO backend (default: auto-detect from path)')
    parser.add_argument('--rtdetr', action='store_true', help='Force RT-DETR backend (default: auto-detect from path)')
    parser.add_argument('--confidence', '-c', type=float, default=0.65, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU threshold (0.0-1.0). Lower values allow more overlapping detections. Default: 0.7')
    parser.add_argument('--max-age', type=int, default=30, help='Maximum frames to keep a track without update')
    parser.add_argument('--max-iou-distance', type=float, default=0.7, help='Maximum IOU distance for track association')
    parser.add_argument('--no-display', action='store_true', default=True,help='Disable live display (faster processing)')
    parser.add_argument('--imgsz', type=int, default=0, help='Inference image size. 0 uses model default.')
    parser.add_argument('--crowd-mode', choices=['off', 'soft-nms'], default='off', help='Crowd-scene postprocess mode.')
    parser.add_argument('--soft-nms-iou', type=float, default=0.4, help='Soft-NMS IoU threshold.')
    parser.add_argument('--soft-nms-sigma', type=float, default=0.3, help='Soft-NMS Gaussian sigma.')
    parser.add_argument('--top-region-pass', action='store_true', default=True, help='Run second high-res pass on upper image region.')
    parser.add_argument('--top-region-ratio', type=float, default=0.35, help='Upper region ratio for second pass (0.0-1.0).')
    parser.add_argument('--top-region-imgsz', type=int, default=0, help='Top-region inference size. 0 uses --imgsz/model default.')
    parser.add_argument('--top-region-confidence', type=float, default=-1.0, help='Top-region confidence threshold. Negative 1.0 uses --confidence.')
    parser.add_argument('--tile-mode', choices=['off', 'sahi'], default='off', help='Optional tiled inference mode.')
    parser.add_argument('--tile-size', type=int, default=480, help='Tile size (pixels) for tiled inference.')
    parser.add_argument('--tile-overlap', type=float, default=0.5, help='Tile overlap fraction (0.0-0.8).')
    parser.add_argument('--tile-imgsz', type=int, default=0, help='Tile inference image size. 0 uses --imgsz/model default.')
    parser.add_argument('--tile-confidence', type=float, default=-1.0, help='Tile confidence threshold. Negative uses --confidence.')
    parser.add_argument('--debug-detections', action='store_true', help='Print per-pass detection counters every 10 frames.')
    
    args = parser.parse_args()

    if not (0.0 <= args.confidence <= 1.0):
        print("Error: --confidence must be in [0.0, 1.0]")
        return
    if not (0.0 <= args.iou <= 1.0):
        print("Error: --iou must be in [0.0, 1.0]")
        return
    if not (0.0 < args.top_region_ratio <= 1.0):
        print("Error: --top-region-ratio must be in (0.0, 1.0]")
        return
    if not (0.0 <= args.tile_overlap <= 0.8):
        print("Error: --tile-overlap must be in [0.0, 0.8]")
        return
    
    # Resolve model type: explicit flag overrides auto-detect
    use_rtdetr = None
    if args.rtdetr:
        use_rtdetr = True
    if args.yolo:
        use_rtdetr = False
    if args.yolo and args.rtdetr:
        print("Warning: both --yolo and --rtdetr given; using --rtdetr")
        use_rtdetr = True
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input video file '{args.input}' not found")
        return
    
    # Set output path if not provided
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_DEEPSort_tracked.mp4"
    
    # Validate model file
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return
    
    # Load model (YOLO or RT-DETR; auto-detect from path if not specified)
    model = load_model(args.model, DEVICE, use_rtdetr=use_rtdetr)
    
    # Process video
    print(f"Starting video analysis with DeepSORT tracking...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"NMS IoU threshold: {args.iou}")
    print(f"Inference imgsz: {'model default' if args.imgsz <= 0 else args.imgsz}")
    print(f"Crowd mode: {args.crowd_mode}")
    if args.crowd_mode == 'soft-nms':
        print(f"Soft-NMS: iou={args.soft_nms_iou}, sigma={args.soft_nms_sigma}")
    print(f"Top-region pass: {'Enabled' if args.top_region_pass else 'Disabled'}")
    if args.top_region_pass:
        top_imgsz = args.top_region_imgsz if args.top_region_imgsz > 0 else (args.imgsz if args.imgsz > 0 else 'model default')
        top_conf = args.top_region_confidence if args.top_region_confidence >= 0 else args.confidence
        print(f"Top-region settings: ratio={args.top_region_ratio}, imgsz={top_imgsz}, conf={top_conf}")
    print(f"Tile mode: {args.tile_mode}")
    if args.tile_mode == 'sahi':
        tile_imgsz = args.tile_imgsz if args.tile_imgsz > 0 else (args.imgsz if args.imgsz > 0 else 'model default')
        tile_conf = args.tile_confidence if args.tile_confidence >= 0 else args.confidence
        print(f"Tile settings: size={args.tile_size}, overlap={args.tile_overlap}, imgsz={tile_imgsz}, conf={tile_conf}")
    print(f"Max age: {args.max_age} frames")
    print(f"Max IOU distance: {args.max_iou_distance}")
    print(f"Display: {'Disabled' if args.no_display else 'Enabled'}")
    
    process_video(
        args.input,
        args.output,
        model,
        args.confidence,
        args.max_age,
        args.max_iou_distance,
        args.iou,
        args.no_display,
        imgsz=(args.imgsz if args.imgsz > 0 else None),
        crowd_mode=args.crowd_mode,
        soft_nms_iou=args.soft_nms_iou,
        soft_nms_sigma=args.soft_nms_sigma,
        top_region_pass=args.top_region_pass,
        top_region_ratio=args.top_region_ratio,
        top_region_imgsz=(args.top_region_imgsz if args.top_region_imgsz > 0 else (args.imgsz if args.imgsz > 0 else None)),
        top_region_confidence=(args.top_region_confidence if args.top_region_confidence >= 0 else None),
        tile_mode=args.tile_mode,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_imgsz=(args.tile_imgsz if args.tile_imgsz > 0 else (args.imgsz if args.imgsz > 0 else None)),
        tile_confidence=(args.tile_confidence if args.tile_confidence >= 0 else None),
        debug_detections=args.debug_detections,
    )

if __name__ == "__main__":
    main()
