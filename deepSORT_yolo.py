import cv2
import argparse
import os
from ultralytics import YOLO, RTDETR
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Configuration ---
EXPERIMENT_NAME = 'pdx_rtdetr_finetune3'#'rtdetr_finetune4'#'yolo_finetune'
#MODEL_PATH = './cyclist_detection_yolo8/'+EXPERIMENT_NAME+'/weights/best.pt' #'yolov8l.pt'  # Base YOLO model
BATCH = 8
DEFAULT_MODEL_PATH = './50epoch_yolo_finetune_pdx3/weights/best.pt'#'./pdx_rtdetr/'+EXPERIMENT_NAME+'/weights/best.pt' #'yolov8l.pt'  # Base YOLO model
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

def process_video(input_video_path, output_video_path, model, confidence_threshold=0.9, max_age=30, max_iou_distance=0.7, iou_threshold=0.1, disable_display=False):
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
    
    # Setup video writer - try codecs in order (H264 often fails on Windows without OpenH264)
    ext = os.path.splitext(output_video_path)[1].lower()
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # mp4v works on most systems
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
    ]
    out = None
    for codec_name, fourcc in codecs_to_try:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"VideoWriter using codec: {codec_name}")
            break
        out.release()
        out = None
    if out is None:
        raise RuntimeError(
            "Could not create VideoWriter. Try installing OpenH264 or use a different output path. "
            "See: https://github.com/cisco/openh264/releases"
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
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection (same API for YOLO and RT-DETR)
                # Lower iou threshold (default 0.45) to prevent NMS from suppressing overlapping detections
                # between different classes (cyclists and pedestrians)
                results = model(frame, conf=confidence_threshold, iou=iou_threshold, agnostic_nms=False)
                
                # Prepare detections for tracking - pre-allocate with estimated capacity
                # Batch CPU-GPU transfers for better performance
                boxes_data = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        # Batch convert all boxes at once (more efficient than per-box conversion)
                        boxes_xyxy = boxes.xyxy.cpu().numpy()
                        boxes_conf = boxes.conf.cpu().numpy()
                        boxes_cls = boxes.cls.cpu().numpy()
                        boxes_data.extend(zip(boxes_xyxy, boxes_conf, boxes_cls))
                
                # Pre-allocate lists with estimated capacity (Python lists grow dynamically, but this helps)
                cyclist_detections = []
                pedestrian_detections = []
                
                # Extract detections - optimized conversion
                for (x1, y1, x2, y2), conf, cls in boxes_data:
                    cls_int = int(cls)
                    # DeepSORT expects format: ([left, top, width, height], confidence, class_id)
                    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    
                    if cls_int == 0:  # cyclist
                        cyclist_detections.append((bbox, float(conf), cls_int))
                    elif cls_int == 1:  # pedestrian
                        pedestrian_detections.append((bbox, float(conf), cls_int))
                
                # CRITICAL: Update trackers on EVERY frame, even with empty detections
                # This ensures tracks age properly and are removed when max_age is exceeded
                # Without this, tracks won't be removed correctly and tracking continuity suffers
                cyclist_tracks = cyclist_tracker.update_tracks(cyclist_detections, frame=frame)
                pedestrian_tracks = pedestrian_tracker.update_tracks(pedestrian_detections, frame=frame)
                
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
                    print(f"Frame {frame_count}: {cyclist_count} cyclist(s), {pedestrian_count} pedestrian(s) | "
                          f"Total unique: {len(cyclist_ids_seen)} cyclists, {len(pedestrian_ids_seen)} pedestrians")
                
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
                if frame_count % 30 == 0:  # Print progress every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
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
        cap.release()
        out.release()
        if display_available:
            try:
                cv2.destroyAllWindows()
            except (cv2.error, Exception):
                pass
        print(f"\nVideo processing completed!")
        print(f"Output saved to: {output_video_path}")
        print(f"Frames processed: {frame_count}/{total_frames}")
        if frame_count == total_frames:
            print(f"✓ All frames processed successfully")
        else:
            print(f"⚠ Warning: Expected {total_frames} frames but processed {frame_count} frames")
        print(f"Total unique cyclists tracked: {len(cyclist_ids_seen)}")
        print(f"Total unique pedestrians tracked: {len(pedestrian_ids_seen)}")

def main():
    parser = argparse.ArgumentParser(description='Analyze video for cyclist and pedestrian tracking using YOLO or RT-DETR + DeepSORT')
    parser.add_argument('--input', '-i', required=False, help='Input video file path', default='trim4.mp4')
    parser.add_argument('--output', '-o', help='Output video file path (default: input_tracked.mp4)')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL_PATH, help='Model path (YOLO or RT-DETR .pt)')
    parser.add_argument('--yolo', action='store_true', help='Force YOLO backend (default: auto-detect from path)')
    parser.add_argument('--rtdetr', action='store_true', help='Force RT-DETR backend (default: auto-detect from path)')
    parser.add_argument('--confidence', '-c', type=float, default=0.70, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.3, help='NMS IoU threshold (0.0-1.0). Lower values allow more overlapping detections. Default: 0.3')
    parser.add_argument('--max-age', type=int, default=25, help='Maximum frames to keep a track without update')
    parser.add_argument('--max-iou-distance', type=float, default=0.7, help='Maximum IOU distance for track association')
    parser.add_argument('--no-display', action='store_true', help='Disable live display (faster processing)')
    
    args = parser.parse_args()
    
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
    print(f"Max age: {args.max_age} frames")
    print(f"Max IOU distance: {args.max_iou_distance}")
    print(f"Display: {'Disabled' if args.no_display else 'Enabled'}")
    
    process_video(args.input, args.output, model, args.confidence, args.max_age, args.max_iou_distance, args.iou, args.no_display)

if __name__ == "__main__":
    main()
