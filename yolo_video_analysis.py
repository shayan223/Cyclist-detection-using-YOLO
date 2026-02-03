import cv2
import argparse
import os
from ultralytics import YOLO, RTDETR
import torch
import numpy as np

# --- Configuration ---
EXPERIMENT_NAME =  'pdx_rtdetr_finetune3'#'rtdetr_finetune4'#'rtdetr_finetune4'#'yolo_finetune2'
CONFIG_FILE_PATH = './training_data/dataset.yaml'#'./training_data/config.yaml'
BATCH = 8
DEFAULT_MODEL_PATH = './pdx_rtdetr/'+EXPERIMENT_NAME+'/weights/best.pt' #'yolov8l.pt'  # Base model (YOLOv8 or RT-DETR)
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

def load_model(model_path, device, model_type='auto'):
    """Load a YOLOv8 or RT-DETR model."""
    model_type = (model_type or 'auto').lower()

    # Auto-detect architecture from path if requested
    inferred_type = model_type
    if model_type == 'auto':
        path_lower = str(model_path).lower()
        if 'rtdetr' in os.path.basename(path_lower) or 'rtdetr' in path_lower:
            inferred_type = 'rtdetr'
        else:
            inferred_type = 'yolo'

    print(f"Loading {inferred_type.upper()} model from: {model_path}")

    if inferred_type == 'rtdetr':
        model = RTDETR(model_path)
    else:
        model = YOLO(model_path)

    model.to(device)
    print(f"Model loaded successfully on device: {device}")
    return model

def process_video(input_video_path, output_video_path, model, confidence_threshold=0.5, iou_threshold=0.1,
                  disable_display=False):
    """Process video file and overlay cyclist bounding boxes with optional live display."""
    
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
    if not disable_display:
        print(f"\nControls:")
        print(f"  Press 'q' to quit")
        print(f"  Press 's' to save current frame")
        print(f"  Press 'p' to pause/resume")
        print(f"  Press '+' to increase speed")
        print(f"  Press '-' to decrease speed")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    paused = False
    speed_multiplier = 1.0  # Speed control (1.0 = normal speed)
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO detection
                # This also outputs logging to the console, giving image size and detected objects.
                # Lower iou threshold (default 0.45) to prevent NMS from suppressing overlapping detections
                # between different classes (cyclists and pedestrians)
                results = model(frame, conf=confidence_threshold, iou=iou_threshold, agnostic_nms=False)
                #Uncomment this for logging info and bounding box coordinates
                '''
                # Extract detected objects
                detected_objects = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            # Store detection info
                            detection = {
                                'class_id': cls,
                                'class_name': CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}",
                                'confidence': float(conf),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            }
                            detected_objects.append(detection)
                
                # Print detected objects (optional - can be commented out)
                if detected_objects:
                    print(f"Frame {frame_count}: {len(detected_objects)} objects detected")
                    for obj in detected_objects:
                        if(obj['class_name'] == 'bicycle'):
                        print(f"  - {obj['class_name']}: {obj['confidence']:.2f}")
                '''
                # Process detections
                annotated_frame = frame.copy()
                cyclist_count = 0  # Counter for cyclists in current frame
                pedestrian_count = 0  # Counter for pedestrians in current frame
                
                # Collect all detections
                all_detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            all_detections.append((cls, conf, (int(x1), int(y1), int(x2), int(y2))))
                
                # Debug: print all detections if cyclists are present
                if any(cls == 0 for cls, _, _ in all_detections):
                    print(f"Frame {frame_count} DEBUG - All detections when cyclists present:")
                    for cls, conf, bbox in all_detections:
                        class_name = "cyclist" if cls == 0 else "pedestrian" if cls == 1 else f"class_{cls}"
                        print(f"  {class_name}: conf={conf:.3f}, bbox={bbox}")
                
                # Draw detections
                for cls, conf, (x1, y1, x2, y2) in all_detections:
                    # Filter for cyclists (class 0) and pedestrians (class 1)
                    if cls == 0:  # cyclist
                        cyclist_count += 1  # Increment cyclist counter
                        
                        # Draw bounding box (green for cyclists)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"cyclist: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    elif cls == 1:  # pedestrian
                        pedestrian_count += 1  # Increment pedestrian counter
                        
                        # Draw bounding box (blue for pedestrians)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        
                        # Draw label
                        label = f"pedestrian: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (255, 0, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Print counts to console
                print(f"Frame {frame_count}: {cyclist_count} cyclist(s), {pedestrian_count} pedestrian(s) detected")
                
                # Draw counts on video (lower right corner)
                cyclist_text = f"Cyclists: {cyclist_count}"
                pedestrian_text = f"Pedestrians: {pedestrian_count}"
                
                # Calculate text sizes
                cyclist_text_size = cv2.getTextSize(cyclist_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                pedestrian_text_size = cv2.getTextSize(pedestrian_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                
                # Calculate maximum width for both texts
                max_width = max(cyclist_text_size[0], pedestrian_text_size[0])
                
                # Position text in lower right corner
                text_x = width - max_width - 20
                text_y_cyclist = height - 50
                text_y_pedestrian = height - 20
                
                # Draw background rectangle for better visibility
                bg_height = text_y_pedestrian + pedestrian_text_size[1] + 10 - (text_y_cyclist - cyclist_text_size[1] - 10)
                cv2.rectangle(annotated_frame, (text_x - 10, text_y_cyclist - cyclist_text_size[1] - 10), 
                            (text_x + max_width + 10, text_y_pedestrian + 10), (0, 0, 0), -1)
                
                # Draw cyclist count (green)
                cv2.putText(annotated_frame, cyclist_text, (text_x, text_y_cyclist), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Draw pedestrian count (blue)
                cv2.putText(annotated_frame, pedestrian_text, (text_x, text_y_pedestrian), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Print progress every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            if not disable_display:
                # Display the frame
                cv2.imshow('Cyclist & Pedestrian Detection - Live View', annotated_frame)
                
                # Calculate actual delay based on speed multiplier
                actual_delay = max(1, int(frame_delay / speed_multiplier))
                
                # Handle keyboard input
                key = cv2.waitKey(actual_delay) & 0xFF
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
        if not disable_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print(f"Video processing completed. Output saved to: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze video for cyclist detection using YOLO')
    parser.add_argument('--input', '-i', required=False, help='Input video file path', default='japan_long_cyclist_video.mp4')
    parser.add_argument('--output', '-o', help='Output video file path (default: input_analyzed.mp4)')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL_PATH, help='YOLO model path')
    parser.add_argument('--model-type', '-t',
                        default='auto',
                        choices=['yolo', 'rtdetr', 'auto'],
                        help="Model architecture: 'yolo' for YOLOv8, 'rtdetr' for RT-DETR, or 'auto' to infer from path")
    parser.add_argument('--confidence', '-c', type=float, default=0.7, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.1, help='NMS IoU threshold (0.0-1.0). Lower values allow more overlapping detections. Default: 0.3')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable live OpenCV window (useful in headless/GUI-less environments)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input video file '{args.input}' not found")
        return
    
    # Set output path if not provided
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_analyzed.mp4"
    
    # Validate model file
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return
    
    # Load model
    model = load_model(args.model, DEVICE, args.model_type)
    
    # Process video
    print(f"Starting video analysis...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model path: {args.model}")
    print(f"Model type: {args.model_type}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"NMS IoU threshold: {args.iou}")
    
    process_video(args.input, args.output, model, args.confidence, args.iou, disable_display=args.no_display)

if __name__ == "__main__":
    main()
