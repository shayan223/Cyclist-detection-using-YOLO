import cv2
import numpy as np
import argparse
from collections import deque
import json
from datetime import timedelta
import os
from pathlib import Path
import time
import sys
import threading
import csv
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")

class MovementDetector:
    def __init__(self, video_path, motion_threshold=30, min_area=100, buffer_size=10, 
                 temporal_frames=3, min_aspect_ratio=0.2, max_aspect_ratio=5.0,
                 min_solidity=0.4, min_compactness=0.25, min_distance_ratio=0.6, max_objects=2,
                 dataset_dir="pdx_cyclist_dataset", auto_pause=True):
        """
        Initialize the movement detector.
        
        Args:
            video_path: Path to the video file
            motion_threshold: Threshold for motion detection (0-255)
            min_area: Minimum area of motion to consider as movement
            buffer_size: Number of frames to average for background subtraction
            temporal_frames: Number of consecutive frames motion must persist (reduces noise)
            min_aspect_ratio: Minimum width/height or height/width ratio (filters elongated noise)
            max_aspect_ratio: Maximum width/height or height/width ratio (filters elongated noise)
        """
        self.video_path = video_path
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        self.buffer_size = buffer_size
        self.temporal_frames = temporal_frames
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_solidity = min_solidity
        self.min_compactness = min_compactness
        self.min_distance_ratio = min_distance_ratio
        self.max_objects = max_objects
        self.auto_pause = auto_pause
        
        # Dataset creation
        self.dataset_dir = Path(dataset_dir)
        self.dataset_images_dir = self.dataset_dir / "train" / "images"
        self.dataset_labels_dir = self.dataset_dir / "train" / "labels"
        self.saved_count = 0
        
        # Initialize dataset directories
        self.dataset_images_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get next available file number to avoid overwriting
        self.next_file_number = self.get_next_file_number()
        
        # Create data.yaml file
        self.create_data_yaml()
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Polygon region
        self.polygon_points = []
        self.polygon_complete = False
        self.mask = None
        
        # Motion detection - increased history and threshold for better noise reduction
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False  # Disable shadows to reduce noise
        )
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Temporal filtering - track motion across frames
        self.motion_history = deque(maxlen=temporal_frames)
        
        # Object tracking for consistency filtering
        self.object_tracks = {}  # Track objects across frames: {track_id: [history]}
        self.next_track_id = 0
        self.track_history_length = max(5, temporal_frames)  # Track objects for several frames
        
        # Movement timestamps
        self.movement_timestamps = []
        self.current_frame_number = 0
        self.last_movement_frame = -1
        self.movement_cooldown = int(self.fps * 0.5)  # 0.5 seconds cooldown between detections
        
        # Dataset annotation state
        self.selected_boxes = []  # Selected bounding boxes for current frame: [(x, y, w, h, class_id), ...]
        self.manual_boxes = []  # Manually drawn bounding boxes: [(x, y, w, h, class_id), ...]
        self.drawing_box = False  # Whether currently drawing a manual box
        self.box_start = None  # Start point for manual box
        self.current_box = None  # Current box being drawn
        self.current_class = 0  # Current annotation class: 0 = cyclist, 1 = pedestrian
        
        # Resume cooldown - prevent auto-pause for N frames after resuming
        self.RESUME_COOLDOWN_FRAMES = 100
        self.resume_cooldown_counter = 0
        
        # Playback speed control
        self.playback_speed = 1.0  # 1x = normal, 2x = 2x speed, etc.
        self.speed_multipliers = [1.0, 2.0, 4.0, 8.0, 16.0]  # Available speed options
        self.speed_index = 0  # Current index in speed_multipliers
        
        # Display scaling for larger window
        self.display_scale = 1.5  # Scale factor for display (1.5x = 50% larger)
        
        # Timestamp navigation (for CSV mode)
        self.timestamp_list = []  # List of timestamps to navigate: [(frame, timestamp_seconds), ...]
        self.current_timestamp_index = -1  # Current index in timestamp_list (-1 = not in timestamp mode)
        
    def get_next_file_number(self):
        """Get the next available file number by checking existing files across all splits.
        
        Checks train, valid, and test directories to ensure no duplicate file numbers
        even if files have been moved to different splits.
        """
        max_num = -1
        
        # Check all possible split directories (train, valid, test)
        # This ensures we don't reuse numbers even if files were moved to other splits
        for split_name in ["train", "valid", "test"]:
            split_images_dir = self.dataset_dir / split_name / "images"
            if not split_images_dir.exists():
                continue
            
            # Find all existing frame files in this split
            existing_files = list(split_images_dir.glob("frame_*.jpg"))
            
            # Extract numbers from filenames and find the maximum
            for file in existing_files:
                try:
                    # Extract number from filename like "frame_00000123.jpg"
                    num_str = file.stem.split('_')[1]  # Get "00000123"
                    num = int(num_str)
                    max_num = max(max_num, num)
                except (ValueError, IndexError):
                    continue
        
        return max_num + 1
    
    def create_data_yaml(self):
        """Create or update data.yaml file for YOLO dataset.
        
        Preserves existing train/val/test paths if data.yaml already exists
        (e.g., after dataset split), only updating class names if needed.
        """
        yaml_path = self.dataset_dir / "data.yaml"
        
        # Default paths
        train_path = "train/images"
        val_path = None
        test_path = None
        
        # Check if data.yaml already exists (e.g., after split)
        # Try to preserve existing paths by parsing the file
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    content = f.read()
                    # Simple parsing to extract paths (works even without yaml module)
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('train:'):
                            train_path = line.split(':', 1)[1].strip()
                        elif line.startswith('val:'):
                            val_path = line.split(':', 1)[1].strip()
                        elif line.startswith('test:'):
                            test_path = line.split(':', 1)[1].strip()
            except Exception:
                # If file exists but can't be read, use defaults
                pass
        
        # Build YAML content
        yaml_lines = [
            "names:",
            "- cyclist",
            "- pedestrian",
            "nc: 2",
            f"train: {train_path}",
        ]
        
        # Add val/test paths if they exist in the original file
        if val_path:
            yaml_lines.append(f"val: {val_path}")
        if test_path:
            yaml_lines.append(f"test: {test_path}")
        
        yaml_content = "\n".join(yaml_lines) + "\n"
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        if yaml_path.exists() and (val_path or test_path):
            print(f"Updated dataset config at: {yaml_path} (preserved existing split paths)")
        else:
            print(f"Created dataset config at: {yaml_path}")
    
    def reset_video(self):
        """Reset video to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_number = 0
    
    def set_video_position_percent(self, percent):
        """
        Set video position to a specific percentage.
        
        Args:
            percent: Percentage (0-100) to start from
        """
        if percent < 0 or percent > 100:
            raise ValueError("Percentage must be between 0 and 100")
        target_frame = int(self.total_frames * percent / 100.0)
        target_frame = min(target_frame, self.total_frames - 1)  # Ensure within bounds
        
        # OpenCV seek can fail silently with some codecs/containers.
        # Try frame-based seek first, then timestamp-based seek, then a grab fallback.
        seek_tolerance = max(2, int(self.fps * 0.25))  # ~250ms tolerance
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if abs(actual_frame - target_frame) > seek_tolerance and self.fps > 0:
            target_msec = (target_frame / self.fps) * 1000.0
            self.cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
            actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if abs(actual_frame - target_frame) > seek_tolerance:
            # Final fallback: rewind and advance using decoder grabs.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            grabbed = 0
            while grabbed < target_frame:
                if not self.cap.grab():
                    break
                grabbed += 1
            actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        self.current_frame_number = actual_frame
        return actual_frame
    
    def parse_timestamp_value(self, value_str):
        """
        Parse a timestamp value from various formats.
        
        Supports:
        - Time format: "H:MM:SS" or "HH:MM:SS" (hours:minutes:seconds)
        - Time format: "MM:SS" (minutes:seconds)
        - Numeric: seconds as float (e.g., "123.45")
        - Numeric: frame number as int (e.g., "1500")
        
        Args:
            value_str: String value to parse
            
        Returns:
            tuple: (timestamp_seconds, is_frame_number)
                   is_frame_number is True if value was interpreted as frame number
        """
        value_str = str(value_str).strip()
        
        # Try parsing as time format (H:MM:SS or MM:SS)
        if ':' in value_str:
            parts = value_str.split(':')
            # Strip whitespace from each part
            parts = [p.strip() for p in parts]
            
            if len(parts) == 3:
                # Format: H:MM:SS or HH:MM:SS
                try:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = float(parts[2])
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    return (total_seconds, False)
                except (ValueError, IndexError) as e:
                    # If time parsing fails, continue to numeric parsing
                    pass
            elif len(parts) == 2:
                # Format: MM:SS
                try:
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                    total_seconds = minutes * 60 + seconds
                    return (total_seconds, False)
                except (ValueError, IndexError) as e:
                    # If time parsing fails, continue to numeric parsing
                    pass
        
        # Try parsing as numeric value (only if it doesn't contain ':' since that's already handled)
        if ':' not in value_str:
            try:
                value = float(value_str)
                # Determine if it's a frame number or timestamp in seconds
                # If value > total_frames, assume it's seconds, otherwise assume frames
                if value > self.total_frames:
                    # It's a timestamp in seconds
                    return (value, False)
                else:
                    # It's a frame number
                    return (value, True)
            except ValueError:
                pass
        
        # If we get here, we couldn't parse it
        raise ValueError(f"Could not parse timestamp value: '{value_str}'. Expected format: H:MM:SS, MM:SS, or numeric value.")
    
    def load_timestamps_from_csv(self, csv_path):
        """
        Load timestamps from a CSV file.
        
        CSV format can be:
        - With header: A column named 'timestamp', 'time', 't', 'frame', or 'frame_number'
        - Without header: First column contains timestamps
        - Timestamps can be in:
          * Time format: "H:MM:SS" or "HH:MM:SS" (e.g., "7:51:02", "4:59:51")
          * Time format: "MM:SS" (e.g., "51:02")
          * Seconds as float (e.g., "123.45")
          * Frame numbers as int (e.g., "1500")
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            List of (frame_number, timestamp_seconds) tuples, sorted by frame
        """
        timestamps = []
        
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                # Read first few lines to determine format
                lines = []
                for i, line in enumerate(f):
                    lines.append(line.strip())
                    if i >= 2:  # Read first 3 lines
                        break
                f.seek(0)
                
                # Check if first line looks like a header (contains common header words)
                has_header = False
                if lines:
                    first_line = lines[0].lower()
                    header_keywords = ['timestamp', 'time', 'frame', 't', 'date', 'id']
                    has_header = any(keyword in first_line for keyword in header_keywords)
                
                # Check if values contain colons (time format) - if so, don't use colon as delimiter
                # Try to detect delimiter, but avoid colon if values contain colons
                sample = f.read(1024)
                f.seek(0)
                sniffer = csv.Sniffer()
                detected_delimiter = sniffer.sniff(sample).delimiter
                
                # If detected delimiter is colon and values contain colons, colon is part of data, not delimiter
                delimiter = detected_delimiter
                if detected_delimiter == ':' and lines and not has_header:
                    # Check if values look like time format (contain colons in H:MM:SS or MM:SS format)
                    if any(':' in line and (line.count(':') == 1 or line.count(':') == 2) for line in lines):
                        # Values contain colons in time format, so colon is part of the data, not delimiter
                        # Try comma or tab as delimiter, or read line by line
                        if any(',' in line for line in lines):
                            delimiter = ','
                        elif any('\t' in line for line in lines):
                            delimiter = '\t'
                        else:
                            # No other delimiter found, read entire line (no delimiter)
                            delimiter = None
                
                if has_header:
                    # Use DictReader for files with headers
                    reader = csv.DictReader(f, delimiter=delimiter)
                    
                    # Find the timestamp column
                    timestamp_col = None
                    for col_name in ['timestamp', 'time', 't', 'frame', 'frame_number', 'frame_num']:
                        if col_name in reader.fieldnames:
                            timestamp_col = col_name
                            break
                    
                    # If no named column found, try to find a column with parseable timestamp data
                    if timestamp_col is None:
                        for col_name in reader.fieldnames:
                            # Check if column contains parseable timestamp data
                            f.seek(0)
                            next(reader)  # Skip header
                            sample_row = next(reader, None)
                            if sample_row and col_name in sample_row:
                                try:
                                    self.parse_timestamp_value(sample_row[col_name])
                                    timestamp_col = col_name
                                    break
                                except (ValueError, KeyError):
                                    continue
                        f.seek(0)
                        next(reader)  # Skip header again
                    
                    if timestamp_col is None:
                        raise ValueError("Could not find a parseable timestamp column in CSV")
                    
                    # Read all timestamps
                    for row in reader:
                        try:
                            value_str = row[timestamp_col]
                            timestamp_sec, is_frame = self.parse_timestamp_value(value_str)
                            
                            if is_frame:
                                # It's a frame number
                                frame_num = int(timestamp_sec)
                                timestamp_sec = frame_num / self.fps
                            else:
                                # It's a timestamp in seconds
                                frame_num = int(timestamp_sec * self.fps)
                            
                            # Clamp to valid range
                            frame_num = max(0, min(frame_num, self.total_frames - 1))
                            timestamps.append((frame_num, timestamp_sec))
                        except (ValueError, KeyError) as e:
                            continue  # Skip invalid rows
                else:
                    # No header - read timestamps
                    if delimiter is None:
                        # No delimiter found, read line by line (entire line is the value)
                        for row_num, line in enumerate(f):
                            if not line.strip():  # Skip empty rows
                                continue
                            try:
                                value_str = line.strip()  # Entire line is the value
                                if not value_str:  # Skip empty values
                                    continue
                                
                                timestamp_sec, is_frame = self.parse_timestamp_value(value_str)
                                
                                if is_frame:
                                    # It's a frame number
                                    frame_num = int(timestamp_sec)
                                    timestamp_sec = frame_num / self.fps
                                else:
                                    # It's a timestamp in seconds
                                    frame_num = int(timestamp_sec * self.fps)
                                
                                # Clamp to valid range
                                frame_num = max(0, min(frame_num, self.total_frames - 1))
                                timestamps.append((frame_num, timestamp_sec))
                            except (ValueError, IndexError) as e:
                                continue  # Skip invalid rows
                    else:
                        # Use regular reader with detected delimiter
                        reader = csv.reader(f, delimiter=delimiter)
                        for row_num, row in enumerate(reader):
                            if not row:  # Skip empty rows
                                continue
                            try:
                                value_str = row[0].strip()  # First column
                                if not value_str:  # Skip empty values
                                    continue
                                
                                timestamp_sec, is_frame = self.parse_timestamp_value(value_str)
                                
                                if is_frame:
                                    # It's a frame number
                                    frame_num = int(timestamp_sec)
                                    timestamp_sec = frame_num / self.fps
                                else:
                                    # It's a timestamp in seconds
                                    frame_num = int(timestamp_sec * self.fps)
                                
                                # Clamp to valid range
                                frame_num = max(0, min(frame_num, self.total_frames - 1))
                                timestamps.append((frame_num, timestamp_sec))
                            except (ValueError, IndexError) as e:
                                continue  # Skip invalid rows
                
                # Sort by frame number
                timestamps.sort(key=lambda x: x[0])
                
                print(f"Loaded {len(timestamps)} timestamps from {csv_path}")
                if timestamps:
                    hours_first = timestamps[0][1] / 3600
                    hours_last = timestamps[-1][1] / 3600
                    print(f"  First timestamp: frame {timestamps[0][0]} ({timestamps[0][1]:.2f}s = {hours_first:.2f} hours)")
                    print(f"  Last timestamp: frame {timestamps[-1][0]} ({timestamps[-1][1]:.2f}s = {hours_last:.2f} hours)")
                
                return timestamps
                
        except Exception as e:
            raise ValueError(f"Error loading CSV file {csv_path}: {e}")
    
    def jump_to_timestamp(self, index):
        """
        Jump to a specific timestamp by index.
        
        Args:
            index: Index in timestamp_list (0-based)
            
        Returns:
            True if successful, False if index out of range
        """
        if not self.timestamp_list or index < 0 or index >= len(self.timestamp_list):
            return False
        
        frame_num, timestamp_sec = self.timestamp_list[index]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.current_frame_number = frame_num
        self.current_timestamp_index = index
        return True
    
    def jump_to_next_timestamp(self):
        """Jump to the next timestamp in the list."""
        if not self.timestamp_list:
            return False
        
        if self.current_timestamp_index < 0:
            # Start at first timestamp
            return self.jump_to_timestamp(0)
        elif self.current_timestamp_index < len(self.timestamp_list) - 1:
            return self.jump_to_timestamp(self.current_timestamp_index + 1)
        else:
            print("Already at last timestamp")
            return False
    
    def jump_to_previous_timestamp(self):
        """Jump to the previous timestamp in the list."""
        if not self.timestamp_list:
            return False
        
        if self.current_timestamp_index <= 0:
            print("Already at first timestamp")
            return False
        else:
            return self.jump_to_timestamp(self.current_timestamp_index - 1)
    
    def bbox_to_yolo(self, x, y, w, h, img_width, img_height):
        """Convert bounding box from pixel coordinates to YOLO format (normalized)."""
        center_x = (x + w / 2) / img_width
        center_y = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        return center_x, center_y, width, height
    
    def save_frame_as_dataset(self, frame, frame_number, bounding_boxes):
        """Save frame and annotations in YOLO format.
        
        Args:
            frame: Frame image to save
            frame_number: Frame number
            bounding_boxes: List of (x, y, w, h, class_id) tuples, or (x, y, w, h) tuples (defaults to class 0)
        """
        # Generate unique filename using sequential numbering across all videos
        filename = f"frame_{self.next_file_number:08d}.jpg"
        image_path = self.dataset_images_dir / filename
        label_path = self.dataset_labels_dir / filename.replace('.jpg', '.txt')
        
        # Save image
        cv2.imwrite(str(image_path), frame)
        
        # Save labels (class 0 = cyclist, class 1 = pedestrian)
        cyclist_count = 0
        pedestrian_count = 0
        with open(label_path, 'w') as f:
            for bbox in bounding_boxes:
                # Handle both (x, y, w, h) and (x, y, w, h, class_id) formats
                if len(bbox) >= 5:
                    x, y, w, h, class_id = bbox[:5]
                else:
                    x, y, w, h = bbox[:4]
                    class_id = 0  # Default to cyclist for backward compatibility
                
                center_x, center_y, width, height = self.bbox_to_yolo(x, y, w, h, self.width, self.height)
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                if class_id == 0:
                    cyclist_count += 1
                elif class_id == 1:
                    pedestrian_count += 1
        
        self.saved_count += 1
        self.next_file_number += 1  # Increment for next file
        class_info = []
        if cyclist_count > 0:
            class_info.append(f"{cyclist_count} cyclist{'s' if cyclist_count > 1 else ''}")
        if pedestrian_count > 0:
            class_info.append(f"{pedestrian_count} pedestrian{'s' if pedestrian_count > 1 else ''}")
        class_str = ", ".join(class_info) if class_info else "0 objects"
        print(f"Saved frame {frame_number} as dataset sample #{self.saved_count} ({class_str}) -> {filename}")
        return image_path, label_path
        
    def draw_polygon_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing polygon."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.polygon_complete:
                self.polygon_points.append([x, y])
                print(f"Point {len(self.polygon_points)}: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.polygon_points) >= 3:
                self.polygon_complete = True
                self.create_mask()
                print("Polygon complete! Right-click again to reset, or press 'c' to confirm.")
    
    def annotation_callback(self, event, x, y, flags, param):
        """Mouse callback for annotation mode (clicking boxes and drawing manual boxes).
        Note: Coordinates are in display space (scaled), need to convert back to original.
        """
        # Convert display coordinates back to original frame coordinates
        scale = param.get('display_scale', 1.0)
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on a detected bounding box
            clicked_box = None
            for i, (bx, by, bw, bh) in enumerate(param['detected_boxes']):
                if bx <= orig_x <= bx + bw and by <= orig_y <= by + bh:
                    clicked_box = i
                    break
            
            if clicked_box is not None:
                # Toggle selection of this box
                bbox = param['detected_boxes'][clicked_box]
                # Check if already selected (with any class)
                selected_bbox_tuple = None
                for sel_bbox in param['selected_boxes']:
                    if len(sel_bbox) >= 4 and sel_bbox[:4] == bbox[:4]:
                        selected_bbox_tuple = sel_bbox
                        break
                
                if selected_bbox_tuple is not None:
                    param['selected_boxes'].remove(selected_bbox_tuple)
                    class_name = "cyclist" if (len(selected_bbox_tuple) < 5 or selected_bbox_tuple[4] == 0) else "pedestrian"
                    print(f"Deselected box {clicked_box + 1} ({class_name})")
                else:
                    # Add with current class
                    bbox_with_class = (bbox[0], bbox[1], bbox[2], bbox[3], param.get('current_class', 0))
                    param['selected_boxes'].append(bbox_with_class)
                    class_name = "cyclist" if param.get('current_class', 0) == 0 else "pedestrian"
                    print(f"Selected box {clicked_box + 1} as {class_name}")
            else:
                # Start drawing a manual bounding box
                self.drawing_box = True
                self.box_start = (orig_x, orig_y)
                self.current_box = None
                param['drawing_box'] = True
                param['box_start'] = (orig_x, orig_y)
                param['current_box'] = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if param.get('drawing_box', False) and param.get('box_start') is not None:
                # Update current box while dragging
                x1, y1 = param['box_start']
                x2, y2 = orig_x, orig_y
                self.current_box = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                param['current_box'] = self.current_box
        
        elif event == cv2.EVENT_LBUTTONUP:
            if param.get('drawing_box', False) and param.get('box_start') is not None:
                # Finish drawing manual box
                x1, y1 = param['box_start']
                x2, y2 = orig_x, orig_y
                # Only add if box has minimum size
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    manual_box = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1), param.get('current_class', 0))
                    # Check if already exists (same coordinates)
                    exists = False
                    for existing in param['manual_boxes']:
                        if len(existing) >= 4 and existing[:4] == manual_box[:4]:
                            exists = True
                            break
                    if not exists:
                        param['manual_boxes'].append(manual_box)
                        class_name = "cyclist" if param.get('current_class', 0) == 0 else "pedestrian"
                        print(f"Added manual bounding box as {class_name}: {manual_box[:4]}")
                self.drawing_box = False
                self.box_start = None
                self.current_box = None
                param['drawing_box'] = False
                param['box_start'] = None
                param['current_box'] = None
    
    def create_mask(self):
        """Create a binary mask from the polygon."""
        if len(self.polygon_points) < 3:
            return
        
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        pts = np.array(self.polygon_points, dtype=np.int32)
        cv2.fillPoly(self.mask, [pts], 255)
    
    def select_region(self):
        """Interactive polygon selection on the first frame."""
        self.reset_video()
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read first frame from video")
        
        window_name = "Select Region - Left click to add points, Right click to finish"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.draw_polygon_callback)
        
        print("\n=== Region Selection ===")
        print("Left click: Add point to polygon")
        print("Right click: Finish polygon (need at least 3 points)")
        print("Press 'r': Reset polygon")
        print("Press 'c': Confirm and continue")
        print("Press 'q': Quit")
        print("=======================\n")
        
        while True:
            display_frame = frame.copy()
            
            # Draw polygon points and lines
            if len(self.polygon_points) > 0:
                pts = np.array(self.polygon_points, dtype=np.int32)
                if len(self.polygon_points) > 1:
                    for i in range(len(self.polygon_points) - 1):
                        cv2.line(display_frame, 
                                tuple(self.polygon_points[i]), 
                                tuple(self.polygon_points[i + 1]), 
                                (0, 255, 0), 2)
                if self.polygon_complete:
                    cv2.fillPoly(display_frame, [pts], (0, 255, 0), cv2.LINE_AA)
                    cv2.polylines(display_frame, [pts], True, (0, 255, 255), 3)
                else:
                    for pt in self.polygon_points:
                        cv2.circle(display_frame, tuple(pt), 5, (0, 0, 255), -1)
            
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
            elif key == ord('r'):
                self.polygon_points = []
                self.polygon_complete = False
                self.mask = None
                print("Polygon reset")
            elif key == ord('c'):
                if self.polygon_complete and len(self.polygon_points) >= 3:
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("Please complete the polygon first (at least 3 points, then right-click)")
        
        cv2.destroyAllWindows()
        return False
    
    def detect_motion_in_region(self, frame, use_edge_refinement=False):
        """
        Detect motion within the selected polygon region and return bounding boxes.
        Objects that overlap with the polygon (even partially) are detected with full bounding boxes.
        
        Args:
            frame: Current video frame
            use_edge_refinement: If True, use edge detection to refine bounding boxes
        
        Returns:
            tuple: (motion_detected, total_area, bounding_boxes)
                bounding_boxes: List of (x, y, w, h) tuples
        """
        if self.mask is None:
            return False, 0, []
        
        # Apply background subtraction on full frame first
        fg_mask = self.background_subtractor.apply(frame)
        
        # Apply stronger morphological operations to reduce noise on full frame
        # This ensures we get complete objects, not just parts clipped by polygon
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Multiple passes of morphological operations on full frame
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large)  # Second pass
        
        # Also create masked version for edge refinement and area calculation
        masked_fg = cv2.bitwise_and(fg_mask, self.mask)
        
        # Optional: Use edge detection to refine the mask
        # Improved approach: Run edge detection on full frame, then filter to motion regions
        if use_edge_refinement:
            # First, expand the motion mask slightly to capture edges at boundaries
            # Use smaller kernel and fewer iterations to avoid picking up nearby artifacts
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_motion = cv2.dilate(masked_fg, kernel_dilate, iterations=1)
            
            # Convert frame to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Apply Gaussian blur to reduce noise before edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection on the full frame
            edges = cv2.Canny(blurred, 100, 200)  # Higher thresholds = less noise
            
            # Apply morphological operations to edge mask to reduce noise
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_small)
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_small)
            
            # Only keep edges that are within or near motion regions in the polygon
            # This focuses edge detection on actual moving objects, not static edges
            edges_in_motion = cv2.bitwise_and(edges, dilated_motion)
            
            # Combine edge detection with background subtraction on full frame
            # Edges help refine boundaries of detected motion
            fg_mask = cv2.bitwise_or(fg_mask, edges_in_motion)
            
            # Additional morphological operations for edge-based mask
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large)
            
            # Update masked version for area calculations
            masked_fg = cv2.bitwise_and(fg_mask, self.mask)
        
        # Find contours from the FULL foreground mask to get complete objects
        # This allows us to detect objects that are partially in the polygon
        full_contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate total motion area and extract bounding boxes
        total_area = 0
        bounding_boxes = []
        
        # Check full contours to see if they overlap with the polygon region
        for contour in full_contours:
            # Get bounding box of full contour (not clipped)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if this bounding box overlaps with the polygon region
            # Calculate center and check if it's in polygon, or if any corner is in polygon
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Check if center or any corner of bounding box is in polygon
            bbox_corners = [
                (x, y),  # top-left
                (x + w, y),  # top-right
                (x, y + h),  # bottom-left
                (x + w, y + h),  # bottom-right
                (center_x, center_y)  # center
            ]
            
            # Check if any corner or center is within the polygon
            pts = np.array(self.polygon_points, dtype=np.int32)
            overlap = False
            for corner in bbox_corners:
                if cv2.pointPolygonTest(pts, corner, False) >= 0:
                    overlap = True
                    break
            
            # Also check if a significant portion of the bounding box overlaps
            # by checking if the center is close to the polygon or if box intersects polygon
            if not overlap:
                # Create a small test region around center
                test_size = min(w, h) // 4
                test_region = np.zeros((self.height, self.width), dtype=np.uint8)
                cv2.rectangle(test_region, 
                             (max(0, int(center_x - test_size)), max(0, int(center_y - test_size))),
                             (min(self.width, int(center_x + test_size)), min(self.height, int(center_y + test_size))),
                             255, -1)
                intersection = cv2.bitwise_and(test_region, self.mask)
                if np.sum(intersection) > 0:
                    overlap = True
            
            if not overlap:
                continue  # Skip objects that don't overlap with polygon
            
            # Now calculate area from the portion inside the polygon
            # Create a mask for this contour
            contour_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            
            # Get area inside polygon
            contour_in_polygon = cv2.bitwise_and(contour_mask, self.mask)
            area = cv2.countNonZero(contour_in_polygon)
            
            if area > self.min_area:
                
                # Filter by aspect ratio to remove elongated noise (lines, streaks)
                if w > 0 and h > 0:
                    aspect_ratio = max(w / h, h / w)  # Get the larger ratio
                    if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                        continue  # Skip this contour - likely noise
                
                # Additional filtering: require minimum dimensions
                if w < 5 or h < 5:  # Too small, likely noise
                    continue
                
                # Contour quality analysis - real objects have more solid shapes
                # Calculate solidity using full contour area (not just polygon portion)
                full_contour_area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = full_contour_area / hull_area
                    # Real objects typically have solidity > 0.4, artifacts are often < 0.3
                    if solidity < self.min_solidity:
                        continue  # Too irregular, likely noise
                
                # Compactness check - real objects are more compact (area/perimeter ratio)
                # Use full contour area for compactness calculation
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    compactness = 4 * np.pi * full_contour_area / (perimeter * perimeter)
                    # Real objects typically have compactness > 0.25, artifacts are often < 0.2
                    if compactness < self.min_compactness:
                        continue  # Too elongated or irregular, likely noise
                
                total_area += area
                bounding_boxes.append((x, y, w, h, area, solidity, compactness))
        
        # Check if motion is significant
        motion_detected = total_area > (self.min_area * 2)
        
        # Only return bounding boxes when motion is detected
        if not motion_detected:
            bounding_boxes = []
        
        return motion_detected, total_area, bounding_boxes
    
    def filter_objects_by_tracking(self, bounding_boxes, frame_number, speed_multiplier=1.0):
        """
        Filter objects by tracking them across frames. Objects that don't persist
        or move erratically are likely artifacts.
        
        Args:
            bounding_boxes: List of (x, y, w, h, area, solidity, compactness) tuples
            frame_number: Current frame number
            speed_multiplier: Playback speed multiplier (adjusts movement threshold for skipped frames)
        
        Returns:
            List of filtered bounding boxes (x, y, w, h) only
        """
        if not bounding_boxes:
            # Clean up old tracks
            self.object_tracks = {tid: track for tid, track in self.object_tracks.items() 
                                 if frame_number - track[-1]['frame'] < self.track_history_length * 2}
            return []
        
        # Calculate centers for each bounding box
        current_objects = []
        for bbox in bounding_boxes:
            x, y, w, h, area, solidity, compactness = bbox
            center_x = x + w / 2
            center_y = y + h / 2
            current_objects.append({
                'bbox': (x, y, w, h),
                'center': (center_x, center_y),
                'area': area,
                'solidity': solidity,
                'compactness': compactness,
                'frame': frame_number
            })
        
        # Match current objects to existing tracks
        matched_tracks = set()
        filtered_boxes = []
        
        for obj in current_objects:
            best_match = None
            best_distance = float('inf')
            
            # Find closest existing track
            for track_id, track_history in self.object_tracks.items():
                if not track_history:
                    continue
                
                last_obj = track_history[-1]
                # Calculate distance from last known position
                dx = obj['center'][0] - last_obj['center'][0]
                dy = obj['center'][1] - last_obj['center'][1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Also check size consistency (real objects don't change size dramatically)
                size_ratio = max(obj['area'] / last_obj['area'], last_obj['area'] / obj['area'])
                
                # Get object dimensions for movement threshold
                x, y, w, h = obj['bbox']
                
                # Match if close enough and size is consistent
                # Scale movement threshold by speed multiplier to account for skipped frames
                # When frames are skipped, objects can move further between processed frames
                max_movement = max(w, h) * 2 * speed_multiplier  # Allow movement up to 2x object size * speed
                if distance < max_movement and size_ratio < 2.0:
                    if distance < best_distance:
                        best_distance = distance
                        best_match = track_id
            
            if best_match is not None:
                # Update existing track
                self.object_tracks[best_match].append(obj)
                matched_tracks.add(best_match)
                
                # Only keep objects that have been tracked for multiple frames
                # Reduced from 3 to 2 frames to prioritize true positives
                if len(self.object_tracks[best_match]) >= 2:  # At least 2 frames
                    filtered_boxes.append(obj['bbox'])
            else:
                # Create new track
                self.object_tracks[self.next_track_id] = [obj]
                self.next_track_id += 1
        
        # Clean up old tracks that weren't matched
        self.object_tracks = {tid: track for tid, track in self.object_tracks.items() 
                             if tid in matched_tracks or 
                             (track and frame_number - track[-1]['frame'] < self.track_history_length)}
        
        return filtered_boxes
    
    def filter_adjacent_artifacts(self, bounding_boxes, size_ratio_threshold=0.3):
        """
        Filter out small artifacts that appear adjacent to larger objects.
        When a real object moves, it can cause nearby artifacts to be detected.
        This removes small detections that are very close to larger ones.
        
        Args:
            bounding_boxes: List of (x, y, w, h) tuples
            size_ratio_threshold: If a small object is within this size ratio of a larger one
                                 and very close, it's likely an artifact
        
        Returns:
            List of filtered bounding boxes
        """
        if not bounding_boxes or len(bounding_boxes) <= 1:
            return bounding_boxes
        
        # Calculate areas and centers
        objects = []
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            area = w * h
            center = (x + w / 2, y + h / 2)
            size = max(w, h)
            objects.append({
                'bbox': bbox,
                'area': area,
                'center': center,
                'size': size
            })
        
        # Sort by area (largest first)
        objects.sort(key=lambda o: o['area'], reverse=True)
        
        filtered = []
        for obj in objects:
            is_artifact = False
            
            # Check if this object is a small artifact near a larger object
            for larger_obj in filtered:
                # Calculate distance between centers
                dx = obj['center'][0] - larger_obj['center'][0]
                dy = obj['center'][1] - larger_obj['center'][1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Calculate size ratio
                size_ratio = obj['size'] / larger_obj['size'] if larger_obj['size'] > 0 else 1.0
                
                # If small object is very close to a much larger object, it's likely an artifact
                # Check if distance is less than 1.5x the larger object's size
                max_distance = larger_obj['size'] * 1.5
                
                if distance < max_distance and size_ratio < size_ratio_threshold:
                    is_artifact = True
                    break
            
            if not is_artifact:
                filtered.append(obj)
        
        return [obj['bbox'] for obj in filtered]
    
    def filter_spatial_clusters(self, bounding_boxes, min_distance_ratio=0.5, max_objects=3):
        """
        Filter objects by spatial clustering. Artifacts often cluster together,
        while real objects are more isolated. Keep only the most significant objects.
        
        Args:
            bounding_boxes: List of (x, y, w, h) tuples
            min_distance_ratio: Minimum distance between objects as ratio of average object size
            max_objects: Maximum number of objects to keep (prioritize largest)
        
        Returns:
            List of filtered bounding boxes
        """
        if not bounding_boxes or len(bounding_boxes) <= 1:
            return bounding_boxes
        
        # Calculate centers and sizes for each bounding box
        objects = []
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            center_x = x + w / 2
            center_y = y + h / 2
            size = max(w, h)  # Use larger dimension as size
            area = w * h
            objects.append({
                'bbox': bbox,
                'center': (center_x, center_y),
                'size': size,
                'area': area
            })
        
        # Sort by area (largest first) - prioritize bigger objects
        objects.sort(key=lambda o: o['area'], reverse=True)
        
        # Calculate average object size
        avg_size = np.mean([o['size'] for o in objects])
        min_distance = avg_size * min_distance_ratio
        
        # Keep objects that are sufficiently separated
        filtered = []
        for obj in objects:
            # Check distance to already filtered objects
            too_close = False
            for kept_obj in filtered:
                dx = obj['center'][0] - kept_obj['center'][0]
                dy = obj['center'][1] - kept_obj['center'][1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # If too close to an existing object, skip it
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(obj)
            
            # Limit total number of objects
            if len(filtered) >= max_objects:
                break
        
        return [obj['bbox'] for obj in filtered]
    
    def check_temporal_consistency(self, current_motion_detected):
        """
        Check if motion has persisted across multiple frames (temporal filtering).
        This helps filter out single-frame noise.
        
        Args:
            current_motion_detected: Whether motion was detected in current frame
        
        Returns:
            bool: True if motion has persisted across required number of frames
        """
        self.motion_history.append(current_motion_detected)
        
        # Need at least temporal_frames entries
        if len(self.motion_history) < self.temporal_frames:
            return False
        
        # Motion must be detected in all recent frames
        return all(self.motion_history)
    
    def process_video(self, display=True, save_output=None, use_edge_refinement=False, headless_scan=False, timestamp_list=None):
        """
        Process video with dataset creation features.
        
        Args:
            display: Whether to display the video during processing
            save_output: Path to save output video (optional)
            use_edge_refinement: Use edge detection to refine bounding boxes
            headless_scan: If True, scan in terminal with progress bar, only show GUI on detections
            timestamp_list: Optional list of (frame, timestamp) tuples to navigate between
        """
        if not self.polygon_complete or self.mask is None:
            print("Error: No region selected. Please select a region first.")
            return
        
        self.reset_video()
        
        # Set starting position if specified
        start_percent = getattr(self, 'start_percent', None)
        start_frame = self.current_frame_number
        if start_percent is not None and start_percent > 0:
            start_frame = self.set_video_position_percent(start_percent)
            print(f"Starting video at {start_percent}% (frame {start_frame})")
        
        self.movement_timestamps = []
        self.last_movement_frame = -1
        self.motion_history.clear()  # Reset temporal filtering history
        self.object_tracks.clear()  # Reset object tracking
        self.next_track_id = 0
        
        # Reset annotation state
        self.selected_boxes = []
        self.manual_boxes = []
        self.drawing_box = False
        self.box_start = None
        self.current_box = None
        self.current_class = 0  # Reset to cyclist
        
        # Reset resume cooldown
        self.resume_cooldown_counter = 0
        
        # Reset playback speed
        self.playback_speed = 1.0
        self.speed_index = 0
        
        # Initialize timestamp navigation if provided
        if timestamp_list:
            self.timestamp_list = timestamp_list
            self.current_timestamp_index = -1  # Will jump to first on start
            print(f"\n=== Timestamp Navigation Mode ===")
            print(f"Loaded {len(timestamp_list)} timestamps for navigation")
            print("Press 'n' for next timestamp, 'p' for previous timestamp")
        else:
            self.timestamp_list = []
            self.current_timestamp_index = -1
        
        # Setup video writer if saving output
        out_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(save_output, fourcc, self.fps, (self.width, self.height))
        
        print("\n=== Dataset Creation Mode ===")
        print(f"Video: {self.video_path}")
        print(f"FPS: {self.fps:.2f}")
        print(f"Total frames: {self.total_frames}")
        print(f"Duration: {self.total_frames / self.fps:.2f} seconds")
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Auto-pause on motion: {'ENABLED' if self.auto_pause else 'DISABLED'}")
        if use_edge_refinement:
            print("Edge refinement: ENABLED")
        print(f"Temporal filtering: {self.temporal_frames} frames (motion must persist)")
        print("\n=== Controls ===")
        print("  Space: Pause/Resume")
        print("  Left Click on box: Select/Deselect detected object")
        print("  Left Click + Drag: Draw manual bounding box")
        print("  't': Toggle annotation class (cyclist/pedestrian)")
        print("  's': Save current frame with selected boxes and manual boxes")
        print("  'c': Clear selected boxes and manual boxes")
        print("  'f': Forward 10 frames")
        print("  'b': Backward 10 frames")
        if self.timestamp_list:
            print("  'n': Jump to next timestamp")
            print("  'p': Jump to previous timestamp")
        print("  'r': Reset to beginning")
        print("  'e': Toggle edge refinement")
        print("  'q': Quit")
        print("=======================\n")
        
        # Initialize frame count from current video position
        frame_count = self.current_frame_number
        paused = False
        current_frame = None
        paused_frame = None  # Store the paused frame (original, not display)
        paused_bounding_boxes = []  # Store bounding boxes when paused
        paused_display_motion = False  # Store motion state when paused
        
        # Mouse callback parameters (use references to actual lists)
        callback_params = {
            'detected_boxes': [],
            'selected_boxes': self.selected_boxes,  # Direct reference
            'manual_boxes': self.manual_boxes,  # Direct reference
            'drawing_box': False,  # Will be updated from self.drawing_box
            'box_start': None,  # Will be updated from self.box_start
            'current_box': None,  # Will be updated from self.current_box
            'display_scale': self.display_scale,  # Display scale for coordinate conversion
            'current_class': self.current_class  # Current annotation class
        }
        
        window_name = "Dataset Creation - Movement Detection"
        gui_initialized = False
        
        # Initialize progress bar (works in both headless and normal mode)
        progress_bar = None
        progress_start_frame = self.current_frame_number
        progress_total_frames = max(0, self.total_frames - progress_start_frame)
        if TQDM_AVAILABLE:
            progress_bar = tqdm(total=progress_total_frames, initial=0, desc="Scanning", unit="frame",
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                              file=sys.stdout, disable=False)
        
        def sync_progress_bar():
            if progress_bar is None:
                return
            processed_frames = frame_count - progress_start_frame
            processed_frames = max(0, min(progress_total_frames, processed_frames))
            progress_bar.n = processed_frames
            progress_bar.set_postfix({'detections': len(self.movement_timestamps), 'saved': self.saved_count})
            progress_bar.refresh()
        
        # Terminal input handling for headless mode
        terminal_continue_pending = False
        
        # If timestamp list is provided, jump to first matching timestamp and pause
        # Respect --start-percent by jumping to the first timestamp at/after start_frame.
        if self.timestamp_list:
            target_timestamp_index = None
            for idx, (ts_frame, _) in enumerate(self.timestamp_list):
                if ts_frame >= start_frame:
                    target_timestamp_index = idx
                    break
            
            if target_timestamp_index is None:
                print(f"No timestamps found at or after start frame {start_frame}. Continuing from current position.")
            elif self.jump_to_timestamp(target_timestamp_index):
                frame_count = self.current_frame_number
                ret, frame = self.cap.read()
                if ret:
                    paused = True
                    paused_frame = frame.copy()
                    # Re-detect motion for this frame
                    motion_detected, motion_area, bounding_boxes = self.detect_motion_in_region(
                        frame, use_edge_refinement=use_edge_refinement
                    )
                    if bounding_boxes:
                        bounding_boxes = self.filter_objects_by_tracking(bounding_boxes, frame_count)
                        if bounding_boxes and len(bounding_boxes) > 1:
                            bounding_boxes = self.filter_adjacent_artifacts(bounding_boxes)
                        if bounding_boxes:
                            bounding_boxes = self.filter_spatial_clusters(bounding_boxes, 
                                                                          min_distance_ratio=self.min_distance_ratio, 
                                                                          max_objects=self.max_objects)
                    paused_bounding_boxes = bounding_boxes.copy() if bounding_boxes else []
                    paused_display_motion = motion_detected
                    self.selected_boxes.clear()
                    self.manual_boxes.clear()
                    print(f"Jumped to timestamp {self.current_timestamp_index + 1}/{len(self.timestamp_list)} "
                          f"(frame {frame_count}, {self.timestamp_list[self.current_timestamp_index][1]:.2f}s)")
        
        while True:
            if not paused:
                # Process every frame for accurate motion detection (don't skip frames)
                # This ensures temporal consistency works correctly regardless of playback speed
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_count += 1
                self.current_frame_number = frame_count
                
                # Detect motion and get bounding boxes (process every frame)
                motion_detected, motion_area, bounding_boxes = self.detect_motion_in_region(
                    frame, use_edge_refinement=use_edge_refinement
                )
                
                # Apply object tracking filter
                if bounding_boxes:
                    bounding_boxes = self.filter_objects_by_tracking(bounding_boxes, frame_count, 1.0)
                    if not bounding_boxes:
                        motion_detected = False
                        motion_area = 0
                
                # Apply adjacent artifact filter
                if bounding_boxes and len(bounding_boxes) > 1:
                    bounding_boxes = self.filter_adjacent_artifacts(bounding_boxes, size_ratio_threshold=0.3)
                    if not bounding_boxes:
                        motion_detected = False
                        motion_area = 0
                
                # Apply spatial clustering filter
                if bounding_boxes:
                    bounding_boxes = self.filter_spatial_clusters(bounding_boxes, 
                                                                  min_distance_ratio=self.min_distance_ratio, 
                                                                  max_objects=self.max_objects)
                    if not bounding_boxes:
                        motion_detected = False
                        motion_area = 0
                
                # Apply temporal filtering
                persistent_motion = self.check_temporal_consistency(motion_detected)
                display_motion = persistent_motion
                record_motion = persistent_motion
                
                # Handle playback speed - only update display at specified speed
                # Process all frames for detection, but only display every Nth frame when speed > 1x
                should_display_this_frame = True
                if self.playback_speed > 1.0:
                    # Only display frames that match the speed interval
                    # For 2x speed: display frames 2, 4, 6, 8...
                    # For 4x speed: display frames 4, 8, 12, 16...
                    should_display_this_frame = (frame_count % int(self.playback_speed) == 0)
                
                # Store frame for display
                current_frame = frame.copy()
                
                # Increment resume cooldown counter when not paused
                # Scale by playback speed so cooldown represents same video time at all speeds
                if not paused:
                    self.resume_cooldown_counter += self.playback_speed
                
                # Auto-pause on motion detection (only if cooldown has passed)
                if (self.auto_pause and persistent_motion and not paused and 
                    self.resume_cooldown_counter >= self.RESUME_COOLDOWN_FRAMES):
                    paused = True
                    paused_frame = current_frame.copy()
                    paused_bounding_boxes = bounding_boxes.copy() if bounding_boxes else []
                    paused_display_motion = display_motion
                    # Reset selection state for new frame
                    self.selected_boxes.clear()
                    self.manual_boxes.clear()
                    print(f"\n[MOTION DETECTED] Auto-paused at frame {frame_count} - Motion detected!")
                    # Initialize GUI if in headless mode and detection occurs
                    if headless_scan and not gui_initialized:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        # Set initial window size (scaled)
                        scaled_width = int(self.width * self.display_scale)
                        scaled_height = int(self.height * self.display_scale)
                        cv2.resizeWindow(window_name, scaled_width, scaled_height)
                        cv2.setMouseCallback(window_name, self.annotation_callback, callback_params)
                        gui_initialized = True
                        print("GUI window opened for annotation. Press 'Enter' in terminal to continue after saving.")
                
                # Record timestamp if persistent motion detected (with cooldown)
                if record_motion and (frame_count - self.last_movement_frame) > self.movement_cooldown:
                    timestamp = frame_count / self.fps
                    self.movement_timestamps.append({
                        'frame': frame_count,
                        'timestamp': timestamp,
                        'timestamp_formatted': str(timedelta(seconds=int(timestamp))),
                        'motion_area': motion_area,
                        'bounding_boxes': bounding_boxes
                    })
                    self.last_movement_frame = frame_count
            else:
                # When paused, use the stored paused frame
                if paused_frame is None:
                    break
                frame = paused_frame.copy()
                bounding_boxes = paused_bounding_boxes.copy()
                display_motion = paused_display_motion
                record_motion = False
                should_display_this_frame = True  # Always display when paused
                
                # Update callback params with current boxes
                callback_params['detected_boxes'] = bounding_boxes
                callback_params['current_class'] = self.current_class  # Update current class
                # selected_boxes and manual_boxes are already references, no need to update
                # But update drawing state to keep in sync
                if self.drawing_box:
                    callback_params['drawing_box'] = self.drawing_box
                    callback_params['box_start'] = self.box_start
                    callback_params['current_box'] = self.current_box
            
            # Keep progress in sync with actual frame position (including jumps/seeks).
            sync_progress_bar()
            
            # Skip rendering during headless scanning (only render when paused)
            # Also skip rendering if this frame shouldn't be displayed due to playback speed
            should_render = ((display and (not headless_scan or paused)) or save_output) and should_display_this_frame
            
            if should_render:
                # Create display frame
                display_frame = frame.copy()
                
                # Draw polygon
                if self.mask is not None:
                    pts = np.array(self.polygon_points, dtype=np.int32)
                    overlay = display_frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                    cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)
                
                # Draw detected bounding boxes
                if display_motion and bounding_boxes:
                    for i, (x, y, w, h) in enumerate(bounding_boxes):
                        # Check if this box is selected and get its class
                        is_selected = False
                        box_class = 0  # Default to cyclist
                        for sel_bbox in self.selected_boxes:
                            if len(sel_bbox) >= 4 and sel_bbox[:4] == (x, y, w, h):
                                is_selected = True
                                if len(sel_bbox) >= 5:
                                    box_class = sel_bbox[4]
                                break
                        
                        # Color coding: Green for cyclist, Magenta for pedestrian
                        if is_selected:
                            color = (0, 255, 0) if box_class == 0 else (255, 0, 255)  # Green for cyclist, Magenta for pedestrian
                        else:
                            color = (255, 0, 0)  # Blue for unselected
                        thickness = 3 if is_selected else 2
                        
                        # Draw rectangle
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
                        
                        # Draw label
                        class_name = "cyclist" if box_class == 0 else "pedestrian"
                        label = f"Object {i+1}" + (f" [{class_name.upper()}]" if is_selected else "")
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        label_y = max(y - 5, label_size[1] + 5)
                        
                        # Draw label background
                        cv2.rectangle(display_frame, 
                                    (x, label_y - label_size[1] - 2), 
                                    (x + label_size[0] + 3, label_y + 2), 
                                    color, -1)
                        
                        # Draw label text
                        cv2.putText(display_frame, label, (x + 1, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Draw manual bounding boxes with class colors
                for manual_bbox in self.manual_boxes:
                    if len(manual_bbox) >= 5:
                        x, y, w, h, box_class = manual_bbox[:5]
                    else:
                        x, y, w, h = manual_bbox[:4]
                        box_class = 0  # Default to cyclist
                    
                    # Color: Cyan for cyclist, Yellow for pedestrian
                    color = (255, 255, 0) if box_class == 0 else (0, 255, 255)  # Cyan for cyclist, Yellow for pedestrian
                    class_name = "CYCLIST" if box_class == 0 else "PEDESTRIAN"
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display_frame, f"MANUAL {class_name}", (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw current box being drawn (with current class color)
                current_drawing_box = callback_params.get('current_box') or self.current_box
                if current_drawing_box is not None:
                    x, y, w, h = current_drawing_box
                    # Color based on current class: Cyan for cyclist, Yellow for pedestrian
                    draw_color = (255, 255, 0) if self.current_class == 0 else (0, 255, 255)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), draw_color, 2)
                
                # Draw status info (smaller font, full labels)
                if paused:
                    current_class_name = "cyclist" if self.current_class == 0 else "pedestrian"
                    status_text = f"PAUSED - Annotate {current_class_name}s"
                    color = (0, 255, 255)  # Cyan for paused
                    cv2.putText(display_frame, status_text, (10, 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    status_text = "MOTION DETECTED!" if display_motion else "Scanning..."
                    color = (0, 165, 255) if display_motion else (0, 255, 0)
                    cv2.putText(display_frame, status_text, (10, 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                cv2.putText(display_frame, f"Frame: {frame_count}/{self.total_frames}", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Time: {frame_count/self.fps:.2f}s", 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Saved: {self.saved_count} samples", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                # Count boxes by class
                cyclist_selected = sum(1 for bbox in self.selected_boxes if len(bbox) < 5 or bbox[4] == 0)
                pedestrian_selected = sum(1 for bbox in self.selected_boxes if len(bbox) >= 5 and bbox[4] == 1)
                cyclist_manual = sum(1 for bbox in self.manual_boxes if len(bbox) < 5 or bbox[4] == 0)
                pedestrian_manual = sum(1 for bbox in self.manual_boxes if len(bbox) >= 5 and bbox[4] == 1)
                cv2.putText(display_frame, f"Selected: {cyclist_selected}C/{pedestrian_selected}P | Manual: {cyclist_manual}C/{pedestrian_manual}P", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                # Show current annotation class
                current_class_name = "CYCLIST" if self.current_class == 0 else "PEDESTRIAN"
                class_color = (0, 255, 0) if self.current_class == 0 else (255, 0, 255)
                cv2.putText(display_frame, f"Class: {current_class_name} (Press 't' to toggle)", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, class_color, 1)
                # Show playback speed
                if not paused:
                    speed_text = f"Speed: {self.playback_speed:.1f}x"
                    cv2.putText(display_frame, speed_text, 
                               (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                # Show resume cooldown status
                if not paused and self.auto_pause:
                    remaining = max(0, self.RESUME_COOLDOWN_FRAMES - self.resume_cooldown_counter)
                    if remaining > 0:
                        # Show remaining in frame-equivalents (accounting for speed)
                        remaining_frames = int(remaining / self.playback_speed) if self.playback_speed > 0 else 0
                        cv2.putText(display_frame, f"Cooldown: {remaining_frames} frames", 
                                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                # Show timestamp navigation info if in timestamp mode
                if self.timestamp_list and self.current_timestamp_index >= 0:
                    timestamp_info = f"Timestamp: {self.current_timestamp_index + 1}/{len(self.timestamp_list)} " \
                                   f"({self.timestamp_list[self.current_timestamp_index][1]:.2f}s)"
                    cv2.putText(display_frame, timestamp_info, 
                               (10, self.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                if save_output:
                    cv2.putText(display_frame, "Recording...", (10, self.height - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    out_writer.write(display_frame)
                
                if display and (not headless_scan or paused):
                    if not gui_initialized:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        # Set initial window size (scaled)
                        scaled_width = int(self.width * self.display_scale)
                        scaled_height = int(self.height * self.display_scale)
                        cv2.resizeWindow(window_name, scaled_width, scaled_height)
                        cv2.setMouseCallback(window_name, self.annotation_callback, callback_params)
                        gui_initialized = True
                    # Ensure window is visible when paused (especially in headless mode)
                    if paused:
                        try:
                            # Try to bring window to front (Windows-specific, but harmless on other platforms)
                            if sys.platform == 'win32':
                                import ctypes
                                hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
                                if hwnd:
                                    ctypes.windll.user32.SetForegroundWindow(hwnd)
                                    ctypes.windll.user32.ShowWindow(hwnd, 1)  # SW_SHOWNORMAL
                        except:
                            pass  # Ignore errors, just try to help
                    
                    # Update callback params to ensure current_class is always up to date
                    callback_params['current_class'] = self.current_class
                    
                    # Scale display frame for larger window
                    scaled_frame = cv2.resize(display_frame, None, fx=self.display_scale, fy=self.display_scale, 
                                             interpolation=cv2.INTER_LINEAR)
                    cv2.imshow(window_name, scaled_frame)
                    
                    # Use longer wait time when paused to ensure window stays responsive
                    wait_time = 30 if paused else 1
                    key = cv2.waitKey(wait_time) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        paused = not paused
                        if paused:
                            paused_frame = current_frame.copy() if current_frame is not None else frame.copy()
                            paused_bounding_boxes = bounding_boxes.copy() if bounding_boxes else []
                            paused_display_motion = display_motion
                            self.selected_boxes.clear()
                            self.manual_boxes.clear()
                            # Ensure window is visible when manually paused
                            if not gui_initialized:
                                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                                scaled_width = int(self.width * self.display_scale)
                                scaled_height = int(self.height * self.display_scale)
                                cv2.resizeWindow(window_name, scaled_width, scaled_height)
                                cv2.setMouseCallback(window_name, self.annotation_callback, callback_params)
                                gui_initialized = True
                            print("Paused - Ready for annotation")
                        else:
                            paused_frame = None
                            paused_bounding_boxes = []
                            paused_display_motion = False
                            self.selected_boxes.clear()
                            self.manual_boxes.clear()
                            # Reset resume cooldown when manually resuming
                            self.resume_cooldown_counter = 0
                            # Calculate effective frames based on current speed
                            effective_frames = int(self.RESUME_COOLDOWN_FRAMES / self.playback_speed) if self.playback_speed > 0 else self.RESUME_COOLDOWN_FRAMES
                            print(f"Resumed scanning (will skip auto-pause for {effective_frames} frames at {self.playback_speed:.1f}x speed)")
                            # In headless mode, hide GUI when resuming
                            if headless_scan and gui_initialized:
                                cv2.destroyWindow(window_name)
                                gui_initialized = False
                                print("GUI hidden. Scanning in background...")
                    elif key == ord('s') and paused:
                        # Save current frame with annotations
                        all_boxes = self.selected_boxes + self.manual_boxes
                        if all_boxes:
                            self.save_frame_as_dataset(paused_frame, frame_count, all_boxes)
                            # Clear selections after saving
                            self.selected_boxes.clear()
                            self.manual_boxes.clear()
                            # In headless mode, allow continuing via terminal
                            if headless_scan:
                                print("Saved! Press 'Enter' in terminal to continue scanning, or 'q' to quit.")
                        else:
                            print("No boxes selected or drawn. Select boxes or draw manual boxes first.")
                    elif key == ord('c') and paused:
                        # Clear selections
                        self.selected_boxes.clear()
                        self.manual_boxes.clear()
                        paused_bounding_boxes = []
                        callback_params['detected_boxes'] = []
                        paused_display_motion = False
                        print("Cleared selections")
                    elif key == ord('t'):
                        # Toggle annotation class (cyclist <-> pedestrian)
                        self.current_class = 1 - self.current_class  # Toggle between 0 and 1
                        callback_params['current_class'] = self.current_class
                        class_name = "cyclist" if self.current_class == 0 else "pedestrian"
                        print(f"Switched annotation class to: {class_name}")
                    elif key == ord('f') and paused:
                        # Forward 10 frames
                        new_frame = min(frame_count + 10, self.total_frames - 1)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                        frame_count = new_frame
                        self.current_frame_number = frame_count
                        ret, frame = self.cap.read()
                        if ret:
                            paused_frame = frame.copy()
                            # Re-detect motion for new frame
                            motion_detected, motion_area, bounding_boxes = self.detect_motion_in_region(
                                frame, use_edge_refinement=use_edge_refinement
                            )
                            if bounding_boxes:
                                bounding_boxes = self.filter_objects_by_tracking(bounding_boxes, frame_count)
                                if bounding_boxes and len(bounding_boxes) > 1:
                                    bounding_boxes = self.filter_adjacent_artifacts(bounding_boxes)
                                if bounding_boxes:
                                    bounding_boxes = self.filter_spatial_clusters(bounding_boxes, 
                                                                                  min_distance_ratio=self.min_distance_ratio, 
                                                                                  max_objects=self.max_objects)
                            paused_bounding_boxes = bounding_boxes.copy() if bounding_boxes else []
                            paused_display_motion = motion_detected
                            self.selected_boxes.clear()
                            self.manual_boxes.clear()
                            print(f"Jumped forward to frame {frame_count}")
                    elif key == ord('b') and paused:
                        # Backward 10 frames
                        new_frame = max(frame_count - 10, 0)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                        frame_count = new_frame
                        self.current_frame_number = frame_count
                        ret, frame = self.cap.read()
                        if ret:
                            paused_frame = frame.copy()
                            # Re-detect motion for new frame
                            motion_detected, motion_area, bounding_boxes = self.detect_motion_in_region(
                                frame, use_edge_refinement=use_edge_refinement
                            )
                            if bounding_boxes:
                                bounding_boxes = self.filter_objects_by_tracking(bounding_boxes, frame_count)
                                if bounding_boxes and len(bounding_boxes) > 1:
                                    bounding_boxes = self.filter_adjacent_artifacts(bounding_boxes)
                                if bounding_boxes:
                                    bounding_boxes = self.filter_spatial_clusters(bounding_boxes, 
                                                                                  min_distance_ratio=self.min_distance_ratio, 
                                                                                  max_objects=self.max_objects)
                            paused_bounding_boxes = bounding_boxes.copy() if bounding_boxes else []
                            paused_display_motion = motion_detected
                            self.selected_boxes.clear()
                            self.manual_boxes.clear()
                            print(f"Jumped backward to frame {frame_count}")
                    elif key == ord('r'):
                        self.reset_video()
                        frame_count = 0
                        self.movement_timestamps = []
                        self.last_movement_frame = -1
                        self.motion_history.clear()
                        paused = False
                        paused_frame = None
                        paused_bounding_boxes = []
                        paused_display_motion = False
                        self.object_tracks.clear()
                        self.next_track_id = 0
                        self.selected_boxes.clear()
                        self.manual_boxes.clear()
                        self.resume_cooldown_counter = 0
                        self.playback_speed = 1.0
                        self.speed_index = 0
                        print("Reset to beginning")
                    elif key == ord('+') or key == ord('='):
                        # Increase playback speed
                        if self.speed_index < len(self.speed_multipliers) - 1:
                            self.speed_index += 1
                            self.playback_speed = self.speed_multipliers[self.speed_index]
                            print(f"Playback speed: {self.playback_speed:.1f}x")
                    elif key == ord('-') or key == ord('_'):
                        # Decrease playback speed
                        if self.speed_index > 0:
                            self.speed_index -= 1
                            self.playback_speed = self.speed_multipliers[self.speed_index]
                            print(f"Playback speed: {self.playback_speed:.1f}x")
                    elif key == ord('e'):
                        use_edge_refinement = not use_edge_refinement
                        print(f"Edge refinement: {'ENABLED' if use_edge_refinement else 'DISABLED'}")
                    elif key == ord('n') and self.timestamp_list:
                        # Jump to next timestamp
                        if self.jump_to_next_timestamp():
                            frame_count = self.current_frame_number
                            ret, frame = self.cap.read()
                            if ret:
                                paused = True
                                paused_frame = frame.copy()
                                current_frame = frame.copy()  # Update current_frame for display
                                # Re-detect motion for this frame
                                motion_detected, motion_area, bounding_boxes = self.detect_motion_in_region(
                                    frame, use_edge_refinement=use_edge_refinement
                                )
                                if bounding_boxes:
                                    bounding_boxes = self.filter_objects_by_tracking(bounding_boxes, frame_count)
                                    if bounding_boxes and len(bounding_boxes) > 1:
                                        bounding_boxes = self.filter_adjacent_artifacts(bounding_boxes)
                                    if bounding_boxes:
                                        bounding_boxes = self.filter_spatial_clusters(bounding_boxes, 
                                                                                      min_distance_ratio=self.min_distance_ratio, 
                                                                                      max_objects=self.max_objects)
                                paused_bounding_boxes = bounding_boxes.copy() if bounding_boxes else []
                                paused_display_motion = motion_detected
                                display_motion = motion_detected
                                self.selected_boxes.clear()
                                self.manual_boxes.clear()
                                print(f"Jumped to timestamp {self.current_timestamp_index + 1}/{len(self.timestamp_list)} "
                                      f"(frame {frame_count}, {self.timestamp_list[self.current_timestamp_index][1]:.2f}s)")
                    elif key == ord('p') and self.timestamp_list:
                        # Jump to previous timestamp
                        if self.jump_to_previous_timestamp():
                            frame_count = self.current_frame_number
                            ret, frame = self.cap.read()
                            if ret:
                                paused = True
                                paused_frame = frame.copy()
                                current_frame = frame.copy()  # Update current_frame for display
                                # Re-detect motion for this frame
                                motion_detected, motion_area, bounding_boxes = self.detect_motion_in_region(
                                    frame, use_edge_refinement=use_edge_refinement
                                )
                                if bounding_boxes:
                                    bounding_boxes = self.filter_objects_by_tracking(bounding_boxes, frame_count)
                                    if bounding_boxes and len(bounding_boxes) > 1:
                                        bounding_boxes = self.filter_adjacent_artifacts(bounding_boxes)
                                    if bounding_boxes:
                                        bounding_boxes = self.filter_spatial_clusters(bounding_boxes, 
                                                                                      min_distance_ratio=self.min_distance_ratio, 
                                                                                      max_objects=self.max_objects)
                                paused_bounding_boxes = bounding_boxes.copy() if bounding_boxes else []
                                paused_display_motion = motion_detected
                                display_motion = motion_detected
                                self.selected_boxes.clear()
                                self.manual_boxes.clear()
                                print(f"Jumped to timestamp {self.current_timestamp_index + 1}/{len(self.timestamp_list)} "
                                      f"(frame {frame_count}, {self.timestamp_list[self.current_timestamp_index][1]:.2f}s)")
            
            # Handle terminal input in headless mode when paused
            # When paused in headless mode, wait for user to press Enter in terminal
            if headless_scan and paused and not terminal_continue_pending:
                # Print prompt once
                if not hasattr(self, '_prompt_shown'):
                    print("\n[PAUSED] Press 'Enter' in terminal to continue scanning after annotation...")
                    self._prompt_shown = True
                
                # Check for terminal input (non-blocking on Windows, blocking on Unix)
                try:
                    if sys.platform == 'win32':
                        import msvcrt
                        if msvcrt.kbhit():
                            key = msvcrt.getch()
                            if key == b'\r':  # Enter key
                                terminal_continue_pending = True
                    else:
                        import select
                        if select.select([sys.stdin], [], [], 0)[0]:
                            user_input = sys.stdin.readline().strip().lower()
                            if user_input == '' or user_input == 'c' or user_input == 'continue':
                                terminal_continue_pending = True
                except:
                    pass
            
            if terminal_continue_pending and paused:
                paused = False
                terminal_continue_pending = False
                self.resume_cooldown_counter = 0
                if gui_initialized:
                    cv2.destroyWindow(window_name)
                    gui_initialized = False
                self._prompt_shown = False
                print("Continuing scan...")
        
        if display and gui_initialized:
            cv2.destroyAllWindows()
        
        if progress_bar is not None:
            progress_bar.close()
        
        if out_writer:
            out_writer.release()
            print(f"Output video saved to: {save_output}")
        
        print(f"\n=== Processing Complete ===")
        print(f"Total movement events detected: {len(self.movement_timestamps)}")
        print(f"Total dataset samples saved: {self.saved_count}")
        print(f"Dataset saved to: {self.dataset_dir}")
        print("============================\n")
        
        return self.movement_timestamps
    
    def save_results(self, output_path):
        """Save movement timestamps to a JSON file."""
        results = {
            'video_path': self.video_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'polygon_points': self.polygon_points,
            'motion_threshold': self.motion_threshold,
            'min_area': self.min_area,
            'movement_events': self.movement_timestamps
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def print_results(self):
        """Print movement timestamps to console."""
        if not self.movement_timestamps:
            print("No movement detected.")
            return
        
        print("\n=== Movement Timestamps ===")
        for event in self.movement_timestamps:
            bbox_info = ""
            if 'bounding_boxes' in event and event['bounding_boxes']:
                bbox_info = f" | Objects: {len(event['bounding_boxes'])}"
            print(f"Frame {event['frame']:6d} | "
                  f"Time: {event['timestamp']:8.2f}s | "
                  f"({event['timestamp_formatted']}) | "
                  f"Area: {event['motion_area']:.0f}{bbox_info}")
        print("===========================\n")
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'cap'):
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(
        description='Detect movement in a selected region of a video using traditional image processing'
    )
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to save results JSON file (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Process without displaying video (faster)')
    parser.add_argument('--motion-threshold', type=int, default=100,
                       help='Motion detection threshold (lower = more sensitive, default: 100)')
    parser.add_argument('--min-area', type=int, default=50,
                       help='Minimum area of motion to consider (default: 50)')
    parser.add_argument('--no-edge-refinement', action='store_true',
                       help='Disable edge detection refinement (enabled by default)')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Path to save output video with bounding boxes (optional)')
    parser.add_argument('--temporal-frames', type=int, default=8,
                       help='Number of consecutive frames motion must persist (lower = faster detection, default: 8)')
    parser.add_argument('--min-aspect-ratio', type=float, default=0.5,
                       help='Minimum aspect ratio to filter elongated noise (lower = more permissive, default: 0.5)')
    parser.add_argument('--max-aspect-ratio', type=float, default=5.5,
                       help='Maximum aspect ratio to filter elongated noise (default: 5.5)')
    parser.add_argument('--min-solidity', type=float, default=0.3,
                       help='Minimum contour solidity (0-1, lower = more permissive, default: 0.3)')
    parser.add_argument('--min-compactness', type=float, default=0.2,
                       help='Minimum contour compactness (0-1, lower = more permissive, default: 0.2)')
    parser.add_argument('--min-distance-ratio', type=float, default=0.4,
                       help='Minimum distance between objects as ratio of avg size (lower = allow closer, default: 0.4)')
    parser.add_argument('--max-objects', type=int, default=3,
                       help='Maximum number of objects to keep per frame (default: 3)')
    parser.add_argument('--dataset-dir', type=str, default='v4_pdx_cyclist_dataset',
                       help='Directory to save dataset (default: v4_pdx_cyclist_dataset)')
    parser.add_argument('--no-auto-pause', action='store_true',
                       help='Disable auto-pause on motion detection')
    parser.add_argument('--headless-scan', action='store_true',default=True,
                       help='Scan in terminal with progress bar, only show GUI on detections (saves IO/compute)')
    parser.add_argument('--start-percent', type=float, default=0.0,
                       help='Start video at this percentage (0-100, default: 0)')
    parser.add_argument('--timestamps-csv', type=str, default=None,
                       help='Path to CSV file with timestamps to navigate. CSV should have a column with timestamps (seconds) or frame numbers.')
    
    args = parser.parse_args()
    
    # Validate start percentage
    if args.start_percent < 0 or args.start_percent > 100:
        parser.error("--start-percent must be between 0 and 100")
    
    # Create detector
    detector = MovementDetector(
        args.video_path,
        motion_threshold=args.motion_threshold,
        min_area=args.min_area,
        temporal_frames=args.temporal_frames,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_solidity=args.min_solidity,
        min_compactness=args.min_compactness,
        min_distance_ratio=args.min_distance_ratio,
        max_objects=args.max_objects,
        dataset_dir=args.dataset_dir,
        auto_pause=not args.no_auto_pause
    )
    
    # Set start percentage if specified
    if args.start_percent > 0:
        detector.start_percent = args.start_percent
    
    # Load timestamps from CSV if provided
    timestamp_list = None
    if args.timestamps_csv:
        try:
            timestamp_list = detector.load_timestamps_from_csv(args.timestamps_csv)
            if not timestamp_list:
                print(f"Warning: No valid timestamps found in {args.timestamps_csv}")
        except Exception as e:
            print(f"Error loading timestamps CSV: {e}")
            return
    
    # Select region
    if not detector.select_region():
        print("Region selection cancelled.")
        return
    
    # Process video
    timestamps = detector.process_video(
        display=not args.no_display,
        save_output=args.save_video,
        use_edge_refinement=not args.no_edge_refinement,
        headless_scan=args.headless_scan,
        timestamp_list=timestamp_list
    )
    
    # Print results
    detector.print_results()
    
    # Save results if requested
    if args.output:
        detector.save_results(args.output)
    elif timestamps:
        print("\nTip: Use --output to save results to a JSON file")
        print("Tip: Use --save-video to save annotated video with bounding boxes")


if __name__ == '__main__':
    main()

