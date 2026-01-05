import cv2
import numpy as np
import argparse
from collections import deque
import json
from datetime import timedelta

class MovementDetector:
    def __init__(self, video_path, motion_threshold=30, min_area=100, buffer_size=10, 
                 temporal_frames=3, min_aspect_ratio=0.2, max_aspect_ratio=5.0,
                 min_solidity=0.4, min_compactness=0.25, min_distance_ratio=0.6, max_objects=2):
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
        
    def reset_video(self):
        """Reset video to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_number = 0
        
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
    
    def filter_objects_by_tracking(self, bounding_boxes, frame_number):
        """
        Filter objects by tracking them across frames. Objects that don't persist
        or move erratically are likely artifacts.
        
        Args:
            bounding_boxes: List of (x, y, w, h, area, solidity, compactness) tuples
            frame_number: Current frame number
        
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
                max_movement = max(w, h) * 2  # Allow movement up to 2x object size
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
    
    def process_video(self, display=True, save_output=None, use_edge_refinement=False):
        """
        Process video and record movement timestamps.
        
        Args:
            display: Whether to display the video during processing
            save_output: Path to save output video (optional)
            use_edge_refinement: Use edge detection to refine bounding boxes
        """
        if not self.polygon_complete or self.mask is None:
            print("Error: No region selected. Please select a region first.")
            return
        
        self.reset_video()
        self.movement_timestamps = []
        self.last_movement_frame = -1
        self.motion_history.clear()  # Reset temporal filtering history
        self.object_tracks.clear()  # Reset object tracking
        self.next_track_id = 0
        
        # Setup video writer if saving output
        out_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(save_output, fourcc, self.fps, (self.width, self.height))
        
        print("\n=== Processing Video ===")
        print(f"Video: {self.video_path}")
        print(f"FPS: {self.fps:.2f}")
        print(f"Total frames: {self.total_frames}")
        print(f"Duration: {self.total_frames / self.fps:.2f} seconds")
        if use_edge_refinement:
            print("Edge refinement: ENABLED")
        print(f"Temporal filtering: {self.temporal_frames} frames (motion must persist)")
        print(f"Aspect ratio filter: {self.min_aspect_ratio:.1f} - {self.max_aspect_ratio:.1f}")
        print(f"Object tracking: ENABLED (filters artifacts by persistence)")
        print(f"Contour quality: ENABLED (solidity & compactness filtering)")
        print(f"Adjacent artifact filter: ENABLED (removes small artifacts near larger objects)")
        print(f"Spatial clustering: ENABLED (removes clustered artifacts, max {self.max_objects} objects)")
        print(f"Contour quality thresholds: solidity > {self.min_solidity:.2f}, compactness > {self.min_compactness:.2f}")
        print("\nControls:")
        print("  Space: Pause/Resume (freezes frame with bounding boxes)")
        print("  'r': Reset to beginning")
        print("  'e': Toggle edge refinement")
        print("  'q': Quit")
        print("=======================\n")
        
        frame_count = 0
        paused = False
        current_frame = None
        paused_frame = None  # Store the paused frame with bounding boxes
        paused_bounding_boxes = []  # Store bounding boxes when paused
        paused_display_motion = False  # Store motion state when paused
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_count += 1
                self.current_frame_number = frame_count
                current_frame = frame.copy()
                
                # Detect motion and get bounding boxes
                motion_detected, motion_area, bounding_boxes = self.detect_motion_in_region(
                    frame, use_edge_refinement=use_edge_refinement
                )
                
                # Apply object tracking filter - only keep objects that persist across frames
                if bounding_boxes:
                    bounding_boxes = self.filter_objects_by_tracking(bounding_boxes, frame_count)
                    # Update motion detection based on filtered boxes
                    if not bounding_boxes:
                        motion_detected = False
                        motion_area = 0
                
                # Apply adjacent artifact filter - remove small artifacts near larger objects
                if bounding_boxes and len(bounding_boxes) > 1:
                    bounding_boxes = self.filter_adjacent_artifacts(bounding_boxes, size_ratio_threshold=0.3)
                    # Update motion detection based on filtered boxes
                    if not bounding_boxes:
                        motion_detected = False
                        motion_area = 0
                
                # Apply spatial clustering filter - remove clustered artifacts
                if bounding_boxes:
                    bounding_boxes = self.filter_spatial_clusters(bounding_boxes, 
                                                                  min_distance_ratio=self.min_distance_ratio, 
                                                                  max_objects=self.max_objects)
                    # Update motion detection based on filtered boxes
                    if not bounding_boxes:
                        motion_detected = False
                        motion_area = 0
                
                # Apply temporal filtering - motion must persist across multiple frames
                persistent_motion = self.check_temporal_consistency(motion_detected)
                
                # Only display and record bounding boxes when motion meets all criteria
                # (temporal persistence, aspect ratio, area, tracking, etc.)
                display_motion = persistent_motion  # Only show boxes if motion meets all criteria
                record_motion = persistent_motion  # Only record if motion persists
            else:
                # When paused, use the stored paused frame and bounding boxes
                if paused_frame is None:
                    break
                frame = paused_frame.copy()
                bounding_boxes = paused_bounding_boxes
                display_motion = paused_display_motion
                record_motion = False  # Don't record when paused
            
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
                print(f"Movement detected at frame {frame_count} (time: {timestamp:.2f}s, area: {motion_area:.0f}, objects: {len(bounding_boxes)})")
            
            if display or save_output:
                # Create display frame
                display_frame = frame.copy()
                
                # Draw polygon
                if self.mask is not None:
                    pts = np.array(self.polygon_points, dtype=np.int32)
                    overlay = display_frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                    cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)
                
                # Draw bounding boxes around moving objects (only when motion is detected)
                if display_motion:
                    for i, (x, y, w, h) in enumerate(bounding_boxes):
                        # Draw rectangle
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        
                        # Draw label with object number
                        label = f"Object {i+1}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        label_y = max(y - 10, label_size[1] + 10)
                        
                        # Draw label background
                        cv2.rectangle(display_frame, 
                                    (x, label_y - label_size[1] - 5), 
                                    (x + label_size[0] + 5, label_y + 5), 
                                    (255, 0, 0), -1)
                        
                        # Draw label text
                        cv2.putText(display_frame, label, (x + 2, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw motion detection info
                if paused:
                    status_text = "PAUSED"
                    color = (0, 255, 255)  # Cyan for paused
                    cv2.putText(display_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                else:
                    status_text = "MOTION DETECTED!" if display_motion else "No motion"
                    color = (0, 165, 255) if display_motion else (0, 255, 0)  # Orange when criteria met
                    cv2.putText(display_frame, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(display_frame, f"Frame: {frame_count}/{self.total_frames}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Time: {frame_count/self.fps:.2f}s", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Detections: {len(self.movement_timestamps)}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Show object count (only when motion meets criteria)
                if display_motion:
                    cv2.putText(display_frame, f"Objects: {len(bounding_boxes)}", 
                               (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(display_frame, f"Objects: 0", 
                               (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if save_output:
                    cv2.putText(display_frame, "Recording...", (10, self.height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    out_writer.write(display_frame)
                
                if display:
                    cv2.imshow("Movement Detection", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        paused = not paused
                        if paused:
                            # Store current frame with bounding boxes when pausing
                            paused_frame = display_frame.copy()
                            paused_bounding_boxes = bounding_boxes.copy() if bounding_boxes else []
                            paused_display_motion = display_motion
                            print("Paused - Frame frozen with bounding boxes")
                        else:
                            # Clear paused frame when resuming
                            paused_frame = None
                            paused_bounding_boxes = []
                            paused_display_motion = False
                            print("Resumed")
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
                        print("Reset to beginning")
                    elif key == ord('e'):
                        use_edge_refinement = not use_edge_refinement
                        print(f"Edge refinement: {'ENABLED' if use_edge_refinement else 'DISABLED'}")
        
        if display:
            cv2.destroyAllWindows()
        
        if out_writer:
            out_writer.release()
            print(f"Output video saved to: {save_output}")
        
        print(f"\n=== Processing Complete ===")
        print(f"Total movement events detected: {len(self.movement_timestamps)}")
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
    parser.add_argument('--edge-refinement', action='store_true',# default=True,
                       help='Use edge detection to refine bounding boxes')
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
    
    args = parser.parse_args()
    
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
        max_objects=args.max_objects
    )
    
    # Select region
    if not detector.select_region():
        print("Region selection cancelled.")
        return
    
    # Process video
    timestamps = detector.process_video(
        display=not args.no_display,
        save_output=args.save_video,
        use_edge_refinement=args.edge_refinement
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

