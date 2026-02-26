import os
from typing import List, Tuple

import numpy as np
import torch
from ultralytics import YOLO

# Type alias for DeepSORT detections: (bbox_xywh, confidence, class_id)
Detection = Tuple[List[int], float, int]


class EnsembleYOLODetections:
    """
    Helper that loads two single-class YOLO models and merges their predictions.

    The typical configuration is:
      - cyclist_model_path: YOLO model trained only on cyclists
      - pedestrian_model_path: YOLO model trained only on pedestrians

    Each single-class model is expected to output only one class internally
    (class index 0). This helper remaps them to global class IDs, e.g.:
      - cyclists -> global class 0
      - pedestrians -> global class 1

    The public API returns detections in DeepSORT's expected format:
        List[(bbox_xywh, confidence, class_id)]
    """

    def __init__(
        self,
        cyclist_model_path: str,
        pedestrian_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cyclist_global_class_id: int = 0,
        pedestrian_global_class_id: int = 1,
    ) -> None:
        if not os.path.exists(cyclist_model_path):
            raise FileNotFoundError(f"Cyclist model not found: {cyclist_model_path}")
        if not os.path.exists(pedestrian_model_path):
            raise FileNotFoundError(f"Pedestrian model not found: {pedestrian_model_path}")

        self.device = device
        self.cyclist_global_class_id = cyclist_global_class_id
        self.pedestrian_global_class_id = pedestrian_global_class_id

        print(f"Loading cyclist model from: {cyclist_model_path}")
        self.cyclist_model = YOLO(cyclist_model_path)
        self.cyclist_model.to(device)

        print(f"Loading pedestrian model from: {pedestrian_model_path}")
        self.pedestrian_model = YOLO(pedestrian_model_path)
        self.pedestrian_model.to(device)

        print(f"Ensemble models loaded successfully on device: {device}")

    def _yolo_results_to_detections(
        self,
        results,
        global_class_id: int,
    ) -> List[Detection]:
        """Convert Ultralytics Results into DeepSORT-style detections."""
        detections: List[Detection] = []

        # Ultralytics returns a list[Results]; we iterate through them
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Batch convert tensors once for efficiency
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            boxes_conf = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), conf in zip(boxes_xyxy, boxes_conf):
                w = x2 - x1
                h = y2 - y1
                bbox_xywh = [int(x1), int(y1), int(w), int(h)]
                detections.append((bbox_xywh, float(conf), global_class_id))

        return detections

    def get_detections(
        self,
        frame: np.ndarray,
        conf: float = 0.5,
        iou: float = 0.5,
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        Run both models on a single frame and return detections.

        Returns:
            cyclist_detections, pedestrian_detections
            where each list contains (bbox_xywh, confidence, class_id).
        """
        # Run the two models independently; they each operate on the same frame.
        # We keep them separate so DeepSORT can use two trackers if desired.
        cyclist_results = self.cyclist_model(
            frame,
            conf=conf,
            iou=iou,
            agnostic_nms=False,
            verbose=False,
        )
        pedestrian_results = self.pedestrian_model(
            frame,
            conf=conf,
            iou=iou,
            agnostic_nms=False,
            verbose=False,
        )

        cyclist_detections = self._yolo_results_to_detections(
            cyclist_results,
            global_class_id=self.cyclist_global_class_id,
        )
        pedestrian_detections = self._yolo_results_to_detections(
            pedestrian_results,
            global_class_id=self.pedestrian_global_class_id,
        )

        return cyclist_detections, pedestrian_detections

