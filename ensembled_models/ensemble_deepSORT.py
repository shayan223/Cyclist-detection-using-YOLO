import cv2
import argparse
import os
import torch
from typing import List

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from ensemble_yolo_models import EnsembleYOLODetections, Detection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def process_video(
    input_video_path: str,
    output_video_path: str,
    ensemble: EnsembleYOLODetections,
    confidence_threshold: float = 0.9,
    max_age: int = 30,
    max_iou_distance: float = 0.7,
    iou_threshold: float = 0.1,
    disable_display: bool = False,
) -> None:
    """Process video using two single-class YOLO models + DeepSORT tracking."""

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {input_video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_delay = int(1000 / fps) if fps > 0 else 33

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

    # Two separate trackers, same as in deepSORT_yolo.py
    try:
        cyclist_tracker = DeepSort(
            max_age=max_age,
            max_iou_distance=max_iou_distance,
            n_init=2,
            embedder="mobilenet",
        )
        pedestrian_tracker = DeepSort(
            max_age=max_age,
            max_iou_distance=max_iou_distance,
            n_init=2,
            embedder="mobilenet",
        )
    except TypeError:
        cyclist_tracker = DeepSort(
            max_age=max_age,
            max_iou_distance=max_iou_distance,
            n_init=2,
        )
        pedestrian_tracker = DeepSort(
            max_age=max_age,
            max_iou_distance=max_iou_distance,
            n_init=2,
        )

    cyclist_ids_seen = set()
    pedestrian_ids_seen = set()

    # Video writer (same robust codec selection as deepSORT_yolo)
    codecs_to_try = [
        ("mp4v", cv2.VideoWriter_fourcc(*"mp4v")),
        ("H264", cv2.VideoWriter_fourcc(*"H264")),
        ("XVID", cv2.VideoWriter_fourcc(*"XVID")),
        ("avc1", cv2.VideoWriter_fourcc(*"avc1")),
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
            "Could not create VideoWriter. Try installing OpenH264 or use a different output path."
        )

    frame_count = 0
    paused = False
    speed_multiplier = 1.0
    annotated_frame = None
    display_available = not disable_display

    # Fixed overlay layout (copied from deepSORT_yolo.py)
    font_scale = 0.8
    font_thickness = 2
    padding = 15
    line_spacing = 5
    max_possible_text = "Pedestrians: 999 (Total: 9999)"
    max_text_size, _ = cv2.getTextSize(
        max_possible_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    max_text_width = max(max_text_size[0], 350)

    sample_text = "Cyclists: 0 (Total: 0)"
    sample_text_size, _ = cv2.getTextSize(
        sample_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    text_height = sample_text_size[1]

    text_x = width - max_text_width - padding
    text_y_pedestrian = height - padding
    text_y_cyclist = text_y_pedestrian - text_height - line_spacing

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

                # Use ensemble helper to get per-class detections
                cyclist_detections, pedestrian_detections = ensemble.get_detections(
                    frame,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                )

                # Update trackers only if there are detections
                cyclist_tracks = (
                    cyclist_tracker.update_tracks(cyclist_detections, frame=frame)
                    if cyclist_detections
                    else []
                )
                pedestrian_tracks = (
                    pedestrian_tracker.update_tracks(pedestrian_detections, frame=frame)
                    if pedestrian_detections
                    else []
                )

                annotated_frame = frame.copy()
                cyclist_count = 0
                pedestrian_count = 0

                confirmed_cyclist_tracks = [
                    t for t in cyclist_tracks if t.is_confirmed()
                ]
                confirmed_pedestrian_tracks = [
                    t for t in pedestrian_tracks if t.is_confirmed()
                ]

                # Cyclist tracks (class 0)
                for track in confirmed_cyclist_tracks:
                    track_id = track.track_id
                    tlbr = track.to_tlbr()
                    x1, y1, x2, y2 = (
                        int(tlbr[0]),
                        int(tlbr[1]),
                        int(tlbr[2]),
                        int(tlbr[3]),
                    )

                    cyclist_ids_seen.add(track_id)
                    cyclist_count += 1

                    cv2.rectangle(
                        annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )

                    label = f"Cyclist #{track_id}"
                    label_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )[0]
                    label_y = y1 - 5
                    label_bg_y1 = y1 - label_size[1] - 10
                    cv2.rectangle(
                        annotated_frame,
                        (x1, label_bg_y1),
                        (x1 + label_size[0], y1),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2,
                    )

                # Pedestrian tracks (class 1)
                for track in confirmed_pedestrian_tracks:
                    track_id = track.track_id
                    tlbr = track.to_tlbr()
                    x1, y1, x2, y2 = (
                        int(tlbr[0]),
                        int(tlbr[1]),
                        int(tlbr[2]),
                        int(tlbr[3]),
                    )

                    pedestrian_ids_seen.add(track_id)
                    pedestrian_count += 1

                    cv2.rectangle(
                        annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2
                    )

                    label = f"Pedestrian #{track_id}"
                    label_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )[0]
                    label_y = y1 - 5
                    label_bg_y1 = y1 - label_size[1] - 10
                    cv2.rectangle(
                        annotated_frame,
                        (x1, label_bg_y1),
                        (x1 + label_size[0], y1),
                        (255, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                if frame_count % 10 == 0:
                    print(
                        f"Frame {frame_count}: {cyclist_count} cyclist(s), {pedestrian_count} pedestrian(s) | "
                        f"Total unique: {len(cyclist_ids_seen)} cyclists, {len(pedestrian_ids_seen)} pedestrians"
                    )

                cyclist_text = (
                    f"Cyclists: {cyclist_count} (Total: {len(cyclist_ids_seen)})"
                )
                pedestrian_text = (
                    f"Pedestrians: {pedestrian_count} (Total: {len(pedestrian_ids_seen)})"
                )

                cv2.rectangle(
                    annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1
                )

                cv2.putText(
                    annotated_frame,
                    cyclist_text,
                    (text_x, text_y_cyclist),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    font_thickness,
                )
                cv2.putText(
                    annotated_frame,
                    pedestrian_text,
                    (text_x, text_y_pedestrian),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 0, 0),
                    font_thickness,
                )

                out.write(annotated_frame)

                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(
                        f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)"
                    )

            if display_available and annotated_frame is not None:
                try:
                    cv2.imshow(
                        "Cyclist & Pedestrian Detection (Ensembled) - Live View",
                        annotated_frame,
                    )
                except (cv2.error, Exception):
                    display_available = False
                    print(
                        "Display not available. Use --no-display to skip GUI. Continuing without live view."
                    )

            actual_delay = (
                max(1, int(frame_delay / speed_multiplier)) if display_available else 1
            )

            key = cv2.waitKey(actual_delay) & 0xFF if display_available else 0
            if key == ord("q"):
                print("\nQuit requested by user")
                break
            elif key == ord("s"):
                frame_filename = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(frame_filename, annotated_frame)
                print(f"Frame saved as: {frame_filename}")
            elif key == ord("p"):
                paused = not paused
                status = "Paused" if paused else "Resumed"
                print(f"Video {status}")
            elif key == ord("+") or key == ord("="):
                speed_multiplier = min(5.0, speed_multiplier + 0.5)
                print(f"Speed increased to {speed_multiplier:.1f}x")
            elif key == ord("-"):
                speed_multiplier = max(0.1, speed_multiplier - 0.5)
                print(f"Speed decreased to {speed_multiplier:.1f}x")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")

    finally:
        cap.release()
        out.release()
        if display_available:
            try:
                cv2.destroyAllWindows()
            except (cv2.error, Exception):
                pass
        print("\nVideo processing completed!")
        print(f"Output saved to: {output_video_path}")
        print(f"Total unique cyclists tracked: {len(cyclist_ids_seen)}")
        print(f"Total unique pedestrians tracked: {len(pedestrian_ids_seen)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track cyclists and pedestrians using two single-class YOLO models "
            "and DeepSORT (ensemble pipeline)."
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        required=False,
        default="../trim_3.mp4",
        help="Input video file path.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output video file path (default: input_ENSEMBLE_DEEPSort_tracked.mp4).",
    )

    # Default paths assume you trained with train_ensembled_yolo.py defaults
    parser.add_argument(
        "--cyclist-model",
        type=str,
        default='./runs/detect/ensembled_models/runs/cyclist_single_class10/weights/best.pt',
        help=(
            "Path to cyclist-only YOLO model weights "
            "(default: ensembled_models/runs/cyclist_single_class/weights/best.pt)."
        ),
    )
    parser.add_argument(
        "--pedestrian-model",
        type=str,
        default='./runs/detect/ensembled_models/runs/pedestrian_single_class6/weights/best.pt',
        help=(
            "Path to pedestrian-only YOLO model weights "
            "(default: ensembled_models/runs/pedestrian_single_class/weights/best.pt)."
        ),
    )

    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.85,
        help="Confidence threshold for both models (0.0-1.0).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold for both models (0.0-1.0).",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=15,
        help="Maximum frames to keep a track without update.",
    )
    parser.add_argument(
        "--max-iou-distance",
        type=float,
        default=0.7,
        help="Maximum IOU distance for track association.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live display (faster processing or for headless environments).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input video file '{args.input}' not found")
        return

    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_ENSEMBLE_DEEPSort_tracked.mp4"

    if not os.path.exists(args.cyclist_model):
        print(f"Error: Cyclist model file '{args.cyclist_model}' not found")
        return
    if not os.path.exists(args.pedestrian_model):
        print(f"Error: Pedestrian model file '{args.pedestrian_model}' not found")
        return

    ensemble = EnsembleYOLODetections(
        cyclist_model_path=args.cyclist_model,
        pedestrian_model_path=args.pedestrian_model,
        device=DEVICE,
        cyclist_global_class_id=0,
        pedestrian_global_class_id=1,
    )

    print("Starting ensemble video analysis with DeepSORT tracking...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Cyclist model: {args.cyclist_model}")
    print(f"Pedestrian model: {args.pedestrian_model}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"NMS IoU threshold: {args.iou}")
    print(f"Max age: {args.max_age} frames")
    print(f"Max IOU distance: {args.max_iou_distance}")
    print(f"Display: {'Disabled' if args.no_display else 'Enabled'}")

    process_video(
        args.input,
        args.output,
        ensemble,
        confidence_threshold=args.confidence,
        max_age=args.max_age,
        max_iou_distance=args.max_iou_distance,
        iou_threshold=args.iou,
        disable_display=args.no_display,
    )


if __name__ == "__main__":
    main()

