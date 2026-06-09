"""
Post Encroachment Time (PET) conflict zone detection.
Uses a grid over the frame to detect when pedestrians and cyclists share or recently
shared the same zone within a time window, computes PET, and outputs conflict events to CSV
and a video with highlighted conflict zones.

PET definition (standard): PET(A1,A2,CA) = t_entry(A2,CA) - t_exit(A1,CA), the time gap
between one actor leaving and the other entering the conflict area; scale [0, inf) s.
PET is undefined when both occupy the conflict area before either leaves (overlap).
Reference: https://criticality-metrics.readthedocs.io/en/latest/time-scale/PET.html
"""
import cv2
import argparse
import os
from collections import defaultdict
from ultralytics import YOLO, RTDETR
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Configuration ---

# --- Configuration ---
BATCH = 8
#'./50epoch_yolo_finetune_pdx3/weights/best.pt'
DEFAULT_MODEL_PATH =  './RT_DETR/runs/detect/cyclist_detection_rtdetr/rtdetr_finetune_imgsz640_mos1.0_mix0.15_fliplr0.52/weights/best.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _is_rtdetr_path(model_path):
    """Return True if path suggests an RT-DETR model."""
    path_lower = model_path.replace('\\', '/').lower()
    return 'rtdetr' in path_lower


def load_model(model_path, device, use_rtdetr=None):
    """Load YOLO or RT-DETR model. If use_rtdetr is None, auto-detect from path."""
    if use_rtdetr is None:
        use_rtdetr = _is_rtdetr_path(model_path)
    if use_rtdetr:
        model = RTDETR(model_path)
    else:
        model = YOLO(model_path)
    model.to(device)
    return model


def _bbox_overlap_cells(bbox_xyxy, width, height, grid_rows, grid_cols):
    """Return set of (row, col) grid cell indices that the bbox overlaps.
    bbox_xyxy: (x1, y1, x2, y2) in pixel coordinates.
    """
    x1, y1, x2, y2 = bbox_xyxy
    cell_w = width / grid_cols
    cell_h = height / grid_rows
    cells = set()
    # Cells that the bbox intersects (using center is too strict; use overlap)
    col_start = max(0, int(x1 / cell_w))
    col_end = min(grid_cols - 1, int(x2 / cell_w))
    row_start = max(0, int(y1 / cell_h))
    row_end = min(grid_rows - 1, int(y2 / cell_h))
    for r in range(row_start, row_end + 1):
        for c in range(col_start, col_end + 1):
            cells.add((r, c))
    return cells


def _get_pet_output_dir(input_video_path, root="PET_Analysis"):
    """
    Return (output_dir_path, run_number) for this video.
    output_dir = root / video_basename_N where N is the next available run (1, 2, 3, ...).
    Creates root if needed; does not create the run subdir (caller creates when writing).
    """
    video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
    if not video_basename:
        video_basename = "video"
    os.makedirs(root, exist_ok=True)
    prefix = video_basename + "_"
    run_number = 1
    for name in os.listdir(root):
        if name.startswith(prefix) and os.path.isdir(os.path.join(root, name)):
            try:
                n = int(name[len(prefix):])
                if n >= run_number:
                    run_number = n + 1
            except ValueError:
                continue
    run_dir_name = f"{video_basename}_{run_number}"
    output_dir = os.path.join(root, run_dir_name)
    return output_dir, run_number


def _neighbor_cells(r, c, grid_rows, grid_cols, include_self=True):
    """Return set of (row, col) for cell (r,c) and its 3x3 neighbors."""
    out = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not include_self and dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
                out.add((nr, nc))
    return out


def _select_grid_cell_for_pet(frame, grid_rows, grid_cols, window_name="Select PET grid cell"):
    """
    Let the user select a single grid cell by clicking on the frame.
    Returns (row, col) or None if selection fails or is cancelled.
    """
    height, width = frame.shape[:2]
    cell_w = width / grid_cols
    cell_h = height / grid_rows
    selected = {'cell': None}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            c = int(x / cell_w)
            r = int(y / cell_h)
            if 0 <= r < grid_rows and 0 <= c < grid_cols:
                selected['cell'] = (r, c)

    try:
        display = frame.copy()
        for i in range(1, grid_rows):
            y = int(i * cell_h)
            cv2.line(display, (0, y), (width, y), (60, 60, 60), 1)
        for j in range(1, grid_cols):
            x = int(j * cell_w)
            cv2.line(display, (x, 0), (x, height), (60, 60, 60), 1)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, _on_mouse)

        while True:
            img = display.copy()
            if selected['cell'] is not None:
                r, c = selected['cell']
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.imshow(window_name, img)
            key = cv2.waitKey(20) & 0xFF
            if selected['cell'] is not None and key in (13, 32, ord('q'), 27):
                break
        cv2.destroyWindow(window_name)
        return selected['cell']
    except (cv2.error, Exception):
        try:
            cv2.destroyWindow(window_name)
        except (cv2.error, Exception):
            pass
        return None


def _build_heatmap_image(cell_pet_values, grid_rows, grid_cols, width, height, fps, max_pet_time, first_frame):
    """
    Build a heatmap: average standard PET per cell (PET in [0, inf), time gap).
    Low PET = more critical = red; high PET = less critical = blue.
    Returns BGR image (faint overlay over first_frame).
    """
    max_sec = max_pet_time / fps if fps > 0 else 1.0
    # Grid of average standard PET (seconds) per cell; NaN where no data
    heat = np.full((grid_rows, grid_cols), np.nan, dtype=np.float64)
    for (r, c), values in cell_pet_values.items():
        if 0 <= r < grid_rows and 0 <= c < grid_cols and values:
            # Standard PET = |signed|; overlap (0) is most critical
            heat[r, c] = np.mean([abs(v) for v in values])

    # Normalize: standard PET in [0, max_sec] -> 0 = red (critical), high = blue (safe)
    heat_uint8 = np.full((grid_rows, grid_cols), 128, dtype=np.uint8)  # no-data = 128 (gray)
    valid = ~np.isnan(heat)
    if np.any(valid):
        v = np.clip(heat[valid], 0.0, max_sec)
        norm = v / max_sec  # 0 -> 0, max_sec -> 1
        heat_uint8[valid] = (255 * (1.0 - norm)).clip(0, 255).astype(np.uint8)  # 0 PET -> 255 (red), high -> 0 (blue)

    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.resize(heat_color, (width, height), interpolation=cv2.INTER_NEAREST)

    # Faint overlay: blend heatmap only where we have data (no-data cells stay as original frame)
    mask = (heat_uint8 != 128)
    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    blended = cv2.addWeighted(heat_color, 0.45, first_frame, 0.55, 0)
    overlay = np.where(mask[:, :, np.newaxis], blended, first_frame).astype(np.uint8)
    return overlay


def process_video(
    input_video_path,
    output_video_path,
    model,
    confidence_threshold=0.9,
    max_age=30,
    max_iou_distance=0.7,
    iou_threshold=0.1,
    grid_size=10,
    max_pet_time=30,
    use_neighbors=True,
    output_csv_path=None,
    output_heatmap_path=None,
    disable_display=True,
    show_grid=False,
    single_cell_mode=False,
):
    """
    Process video: detect and track cyclists/pedestrians, maintain grid occupancy
    timestamps, detect conflict zones (PET), write output video and CSV.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {input_video_path}")

    grid_rows = grid_cols = grid_size
    selected_cell = None

    if single_cell_mode:
        ret_sel, frame_sel = cap.read()
        if ret_sel and frame_sel is not None:
            tqdm.write(
                "Select a grid cell by clicking on the frame; "
                "press Enter/Space or q/ESC to confirm."
            )
            chosen = _select_grid_cell_for_pet(frame_sel, grid_rows, grid_cols)
            if chosen is not None:
                selected_cell = chosen
                tqdm.write(
                    f"Using single grid cell (row={selected_cell[0]}, col={selected_cell[1]}) "
                    "for PET computation."
                )
            else:
                tqdm.write("No grid cell selected; falling back to full-grid PET computation.")
        else:
            tqdm.write("Could not read first frame for grid selection; falling back to full-grid PET computation.")

        cap.release()
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not re-open video file {input_video_path} after grid selection")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cell_w = width / grid_cols
    cell_h = height / grid_rows

    try:
        cyclist_tracker = DeepSort(
            max_age=max_age,
            max_iou_distance=max_iou_distance,
            n_init=2,
            embedder='mobilenet',
        )
        pedestrian_tracker = DeepSort(
            max_age=max_age,
            max_iou_distance=max_iou_distance,
            n_init=2,
            embedder='mobilenet',
        )
    except TypeError:
        cyclist_tracker = DeepSort(max_age=max_age, max_iou_distance=max_iou_distance, n_init=2)
        pedestrian_tracker = DeepSort(max_age=max_age, max_iou_distance=max_iou_distance, n_init=2)

    # grid[(r,c)] = {'pedestrian': [(frame, track_id), ...], 'cyclist': [(frame, track_id), ...]}
    grid_occupancy = defaultdict(lambda: {'pedestrian': [], 'cyclist': []})

    # Conflict events for CSV: list of dicts
    conflict_events = []

    # Per-cell signed PET values for heatmap (average PET per cell); negative = overlap = higher risk
    cell_pet_values = defaultdict(list)

    # Per-frame PET values for "average PET over time" plot
    frame_to_pets = defaultdict(list)

    # Only record when a cell newly enters conflict (not every frame) to get one row per incident
    conflict_cells_previous_frame = set()

    ext = os.path.splitext(output_video_path)[1].lower()
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
    ]
    out = None
    for _codec_name, fourcc in codecs_to_try:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if out.isOpened():
            break
        out.release()
        out = None
    if out is None:
        raise RuntimeError("Could not create VideoWriter.")

    frame_count = 0
    display_available = not disable_display
    pbar = tqdm(total=total_frames if total_frames > 0 else None, unit="frame", desc="PET analysis")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                agnostic_nms=False,
                verbose=False,
            )

            boxes_data = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    boxes_xyxy = boxes.xyxy.cpu().numpy()
                    boxes_conf = boxes.conf.cpu().numpy()
                    boxes_cls = boxes.cls.cpu().numpy()
                    boxes_data.extend(zip(boxes_xyxy, boxes_conf, boxes_cls))

            cyclist_detections = []
            pedestrian_detections = []
            for (x1, y1, x2, y2), conf, cls in boxes_data:
                cls_int = int(cls)
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                if cls_int == 0:
                    cyclist_detections.append((bbox, float(conf), cls_int))
                elif cls_int == 1:
                    pedestrian_detections.append((bbox, float(conf), cls_int))

            cyclist_tracks = cyclist_tracker.update_tracks(cyclist_detections, frame=frame)
            pedestrian_tracks = pedestrian_tracker.update_tracks(pedestrian_detections, frame=frame)

            confirmed_cyclist = [t for t in cyclist_tracks if t.is_confirmed()]
            confirmed_pedestrian = [t for t in pedestrian_tracks if t.is_confirmed()]

            # Update grid: which cells does each track overlap this frame?
            for track in confirmed_cyclist:
                tlbr = track.to_tlbr()
                cells = _bbox_overlap_cells(
                    (tlbr[0], tlbr[1], tlbr[2], tlbr[3]),
                    width, height, grid_rows, grid_cols,
                )
                for (r, c) in cells:
                    grid_occupancy[(r, c)]['cyclist'].append((frame_count, track.track_id))

            for track in confirmed_pedestrian:
                tlbr = track.to_tlbr()
                cells = _bbox_overlap_cells(
                    (tlbr[0], tlbr[1], tlbr[2], tlbr[3]),
                    width, height, grid_rows, grid_cols,
                )
                for (r, c) in cells:
                    grid_occupancy[(r, c)]['pedestrian'].append((frame_count, track.track_id))

            # Prune old timestamps (older than current_frame - max_pet_time)
            cutoff = frame_count - max_pet_time
            for key in list(grid_occupancy.keys()):
                grid_occupancy[key]['pedestrian'] = [
                    (f, tid) for (f, tid) in grid_occupancy[key]['pedestrian'] if f > cutoff
                ]
                grid_occupancy[key]['cyclist'] = [
                    (f, tid) for (f, tid) in grid_occupancy[key]['cyclist'] if f > cutoff
                ]
                if not grid_occupancy[key]['pedestrian'] and not grid_occupancy[key]['cyclist']:
                    del grid_occupancy[key]

            # Detect conflict: for each cell (and optionally its neighbors), check if both
            # pedestrian and cyclist have timestamps in the window; compute PET and record.
            conflict_cells_this_frame = set()
            if selected_cell is not None:
                cells_to_check = {selected_cell} if selected_cell in grid_occupancy else set()
            else:
                cells_to_check = set(grid_occupancy.keys())

            for (r, c) in list(cells_to_check):
                if (r, c) in conflict_cells_this_frame:
                    continue
                # Aggregate timestamps from this cell and optionally its 3x3 neighbors
                ped_list = []
                cyc_list = []
                if use_neighbors and not single_cell_mode:
                    cells_region = _neighbor_cells(r, c, grid_rows, grid_cols, include_self=True)
                else:
                    cells_region = {(r, c)}
                for (nr, nc) in cells_region:
                    if (nr, nc) not in grid_occupancy:
                        continue
                    ped_list.extend(grid_occupancy[(nr, nc)]['pedestrian'])
                    cyc_list.extend(grid_occupancy[(nr, nc)]['cyclist'])

                if not ped_list or not cyc_list:
                    continue

                # Post Encroachment Time (PET): time gap between one actor leaving and the other
                # entering the conflict area (CA). Definition: PET(A1,A2,CA) = t_entry(A2,CA) - t_exit(A1,CA)
                # with t_entry(A2) >= t_exit(A1); scale [0, inf) s. PET undefined when both occupy CA
                # before either leaves (overlap). See: https://criticality-metrics.readthedocs.io/en/latest/time-scale/PET.html
                # We use frame indices as proxies for presence in the cell; minimum gap over (ped,cyclist)
                # pairs gives PET = |t2 - t1|/fps (standard PET, non-negative). Overlap = same frame -> PET undefined (we output 0).
                pet_frames = None
                signed_pet_frames = None
                for (fp, idp) in ped_list:
                    for (fc, idc) in cyc_list:
                        d = abs(fp - fc)
                        if pet_frames is None or d < pet_frames:
                            pet_frames = d
                            # Order: positive = ped exited before cyc entered (PET = (fc-fp)/fps); negative = overlap / cyc first
                            signed_pet_frames = fp - fc
                            best_ped_id = idp
                            best_cyclist_id = idc

                if pet_frames is None:
                    continue

                conflict_cells_this_frame.add((r, c))
                # Standard PET (non-negative, seconds): time between one leaving and other entering
                pet_seconds = (pet_frames / fps) if fps > 0 else 0.0
                signed_pet_seconds = (signed_pet_frames / fps) if (fps > 0 and signed_pet_frames is not None) else 0.0
                time_sec = frame_count / fps if fps > 0 else 0.0
                overlap = pet_frames == 0  # PET undefined per definition when both in CA
                frame_to_pets[frame_count].append(signed_pet_seconds)

                # Record event when this cell newly enters conflict (not already in previous frame)
                if (r, c) not in conflict_cells_previous_frame:
                    conflict_events.append({
                        'frame': frame_count,
                        'time_sec': round(time_sec, 3),
                        'cell_row': r,
                        'cell_col': c,
                        'pet_frames': pet_frames,
                        'pet_seconds': round(pet_seconds, 3),
                        'pet_undefined_overlap': overlap,
                        'signed_pet_frames': signed_pet_frames,
                        'signed_pet_seconds': round(signed_pet_seconds, 3),
                        'pedestrian_id': best_ped_id,
                        'cyclist_id': best_cyclist_id,
                    })
                    cell_pet_values[(r, c)].append(signed_pet_seconds)

            # Build annotated frame: grid (faint), bboxes, conflict zones
            annotated_frame = frame.copy()

            # Draw grid (faint) when flag is set (for tuning grid size N)
            if show_grid:
                for i in range(1, grid_rows):
                    y = int(i * cell_h)
                    cv2.line(annotated_frame, (0, y), (width, y), (60, 60, 60), 1)
                for j in range(1, grid_cols):
                    x = int(j * cell_w)
                    cv2.line(annotated_frame, (x, 0), (x, height), (60, 60, 60), 1)

            # Highlight conflict zones (semi-transparent red)
            for (r, c) in conflict_cells_this_frame:
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw cyclist boxes (green)
            for track in confirmed_cyclist:
                tlbr = track.to_tlbr()
                x1, y1, x2, y2 = int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"C#{track.track_id}"
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

            # Draw pedestrian boxes (blue)
            for track in confirmed_pedestrian:
                tlbr = track.to_tlbr()
                x1, y1, x2, y2 = int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"P#{track.track_id}"
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                )

            # Conflict count this frame
            if conflict_cells_this_frame:
                cv2.putText(
                    annotated_frame, f"CONFLICT ZONES: {len(conflict_cells_this_frame)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )

            out.write(annotated_frame)
            conflict_cells_previous_frame = conflict_cells_this_frame
            frame_count += 1
            pbar.update(1)

            if display_available and annotated_frame is not None:
                try:
                    cv2.imshow('PET Conflict Detection', annotated_frame)
                except (cv2.error, Exception):
                    display_available = False
            if display_available:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        tqdm.write("Interrupted by user")
    finally:
        pbar.close()
        cap.release()
        out.release()
        if display_available:
            try:
                cv2.destroyAllWindows()
            except (cv2.error, Exception):
                pass

    # CSV output
    base = os.path.splitext(output_video_path)[0]
    if output_csv_path is None:
        output_csv_path = base + "_PET_conflicts.csv"

    df = pd.DataFrame(conflict_events)
    if not df.empty:
        df.to_csv(output_csv_path, index=False)

    # Heatmap image: average signed PET per cell (negative = overlap = higher risk = red)
    if output_heatmap_path is None:
        output_heatmap_path = base + "_PET_heatmap.png"
    cap_heat = cv2.VideoCapture(input_video_path)
    if cap_heat.isOpened():
        ret, first_frame = cap_heat.read()
        cap_heat.release()
        if ret and first_frame is not None and cell_pet_values:
            heatmap_img = _build_heatmap_image(
                cell_pet_values, grid_rows, grid_cols, width, height, fps, max_pet_time, first_frame
            )
            cv2.imwrite(output_heatmap_path, heatmap_img)

    # Plot: average standard PET over time (PET = time gap [0, inf) s; lower = more critical)
    output_plot_path = base + "_PET_over_time.png"
    if frame_to_pets:
        # Build series covering the full processed video duration (all frames 0..frame_count-1)
        frames_all = np.arange(frame_count, dtype=int)
        time_sec_arr = frames_all.astype(float) / fps if fps > 0 else frames_all.astype(float)
        # Standard PET = |signed| per value; average per frame; NaN where no PET events
        avg_pet_arr = np.full_like(time_sec_arr, np.nan, dtype=float)
        for f, vals in frame_to_pets.items():
            if 0 <= f < frame_count and vals:
                avg_pet_arr[f] = np.mean([abs(p) for p in vals])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_sec_arr, avg_pet_arr, color="steelblue", linewidth=1)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, None)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Average PET (s)")
        ax.set_title("Average PET over time (lower = more critical; PET = time gap between one leaving and other entering)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_plot_path, dpi=150)
        plt.close(fig)

    # Plot: Risk = 1/(1+PET); higher risk when PET is low (critical). Standard definition: lower PET = more critical.
    output_risk_plot_path = base + "_Risk_PET_over_time.png"
    if frame_to_pets:
        frames_sorted = sorted(frame_to_pets.keys())
        time_sec_arr = np.array(frames_sorted, dtype=float) / fps if fps > 0 else np.array(frames_sorted)
        # Risk = 1/(1+PET_standard); PET=0 (or overlap) -> risk=1, large PET -> risk~0
        risk_per_frame = np.array([
            np.mean([1.0 / (1.0 + abs(p)) for p in frame_to_pets[f]]) for f in frames_sorted
        ])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_sec_arr, risk_per_frame, color="crimson", linewidth=1)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Risk 1/(1+PET)")
        ax.set_title("Risk over time (higher = lower PET = more critical)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_risk_plot_path, dpi=150)
        plt.close(fig)

    tqdm.write(f"Video:   {os.path.abspath(output_video_path)}")
    tqdm.write(f"CSV:     {os.path.abspath(output_csv_path)}")
    tqdm.write(f"Heatmap: {os.path.abspath(output_heatmap_path)}")
    if frame_to_pets:
        tqdm.write(f"Plot:    {os.path.abspath(output_plot_path)}")
        tqdm.write(f"Risk:    {os.path.abspath(output_risk_plot_path)}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='PET conflict zone detection: grid-based pedestrian/cyclist overlap with Post Encroachment Time'
    )
    parser.add_argument('--input', '-i', required=False, default='trim5.mp4', help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path (default: input_PET_conflicts.mp4)')
    parser.add_argument('--csv', help='Output CSV path (default: output base + _PET_conflicts.csv)')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL_PATH, help='Model path (YOLO or RT-DETR .pt)')
    parser.add_argument('--yolo', action='store_true', help='Force YOLO backend')
    parser.add_argument('--rtdetr', action='store_true', help='Force RT-DETR backend')
    parser.add_argument('--confidence', '-c', type=float, default=0.70, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-age', type=int, default=25, help='Max frames to keep track without update')
    parser.add_argument('--max-iou-distance', type=float, default=0.7, help='Max IOU distance for association')
    parser.add_argument('--grid-size', type=int, default=100, help='N for NxN grid')
    parser.add_argument('--max-pet-time', type=int, default=10, help='Max frames to keep occupancy (conflict window)')
    parser.add_argument('--no-neighbors', action='store_true', help='Do not extend conflict to neighbor cells')
    parser.add_argument('--show-grid', action='store_true', help='Draw faint grid lines on output video (for tuning --grid-size N)')
    parser.add_argument(
        '--no-grid',
        action='store_true',
        help='Use only a single user-selected grid cell for PET (disables neighbor-based conflicts)',
    )
    parser.add_argument('--heatmap', metavar='PATH', help='Output heatmap image path (default: output base + _PET_heatmap.png)')
    parser.add_argument('--display', action='store_true', help='Show live window (default: off for speed)')

    args = parser.parse_args()

    use_rtdetr = None
    if args.rtdetr:
        use_rtdetr = True
    if args.yolo:
        use_rtdetr = False
    if args.yolo and args.rtdetr:
        use_rtdetr = True

    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found")
        return
    if args.output is None:
        output_dir, run_number = _get_pet_output_dir(args.input)
        os.makedirs(output_dir, exist_ok=True)
        video_basename = os.path.splitext(os.path.basename(args.input))[0] or "video"
        args.output = os.path.join(output_dir, f"{video_basename}.mp4")
        tqdm.write(f"Results directory: {os.path.abspath(output_dir)} (run {run_number})")
    if not os.path.exists(args.model):
        print(f"Error: Model '{args.model}' not found")
        return

    model = load_model(args.model, DEVICE, use_rtdetr=use_rtdetr)
    process_video(
        args.input,
        args.output,
        model,
        confidence_threshold=args.confidence,
        max_age=args.max_age,
        max_iou_distance=args.max_iou_distance,
        iou_threshold=args.iou,
        grid_size=args.grid_size,
        max_pet_time=args.max_pet_time,
        use_neighbors=not args.no_neighbors,
        output_csv_path=args.csv,
        output_heatmap_path=args.heatmap,
        disable_display=not args.display,
        show_grid=args.show_grid,
        single_cell_mode=args.no_grid,
    )


if __name__ == "__main__":
    main()
