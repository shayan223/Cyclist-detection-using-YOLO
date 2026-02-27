#!/usr/bin/env python3
"""
Pre-generate augmented training samples to match the same variety of augmentations
as Ultralytics YOLO (fine_tune_yolo.py). Uses defaults from Ultralytics: HSV, flips,
translate, scale plus project-specific crowd/small-object augmentations.

Reads a YOLO-format dataset (data.yaml + train/images, train/labels) and creates
a new dataset with Nx training samples (N set by --multiplier). For each image you
get N samples: 1 original plus (N-1) randomly chosen augmentations from:
  - Horizontal flip (fliplr; image + labels)
  - Vertical flip (flipud; image + labels)
  - HSV color jitter (hsv_h/s/v; image only)
  - Random translate (±translate=0.1; image + labels)
  - Random scale (scale=0.5; letterboxed; image + labels)
  - Random perspective warp (street-cam tilt simulation)
  - Large-scale jitter (zoom-out/zoom-in with random crop/pad)
  - Top-half blur/noise simulation
  - BBox copy-paste into upper image region (small distant objects)
Augmentation types are drawn at random (with replacement), so N=3 triples the
dataset with a random mix of all variation types.

Val and test splits are copied as-is. Use the output dataset with fine_tune_yolo.py
or fine_tune_rtdetr.py by setting CONFIG_FILE_PATH to the new data.yaml.

Usage:
    python augment_dataset_3x.py --dataset-dir pdx_cyclist_dataset --output-dir pdx_cyclist_dataset_3x
    python augment_dataset_3x.py --dataset-dir pdx_cyclist_dataset --output-dir pdx_cyclist_dataset_5x -n 5
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

# Match Ultralytics YOLO (fine_tune_yolo.py) default augmentation ranges
HSV_H = 0.015   # Hue (± fraction of 180)
HSV_S = 0.7     # Saturation (± fraction of 255)
HSV_V = 0.4     # Value (± fraction of 255)
TRANSLATE = 0.1   # Translation fraction (± of image size), YOLO default
SCALE_GAIN = 0.5  # Scale gain (YOLO default scale=0.5 -> scale in ~[0.5, 1.5])
PERSPECTIVE_FRAC = 0.08  # Corner perturbation as fraction of width/height
LSJ_MIN_SCALE = 0.35
LSJ_MAX_SCALE = 1.9
TOP_BLUR_RATIO = 0.5
TOP_BLUR_KERNEL = 7
COPY_PASTE_MAX_INSTANCES = 3
COPY_PASTE_SCALE_MIN = 0.35
COPY_PASTE_SCALE_MAX = 0.8
MIN_BOX_W = 0.005
MIN_BOX_H = 0.005
MIN_BOX_AREA = 1e-4

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
# Augmentation type indices:
# 0=hflip, 1=vflip, 2=hsv, 3=translate, 4=scale, 5=perspective, 6=lsj, 7=top-blur, 8=copy-paste
NUM_AUG_TYPES = 9


def list_images(images_dir: Path):
    """Return sorted list of image paths in directory."""
    if not images_dir.exists():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_yolo_labels(label_path: Path):
    """Read YOLO format labels: one line per object = class_id x_center y_center width height (normalized)."""
    if not label_path.exists():
        return []
    lines = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                lines.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return lines


def write_yolo_labels(label_path: Path, rows):
    """Write YOLO format labels."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for row in rows:
            f.write(f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")


def clip_and_filter_rows(rows, nc=None, min_w=MIN_BOX_W, min_h=MIN_BOX_H, min_area=MIN_BOX_AREA):
    """Clip normalized boxes to [0,1] and remove tiny/invalid boxes (optionally invalid classes)."""
    out = []
    for cls, xc, yc, ww, hh in rows:
        cls_int = int(cls)
        if nc is not None and not (0 <= cls_int < int(nc)):
            continue
        x1 = max(0.0, min(1.0, float(xc) - float(ww) / 2.0))
        y1 = max(0.0, min(1.0, float(yc) - float(hh) / 2.0))
        x2 = max(0.0, min(1.0, float(xc) + float(ww) / 2.0))
        y2 = max(0.0, min(1.0, float(yc) + float(hh) / 2.0))
        if x2 <= x1 or y2 <= y1:
            continue
        new_w = x2 - x1
        new_h = y2 - y1
        if new_w < min_w or new_h < min_h or (new_w * new_h) < min_area:
            continue
        out.append([cls_int, (x1 + x2) / 2.0, (y1 + y2) / 2.0, new_w, new_h])
    return out


def _xyxy_norm_to_px(row, w, h):
    """Convert one normalized YOLO row to pixel xyxy."""
    _, xc, yc, ww, hh = row
    x1 = (xc - ww / 2.0) * w
    y1 = (yc - hh / 2.0) * h
    x2 = (xc + ww / 2.0) * w
    y2 = (yc + hh / 2.0) * h
    return x1, y1, x2, y2


def labels_hflip(rows):
    """Transform YOLO labels for horizontal flip: x_center -> 1 - x_center."""
    out = []
    for cls, xc, yc, w, h in rows:
        out.append([cls, 1.0 - xc, yc, w, h])
    return out


def labels_vflip(rows):
    """Transform YOLO labels for vertical flip: y_center -> 1 - y_center."""
    out = []
    for cls, xc, yc, w, h in rows:
        out.append([cls, xc, 1.0 - yc, w, h])
    return out


def image_hflip(img):
    """Horizontal flip image."""
    return cv2.flip(img, 1)


def image_vflip(img):
    """Vertical flip image."""
    return cv2.flip(img, 0)


def image_hsv_jitter(img, h=HSV_H, s=HSV_S, v=HSV_V, rng=None):
    """Apply random HSV jitter; labels unchanged. h,s,v are max absolute fractions."""
    if rng is None:
        rng = random
    img = img.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    H = np.clip(H + rng.uniform(-h * 180, h * 180), 0, 180).astype(np.uint8)
    S = np.clip(S + rng.uniform(-s * 255, s * 255), 0, 255).astype(np.uint8)
    V = np.clip(V + rng.uniform(-v * 255, v * 255), 0, 255).astype(np.uint8)
    hsv = np.stack([H, S, V], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def image_translate_labels(img, rows, tx_frac, ty_frac, rng=None):
    """
    Translate image by (tx_frac, ty_frac) in normalized coords; pad/crop and update labels.
    tx_frac, ty_frac in [-1, 1] (fraction of width/height). Returns (img_out, rows_out).
    Boxes that end up fully outside [0,1] are dropped.
    """
    if rng is None:
        rng = random
    h, w = img.shape[:2]
    tx_px = int(round(tx_frac * w))
    ty_px = int(round(ty_frac * h))
    M = np.float32([[1, 0, tx_px], [0, 1, ty_px]])
    img_out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # warpAffine with (+tx, +ty) moves image content right/down; so bbox center (xc,yc) moves to (xc+tx_frac, yc+ty_frac)
    out_rows = []
    for cls, xc, yc, ww, hh in rows:
        xc_new = xc + tx_frac
        yc_new = yc + ty_frac
        # Clip box to [0,1]; if box is fully outside, drop it
        x1 = xc_new - ww / 2
        y1 = yc_new - hh / 2
        x2 = xc_new + ww / 2
        y2 = yc_new + hh / 2
        x1 = max(0, min(1, x1))
        y1 = max(0, min(1, y1))
        x2 = max(0, min(1, x2))
        y2 = max(0, min(1, y2))
        if x1 >= x2 or y1 >= y2:
            continue
        xc_new = (x1 + x2) / 2
        yc_new = (y1 + y2) / 2
        ww_new = x2 - x1
        hh_new = y2 - y1
        out_rows.append([cls, xc_new, yc_new, ww_new, hh_new])
    return img_out, out_rows


def image_scale_letterbox_labels(img, rows, scale_gain, rng=None):
    """
    Scale image by factor 1 + scale_gain (random ±), then letterbox to original size.
    Returns (img_out, rows_out). scale_gain is the max absolute e.g. 0.5 -> scale in [0.5, 1.5].
    """
    if rng is None:
        rng = random
    h, w = img.shape[:2]
    scale = 1.0 + rng.uniform(-scale_gain, scale_gain)
    scale = max(0.5, min(1.5, scale))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img_scaled = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # Letterbox: fit into (w, h), add padding
    r = min(w / nw, h / nh)
    rw, rh = int(round(nw * r)), int(round(nh * r))
    img_scaled = cv2.resize(img_scaled, (rw, rh), interpolation=cv2.INTER_LINEAR)
    top = (h - rh) // 2
    left = (w - rw) // 2
    img_out = np.full((h, w, img.shape[2]), 114, dtype=img.dtype)
    img_out[top : top + rh, left : left + rw] = img_scaled
    # Transform labels: normalized (xc,yc,w,h) in original -> scale to (rw,rh) then offset by (left,top)
    # Original norm is relative to (w,h). After letterbox, content is in (left, top, rw, rh).
    # So norm x_px = xc*w, and in letterbox: x_in_canvas = left + xc*w * (rw/w) = left + xc*rw. Normalized: (left + xc*rw)/w = left/w + xc*rw/w.
    # Simpler: new normalized xc = (left + xc * rw) / w, new w = (ww * rw) / w.
    left_n = left / w
    top_n = top / h
    rw_n = rw / w
    rh_n = rh / h
    out_rows = []
    for cls, xc, yc, ww, hh in rows:
        xc_new = left_n + xc * rw_n
        yc_new = top_n + yc * rh_n
        ww_new = ww * rw_n
        hh_new = hh * rh_n
        # Clip to [0,1]
        x1 = max(0, xc_new - ww_new / 2)
        y1 = max(0, yc_new - hh_new / 2)
        x2 = min(1, xc_new + ww_new / 2)
        y2 = min(1, yc_new + hh_new / 2)
        if x1 >= x2 or y1 >= y2:
            continue
        xc_new = (x1 + x2) / 2
        yc_new = (y1 + y2) / 2
        ww_new = x2 - x1
        hh_new = y2 - y1
        out_rows.append([cls, xc_new, yc_new, ww_new, hh_new])
    return img_out, out_rows


def image_perspective_labels(img, rows, perspective_frac=PERSPECTIVE_FRAC, rng=None):
    """Apply random perspective warp and remap YOLO boxes by transforming corners."""
    if rng is None:
        rng = random
    h, w = img.shape[:2]
    dx = perspective_frac * w
    dy = perspective_frac * h

    src = np.float32(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    )
    dst = np.float32(
        [
            [rng.uniform(0, dx), rng.uniform(0, dy)],
            [w - 1 - rng.uniform(0, dx), rng.uniform(0, dy)],
            [w - 1 - rng.uniform(0, dx), h - 1 - rng.uniform(0, dy)],
            [rng.uniform(0, dx), h - 1 - rng.uniform(0, dy)],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    out_rows = []
    for row in rows:
        cls = row[0]
        x1, y1, x2, y2 = _xyxy_norm_to_px(row, w, h)
        corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
        xs = np.clip(warped_corners[:, 0], 0, w - 1)
        ys = np.clip(warped_corners[:, 1], 0, h - 1)
        nx1, ny1 = float(np.min(xs)), float(np.min(ys))
        nx2, ny2 = float(np.max(xs)), float(np.max(ys))
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        xc = ((nx1 + nx2) / 2.0) / w
        yc = ((ny1 + ny2) / 2.0) / h
        ww = (nx2 - nx1) / w
        hh = (ny2 - ny1) / h
        out_rows.append([cls, xc, yc, ww, hh])
    return warped, out_rows


def image_lsj_labels(img, rows, min_scale=LSJ_MIN_SCALE, max_scale=LSJ_MAX_SCALE, rng=None):
    """Large scale jitter: random zoom-out/zoom-in with random crop/pad back to original size."""
    if rng is None:
        rng = random
    h, w = img.shape[:2]
    scale = rng.uniform(min_scale, max_scale)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, img.shape[2]), 114, dtype=img.dtype)

    if nw >= w:
        src_x = rng.randint(0, nw - w)
        dst_x = 0
        copy_w = w
    else:
        src_x = 0
        dst_x = rng.randint(0, w - nw)
        copy_w = nw

    if nh >= h:
        src_y = rng.randint(0, nh - h)
        dst_y = 0
        copy_h = h
    else:
        src_y = 0
        dst_y = rng.randint(0, h - nh)
        copy_h = nh

    canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = resized[src_y : src_y + copy_h, src_x : src_x + copy_w]

    out_rows = []
    shift_x = dst_x - src_x
    shift_y = dst_y - src_y
    for row in rows:
        cls = row[0]
        x1, y1, x2, y2 = _xyxy_norm_to_px(row, w, h)
        nx1 = x1 * scale + shift_x
        ny1 = y1 * scale + shift_y
        nx2 = x2 * scale + shift_x
        ny2 = y2 * scale + shift_y
        nx1 = max(0.0, min(w - 1.0, nx1))
        ny1 = max(0.0, min(h - 1.0, ny1))
        nx2 = max(0.0, min(w - 1.0, nx2))
        ny2 = max(0.0, min(h - 1.0, ny2))
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        xc = ((nx1 + nx2) / 2.0) / w
        yc = ((ny1 + ny2) / 2.0) / h
        ww = (nx2 - nx1) / w
        hh = (ny2 - ny1) / h
        out_rows.append([cls, xc, yc, ww, hh])
    return canvas, out_rows


def image_top_half_blur(img, top_ratio=TOP_BLUR_RATIO, kernel_size=TOP_BLUR_KERNEL):
    """Blur top region to simulate distant low-quality objects from tilted surveillance perspective."""
    h, _ = img.shape[:2]
    top_h = int(max(1, min(h, round(h * top_ratio))))
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    out = img.copy()
    top = out[:top_h, :]
    blurred = cv2.GaussianBlur(top, (k, k), 0)
    noise = np.random.normal(0, 3.0, size=blurred.shape).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    out[:top_h, :] = noisy
    return out


def image_copy_paste_upper_half(img, rows, rng=None):
    """Copy object patches and paste scaled instances into upper-half region."""
    if rng is None:
        rng = random
    h, w = img.shape[:2]
    out = img.copy()
    out_rows = [list(r) for r in rows]
    candidates = []

    for row in rows:
        cls = int(row[0])
        x1, y1, x2, y2 = _xyxy_norm_to_px(row, w, h)
        x1i, y1i = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
        x2i, y2i = int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))
        bw = x2i - x1i
        bh = y2i - y1i
        if bw < 6 or bh < 6:
            continue
        patch = img[y1i:y2i, x1i:x2i]
        if patch.size == 0:
            continue
        candidates.append((cls, patch))

    if not candidates:
        return out, out_rows

    n_paste = rng.randint(1, min(COPY_PASTE_MAX_INSTANCES, len(candidates)))
    for _ in range(n_paste):
        cls, patch = rng.choice(candidates)
        ph, pw = patch.shape[:2]
        scale = rng.uniform(COPY_PASTE_SCALE_MIN, COPY_PASTE_SCALE_MAX)
        nw = max(4, int(round(pw * scale)))
        nh = max(4, int(round(ph * scale)))
        if nw >= w or nh >= h:
            continue
        patch_rs = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0 = rng.randint(0, max(0, w - nw))
        y0 = rng.randint(0, max(0, int(h * 0.5) - nh))
        out[y0 : y0 + nh, x0 : x0 + nw] = patch_rs
        xc = (x0 + nw / 2.0) / w
        yc = (y0 + nh / 2.0) / h
        ww = nw / w
        hh = nh / h
        out_rows.append([cls, xc, yc, ww, hh])
    return out, out_rows


def _write_variant(
    out_train_images,
    out_train_labels,
    stem,
    ext,
    img,
    labels,
    variant_type,
    rng,
    filename_suffix,
    nc=None,
    min_box_w=MIN_BOX_W,
    min_box_h=MIN_BOX_H,
    min_box_area=MIN_BOX_AREA,
):
    """
    Write one augmented sample. variant_type in 0..8.
    filename_suffix is used for the output name (e.g. 'aug0') so names are unique when types repeat.
    """
    out_img = out_train_images / f"{stem}_{filename_suffix}{ext}"
    out_lbl = out_train_labels / f"{stem}_{filename_suffix}.txt"
    if variant_type == 0:
        cv2.imwrite(str(out_img), image_hflip(img))
        write_yolo_labels(out_lbl, clip_and_filter_rows(labels_hflip(labels), nc=nc, min_w=min_box_w, min_h=min_box_h, min_area=min_box_area))
    elif variant_type == 1:
        cv2.imwrite(str(out_img), image_vflip(img))
        write_yolo_labels(out_lbl, clip_and_filter_rows(labels_vflip(labels), nc=nc, min_w=min_box_w, min_h=min_box_h, min_area=min_box_area))
    elif variant_type == 2:
        cv2.imwrite(str(out_img), image_hsv_jitter(img, rng=rng))
        write_yolo_labels(out_lbl, clip_and_filter_rows(labels, nc=nc, min_w=min_box_w, min_h=min_box_h, min_area=min_box_area))
    elif variant_type == 3:
        tx_frac = rng.uniform(-TRANSLATE, TRANSLATE)
        ty_frac = rng.uniform(-TRANSLATE, TRANSLATE)
        img_t, labels_t = image_translate_labels(img, labels, tx_frac, ty_frac, rng=rng)
        cv2.imwrite(str(out_img), img_t)
        write_yolo_labels(out_lbl, clip_and_filter_rows(labels_t, nc=nc, min_w=min_box_w, min_h=min_box_h, min_area=min_box_area))
    elif variant_type == 4:
        img_s, labels_s = image_scale_letterbox_labels(img, labels, SCALE_GAIN, rng=rng)
        cv2.imwrite(str(out_img), img_s)
        write_yolo_labels(out_lbl, clip_and_filter_rows(labels_s, nc=nc, min_w=min_box_w, min_h=min_box_h, min_area=min_box_area))
    elif variant_type == 5:
        img_p, labels_p = image_perspective_labels(img, labels, perspective_frac=PERSPECTIVE_FRAC, rng=rng)
        cv2.imwrite(str(out_img), img_p)
        write_yolo_labels(out_lbl, clip_and_filter_rows(labels_p, nc=nc, min_w=min_box_w, min_h=min_box_h, min_area=min_box_area))
    elif variant_type == 6:
        img_lsj, labels_lsj = image_lsj_labels(img, labels, min_scale=LSJ_MIN_SCALE, max_scale=LSJ_MAX_SCALE, rng=rng)
        cv2.imwrite(str(out_img), img_lsj)
        write_yolo_labels(out_lbl, clip_and_filter_rows(labels_lsj, nc=nc, min_w=min_box_w, min_h=min_box_h, min_area=min_box_area))
    elif variant_type == 7:
        cv2.imwrite(str(out_img), image_top_half_blur(img, top_ratio=TOP_BLUR_RATIO, kernel_size=TOP_BLUR_KERNEL))
        write_yolo_labels(out_lbl, clip_and_filter_rows(labels, nc=nc, min_w=min_box_w, min_h=min_box_h, min_area=min_box_area))
    else:  # 8 = copy-paste
        img_cp, labels_cp = image_copy_paste_upper_half(img, labels, rng=rng)
        cv2.imwrite(str(out_img), img_cp)
        write_yolo_labels(out_lbl, clip_and_filter_rows(labels_cp, nc=nc, min_w=min_box_w, min_h=min_box_h, min_area=min_box_area))


def augment_train_split(
    dataset_dir: Path,
    output_dir: Path,
    data_cfg: dict,
    seed: int = 42,
    multiplier: int = 3,
    warn_missing_labels: bool = False,
    min_box_w: float = MIN_BOX_W,
    min_box_h: float = MIN_BOX_H,
    min_box_area: float = MIN_BOX_AREA,
):
    """Create Nx training samples: 1 original + (N-1) random augmentations (drawn from all configured types)."""
    if multiplier < 1:
        raise ValueError(f"multiplier must be >= 1, got {multiplier}")
    rng = random.Random(seed)
    train_images_rel = data_cfg.get("train")
    if not train_images_rel:
        raise ValueError("data.yaml has no 'train' key")
    train_images_dir = dataset_dir / train_images_rel
    train_labels_dir = dataset_dir / train_images_rel.replace("/images", "/labels")

    out_train_images = output_dir / train_images_rel
    out_train_labels = output_dir / train_images_rel.replace("/images", "/labels")
    out_train_images.mkdir(parents=True, exist_ok=True)
    out_train_labels.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(train_images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images in {train_images_dir}")

    nc = data_cfg.get("nc")
    missing_label_count = 0
    total_train = 0
    for img_path in image_paths:
        stem = img_path.stem
        ext = img_path.suffix
        label_path = train_labels_dir / f"{stem}.txt"
        if warn_missing_labels and not label_path.exists():
            missing_label_count += 1
        labels = read_yolo_labels(label_path)
        labels = clip_and_filter_rows(labels, nc=nc, min_w=min_box_w, min_h=min_box_h, min_area=min_box_area)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 1) Original
        out_img = out_train_images / f"{stem}{ext}"
        out_lbl = out_train_labels / f"{stem}.txt"
        shutil.copy2(img_path, out_img)
        write_yolo_labels(out_lbl, labels)
        total_train += 1

        # 2) (N-1) random augmentations from all configured types (with replacement)
        for i in range(multiplier - 1):
            variant_type = rng.randint(0, NUM_AUG_TYPES - 1)
            _write_variant(
                out_train_images, out_train_labels,
                stem, ext, img, labels,
                variant_type, rng,
                filename_suffix=f"aug{i}",
                nc=nc,
                min_box_w=min_box_w,
                min_box_h=min_box_h,
                min_box_area=min_box_area,
            )
            total_train += 1

    if warn_missing_labels and missing_label_count > 0:
        print(f"Warning: {missing_label_count} train images were missing label files. Empty labels were written for those images.")
    print(f"Train: {len(image_paths)} originals -> {total_train} samples ({multiplier}x, random mix of all augmentation types)")
    return total_train


def copy_split_as_is(dataset_dir: Path, output_dir: Path, split_key: str, images_rel: str):
    """Copy a split (e.g. val, test) without augmentation."""
    if not images_rel:
        return 0
    src_images = dataset_dir / images_rel
    src_labels = dataset_dir / images_rel.replace("/images", "/labels")
    dst_images = output_dir / images_rel
    dst_labels = output_dir / images_rel.replace("/images", "/labels")
    if not src_images.exists():
        return 0
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(src_images)
    for img_path in image_paths:
        shutil.copy2(img_path, dst_images / img_path.name)
        label_path = src_labels / img_path.with_suffix(".txt").name
        if label_path.exists():
            shutil.copy2(label_path, dst_labels / label_path.name)
    print(f"{split_key}: {len(image_paths)} images copied")
    return len(image_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Expand training set with baseline + crowd/small-object augmentations (flip, hsv, translate, scale, perspective, LSJ, top-blur, copy-paste)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="v4_pdx_cyclist_dataset",
        help="Path to YOLO dataset (must contain data.yaml and train/images, train/labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="v4_pdx_cyclist_dataset_3x",
        help="Output dataset path (train will have Nx images, val/test copied as-is).",
    )
    parser.add_argument(
        "--multiplier",
        "-n",
        type=int,
        default=3,
        metavar="N",
        help="Multiply training set by N: 1 original + (N-1) random augmentations from all types. Default: %(default)s.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for augmentations.",
    )
    parser.add_argument(
        "--warn-missing-labels",
        action="store_true",
        help="Warn when a train image has no label file.",
    )
    parser.add_argument(
        "--min-box-w",
        type=float,
        default=MIN_BOX_W,
        help="Drop labels with normalized width below this threshold.",
    )
    parser.add_argument(
        "--min-box-h",
        type=float,
        default=MIN_BOX_H,
        help="Drop labels with normalized height below this threshold.",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=MIN_BOX_AREA,
        help="Drop labels with normalized area below this threshold.",
    )
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    if args.multiplier < 1:
        parser.error("--multiplier must be >= 1")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    n_train = augment_train_split(
        dataset_dir,
        output_dir,
        data_cfg,
        seed=args.seed,
        multiplier=args.multiplier,
        warn_missing_labels=args.warn_missing_labels,
        min_box_w=args.min_box_w,
        min_box_h=args.min_box_h,
        min_box_area=args.min_box_area,
    )
    n_val = copy_split_as_is(dataset_dir, output_dir, "Val", data_cfg.get("val", ""))
    n_test = copy_split_as_is(dataset_dir, output_dir, "Test", data_cfg.get("test", ""))

    out_yaml = {
        "names": data_cfg.get("names", ["cyclist", "pedestrian"]),
        "nc": data_cfg.get("nc", 2),
        "train": data_cfg["train"],
        "val": data_cfg.get("val", "valid/images"),
        "test": data_cfg.get("test", "test/images"),
    }
    out_yaml_path = output_dir / "data.yaml"
    with open(out_yaml_path, "w") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False, default_flow_style=False)

    print(f"\nWrote {out_yaml_path}")
    print(f"Training samples: {n_train} ({args.multiplier}x original train size)")
    print("In fine_tune_yolo.py or fine_tune_rtdetr.py set: CONFIG_FILE_PATH = '%s'" % (output_dir / "data.yaml"))


if __name__ == "__main__":
    main()
