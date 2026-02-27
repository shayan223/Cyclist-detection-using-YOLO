import argparse
import json
import shutil
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_yolo_names(data_cfg):
    names = data_cfg.get("names")
    if isinstance(names, dict):
        max_id = max(int(k) for k in names.keys())
        ordered = []
        for idx in range(max_id + 1):
            ordered.append(str(names.get(idx, names.get(str(idx), f"class_{idx}"))))
        return ordered
    if isinstance(names, list):
        return [str(n) for n in names]
    nc = int(data_cfg.get("nc", 0))
    return [f"class_{i}" for i in range(nc)]


def _resolve_split_dir(yaml_dir: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = (yaml_dir / split_path).resolve()
    return split_path


def _yolo_label_dir_from_images(images_dir: Path) -> Path:
    if images_dir.name.lower() == "images":
        return images_dir.parent / "labels"
    return images_dir.parent / "labels"


def _to_coco_bbox(cx, cy, w, h, img_w, img_h):
    x = (cx - (w / 2.0)) * img_w
    y = (cy - (h / 2.0)) * img_h
    bw = w * img_w
    bh = h * img_h
    x = max(0.0, min(x, img_w - 1))
    y = max(0.0, min(y, img_h - 1))
    bw = max(0.0, min(bw, img_w - x))
    bh = max(0.0, min(bh, img_h - y))
    return [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)]


def _read_yolo_annotations(label_path: Path):
    anns = []
    if not label_path.exists():
        return anns
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        anns.append((cls, cx, cy, w, h))
    return anns


def convert_yolo_to_rfdetr(data_yaml_path: Path, output_dir: Path):
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    yaml_dir = data_yaml_path.parent.resolve()
    names = _load_yolo_names(data_cfg)
    categories = [
        {"id": idx, "name": name, "supercategory": "object"}
        for idx, name in enumerate(names)
    ]

    split_key_candidates = {
        "train": ["train"],
        "valid": ["val", "valid"],
        "test": ["test"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for out_split, candidates in split_key_candidates.items():
        split_value = None
        for key in candidates:
            if key in data_cfg:
                split_value = data_cfg[key]
                break
        if not split_value:
            continue

        images_dir = _resolve_split_dir(yaml_dir, split_value)
        labels_dir = _yolo_label_dir_from_images(images_dir)
        if not images_dir.exists():
            print(f"[skip] {out_split}: images path not found: {images_dir}")
            continue

        split_out = output_dir / out_split
        split_out.mkdir(parents=True, exist_ok=True)

        images = []
        annotations = []
        ann_id = 1
        image_id = 1

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                continue

            dst_img = split_out / img_path.name
            shutil.copy2(img_path, dst_img)

            images.append(
                {
                    "id": image_id,
                    "file_name": img_path.name,
                    "width": img_w,
                    "height": img_h,
                }
            )

            label_path = labels_dir / f"{img_path.stem}.txt"
            yolo_anns = _read_yolo_annotations(label_path)
            for cls, cx, cy, w, h in yolo_anns:
                if cls < 0 or cls >= len(categories):
                    continue
                bbox = _to_coco_bbox(cx, cy, w, h, img_w, img_h)
                if bbox[2] <= 0.0 or bbox[3] <= 0.0:
                    continue
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls,
                        "bbox": bbox,
                        "area": round(bbox[2] * bbox[3], 2),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

            image_id += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        with (split_out / "_annotations.coco.json").open("w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

        print(
            f"[ok] {out_split}: {len(images)} images, {len(annotations)} annotations -> {split_out}"
        )

    print(f"\nRF-DETR dataset copy created at: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO dataset to RF-DETR COCO format (copied dataset)."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to YOLO data.yaml",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for RF-DETR-compatible COCO copy",
    )
    args = parser.parse_args()

    data_arg = Path(args.data).resolve()
    output_dir = Path(args.output).resolve()

    if data_arg.is_dir():
        preferred = data_arg / "data.yaml"
        if preferred.exists():
            data_yaml_path = preferred
        else:
            yaml_candidates = sorted(list(data_arg.glob("*.yaml")) + list(data_arg.glob("*.yml")))
            if len(yaml_candidates) == 1:
                data_yaml_path = yaml_candidates[0]
            elif len(yaml_candidates) > 1:
                names = ", ".join(p.name for p in yaml_candidates)
                raise ValueError(
                    f"Multiple YAML files found in {data_arg}. "
                    f"Please pass --data <path-to-yaml>. Candidates: {names}"
                )
            else:
                raise FileNotFoundError(
                    f"No data.yaml/.yml found in directory: {data_arg}. "
                    "Pass --data <path-to-yolo-data.yaml>."
                )
    else:
        data_yaml_path = data_arg

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml_path}")
    if not data_yaml_path.is_file():
        raise ValueError(f"--data must point to a YAML file or dataset directory, got: {data_yaml_path}")

    print(f"Using YOLO data config: {data_yaml_path}")
    convert_yolo_to_rfdetr(data_yaml_path, output_dir)


if __name__ == "__main__":
    main()
