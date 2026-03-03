import argparse
import os
import torch
from ultralytics import YOLO

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MODEL = 'cyclist_detection_yolo26/weights/best.pt'
DEFAULT_DATA = 'v5_pdx_cyclist_dataset/data.yaml'
DEFAULT_IMGSZ = 960


def validate(
    model_path,
    data_path,
    imgsz,
    device,
    split='val',
    conf=0.001,
    iou=0.6,
    verbose=True,
):
    """Run validation on a fine-tuned YOLO26 model and print mAP metrics."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    print(f"Loading YOLO26 model: {model_path}")
    model = YOLO(model_path)
    model.to(device)

    print(f"Validating on split='{split}' | imgsz={imgsz} | conf={conf} | iou={iou}")
    results = model.val(
        data=data_path,
        imgsz=imgsz,
        device=device,
        split=split,
        conf=conf,
        iou=iou,
        verbose=verbose,
    )
    print(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate a fine-tuned YOLO26 model.")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Path to YOLO26 .pt checkpoint.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to YOLO data.yaml.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Validation image size.")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device: cuda or cpu.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Dataset split to evaluate.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for evaluation.")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS during validation.")
    parser.add_argument("--no-verbose", action="store_true", help="Suppress per-class verbose output.")
    args = parser.parse_args()

    validate(
        model_path=args.model,
        data_path=args.data,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        verbose=not args.no_verbose,
    )


if __name__ == "__main__":
    main()
