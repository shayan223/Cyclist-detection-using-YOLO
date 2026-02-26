import argparse
import os

from ultralytics import YOLO


def train_single_model(
    base_model: str,
    data_yaml: str,
    classes: list[int],
    project: str,
    name: str,
    device: str = "cuda",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    lr0: float = 1e-4,
) -> None:
    """
    Train a single-class YOLO model on a subset of dataset classes.

    Args:
        base_model: Base YOLO checkpoint (e.g. 'yolov8n.pt' or a finetuned .pt).
        data_yaml: Path to Ultralytics data YAML file.
        classes: List of class indices to include in training (e.g. [0] or [1]).
        project: Parent directory for Ultralytics runs.
        name: Run name under project (e.g. 'cyclist_single_class').
        device: Device string for Ultralytics (e.g. 'cuda', 'cpu', '0', '0,1').
        epochs: Number of training epochs.
        imgsz: Training image size.
        batch: Optional batch size override.
        lr0: Initial learning rate.
    """
    model = YOLO(base_model)

    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
        "project": project,
        "name": name,
        "classes": classes,
        "lr0": lr0,
    }
    if batch is not None:
        train_kwargs["batch"] = batch

    print(
        f"\n[TRAIN] Starting run '{name}'\n"
        f"  base_model: {base_model}\n"
        f"  data: {data_yaml}\n"
        f"  classes: {classes}\n"
        f"  device: {device}\n"
        f"  epochs: {epochs}\n"
        f"  imgsz: {imgsz}\n"
        f"  batch: {batch}\n"
        f"  lr0: {lr0}\n"
        f"  project: {project}\n"
    )

    model.train(**train_kwargs)

    print(f"[TRAIN] Finished run '{name}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train two single-class YOLO models sequentially on a shared dataset. "
            "One model is trained only on cyclists, the other only on pedestrians."
        )
    )

    # Dataset config
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("../v4_pdx_cyclist_dataset_3x", "data.yaml"),
        help=(
            "Path to Ultralytics data YAML for the v4_pdx_cyclist_dataset "
            "(default: v4_pdx_cyclist_dataset/data.yaml). Override for a different dataset."
        ),
    )

    # Base checkpoint
    parser.add_argument(
        "--base-model",
        type=str,
        default="../yolo_finetune4/weights/best.pt", #"yolov8n.pt",
        help="Base YOLO checkpoint to finetune (default: yolov8n.pt).",
    )

    # Output layout
    parser.add_argument(
        "--project",
        type=str,
        default=os.path.join("ensembled_models", "runs"),
        help="Ultralytics project directory where both runs will be saved.",
    )
    parser.add_argument(
        "--cyclist-name",
        type=str,
        default="cyclist_single_class",
        help="Run name for the cyclist-only model.",
    )
    parser.add_argument(
        "--pedestrian-name",
        type=str,
        default="pedestrian_single_class",
        help="Run name for the pedestrian-only model.",
    )

    # Class indices mapping (dataset-specific but overrideable)
    parser.add_argument(
        "--cyclist-class",
        type=int,
        default=0,
        help="Dataset class index for cyclists (default: 0).",
    )
    parser.add_argument(
        "--pedestrian-class",
        type=int,
        default=1,
        help="Dataset class index for pedestrians (default: 1).",
    )

    # Training hyperparameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string passed to Ultralytics (e.g. 'cuda', 'cpu', '0', '0,1').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs for each model.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Optional batch size override. If None, Ultralytics auto-selects.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=1e-4,
        help="Initial learning rate for training (default: 1e-4).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "cyclist", "pedestrian"],
        default="both",
        help=(
            "Which model(s) to train: 'cyclist' only, 'pedestrian' only, "
            "or 'both' sequentially (default: both)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.project, exist_ok=True)

    # Prepare arguments for each model
    cyclist_kwargs = dict(
        base_model=args.base_model,
        data_yaml=args.data,
        classes=[args.cyclist_class],
        project=args.project,
        name=args.cyclist_name,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
    )
    pedestrian_kwargs = dict(
        base_model=args.base_model,
        data_yaml=args.data,
        classes=[args.pedestrian_class],
        project=args.project,
        name=args.pedestrian_name,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
    )

    if args.mode in ("both", "cyclist"):
        print("\nRunning cyclist-only training.")
        train_single_model(**cyclist_kwargs)
    else:
        print("\nSkipping cyclist-only training (mode != 'both' or 'cyclist').")

    if args.mode in ("both", "pedestrian"):
        print("\nRunning pedestrian-only training.")
        train_single_model(**pedestrian_kwargs)
    else:
        print("\nSkipping pedestrian-only training (mode != 'both' or 'pedestrian').")

    print("\nRequested training runs completed.")


if __name__ == "__main__":
    main()

