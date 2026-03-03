from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import torch
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from convert_yolo_to_rfdetr import convert_yolo_to_rfdetr

DEFAULT_RECIPE = {
    "mosaic": 1.0,
    "mixup": 0.20,
    "perspective": 0.0008,
    "close_mosaic": 10,
    "degrees": 2.0,
    "translate": 0.1,
    "scale": 0.6,
    "fliplr": 0.5,
    "flipud": 0.0,
}

BASELINE_RECIPE = {
    "mosaic": 0.0,
    "mixup": 0.0,
    "perspective": 0.0,
    "close_mosaic": 0,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "flipud": 0.0,
}

MODEL_MAP = {
    "RFDETRNano": RFDETRNano,
    "RFDETRSmall": RFDETRSmall,
    "RFDETRMedium": RFDETRMedium,
    "RFDETRLarge": RFDETRLarge,
}


def _infer_model_size_from_weights(weights_path: str | None) -> str | None:
    if not weights_path:
        return None
    p = weights_path.replace("\\", "/").lower()
    if "nano" in p:
        return "RFDETRNano"
    if "small" in p:
        return "RFDETRSmall"
    if "medium" in p:
        return "RFDETRMedium"
    # Handle unsupported larger variants explicitly so we do not misclassify as Large.
    if "2xlarge" in p or "xlarge" in p:
        return None
    if "large" in p:
        return "RFDETRLarge"
    return None


def _build_run_name(base_name: str, resolution: int, recipe: dict) -> str:
    return (
        f"{base_name}_res{resolution}"
        f"_mos{recipe['mosaic']}"
        f"_mix{recipe['mixup']}"
        f"_persp{recipe['perspective']}"
    )


def _supported_augmentation_kwargs(model, recipe: dict):
    """
    RF-DETR's Python API evolves quickly. Only pass augmentation keys that are
    explicitly accepted by the installed model.train() signature.
    """
    sig = inspect.signature(model.train)
    accepted = set(sig.parameters.keys())
    supported = {k: v for k, v in recipe.items() if k in accepted}
    unsupported = [k for k in recipe.keys() if k not in accepted]
    return supported, unsupported


def train_rfdetr(
    dataset_dir: Path,
    output_root: Path,
    run_name: str,
    model_size: str,
    pretrain_weights: str | None,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    lr_encoder: float | None,
    resolution: int,
    device: str,
    use_ema: bool,
    checkpoint_interval: int,
    early_stopping: bool,
    early_stopping_patience: int,
    augment_profile: str,
    resume: str | None,
):
    model_cls = MODEL_MAP[model_size]
    model = model_cls(pretrain_weights=pretrain_weights) if pretrain_weights else model_cls()

    recipe = DEFAULT_RECIPE if augment_profile == "yolo-latest" else BASELINE_RECIPE
    effective_name = _build_run_name(run_name, resolution, recipe)
    output_dir = output_root / effective_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "resolution": resolution,
        "device": device,
        "use_ema": use_ema,
        "checkpoint_interval": checkpoint_interval,
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
    }
    if lr_encoder is not None:
        train_kwargs["lr_encoder"] = lr_encoder
    if resume:
        train_kwargs["resume"] = resume

    if augment_profile != "none":
        supported_aug, unsupported_aug = _supported_augmentation_kwargs(model, recipe)
        train_kwargs.update(supported_aug)
        if supported_aug:
            print(f"Applying augmentation overrides supported by this RF-DETR build: {supported_aug}")
        if unsupported_aug:
            print(
                "These YOLO-style augmentation keys are not exposed by the installed RF-DETR train() API "
                f"and will use RF-DETR defaults: {unsupported_aug}"
            )

    print(f"Training on device: {device}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Run name: {effective_name}")
    print(f"Model: {model_size}")
    print(
        f"epochs={epochs}, batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, "
        f"lr={lr}, resolution={resolution}"
    )

    model.train(**train_kwargs)
    print(f"Training finished. Artifacts in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR for cyclist/pedestrian detection.")
    parser.add_argument("--data", type=str, default="v5_pdx_cyclist_dataset/data.yaml", help="YOLO data.yaml path")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="coco_v5_pdx_cyclist_dataset",
        help="RF-DETR COCO dataset directory. If omitted, a converted copy is created from --data.",
    )
    parser.add_argument(
        "--converted-output",
        type=str,
        default="RF_DETR/datasets/v5_pdx_cyclist_dataset_coco_rfdetr",
        help="Where to create the converted COCO copy when --dataset-dir is omitted.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip YOLO->COCO conversion and require --dataset-dir.",
    )
    parser.add_argument(
        "--model-size",
        choices=list(MODEL_MAP.keys()),
        default='RFDETRSmall',
        help="RF-DETR model size. Default is RFDETRLarge unless inferred from --pretrain-weights.",
    )
    parser.add_argument(
        "--pretrain-weights",
        type=str,
        default="",
        help="Optional checkpoint path for continued fine-tuning.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=576) #Should be divisible by 56
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Training device, e.g. "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument("--output-root", type=str, default="RF_DETR/runs")
    parser.add_argument("--name", type=str, default="rfdetr_finetune")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--early-stopping", action="store_true", default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument(
        "--augment-profile",
        choices=["none", "baseline", "yolo-latest"],
        default="yolo-latest",
        help=(
            "none=RF-DETR defaults only, baseline=baseline recipe from fine_tune_yolo.py, "
            "yolo-latest=latest recipe from fine_tune_yolo.py"
        ),
    )
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint.pth to resume training.")
    args = parser.parse_args()

    os.environ["WANDB_DISABLED"] = "true"

    if args.resolution % 56 != 0:
        print("Warning: RF-DETR docs recommend resolution divisible by 56.")

    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else None

    if not args.skip_convert and dataset_dir is None:
        data_yaml = Path(args.data).resolve()
        if not data_yaml.exists():
            raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}")
        convert_out = Path(args.converted_output).resolve()
        convert_yolo_to_rfdetr(data_yaml, convert_out)
        dataset_dir = convert_out

    if dataset_dir is None:
        raise ValueError("Provide --dataset-dir or omit --skip-convert to auto-convert from --data.")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    inferred_model_size = _infer_model_size_from_weights(args.pretrain_weights or None)
    model_size = args.model_size or inferred_model_size or "RFDETRLarge"
    if args.model_size is None and inferred_model_size is not None:
        print(f"Using model size inferred from weights: {inferred_model_size}")
    elif args.model_size is None and args.pretrain_weights and inferred_model_size is None:
        print(
            "Could not infer model size from --pretrain-weights path; defaulting to RFDETRLarge. "
            "Set --model-size explicitly if needed."
        )

    train_rfdetr(
        dataset_dir=dataset_dir,
        output_root=Path(args.output_root).resolve(),
        run_name=args.name,
        model_size=model_size,
        pretrain_weights=args.pretrain_weights or None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        resolution=args.resolution,
        device=args.device,
        use_ema=args.use_ema,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        augment_profile=args.augment_profile,
        resume=args.resume or None,
    )


if __name__ == "__main__":
    main()
