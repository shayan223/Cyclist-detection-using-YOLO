import os
import argparse
from ultralytics import RTDETR
import torch

# --- Configuration ---
CONFIG_FILE_PATH = '../v5_pdx_cyclist_dataset/data.yaml'
MODEL_PATH = 'rt_detr_macro_augmented.pt'  # Base RT-DETR model. Options: rtdetr-l.pt, rtdetr-x.pt
EPOCHS = 100
BATCH = 8
LR0 = 0.001
IMGSZ = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_NAME = "cyclist_detection_rtdetr"
BASE_RUN_NAME = "rtdetr_finetune"

# --- Augmentation recipes ---
DEFAULT_RECIPE = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.15,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "auto_augment": "randaugment",
    "erasing": 0.4,
}

BASELINE_RECIPE = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "auto_augment": None,
    "erasing": 0.0,
}


def build_run_name(base_name, imgsz, recipe):
    """Create compact run names that encode key augmentation settings."""
    return (
        f"{base_name}_imgsz{imgsz}"
        f"_mos{recipe['mosaic']}"
        f"_mix{recipe['mixup']}"
        f"_fliplr{recipe['fliplr']}"
    )


def fine_tune_rtdetr(
    config_file_path,
    model_path,
    epochs,
    batch,
    device,
    lr0=LR0,
    imgsz=IMGSZ,
    recipe=None,
    project_name=PROJECT_NAME,
    run_name=BASE_RUN_NAME,
):
    """Fine-tune an RT-DETR model on the cyclist/pedestrian dataset."""
    os.environ['WANDB_DISABLED'] = 'true'
    recipe = dict(DEFAULT_RECIPE if recipe is None else recipe)

    model = RTDETR(model_path)
    model.to(device)

    effective_run_name = build_run_name(run_name, imgsz, recipe)
    print(f"Training on device: {device}")
    print(f"Run name: {effective_run_name}")
    print(f"Initial learning rate: {lr0}")
    print(f"Recipe: {recipe}")

    model.train(
        data=config_file_path,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        lr0=lr0,
        patience=10,
        save_period=10,
        project=project_name,
        name=effective_run_name,
        resume=False,
        hsv_h=recipe["hsv_h"],
        hsv_s=recipe["hsv_s"],
        hsv_v=recipe["hsv_v"],
        degrees=recipe["degrees"],
        translate=recipe["translate"],
        scale=recipe["scale"],
        shear=recipe["shear"],
        perspective=recipe["perspective"],
        flipud=recipe["flipud"],
        fliplr=recipe["fliplr"],
        mosaic=recipe["mosaic"],
        mixup=recipe["mixup"],
        cutmix=recipe["cutmix"],
        copy_paste=recipe["copy_paste"],
        auto_augment=recipe["auto_augment"],
        erasing=recipe["erasing"],
    )

    validation_results = model.val(data=config_file_path, device=device, imgsz=imgsz)
    print(validation_results)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR for cyclist/pedestrian detection.")
    parser.add_argument("--data", type=str, default=CONFIG_FILE_PATH, help="Path to YOLO data.yaml.")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to base or fine-tuned RT-DETR .pt checkpoint.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=BATCH, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Training image size.")
    parser.add_argument("--lr0", type=float, default=LR0, help="Initial learning rate.")
    parser.add_argument("--device", type=str, default=DEVICE, help="Training device (cuda/cpu).")
    parser.add_argument("--project", type=str, default=PROJECT_NAME, help="Ultralytics project directory name.")
    parser.add_argument("--name", type=str, default=BASE_RUN_NAME, help="Base run name prefix.")
    parser.add_argument("--baseline-recipe", action="store_true", help="Disable all augmentation (baseline recipe).")
    args = parser.parse_args()

    recipe = BASELINE_RECIPE if args.baseline_recipe else DEFAULT_RECIPE
    fine_tune_rtdetr(
        config_file_path=args.data,
        model_path=args.model,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        imgsz=args.imgsz,
        recipe=recipe,
        project_name=args.project,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
