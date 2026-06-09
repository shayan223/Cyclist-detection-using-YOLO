import os
import argparse
from ultralytics import YOLO
import torch

# --- Configuration ---
CONFIG_FILE_PATH = 'v5_pdx_cyclist_dataset/data.yaml'
MODEL_PATH = './yolo26l_macro.pt'  # Keep using your YOLO26 pretrained checkpoint as default
EPOCHS = 100
BATCH = 4
# Your current augmented dataset is 640x480; 960 is a better default trade-off
# than 1280 (less VRAM/latency while still giving denser feature maps for small objects).
IMGSZ = 960
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_NAME = "cyclist_detection_yolo26"
BASE_RUN_NAME = "yolo26_finetune"

DEFAULT_RECIPE = {
    # Crowd/small-object focused train-time augmentation knobs.
    "mosaic": 1.0,
    "mixup": 0.20,
    "perspective": 0.0008,
    "close_mosaic": 10,  # Disable heavy mosaic in final epochs for cleaner convergence
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


def build_run_name(base_name, imgsz, recipe):
    """Create compact run names that encode key augmentation settings."""
    return (
        f"{base_name}_imgsz{imgsz}"
        f"_mos{recipe['mosaic']}"
        f"_mix{recipe['mixup']}"
        f"_persp{recipe['perspective']}"
    )


def fine_tune_yolo(
    config_file_path,
    model_path,
    epochs,
    batch,
    device,
    imgsz=IMGSZ,
    recipe=None,
    project_name=PROJECT_NAME,
    run_name=BASE_RUN_NAME,
):
    """Fine-tune a YOLO model with explicit crowd/small-object recipe controls."""
    os.environ['WANDB_DISABLED'] = 'true'  # Disable Weights & Biases logging
    recipe = dict(DEFAULT_RECIPE if recipe is None else recipe)

    model = YOLO(model_path)
    model.to(device)

    effective_run_name = build_run_name(run_name, imgsz, recipe)
    print(f"Training on device: {device}")
    print(f"Run name: {effective_run_name}")
    print(f"Recipe: {recipe}")

    model.train(
        data=config_file_path,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        patience=10,
        save_period=25,
        project=project_name,
        name=effective_run_name,
        resume=False,
        mosaic=recipe["mosaic"],
        mixup=recipe["mixup"],
        perspective=recipe["perspective"],
        close_mosaic=recipe["close_mosaic"],
        degrees=recipe["degrees"],
        translate=recipe["translate"],
        scale=recipe["scale"],
        fliplr=recipe["fliplr"],
        flipud=recipe["flipud"],
    )

    validation_results = model.val(data=config_file_path, device=device, imgsz=imgsz)
    print(validation_results)


def run_ablation_protocol(
    config_file_path,
    model_path,
    epochs,
    batch,
    device,
    imgsz,
    run_experiments=False,
):
    """
    Stage ablations from the plan:
    1) baseline
    2) +train aug only
    3) +top-region pass
    4) +soft-nms
    5) +tiled inference (SAHI-style)
    """
    experiments = [
        ("ablation_1_baseline", BASELINE_RECIPE),
        ("ablation_2_train_aug", DEFAULT_RECIPE),
        ("ablation_3_top_region", DEFAULT_RECIPE),
        ("ablation_4_soft_nms", DEFAULT_RECIPE),
        ("ablation_5_tiled", DEFAULT_RECIPE),
    ]

    print("\nAblation protocol (training + inference eval steps):")
    for name, recipe in experiments:
        run_name = build_run_name(name, imgsz, recipe)
        print(f"- {run_name}")

    print("\nInference command templates for deepSORT_yolo.py:")
    print("  # Ablation 3 (+top-region pass)")
    print("  python deepSORT_yolo.py --model <weights.pt> --top-region-pass --top-region-ratio 0.35 --top-region-imgsz 1536")
    print("  # Ablation 4 (+soft-nms)")
    print("  python deepSORT_yolo.py --model <weights.pt> --top-region-pass --crowd-mode soft-nms --soft-nms-iou 0.5 --soft-nms-sigma 0.5")
    print("  # Ablation 5 (+tiled SAHI-style pass)")
    print("  python deepSORT_yolo.py --model <weights.pt> --top-region-pass --crowd-mode soft-nms --tile-mode sahi --tile-size 960 --tile-overlap 0.2")

    if not run_experiments:
        print("\nDry run only. Re-run with --run-ablation to execute training experiments.")
        return

    for name, recipe in experiments:
        fine_tune_yolo(
            config_file_path=config_file_path,
            model_path=model_path,
            epochs=epochs,
            batch=batch,
            device=device,
            imgsz=imgsz,
            recipe=recipe,
            run_name=name,
        )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 for cyclist/pedestrian detection.")
    parser.add_argument("--data", type=str, default=CONFIG_FILE_PATH, help="Path to YOLO data.yaml.")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to pretrained YOLO model/checkpoint.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=BATCH, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Training image size.")
    parser.add_argument("--device", type=str, default=DEVICE, help="Training device.")
    parser.add_argument("--project", type=str, default=PROJECT_NAME, help="Ultralytics project name.")
    parser.add_argument("--name", type=str, default=BASE_RUN_NAME, help="Base run name prefix.")
    parser.add_argument("--baseline-recipe", action="store_true", help="Use baseline recipe instead of crowd/small-object recipe.")
    parser.add_argument("--ablation-protocol", action="store_true", help="Print and/or execute staged ablation protocol.")
    parser.add_argument("--run-ablation", action="store_true", help="When used with --ablation-protocol, execute ablation trainings.")
    args = parser.parse_args()

    if args.ablation_protocol:
        run_ablation_protocol(
            config_file_path=args.data,
            model_path=args.model,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            imgsz=args.imgsz,
            run_experiments=args.run_ablation,
        )
        return

    recipe = BASELINE_RECIPE if args.baseline_recipe else DEFAULT_RECIPE
    fine_tune_yolo(
        config_file_path=args.data,
        model_path=args.model,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        imgsz=args.imgsz,
        recipe=recipe,
        project_name=args.project,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
