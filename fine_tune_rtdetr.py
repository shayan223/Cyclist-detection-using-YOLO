import os
from ultralytics import RTDETR
import torch

# --- Configuration ---
# To triple training samples: run augment_dataset_3x.py, then use the 3x data.yaml below.
CONFIG_FILE_PATH = 'v4_augmented_10x_pdx_cyclist_dataset/data.yaml'  # or 'pdx_cyclist_dataset_3x/data.yaml' for 3x augmented train set
# CONFIG_FILE_PATH = 'pdx_cyclist_dataset_3x/data.yaml'  # uncomment after running: python augment_dataset_3x.py --dataset-dir pdx_cyclist_dataset --output-dir pdx_cyclist_dataset_3x
#MODEL_PATH = 'rtdetr-l.pt'  # Base RT-DETR model (options: rtdetr-l.pt, rtdetr-x.pt) USE THIS FOR FIRST TIME TRAINING
MODEL_PATH = './cyclist_detection_rtdetr/rtdetr_finetune4/weights/best.pt' # pre-finetuned model on cyclist dataset, for further fine tuning on pdx dataset  
EPOCHS = 50
BATCH = 8
# Initial learning rate (lr0). Use a smaller value (e.g. 1e-4, 5e-5) for gentler fine-tuning.
LR0 = 0.0001  # Ultralytics default; try 0.001 or 0.0001 for smaller updates
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Runtime Augmentation ---
ENABLE_AUGMENTATION = False  # Set to False to disable all data augmentation during training

# --- Data Augmentation Configuration (used only when ENABLE_AUGMENTATION is True) ---
# HSV augmentation (color space)
HSV_H = 0.015  # Hue augmentation factor
HSV_S = 0.7    # Saturation augmentation factor
HSV_V = 0.4    # Value (brightness) augmentation factor

# Geometric transformations
DEGREES = 0.0      # Rotation degrees (+/-)
TRANSLATE = 0.1    # Translation fraction (+/-)
SCALE = 0.5        # Scale gain (+/-)
SHEAR = 0.0        # Shear degrees (+/-)
PERSPECTIVE = 0.0  # Perspective (+/- fraction)

# Flip augmentations
FLIPUD = 0.0   # Vertical flip probability (0.0-1.0)
FLIPLR = 0.5   # Horizontal flip probability (0.0-1.0)

# Advanced augmentations
MOSAIC = 1.0        # Mosaic augmentation probability (0.0-1.0)
MIXUP = 0.15        # Mixup augmentation probability (0.0-1.0)
CUTMIX = 0.0        # CutMix augmentation probability (0.0-1.0)
COPY_PASTE = 0.0    # Copy-paste augmentation probability (0.0-1.0)

# Auto augmentation
AUTO_AUGMENT = 'randaugment'  # Auto augmentation policy (randaugment, autoaugment, or None)
ERASING = 0.4       # Random erasing probability (0.0-1.0)


def fine_tune_rtdetr(config_file_path, model_path, epochs, batch, device,
                     lr0=LR0,
                     augment=None,
                     hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
                     degrees=DEGREES, translate=TRANSLATE, scale=SCALE,
                     shear=SHEAR, perspective=PERSPECTIVE,
                     flipud=FLIPUD, fliplr=FLIPLR,
                     mosaic=MOSAIC, mixup=MIXUP, cutmix=CUTMIX, copy_paste=COPY_PASTE,
                     auto_augment=AUTO_AUGMENT, erasing=ERASING):
    """Fine-tunes an RT-DETR model on the cyclist dataset with optional data augmentation."""
    if augment is None:
        augment = ENABLE_AUGMENTATION

    if not augment:
        hsv_h = hsv_s = hsv_v = 0.0
        degrees = translate = scale = shear = perspective = 0.0
        flipud = fliplr = 0.0
        mosaic = mixup = cutmix = copy_paste = 0.0
        auto_augment = None
        erasing = 0.0

    os.environ['WANDB_DISABLED'] = 'true'  # Disable Weights & Biases logging

    model = RTDETR(model_path)  # Load the pre-trained RT-DETR model
    model.to(device)

    print(f"Training on device: {device}")
    print(f"Initial learning rate (lr0): {lr0}")
    print(f"Runtime augmentation: {'enabled' if augment else 'disabled'}")
    if augment:
        print(f"  - HSV: H={hsv_h}, S={hsv_s}, V={hsv_v}")
        print(f"  - Geometric: degrees={degrees}, translate={translate}, scale={scale}")
        print(f"  - Flips: horizontal={fliplr}, vertical={flipud}")
        print(f"  - Advanced: mosaic={mosaic}, mixup={mixup}, cutmix={cutmix}, copy_paste={copy_paste}")
        print(f"  - Auto augment: {auto_augment}, erasing={erasing}")

    results = model.train(
        data=config_file_path,
        epochs=epochs,
        batch=batch,
        device=device,
        lr0=lr0,
        patience=5,  # Stop training early if no improvement
        save_period=5,  # Save model after each epoch
        project="pdx_rtdetr",
        name="pdx_rtdetr_finetune_augmented",
        resume=False,
        # Data augmentation parameters
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        cutmix=cutmix,
        copy_paste=copy_paste,
        auto_augment=auto_augment,
        erasing=erasing
    )

    validation_results = model.val(device=device)
    print(validation_results)


if __name__ == "__main__":
    fine_tune_rtdetr(
        CONFIG_FILE_PATH, MODEL_PATH, EPOCHS, BATCH, DEVICE,
        lr0=LR0,
        augment=ENABLE_AUGMENTATION,
    )

