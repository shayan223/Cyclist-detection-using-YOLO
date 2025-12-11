import os
from ultralytics import RTDETR
import torch

# --- Configuration ---
CONFIG_FILE_PATH = 'Cyclist_Pedestrian_Dataset/data.yaml'#'eurocity_yolo/data.yaml'#'./training_data/dataset.yaml'#'./training_data/config.yaml'
MODEL_PATH = 'rtdetr-l.pt'  # Base RT-DETR model (options: rtdetr-l.pt, rtdetr-x.pt)
EPOCHS = 100
BATCH = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Data Augmentation Configuration ---
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
                     hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
                     degrees=DEGREES, translate=TRANSLATE, scale=SCALE,
                     shear=SHEAR, perspective=PERSPECTIVE,
                     flipud=FLIPUD, fliplr=FLIPLR,
                     mosaic=MOSAIC, mixup=MIXUP, cutmix=CUTMIX, copy_paste=COPY_PASTE,
                     auto_augment=AUTO_AUGMENT, erasing=ERASING):
    """Fine-tunes an RT-DETR model on the cyclist dataset with data augmentation."""

    os.environ['WANDB_DISABLED'] = 'true'  # Disable Weights & Biases logging

    model = RTDETR(model_path)  # Load the pre-trained RT-DETR model
    model.to(device)

    print(f"Training on device: {device}")
    print(f"Data augmentation enabled:")
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
        patience=5,  # Stop training early if no improvement
        save_period=5,  # Save model after each epoch
        project="cyclist_detection_rtdetr",
        name="rtdetr_finetune",
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
    fine_tune_rtdetr(CONFIG_FILE_PATH, MODEL_PATH, EPOCHS, BATCH, DEVICE)

