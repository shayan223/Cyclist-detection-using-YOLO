import os
from ultralytics import YOLO
import torch

# --- Configuration ---
CONFIG_FILE_PATH = 'Cyclist_Pedestrian_Dataset/data.yaml'#'eurocity_yolo/data.yaml'#'./training_data/dataset.yaml'#'./training_data/config.yaml'
MODEL_PATH = './yolo11n.pt'#'./yolo11n.pt' #'yolov8l.pt'  # Base YOLO model
EPOCHS = 100
BATCH = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def fine_tune_yolo(config_file_path, model_path, epochs, batch, device):
    """Fine-tunes a YOLO model on the cyclist dataset."""

    os.environ['WANDB_DISABLED'] = 'true'  # Disable Weights & Biases logging

    model = YOLO(model_path)  # Load the pre-trained YOLO model
    model.to(device)

    print(f"Training on device: {device}")

    results = model.train(
        data=config_file_path,
        epochs=epochs,
        batch=batch,
        device=device,
        patience=5,  # Stop training early if no improvement
        save_period=25,  # Save model after each epoch
        project="cyclist_detection_yolo11n",
        name="yolo_finetune",
        resume=False
    )

    validation_results = model.val(device=device)
    print(validation_results)


if __name__ == "__main__":
    fine_tune_yolo(CONFIG_FILE_PATH, MODEL_PATH, EPOCHS, BATCH, DEVICE)
