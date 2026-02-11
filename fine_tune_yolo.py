import os
from ultralytics import YOLO
import torch

# --- Configuration ---
CONFIG_FILE_PATH = './v4_pdx_cyclist_dataset/data.yaml'#'Cyclist_Pedestrian_Dataset/data.yaml'#'eurocity_yolo/data.yaml'#'./training_data/dataset.yaml'#'./training_data/config.yaml'
MODEL_PATH = './euro_pretrain_yolo26/best.pt'#'./yolo11n.pt'#'./yolo11n.pt' #'yolov8l.pt'  # Base YOLO model
EPOCHS = 50
BATCH = 8
LR0 = 0.001  # Starting learning rate
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def fine_tune_yolo(config_file_path, model_path, epochs, batch, lr0, device):
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
        lr0=lr0,  # Starting learning rate
        patience=10,  # Stop training early if no improvement
        save_period=10,  # Save model after each epoch
        project="cyclist_detection_yolo26l",
        name="yolo_finetune_pdx",
        resume=False
    )

    validation_results = model.val(device=device)
    print(validation_results)

if __name__ == "__main__":
    fine_tune_yolo(CONFIG_FILE_PATH, MODEL_PATH, EPOCHS, BATCH, LR0, DEVICE)
