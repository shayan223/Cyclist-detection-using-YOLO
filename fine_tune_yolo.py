import os
from pathlib import Path
import torch
from ultralytics import YOLO
from deepSORT_yolo import process_video

# --- Configuration ---
CONFIG_FILE_PATH = "Cyclist_Pedestrian_Dataset/data.yaml"
BASE_MODEL_PATH  = "./yolo26n.pt"      # starting weights
EPOCHS = 100
BATCH  = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE_PATH = './v4_pdx_cyclist_dataset/data.yaml'#'Cyclist_Pedestrian_Dataset/data.yaml'#'eurocity_yolo/data.yaml'#'./training_data/dataset.yaml'#'./training_data/config.yaml'
MODEL_PATH = './euro_pretrain_yolo26/best.pt'#'./yolo11n.pt'#'./yolo11n.pt' #'yolov8l.pt'  # Base YOLO model
EPOCHS = 50
BATCH = 8
LR0 = 0.001  # Starting learning rate
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

GENERATE_VIDEO = True
CONFIDENCE     = 0.85
IOU_NMS        = 0.7
MAX_AGE        = 15
MAX_IOU_DIST   = 0.7
NO_DISPLAY     = True

PROJECT_DIR    = "cyclist_detection_yolo11n"
RUN_NAME       = "yolo_finetune"
def fine_tune_yolo(config_file_path, model_path, epochs, batch, lr0, device):
    """Fine-tunes a YOLO model on the cyclist dataset."""

VIDEO_INPUT_DIR    = "video/samples"
VIDEO_OUTPUT_DIR   = "video/{}".format(RUN_NAME)

def fine_tune_yolo(config_file_path, model_path, epochs, batch, device):
    os.environ["WANDB_DISABLED"] = "true"

    model = YOLO(model_path).to(device)
    print(f"Training on device: {device}")

    results = model.train(
        data=config_file_path,
        epochs=epochs,
        batch=batch,
        device=device,
        patience=5,
        save_period=25,
        project=PROJECT_DIR,
        name=RUN_NAME,
        lr0=lr0,  # Starting learning rate
        patience=10,  # Stop training early if no improvement
        save_period=10,  # Save model after each epoch
        project="cyclist_detection_yolo26l",
        name="yolo_finetune_pdx",
        resume=False
    )

    # Optional validation
    val_results = model.val(device=device)
    print(val_results)

    # Ultralytics returns a "save_dir" we can use to find best.pt
    # Usually: runs/detect/<name> or <project>/<name> depending on settings.
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        # fallback: expected location from project/name
        save_dir = Path(PROJECT_DIR) / RUN_NAME
    else:
        save_dir = Path(save_dir)

    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    if best_pt.exists():
        ckpt = best_pt
    elif last_pt.exists():
        ckpt = last_pt
    else:
        raise FileNotFoundError(f"Could not find best.pt or last.pt under: {save_dir}/weights")

    print(f"Using checkpoint: {ckpt}")
    return str(ckpt)


def generate_inference_result(ckpt_path, input_dir, output_dir):
    """
    Run DeepSORT inference on all videos in input_dir
    and save results into output_dir.
    """

    # --- checks ---
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # --- load model once (important for speed) ---
    model = YOLO(ckpt_path, DEVICE)

    # supported video formats
    video_ext = {".mp4", ".mov", ".avi", ".mkv"}

    files = sorted(Path(input_dir).iterdir())

    for file in files:
        if file.suffix.lower() not in video_ext:
            continue

        input_video = str(file)
        output_video = str(Path(output_dir) / f"{file.stem}_tracked.mp4")

        print("\n==============================")
        print(f"Processing: {input_video}")
        print(f"Output:     {output_video}")

        process_video(
            input_video,
            output_video,
            model,
            CONFIDENCE,
            MAX_AGE,
            MAX_IOU_DIST,
            IOU_NMS,
            NO_DISPLAY
        )

    print("\nAll videos processed.")

if __name__ == "__main__":
    ckpt_path = fine_tune_yolo(CONFIG_FILE_PATH, BASE_MODEL_PATH, EPOCHS, BATCH, DEVICE)

    if GENERATE_VIDEO:
        generate_inference_result(ckpt_path, VIDEO_INPUT_DIR, VIDEO_OUTPUT_DIR)
