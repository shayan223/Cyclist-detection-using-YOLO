# Cyclist-detection-using-YOLO

## Prerequisites:
    * Python 3.x
    * `opencv-python` (`pip install opencv-python`)
    * `ultralytics` (`pip install ultralytics`)
    * `torch` (`pip install torch`) (PyTorch is a dependency of `ultralytics`)
    * A pre-trained YOLO model weights file (e.g., `yolov8l.pt`). You can download these from the official YOLOv8 repository or other sources.

## Labeling Your Own Data/Videos

To label your own videos and add them to the training dataset:

1. **Label videos using movement_window.py:**
   ```bash
   python .\movement_window.py <video.mp4> --timestamps-csv <timestamp_file.csv>
   ```
   This will process your video and generate labeled training data in YOLO format. The script extracts frames from your video and allows you to annotate cyclists and pedestrians.

2. **Re-split the dataset after adding new data:**
   ```bash
   python .\split_pdx_dataset.py --allow-resplit
   ```
   After adding new labeled data to your dataset, run this command to combine all data from train/valid/test splits and redistribute them according to the configured ratios (default: 70% train, 20% validation, 10% test). This ensures your new data is properly integrated into all splits.

## annotate_raw_data.py:

### Key Features

* **Object Detection:** Utilizes the `ultralytics` YOLO library to detect 'person' and 'bicycle' objects in images.
* **Cyclist Identification:** Implements an IoU (Intersection over Union) calculation to determine if a detected person and bicycle overlap sufficiently to be considered a 'cyclist'.
* **YOLO Annotation Generation:** Converts the detected bounding box coordinates (for cyclists, persons, and bicycles) into the normalized YOLO format (`center_x`, `center_y`, `width`, `height`).
* **Class Labeling:** Assigns specific class labels in the annotation files:
    * `0`: cyclist (when a person and bicycle have a high IoU)
    * `1`: person (when a person is detected without a closely overlapping bicycle)
    * `2`: bicycle (when a bicycle is detected without a closely overlapping person)
* **Batch Processing:** Processes all images within a specified input directory.
* **Output Directory:** Saves the generated annotation files (in `.txt` format, one per image) in a designated output directory.
* **Device Agnostic:** Automatically utilizes a CUDA-enabled GPU if available, falling back to CPU if not.
* **Error Handling:** Includes basic error handling for image loading and directory existence.


## process_video_for_cyclist.py

### Key Features

* **YouTube Video Downloading:** Downloads videos directly from YouTube URLs using `yt-dlp` for processing.
* **Local Video Processing:** Processes video files (MP4, AVI, MOV, MKV) from a specified local folder.
* **YOLO-Powered Bicycle Detection:** Utilizes the `ultralytics` YOLO library for robust and efficient bicycle detection in video frames.
* **Intelligent Frame Extraction:** Employs Intersection over Union (IoU) to compare consecutive bicycle detections and skips saving frames with high overlap, reducing redundancy in the output image dataset.
* **Configurable Overlap Threshold:** The `overlap_between_frames` parameter allows customization of the sensitivity for frame skipping.
* **Confidence Filtering:** The `confidence_threshold` parameter enables filtering bicycle detections based on the YOLO model's confidence score.
* **Organized Output:** Saves extracted frames as sequentially numbered `.jpg` images in a designated output directory.
* **Hardware Acceleration:** Automatically leverages CUDA-enabled GPUs for faster processing, with CPU fallback.
* **Basic Error Handling:** Includes checks for directory existence, video file accessibility, and YouTube download issues.
* **Sequential Image Naming:** Output images are named with sequential numbers for easy organization.

## fine_tune_yolo.py

## Key Features

* **YOLO Model Fine-tuning:** Fine-tunes a pre-trained YOLO model (specified by `MODEL_PATH`, default: `yolov8l.pt`) for cyclist detection.
* **Custom Dataset Training:** Trains the model using a custom dataset defined by the `CONFIG_FILE_PATH` (default: `./training_data/config.yaml`), which should be in YOLO format.
* **Configurable Training Parameters:** Allows easy adjustment of key training parameters:
    * `EPOCHS`: Number of training epochs (default: 20).
    * `BATCH`: Batch size for training (default: 16).
    * `DEVICE`: Specifies the training device, automatically using CUDA if available (default: 'cuda' or 'cpu').
* **Weights & Biases Disabling:** Includes an option to disable Weights & Biases logging during training by default.
* **Early Stopping:** Implements `patience` to stop training early if no improvement is observed on the validation set, preventing overfitting.
* **Periodic Model Saving:** Saves the model weights after each epoch (`save_period=1`).
* **Organized Training Output:** Saves training results and model checkpoints within a `cyclist_detection` project and a `yolo_finetune` run.
* **Validation After Training:** Automatically performs validation on the trained model after the fine-tuning process is complete, providing evaluation metrics.


## test_fine_tuned_yolo.py

## Key Features

* **End-to-End Cyclist Tracking and Counting:** Processes video clips to detect, track, and count unique cyclists.
* **YOLO for Detection:** Utilizes a fine-tuned YOLO model (specified by `MODEL_PATH`) to accurately detect cyclists in each video frame.
* **DeepSORT for Tracking:** Integrates the DeepSORT algorithm (`deep_sort_realtime`) for robust tracking of detected cyclists across frames, even with occlusions.
* **Deep Feature Extraction:** Employs a ResNet-50 based feature extractor to generate deep appearance features for each detected cyclist, improving tracking accuracy during occlusions and re-identification.
* **Configurable Tracking Parameters:** Allows adjustment of DeepSORT parameters:
    * `MAX_INACTIVE_FRAMES`: Maximum number of frames a track can be inactive before being deleted.
    * `MAX_IOU_DISTANCE`: Maximum Intersection over Union (IoU) distance for associating detections with existing tracks.
* **Confidence Threshold:** Filters cyclist detections based on a `CONF_THRESHOLD` to ensure only high-confidence detections are tracked.
* **Video Input:** Accepts a list of input video clip paths (`INPUT_VIDEO_CLIPS`).
* **Video Output:** Saves the processed video with bounding boxes and track IDs overlaid, along with a unique cyclist count, to an output file (`OUTPUT_VIDEO_PATH`).
* **Image Preprocessing:** Includes image resizing and normalization for the feature extractor.
* **Clear Visual Output:** Draws bounding boxes and track IDs around detected and tracked cyclists in the output video, and displays the total unique cyclist count.

