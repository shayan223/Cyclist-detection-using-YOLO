import cv2
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Configuration ---
#INPUT_VIDEO_CLIPS = ['./testing/downloaded_video.mp4']
INPUT_VIDEO_CLIPS = ['./June2023/SE Division St and SE 162nd EB Stop - 6-15-2023 - 0700-1900.mp4']
OUTPUT_VIDEO_PATH = 'deepsort_cyclist_count_output'
MODEL_PATH = 'Old_runs/yolo_finetune9/weights/best.pt'
CONF_THRESHOLD = 0.5
MAX_INACTIVE_FRAMES = 15
MAX_IOU_DISTANCE = 0.2
YOLO_IMGSIZE = 640
FEATURE_EXTRACTOR_RESIZE = (128, 256)

# --- Feature Extraction Setup ---
def setup_feature_extractor():
    """Sets up the feature extractor model."""
    feature_extractor = models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.eval().cuda()
    return feature_extractor

def setup_image_transform():
    """Sets up the image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize(FEATURE_EXTRACTOR_RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Feature Extraction ---
def extract_features(frame, bbox, feature_extractor, transform):
    """Extracts deep appearance features from a bounding box."""
    x1, y1, x2, y2 = bbox
    cropped_img = frame[y1:y2, x1:x2]
    if cropped_img.size == 0:
        return None
    pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).cuda()
    with torch.no_grad():
        features = feature_extractor(img_tensor)
        return features.view(-1).cpu().numpy()

# --- Tracking and Counting ---
def track_and_count_people(input_video, output_video, model_path):
    """Tracks cyclists in a video and counts unique cyclists."""

    model = YOLO(model_path).to('cuda')
    tracker = DeepSort(max_age=MAX_INACTIVE_FRAMES, max_iou_distance=MAX_IOU_DISTANCE)
    feature_extractor = setup_feature_extractor()
    transform = setup_image_transform()

    input_filename = os.path.splitext(os.path.basename(input_video))[0]
    output_video_file = f"{output_video}_{input_filename}.mp4"

    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    tracked_ids = set()
    unique_count = 0
    cyclist_counter = 0
    conf_thresholds = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=YOLO_IMGSIZE)
        detections = []

        for result in results:
            for box in result.boxes:
                if model.names[int(box.cls.item())] == "cyclist":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf >= CONF_THRESHOLD:
                        feature_vector = extract_features(frame, [x1, y1, x2, y2], feature_extractor, transform)
                        detections.append(([x1, y1, x2, y2], conf, feature_vector))
                        conf_thresholds[cyclist_counter] = conf
                        cyclist_counter += 1

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            tracked_ids.add(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"cyclist {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        unique_count = len(tracked_ids)
        cv2.putText(frame, f"Unique cyclist: {unique_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Finished processing video. Total unique cyclists: {unique_count}. Output: {output_video_file}")

# --- Main Execution ---
if __name__ == "__main__":
    for clip in INPUT_VIDEO_CLIPS:
        if not os.path.exists(clip):
            print(f"Video file does not exist: {clip}")
            continue
        track_and_count_people(clip, OUTPUT_VIDEO_PATH, MODEL_PATH)