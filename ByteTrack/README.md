# ByteTrack — RF-DETR + ByteTrack Tracking

Multi-object tracking using ByteTrack with two interchangeable model backends.

## Files

| File | Purpose |
|---|---|
| `bytetrack_rfdetr.py` | Main inference + tracking script |
| `fine_tune_rfdetr.py` | Fine-tune RF-DETR on the PDX cyclist/pedestrian dataset |

## Installation

```bash
pip install supervision inference-gpu
pip install git+https://github.com/roboflow/trackers.git
pip install rfdetr ultralytics
```

> Use `inference` instead of `inference-gpu` if you don't have a CUDA GPU.

---

## Inference — `bytetrack_rfdetr.py`

### Model backends

| Backend | When it's used | Model |
|---|---|---|
| `inference` (default) | No `--model` flag | `rfdetr-seg-medium` hosted on Roboflow |
| `ultralytics` | `--model path/to/model.pt` | Your local RT-DETR or YOLO `.pt` file |

The backend is detected automatically from the file extension. Use `--backend` to force one.

---

### RF-DETR (Roboflow inference backend)

**Default — segmentation model, all street classes tracked:**
```bash
python ByteTrack/bytetrack_rfdetr.py --input trim_3.mp4
```

**Bounding-box model instead of segmentation:**
```bash
python ByteTrack/bytetrack_rfdetr.py --input trim_3.mp4 --bbox
```

**Different hosted model:**
```bash
python ByteTrack/bytetrack_rfdetr.py --input trim_3.mp4 --model-id rfdetr-large
```

**Adjust confidence and NMS:**
```bash
python ByteTrack/bytetrack_rfdetr.py --input trim_3.mp4 --confidence 0.3 --nms-threshold 0.4
```

**Reduce inference resolution for faster processing (recommended for 1080p+):**
```bash
# 640px on longest edge — good speed/accuracy balance
python ByteTrack/bytetrack_rfdetr.py --input trim_3.mp4 --imgsz 640

# 480px — faster, some accuracy loss on small/distant objects
python ByteTrack/bytetrack_rfdetr.py --input trim_3.mp4 --imgsz 480
```

**Draw deadzones interactively before processing:**
```bash
# Left-click + drag to draw exclusion zones, Enter to confirm
python ByteTrack/bytetrack_rfdetr.py --input trim_3.mp4 --deadzone

# Also render deadzone overlays in the output video
python ByteTrack/bytetrack_rfdetr.py --input trim_3.mp4 --deadzone --show-deadzones
```

**Detector-only (no tracking, raw detections):**
```bash
python ByteTrack/bytetrack_rfdetr.py --input trim_3.mp4 --inference-only
```

---

### In-house RT-DETR model (ultralytics backend)

The custom model mode filters to **cyclist and pedestrian only** with separate
trackers per class. Class IDs default to `0=cyclist, 1=pedestrian` — adjust
with `--cyclist-class-id` and `--pedestrian-class-id` if your model differs.

**Basic — auto-detected as ultralytics from the `.pt` extension:**
```bash
python ByteTrack/bytetrack_rfdetr.py \
  --input trim_3.mp4 \
  --model RT_DETR/runs/train/weights/best.pt
```

**With resolution scaling for faster inference:**
```bash
python ByteTrack/bytetrack_rfdetr.py \
  --input trim_3.mp4 \
  --model RT_DETR/runs/train/weights/best.pt \
  --imgsz 640
```

**Custom class IDs (if your model uses different indices):**
```bash
python ByteTrack/bytetrack_rfdetr.py \
  --input trim_3.mp4 \
  --model RT_DETR/runs/train/weights/best.pt \
  --cyclist-class-id 1 \
  --pedestrian-class-id 0
```

**With deadzones:**
```bash
python ByteTrack/bytetrack_rfdetr.py \
  --input trim_3.mp4 \
  --model RT_DETR/runs/train/weights/best.pt \
  --deadzone --show-deadzones
```

**Adjust ByteTrack parameters:**
```bash
python ByteTrack/bytetrack_rfdetr.py \
  --input trim_3.mp4 \
  --model RT_DETR/runs/train/weights/best.pt \
  --track-activation-threshold 0.3 \
  --lost-track-buffer 45 \
  --minimum-matching-threshold 0.75
```

---

### All inference flags

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | `../trim_3.mp4` | Input video |
| `--output` / `-o` | auto | Output video path |
| `--model` / `-m` | *(none)* | Local `.pt`/`.onnx` — triggers ultralytics backend |
| `--backend` | `auto` | `auto` / `inference` / `ultralytics` |
| `--bbox` | off | Use `rfdetr-medium` (no masks) instead of `rfdetr-seg-medium` |
| `--model-id` | `rfdetr-seg-medium` | Roboflow inference model ID |
| `--confidence` / `-c` | `0.2` | Detection confidence threshold |
| `--nms-threshold` | `0.3` | NMS IoU threshold |
| `--imgsz` | `0` (full res) | Resize longest edge before inference for speed |
| `--cyclist-class-id` | `0` | Cyclist class ID (ultralytics backend) |
| `--pedestrian-class-id` | `1` | Pedestrian class ID (ultralytics backend) |
| `--track-activation-threshold` | `0.25` | Min confidence to activate a new track |
| `--lost-track-buffer` | `30` | Frames to keep a lost track alive |
| `--minimum-matching-threshold` | `0.8` | IoU threshold for matching detections to tracks |
| `--minimum-consecutive-frames` | `1` | Frames before a track is confirmed |
| `--deadzone` | off | Interactively draw exclusion zones on the first frame |
| `--show-deadzones` | off | Render deadzone overlays in output video |
| `--inference-only` | off | Detect only, no tracking |

---

## Fine-tuning — `fine_tune_rfdetr.py`

Fine-tunes `rfdetr-medium` (bounding box) on the PDX cyclist/pedestrian dataset.
The segmentation variant (`rfdetr-seg-medium`) is not directly fine-tuneable with
this dataset as it requires polygon mask annotations — see `--seg` note below.

### Standard fine-tune (2-class cyclist/pedestrian)

Replaces the COCO head with a 2-class head. Best accuracy for cyclist/pedestrian,
but loses all other COCO class detections.

**Default — uses `../v5_pdx_cyclist_dataset/data.yaml`:**
```bash
python ByteTrack/fine_tune_rfdetr.py
```

**Custom dataset:**
```bash
python ByteTrack/fine_tune_rfdetr.py --data ../my_dataset/data.yaml
```

**Larger model:**
```bash
python ByteTrack/fine_tune_rfdetr.py --size large
```

**Custom training settings:**
```bash
python ByteTrack/fine_tune_rfdetr.py \
  --epochs 150 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --lr 5e-5 \
  --resolution 560
```

**Resume an interrupted run:**
```bash
python ByteTrack/fine_tune_rfdetr.py \
  --resume ByteTrack/runs/rfdetr-medium_res560/checkpoint.pth
```

**Continue from a previous fine-tuned checkpoint:**
```bash
python ByteTrack/fine_tune_rfdetr.py \
  --pretrain-weights ByteTrack/runs/rfdetr-medium_res560/checkpoint_best_total.pth
```

---

### Class-preserving fine-tune (add cyclist, keep other COCO classes)

Goal: add cyclist detection from our dataset while minimising degradation of
cars, buses, people, and the other 77 COCO classes.

What `--preserve-classes` does automatically:
- **Freezes the encoder** (`lr_encoder ≈ 0`) — backbone features stay intact
- **Remaps class IDs** onto COCO — cyclist → bicycle (ID 1), pedestrian → person (ID 0)
- **Conservative augmentation** — reduced mosaic/mixup to limit weight drift
- **Lowers defaults** — LR to `1e-5`, epochs to `40`

> **Note:** Without COCO data mixed in, the other 78 classes will still degrade
> over time — just slower. `--coco-data` is the best fix.

**Class-preserving, no COCO data (faster, some other-class degradation):**
```bash
python ByteTrack/fine_tune_rfdetr.py --preserve-classes
```

**Class-preserving + COCO data mixing (recommended — keeps all classes active):**
```bash
python ByteTrack/fine_tune_rfdetr.py \
  --preserve-classes \
  --coco-data ../coco_subset
```

**Class-preserving with explicit hyperparameters:**
```bash
python ByteTrack/fine_tune_rfdetr.py \
  --preserve-classes \
  --lr 1e-5 \
  --epochs 50 \
  --lr-encoder 1e-7
```

---

### Segmentation model fine-tuning

`rfdetr-seg-medium` requires COCO JSON with polygon mask annotations. Our
dataset has bounding boxes only so this will show an error with instructions.

If you have a segmentation-annotated dataset (e.g. exported from Roboflow as
"COCO Segmentation JSON"):
```bash
python ByteTrack/fine_tune_rfdetr.py --seg --data ../my_seg_dataset
```

---

### Using a fine-tuned checkpoint for inference

After training, use the output checkpoint with `bytetrack_rfdetr.py` via the
ultralytics backend:

```bash
python ByteTrack/bytetrack_rfdetr.py \
  --input trim_3.mp4 \
  --model ByteTrack/runs/rfdetr-medium_res560/checkpoint_best_total.pth \
  --bbox
```

> The `rfdetr` package loads `.pth` checkpoints directly. The ultralytics
> backend auto-detects the file extension. Use `--bbox` to skip mask annotation
> since this is a detection (not segmentation) checkpoint.

---

### All fine-tuning flags

| Flag | Default | Description |
|---|---|---|
| `--data` | `../v5_pdx_cyclist_dataset/data.yaml` | YOLO yaml, YOLO dir, or COCO dir |
| `--bbox` | *(default)* | Fine-tune detection model (compatible with YOLO datasets) |
| `--seg` | off | Fine-tune segmentation model (requires polygon mask annotations) |
| `--size` | `medium` | `small` / `medium` / `large` |
| `--pretrain-weights` | *(none)* | Continue from an existing `.pth` checkpoint |
| `--preserve-classes` | off | Freeze encoder + remap IDs + conservative settings |
| `--coco-data` | *(none)* | COCO dataset to mix in (prevents other-class forgetting) |
| `--epochs` | `100` (40 with `--preserve-classes`) | Training epochs |
| `--batch-size` | `4` | Batch size per GPU |
| `--grad-accum-steps` | `4` | Gradient accumulation (effective batch = batch × accum) |
| `--lr` | `1e-4` (1e-5 with `--preserve-classes`) | Learning rate for decoder/head |
| `--lr-encoder` | *(same as `--lr`)* | Separate LR for encoder backbone |
| `--resolution` | `560` | Input resolution (must be divisible by 56) |
| `--checkpoint-interval` | `10` | Save a checkpoint every N epochs |
| `--no-early-stopping` | off | Disable early stopping |
| `--early-stopping-patience` | `15` | Epochs without improvement before stopping |
| `--no-augment` | off | Use rfdetr augmentation defaults |
| `--resume` | *(none)* | Resume from `checkpoint.pth` after an interrupted run |
| `--output-dir` | `ByteTrack/runs/rfdetr-<size>_res<N>/` | Where to save checkpoints |
