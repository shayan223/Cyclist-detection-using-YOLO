# Video annotation (keyframe + interpolation)

Label cyclists and pedestrians in video. You save **keyframes** by hand; the tool can write labels for frames **between** two keyframes when the span is within the limit.

---

## Quick start

```bash
cd annotation
python interpolated_annotate.py ../your_video.mp4 --interpolate-frames 10
```

| Flag | Values | Meaning |
|------|--------|---------|
| `--interpolate-frames` | `0`, `5`, `10` | Max gap between keyframes to auto-fill (`0` = off) |

Output (each session): `run_001/images/`, `run_001/labels/` (YOLO), `run_001/data.yaml`.

---

## Gap rule

**Gap** = frame number of the keyframe you just saved − previous keyframe.

Interpolation runs only when:

```
2 ≤ gap ≤ n
```

- `gap < 2` — no room between frames (e.g. 10 → 11)
- `gap > n` — span too wide; only the new keyframe is saved
- `n = 0` — interpolation disabled

---

## Example: keyframes at 10, 20, 30, 40, 50

Pause, annotate boxes, press **`s`** on each frame below. Use **`--interpolate-frames 10`** so gap 10 is allowed.

| Save at | Gap | Middle frames auto-labeled |
|---------|-----|----------------------------|
| 10 | — | (none — first keyframe) |
| 20 | 10 | 11, 12, 13, 14, 15, 16, 17, 18, 19 |
| 30 | 10 | 21–29 |
| 40 | 10 | 31–39 |
| 50 | 10 | 41–49 |

With default **`n = 5`**, gap 10 is too large: only frames 10, 20, 30, 40, 50 are saved.

For each middle frame `F` between keyframe **A** and **B**:

```
t = (F - A) / (B - A)
```

Example **A = 10**, **B = 20**: frame 15 has `t = 5/10 = 0.5` (halfway between A and B boxes).

---

## Many boxes in the scene: how matching works

When you save keyframe **B**, the tool compares boxes on the **previous** keyframe **A** to boxes on **B**. It does **not** track IDs across the whole video—only this one pair (A, B).

### Step 1 — Pair objects (once per span)

For every box on **A**, in list order:

1. Consider boxes on **B** with the **same class** (cyclist `0` or pedestrian `1`).
2. Measure **center-to-center** distance.
3. Accept a candidate only if distance `< 2 × max(width, height)` of the **A** box.
4. Pick the **closest** unused candidate on **B**.
5. Mark that **B** box as used (one-to-one).

Then:

- **A** box with no valid **B** match → `(A, None)` — left the scene
- **B** box never used → `(None, B)` — entered the scene

### Step 2 — Fill every middle frame

The same pairs are used for **all** frames A+1 … B−1. Only **`t`** changes per frame.

| Pair type | Middle frames |
|-----------|----------------|
| Matched (A and B) | `lerp`: position and size blend from A → B |
| (A, None) | Keep A’s box while `t < 0.35`, then drop |
| (None, B) | No box until `t > 0.65`, then use B’s box as-is |

---

## Worked example: 3 objects, keyframes 10 → 20

**Frame 10** (you save):

| ID | Class | Role |
|----|-------|------|
| Alice | cyclist | moving right |
| Bob | pedestrian | center |
| Carol | cyclist | far left |

**Frame 20** (you save):

| ID | Class | Role |
|----|-------|------|
| Alice′ | cyclist | right of Alice on 10 |
| Bob′ | pedestrian | near Bob on 10 |
| Dave | cyclist | new on the right (Carol gone) |

**Matching result:**

| Pair | Meaning |
|------|---------|
| (Alice, Alice′) | Same class, closest centers → interpolate 10→20 |
| (Bob, Bob′) | Same class, closest → interpolate |
| (Carol, None) | No cyclist on 20 close enough → exit rules |
| (None, Dave) | New cyclist on 20 → enter rules |

**Frame 12** (`t = 0.2`):

- Alice, Bob: lerped toward frame 20
- Carol: frozen box from frame 10 (`t < 0.35`)
- Dave: not on disk yet

**Frame 15** (`t = 0.5`):

- Alice, Bob: lerped (halfway)
- Carol: **gone** (`t` is not `< 0.35`)
- Dave: **gone** (`t` is not `> 0.65`)

**Frame 18** (`t = 0.8`):

- Alice, Bob: lerped (near frame 20)
- Carol: still gone
- Dave: full box from frame 20 (`t > 0.65`)

---

## Output from `interpolated_annotate.py`

Each session creates a new folder in the directory where you run the tool (e.g. `annotation/run_001/`).

```
run_001/
├── images/
│   ├── frame_00000010.jpg    # one JPEG per saved frame (video frame index in name)
│   ├── frame_00000011.jpg
│   └── ...
├── labels/
│   ├── frame_00000010.txt    # YOLO labels (same stem as image)
│   ├── frame_00000011.txt
│   └── ...
└── data.yaml                 # class names + train image path for YOLO training
```

| Console tag | Meaning |
|-------------|---------|
| `[keyframe]` | You pressed **`s`** on that frame |
| `[interp]` | Auto-filled between two keyframes |

**Only annotated frames are written** — there is no file for frames you never saved.

### YOLO label file (`.txt`)

One line per object. Values are **normalized** to image width/height (0–1):

```
<class_id> <cx> <cy> <w> <h>
```

| Field | Values |
|-------|--------|
| `class_id` | `0` = cyclist, `1` = pedestrian |
| `cx`, `cy` | Box center |
| `w`, `h` | Box width and height |

Example `labels/frame_00000010.txt` with **two** boxes:

```
1 0.064236 0.275463 0.031250 0.089815
0 0.131944 0.237037 0.033333 0.070370
```

Line 1 = pedestrian; line 2 = cyclist. Order in the file is the order boxes were saved (used internally for matching on the next span).

### `data.yaml`

```yaml
names:
- cyclist
- pedestrian
nc: 2
train: /absolute/path/to/run_001/images
```

---

## Output conversion: `convert_annotations_to_csv.py`

Turns a `run_NNN` folder into a **per-video-frame CSV** in the same shape as `deepSORT_rtdetr.py --csv` (for eval or comparison with detector output).

```bash
python convert_annotations_to_csv.py \
  --run-dir run_001 \
  --video ../your_video.mp4 \
  --output run_001_annotations.csv
```

| Argument | Required | Role |
|----------|----------|------|
| `--run-dir` | yes | Folder with `images/` and `labels/` |
| `--video` | no* | Frame count + image size (width/height) |
| `--output` | no | CSV path (default: `<run-dir>/annotations.csv`) |
| `--total-frames` | no | Pad CSV length without a video file |

\* Without `--video`, image size comes from the first JPEG in `images/`; frame count falls back to `max(annotated frame) + 1` unless `--total-frames` is set.

### What the converter does

1. Read every `labels/frame_XXXXXXXX.txt`.
2. Parse **frame number** from the filename (e.g. `frame_00000010` → `10`).
3. Convert each YOLO line → pixel bbox `[x1, y1, x2, y2]` using video (or image) dimensions.
4. Write **one CSV row per frame** from `0` to `total_frames - 1`.
5. Frames with no label file get an **empty** prediction list `[]`.

### CSV columns

| Column | Type | Content |
|--------|------|---------|
| `frame` | int | Frame index (0-based, aligned with video) |
| `predictions_json` | JSON string | List of detections on that frame |

### `predictions_json` — one object per box

Each annotated box becomes one dict:

```json
{
  "class_id": 1,
  "class_name": "Pedestrian",
  "confidence": 1.0,
  "bbox": [260, 73, 276, 113]
}
```

| Field | Source |
|-------|--------|
| `class_id` | YOLO class column (`0` / `1`) |
| `class_name` | `"Cyclist"` or `"Pedestrian"` |
| `confidence` | Always `1.0` (ground truth, not a model score) |
| `bbox` | `[x1, y1, x2, y2]` in **pixels** (top-left, bottom-right) |

### Example CSV rows (multiple boxes)

Frame **5** with two pedestrians (from a real run):

```csv
frame,predictions_json
5,"[{""class_id"": 1, ""class_name"": ""Pedestrian"", ""confidence"": 1.0, ""bbox"": [260, 73, 276, 113]}, {""class_id"": 1, ""class_name"": ""Pedestrian"", ""confidence"": 1.0, ""bbox"": [149, 434, 199, 476]}]"
```

Frame **4** with no labels on disk:

```csv
4,[]
```

### Pipeline overview

```
interpolated_annotate.py
        │
        ▼
  run_001/images/*.jpg  +  labels/*.txt  (sparse frames only)
        │
        ▼
convert_annotations_to_csv.py  +  --video
        │
        ▼
  annotations.csv  (every frame 0…N-1; empty [] where unannotated)
        │
        ▼
  compare with deepSORT_rtdetr.py --csv output / evaluate_detections.py
        │
        ▼
  visualize.py  (optional QC on saved images + labels)
```

---

## Visualization: `visualize.py`

Lightweight viewer to **check saved annotations** after a run. It steps through every JPEG in a `run_NNN/images/` folder and draws the matching YOLO boxes from `labels/`.

### Setup

```bash
cd annotation
pip install opencv-python natsort
```

Edit the run folder at the top of `visualize.py` (line 6):

```python
run = Path("./run_002")   # change to run_001, run_002, etc.
```

There is a commented one-liner to auto-pick the latest `run_*` folder by number; uncomment it if you prefer that.

### What it does

1. List all `images/*.jpg` in **natural sort order** (`frame_…` order is correct).
2. For each image, load the paired `labels/<same_stem>.txt` if it exists.
3. Convert each YOLO line (`class cx cy w h`, normalized) → pixel rectangle.
4. Draw box + class label on the image and show it in a window.

### Colors (BGR)

| Class | `class_id` | Color | Label text |
|-------|------------|-------|------------|
| cyclist | `0` | Green `(0, 255, 0)` | `cyclist` |
| pedestrian | `1` | Magenta `(255, 0, 255)` | `pedestrian` |

Matches the annotator palette for cyclists (green) and pedestrians (magenta/pink).

### Viewer controls

| Key | Action |
|-----|--------|
| *(any other key)* | Advance to next image (~16 ms delay) |
| `Space` | Pause on current frame (wait for another key) |
| `q` | Quit |

Frames **without** a `.txt` file still display (image only, no boxes).

### When to use

- After `interpolated_annotate.py`: spot-check keyframes and interpolated frames.
- Before `convert_annotations_to_csv.py`: confirm boxes look correct on disk.
- Compare two runs by changing `run = Path(...)` and re-running.

`visualize.py` does **not** read CSV or the source video—only `run_NNN/images` + `labels`.

---

## Controls (while paused)

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `s` | Save keyframe (+ interpolate if gap OK) |
| `t` | Toggle cyclist / pedestrian |
| `c` | Clear selections |
| `f` / `b` | Step +10 / −10 frames |
| `n` / `p` | Next / previous CSV timestamp (if `--timestamps-csv` set) |
| `q` | Quit |

After **`s`**, the playhead jumps forward **`n`** frames to help place the next keyframe.

---

## Summary

1. Save keyframes on sparse frames (e.g. 10, 20, …, 50); set **`n`** ≥ your spacing.
2. Each span: **match** boxes A↔B, then **lerp** middle frames (see matching section above).
3. Native output: `run_NNN/images` + `labels` (YOLO `.txt`, sparse frames).
4. Optional: `convert_annotations_to_csv.py` → full-length CSV for eval (`frame`, `predictions_json`).
5. Optional: `visualize.py` → step through saved images and overlay labels for QC.
