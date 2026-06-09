# RTDETR_AUTO_EVAL

Auto-generate RT-DETR + DeepSORT inference YAML trials, run them on a clip with ground-truth labels, score detections against GT, and write `best_config.yaml` plus a leaderboard.

Tuning changes **hyperparameters in YAML only** — it does not train a new detector checkpoint.

## Pipeline

```
                         RTDETR_AUTO_EVAL PIPELINE
                         =========================

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  OPTIONAL: GROUND TRUTH (interpolate_annotate/)                           │
  └─────────────────────────────────────────────────────────────────────────┘

       your_video.mp4
            │
            ▼
   ┌────────────────────────┐
   │ interpolated_annotate  │  keyframes; optional gap fill (--interpolate-frames n)
   │        .py             │  → run_NNN/images/ + labels/ (YOLO)
   └───────────┬────────────┘
               ▼
   ┌────────────────────────┐
   │ convert_annotations_   │  sparse labels → full-length CSV
   │      to_csv.py         │
   └───────────┬────────────┘
               ▼
        ground_truth.csv          columns: frame, predictions_json


  ┌─────────────────────────────────────────────────────────────────────────┐
  │  CAMERA SETUP (once per camera, in data/base.yaml)                      │
  └─────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────┐
   │  base.yaml                                                           │
   │  • model: ./best.pt              weights (not tuned by sweep)        │
   │  • passes.warp:                  perspective ROI [CAMERA, not swept]   │
   │      enabled, src_points (4 corners, top-left clockwise),            │
   │      dst_size [W,H]              calibrate with warp_calibrate.py    │
   │  • passes.top_region / sahi      on/off + camera defaults              │
   │  • nms / classes / inference     template; sweep randomizes subsets  │
   └──────────────────────────────────────────────────────────────────────┘

   data/best.pt


  ┌─────────────────────────────────────────────────────────────────────────┐
  │  AUTO-EVAL (python -m rtdetr_eval)                                      │
  └─────────────────────────────────────────────────────────────────────────┘

        base.yaml + --n
               │
               ▼
   ┌────────────────────────┐
   │      explore           │  explore_001.yaml … explore_NNN.yaml + manifest.json
   └───────────┬────────────┘
               │
               │     video.mp4 + ground_truth.csv
               ▼
   ┌────────────────────────┐
   │     run-trials         │  loop: infer → score → rank
   │   (or `full` = both)   │
   └───────────┬────────────┘
               │
               ▼
   ┌────────────────────────┐
   │   deepSORT_rtdetr.py   │  RT-DETR + DeepSORT; --csv → predictions CSV
   └───────────┬────────────┘
               │
               ▼
     {video_stem}_rtdetr.csv
               │
               ▼
   ┌────────────────────────┐
   │      evaluate          │  GT vs pred → AP, IoU, trial score
   └───────────┬────────────┘
               ▼
     trial_leaderboard.json
     best_config.yaml


  WARP (one-time per camera, in base.yaml)
  ────────────────────────────────────────
    4 corners in raw frame  ──H──►  rectified patch  ──detect──►  boxes back via H_inv
    Calibrate with warp_calibrate.py; paste src_points + dst_size into YAML.
```

**Inference script (brief):** `deepSORT_rtdetr.py` loads YAML + RT-DETR, runs multi-pass detection (full frame, top crop, optional tiles/warp), NMS, then DeepSORT per class. The sweep calls `--config trial.yaml -i video.mp4 --csv`, which writes `{stem}_rtdetr.csv` next to the video.

## High-level steps

1. **Install** — venv + `pip install -r requirements.txt` from repo root.
2. **Prepare assets** — weights (`data/best.pt`), eval clip (`*.mp4`), ground-truth CSV (`frame`, `predictions_json`). Use bundled `data/camera_1/` or create GT with `interpolate_annotate/` (see that folder’s README).
3. **Calibrate camera** — edit `data/base.yaml`: model path, warp `src_points` / `dst_size`, and any fixed pass toggles. Run `warp_calibrate.py` (external) for perspective corners if needed.
4. **Sweep** — `explore` samples random trial YAMLs; `run-trials` runs `deepSORT_rtdetr.py` on the clip for each trial and scores vs GT; `refine` warm-starts a focused search around the best; `search` chains all of it. `full` does explore + run once.
5. **Use results** — `best_config.yaml` is the winning hyperparameters; `trial_leaderboard.json` ranks all trials.

**Standalone (no sweep):** run `deepSORT_rtdetr.py` once, then `python -m rtdetr_eval eval --gt … --pred …`.

## Setup

From the **repository root** (required so `python -m rtdetr_eval` and model paths resolve):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check resolved defaults:

```bash
python3 -m rtdetr_eval print-paths
```

## Quick run (bundled `data/` layout)

Sample assets live under `data/` (`best.pt`, `camera_1/trim5.mp4`, `ground_truth.csv`). Use explicit paths and set inference cwd to `data/` so `model: ./best.pt` resolves:

```bash
python3 -m rtdetr_eval full \
  --config data/base.yaml \
  --n 20 \
  --seed 42 \
  --trials-dir data/trials/camera_1 \
  --video data/camera_1/trim5.mp4 \
  --gt data/camera_1/ground_truth.csv \
  --inference-cwd data
```

Outputs: `data/trials/camera_1/best_config.yaml`, `top_params.json`, `trial_leaderboard.json`, `explore_*.yaml`, `manifest.json`.

## Project experiment suite

From `RT_DETR/RTDETR_AUTO_EVAL`, run this command to execute the project
baseline/manual/tuned comparison suite and generate study-ready results:

```powershell
python -m rtdetr_eval suite --model ..\pdx_finetuned_rtdetr.pt --video ..\..\trim5.mp4 --gt data\camera_1\ground_truth.csv --manual-config ..\config_trim5.yaml --infer-script ..\deepSORT_rtdetr.py --inference-cwd .. --n 50 --refine-n 30 --seed 42
```

This suite runs four comparable experiments on the same staged copy of
`trim5.mp4`: bare-bones detector-only inference, the manually chosen
`config_trim5.yaml`, the best trial from a two-phase AUTO_EVAL search, and a
rerun of the winning `best_config.yaml`. Outputs are written under
`runs/camera_1_trim5/<timestamp>/` and include generated YAML configs,
prediction CSVs, evaluation JSON files, PR/IoU plots, the trial leaderboard,
the tuned best config, and `summary.csv`.

`summary.csv` is the main table for further study. It includes the suite score,
mAP, per-class AP/precision/recall for Cyclist and Pedestrian, IoU threshold,
matched-box count, mean/median/min/max IoU, and percent of matched boxes above
0.75 IoU.

## CLI overview

| Command | Purpose |
| --- | --- |
| `explore` | Phase 1: sample random `explore_*.yaml` from a base config |
| `run-trials` | Run inference for each trial, rank, write `best_config.yaml` + `top_params.json` |
| `refine` | Phase 2: warm-start `exploit_*.yaml` by perturbing around the top-K seeds |
| `search` | Two-phase warm-start: `explore` -> `run-trials` -> `refine` -> `run-trials` |
| `suite` | Run bare-bones, manual-config, and tuned-search comparisons; write `summary.csv` |
| `full` | `explore` then `run-trials` (single pass, no warm-start) |
| `eval` | One-off GT vs predictions CSV (+ plots) |
| `print-paths` | Print resolved default paths for this repo |

(`generate` remains as a hidden alias of `explore` for back-compat.)

## Two-phase warm-start search

Phase 1 samples the full space randomly; Phase 2 exploits it by drawing Gaussian
perturbations around the **top-K** trials (default 3), with the perturbation
budget split **weighted by score**. This is more robust than seeding from a
single best trial, since each trial score comes from one noisy video pass.

One command (explore -> run -> refine -> run):

```bash
python3 -m rtdetr_eval search \
  --config data/base.yaml \
  --n 50 --refine-n 30 --sigma 0.15 --top-k 3 --seed 42 \
  --trials-dir data/trials/camera_1 \
  --video data/camera_1/trim5.mp4 \
  --gt data/camera_1/ground_truth.csv \
  --inference-cwd data
```

Or step the phases manually:

```bash
# Phase 1
python3 -m rtdetr_eval explore --config data/base.yaml --n 50 --seed 42 \
  --out-dir data/trials/camera_1
python3 -m rtdetr_eval run-trials --trials-dir data/trials/camera_1 \
  --video data/camera_1/trim5.mp4 --gt data/camera_1/ground_truth.csv --inference-cwd data

# Phase 2 (reads data/trials/camera_1/top_params.json)
python3 -m rtdetr_eval refine --config data/base.yaml --trials-dir data/trials/camera_1 \
  --n 30 --sigma 0.15 --top-k 3
python3 -m rtdetr_eval run-trials --trials-dir data/trials/camera_1 \
  --video data/camera_1/trim5.mp4 --gt data/camera_1/ground_truth.csv --inference-cwd data
```

## Defaults (paths)

Paths are centralized in `rtdetr_eval/paths.py`. Code defaults point at `inference_parameter/`; if that tree is missing, pass `--video`, `--gt`, `--trials-dir`, and `--inference-cwd` explicitly (as in the quick run above).

- **Trials directory:** `inference_parameter/trials/camera_1/`
- **Ground truth:** `inference_parameter/camera_1/ground_truth.csv` (or `gt.csv`)
- **Video:** `inference_parameter/camera_1/trim5.mp4`, or the only `.mp4` in that folder

`run-trials` and `full` invoke `deepSORT_rtdetr.py` with cwd `inference_parameter/` by default so relative `model:` paths in YAML resolve. Override with `--inference-cwd data` when weights and YAML live under `data/`.

## Step-by-step sweep

**1. Explore (random trial YAMLs)**

```bash
python3 -m rtdetr_eval explore \
  --config data/base.yaml \
  --n 50 \
  --seed 42 \
  --out-dir data/trials/camera_1
```

**2. Run the sweep**

```bash
python3 -m rtdetr_eval run-trials \
  --trials-dir data/trials/camera_1 \
  --video data/camera_1/trim5.mp4 \
  --gt data/camera_1/ground_truth.csv \
  --inference-cwd data
```

Optional: `--eval-plots` for PR/IoU plots per trial (slower).

**3. Read outputs** (in the trials directory)

- `explore_*.yaml` / `exploit_*.yaml`, `manifest.json` — sampled hyperparameters
- `best_config.yaml` — winner for production inference on this camera
- `top_params.json` — top-K seeds (params + score) consumed by `refine`
- `trial_leaderboard.json` — ranked scores and per-class AP

For the focused second pass, run `refine` then `run-trials` again (see the
two-phase section above), or just use the one-command `search`.

## Ground truth (optional)

If you need labels for a new clip:

```bash
cd interpolate_annotate
python3 interpolated_annotate.py /path/to/video.mp4 --interpolate-frames 10
python3 convert_annotations_to_csv.py \
  --run-dir run_001 \
  --video /path/to/video.mp4 \
  --output /path/to/ground_truth.csv
```

See `interpolate_annotate/README.md` for controls and interpolation rules.

## Single evaluation (no sweep)

```bash
python3 -m rtdetr_eval eval \
  --gt data/camera_1/ground_truth.csv \
  --pred data/camera_1/trim5_rtdetr.csv \
  --out ./eval_out
```

## Layout reference

| Path | Role |
| --- | --- |
| `deepSORT_rtdetr.py` | Inference entrypoint (RT-DETR + DeepSORT, CSV export) |
| `data/` | Weights, base YAML, sample video/GT, trial outputs |
| `inference_parameter/` | Alternate layout expected by code defaults |
| `rtdetr_eval/` | `search` (explore + refine), `evaluate`, `trials`, CLI |
| `interpolate_annotate/` | Labeling + CSV conversion for GT |
| `tests/` | Unit tests |

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
