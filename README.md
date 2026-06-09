# Cyclist and Pedestrian Detection with RT-DETR

This repository is organized around the current RT-DETR cyclist/pedestrian
pipeline. Older YOLO, RF-DETR, RTMDet, ensemble, ByteTrack, dataset, checkpoint,
and media artifacts have been moved into `deprecated_content/` so the active
workflow is easier to find.

## Active Workflow

The maintained pipeline is:

1. Prepare or label PDX video data.
2. Split and optionally augment the YOLO-format dataset.
3. Train or fine-tune RT-DETR on the PDX dataset.
4. Run RT-DETR inference and DeepSORT tracking on target videos.
5. Tune inference/tracking config with `RT_DETR/RTDETR_AUTO_EVAL`.
6. Generate reproducible result tables, figures, configs, and videos with
   `results_reporting`.

## Active Layout

- `RT_DETR/` - current RT-DETR training, validation, inference, PET analysis,
  config tuning, tests, checkpoints, and AUTO_EVAL tooling.
- `RT_DETR/RTDETR_AUTO_EVAL/` - current inference-parameter search and
  evaluation suite.
- `results_reporting/` - reproducible report generation for the completed
  AUTO_EVAL run.
- `v5_pdx_cyclist_dataset/` - current PDX training dataset used by RT-DETR.
- `raw_pdx_videos/` - raw and labeling source material.
- Root `trim*.mp4` and the SE Division video - active inference inputs
  referenced by RT-DETR config files.
- Root data-prep scripts - active dataset creation, splitting, augmentation,
  and conversion helpers.
- `deprecated_content/` - archived legacy content kept for reference, but not
  part of the current pipeline.

## Environment

Install the project dependencies from the repository root:

```powershell
pip install -r requirements.txt
```

Key runtime packages include `ultralytics`, `torch`, `opencv-python`, `pandas`,
`pyyaml`, `matplotlib`, and `deep_sort_realtime`.

## Data Preparation

Label PDX video data into a YOLO-format dataset:

```powershell
python .\movement_window.py <video.mp4> --timestamps-csv <timestamp_file.csv> --dataset-dir v5_pdx_cyclist_dataset
```

Re-split the dataset after adding or editing labels:

```powershell
python .\split_pdx_dataset.py --dataset-dir v5_pdx_cyclist_dataset --allow-resplit
```

Optionally generate an augmented dataset copy:

```powershell
python .\augment_dataset_3x.py --dataset-dir v5_pdx_cyclist_dataset --output-dir <augmented_dataset_dir> --multiplier 3
```

Combine external YOLO-format datasets when rebuilding a training set:

```powershell
python .\combine_datasets.py --input-dirs <dataset_a> <dataset_b> --output-dir <merged_dataset> --filter-classes cyclist pedestrian
```

## RT-DETR Training

The current training entrypoint is:

```powershell
cd .\RT_DETR
python .\fine_tune_rtdetr.py --data ..\v5_pdx_cyclist_dataset\data.yaml --model .\rt_detr_macro_augmented.pt
```

Useful defaults live in `RT_DETR/fine_tune_rtdetr.py`:

- Dataset: `../v5_pdx_cyclist_dataset/data.yaml`
- Base checkpoint: `rt_detr_macro_augmented.pt`
- Project name: `cyclist_detection_rtdetr`
- Run name prefix: `rtdetr_finetune`

Validate a trained checkpoint:

```powershell
cd .\RT_DETR
python .\validate_rtdetr.py --model <checkpoint.pt> --data ..\v5_pdx_cyclist_dataset\data.yaml
```

Export a PyTorch checkpoint for inference:

```powershell
cd .\RT_DETR
python .\export_engine.py --model .\pdx_finetuned_rtdetr.pt --format onnx
```

## Inference And PET

RT-DETR inference configs are in `RT_DETR/config_*.yaml`. They intentionally
reference root input videos such as `../trim5.mp4`, so those videos remain in
the repository root.

Run RT-DETR plus DeepSORT inference:

```powershell
cd .\RT_DETR
python .\deepSORT_rtdetr.py --config .\config_trim5.yaml
```

Export frame-level predictions while running inference:

```powershell
cd .\RT_DETR
python .\deepSORT_rtdetr.py --config .\config_trim5.yaml --csv
```

Run PET analysis:

```powershell
cd .\RT_DETR
python .\PET_deepSORT.py --config .\config_trim5.yaml
```

For single-cell PET analysis:

```powershell
cd .\RT_DETR
python .\PET_deepSORT.py --config .\config_trim5.yaml --no-grid --grid-size 8
```

## Config Tuning

`RT_DETR/RTDETR_AUTO_EVAL` tunes inference and tracking YAML parameters. It does
not retrain the detector checkpoint.

From `RT_DETR/RTDETR_AUTO_EVAL`, run the current project suite:

```powershell
python -m rtdetr_eval suite --model ..\pdx_finetuned_rtdetr.pt --video ..\..\trim5.mp4 --gt data\camera_1\ground_truth.csv --manual-config ..\config_trim5.yaml --infer-script ..\deepSORT_rtdetr.py --inference-cwd .. --n 50 --refine-n 30 --seed 42
```

The current preserved suite run is:

```text
RT_DETR/RTDETR_AUTO_EVAL/runs/camera_1_trim5/20260608_220235
```

## Results Reporting

Regenerate reproducible report outputs from the preserved AUTO_EVAL suite:

```powershell
python .\results_reporting\generate_results.py
```

This writes tables, figures, copied comparison configs, copied report videos,
and a manifest under `results_reporting/outputs/`.

## Deprecated Content

Legacy material was moved, not deleted. See `deprecated_content/README.md` for
the archive map. Archived files are kept for reference and comparison, but the
current pipeline should not depend on them.
