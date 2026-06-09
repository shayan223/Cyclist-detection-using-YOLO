# Deprecated Content Archive

This directory contains legacy project material that is no longer part of the
active RT-DETR workflow. Files were moved here to keep the repository root
presentable while preserving older experiments for reference.

## Archive Map

- `legacy_model_code/` - older non-current model families and experiments:
  YOLO26, RF-DETR, RTMDet, ensembles, and ByteTrack.
- `legacy_training_outputs/` - old generated training, validation, PET, and run
  output directories plus legacy root checkpoints.
- `legacy_datasets/` - older dataset variants, EuroCity conversion data,
  Cyclist/Pedestrian datasets, and zipped dataset archives.
- `legacy_media_outputs/` - rendered videos, CSVs, and test media from older
  YOLO, RF-DETR, ensemble, ByteTrack, and early RT-DETR runs.
- `legacy_scripts/` - root-level training and analysis scripts from earlier
  workflows.
- `tracked_cache/` - Python cache files that were previously present in the
  working tree.

## Active Pipeline Note

The active project now lives in the repository root, `RT_DETR/`,
`results_reporting/`, `v5_pdx_cyclist_dataset/`, and `raw_pdx_videos/`.
Anything in this archive should be treated as historical unless it is
intentionally restored and its paths are updated.

Current RT-DETR configs still reference active root input videos such as
`trim5.mp4`, and current reporting depends on
`RT_DETR/RTDETR_AUTO_EVAL/runs/camera_1_trim5/20260608_220235`.
