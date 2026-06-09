"""Validate a fine-tuned RTMDet checkpoint on the cyclist/pedestrian COCO dataset.

Install:
    pip install mmdet mmengine mmcv

Usage:
    python validate_rtmdet.py --checkpoint runs/rtmdet_finetune_l_imgsz640_mos1_mix1_fliplr0.5/best_coco_bbox_mAP_epoch_*.pth
    python validate_rtmdet.py --checkpoint best.pth --split test
    python validate_rtmdet.py --checkpoint best.pth --model-size m --data ../RF_DETR/coco_v5_pdx_cyclist_dataset
"""

from __future__ import annotations

import argparse
import os

import torch
from mmengine.config import Config
from mmengine.runner import Runner

try:
    from mmdet.utils import register_all_modules
    register_all_modules()
except ImportError as exc:
    raise ImportError(
        "mmdet is not installed. Install with: pip install mmdet"
    ) from exc

from fine_tune_rtmdet import (
    DATA_ROOT,
    CLASS_NAMES,
    NUM_CLASSES,
    IMGSZ,
    DEVICE,
    MODEL_SIZES,
    _build_test_pipeline,
)

DEFAULT_CHECKPOINT = 'runs/rtmdet_finetune_l_imgsz640_mos1_mix1_fliplr0.5/best_coco_bbox_mAP.pth'


def validate(
    checkpoint: str,
    data_root: str = DATA_ROOT,
    model_size: str = 'l',
    imgsz: int = IMGSZ,
    device: str = DEVICE,
    split: str = 'val',
    score_thr: float = 0.001,
    iou_threshold: float = 0.65,
    work_dir: str = 'runs/val',
) -> None:
    """Run COCO evaluation on a fine-tuned RTMDet checkpoint."""
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    split_dir_map = {'val': 'valid', 'test': 'test', 'train': 'train'}
    split_dir = split_dir_map.get(split, split)
    ann_file = os.path.join(data_root, split_dir, '_annotations.coco.json')
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    deepen, widen = MODEL_SIZES[model_size]
    ch256  = int(256  * widen)
    ch512  = int(512  * widen)
    ch1024 = int(1024 * widen)
    neck_out = ch256

    test_pipeline = _build_test_pipeline(imgsz)

    cfg_dict = dict(
        model=dict(
            type='RTMDet',
            data_preprocessor=dict(
                type='DetDataPreprocessor',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                bgr_to_rgb=False,
                batch_augments=None,
            ),
            backbone=dict(
                type='CSPNeXt',
                arch='P5',
                expand_ratio=0.5,
                deepen_factor=deepen,
                widen_factor=widen,
                channel_attention=True,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='SiLU', inplace=True),
            ),
            neck=dict(
                type='CSPNeXtPAFPN',
                in_channels=[ch256, ch512, ch1024],
                out_channels=neck_out,
                num_csp_blocks=max(1, round(3 * deepen)),
                expand_ratio=0.5,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='SiLU', inplace=True),
            ),
            bbox_head=dict(
                type='RTMDetSepBNHead',
                num_classes=NUM_CLASSES,
                in_channels=neck_out,
                stacked_convs=2,
                feat_channels=neck_out,
                anchor_generator=dict(
                    type='MlvlPointGenerator',
                    offset=0,
                    strides=[8, 16, 32],
                ),
                bbox_coder=dict(type='DistancePointBBoxCoder'),
                loss_cls=dict(
                    type='QualityFocalLoss',
                    use_sigmoid=True,
                    beta=2.0,
                    loss_weight=1.0,
                ),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                with_objectness=False,
                exp_on_reg=False,
                share_conv=True,
                pred_kernel_size=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='SiLU', inplace=True),
            ),
            train_cfg=None,
            test_cfg=dict(
                nms_pre=30000,
                min_bbox_size=0,
                score_thr=score_thr,
                nms=dict(type='nms', iou_threshold=iou_threshold),
                max_per_img=300,
            ),
        ),
        test_dataloader=dict(
            batch_size=1,
            num_workers=2,
            persistent_workers=True,
            drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(
                type='CocoDataset',
                data_root=data_root,
                metainfo=dict(classes=CLASS_NAMES),
                ann_file=f'{split_dir}/_annotations.coco.json',
                data_prefix=dict(img=f'{split_dir}/'),
                test_mode=True,
                pipeline=test_pipeline,
            ),
        ),
        test_evaluator=dict(
            type='CocoMetric',
            ann_file=ann_file,
            metric='bbox',
            format_only=False,
        ),
        test_cfg=dict(type='TestLoop'),
        default_scope='mmdet',
        env_cfg=dict(
            cudnn_benchmark=False,
            mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
            dist_cfg=dict(backend='nccl'),
        ),
        vis_backends=[dict(type='LocalVisBackend')],
        visualizer=dict(
            type='DetLocalVisualizer',
            vis_backends=[dict(type='LocalVisBackend')],
            name='visualizer',
        ),
        log_processor=dict(type='LogProcessor', window_size=50, by_epoch=True),
        log_level='INFO',
        load_from=checkpoint,
        work_dir=work_dir,
    )

    print(f"Loading checkpoint: {checkpoint}")
    print(f"Dataset split:      {split} ({split_dir})")
    print(f"Model size:         rtmdet-{model_size}")
    print(f"imgsz: {imgsz}  |  score_thr: {score_thr}  |  iou_threshold: {iou_threshold}")

    cfg = Config(cfg_dict)
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    print("\n--- Validation Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a fine-tuned RTMDet checkpoint.")
    parser.add_argument(
        '--checkpoint', '-c', type=str, default=DEFAULT_CHECKPOINT,
        help='Path to the RTMDet .pth checkpoint.',
    )
    parser.add_argument(
        '--data', type=str, default=DATA_ROOT,
        help='Root directory of the COCO-format dataset.',
    )
    parser.add_argument(
        '--model-size', choices=list(MODEL_SIZES), default='l',
        help='RTMDet model size. Must match the checkpoint. Default: l.',
    )
    parser.add_argument('--imgsz', type=int, default=IMGSZ)
    parser.add_argument(
        '--device', type=str, default=DEVICE,
        help='Device: "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        '--split', type=str, default='val',
        choices=['val', 'test', 'train'],
        help='Dataset split to evaluate.',
    )
    parser.add_argument(
        '--score-thr', type=float, default=0.001,
        help='Score threshold for evaluation (lower = measure full recall curve).',
    )
    parser.add_argument(
        '--iou', type=float, default=0.65,
        help='IoU threshold for NMS during evaluation.',
    )
    parser.add_argument('--work-dir', type=str, default='runs/val')
    args = parser.parse_args()

    validate(
        checkpoint=args.checkpoint,
        data_root=args.data,
        model_size=args.model_size,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        score_thr=args.score_thr,
        iou_threshold=args.iou,
        work_dir=args.work_dir,
    )


if __name__ == '__main__':
    main()
