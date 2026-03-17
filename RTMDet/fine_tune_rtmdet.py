"""Fine-tune RTMDet (mmdetection v3.x) on the cyclist/pedestrian COCO dataset.

Install:
    pip install mmdet mmengine mmcv openmim

Usage:
    python fine_tune_rtmdet.py
    python fine_tune_rtmdet.py --model-size m --epochs 100 --lr 0.004
    python fine_tune_rtmdet.py --checkpoint /path/to/rtmdet.pth --model-size l
    python fine_tune_rtmdet.py --baseline-recipe
    python fine_tune_rtmdet.py --model-size tiny --batch 16 --epochs 50

Dataset format: COCO JSON  (ann_file = <split>/_annotations.coco.json)
"""

from __future__ import annotations

import argparse
import os

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.utils import register_all_modules
register_all_modules()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_ROOT = '../RF_DETR/coco_v5_pdx_cyclist_dataset'
CLASS_NAMES = ('cyclist', 'pedestrian')
NUM_CLASSES = len(CLASS_NAMES)
EPOCHS = 100
BATCH = 8
LR = 0.004
IMGSZ = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_RUN_NAME = 'rtmdet_finetune'
WORK_DIR_ROOT = 'runs'

# (deepen_factor, widen_factor) per model size, matching official RTMDet configs
MODEL_SIZES: dict[str, tuple[float, float]] = {
    'tiny': (0.167, 0.375),
    's':    (0.33,  0.5),
    'm':    (0.67,  0.75),
    'l':    (1.0,   1.0),
    'x':    (1.33,  1.25),
}

PRETRAINED_CHECKPOINTS: dict[str, str] = {
    'tiny': 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
    's':    'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387f244c.pth',
    'm':    'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth',
    'l':    'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth',
    'x':    'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220719_113843-91adff55.pth',
}

# ---------------------------------------------------------------------------
# Augmentation recipes
# ---------------------------------------------------------------------------
DEFAULT_RECIPE: dict = {
    'mosaic':        True,
    'mixup':         True,
    'close_mosaic':  10,     # disable mosaic/mixup in last N epochs
    'hsv':           True,
    'fliplr':        0.5,
    'flipud':        0.0,
    'scale':         0.5,    # RandomResize ratio offset from 1.0
    'translate':     0.1,
}

BASELINE_RECIPE: dict = {
    'mosaic':        False,
    'mixup':         False,
    'close_mosaic':  0,
    'hsv':           False,
    'fliplr':        0.0,
    'flipud':        0.0,
    'scale':         0.0,
    'translate':     0.0,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _channel(base: int, widen: float) -> int:
    return int(base * widen)


def _num_csp_blocks(deepen: float) -> int:
    return max(1, round(3 * deepen))


def _build_train_pipeline(imgsz: int, recipe: dict, mosaic_on: bool) -> list:
    """Augmentation pipeline for the main training phase."""
    if not mosaic_on:
        return _build_close_mosaic_pipeline(imgsz, recipe)

    pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='CachedMosaic',
            img_scale=(imgsz, imgsz),
            pad_val=114.0,
            max_cached_images=20,
            random_pop=False,
        ),
        dict(
            type='RandomResize',
            scale=(2 * imgsz, 2 * imgsz),
            ratio_range=(1.0 - recipe['scale'], 1.0 + recipe['scale']),
            keep_ratio=True,
        ),
        dict(type='RandomCrop', crop_size=(imgsz, imgsz)),
    ]
    if recipe['hsv']:
        pipeline.append(dict(type='YOLOXHSVRandomAug'))
    if recipe['fliplr'] > 0:
        pipeline.append(dict(type='RandomFlip', prob=recipe['fliplr'], direction='horizontal'))
    if recipe['flipud'] > 0:
        pipeline.append(dict(type='RandomFlip', prob=recipe['flipud'], direction='vertical'))
    pipeline.append(dict(type='Pad', size=(imgsz, imgsz), pad_val=dict(img=(114, 114, 114))))
    if recipe['mixup']:
        pipeline.append(dict(
            type='CachedMixUp',
            img_scale=(imgsz, imgsz),
            ratio_range=(1.0, 1.0),
            max_cached_images=10,
            random_pop=False,
            pad_val=(114, 114, 114),
            prob=0.5,
        ))
    pipeline.append(dict(type='PackDetInputs'))
    return pipeline


def _build_close_mosaic_pipeline(imgsz: int, recipe: dict) -> list:
    """Simpler pipeline used in the final close_mosaic epochs (no mosaic/mixup)."""
    pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='RandomResize',
            scale=(imgsz, imgsz),
            ratio_range=(1.0 - recipe['scale'], 1.0 + recipe['scale']),
            keep_ratio=True,
        ),
        dict(type='RandomCrop', crop_size=(imgsz, imgsz)),
    ]
    if recipe['hsv']:
        pipeline.append(dict(type='YOLOXHSVRandomAug'))
    if recipe['fliplr'] > 0:
        pipeline.append(dict(type='RandomFlip', prob=recipe['fliplr'], direction='horizontal'))
    if recipe['flipud'] > 0:
        pipeline.append(dict(type='RandomFlip', prob=recipe['flipud'], direction='vertical'))
    pipeline.append(dict(type='Pad', size=(imgsz, imgsz), pad_val=dict(img=(114, 114, 114))))
    pipeline.append(dict(type='PackDetInputs'))
    return pipeline


def _build_test_pipeline(imgsz: int) -> list:
    return [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='Resize', scale=(imgsz, imgsz), keep_ratio=True),
        dict(type='Pad', size=(imgsz, imgsz), pad_val=dict(img=(114, 114, 114))),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
        ),
    ]


def build_rtmdet_config(
    model_size: str = 'l',
    num_classes: int = NUM_CLASSES,
    class_names: tuple = CLASS_NAMES,
    data_root: str = DATA_ROOT,
    epochs: int = EPOCHS,
    batch: int = BATCH,
    lr: float = LR,
    imgsz: int = IMGSZ,
    recipe: dict | None = None,
    work_dir: str = 'runs/rtmdet_finetune',
    checkpoint: str | None = None,
    device: str = DEVICE,
) -> Config:
    """Build a complete mmdet Config dict for RTMDet fine-tuning."""
    recipe = dict(DEFAULT_RECIPE if recipe is None else recipe)
    deepen, widen = MODEL_SIZES[model_size]

    # Channel dimensions that scale with widen_factor
    ch256  = _channel(256,  widen)
    ch512  = _channel(512,  widen)
    ch1024 = _channel(1024, widen)
    neck_out = ch256

    # Pretrained init config
    if checkpoint and os.path.isfile(checkpoint):
        init_cfg = dict(type='Pretrained', checkpoint=checkpoint)
    else:
        init_cfg = dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=PRETRAINED_CHECKPOINTS[model_size],
        )

    train_pipeline = _build_train_pipeline(imgsz, recipe, mosaic_on=recipe['mosaic'])
    close_pipeline = _build_close_mosaic_pipeline(imgsz, recipe)
    test_pipeline  = _build_test_pipeline(imgsz)

    # Hooks: checkpoint, logger, pipeline switch
    hooks = [
        dict(type='CheckpointHook', interval=10, max_keep_ckpts=3, save_best='coco/bbox_mAP'),
        dict(type='LoggerHook', interval=50),
        dict(type='ParamSchedulerHook'),
        dict(type='IterTimerHook'),
        dict(type='NumClassCheckHook'),
    ]
    if recipe['mosaic'] and recipe['close_mosaic'] > 0:
        switch_epoch = max(1, epochs - recipe['close_mosaic'])
        hooks.append(dict(
            type='PipelineSwitchHook',
            switch_epoch=switch_epoch,
            switch_pipeline=close_pipeline,
        ))

    cfg_dict = dict(
        # ------------------------------------------------------------------ model
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
                init_cfg=init_cfg,
            ),
            neck=dict(
                type='CSPNeXtPAFPN',
                in_channels=[ch256, ch512, ch1024],
                out_channels=neck_out,
                num_csp_blocks=_num_csp_blocks(deepen),
                expand_ratio=0.5,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='SiLU', inplace=True),
            ),
            bbox_head=dict(
                type='RTMDetSepBNHead',
                num_classes=num_classes,
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
            train_cfg=dict(
                assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            test_cfg=dict(
                nms_pre=30000,
                min_bbox_size=0,
                score_thr=0.001,
                nms=dict(type='nms', iou_threshold=0.65),
                max_per_img=300,
            ),
        ),

        # ------------------------------------------------------------------ data
        train_dataloader=dict(
            batch_size=batch,
            num_workers=4,
            persistent_workers=True,
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_sampler=dict(type='AspectRatioBatchSampler'),
            dataset=dict(
                type='CocoDataset',
                data_root=data_root,
                metainfo=dict(classes=class_names),
                ann_file='train/_annotations.coco.json',
                data_prefix=dict(img='train/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
            ),
        ),
        val_dataloader=dict(
            batch_size=1,
            num_workers=2,
            persistent_workers=True,
            drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(
                type='CocoDataset',
                data_root=data_root,
                metainfo=dict(classes=class_names),
                ann_file='valid/_annotations.coco.json',
                data_prefix=dict(img='valid/'),
                test_mode=True,
                pipeline=test_pipeline,
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
                metainfo=dict(classes=class_names),
                ann_file='test/_annotations.coco.json',
                data_prefix=dict(img='test/'),
                test_mode=True,
                pipeline=test_pipeline,
            ),
        ),

        # ------------------------------------------------------------------ evaluator
        val_evaluator=dict(
            type='CocoMetric',
            ann_file=os.path.join(data_root, 'valid/_annotations.coco.json'),
            metric='bbox',
            format_only=False,
        ),
        test_evaluator=dict(
            type='CocoMetric',
            ann_file=os.path.join(data_root, 'test/_annotations.coco.json'),
            metric='bbox',
            format_only=False,
        ),

        # ------------------------------------------------------------------ training schedule
        train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=5),
        val_cfg=dict(type='ValLoop'),
        test_cfg=dict(type='TestLoop'),

        # ------------------------------------------------------------------ optimizer
        optim_wrapper=dict(
            type='OptimWrapper',
            optimizer=dict(type='AdamW', lr=lr, weight_decay=0.05),
            paramwise_cfg=dict(
                norm_decay_mult=0,
                bias_decay_mult=0,
                bypass_duplicate=True,
            ),
            clip_grad=dict(max_norm=35, norm_type=2),
        ),

        # ------------------------------------------------------------------ LR schedule
        param_scheduler=[
            dict(
                type='LinearLR',
                start_factor=1.0e-5,
                by_epoch=False,
                begin=0,
                end=1000,
            ),
            dict(
                type='CosineAnnealingLR',
                eta_min=lr * 0.05,
                begin=epochs // 2,
                end=epochs,
                T_max=epochs // 2,
                by_epoch=True,
                convert_to_iter_based=True,
            ),
        ],

        # ------------------------------------------------------------------ hooks & logging
        default_hooks=dict(
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook', interval=50),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(
                type='CheckpointHook',
                interval=10,
                max_keep_ckpts=3,
                save_best='coco/bbox_mAP',
            ),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            visualization=dict(type='DetVisualizationHook'),
        ),
        custom_hooks=hooks[4:],  # PipelineSwitchHook if present

        # ------------------------------------------------------------------ runner / env
        default_scope='mmdet',
        env_cfg=dict(
            cudnn_benchmark=False,
            mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
            dist_cfg=dict(backend='nccl'),
        ),
        vis_backends=[dict(type='LocalVisBackend')],
        visualizer=dict(type='DetLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')], name='visualizer'),
        log_processor=dict(type='LogProcessor', window_size=50, by_epoch=True),
        log_level='INFO',
        load_from=None,
        resume=False,
        work_dir=work_dir,
    )

    # If a pre-trained checkpoint file is given, load it directly instead of
    # initialising backbone weights from the COCO pretrain.
    if checkpoint and os.path.isfile(checkpoint):
        cfg_dict['load_from'] = checkpoint

    return Config(cfg_dict)


def build_run_name(base_name: str, model_size: str, imgsz: int, recipe: dict) -> str:
    return (
        f"{base_name}_{model_size}"
        f"_imgsz{imgsz}"
        f"_mos{int(recipe['mosaic'])}"
        f"_mix{int(recipe['mixup'])}"
        f"_fliplr{recipe['fliplr']}"
    )


def fine_tune_rtmdet(
    data_root: str = DATA_ROOT,
    model_size: str = 'l',
    checkpoint: str | None = None,
    epochs: int = EPOCHS,
    batch: int = BATCH,
    lr: float = LR,
    imgsz: int = IMGSZ,
    device: str = DEVICE,
    recipe: dict | None = None,
    work_dir_root: str = WORK_DIR_ROOT,
    run_name: str = BASE_RUN_NAME,
) -> None:
    """Fine-tune an RTMDet model on the cyclist/pedestrian dataset."""
    os.environ['WANDB_DISABLED'] = 'true'
    recipe = dict(DEFAULT_RECIPE if recipe is None else recipe)

    effective_name = build_run_name(run_name, model_size, imgsz, recipe)
    work_dir = os.path.join(work_dir_root, effective_name)

    print(f"Device:     {device}")
    print(f"Model size: rtmdet-{model_size}")
    print(f"Run name:   {effective_name}")
    print(f"Work dir:   {work_dir}")
    print(f"Epochs:     {epochs}  |  Batch: {batch}  |  LR: {lr}  |  imgsz: {imgsz}")
    print(f"Recipe:     {recipe}")

    cfg = build_rtmdet_config(
        model_size=model_size,
        data_root=data_root,
        epochs=epochs,
        batch=batch,
        lr=lr,
        imgsz=imgsz,
        recipe=recipe,
        work_dir=work_dir,
        checkpoint=checkpoint,
        device=device,
    )

    runner = Runner.from_cfg(cfg)
    runner.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RTMDet for cyclist/pedestrian detection.")
    parser.add_argument(
        '--data', type=str, default=DATA_ROOT,
        help='Root directory of the COCO-format dataset '
             '(expects <data>/{train,valid,test}/_annotations.coco.json).',
    )
    parser.add_argument(
        '--model-size', choices=list(MODEL_SIZES), default='l',
        help='RTMDet model size. Default: l.',
    )
    parser.add_argument(
        '--checkpoint', type=str, default='',
        help='Path to an existing checkpoint for continued fine-tuning. '
             'If omitted, loads COCO pretrained weights.',
    )
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--imgsz', type=int, default=IMGSZ)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument(
        '--device', type=str, default=DEVICE,
        help='Training device, e.g. "cuda", "cuda:0", "cpu".',
    )
    parser.add_argument('--work-dir-root', type=str, default=WORK_DIR_ROOT)
    parser.add_argument('--name', type=str, default=BASE_RUN_NAME)
    parser.add_argument(
        '--baseline-recipe', action='store_true',
        help='Disable all augmentation (baseline recipe).',
    )
    args = parser.parse_args()

    recipe = BASELINE_RECIPE if args.baseline_recipe else DEFAULT_RECIPE
    fine_tune_rtmdet(
        data_root=args.data,
        model_size=args.model_size,
        checkpoint=args.checkpoint or None,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        imgsz=args.imgsz,
        device=args.device,
        recipe=recipe,
        work_dir_root=args.work_dir_root,
        run_name=args.name,
    )


if __name__ == '__main__':
    main()
