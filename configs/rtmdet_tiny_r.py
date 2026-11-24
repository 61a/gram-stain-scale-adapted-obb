"""Four-class Rotated RTMDet-tiny configurations used in the paper.

Derived from OpenMMLab MMRotate's Rotated RTMDet-tiny DOTA configuration.
Copyright (c) OpenMMLab. All rights reserved.
Upstream-derived portions are licensed under the Apache License, Version 2.0.
"""

CLASSES = ("G_coccus", "Gp_coccus", "G_rod", "Gp_rod")
PARAMETERS = {"baseline": 4874343, "two_scale": 1403934, "coarse_narrow": 3824007}
STRIDES = {"baseline": (8, 16, 32), "two_scale": (8, 16), "coarse_narrow": (8, 16, 32)}
INPUT_SIZE = 640


def model_config(variant):
    """Build the native MMRotate model configuration without loading weights."""
    strides = list(STRIDES[variant])
    backbone = dict(
        type="mmdet.CSPNeXt", arch="P5", channel_attention=True,
        deepen_factor=0.167, widen_factor=0.375, expand_ratio=0.5,
        act_cfg=dict(type="SiLU"), norm_cfg=dict(type="BN"), init_cfg=None,
    )
    neck_channels = [96, 192, 384]
    stages = [
        [64, 128, 3, True, False],
        [128, 256, 6, True, False],
        [256, 512, 6, True, False],
        [512, 1024, 3, False, True],
    ]
    if variant == "two_scale":
        backbone.update(arch_ovewrite=stages[:3], out_indices=(2, 3))
        neck_channels = [96, 192]
    elif variant == "coarse_narrow":
        stages[-1][1] = 768
        backbone["arch_ovewrite"] = stages
        neck_channels[-1] = 288
    return dict(
        type="mmdet.RTMDet",
        backbone=backbone,
        neck=dict(
            type="mmdet.CSPNeXtPAFPN", in_channels=neck_channels, out_channels=96,
            num_csp_blocks=1, expand_ratio=0.5,
            act_cfg=dict(type="SiLU"), norm_cfg=dict(type="BN"),
        ),
        bbox_head=dict(
            type="RotatedRTMDetSepBNHead", in_channels=96, feat_channels=96,
            stacked_convs=2, num_classes=len(CLASSES), pred_kernel_size=1,
            share_conv=True, exp_on_reg=False, scale_angle=False, with_objectness=False,
            use_hbbox_loss=False, angle_version="le90", loss_angle=None,
            act_cfg=dict(type="SiLU"), norm_cfg=dict(type="BN"),
            anchor_generator=dict(type="mmdet.MlvlPointGenerator", offset=0, strides=strides),
            bbox_coder=dict(type="DistanceAnglePointCoder", angle_version="le90"),
            loss_cls=dict(type="mmdet.QualityFocalLoss", beta=2.0, loss_weight=1.0, use_sigmoid=True),
            loss_bbox=dict(type="RotatedIoULoss", mode="linear", loss_weight=2.0),
        ),
        data_preprocessor=dict(
            type="mmdet.DetDataPreprocessor", batch_augments=None, bgr_to_rgb=False,
            boxtype2tensor=False, mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
        ),
        train_cfg=dict(
            allowed_border=-1, debug=False, pos_weight=-1,
            assigner=dict(
                type="mmdet.DynamicSoftLabelAssigner", topk=13,
                iou_calculator=dict(type="RBboxOverlaps2D"),
            ),
        ),
        test_cfg=dict(
            score_thr=0.01, nms_pre=2000, min_bbox_size=0, max_per_img=300,
            nms=dict(type="nms_rotated", iou_threshold=0.7),
        ),
    )


def train_pipeline():
    return [
        dict(type="mmdet.LoadImageFromFile", backend_args=None),
        dict(type="mmdet.LoadAnnotations", box_type="qbox", with_bbox=True),
        dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
        dict(type="mmdet.Resize", scale=(INPUT_SIZE, INPUT_SIZE), keep_ratio=True),
        dict(type="mmdet.RandomFlip", prob=0.75, direction=["horizontal", "vertical", "diagonal"]),
        dict(type="RandomRotate", prob=0.5, angle_range=180, rect_obj_labels=[]),
        dict(type="mmdet.Pad", size=(INPUT_SIZE, INPUT_SIZE), pad_val=dict(img=(114, 114, 114))),
        dict(type="mmdet.PackDetInputs"),
    ]


def inference_pipeline():
    return [
        dict(type="mmdet.LoadImageFromNDArray"),
        dict(type="mmdet.Resize", scale=(INPUT_SIZE, INPUT_SIZE), keep_ratio=True),
        dict(type="mmdet.PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
    ]


def fine_tune_config(variant, seed, epochs, data_root, manifest, initial, work_dir):
    if seed not in (42, 43, 44) or epochs not in (10, 30):
        raise ValueError("The paper's seeds are 42–44, with fixed 10- or 30-epoch schedules.")
    return dict(
        custom_imports=dict(imports=["rtmdet_runtime"], allow_failed_imports=False),
        default_scope="mmrotate",
        model=model_config(variant),
        work_dir=str(work_dir),
        load_from=str(initial),
        resume=False,
        randomness=dict(seed=seed, deterministic=False),
        log_level="INFO",
        log_processor=dict(type="LogProcessor", window_size=50, by_epoch=True),
        env_cfg=dict(
            cudnn_benchmark=False, dist_cfg=dict(backend="nccl"),
            mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
        ),
        train_dataloader=dict(
            batch_size=8, batch_sampler=None, num_workers=2,
            persistent_workers=True, pin_memory=False,
            sampler=dict(type="DefaultSampler", shuffle=True),
            dataset=dict(
                type="PaperDotaDataset", data_root=str(data_root), manifest_path=str(manifest),
                img_suffix="jpg", metainfo=dict(classes=CLASSES),
                filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline(),
            ),
        ),
        train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=epochs, val_begin=epochs + 1, val_interval=epochs + 1),
        val_cfg=None, val_dataloader=None, val_evaluator=None,
        test_cfg=None, test_dataloader=None, test_evaluator=None,
        optim_wrapper=dict(
            type="FiniteAmpOptimWrapper", dtype="bfloat16",
            optimizer=dict(type="AdamW", lr=5e-5, weight_decay=0.05),
            paramwise_cfg=dict(bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
        ),
        param_scheduler=[
            dict(type="LinearLR", begin=0, end=100, by_epoch=False, start_factor=0.1),
            dict(type="CosineAnnealingLR", begin=0, end=epochs, by_epoch=True, T_max=epochs,
                 eta_min=2.5e-6, convert_to_iter_based=True),
        ],
        default_hooks=dict(
            checkpoint=dict(type="CheckpointHook", interval=epochs, max_keep_ckpts=1, save_best=None, save_last=True),
            logger=dict(type="LoggerHook", interval=100),
            param_scheduler=dict(type="ParamSchedulerHook"),
            sampler_seed=dict(type="DistSamplerSeedHook"),
            timer=dict(type="IterTimerHook"),
            visualization=dict(type="mmdet.DetVisualizationHook"),
        ),
        custom_hooks=[
            dict(type="mmdet.NumClassCheckHook"),
            dict(type="EMAHook", ema_type="mmdet.ExpMomentumEMA", momentum=0.0002,
                 priority=49, update_buffers=True),
            dict(type="FixedStartHook", seed=seed),
        ],
        visualizer=dict(name="visualizer", type="RotLocalVisualizer", vis_backends=[dict(type="LocalVisBackend")]),
    )
