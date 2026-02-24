# DINOv2 ViT-S/14 + multi-scale head fine-tuning on CTEM (head-only).

custom_imports = dict(
    imports=["dinov2.eval.segmentation.models"],
    allow_failed_imports=False,
)

dataset_type = "CustomDataset"
# TODO: update these paths to your local dataset.
data_root = "/home/tipriest/Documents/CVTask/CTEM"
test_root = "/home/tipriest/Documents/CVTask/CTEM/TestSet"

classes = (
    "Bedrock",
    "Soil",
    "Gravel",
    "Stony soil",
    "Rock",
    "Background",
)

palette = [
    [128, 64, 64],  # Bedrock
    [128, 128, 0],  # Soil
    [64, 128, 128],  # Gravel
    [192, 64, 0],  # Stony soil
    [0, 128, 64],  # Rock
    [0, 0, 0],  # Background
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

crop_size = (540, 960)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(960, 540), ratio_range=(1.0, 3.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(960, 540),
        # img_ratios=[1.0, 1.32, 1.73, 2.28, 3.0],
        # flip=True,
        # img_ratios=[1.0, 1.32, 1.73],
        # flip=True,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="JPEGImages",
        ann_dir="SegmentationClass",
        img_suffix=".jpg",
        seg_map_suffix=".png",
        split="ImageSets/train.txt",
        classes=classes,
        palette=palette,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="JPEGImages",
        ann_dir="SegmentationClass",
        img_suffix=".jpg",
        seg_map_suffix=".png",
        split="ImageSets/val.txt",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=test_root,
        img_dir="JPEGImages",
        ann_dir="SegmentationClass",
        img_suffix=".jpg",
        seg_map_suffix=".png",
        split="ImageSets/test.txt",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline,
    ),
)

norm_cfg = dict(type="SyncBN", requires_grad=True)

model = dict(
    type="EncoderDecoder",
    pretrained=None,
    backbone=dict(type="DinoVisionTransformer", out_indices=[8, 9, 10, 11]),
    decode_head=dict(
        type="BNHead",
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        input_transform="resize_concat",
        channels=4096,  # 1024*4
        dropout_ratio=0.0,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(270, 480)),
)

optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))
optimizer_config = dict(type="OptimizerHook", grad_clip=None)

lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

runner = dict(type="IterBasedRunner", max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)

evaluation = dict(interval=2000, metric="mIoU", pre_eval=True)

log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook", by_epoch=False)])

log_level = "INFO"

gpu_ids = range(1)

seed = 0

device = "cuda"

cudnn_benchmark = True
find_unused_parameters = True

work_dir = "./work_dirs/dinov2_vitl14_ctem_ms_540_960"

resume_from = None
auto_resume = True

load_from = None

workflow = [("train", 1)]
