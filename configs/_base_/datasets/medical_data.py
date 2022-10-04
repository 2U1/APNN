# dataset settings
dataset_type = 'CocoDataset'
data_root = "/home/workspace/dataset"

test_anno = "test_annotations.json" # 출력은 "test_partial_annotations.json"
valid_anno = "valid_annotations.json" # 출력은 "valid_partial_annotations.json"
train_anno = "train_annotations.json" # 출력은 "train_partial_annotations.json"

test_img = 'test_img' # 출력은 "test_100000"
valid_img =  "valid_img" # 출력은 "valid_100000"
train_img = "train_img" # 출력은 "train_100000"

data_classes = ('01_ulcer', '02_mass', '04_lymph', '05_bleeding')

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/' + train_anno,
        img_prefix=data_root + '/' + train_img,
        classes = data_classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/' + valid_anno,
        img_prefix=data_root + '/' + valid_img,
        classes = data_classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/' + test_anno,
        img_prefix=data_root + '/' + test_img,
        classes = data_classes,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')