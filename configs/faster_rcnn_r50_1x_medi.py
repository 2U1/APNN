_base_ = [
    '../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../configs/_base_/datasets/medical_data.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

model = dict(
    backbone = dict(
        norm_cfg = dict(type='SyncBN', requires_grad=True),
    ),
    roi_head = dict(
        bbox_head = dict(
            num_classes=4
        )
    )
)

load_from = "/home/workspace/FasterRCNN/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

checkpoint_config = dict(interval=4)

optimizer = dict(type='SGD', lr=0.02/4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(warmup=None)

runner = dict(max_epochs=24)

evaluation = dict(interval=4, metric='bbox')

work_dir = './work_dirs/faster_rcnn_r50_2x_medi'