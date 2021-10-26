# dataset settings
dataset_type = "CocoPointSupDataset"
data_root = "data/coco/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="LoadAnnotationsWithSites",
        with_bbox=True,
        with_mask=True,
        with_site=True,
        kind="own",
    ),
    dict(
        type="ResizeShortestEdge",
        shortest_edge_scale=[640, 672, 704, 736, 768, 800],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(
        type="ToDataContainer",
        fields=[
            dict(key="rand_labels", stack=False),
            dict(key="rand_sites", stack=False),
        ],
    ),
    dict(
        type="Collect",
        keys=[
            "img",
            "gt_bboxes",
            "gt_labels",
            "gt_masks",
            "rand_labels",
            "rand_sites",
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        pipeline=train_pipeline,
        N=10,
        kind="own",
    ),
)
evaluation = dict(metric=["bbox", "segm"])
