_base_ = [
    "./point_sup_r50_fpn.py",
    "./point_coco_instance.py",
    "./schedule_1x.py",
    "./default_runtime.py",
]
relative_imports = dict(imports={".PointSup": "point_sup"}, allow_failed_imports=False)
