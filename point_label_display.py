# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import torch
from mmcv import Config, DictAction

from mmdet.datasets import build_dataloader, build_dataset

from utils.draw_point import draw_point


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet display the point label")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where painted images will be saved"
    )

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )

    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    assert args.show or args.show_dir, (
        "Please specify at least one operation (show the "
        ' results of point sample labels) with the argument  "--show" or "--show-dir"'
    )

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get("relative_imports", None):
        from utils.relative_import import import_modules_with_relative

        import_modules_with_relative(**cfg["relative_imports"])
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    draw_point(
        dataset.CLASSES, data_loader, max_img=10, show=args.show, out_dir=args.show_dir
    )


if __name__ == "__main__":
    main()
