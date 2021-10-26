# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch
import numpy as np
import mmcv
from mmcv.image import tensor2imgs
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt


EPS = 1e-2


def draw_point(
    class_names,
    data_loader,
    max_img=10,
    show=False,
    out_dir=None,
    with_origin=False,
    with_mask=False,
    with_point=True,
    with_point_mask=True,
):
    prog_bar = mmcv.ProgressBar(max_img)
    for i, data in enumerate(data_loader):
        # print('sgljsf')
        prog_bar.update()
        if i > max_img:
            break
        batch_size = 1
        if show or out_dir:
            if batch_size == 1 and isinstance(data["img"], torch.Tensor):
                img_tensor = data["img"]
            else:
                img_tensor = data["img"].data[0]
            img_metas = data["img_metas"].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]["img_norm_cfg"])
            assert len(imgs) == len(img_metas)

            gt_sample_sites = data["rand_sites"].data[0]
            gt_sample_labels = data["rand_labels"].data[0]
            gt_bboxes = data["gt_bboxes"].data[0]
            gt_masks = data["gt_masks"].data[0]
            gt_labels = data["gt_labels"].data[0]

            for i, (
                img,
                img_meta,
                bboxes,
                labels,
                masks,
                sample_sites,
                sample_labels,
            ) in enumerate(
                zip(
                    imgs,
                    img_metas,
                    gt_bboxes,
                    gt_labels,
                    gt_masks,
                    gt_sample_sites,
                    gt_sample_labels,
                )
            ):
                if with_origin:
                    out_file = osp.join(out_dir, img_meta["ori_filename"])
                    disp_point_pic(
                        img,
                        bboxes.detach().numpy(),
                        labels.detach().numpy(),
                        class_names,
                        show=show,
                        out_file=out_file,
                    )
                if with_mask:
                    out_file = osp.join(out_dir, "mask_" + img_meta["ori_filename"])
                    disp_point_pic(
                        img,
                        bboxes.detach().numpy(),
                        labels.detach().numpy(),
                        class_names,
                        segms=masks.to_ndarray(),
                        show=show,
                        out_file=out_file,
                    )
                if with_point:
                    out_file = osp.join(out_dir, "point_" + img_meta["ori_filename"])
                    disp_point_pic(
                        img,
                        bboxes.detach().numpy(),
                        labels.detach().numpy(),
                        class_names,
                        sample_sites=sample_sites.detach().numpy().astype(np.int32),
                        sample_labels=sample_labels.detach().numpy(),
                        show=show,
                        out_file=out_file,
                    )
                if with_point_mask:
                    out_file = osp.join(
                        out_dir, "point_mask_" + img_meta["ori_filename"]
                    )
                    disp_point_pic(
                        img,
                        bboxes.detach().numpy(),
                        labels.detach().numpy(),
                        class_names,
                        segms=masks.to_ndarray(),
                        sample_sites=sample_sites.detach().numpy().astype(np.int32),
                        sample_labels=sample_labels.detach().numpy(),
                        show=show,
                        out_file=out_file,
                    )


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def disp_point_pic(
    img,
    bboxes,
    labels,
    class_names,
    segms=None,
    sample_sites=None,
    sample_labels=None,
    bbox_color=(72, 101, 241),
    text_color=(72, 101, 241),
    mask_color=None,
    thickness=2,
    font_size=13,
    win_name="",
    show=False,
    wait_time=0,
    out_file=None,
):
    mask_colors = []
    if segms is not None or sample_sites is not None:
        if labels.shape[0] > 0:
            if mask_color is None:
                # Get random state before set seed, and restore random state later.
                # Prevent loss of randomness.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                # random color
                np.random.seed(42)
                mask_colors = [
                    np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    for _ in range(max(labels) + 1)
                ]
                np.random.set_state(state)
            else:
                # specify  color
                mask_colors = [
                    np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
                ] * (max(labels) + 1)
    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)
    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis("off")
    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):

        bbox_int = bbox.astype(np.int32)
        poly = [
            [bbox_int[0], bbox_int[1]],
            [bbox_int[0], bbox_int[3]],
            [bbox_int[2], bbox_int[3]],
            [bbox_int[2], bbox_int[1]],
        ]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[label] if class_names is not None else f"class {label}"
        if len(bbox) > 4:
            label_text += f"|{bbox[-1]:.02f}"
        ax.text(
            bbox_int[0],
            bbox_int[1],
            f"{label_text}",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            color=text_color,
            fontsize=font_size,
            verticalalignment="top",
            horizontalalignment="left",
        )
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
        if sample_sites is not None:
            point_color = mask_colors[labels[i]].reshape(3)
            sample_site = sample_sites[i]
            sample_label = sample_labels[i]
            sample_site_fg = sample_site[:, sample_label == 1]
            sample_site_bg = sample_site[:, sample_label == 0]
            sample_site_out = sample_site[:, sample_label == 2]
            ax.plot(
                sample_site_fg[0],
                sample_site_fg[1],
                "o",
                color=color_val_matplotlib(point_color),
            )
            ax.plot(
                sample_site_bg[0],
                sample_site_bg[1],
                "x",
                color=color_val_matplotlib(point_color),
            )
            ax.plot(
                sample_site_out[0],
                sample_site_out[1],
                "*",
                color=color_val_matplotlib(point_color),
            )
    plt.imshow(img)
    p = PatchCollection(
        polygons, facecolor="none", edgecolors=color, linewidths=thickness
    )
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype="uint8")
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype("uint8")
    img = mmcv.rgb2bgr(img)
    # plt.imshow(img)
    mmcv.imwrite(img, out_file)

    plt.close()
