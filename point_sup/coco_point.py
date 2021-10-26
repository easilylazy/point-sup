# Author: leezeeyee
# Date: 2021/10/26
import numpy as np

from mmdet.datasets.pipelines import Compose
from mmdet.datasets.builder import DATASETS
from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO


@DATASETS.register_module()
class CocoPointSupDataset(CocoDataset):
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results["img_prefix"] = self.img_prefix
        results["seg_prefix"] = self.seg_prefix
        results["proposal_file"] = self.proposal_file
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["site_fields"] = []
        results["label_fields"] = []

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_sample_sites = []
        gt_sample_labels = []

        gt_point_coords = []
        gt_point_labels = []

        for i, ann in enumerate(ann_info):
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                gt_masks_ann.append(ann.get("segmentation", None))
                # gt_sample_sites.append(ann["sample_sites"])
                # gt_sample_labels.append(ann["sample_labels"])
                gt_point_coords.append(ann["point_coords"])
                gt_point_labels.append(ann["point_labels"])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info["filename"].replace("jpg", "png")

        ann = dict(
            point_labels=gt_point_labels,
            point_coords=gt_point_coords,
            sample_sites=gt_sample_sites,
            sample_labels=gt_sample_labels,
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
        )

        return ann
