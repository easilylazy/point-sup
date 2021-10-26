# Author: leezeeyee
# Date: 2021/10/26
import torch
import numpy as np

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadAnnotationsWithSites:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        with_site (bool): Whether to parse and load the mask random point sites.
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
        self,
        with_bbox=True,
        with_label=True,
        with_mask=False,
        with_seg=False,
        with_site=False,
        poly2mask=True,
        file_client_args=dict(backend="disk"),
        Point_N=10,
    ):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_site = with_site
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.Point_N = Point_N
        self.Rand_N = Point_N // 2

    def _load_sites(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        ann_info = results["ann_info"]

        rand_points_labels = []
        rand_points_sites = []
        for coords, label in zip(ann_info["point_coords"], ann_info["point_labels"]):
            rand_index = np.random.choice(
                self.Point_N, self.Rand_N, replace=False
            ).astype(int)
            site = np.array(coords).transpose(1, 0)
            label = np.array(label)
            rand_site = site[:, rand_index]
            rand_label = label[:, None][rand_index].flatten()
            rand_points_sites.append(rand_site)
            rand_points_labels.append(rand_label)
            assert rand_site.shape == (2, self.Rand_N)

        results["rand_sites"] = torch.Tensor(rand_points_sites)
        results["rand_labels"] = torch.Tensor(rand_points_labels)

        results["site_fields"].append("rand_sites")
        results["label_fields"].append("rand_labels")

        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_site:
            results = self._load_sites(results)
        return results
