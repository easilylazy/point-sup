# Author: leezeeyee
# Date: 2021/10/26
import torch
import torch.nn.functional as F
from mmcv.ops import point_sample
from mmcv.runner import force_fp32

from mmdet.core import bbox2roi
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.builder import HEADS


from mmdet.core import (
    bbox2roi,
)


@HEADS.register_module()
class PointSupRoIHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def forward_train(
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        my_sites_imgs=None,
        points_labels=None,
        **kwargs
    ):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x, sampling_results, gt_bboxes, gt_labels, img_metas
            )
            losses.update(bbox_results["loss_bbox"])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(
                x,
                sampling_results,
                bbox_results["bbox_feats"],
                sites_img=my_sites_imgs,
                points_labels=points_labels,
                **kwargs
            )
            losses.update(mask_results["loss_mask"])

        return losses

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[: self.mask_roi_extractor.num_inputs], rois
            )
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def _mask_forward_train(
        self, x, sampling_results, bbox_feats, points_labels, sites_img, **kwargs
    ):
        # sites already sampled
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0], device=device, dtype=torch.uint8
                    )
                )
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0], device=device, dtype=torch.uint8
                    )
                )
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats
            )
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        assert points_labels[0].shape[1] == 5
        assert sites_img[0].shape[2] == 5
        # res=sampling_results[1]
        # res.pos_gt_labels
        try:
            pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results])
            sites = torch.cat(
                [
                    site_img[res.pos_assigned_gt_inds, :, :]
                    for site_img, res in zip(sites_img, sampling_results)
                ]
            )
            mask_targets = torch.cat(
                [
                    labels[
                        res.pos_assigned_gt_inds,
                    ]
                    for labels, res in zip(points_labels, sampling_results)
                ]
            )
            new_sites = get_point_coords_wrt_box(pos_bboxes, sites)
            point_ignores = (
                (new_sites[:, :, 0] < 0)
                | (new_sites[:, :, 0] > 1)
                | (new_sites[:, :, 1] < 0)
                | (new_sites[:, :, 1] > 1)
            )
            mask_targets[point_ignores] = 2

            point_preds = point_sample(
                mask_results["mask_pred"],
                new_sites,
                align_corners=False,
            )
            loss_mask = self.mask_head.loss(
                point_preds, mask_targets.to(torch.float32).squeeze(1), pos_labels
            )
            mask_results.update(loss_mask=loss_mask)  # , mask_targets=mask_targets)
            return mask_results
        except Exception as e:
            print("error in _mask_forward_train: ", e)


def get_point_coords_wrt_box(boxes_coords, point_coords):
    """
    Convert image-level absolute coordinates to box-normalized [0, 1] x [0, 1] point cooordinates.
    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    Returns:
        point_coords_wrt_box (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.
    """
    with torch.no_grad():
        point_coords_wrt_box = point_coords.clone().permute(0, 2, 1)
        point_coords_wrt_box[:, :, 0] -= boxes_coords[:, None, 0]
        point_coords_wrt_box[:, :, 1] -= boxes_coords[:, None, 1]
        point_coords_wrt_box[:, :, 0] = point_coords_wrt_box[:, :, 0] / (
            boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
        )
        point_coords_wrt_box[:, :, 1] = point_coords_wrt_box[:, :, 1] / (
            boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
        )
    return point_coords_wrt_box


@HEADS.register_module()
class PointSupHead(FCNMaskHead):
    @force_fp32(apply_to=("mask_pred",))
    def loss(self, mask_pred, mask_targets, labels):
        """
        Example:
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> # There are lots of variations depending on the configuration
            >>> self = MyPointHead(num_classes=C, num_convs=1)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> sf = self.scale_factor
            >>> labels = torch.randint(0, C, size=(N,))
            >>> # With the default properties the mask targets should indicate
            >>> # a (potentially soft) single-class label
            >>> mask_targets = torch.rand(N, H * sf, W * sf)
            >>> loss = self.loss(mask_pred, mask_targets, labels)
            >>> print('loss = {!r}'.format(loss))
        """
        loss = dict()
        total_num_masks = mask_pred.size(0)
        if total_num_masks == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(
                    mask_pred, mask_targets, torch.zeros_like(labels)
                )
            else:
                indices = torch.arange(total_num_masks)
                point_ignores = mask_targets == 2
                mask_logits = mask_pred[indices, labels]
                loss_mask = F.binary_cross_entropy_with_logits(
                    mask_logits,
                    mask_targets.to(dtype=torch.float32),
                    reduction="mean",
                    weight=~point_ignores,
                )

        loss["loss_mask"] = loss_mask
        return loss
