# Author: leezeeyee
# Date: 2021/10/26
import mmcv
import numpy as np

from mmdet.datasets.pipelines import Resize, RandomFlip
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class ResizeWithSites(Resize):
    def _resize_sites(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""

        for key in results.get("site_fields", []):
            assert results[key].shape[-2] == 2

            sites = results[key] * results["scale_factor"][:2][:, None]

            results[key] = sites


@PIPELINES.register_module()
class ResizeShortestEdge(ResizeWithSites):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give shortest_edge_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `shortest_edge_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        shortest_edge_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(
        self,
        shortest_edge_scale=None,
        multiscale_mode="value",
        ratio_range=None,
        keep_ratio=True,
        bbox_clip_border=True,
        backend="cv2",
        override=False,
        max_size=1333,
    ):
        if shortest_edge_scale is None:
            self.shortest_edge_scale = None
        else:
            if isinstance(shortest_edge_scale, list):
                self.shortest_edge_scale = shortest_edge_scale
            else:
                self.shortest_edge_scale = [shortest_edge_scale]
            assert type(self.shortest_edge_scale) is list

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.shortest_edge_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]
        self.max_size = max_size
        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(shortest_edge_scales):
        """Randomly select an shortest_edge_scale from given candidates.

        Args:
            shortest_edge_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(shortest_edge_scale, scale_dix)``, \
                where ``shortest_edge_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        scale_idx = np.random.randint(len(shortest_edge_scales))
        shortest_edge_scale = shortest_edge_scales[scale_idx]
        return shortest_edge_scale, scale_idx

    def _random_scale(self, results):
        """Randomly sample an shortest_edge_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``shortest_edge_scale``.
        If multiple scales are specified by ``shortest_edge_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.shortest_edge_scale[0], self.ratio_range
            )
        elif len(self.shortest_edge_scale) == 1:
            scale, scale_idx = self.shortest_edge_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.shortest_edge_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.shortest_edge_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if "scale" not in results:
            if "scale_factor" in results:
                img_shape = results["img"].shape[:2]
                scale_factor = results["scale_factor"]
                assert isinstance(scale_factor, float)
                results["scale"] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1]
                )
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert (
                    "scale_factor" not in results
                ), "scale and scale_factor cannot be both set."
            else:
                results.pop("scale")
                if "scale_factor" in results:
                    results.pop("scale_factor")
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_sites(results)
        self._resize_seg(results)
        return results

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get("img_fields", ["img"]):
            h, w = results[key].shape[:2]
            # print('before resize: ', h,w)
            size = results["scale"]
            scale = size * 1.0 / min(h, w)
            if h < w:
                newh, neww = size, scale * w
            else:
                newh, neww = scale * h, size

            # print('new h, new w: ', newh, neww)

            if max(newh, neww) > self.max_size:
                scale = self.max_size * 1.0 / max(newh, neww)
                newh = newh * scale
                neww = neww * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)
            # print('new h, new w: ', newh, neww)

            results["scale"] = (neww, newh)
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results["scale"],
                    return_scale=True,
                    backend=self.backend,
                )
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
                # print('after resize: ', new_h, new_w)

            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results["scale"],
                    return_scale=True,
                    backend=self.backend,
                )
            results[key] = img

            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
            )
            results["img_shape"] = img.shape
            # in case that there is no padding
            results["pad_shape"] = img.shape
            results["scale_factor"] = scale_factor
            results["keep_ratio"] = self.keep_ratio


@PIPELINES.register_module()
class RandomFlipWithSites(RandomFlip):
    def site_flip(self, sites, img_shape, direction):
        """Flip sites horizontally.

        Args:
            sites (numpy.ndarray): Bounding boxes, shape (..., 2, P)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        assert sites.shape[-2] == 2
        flipped = sites.clone()
        if direction == "horizontal":
            w = img_shape[1]
            flipped[:, 0, :] = w - sites[:, 0, :]
        elif direction == "vertical":
            h = img_shape[0]
            flipped[:, 1, :] = h - sites[:, 1, :]
        elif direction == "diagonal":
            w = img_shape[1]
            h = img_shape[0]
            flipped[:, 0, :] = w - sites[:, 0, :]
            flipped[:, 1, :] = h - sites[:, 1, :]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if "flip" not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) - 1) + [
                    non_flip_ratio
                ]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results["flip"] = cur_dir is not None
        if "flip_direction" not in results:
            results["flip_direction"] = cur_dir
        if results["flip"]:
            # flip image
            for key in results.get("img_fields", ["img"]):
                results[key] = mmcv.imflip(
                    results[key], direction=results["flip_direction"]
                )
            # flip bboxes
            for key in results.get("bbox_fields", []):
                results[key] = self.bbox_flip(
                    results[key], results["img_shape"], results["flip_direction"]
                )
            # flip sites
            for key in results.get("site_fields", []):
                results[key] = self.site_flip(
                    results[key], results["img_shape"], results["flip_direction"]
                )

            # flip masks
            for key in results.get("mask_fields", []):
                results[key] = results[key].flip(results["flip_direction"])

            # flip segs
            for key in results.get("seg_fields", []):
                results[key] = mmcv.imflip(
                    results[key], direction=results["flip_direction"]
                )
        return results
