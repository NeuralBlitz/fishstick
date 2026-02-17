"""
Fishstick Computer Vision Pipelines
===================================

Comprehensive collection of computer vision model implementations including:
- Object Detection (YOLO, SSD, FCOS)
- Instance Segmentation (Mask R-CNN, YOLACT)
- Semantic Segmentation (UNet, DeepLabV3, PSPNet)
- Keypoint Detection (Keypoint R-CNN)
- Panoptic Segmentation (Panoptic FPN)
- Vision Utilities and Data Augmentation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import math
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Vision Utilities
# =============================================================================


class AnchorGenerator(nn.Module):
    """Generate anchor boxes for object detection.

    Generates anchors at multiple scales and aspect ratios for each feature
    map location. Used in two-stage detectors and some single-stage detectors.

    Args:
        sizes: List of anchor sizes for each feature map level
        aspect_ratios: List of aspect ratios for each feature map level
    """

    def __init__(
        self, sizes: List[List[int]] = None, aspect_ratios: List[List[float]] = None
    ):
        super().__init__()

        if sizes is None:
            sizes = [[32], [64], [128], [256], [512]]
        if aspect_ratios is None:
            aspect_ratios = [[0.5, 1.0, 2.0]] * len(sizes)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

        self.cell_anchors = self._create_cell_anchors()

    def _create_cell_anchors(self) -> List[Tensor]:
        """Create anchor templates for each feature map level."""
        cell_anchors = []

        for size, ratios in zip(self.sizes, self.aspect_ratios):
            anchors = []
            for s in size:
                for ratio in ratios:
                    w = s * math.sqrt(ratio)
                    h = s / math.sqrt(ratio)
                    anchors.append([-w / 2, -h / 2, w / 2, h / 2])
            cell_anchors.append(torch.tensor(anchors, dtype=torch.float32))

        return cell_anchors

    def forward(
        self, feature_maps: List[Tensor], image_sizes: List[Tuple[int, int]]
    ) -> List[Tensor]:
        """Generate anchors for all feature map locations.

        Args:
            feature_maps: List of feature maps from backbone
            image_sizes: List of (H, W) tuples for each image

        Returns:
            List of anchor tensors, one per image
        """
        anchors = []

        for idx, (feature_map, cell_anchor) in enumerate(
            zip(feature_maps, self.cell_anchors)
        ):
            grid_height, grid_width = feature_map.shape[-2:]
            stride_h = image_sizes[0][0] / grid_height
            stride_w = image_sizes[0][1] / grid_width

            shifts_x = torch.arange(grid_width, dtype=torch.float32) * stride_w
            shifts_y = torch.arange(grid_height, dtype=torch.float32) * stride_h

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1)

            all_anchors = shifts.view(-1, 1, 4) + cell_anchor.view(1, -1, 4)
            anchors.append(all_anchors.view(-1, 4))

        return anchors


class BoxCoder(nn.Module):
    """Encode and decode bounding boxes.

    Converts between normalized box representations (center, size) and
    absolute coordinates (x1, y1, x2, y2).

    Args:
        weights: Weights for box regression targets (dx, dy, dw, dh)
        bbox_xform_clip: Clipping value for transformed boxes
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        bbox_xform_clip: float = math.log(1000.0 / 16),
    ):
        super().__init__()
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """Encode proposals relative to reference boxes.

        Args:
            reference_boxes: Ground truth boxes (N, 4)
            proposals: Region proposals (N, 4)

        Returns:
            Encoded box targets (N, 4)
        """
        ex_widths = proposals[:, 2] - proposals[:, 0]
        ex_heights = proposals[:, 3] - proposals[:, 1]
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0]
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1]
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        targets_dx = self.weights[0] * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = self.weights[1] * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = self.weights[2] * torch.log(gt_widths / ex_widths)
        targets_dh = self.weights[3] * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """Decode relative codes to absolute box coordinates.

        Args:
            rel_codes: Encoded box deltas (N, 4)
            boxes: Reference boxes (N, 4)

        Returns:
            Decoded box coordinates (N, 4)
        """
        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = rel_codes[:, 0::4] / self.weights[0]
        dy = rel_codes[:, 1::4] / self.weights[1]
        dw = rel_codes[:, 2::4] / self.weights[2]
        dh = rel_codes[:, 3::4] / self.weights[3]

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Non-maximum suppression.

    Removes overlapping bounding boxes based on IoU threshold.

    Args:
        boxes: Bounding boxes (N, 4) in format (x1, y1, x2, y2)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def batched_nms(
    boxes: Tensor, scores: Tensor, idxs: Tensor, iou_threshold: float
) -> Tensor:
    """Batched non-maximum suppression for multiple classes.

    Args:
        boxes: Bounding boxes (N, 4)
        scores: Confidence scores (N,)
        idxs: Class indices (N,)
        iou_threshold: IoU threshold

    Returns:
        Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


class ROIAlign(nn.Module):
    """Region of Interest Align layer.

    Extracts fixed-size features from feature maps using bilinear interpolation
    for better gradient flow compared to RoI Pooling.

    Args:
        output_size: Output spatial size (H, W)
        spatial_scale: Scale factor to map boxes from input to feature space
        sampling_ratio: Number of sampling points per bin
        aligned: Use aligned RoI pooling
    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        spatial_scale: float,
        sampling_ratio: int = -1,
        aligned: bool = True,
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        """Apply RoI Align.

        Args:
            input: Feature map (N, C, H, W)
            rois: Regions of interest (M, 5) with batch index

        Returns:
            Pooled features (M, C, output_size[0], output_size[1])
        """
        # Manual implementation
        batch_size, channels, _, _ = input.shape
        num_rois = rois.shape[0]

        output = torch.zeros(
            num_rois,
            channels,
            self.output_size[0],
            self.output_size[1],
            device=input.device,
            dtype=input.dtype,
        )

        for roi_idx in range(num_rois):
            batch_idx = int(rois[roi_idx, 0].item())
            x1, y1, x2, y2 = rois[roi_idx, 1:5]

            x1 *= self.spatial_scale
            y1 *= self.spatial_scale
            x2 *= self.spatial_scale
            y2 *= self.spatial_scale

            roi_width = max(x2 - x1, 1)
            roi_height = max(y2 - y1, 1)

            bin_width = roi_width / self.output_size[1]
            bin_height = roi_height / self.output_size[0]

            for ph in range(self.output_size[0]):
                for pw in range(self.output_size[1]):
                    ystart = y1 + ph * bin_height
                    xstart = x1 + pw * bin_width
                    yend = ystart + bin_height
                    xend = xstart + bin_width

                    # Sample center point with bilinear interpolation
                    y_sample = (ystart + yend) / 2
                    x_sample = (xstart + xend) / 2

                    # Normalize to [-1, 1] for grid_sample
                    grid_y = 2.0 * y_sample / input.shape[2] - 1
                    grid_x = 2.0 * x_sample / input.shape[3] - 1

                    grid = torch.tensor(
                        [[[grid_x, grid_y]]], device=input.device, dtype=input.dtype
                    )

                    sampled = F.grid_sample(
                        input[batch_idx : batch_idx + 1],
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )

                    output[roi_idx, :, ph, pw] = sampled[0, :, 0, 0]

        return output


class ROIPool(nn.Module):
    """Region of Interest Pooling layer.

    Quantizes RoI into fixed-size bins using max pooling.

    Args:
        output_size: Output spatial size (H, W)
        spatial_scale: Scale factor to map boxes from input to feature space
    """

    def __init__(self, output_size: Tuple[int, int], spatial_scale: float):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        """Apply RoI Pooling.

        Args:
            input: Feature map (N, C, H, W)
            rois: Regions of interest (M, 5) with batch index

        Returns:
            Pooled features (M, C, output_size[0], output_size[1])
        """
        batch_size, channels, _, _ = input.shape
        num_rois = rois.shape[0]

        output = torch.zeros(
            num_rois,
            channels,
            self.output_size[0],
            self.output_size[1],
            device=input.device,
            dtype=input.dtype,
        )

        for roi_idx in range(num_rois):
            batch_idx = int(rois[roi_idx, 0].item())
            x1, y1, x2, y2 = rois[roi_idx, 1:5]

            x1 = int(x1 * self.spatial_scale)
            y1 = int(y1 * self.spatial_scale)
            x2 = int(x2 * self.spatial_scale)
            y2 = int(y2 * self.spatial_scale)

            roi = input[batch_idx, :, y1:y2, x1:x2]

            if roi.numel() > 0:
                pooled = F.adaptive_max_pool2d(roi, self.output_size)
                output[roi_idx] = pooled

        return output


class PostProcessor:
    """Post-process detection outputs.

    Applies score filtering, NMS, and formats final detections.

    Args:
        score_threshold: Minimum score to keep a detection
        nms_threshold: IoU threshold for NMS
        detections_per_img: Maximum detections per image
    """

    def __init__(
        self,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        detections_per_img: int = 100,
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.detections_per_img = detections_per_img

    def __call__(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        anchors: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        """Post-process detection outputs."""
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(anchors[i]) for i in range(len(anchors))]
        concat_anchors = torch.cat(anchors, dim=0)

        # Decode box predictions
        box_coder = BoxCoder()
        box_regression = box_regression.reshape(-1, 4)
        pred_boxes = box_coder.decode(box_regression, concat_anchors)

        # Split by image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        class_logits = class_logits.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores, image_shape in zip(pred_boxes, class_logits, image_shapes):
            boxes = self._clip_boxes_to_image(boxes, image_shape)
            scores = torch.sigmoid(scores)

            image_boxes = []
            image_scores = []
            image_labels = []

            for label in range(1, num_classes):
                score = scores[:, label]

                # Filter by score threshold
                keep_idxs = score > self.score_threshold
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # Apply NMS
                keep = nms(box, score, self.nms_threshold)
                box = box[keep]
                score = score[keep]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(
                    torch.full((len(box),), label, dtype=torch.int64, device=device)
                )

            if len(image_boxes) > 0:
                image_boxes = torch.cat(image_boxes, dim=0)
                image_scores = torch.cat(image_scores, dim=0)
                image_labels = torch.cat(image_labels, dim=0)

                # Keep top detections
                if len(image_scores) > self.detections_per_img:
                    keep = torch.topk(image_scores, self.detections_per_img)[1]
                    image_boxes = image_boxes[keep]
                    image_scores = image_scores[keep]
                    image_labels = image_labels[keep]
            else:
                image_boxes = torch.zeros(0, 4, device=device)
                image_scores = torch.zeros(0, device=device)
                image_labels = torch.zeros(0, dtype=torch.int64, device=device)

            all_boxes.append(image_boxes)
            all_scores.append(image_scores)
            all_labels.append(image_labels)

        results = []
        for boxes, scores, labels in zip(all_boxes, all_scores, all_labels):
            results.append({"boxes": boxes, "scores": scores, "labels": labels})

        return results

    def _clip_boxes_to_image(self, boxes: Tensor, size: Tuple[int, int]) -> Tensor:
        """Clip boxes to image boundaries."""
        boxes = boxes.clone()
        boxes[..., 0::2].clamp_(min=0, max=size[1])
        boxes[..., 1::2].clamp_(min=0, max=size[0])
        return boxes
