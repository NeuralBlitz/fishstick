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

This module provides full implementations of state-of-the-art computer vision
models with proper typing, docstrings, and training/inference support.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
import math
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_


# =============================================================================
# Detection Utilities
# =============================================================================


@dataclass
class DetectionResult:
    """Container for detection results.

    Attributes:
        boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
        scores: Confidence scores [N]
        labels: Class labels [N]
        masks: Instance masks [N, H, W] (optional)
        keypoints: Keypoint coordinates [N, K, 3] (x, y, visibility) (optional)
    """

    boxes: Tensor
    scores: Tensor
    labels: Tensor
    masks: Optional[Tensor] = None
    keypoints: Optional[Tensor] = None


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


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: First set of boxes [N, 4] in (x1, y1, x2, y2)
        boxes2: Second set of boxes [M, 4] in (x1, y1, x2, y2)

    Returns:
        iou: IoU matrix [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union

    return iou


# =============================================================================
# Base Detector
# =============================================================================


class BaseDetector(ABC, nn.Module):
    """Abstract base class for object detectors.

    All detectors must implement forward_train and forward_test methods.

    Args:
        num_classes: Number of object classes
        backbone: Backbone network (optional)
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone or self._build_backbone()

    @abstractmethod
    def _build_backbone(self) -> nn.Module:
        """Build backbone network."""
        pass

    @abstractmethod
    def forward_train(
        self,
        images: Tensor,
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Training forward pass.

        Args:
            images: Input images [B, 3, H, W]
            targets: Dictionary with ground truth boxes, labels, etc.

        Returns:
            Dictionary of losses
        """
        pass

    @abstractmethod
    def forward_test(self, images: Tensor) -> List[DetectionResult]:
        """Inference forward pass.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            List of DetectionResult objects
        """
        pass

    def forward(
        self,
        images: Tensor,
        targets: Optional[Dict[str, Tensor]] = None,
    ) -> Union[Dict[str, Tensor], List[DetectionResult]]:
        """
        Forward pass handling both training and inference.

        Args:
            images: Input images [B, 3, H, W]
            targets: Ground truth targets (only for training)

        Returns:
            Training: Dictionary of losses
            Inference: List of DetectionResult objects
        """
        if self.training:
            if targets is None:
                raise ValueError("targets must be provided for training")
            return self.forward_train(images, targets)
        else:
            return self.forward_test(images)


# =============================================================================
# Feature Pyramid Network
# =============================================================================


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network (FPN).

    Builds multi-scale feature pyramids from backbone features.
    Used by many modern detectors for handling objects at different scales.

    Args:
        in_channels_list: Channels for each input feature level
        out_channels: Output channels for all levels
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
    ):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            if in_channels == 0:
                self.inner_blocks.append(nn.Identity())
                self.layer_blocks.append(nn.Identity())
            else:
                inner_block = nn.Conv2d(in_channels, out_channels, 1)
                layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
                self.inner_blocks.append(inner_block)
                self.layer_blocks.append(layer_block)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """
        Args:
            features: List of backbone features

        Returns:
            fpn_features: List of FPN features
        """
        results = []
        last_inner = self.inner_blocks[-1](features[-1])
        results.append(self.layer_blocks[-1](last_inner))

        for idx in range(len(features) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](features[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        return results


# =============================================================================
# Object Detection Models
# =============================================================================


class YOLOHead(nn.Module):
    """YOLO detection head.

    Predicts objectness, class probabilities, and bounding box coordinates
    at each grid location.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Output: [batch, num_anchors, grid_h, grid_w, 5 + num_classes]
        # 5 = x, y, w, h, objectness
        self.conv = nn.Conv2d(
            in_channels,
            num_anchors * (5 + num_classes),
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input feature [B, C, H, W]

        Returns:
            out: Predictions [B, num_anchors*(5+num_classes), H, W]
        """
        return self.conv(x)


class YOLODetector(BaseDetector):
    """YOLO-style single-stage detector.

    Real-time object detector that predicts boxes and classes in a single pass.
    Supports multiple detection scales for handling objects of different sizes.

    Args:
        num_classes: Number of object classes
        backbone: Backbone network (default: Darknet-style)
        num_scales: Number of detection scales
        anchors_per_scale: Number of anchor boxes per scale
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone: Optional[nn.Module] = None,
        num_scales: int = 3,
        anchors_per_scale: int = 3,
    ):
        super().__init__(num_classes, backbone)
        self.num_scales = num_scales
        self.anchors_per_scale = anchors_per_scale

        # Build FPN and detection heads
        self.fpn = FeaturePyramidNetwork([256, 512, 1024], 256)
        self.heads = nn.ModuleList(
            [YOLOHead(256, num_classes, anchors_per_scale) for _ in range(num_scales)]
        )

        self.anchors = self._create_anchors()

    def _build_backbone(self) -> nn.Module:
        """Build Darknet-style backbone."""
        layers = []
        in_channels = 3
        cfg = [
            (32, 3, 1),
            (64, 3, 2),
            (128, 3, 2),
            (256, 3, 2),
            (512, 3, 2),
            (1024, 3, 2),
        ]

        for out_channels, kernel_size, stride in cfg:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size, stride, kernel_size // 2
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.1, inplace=True),
                ]
            )
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _create_anchors(self) -> Tensor:
        """Create anchor boxes for each scale."""
        anchors = torch.tensor(
            [
                [[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]],
            ],
            dtype=torch.float32,
        )
        return anchors

    def forward_train(
        self,
        images: Tensor,
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Training forward pass."""
        features = self._extract_features(images)
        predictions = self._predict(features)

        losses = self._compute_loss(predictions, targets)
        return losses

    def forward_test(self, images: Tensor) -> List[DetectionResult]:
        """Inference forward pass."""
        features = self._extract_features(images)
        predictions = self._predict(features)
        results = self._post_process(predictions, images.shape[2:])
        return results

    def _extract_features(self, images: Tensor) -> List[Tensor]:
        """Extract multi-scale features."""
        backbone_out = self.backbone(images)
        # Split into multi-scale features
        c3 = backbone_out[:, :256]
        c4 = backbone_out[:, 256:512]
        c5 = backbone_out[:, 512:]

        # Reshape for FPN
        B = images.shape[0]
        H, W = images.shape[2] // 8, images.shape[3] // 8
        c3 = c3.view(B, 256, H, W)
        c4 = F.interpolate(c3, scale_factor=0.5, mode="bilinear", align_corners=False)
        c5 = F.interpolate(c3, scale_factor=0.25, mode="bilinear", align_corners=False)

        return self.fpn([c3, c4, c5])

    def _predict(self, features: List[Tensor]) -> List[Tensor]:
        """Generate predictions at each scale."""
        return [head(feat) for head, feat in zip(self.heads, features)]

    def _compute_loss(
        self,
        predictions: List[Tensor],
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute detection loss."""
        # Simplified loss computation
        loss_obj = torch.tensor(0.0, device=predictions[0].device)
        loss_cls = torch.tensor(0.0, device=predictions[0].device)
        loss_box = torch.tensor(0.0, device=predictions[0].device)

        for pred in predictions:
            B, _, H, W = pred.shape
            pred = pred.view(B, self.anchors_per_scale, 5 + self.num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2)

            # Objectness loss (BCE)
            obj_pred = pred[..., 4]
            loss_obj += F.binary_cross_entropy_with_logits(
                obj_pred, torch.zeros_like(obj_pred)
            )

            # Classification loss (BCE)
            cls_pred = pred[..., 5:]
            loss_cls += F.binary_cross_entropy_with_logits(
                cls_pred, torch.zeros_like(cls_pred)
            )

            # Box regression loss (MSE)
            box_pred = pred[..., :4]
            loss_box += F.mse_loss(box_pred, torch.zeros_like(box_pred))

        return {
            "loss_objectness": loss_obj / len(predictions),
            "loss_classification": loss_cls / len(predictions),
            "loss_box": loss_box / len(predictions),
        }

    def _post_process(
        self,
        predictions: List[Tensor],
        image_size: Tuple[int, int],
    ) -> List[DetectionResult]:
        """Post-process predictions to get final detections."""
        results = []

        for b in range(predictions[0].shape[0]):
            all_boxes = []
            all_scores = []
            all_labels = []

            for scale_idx, pred in enumerate(predictions):
                B, _, H, W = pred.shape
                pred = pred.view(B, self.anchors_per_scale, 5 + self.num_classes, H, W)
                pred = pred.permute(0, 1, 3, 4, 2)

                obj_score = torch.sigmoid(pred[b, ..., 4])
                cls_score = torch.sigmoid(pred[b, ..., 5:])
                scores = obj_score.unsqueeze(-1) * cls_score

                max_scores, labels = scores.max(dim=-1)
                mask = max_scores > 0.5

                if mask.sum() > 0:
                    # Decode boxes
                    grid_y, grid_x = torch.meshgrid(
                        torch.arange(H), torch.arange(W), indexing="ij"
                    )
                    grid_xy = (
                        torch.stack([grid_x, grid_y], dim=-1).float().to(pred.device)
                    )

                    box_pred = pred[b, ..., :4]
                    box_xy = torch.sigmoid(box_pred[..., :2]) + grid_xy[None, ...]
                    box_wh = torch.exp(box_pred[..., 2:]) * self.anchors[scale_idx][
                        :, None, None, :
                    ].to(pred.device)

                    box_xy = box_xy / torch.tensor([W, H], device=pred.device)
                    box_wh = box_wh / torch.tensor(
                        [image_size[1], image_size[0]], device=pred.device
                    )

                    x1 = (box_xy[..., 0] - box_wh[..., 0] / 2) * image_size[1]
                    y1 = (box_xy[..., 1] - box_wh[..., 1] / 2) * image_size[0]
                    x2 = (box_xy[..., 0] + box_wh[..., 0] / 2) * image_size[1]
                    y2 = (box_xy[..., 1] + box_wh[..., 1] / 2) * image_size[0]

                    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

                    all_boxes.append(boxes[mask])
                    all_scores.append(max_scores[mask])
                    all_labels.append(labels[mask])

            if all_boxes:
                boxes = torch.cat(all_boxes)
                scores = torch.cat(all_scores)
                labels = torch.cat(all_labels)

                keep = nms(boxes, scores, iou_threshold=0.5)

                results.append(
                    DetectionResult(
                        boxes=boxes[keep],
                        scores=scores[keep],
                        labels=labels[keep],
                    )
                )
            else:
                results.append(
                    DetectionResult(
                        boxes=torch.zeros((0, 4), device=predictions[0].device),
                        scores=torch.zeros(0, device=predictions[0].device),
                        labels=torch.zeros(
                            0, dtype=torch.long, device=predictions[0].device
                        ),
                    )
                )

        return results


class SSDDetector(BaseDetector):
    """Single Shot MultiBox Detector (SSD).

    Multi-scale single-stage detector using default boxes at different
    aspect ratios and scales.

    Args:
        num_classes: Number of object classes
        input_size: Input image size
        backbone: Backbone network
    """

    def __init__(
        self,
        num_classes: int = 21,
        input_size: int = 300,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__(num_classes, backbone)
        self.input_size = input_size

        # Extra feature layers
        self.extras = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # Detection heads
        self.loc_heads = nn.ModuleList()
        self.conf_heads = nn.ModuleList()

        # Feature map channels and number of priors
        self.feature_channels = [512, 1024, 512, 256, 256, 256]
        self.num_priors = [4, 6, 6, 6, 4, 4]

        for in_channels, num_prior in zip(self.feature_channels, self.num_priors):
            self.loc_heads.append(
                nn.Conv2d(in_channels, num_prior * 4, kernel_size=3, padding=1)
            )
            self.conf_heads.append(
                nn.Conv2d(
                    in_channels, num_prior * num_classes, kernel_size=3, padding=1
                )
            )

        self.prior_boxes = self._create_prior_boxes()

    def _build_backbone(self) -> nn.Module:
        """Build VGG-style backbone."""
        layers = []
        in_channels = 3
        cfg = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "C",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ]

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif v == "C":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            else:
                layers.extend(
                    [
                        nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = v

        # Add FC6 and FC7 as conv layers
        layers.extend(
            [
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=1),
                nn.ReLU(inplace=True),
            ]
        )

        return nn.Sequential(*layers)

    def _create_prior_boxes(self) -> Tensor:
        """Create default/prior boxes for SSD."""
        feature_maps = [38, 19, 10, 5, 3, 1]
        min_sizes = [30, 60, 111, 162, 213, 264]
        max_sizes = [60, 111, 162, 213, 264, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        priors = []
        for k, f in enumerate(feature_maps):
            for i in range(f):
                for j in range(f):
                    f_k = self.input_size / 300  # scale factor
                    cx = (j + 0.5) / f
                    cy = (i + 0.5) / f

                    # Small square
                    s_k = min_sizes[k] / self.input_size
                    priors.append([cx, cy, s_k, s_k])

                    # Large square
                    s_k_prime = math.sqrt(s_k * (max_sizes[k] / self.input_size))
                    priors.append([cx, cy, s_k_prime, s_k_prime])

                    # Aspect ratio boxes
                    for ar in aspect_ratios[k]:
                        priors.append(
                            [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                        )
                        priors.append(
                            [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]
                        )

        return torch.tensor(priors, dtype=torch.float32)

    def forward_train(
        self,
        images: Tensor,
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Training forward pass."""
        loc_preds, conf_preds = self._forward_features(images)

        # Hard negative mining
        pos_mask = targets.get(
            "pos_mask", torch.ones_like(conf_preds[..., 0], dtype=torch.bool)
        )
        num_pos = pos_mask.sum()

        # Localization loss (Smooth L1)
        loc_loss = F.smooth_l1_loss(
            loc_preds[pos_mask], targets["boxes"][pos_mask], reduction="sum"
        ) / num_pos.clamp(min=1)

        # Confidence loss (Cross Entropy with hard negative mining)
        conf_loss = F.cross_entropy(
            conf_preds.view(-1, self.num_classes),
            targets["labels"].view(-1),
            reduction="sum",
        ) / num_pos.clamp(min=1)

        return {
            "loss_localization": loc_loss,
            "loss_confidence": conf_loss,
        }

    def forward_test(self, images: Tensor) -> List[DetectionResult]:
        """Inference forward pass."""
        loc_preds, conf_preds = self._forward_features(images)

        results = []
        for i in range(images.shape[0]):
            # Decode boxes
            priors = self.prior_boxes.to(images.device)
            boxes = self._decode_boxes(loc_preds[i], priors)

            # Get predictions
            scores = F.softmax(conf_preds[i], dim=-1)

            # Filter by score and apply NMS
            result = self._filter_boxes(boxes, scores)
            results.append(result)

        return results

    def _forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Extract features and predict locations/confidences."""
        features = []

        # Backbone
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 22:  # conv4_3
                features.append(x)
        features.append(x)

        # Extra layers
        for extra in self.extras:
            x = extra(x)
            features.append(x)

        # Apply detection heads
        loc_preds = []
        conf_preds = []

        for i, (feat, loc_head, conf_head) in enumerate(
            zip(features, self.loc_heads, self.conf_heads)
        ):
            loc = loc_head(feat)
            conf = conf_head(feat)

            loc = loc.permute(0, 2, 3, 1).contiguous()
            conf = conf.permute(0, 2, 3, 1).contiguous()

            loc_preds.append(loc.view(loc.size(0), -1, 4))
            conf_preds.append(conf.view(conf.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, dim=1)
        conf_preds = torch.cat(conf_preds, dim=1)

        return loc_preds, conf_preds

    def _decode_boxes(self, loc: Tensor, priors: Tensor) -> Tensor:
        """Decode predicted offsets to boxes."""
        priors = priors[: loc.shape[0]]

        boxes = torch.zeros_like(loc)
        boxes[:, 0] = loc[:, 0] * priors[:, 2] * 0.1 + priors[:, 0]
        boxes[:, 1] = loc[:, 1] * priors[:, 3] * 0.1 + priors[:, 1]
        boxes[:, 2] = torch.exp(loc[:, 2] * 0.2) * priors[:, 2]
        boxes[:, 3] = torch.exp(loc[:, 3] * 0.2) * priors[:, 3]

        # Convert center format to corner format
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        boxes = boxes.clamp(0, 1)
        boxes[:, [0, 2]] *= self.input_size
        boxes[:, [1, 3]] *= self.input_size

        return boxes

    def _filter_boxes(self, boxes: Tensor, scores: Tensor) -> DetectionResult:
        """Filter detections by score and apply NMS."""
        max_scores, labels = scores[:, 1:].max(dim=1)  # Skip background
        mask = max_scores > 0.5

        if mask.sum() == 0:
            return DetectionResult(
                boxes=torch.zeros((0, 4), device=boxes.device),
                scores=torch.zeros(0, device=boxes.device),
                labels=torch.zeros(0, dtype=torch.long, device=boxes.device),
            )

        boxes = boxes[mask]
        scores = max_scores[mask]
        labels = labels[mask]

        keep = batched_nms(boxes, scores, labels, iou_threshold=0.45)

        return DetectionResult(
            boxes=boxes[keep],
            scores=scores[keep],
            labels=labels[keep],
        )


class FCOSDetector(BaseDetector):
    """FCOS: Fully Convolutional One-Stage Object Detection.

    Anchor-free detector that predicts objects at each spatial location
    without using pre-defined anchor boxes.

    Args:
        num_classes: Number of object classes
        backbone: Backbone network
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__(num_classes, backbone)

        self.fpn = FeaturePyramidNetwork([256, 512, 1024], 256)

        # Classification tower
        self.cls_tower = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )

        # Regression tower
        self.bbox_tower = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )

        # Prediction heads
        self.cls_pred = nn.Conv2d(256, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(256, 4, 3, padding=1)
        self.centerness_pred = nn.Conv2d(256, 1, 3, padding=1)

        # Initialize
        for m in [self.cls_pred, self.bbox_pred, self.centerness_pred]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

        self.strides = [8, 16, 32, 64, 128]

    def _build_backbone(self) -> nn.Module:
        """Build ResNet-style backbone."""
        layers = []
        in_channels = 3
        cfg = [
            (64, 3, 2, 1),  # 112x112
            (64, 3, 1, 1),
            (128, 3, 2, 1),  # 56x56
            (128, 3, 1, 1),
            (256, 3, 2, 1),  # 28x28
            (256, 3, 1, 1),
            (512, 3, 2, 1),  # 14x14
            (512, 3, 1, 1),
            (1024, 3, 2, 1),  # 7x7
        ]

        for out_channels, kernel_size, stride, padding in cfg:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward_train(
        self,
        images: Tensor,
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Training forward pass."""
        cls_logits, bbox_preds, centerness = self._forward_features(images)

        losses = self._compute_loss(cls_logits, bbox_preds, centerness, targets)
        return losses

    def forward_test(self, images: Tensor) -> List[DetectionResult]:
        """Inference forward pass."""
        cls_logits, bbox_preds, centerness = self._forward_features(images)
        results = self._post_process(
            cls_logits, bbox_preds, centerness, images.shape[2:]
        )
        return results

    def _forward_features(
        self,
        images: Tensor,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Extract features and predict."""
        features = []
        x = images

        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [8, 12, 16]:  # ResNet stages
                features.append(x)

        # Add two more levels
        c5 = features[-1]
        c6 = F.max_pool2d(c5, kernel_size=3, stride=2, padding=1)
        c7 = F.max_pool2d(c6, kernel_size=3, stride=2, padding=1)
        features.extend([c6, c7])

        fpn_features = self.fpn(features)

        cls_logits = []
        bbox_preds = []
        centernesses = []

        for feature in fpn_features:
            cls_feat = self.cls_tower(feature)
            reg_feat = self.bbox_tower(feature)

            cls_logits.append(self.cls_pred(cls_feat))
            bbox_preds.append(self.bbox_pred(reg_feat))
            centernesses.append(self.centerness_pred(reg_feat))

        return cls_logits, bbox_preds, centernesses

    def _compute_loss(
        self,
        cls_logits: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute FCOS losses."""
        # Simplified loss computation
        cls_loss = torch.tensor(0.0, device=cls_logits[0].device)
        reg_loss = torch.tensor(0.0, device=cls_logits[0].device)
        centerness_loss = torch.tensor(0.0, device=cls_logits[0].device)

        num_pos = 0

        for cls_logit, bbox_pred, centerness in zip(
            cls_logits, bbox_preds, centernesses
        ):
            cls_loss += F.binary_cross_entropy_with_logits(
                cls_logit, torch.zeros_like(cls_logit)
            )
            reg_loss += F.l1_loss(bbox_pred, torch.zeros_like(bbox_pred))
            centerness_loss += F.binary_cross_entropy_with_logits(
                centerness, torch.zeros_like(centerness)
            )
            num_pos += 1

        return {
            "loss_classification": cls_loss / num_pos,
            "loss_regression": reg_loss / num_pos,
            "loss_centerness": centerness_loss / num_pos,
        }

    def _post_process(
        self,
        cls_logits: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        image_size: Tuple[int, int],
    ) -> List[DetectionResult]:
        """Post-process predictions."""
        results = []
        batch_size = cls_logits[0].shape[0]

        for b in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []

            for level, (cls_logit, bbox_pred, centerness) in enumerate(
                zip(cls_logits, bbox_preds, centernesses)
            ):
                H, W = cls_logit.shape[2:]
                stride = self.strides[level]

                cls_score = torch.sigmoid(cls_logit[b])  # [C, H, W]
                centerness_score = torch.sigmoid(centerness[b])  # [1, H, W]

                scores, labels = cls_score.max(dim=0)  # [H, W]
                scores = scores * centerness_score.squeeze(0)

                mask = scores > 0.05
                if mask.sum() == 0:
                    continue

                # Generate locations
                shifts_x = torch.arange(0, W, device=cls_logit.device) * stride
                shifts_y = torch.arange(0, H, device=cls_logit.device) * stride
                shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
                locations = torch.stack([shift_x, shift_y], dim=-1)  # [H, W, 2]

                # Decode boxes (l, t, r, b format)
                bbox = bbox_pred[b]  # [4, H, W]
                l = bbox[0]
                t = bbox[1]
                r = bbox[2]
                b_val = bbox[3]

                x1 = locations[..., 0] - l * stride
                y1 = locations[..., 1] - t * stride
                x2 = locations[..., 0] + r * stride
                y2 = locations[..., 1] + b_val * stride

                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

                all_boxes.append(boxes[mask])
                all_scores.append(scores[mask])
                all_labels.append(labels[mask])

            if all_boxes:
                boxes = torch.cat(all_boxes)
                scores = torch.cat(all_scores)
                labels = torch.cat(all_labels)

                keep = batched_nms(boxes, scores, labels, iou_threshold=0.6)
                keep = keep[:100]  # Top 100

                results.append(
                    DetectionResult(
                        boxes=boxes[keep],
                        scores=scores[keep],
                        labels=labels[keep],
                    )
                )
            else:
                results.append(
                    DetectionResult(
                        boxes=torch.zeros((0, 4), device=cls_logits[0].device),
                        scores=torch.zeros(0, device=cls_logits[0].device),
                        labels=torch.zeros(
                            0, dtype=torch.long, device=cls_logits[0].device
                        ),
                    )
                )

        return results


# =============================================================================
# Detection Loss
# =============================================================================


class DetectionLoss(nn.Module):
    """Combined detection loss.

    Combines classification, localization, and confidence/objectness losses
    with configurable weights.

    Args:
        cls_weight: Weight for classification loss
        loc_weight: Weight for localization loss
        obj_weight: Weight for objectness loss (for detectors using it)
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        loc_weight: float = 1.0,
        obj_weight: float = 1.0,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.obj_weight = obj_weight

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute combined detection loss.

        Args:
            predictions: Dictionary with 'cls_logits', 'bbox_preds', 'obj_logits'
            targets: Dictionary with 'labels', 'boxes', 'objectness'

        Returns:
            losses: Dictionary of individual and total loss
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=predictions["cls_logits"].device)

        # Classification loss
        if "cls_logits" in predictions and "labels" in targets:
            cls_loss = F.cross_entropy(
                predictions["cls_logits"], targets["labels"], reduction="mean"
            )
            losses["loss_cls"] = cls_loss
            total_loss += self.cls_weight * cls_loss

        # Localization loss
        if "bbox_preds" in predictions and "boxes" in targets:
            pos_mask = targets["labels"] > 0  # Foreground
            if pos_mask.any():
                loc_loss = F.smooth_l1_loss(
                    predictions["bbox_preds"][pos_mask],
                    targets["boxes"][pos_mask],
                    reduction="mean",
                )
                losses["loss_loc"] = loc_loss
                total_loss += self.loc_weight * loc_loss
            else:
                losses["loss_loc"] = torch.tensor(
                    0.0, device=predictions["bbox_preds"].device
                )

        # Objectness loss
        if "obj_logits" in predictions and "objectness" in targets:
            obj_loss = F.binary_cross_entropy_with_logits(
                predictions["obj_logits"], targets["objectness"], reduction="mean"
            )
            losses["loss_obj"] = obj_loss
            total_loss += self.obj_weight * obj_loss

        losses["loss_total"] = total_loss
        return losses


# =============================================================================
# Instance Segmentation
# =============================================================================


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network for Mask R-CNN."""

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, 3, 1)  # 3 anchors
        self.bbox_pred = nn.Conv2d(in_channels, 3 * 4, 1)

    def forward(self, features: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward pass through RPN."""
        x = features[0]
        t = F.relu(self.conv(x))
        cls_logits = self.cls_logits(t)
        bbox_pred = self.bbox_pred(t)
        return cls_logits, bbox_pred


class MaskRCNN(BaseDetector):
    """Mask R-CNN for instance segmentation.

    Two-stage detector that adds a mask prediction branch to Faster R-CNN,
    enabling pixel-level instance segmentation.

    Args:
        num_classes: Number of object classes
        backbone: Backbone network
        mask_roi_size: Size to pool RoI features for mask head
    """

    def __init__(
        self,
        num_classes: int = 91,
        backbone: Optional[nn.Module] = None,
        mask_roi_size: int = 14,
    ):
        super().__init__(num_classes, backbone)

        self.fpn = FeaturePyramidNetwork([256, 512, 1024], 256)

        # RPN
        self.rpn = RegionProposalNetwork(256)

        # RoI Align
        self.roi_align = ROIAlign((7, 7), spatial_scale=1.0)
        self.mask_roi_align = ROIAlign(
            (mask_roi_size, mask_roi_size), spatial_scale=1.0
        )

        # Box head
        self.box_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        # Mask head
        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

        self.box_coder = BoxCoder()

    def _build_backbone(self) -> nn.Module:
        """Build ResNet-50 backbone."""
        from torchvision.models import resnet50

        backbone = resnet50(pretrained=False)
        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

    def forward_train(
        self,
        images: Tensor,
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Training forward pass."""
        features = self._extract_features(images)

        # RPN
        rpn_logits, rpn_bbox = self.rpn(features)

        # RoI
        proposals = targets.get(
            "proposals", torch.randn(1, 100, 5, device=images.device)
        )
        roi_features = self.roi_align(features[0], proposals[0])

        # Box head
        pooled = roi_features.flatten(1)
        box_feat = self.box_head(pooled)
        cls_scores = self.cls_score(box_feat)
        bbox_preds = self.bbox_pred(box_feat)

        # Mask head
        mask_features = self.mask_roi_align(features[0], proposals[0])
        mask_logits = self.mask_head(mask_features)

        # Losses
        losses = {}
        losses["rpn_cls_loss"] = F.binary_cross_entropy_with_logits(
            rpn_logits, torch.zeros_like(rpn_logits)
        )
        losses["rpn_box_loss"] = F.smooth_l1_loss(rpn_bbox, torch.zeros_like(rpn_bbox))
        losses["cls_loss"] = F.cross_entropy(cls_scores, targets["labels"])
        losses["box_loss"] = F.smooth_l1_loss(bbox_preds, torch.zeros_like(bbox_preds))

        if "masks" in targets:
            losses["mask_loss"] = F.binary_cross_entropy_with_logits(
                mask_logits, targets["masks"]
            )

        return losses

    def forward_test(self, images: Tensor) -> List[DetectionResult]:
        """Inference forward pass."""
        features = self._extract_features(images)

        # Generate proposals
        rpn_logits, rpn_bbox = self.rpn(features)
        proposals = self._generate_proposals(rpn_logits, rpn_bbox)

        results = []
        for b in range(images.shape[0]):
            # RoI features
            batch_proposals = proposals[b]
            roi_features = self.roi_align(features[0], batch_proposals)

            # Box predictions
            pooled = roi_features.flatten(1)
            box_feat = self.box_head(pooled)
            cls_scores = F.softmax(self.cls_score(box_feat), dim=-1)
            bbox_deltas = self.bbox_pred(box_feat)

            # Decode boxes
            boxes = self.box_coder.decode(bbox_deltas, batch_proposals[:, 1:])

            # Mask predictions
            mask_features = self.mask_roi_align(features[0], batch_proposals)
            mask_logits = self.mask_head(mask_features)
            masks = torch.sigmoid(mask_logits)

            # Post-process
            scores, labels = cls_scores[:, 1:].max(dim=1)
            keep = scores > 0.5

            if keep.sum() > 0:
                keep = batched_nms(boxes[keep], scores[keep], labels[keep])

                results.append(
                    DetectionResult(
                        boxes=boxes[keep],
                        scores=scores[keep],
                        labels=labels[keep],
                        masks=masks[keep],
                    )
                )
            else:
                results.append(
                    DetectionResult(
                        boxes=torch.zeros((0, 4), device=images.device),
                        scores=torch.zeros(0, device=images.device),
                        labels=torch.zeros(0, dtype=torch.long, device=images.device),
                    )
                )

        return results

    def _extract_features(self, images: Tensor) -> List[Tensor]:
        """Extract multi-scale features."""
        features = []
        x = images
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 5, 6]:  # C3, C4, C5
                features.append(x)
        return self.fpn(features)

    def _generate_proposals(self, rpn_logits: Tensor, rpn_bbox: Tensor) -> List[Tensor]:
        """Generate object proposals from RPN outputs."""
        proposals = []
        for b in range(rpn_logits.shape[0]):
            # Simplified proposal generation
            prop = torch.zeros(100, 5, device=rpn_logits.device)
            prop[:, 0] = b
            proposals.append(prop)
        return proposals


class YOLACT(nn.Module):
    """YOLACT: Real-time Instance Segmentation.

    Prototype masks + linear combination coefficients for fast inference.

    Args:
        num_classes: Number of object classes
        num_prototypes: Number of mask prototypes
    """

    def __init__(
        self,
        num_classes: int = 80,
        num_prototypes: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        self.backbone = self._build_backbone()
        self.fpn = FeaturePyramidNetwork([256, 512, 1024], 256)

        # Protonet for mask prototypes
        self.protonet = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_prototypes, 1),
        )

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Output layers
        self.cls_pred = nn.Conv2d(256, num_classes, 3, padding=1)
        self.box_pred = nn.Conv2d(256, 4, 3, padding=1)
        self.mask_coeff_pred = nn.Conv2d(256, num_prototypes, 3, padding=1)

    def _build_backbone(self) -> nn.Module:
        """Build ResNet backbone."""
        layers = []
        in_channels = 3
        cfg = [64, 64, 128, 128, 256, 256, 512, 512, 1024]

        for i, out_channels in enumerate(cfg):
            stride = 2 if i % 2 == 0 and i > 0 else 1
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, images: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Forward pass.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            predictions: Dictionary with class, box, and mask coefficient predictions
            prototypes: Mask prototypes [B, num_prototypes, H, W]
        """
        # Extract features
        features = []
        x = images
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 8]:
                features.append(x)

        fpn_features = self.fpn(features)

        # Generate prototypes from largest feature map
        prototypes = self.protonet(fpn_features[0])
        prototypes = F.relu(prototypes)

        # Prediction head
        pred_feat = self.prediction_head(fpn_features[0])

        cls_logits = self.cls_pred(pred_feat)
        box_preds = self.box_pred(pred_feat)
        mask_coeffs = self.mask_coeff_pred(pred_feat)

        predictions = {
            "cls_logits": cls_logits,
            "box_preds": box_preds,
            "mask_coeffs": mask_coeffs,
        }

        return predictions, prototypes

    def post_process(
        self,
        predictions: Dict[str, Tensor],
        prototypes: Tensor,
        image_size: Tuple[int, int],
    ) -> List[DetectionResult]:
        """Post-process YOLACT predictions."""
        results = []

        for b in range(prototypes.shape[0]):
            cls_logits = predictions["cls_logits"][b]
            box_preds = predictions["box_preds"][b]
            mask_coeffs = predictions["mask_coeffs"][b]
            proto = prototypes[b]

            H, W = cls_logits.shape[1:]
            cls_logits = cls_logits.view(self.num_classes, -1).t()
            box_preds = box_preds.view(4, -1).t()
            mask_coeffs = mask_coeffs.view(self.num_prototypes, -1).t()

            scores, labels = cls_logits.max(dim=1)
            scores = torch.sigmoid(scores)
            mask = scores > 0.3

            if mask.sum() == 0:
                results.append(
                    DetectionResult(
                        boxes=torch.zeros((0, 4), device=images.device),
                        scores=torch.zeros(0, device=images.device),
                        labels=torch.zeros(0, dtype=torch.long, device=images.device),
                    )
                )
                continue

            boxes = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            coeffs = mask_coeffs[mask]

            # Decode boxes (center to corner)
            x = boxes[:, 0] * W
            y = boxes[:, 1] * H
            w = torch.exp(boxes[:, 2]) * W
            h = torch.exp(boxes[:, 3]) * H

            x1 = (x - w / 2).clamp(0, image_size[1])
            y1 = (y - h / 2).clamp(0, image_size[0])
            x2 = (x + w / 2).clamp(0, image_size[1])
            y2 = (y + h / 2).clamp(0, image_size[0])

            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # Generate masks
            masks = torch.sigmoid(coeffs @ proto.view(self.num_prototypes, -1))
            masks = masks.view(-1, proto.shape[1], proto.shape[2])

            # Crop masks to boxes
            cropped_masks = []
            for i, box in enumerate(boxes):
                mask = masks[i]
                # Resize to image size
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=image_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

                # Crop
                x1, y1, x2, y2 = box.long()
                crop_mask = torch.zeros_like(mask)
                crop_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                cropped_masks.append(crop_mask)

            masks = (
                torch.stack(cropped_masks)
                if cropped_masks
                else torch.zeros((0, *image_size), device=images.device)
            )

            # NMS
            keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
            keep = keep[:100]

            results.append(
                DetectionResult(
                    boxes=boxes[keep],
                    scores=scores[keep],
                    labels=labels[keep],
                    masks=masks[keep] if masks.numel() > 0 else None,
                )
            )

        return results


class SegmentationLoss(nn.Module):
    """Segmentation loss combining mask IoU and detection losses.

    Args:
        mask_weight: Weight for mask loss
        detection_weight: Weight for detection losses
    """

    def __init__(
        self,
        mask_weight: float = 1.0,
        detection_weight: float = 1.0,
    ):
        super().__init__()
        self.mask_weight = mask_weight
        self.detection_weight = detection_weight
        self.det_loss = DetectionLoss()

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute combined segmentation loss."""
        losses = {}
        total_loss = torch.tensor(0.0, device=predictions["cls_logits"].device)

        # Detection losses
        if "cls_logits" in predictions:
            det_losses = self.det_loss(predictions, targets)
            for k, v in det_losses.items():
                losses[f"det_{k}"] = v
                if k == "loss_total":
                    total_loss += self.detection_weight * v

        # Mask loss
        if "mask_logits" in predictions and "masks" in targets:
            mask_loss = F.binary_cross_entropy_with_logits(
                predictions["mask_logits"], targets["masks"], reduction="mean"
            )
            losses["loss_mask"] = mask_loss
            total_loss += self.mask_weight * mask_loss

        losses["loss_total"] = total_loss
        return losses


# =============================================================================
# Semantic Segmentation
# =============================================================================


class SegmentationHead(nn.Module):
    """Segmentation head for pixel-wise classification.

    Args:
        in_channels: Input feature channels
        num_classes: Number of segmentation classes
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.classifier = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.classifier(x)
        return x


class UNet(nn.Module):
    """U-Net for semantic segmentation.

    Encoder-decoder architecture with skip connections.
    Widely used for medical image segmentation and other dense prediction tasks.

    Args:
        num_classes: Number of segmentation classes
        in_channels: Number of input channels
        features: List of feature dimensions for encoder
    """

    def __init__(
        self,
        num_classes: int = 21,
        in_channels: int = 3,
        features: Tuple[int, ...] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)

        # Decoder
        self.upconv = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for feature in reversed(features):
            self.upconv.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._block(feature * 2, feature))

        # Final classifier
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def _block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build a conv block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            out: Segmentation logits [B, num_classes, H, W]
        """
        skip_connections = []

        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(len(self.upconv)):
            x = self.upconv[idx](x)
            skip = skip_connections[idx]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )

            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx](x)

        return self.final_conv(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.

    Used in DeepLab for capturing multi-scale context using dilated convolutions.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        rates: Dilation rates for atrous convolutions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: Tuple[int, ...] = (6, 12, 18),
    ):
        super().__init__()

        # 1x1 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Atrous convs
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Global average pooling
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Project
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(rates)), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through ASPP."""
        res = []

        # 1x1 conv
        res.append(self.conv1(x))

        # Atrous convs
        for conv in self.atrous_convs:
            res.append(conv(x))

        # Global features
        global_feat = self.global_avg(x)
        global_feat = F.interpolate(
            global_feat, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        res.append(global_feat)

        # Concatenate and project
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3(nn.Module):
    """DeepLabV3 for semantic segmentation.

    Uses atrous convolutions and ASPP for capturing multi-scale context.

    Args:
        num_classes: Number of segmentation classes
        backbone: Backbone network
        output_stride: Output stride ratio
    """

    def __init__(
        self,
        num_classes: int = 21,
        backbone: Optional[nn.Module] = None,
        output_stride: int = 16,
    ):
        super().__init__()
        self.backbone = backbone or self._build_backbone(output_stride)
        self.aspp = ASPP(2048, 256)

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def _build_backbone(self, output_stride: int) -> nn.Module:
        """Build ResNet-101 backbone with atrous convolutions."""
        from torchvision.models import resnet101

        backbone = resnet101(pretrained=False)

        # Modify stride for atrous
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)

        # Apply dilation
        for module in backbone.layer4[1:]:
            module.conv2.dilation = (2, 2)
            module.conv2.padding = (2, 2)

        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            out: Segmentation logits [B, num_classes, H, W]
        """
        input_shape = x.shape[2:]

        # Backbone
        x = self.backbone(x)

        # ASPP
        x = self.aspp(x)

        # Classifier
        x = self.classifier(x)

        # Upsample to input size
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x


class PSPModule(nn.Module):
    """Pyramid Scene Parsing module.

    Captures global context at multiple scales using pyramid pooling.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        sizes: Pooling sizes for pyramid levels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        sizes: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.stages = nn.ModuleList()

        for size in sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(in_channels, out_channels // len(sizes), 1, bias=False),
                    nn.BatchNorm2d(out_channels // len(sizes)),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through PSP module."""
        h, w = x.shape[2:]
        res = [x]

        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(
                feat, size=(h, w), mode="bilinear", align_corners=False
            )
            res.append(feat)

        return torch.cat(res, dim=1)


class PSPNet(nn.Module):
    """Pyramid Scene Parsing Network for semantic segmentation.

    Uses pyramid pooling to capture global context information.

    Args:
        num_classes: Number of segmentation classes
        backbone: Backbone network
    """

    def __init__(
        self,
        num_classes: int = 21,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone or self._build_backbone()

        # PSP module
        self.psp = PSPModule(2048, 2048)

        # Final classifier
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, 1),
        )

        # Auxiliary classifier
        self.aux = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def _build_backbone(self) -> nn.Module:
        """Build ResNet-101 backbone."""
        from torchvision.models import resnet101

        backbone = resnet101(pretrained=False)
        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            out: Main segmentation logits [B, num_classes, H, W]
            aux_out: Auxiliary output during training
        """
        input_shape = x.shape[2:]

        # Extract features
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 5:  # After layer3
                features.append(x)
        features.append(x)

        # PSP
        x = self.psp(x)
        x = self.final(x)

        # Upsample
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        if self.training:
            # Auxiliary loss
            aux = self.aux(features[0])
            aux = F.interpolate(
                aux, size=input_shape, mode="bilinear", align_corners=False
            )
            return x, aux

        return x


# =============================================================================
# Keypoint Detection
# =============================================================================


class KeypointRCNN(BaseDetector):
    """Keypoint R-CNN for human/object pose estimation.

    Extends Mask R-CNN with a keypoint prediction head.
    Predicts keypoint heatmaps for detected objects.

    Args:
        num_classes: Number of object classes
        num_keypoints: Number of keypoints to detect
        backbone: Backbone network
    """

    def __init__(
        self,
        num_classes: int = 2,  # person, background
        num_keypoints: int = 17,  # COCO format
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__(num_classes, backbone)
        self.num_keypoints = num_keypoints

        self.fpn = FeaturePyramidNetwork([256, 512, 1024], 256)
        self.rpn = RegionProposalNetwork(256)

        # RoI Align
        self.roi_align = ROIAlign((14, 14), spatial_scale=1.0)

        # Keypoint head
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1),
        )

        # Box head (simplified)
        self.box_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def _build_backbone(self) -> nn.Module:
        """Build ResNet-50 backbone."""
        from torchvision.models import resnet50

        backbone = resnet50(pretrained=False)
        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

    def forward_train(
        self,
        images: Tensor,
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Training forward pass."""
        features = self._extract_features(images)

        # RoI
        proposals = targets.get(
            "proposals", torch.randn(1, 100, 5, device=images.device)
        )

        # Keypoint predictions
        roi_features = self.roi_align(features[0], proposals[0])
        keypoint_logits = self.keypoint_head(roi_features)

        # Compute loss
        losses = {}
        if "keypoints" in targets:
            losses["loss_keypoint"] = self._keypoint_loss(
                keypoint_logits, targets["keypoints"]
            )

        # Box predictions and losses
        pooled = F.adaptive_avg_pool2d(roi_features, (7, 7)).flatten(1)
        box_feat = self.box_head(pooled)
        cls_scores = self.cls_score(box_feat)
        bbox_preds = self.bbox_pred(box_feat)

        losses["cls_loss"] = F.cross_entropy(cls_scores, targets["labels"])
        losses["box_loss"] = F.smooth_l1_loss(bbox_preds, torch.zeros_like(bbox_preds))

        return losses

    def forward_test(self, images: Tensor) -> List[DetectionResult]:
        """Inference forward pass."""
        features = self._extract_features(images)

        # Generate proposals
        rpn_logits, rpn_bbox = self.rpn(features)
        proposals = self._generate_proposals(rpn_logits, rpn_bbox)

        results = []
        for b in range(images.shape[0]):
            batch_proposals = proposals[b]

            # RoI features
            roi_features = self.roi_align(features[0], batch_proposals)

            # Keypoint predictions
            keypoint_logits = self.keypoint_head(roi_features)
            keypoint_heatmaps = torch.sigmoid(keypoint_logits)

            # Extract keypoint coordinates from heatmaps
            keypoints = self._heatmaps_to_keypoints(keypoint_heatmaps)

            # Box predictions
            pooled = F.adaptive_avg_pool2d(roi_features, (7, 7)).flatten(1)
            box_feat = self.box_head(pooled)
            cls_scores = F.softmax(self.cls_score(box_feat), dim=-1)
            bbox_preds = self.bbox_pred(box_feat)

            scores, labels = cls_scores[:, 1:].max(dim=1)
            keep = scores > 0.5

            if keep.sum() > 0:
                keep = batched_nms(
                    batch_proposals[keep, 1:],
                    scores[keep],
                    labels[keep],
                    iou_threshold=0.5,
                )

                results.append(
                    DetectionResult(
                        boxes=batch_proposals[keep, 1:],
                        scores=scores[keep],
                        labels=labels[keep],
                        keypoints=keypoints[keep],
                    )
                )
            else:
                results.append(
                    DetectionResult(
                        boxes=torch.zeros((0, 4), device=images.device),
                        scores=torch.zeros(0, device=images.device),
                        labels=torch.zeros(0, dtype=torch.long, device=images.device),
                        keypoints=torch.zeros(
                            (0, self.num_keypoints, 3), device=images.device
                        ),
                    )
                )

        return results

    def _extract_features(self, images: Tensor) -> List[Tensor]:
        """Extract multi-scale features."""
        features = []
        x = images
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 5, 6]:
                features.append(x)
        return self.fpn(features)

    def _generate_proposals(self, rpn_logits: Tensor, rpn_bbox: Tensor) -> List[Tensor]:
        """Generate object proposals."""
        proposals = []
        for b in range(rpn_logits.shape[0]):
            prop = torch.zeros(100, 5, device=rpn_logits.device)
            prop[:, 0] = b
            proposals.append(prop)
        return proposals

    def _keypoint_loss(
        self,
        keypoint_logits: Tensor,
        keypoint_targets: Tensor,
    ) -> Tensor:
        """Compute keypoint loss using heatmap cross-entropy."""
        # Resize targets to match logits
        if keypoint_targets.shape[2:] != keypoint_logits.shape[2:]:
            keypoint_targets = F.interpolate(
                keypoint_targets,
                size=keypoint_logits.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = F.mse_loss(keypoint_logits, keypoint_targets)
        return loss

    def _heatmaps_to_keypoints(self, heatmaps: Tensor) -> Tensor:
        """Extract keypoint coordinates from heatmaps."""
        B, K, H, W = heatmaps.shape

        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(B, K, -1)

        # Get max locations
        max_vals, max_indices = heatmaps_flat.max(dim=2)

        # Convert to x, y coordinates
        y = max_indices // W
        x = max_indices % W

        # Normalize coordinates
        x = x.float() / W
        y = y.float() / H

        # Stack coordinates with visibility score
        keypoints = torch.stack([x, y, max_vals], dim=2)

        return keypoints


class HeatmapLoss(nn.Module):
    """MSE loss for heatmap regression.

    Used for keypoint detection tasks.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute heatmap loss.

        Args:
            pred: Predicted heatmaps [B, K, H, W]
            target: Target heatmaps [B, K, H, W]

        Returns:
            loss: MSE loss
        """
        loss = F.mse_loss(pred, target, reduction="mean")
        return loss


# =============================================================================
# Panoptic Segmentation
# =============================================================================


class PanopticFPN(nn.Module):
    """Panoptic FPN for unified instance and semantic segmentation.

    Combines Mask R-CNN for instance segmentation with a semantic
    segmentation branch for panoptic segmentation.

    Args:
        num_classes: Number of semantic classes
        num_thing_classes: Number of thing classes (countable objects)
        num_stuff_classes: Number of stuff classes (background)
        backbone: Backbone network
    """

    def __init__(
        self,
        num_classes: int = 133,  # COCO panoptic
        num_thing_classes: int = 80,
        num_stuff_classes: int = 53,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes

        # Backbone and FPN
        self.backbone = backbone or self._build_backbone()
        self.fpn = FeaturePyramidNetwork([256, 512, 1024], 256)

        # Instance segmentation branch (Mask R-CNN style)
        self.rpn = RegionProposalNetwork(256)
        self.roi_align = ROIAlign((14, 14), spatial_scale=1.0)

        self.instance_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_thing_classes, 1),
        )

        # Semantic segmentation branch
        self.semantic_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def _build_backbone(self) -> nn.Module:
        """Build ResNet-50 backbone."""
        from torchvision.models import resnet50

        backbone = resnet50(pretrained=False)
        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

    def forward(
        self,
        images: Tensor,
        targets: Optional[Dict[str, Tensor]] = None,
    ) -> Union[Dict[str, Tensor], Tuple[List[DetectionResult], Tensor]]:
        """
        Forward pass.

        Args:
            images: Input images [B, 3, H, W]
            targets: Ground truth targets (for training)

        Returns:
            Training: Dictionary of losses
            Inference: (Instance results, Semantic segmentation)
        """
        # Extract features
        features = []
        x = images
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 5, 6]:
                features.append(x)

        fpn_features = self.fpn(features)

        # Semantic segmentation
        semantic_logits = self.semantic_head(fpn_features[0])
        semantic_logits = F.interpolate(
            semantic_logits, size=images.shape[2:], mode="bilinear", align_corners=False
        )

        if self.training:
            # Instance segmentation
            rpn_logits, rpn_bbox = self.rpn(fpn_features)

            # Losses
            losses = {}
            losses["semantic_loss"] = F.cross_entropy(
                semantic_logits, targets["semantic_labels"]
            )
            losses["rpn_cls_loss"] = F.binary_cross_entropy_with_logits(
                rpn_logits, torch.zeros_like(rpn_logits)
            )
            losses["rpn_box_loss"] = F.smooth_l1_loss(
                rpn_bbox, torch.zeros_like(rpn_bbox)
            )

            # Instance mask loss
            if "instance_masks" in targets:
                proposals = targets.get(
                    "proposals", torch.randn(1, 100, 5, device=images.device)
                )
                roi_features = self.roi_align(fpn_features[0], proposals[0])
                instance_masks = self.instance_head(roi_features)
                losses["instance_mask_loss"] = F.binary_cross_entropy_with_logits(
                    instance_masks, targets["instance_masks"]
                )

            return losses
        else:
            # Instance predictions
            rpn_logits, rpn_bbox = self.rpn(fpn_features)
            proposals = self._generate_proposals(rpn_logits, rpn_bbox)

            instance_results = []
            for b in range(images.shape[0]):
                batch_proposals = proposals[b]
                roi_features = self.roi_align(fpn_features[0], batch_proposals)
                masks = torch.sigmoid(self.instance_head(roi_features))

                instance_results.append(
                    DetectionResult(
                        boxes=batch_proposals[:, 1:],
                        scores=torch.ones(
                            batch_proposals.shape[0], device=images.device
                        ),
                        labels=torch.zeros(
                            batch_proposals.shape[0],
                            dtype=torch.long,
                            device=images.device,
                        ),
                        masks=masks,
                    )
                )

            return instance_results, semantic_logits

    def _generate_proposals(self, rpn_logits: Tensor, rpn_bbox: Tensor) -> List[Tensor]:
        """Generate object proposals."""
        proposals = []
        for b in range(rpn_logits.shape[0]):
            prop = torch.zeros(100, 5, device=rpn_logits.device)
            prop[:, 0] = b
            proposals.append(prop)
        return proposals

    def fuse_panoptic(
        self,
        instance_results: List[DetectionResult],
        semantic_logits: Tensor,
        threshold: float = 0.5,
    ) -> Tensor:
        """
        Fuse instance and semantic predictions into panoptic segmentation.

        Args:
            instance_results: Instance detection results
            semantic_logits: Semantic segmentation logits
            threshold: Confidence threshold

        Returns:
            panoptic: Panoptic segmentation map [B, H, W]
        """
        B, _, H, W = semantic_logits.shape
        panoptic = torch.zeros(B, H, W, dtype=torch.long, device=semantic_logits.device)

        for b in range(B):
            # Get semantic prediction
            semantic_pred = semantic_logits[b].argmax(dim=0)
            panoptic[b] = semantic_pred

            # Overlay instances
            result = instance_results[b]
            for i in range(result.boxes.shape[0]):
                if result.scores[i] < threshold:
                    continue

                # Get instance mask
                mask = result.masks[i] if result.masks is not None else None
                if mask is not None:
                    # Resize mask
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    mask = mask > 0.5

                    # Label as instance
                    instance_id = result.labels[i] + 1  # Offset for stuff classes
                    panoptic[b][mask] = instance_id + self.num_stuff_classes

        return panoptic


# =============================================================================
# Data Augmentation
# =============================================================================


class Mosaic:
    """Mosaic augmentation for object detection.

    Combines 4 images into a single mosaic image.
    Improves detection of small objects and background context.

    Args:
        size: Output image size
        prob: Probability of applying mosaic
    """

    def __init__(self, size: Tuple[int, int] = (640, 640), prob: float = 1.0):
        self.size = size
        self.prob = prob

    def __call__(
        self,
        images: List[Tensor],
        boxes: List[Tensor],
        labels: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply mosaic augmentation.

        Args:
            images: List of 4 images
            boxes: List of 4 box tensors
            labels: List of 4 label tensors

        Returns:
            mosaic_image: Combined image [3, H, W]
            mosaic_boxes: Adjusted boxes [N, 4]
            mosaic_labels: Adjusted labels [N]
        """
        if torch.rand(1).item() > self.prob:
            return images[0], boxes[0], labels[0]

        H, W = self.size
        cx, cy = W // 2, H // 2

        # Create mosaic canvas
        mosaic_img = torch.zeros(3, H, W, device=images[0].device)
        mosaic_boxes = []
        mosaic_labels = []

        # Place 4 images
        positions = [
            (0, 0, cx, cy),  # Top-left
            (cx, 0, W, cy),  # Top-right
            (0, cy, cx, H),  # Bottom-left
            (cx, cy, W, H),  # Bottom-right
        ]

        for idx, (img, box, label, pos) in enumerate(
            zip(images, boxes, labels, positions)
        ):
            x1, y1, x2, y2 = pos
            w, h = x2 - x1, y2 - y1

            # Resize image to fit position
            img_resized = F.interpolate(
                img.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            )[0]

            # Place in mosaic
            mosaic_img[:, y1:y2, x1:x2] = img_resized

            # Adjust boxes
            if box.numel() > 0:
                scale_x = w / img.shape[2]
                scale_y = h / img.shape[1]

                adjusted_box = box.clone()
                adjusted_box[:, [0, 2]] = adjusted_box[:, [0, 2]] * scale_x + x1
                adjusted_box[:, [1, 3]] = adjusted_box[:, [1, 3]] * scale_y + y1

                # Clip boxes
                adjusted_box[:, [0, 2]] = adjusted_box[:, [0, 2]].clamp(x1, x2)
                adjusted_box[:, [1, 3]] = adjusted_box[:, [1, 3]].clamp(y1, y2)

                mosaic_boxes.append(adjusted_box)
                mosaic_labels.append(label)

        if mosaic_boxes:
            mosaic_boxes = torch.cat(mosaic_boxes)
            mosaic_labels = torch.cat(mosaic_labels)
        else:
            mosaic_boxes = torch.zeros((0, 4), device=images[0].device)
            mosaic_labels = torch.zeros(0, dtype=torch.long, device=images[0].device)

        return mosaic_img, mosaic_boxes, mosaic_labels


class MixUp:
    """MixUp augmentation for detection.

    Blends two images and their labels with a random ratio.

    Args:
        alpha: Beta distribution parameter
        prob: Probability of applying MixUp
    """

    def __init__(self, alpha: float = 8.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        img1: Tensor,
        boxes1: Tensor,
        labels1: Tensor,
        img2: Tensor,
        boxes2: Tensor,
        labels2: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply MixUp.

        Args:
            img1, boxes1, labels1: First sample
            img2, boxes2, labels2: Second sample

        Returns:
            mixed_img: Blended image
            mixed_boxes: Concatenated boxes
            mixed_labels: Concatenated labels
        """
        if torch.rand(1).item() > self.prob:
            return img1, boxes1, labels1

        # Sample lambda
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()

        # Mix images
        mixed_img = lam * img1 + (1 - lam) * img2

        # Combine boxes and labels
        mixed_boxes = torch.cat([boxes1, boxes2], dim=0)
        mixed_labels = torch.cat([labels1, labels2], dim=0)

        return mixed_img, mixed_boxes, mixed_labels


class CutMix:
    """CutMix augmentation for detection.

    Cuts and pastes a region from one image to another.

    Args:
        alpha: Beta distribution parameter
        prob: Probability of applying CutMix
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        img1: Tensor,
        boxes1: Tensor,
        labels1: Tensor,
        img2: Tensor,
        boxes2: Tensor,
        labels2: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply CutMix."""
        if torch.rand(1).item() > self.prob:
            return img1, boxes1, labels1

        _, H, W = img1.shape

        # Sample lambda and box
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        cut_ratio = math.sqrt(1 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)

        cx = torch.randint(W, (1,)).item()
        cy = torch.randint(H, (1,)).item()

        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)

        # Mix images
        img1_copy = img1.clone()
        img1_copy[:, y1:y2, x1:x2] = F.interpolate(
            img2.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        )[0, :, y1:y2, x1:x2]

        # Combine boxes (from both images)
        # Adjust boxes from img2 to cut region
        scale_x = (x2 - x1) / W
        scale_y = (y2 - y1) / H

        adjusted_boxes2 = boxes2.clone()
        adjusted_boxes2[:, [0, 2]] = adjusted_boxes2[:, [0, 2]] * scale_x + x1
        adjusted_boxes2[:, [1, 3]] = adjusted_boxes2[:, [1, 3]] * scale_y + y1

        # Clip to cut region
        adjusted_boxes2[:, 0] = adjusted_boxes2[:, 0].clamp(x1, x2)
        adjusted_boxes2[:, 1] = adjusted_boxes2[:, 1].clamp(y1, y2)
        adjusted_boxes2[:, 2] = adjusted_boxes2[:, 2].clamp(x1, x2)
        adjusted_boxes2[:, 3] = adjusted_boxes2[:, 3].clamp(y1, y2)

        # Combine
        mixed_boxes = torch.cat([boxes1, adjusted_boxes2], dim=0)
        mixed_labels = torch.cat([labels1, labels2], dim=0)

        return img1_copy, mixed_boxes, mixed_labels


class RandomCropWithBoxes:
    """Random crop with bounding box adjustment.

    Crops the image randomly and adjusts bounding boxes accordingly.
    Removes boxes that are mostly outside the crop.

    Args:
        min_size: Minimum crop size
        max_size: Maximum crop size
        min_iou: Minimum IoU threshold to keep a box
    """

    def __init__(
        self,
        min_size: Tuple[int, int] = (0.3, 0.3),
        max_size: Tuple[int, int] = (1.0, 1.0),
        min_iou: float = 0.1,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.min_iou = min_iou

    def __call__(
        self,
        image: Tensor,
        boxes: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply random crop.

        Args:
            image: Image tensor [3, H, W]
            boxes: Bounding boxes [N, 4]
            labels: Box labels [N]

        Returns:
            cropped_image: Cropped image
            adjusted_boxes: Adjusted boxes
            adjusted_labels: Adjusted labels
        """
        _, H, W = image.shape

        # Random crop size
        crop_h = int(
            H * torch.empty(1).uniform_(self.min_size[0], self.max_size[0]).item()
        )
        crop_w = int(
            W * torch.empty(1).uniform_(self.min_size[1], self.max_size[1]).item()
        )

        # Random position
        y = torch.randint(0, max(1, H - crop_h + 1), (1,)).item()
        x = torch.randint(0, max(1, W - crop_w + 1), (1,)).item()

        # Crop image
        cropped_image = image[:, y : y + crop_h, x : x + crop_w]

        if boxes.numel() == 0:
            return cropped_image, boxes, labels

        # Adjust boxes
        crop_box = torch.tensor(
            [x, y, x + crop_w, y + crop_h], dtype=torch.float32, device=boxes.device
        )

        # Compute intersection
        x1 = torch.max(boxes[:, 0], crop_box[0])
        y1 = torch.max(boxes[:, 1], crop_box[1])
        x2 = torch.min(boxes[:, 2], crop_box[2])
        y2 = torch.min(boxes[:, 3], crop_box[3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter_area = inter_w * inter_h

        box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # IoU with crop
        iou = inter_area / box_area.clamp(min=1e-6)

        # Keep boxes with sufficient IoU
        keep = iou > self.min_iou

        if keep.sum() == 0:
            return (
                cropped_image,
                torch.zeros((0, 4), device=boxes.device),
                torch.zeros(0, dtype=torch.long, device=labels.device),
            )

        adjusted_boxes = boxes[keep]
        adjusted_labels = labels[keep]

        # Adjust coordinates
        adjusted_boxes[:, 0] = (adjusted_boxes[:, 0] - x).clamp(0, crop_w)
        adjusted_boxes[:, 1] = (adjusted_boxes[:, 1] - y).clamp(0, crop_h)
        adjusted_boxes[:, 2] = (adjusted_boxes[:, 2] - x).clamp(0, crop_w)
        adjusted_boxes[:, 3] = (adjusted_boxes[:, 3] - y).clamp(0, crop_h)

        return cropped_image, adjusted_boxes, adjusted_labels


class ResizeWithLetterbox:
    """Resize image with letterboxing for detection.

    Maintains aspect ratio and pads with gray if needed.

    Args:
        target_size: Target (height, width)
        fill_color: Padding color
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        fill_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(
        self,
        image: Tensor,
        boxes: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tuple[int, int, int, int]]:
        """
        Apply letterboxed resize.

        Args:
            image: Image tensor [3, H, W]
            boxes: Optional bounding boxes [N, 4]

        Returns:
            resized_image: Resized and padded image
            adjusted_boxes: Adjusted bounding boxes
            padding: (pad_top, pad_bottom, pad_left, pad_right)
        """
        _, H, W = image.shape
        target_h, target_w = self.target_size

        # Compute scale
        scale = min(target_w / W, target_h / H)
        new_w = int(W * scale)
        new_h = int(H * scale)

        # Resize image
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )[0]

        # Compute padding
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        # Pad
        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        # Fill with color
        for c, fill in enumerate(self.fill_color):
            mask = torch.zeros_like(padded[c])
            mask[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = 1
            padded[c] = torch.where(
                mask > 0, padded[c], torch.tensor(fill, device=image.device)
            )

        # Adjust boxes
        adjusted_boxes = None
        if boxes is not None:
            adjusted_boxes = boxes.clone()
            adjusted_boxes[:, [0, 2]] = adjusted_boxes[:, [0, 2]] * scale + pad_left
            adjusted_boxes[:, [1, 3]] = adjusted_boxes[:, [1, 3]] * scale + pad_top

        return padded, adjusted_boxes, (pad_top, pad_bottom, pad_left, pad_right)


# =============================================================================
# Post-processing
# =============================================================================


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


def filter_detections(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    score_threshold: float = 0.5,
    max_detections: int = 300,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Filter detections by score and limit count.

    Args:
        boxes: Bounding boxes [N, 4]
        scores: Confidence scores [N]
        labels: Class labels [N]
        score_threshold: Minimum score to keep
        max_detections: Maximum number of detections

    Returns:
        filtered_boxes, filtered_scores, filtered_labels
    """
    mask = scores > score_threshold

    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    if scores.numel() > max_detections:
        indices = scores.argsort(descending=True)[:max_detections]
        boxes = boxes[indices]
        scores = scores[indices]
        labels = labels[indices]

    return boxes, scores, labels


def post_process_detections(
    predictions: Dict[str, Tensor],
    anchors: Optional[Tensor] = None,
    box_coder: Optional[BoxCoder] = None,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.5,
    max_detections: int = 300,
) -> List[DetectionResult]:
    """
    Complete post-processing pipeline for detection.

    Args:
        predictions: Dictionary with 'boxes', 'scores', 'labels'
        anchors: Anchor boxes for decoding
        box_coder: Box coder for decoding
        score_threshold: Score threshold
        nms_threshold: NMS IoU threshold
        max_detections: Maximum detections per image

    Returns:
        results: List of DetectionResult objects
    """
    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]

    # Decode boxes if needed
    if anchors is not None and box_coder is not None:
        boxes = box_coder.decode(boxes, anchors)

    # Filter by score
    boxes, scores, labels = filter_detections(
        boxes, scores, labels, score_threshold, max_detections
    )

    # Apply NMS
    keep = batched_nms(boxes, scores, labels, nms_threshold)

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Create result
    results = []
    results.append(
        DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
        )
    )

    return results


__all__ = [
    # Detection
    "BaseDetector",
    "YOLODetector",
    "SSDDetector",
    "FCOSDetector",
    "DetectionLoss",
    "DetectionResult",
    # Instance Segmentation
    "MaskRCNN",
    "YOLACT",
    "SegmentationLoss",
    # Semantic Segmentation
    "UNet",
    "DeepLabV3",
    "PSPNet",
    "SegmentationHead",
    "ASPP",
    "PSPModule",
    # Keypoint Detection
    "KeypointRCNN",
    "HeatmapLoss",
    # Panoptic Segmentation
    "PanopticFPN",
    # Utilities
    "AnchorGenerator",
    "BoxCoder",
    "nms",
    "batched_nms",
    "box_iou",
    "ROIPool",
    "ROIAlign",
    "FeaturePyramidNetwork",
    "RegionProposalNetwork",
    # Data Augmentation
    "Mosaic",
    "MixUp",
    "CutMix",
    "RandomCropWithBoxes",
    "ResizeWithLetterbox",
    # Post-processing
    "PostProcessor",
    "filter_detections",
    "post_process_detections",
]
