"""
Object Detection Heads

Advanced detection heads including RetinaNet, FCOS, YOLO-style,
and anchor-free detection heads for modern object detection.

References:
    - RetinaNet: https://arxiv.org/abs/1708.02002
    - FCOS: https://arxiv.org/abs/1904.01355
    - YOLOv5: https://github.com/ultralytics/yolov5
"""

from typing import List, Tuple, Optional, Dict, Any
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class RetinaNetHead(nn.Module):
    """
    RetinaNet Detection Head.

    Multi-scale feature pyramid detection with classification
    and regression branches per anchor.

    Args:
        in_channels: List of input channel dimensions per pyramid level
        num_anchors: Number of anchors per spatial location
        num_classes: Number of object classes
        feature_size: Intermediate feature size
    """

    def __init__(
        self,
        in_channels: List[int],
        num_anchors: int = 3,
        num_classes: int = 80,
        feature_size: int = 256,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for in_ch in in_channels:
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, feature_size, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature_size, feature_size, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, feature_size, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature_size, feature_size, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        self.cls_logits = nn.Conv2d(feature_size, num_anchors * num_classes, 1)
        self.bbox_pred = nn.Conv2d(feature_size, num_anchors * 4, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(
        self,
        features: List[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            features: List of [B, C, H, W] feature pyramid tensors
        Returns:
            Tuple of (classification logits, bounding box predictions)
        """
        cls_logits = []
        bbox_preds = []

        for i, feat in enumerate(features):
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)

            cls_logits.append(self.cls_logits(cls_feat))
            bbox_preds.append(self.bbox_pred(reg_feat))

        return cls_logits, bbox_preds


class FCOSHead(nn.Module):
    """
    FCOS (Fully Convolutional One-Stage) Detection Head.

    Anchor-free detection using center-ness and regression heads
    on fully convolutional feature maps.

    Args:
        in_channels: Input channel dimension
        num_classes: Number of object classes
        feature_size: Intermediate feature size
        num_convs: Number of convolutional layers per branch
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        feature_size: int = 256,
        num_convs: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        cls_subnet = []
        reg_subnet = []
        centerness_subnet = []

        for _ in range(num_convs):
            cls_subnet.append(nn.Conv2d(in_channels, feature_size, 3, padding=1))
            cls_subnet.append(nn.ReLU(inplace=True))

            reg_subnet.append(nn.Conv2d(in_channels, feature_size, 3, padding=1))
            reg_subnet.append(nn.ReLU(inplace=True))

            centerness_subnet.append(nn.Conv2d(in_channels, feature_size, 3, padding=1))
            centerness_subnet.append(nn.ReLU(inplace=True))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.reg_subnet = nn.Sequential(*reg_subnet)
        self.centerness_subnet = nn.Sequential(*centerness_subnet)

        self.cls_logits = nn.Conv2d(feature_size, num_classes, 1)
        self.bbox_pred = nn.Conv2d(feature_size, 4, 1)
        self.centerness = nn.Conv2d(feature_size, 1, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        features: List[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Args:
            features: List of [B, C, H, W] feature pyramid tensors
        Returns:
            Tuple of (classification logits, bbox predictions, centerness)
        """
        cls_logits = []
        bbox_preds = []
        centerness_preds = []

        for feat in features:
            cls_out = self.cls_subnet(feat)
            reg_out = self.reg_subnet(feat)
            cent_out = self.centerness_subnet(feat)

            cls_logits.append(self.cls_logits(cls_out))
            bbox_preds.append(self.bbox_pred(reg_out))
            centerness_preds.append(self.centerness(cent_out))

        return cls_logits, bbox_preds, centerness_preds


class YOLOHead(nn.Module):
    """
    YOLO Detection Head.

    Anchor-based detection head for YOLO-style detectors
    with detection predictions at multiple scales.

    Args:
        in_channels: List of input channel dimensions per scale
        num_classes: Number of object classes
        num_anchors: Number of anchors per location
        strides: List of feature map strides
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int = 80,
        num_anchors: int = 3,
        strides: List[int] = [8, 16, 32],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.strides = strides

        self.convs = nn.ModuleList()
        self_preds = nn.ModuleList()

        for in_ch, stride in zip(in_channels, strides):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, 3, padding=1),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch, in_ch, 3, padding=1),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                )
            )

            num_outputs = num_anchors * (5 + num_classes)
            self_preds.append(nn.Conv2d(in_ch, num_outputs, 1))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        features: List[Tensor],
    ) -> List[Tensor]:
        """
        Args:
            features: List of [B, C, H, W] feature tensors at different scales
        Returns:
            List of [B, num_anchors*(5+num_classes), H, W] predictions
        """
        outputs = []

        for feat, conv, pred in zip(features, self.convs, self_preds):
            x = conv(feat)
            out = pred(x)
            outputs.append(out)

        return outputs


class AnchorFreeHead(nn.Module):
    """
    Anchor-Free Detection Head.

    Simplified anchor-free detection without predefined anchors.
    Predicts classification and regression directly from feature maps.

    Args:
        in_channels: Input channel dimension
        num_classes: Number of object classes
        feature_size: Intermediate feature size
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        feature_size: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, feature_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, num_classes, 1),
        )

        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, feature_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, 4, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, C, H, W] feature tensor
        Returns:
            Tuple of (cls_logits [B, num_classes, H, W], bbox [B, 4, H, W])
        """
        cls_out = self.cls_conv(x)
        reg_out = self.reg_conv(x)

        return cls_out, reg_out


class DetectionOutput:
    """
    Container for detection outputs.

    Standardizes detection format across different detection heads.
    """

    def __init__(
        self,
        boxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        mask: Optional[Tensor] = None,
    ):
        self.boxes = boxes
        self.scores = scores
        self.labels = labels
        self.mask = mask

    def __len__(self) -> int:
        return len(self.boxes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "boxes": self.boxes,
            "scores": self.scores,
            "labels": self.labels,
        }
        if self.mask is not None:
            result["masks"] = self.mask
        return result


def decode_outputs(
    outputs: List[Tensor],
    strides: List[int],
    num_classes: int,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.5,
) -> List[DetectionOutput]:
    """
    Decode raw detection outputs to bounding boxes.

    Args:
        outputs: List of raw detection tensors
        strides: Strides for each output level
        num_classes: Number of classes
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold

    Returns:
        List of DetectionOutput objects
    """
    all_boxes = []
    all_scores = []
    all_labels = []

    for level_idx, out in enumerate(outputs):
        b, _, h, w = out.shape
        stride = strides[level_idx]

        out = out.view(b, 3, 5 + num_classes, h, w).permute(0, 1, 3, 4, 2)

        box_xy = torch.sigmoid(out[..., 0:2])
        box_wh = torch.sigmoid(out[..., 2:4])
        objectness = torch.sigmoid(out[..., 4:5])
        class_probs = torch.sigmoid(out[..., 5:])

        class_scores, class_labels = class_probs.max(dim=-1)

        final_scores = objectness.squeeze(-1) * class_scores

        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=out.device),
            torch.arange(w, device=out.device),
            indexing="ij",
        )

        grid = torch.stack([x_grid, y_grid], dim=-1).float()

        boxes = torch.cat(
            [
                (box_xy + grid.unsqueeze(0).unsqueeze(1)) * stride,
                box_wh * stride,
            ],
            dim=-1,
        )

        boxes = boxes.view(-1, 4)
        final_scores = final_scores.view(-1)
        class_labels = class_labels.view(-1)

        mask = final_scores > conf_threshold
        boxes = boxes[mask]
        final_scores = final_scores[mask]
        class_labels = class_labels[mask]

        all_boxes.append(boxes)
        all_scores.append(final_scores)
        all_labels.append(class_labels)

    if len(all_boxes) == 0:
        return [
            DetectionOutput(
                torch.empty(0, 4),
                torch.empty(0),
                torch.empty(0, dtype=torch.long),
            )
        ]

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return [DetectionOutput(all_boxes, all_scores, all_labels)]


def create_detection_head(
    head_type: str,
    in_channels: List[int],
    num_classes: int = 80,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create detection heads.

    Args:
        head_type: Type of detection head ('retinanet', 'fcos', 'yolo', 'anchorfree')
        in_channels: Input channel dimensions
        num_classes: Number of object classes
        **kwargs: Additional arguments for specific heads

    Returns:
        Detection head module

    Raises:
        ValueError: If head_type is not recognized
    """
    heads = {
        "retinanet": RetinaNetHead,
        "fcos": FCOSHead,
        "yolo": YOLOHead,
        "anchorfree": AnchorFreeHead,
    }

    if head_type.lower() not in heads:
        raise ValueError(
            f"Unknown head type: {head_type}. Available: {list(heads.keys())}"
        )

    if head_type.lower() == "anchorfree":
        in_channels = in_channels[0] if isinstance(in_channels, list) else in_channels

    return heads[head_type.lower()](in_channels, num_classes, **kwargs)
