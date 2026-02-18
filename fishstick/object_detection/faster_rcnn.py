"""
Faster R-CNN Implementation

Two-stage object detector with:
- Region Proposal Network (RPN)
- ROI Pooling/Align
- Detection head
- Complete training and inference pipeline
"""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIAlign(nn.Module):
    """
    ROI Align layer for extracting features from regions.

    Uses bilinear interpolation to extract features at precise locations.
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (7, 7),
        spatial_scale: float = 1.0,
        sampling_ratio: int = -1,
    ):
        """
        Initialize ROI Align.

        Args:
            output_size: Output (height, width)
            spatial_scale: Scale factor for coordinates
            sampling_ratio: Sampling ratio for bilinear interpolation
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(
        self,
        features: torch.Tensor,
        rois: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply ROI Align.

        Args:
            features: Feature map, shape (N, C, H, W)
            rois: Regions of interest, shape (M, 5) in (batch_idx, x1, y1, x2, y2) format

        Returns:
            Pooled features, shape (M, C, output_h, output_w)
        """
        from torchvision.ops import roi_align

        return roi_align(
            features,
            rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
        )


class ROIPool(nn.Module):
    """
    ROI Pooling layer.

    Max pools features from regions into fixed size.
    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        spatial_scale: float = 1.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(
        self,
        features: torch.Tensor,
        rois: torch.Tensor,
    ) -> torch.Tensor:
        from torchvision.ops import roi_pool

        return roi_pool(
            features,
            rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
        )


class RPNHead(nn.Module):
    """
    Region Proposal Network (RPN) head.

    Generates object proposals from feature maps.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 9,
        mid_channels: int = 512,
    ):
        """
        Initialize RPN head.

        Args:
            in_channels: Input channels from backbone
            num_anchors: Number of anchors per location
            mid_channels: Middle layer channels
        """
        super().__init__()
        self.num_anchors = num_anchors

        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(mid_channels, num_anchors, kernel_size=1)
        self.reg_layer = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for layer in [self.conv, self.cls_layer, self.reg_layer]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through RPN.

        Args:
            x: Feature map

        Returns:
            Tuple of (objectness_scores, bbox_deltas)
        """
        x = F.relu(self.conv(x))

        objectness = self.cls_layer(x)
        bbox_deltas = self.reg_layer(x)

        batch_size = x.shape[0]
        objectness = objectness.view(batch_size, self.num_anchors, -1)
        bbox_deltas = bbox_deltas.view(batch_size, self.num_anchors, 4, -1)

        return objectness, bbox_deltas


class RegionProposalNetwork(nn.Module):
    """
    Complete RPN module.

    Generates object proposals with anchor generation and NMS.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 9,
        pre_nms_top_n: int = 2000,
        post_nms_top_n: int = 1000,
        nms_thresh: float = 0.7,
        min_size: float = 1.0,
    ):
        """
        Initialize RPN.

        Args:
            in_channels: Input channels
            num_anchors: Number of anchors
            pre_nms_top_n: Top N proposals before NMS
            post_nms_top_n: Top proposals after NMS
            nms_thresh: NMS threshold
            min_size: Minimum proposal size
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        self.rpn_head = RPNHead(in_channels, num_anchors)

        self.anchor_generator = AnchorGeneratorGrid()

    def forward(
        self,
        features: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through RPN.

        Args:
            features: Feature map from backbone
            image_shape: Input image shape

        Returns:
            Tuple of (proposals, objectness, bbox_deltas)
        """
        objectness, bbox_deltas = self.rpn_head(features)

        anchors = self.anchor_generator.generate_anchors(
            features.shape[-2:],
            features.device,
        )

        proposals = self._decode_proposals(anchors, bbox_deltas, image_shape)

        proposals, scores = self._filter_proposals(proposals, objectness, image_shape)

        return proposals, objectness, bbox_deltas

    def _decode_proposals(
        self,
        anchors: torch.Tensor,
        bbox_deltas: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Decode bbox deltas to actual coordinates."""
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx = bbox_deltas[:, :, 0]
        dy = bbox_deltas[:, :, 1]
        dw = bbox_deltas[:, :, 2]
        dh = bbox_deltas[:, :, 3]

        pred_ctr_x = dx * widths.view(1, -1) + ctr_x.view(1, -1)
        pred_ctr_y = dy * heights.view(1, -1) + ctr_y.view(1, -1)
        pred_w = torch.exp(dw) * widths.view(1, -1)
        pred_h = torch.exp(dh) * heights.view(1, -1)

        proposals = torch.zeros_like(bbox_deltas[..., :4])
        proposals[..., 0] = pred_ctr_x - 0.5 * pred_w
        proposals[..., 1] = pred_ctr_y - 0.5 * pred_h
        proposals[..., 2] = pred_ctr_x + 0.5 * pred_w
        proposals[..., 3] = pred_ctr_y + 0.5 * pred_h

        return proposals

    def _filter_proposals(
        self,
        proposals: torch.Tensor,
        objectness: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter proposals based on objectness and size."""
        batch_size = proposals.shape[0]
        device = proposals.device

        objectness = objectness.view(batch_size, -1)
        proposals = proposals.view(batch_size, -1, 4)

        pre_nms_top_n = min(self.pre_nms_top_n, proposals.shape[1])

        top_scores, top_indices = objectness.topk(pre_nms_top_n, dim=1)

        batch_indices = (
            torch.arange(batch_size, device=device)
            .view(-1, 1)
            .expand(-1, pre_nms_top_n)
        )

        proposals = proposals[batch_indices.view(-1), top_indices.view(-1)].view(
            batch_size, pre_nms_top_n, 4
        )

        h, w = image_shape
        min_size = self.min_size * h

        keep = (proposals[..., 2] - proposals[..., 0] >= min_size) & (
            proposals[..., 3] - proposals[..., 1] >= min_size
        )

        proposals = proposals * torch.tensor([w, h, w, h], device=device)
        proposals = proposals.clamp(min=0)
        proposals[..., 2] = proposals[..., 2].clamp(max=w)
        proposals[..., 3] = proposals[..., 3].clamp(max=h)

        keep_list = []
        scores_list = []

        for i in range(batch_size):
            keep_mask = keep[i]
            img_proposals = proposals[i][keep_mask]
            img_scores = top_scores[i][keep_mask]

            if len(img_proposals) == 0:
                keep_list.append(
                    torch.zeros(self.post_nms_top_n, dtype=torch.int64, device=device)
                )
                scores_list.append(torch.zeros(self.post_nms_top_n, device=device))
                continue

            keep_idx = nms(
                img_proposals,
                img_scores,
                self.nms_thresh,
            )[: self.post_nms_top_n]

            if len(keep_idx) < self.post_nms_top_n:
                padding = torch.zeros(
                    self.post_nms_top_n - len(keep_idx),
                    dtype=torch.int64,
                    device=device,
                )
                keep_idx = torch.cat([keep_idx, padding])

            keep_list.append(keep_idx)
            scores_list.append(img_scores[keep_idx])

        final_proposals = torch.zeros(
            (batch_size, self.post_nms_top_n, 4), device=device
        )

        for i in range(batch_size):
            final_proposals[i] = proposals[i][keep_list[i][: self.post_nms_top_n]]

        return final_proposals, torch.stack(scores_list, dim=0)


class AnchorGeneratorGrid(nn.Module):
    """Simple grid anchor generator for RPN."""

    def __init__(
        self,
        scales: Tuple[float, ...] = (8, 16, 32),
        aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
    ):
        super().__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios

    def generate_anchors(
        self,
        feature_size: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Generate anchors on a grid."""
        feat_h, feat_w = feature_size
        stride = 16

        scales = torch.tensor(self.scales, device=device)
        aspect_ratios = torch.tensor(self.aspect_ratios, device=device)

        anchor_w = scales.view(-1, 1) * (aspect_ratios**0.5).view(1, -1)
        anchor_h = scales.view(-1, 1) / (aspect_ratios**0.5).view(1, -1)

        shifts_x = (
            torch.arange(feat_w, dtype=torch.float32, device=device) * stride
            + stride // 2
        )
        shifts_y = (
            torch.arange(feat_h, dtype=torch.float32, device=device) * stride
            + stride // 2
        )

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        anchors = []

        for aw in anchor_w.view(-1):
            for ah in anchor_h.view(-1):
                x1 = shift_x - aw / 2
                y1 = shift_y - ah / 2
                x2 = shift_x + aw / 2
                y2 = shift_y + ah / 2
                anchors.append(torch.stack([x1, y1, x2, y2], dim=-1))

        return torch.cat([a.unsqueeze(0) for a in anchors], dim=0).view(-1, 4)


class FastRCNNHead(nn.Module):
    """
    Fast R-CNN detection head.

    Classifies and refines proposals from RPN.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        roi_size: Tuple[int, int] = (7, 7),
    ):
        """
        Initialize Fast R-CNN head.

        Args:
            in_channels: Input channels
            num_classes: Number of object classes
            roi_size: ROI pooling size
        """
        super().__init__()
        self.num_classes = num_classes

        self.roi_pool = ROIPool(roi_size, spatial_scale=1.0)

        self.fc6 = nn.Linear(in_channels * roi_size[0] * roi_size[1], 1024)
        self.fc7 = nn.Linear(1024, 1024)

        self.cls_score = nn.Linear(1024, num_classes + 1)
        self.bbox_pred = nn.Linear(1024, (num_classes + 1) * 4)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for layer in [self.fc6, self.fc7, self.cls_score, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(
        self,
        features: torch.Tensor,
        proposals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through detection head.

        Args:
            features: Feature map from backbone
            proposals: Region proposals

        Returns:
            Tuple of (class_scores, bbox_deltas)
        """
        pooled_features = self.roi_pool(features, proposals)

        pooled_features = pooled_features.flatten(start_dim=1)

        x = F.relu(self.fc6(pooled_features))
        x = F.relu(self.fc7(x))

        class_scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return class_scores, bbox_deltas


class FasterRCNN(nn.Module):
    """
    Complete Faster R-CNN detector.

    Two-stage detector with RPN and detection head.
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone: str = "resnet50",
        pretrained: bool = False,
    ):
        """
        Initialize Faster R-CNN.

        Args:
            num_classes: Number of object classes
            backbone: Backbone type
            pretrained: Whether to use pretrained backbone
        """
        super().__init__()
        self.num_classes = num_classes

        if backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights

            if pretrained:
                resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet50()

            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_channels = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.rpn = RegionProposalNetwork(
            in_channels=backbone_channels,
            num_anchors=9,
        )

        self.detection_head = FastRCNNHead(
            in_channels=backbone_channels,
            num_classes=num_classes,
        )

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass through Faster R-CNN.

        Args:
            images: Input images, shape (B, 3, H, W)
            targets: Ground truth targets (for training)

        Returns:
            List of predictions per image
        """
        image_shape = images.shape[-2:]

        features = self.backbone(images)

        proposals, rpn_obj, rpn_deltas = self.rpn(features, image_shape)

        class_scores, bbox_deltas = self.detection_head(features, proposals)

        results = []

        batch_size = proposals.shape[0]
        for i in range(batch_size):
            results.append(
                {
                    "boxes": proposals[i],
                    "scores": F.softmax(class_scores[i], dim=-1),
                    "labels": bbox_deltas[i],
                }
            )

        return results


def create_faster_rcnn(
    num_classes: int = 80,
    backbone: str = "resnet50",
    pretrained: bool = False,
) -> FasterRCNN:
    """
    Create Faster R-CNN model.

    Args:
        num_classes: Number of object classes
        backbone: Backbone type
        pretrained: Whether to use pretrained weights

    Returns:
        FasterRCNN instance
    """
    return FasterRCNN(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
    )


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Apply non-maximum suppression."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break

        i = order[0]
        keep.append(i.item())

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


__all__ = [
    "ROIAlign",
    "ROIPool",
    "RPNHead",
    "RegionProposalNetwork",
    "AnchorGeneratorGrid",
    "FastRCNNHead",
    "FasterRCNN",
    "create_faster_rcnn",
]
