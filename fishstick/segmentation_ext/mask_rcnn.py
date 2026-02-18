"""
Mask R-CNN for Instance Segmentation

Implementation of Mask R-CNN with:
- Region Proposal Network (RPN)
- RoIAlign for feature extraction
- Mask head for instance masks
- Bounding box head

References:
    - Mask R-CNN: https://arxiv.org/abs/1703.06870

Author: fishstick AI Framework
Version: 0.1.0
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import nms, box_iou


class RPNHead(nn.Module):
    """
    Region Proposal Network head.

    Args:
        in_channels: Number of input channels
        num_anchors: Number of anchor boxes per location
        anchor_sizes: List of anchor box sizes
        anchor_ratios: List of anchor aspect ratios
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_anchors: int = 3,
        anchor_sizes: List[int] = (32, 64, 128, 256, 512),
        anchor_ratios: List[float] = (0.5, 1.0, 2.0),
    ):
        super().__init__()

        self.num_anchors = num_anchors

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * len(anchor_ratios), kernel_size=1
        )
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        self._normal_init(self.cls_logits, 0, 0.01)
        self._normal_init(self.bbox_pred, 0, 0.01)

    def _normal_init(self, m: nn.Module, mean: float = 0, std: float = 0.01) -> None:
        nn.init.normal_(m.weight, mean, std)
        nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, C, H, W] Feature map
        Returns:
            Tuple of (objectness logits, bounding box deltas)
        """
        logits = []
        bbox_reg = []

        feat = F.relu(self.conv(x))

        logits.append(self.cls_logits(feat))
        bbox_reg.append(self.bbox_pred(feat))

        return logits[0], bbox_reg[0]


class AnchorGenerator(nn.Module):
    """
    Anchor generator for RPN.

    Args:
        sizes: Anchor box sizes
        aspect_ratios: Anchor aspect ratios
    """

    def __init__(
        self,
        sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
        aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
    ):
        super().__init__()

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None

    def generate_anchors(
        self,
        scales: Tuple[int, ...],
        aspect_ratios: Tuple[float, ...],
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """
        Generate anchor templates.
        """
        scales = torch.as_tensor(scales, dtype=dtype)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype)

        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def forward(
        self, feature_maps: List[Tensor], image_size: Tuple[int, int]
    ) -> List[Tensor]:
        """
        Args:
            feature_maps: List of feature maps at different scales
            image_size: (H, W) of input image
        Returns:
            List of anchor tensors
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = tuple(image_size)

        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [int(image_size[0] / g[0]), int(image_size[1] / g[1])] for g in grid_sizes
        ]

        anchors = []

        for size, stride, grid_size in zip(self.sizes, strides, grid_sizes):
            cell_anchors = self.generate_anchors((size,), self.aspect_ratios, dtype).to(
                device
            )

            grid_x = torch.arange(grid_size[1], dtype=dtype, device=device)
            grid_y = torch.arange(grid_size[0], dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

            shift_x = (grid_x + 0.5) * stride[1]
            shift_y = (grid_y + 0.5) * stride[0]

            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + cell_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors


class RoIAlign(nn.Module):
    """
    Region of Interest Align pooling.

    Args:
        output_size: (H, W) output size
        spatial_scale: Scale factor for coordinates
        sampling_ratio: Sampling ratio for bilinear interpolation
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (7, 7),
        spatial_scale: float = 1.0,
        sampling_ratio: int = 2,
    ):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, features: Tensor, boxes: Tensor) -> Tensor:
        """
        Args:
            features: [B, C, H, W] Feature map
            boxes: [N, 4] Bounding boxes (x1, y1, x2, y2)
        Returns:
            [N, C, output_H, output_W] Pooled features
        """
        from torchvision.ops import roi_align

        return roi_align(
            features,
            [boxes],
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
        )


class MaskRCNNPredictor(nn.Module):
    """
    Mask R-CNN box and mask predictor heads.

    Args:
        in_channels: Number of input channels
        num_classes: Number of classes
        conv_channels: Number of channels in intermediate layers
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 91,
        conv_channels: int = 1024,
    ):
        super().__init__()

        self.fc1 = nn.Linear(in_channels * 7 * 7, conv_channels)
        self.fc2 = nn.Linear(conv_channels, conv_channels)

        self.class_pred = nn.Linear(conv_channels, num_classes)
        self.box_pred = nn.Linear(conv_channels, num_classes * 4)

        self.mask_fcn1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.mask_fcn2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.mask_fcn3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.mask_fcn4 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        self.mask_pred = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x: Tensor, proposals: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: [N, C, 7, 7] RoI features
            proposals: [N, 4] Bounding box proposals
        Returns:
            Tuple of (class logits, box deltas, mask logits)
        """
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        class_logits = self.class_pred(x)
        box_regression = self.box_pred(x)

        mask_features = x.view(-1, 256, 7, 7)

        mask_features = F.relu(self.mask_fcn1(mask_features))
        mask_features = F.relu(self.mask_fcn2(mask_features))
        mask_features = F.relu(self.mask_fcn3(mask_features))
        mask_features = F.relu(self.mask_fcn4(mask_features))

        mask_logits = self.mask_pred(mask_features)

        return class_logits, box_regression, mask_logits


class MaskRCNN(nn.Module):
    """
    Mask R-CNN for instance segmentation.

    Args:
        num_classes: Number of classes (including background)
        backbone: Backbone network type
        pretrained: Whether to use pretrained backbone
    """

    def __init__(
        self,
        num_classes: int = 91,
        backbone: str = "resnet50",
        pretrained: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes

        if backbone == "resnet50":
            import torchvision.models as models

            backbone_model = models.resnet50(pretrained=pretrained)
            self.feature_channels = 2048
            self.backbone_channels = [256, 512, 1024, 2048]
        else:
            import torchvision.models as models

            backbone_model = models.resnet101(pretrained=pretrained)
            self.feature_channels = 2048
            self.backbone_channels = [256, 512, 1024, 2048]

        self.backbone = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool,
            backbone_model.layer1,
            backbone_model.layer2,
            backbone_model.layer3,
            backbone_model.layer4,
        )

        self.rpn = RPNHead(in_channels=512)

        self.anchor_generator = AnchorGenerator()

        self.roi_pool = RoIAlign(output_size=(7, 7), spatial_scale=1.0 / 16)

        self.predictor = MaskRCNNPredictor(in_channels=256, num_classes=num_classes)

    def forward(
        self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> List[Dict[str, Tensor]]:
        """
        Args:
            images: [B, 3, H, W] Input images
            targets: Optional list of target dictionaries
        Returns:
            List of predictions with 'boxes', 'labels', 'scores', 'masks'
        """
        features = self.backbone(images)

        features = list(features.values())
        feature = features[-1]

        objectness, pred_bbox_deltas = self.rpn(feature)

        anchors = self.anchor_generator([feature], images.shape[-2:])
        anchors = anchors[0]

        proposals = self._decode_boxes(anchors, pred_bbox_deltas)

        proposals = self._clip_boxes(proposals, images.shape[-2:])

        proposals, scores = self._filter_proposals(
            proposals, objectness, images.shape[-2:]
        )

        roi_features = self.roi_pool(feature, proposals)

        class_logits, box_regression, mask_logits = self.predictor(
            roi_features, proposals
        )

        proposals = self._decode_boxes(proposals.unsqueeze(0), box_regression)

        results = []

        boxes = proposals[0]
        labels = class_logits.argmax(dim=-1)
        scores = F.softmax(class_logits, dim=-1).max(dim=-1)[0]
        masks = mask_logits[0].sigmoid()

        results.append(
            {
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "masks": masks,
            }
        )

        return results

    def _decode_boxes(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        """Apply deltas to anchors."""
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        dx = dx * widths[:, None]
        dy = dy * heights[:, None]
        dw = dw * widths[:, None]
        dh = dh * heights[:, None]

        ctr_x = ctr_x[:, None] + dx
        ctr_y = ctr_y[:, None] + dy
        widths = widths[:, None] * torch.exp(dw)
        heights = heights[:, None] * torch.exp(dh)

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = ctr_x - 0.5 * widths
        pred_boxes[:, 1::4] = ctr_y - 0.5 * heights
        pred_boxes[:, 2::4] = ctr_x + 0.5 * widths
        pred_boxes[:, 3::4] = ctr_y + 0.5 * heights

        return pred_boxes

    def _clip_boxes(self, boxes: Tensor, image_size: Tuple[int, int]) -> Tensor:
        """Clip boxes to image boundaries."""
        boxes[:, 0::4].clamp_(min=0, max=image_size[1])
        boxes[:, 1::4].clamp_(min=0, max=image_size[0])
        boxes[:, 2::4].clamp_(min=0, max=image_size[1])
        boxes[:, 3::4].clamp_(min=0, max=image_size[0])
        return boxes

    def _filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_size: Tuple[int, int],
        pre_nms_top_n: int = 2000,
        post_nms_top_n: int = 1000,
        nms_thresh: float = 0.7,
    ) -> Tuple[Tensor, Tensor]:
        """Filter proposals using NMS."""
        objectness = objectness.flatten(1).sigmoid()

        top_indices = objectness.topk(pre_nms_top_n, dim=1)[1]

        batch_indices = (
            torch.arange(proposals.shape[0], device=proposals.device)
            .unsqueeze(1)
            .expand_as(top_indices)
        )

        proposals = proposals[batch_indices, top_indices]
        objectness = objectness[batch_indices, top_indices]

        final_boxes = []
        final_scores = []

        for boxes, scores in zip(proposals, objectness):
            boxes = self._clip_boxes(boxes, image_size)

            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            scores = scores[keep]

            if len(boxes) > 0:
                keep_nms = nms(boxes, scores, nms_thresh)[:post_nms_top_n]
                final_boxes.append(boxes[keep_nms])
                final_scores.append(scores[keep_nms])
            else:
                final_boxes.append(boxes)
                final_scores.append(scores)

        return final_boxes[0], final_scores[0]


class MaskRCNNResNetFPN(nn.Module):
    """
    Mask R-CNN with ResNet-FPN backbone.

    Args:
        num_classes: Number of classes
        pretrained: Whether to use pretrained backbone
    """

    def __init__(
        self,
        num_classes: int = 91,
        pretrained: bool = False,
    ):
        super().__init__()

        import torchvision.models as models
        from torchvision.ops import FeaturePyramidNetwork

        resnet = models.resnet50(pretrained=pretrained)

        self.body = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
        )

        self.rpn = RPNHead(in_channels=256)

        self.anchor_generator = AnchorGenerator()

        self.roi_pool = RoIAlign(output_size=(7, 7), spatial_scale=1.0 / 16)

        self.predictor = MaskRCNNPredictor(in_channels=256, num_classes=num_classes)

    def forward(
        self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> List[Dict[str, Tensor]]:
        """
        Args:
            images: [B, 3, H, W] Input images
            targets: Optional list of targets
        Returns:
            List of predictions
        """
        features = self.body(images)

        fpn_features = self.fpn(
            {
                "0": features[1],
                "1": features[2],
                "2": features[3],
                "3": features[4],
            }
        )

        fpn_features_list = list(fpn_features.values())

        objectness, pred_bbox_deltas = self.rpn(fpn_features_list[0])

        anchors = self.anchor_generator([fpn_features_list[0]], images.shape[-2:])

        results = self._simple_forward(
            images, fpn_features_list[-1], anchors[0], objectness, pred_bbox_deltas
        )

        return results

    def _simple_forward(
        self,
        images: Tensor,
        features: Tensor,
        anchors: Tensor,
        objectness: Tensor,
        pred_bbox_deltas: Tensor,
    ) -> List[Dict[str, Tensor]]:
        """Simplified forward pass."""
        proposals = self._decode_boxes(anchors, pred_bbox_deltas)
        proposals = self._clip_boxes(proposals, images.shape[-2:])

        objectness = objectness.flatten(1).sigmoid()
        scores = objectness.max(dim=1)[0]

        keep = scores > 0.5
        proposals = proposals[keep]
        scores = scores[keep]

        if len(proposals) > 1000:
            keep_top = scores.topk(1000)[1]
            proposals = proposals[keep_top]
            scores = scores[keep_top]

        roi_features = self.roi_pool(features, proposals)

        class_logits, box_regression, mask_logits = self.predictor(
            roi_features, proposals
        )

        masks = mask_logits.sigmoid()

        return [
            {
                "boxes": proposals,
                "labels": class_logits.argmax(dim=-1),
                "scores": F.softmax(class_logits, dim=-1).max(dim=-1)[0],
                "masks": masks,
            }
        ]

    def _decode_boxes(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4] * widths[:, None]
        dy = deltas[:, 1::4] * heights[:, None]
        dw = deltas[:, 2::4] * widths[:, None]
        dh = deltas[:, 3::4] * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = ctr_x[:, None] + dx - torch.exp(dw) / 2
        pred_boxes[:, 1::4] = ctr_y[:, None] + dy - torch.exp(dh) / 2
        pred_boxes[:, 2::4] = ctr_x[:, None] + dx + torch.exp(dw) / 2
        pred_boxes[:, 3::4] = ctr_y[:, None] + dy + torch.exp(dh) / 2

        return pred_boxes

    def _clip_boxes(self, boxes: Tensor, image_size: Tuple[int, int]) -> Tensor:
        boxes[:, 0::4].clamp_(min=0, max=image_size[1])
        boxes[:, 1::4].clamp_(min=0, max=image_size[0])
        boxes[:, 2::4].clamp_(min=0, max=image_size[1])
        boxes[:, 3::4].clamp_(min=0, max=image_size[0])
        return boxes


def create_mask_rcnn(
    variant: str = "mask_rcnn_resnet_fpn",
    num_classes: int = 91,
    pretrained: bool = False,
) -> nn.Module:
    """
    Factory function to create Mask R-CNN models.

    Args:
        variant: Model variant
        num_classes: Number of classes
        pretrained: Whether to use pretrained backbone

    Returns:
        Mask R-CNN model instance

    Examples:
        >>> model = create_mask_rcnn('mask_rcnn_resnet_fpn', num_classes=80)
    """
    if variant == "mask_rcnn":
        return MaskRCNN(num_classes, pretrained=pretrained)
    elif variant == "mask_rcnn_resnet_fpn":
        return MaskRCNNResNetFPN(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown variant: {variant}")
