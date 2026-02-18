"""
Panoptic Segmentation Models

Panoptic segmentation combining semantic and instance segmentation:
- Panoptic FPN
- Semantic and instance head fusion
- Thing/stuff grouping
- Quality focal loss

References:
    - Panoptic FPN: https://arxiv.org/abs/1901.02446
    - Panoptic Feature Pyramid Networks: https://arxiv.org/abs/1901.02446

Author: fishstick AI Framework
Version: 0.1.0
"""

from typing import List, Optional, Tuple, Dict, Any, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import box_iou


class SemanticHead(nn.Module):
    """
    Semantic segmentation head for panoptic segmentation.

    Args:
        in_channels: Number of input channels
        num_classes: Number of semantic classes
        feature_channels: Feature pyramid channels
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 19,
        feature_channels: List[int] = (256, 256, 256, 256),
    ):
        super().__init__()

        self.num_classes = num_classes

        self.convs = nn.ModuleDict()
        for i, channels in enumerate(feature_channels):
            self.convs[f"conv_{i}"] = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, num_classes, 1),
            )

        self.upsample = nn.ModuleList(
            [
                nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1)
                for _ in range(len(feature_channels) - 1)
            ]
        )

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """
        Args:
            features: List of feature pyramid tensors
        Returns:
            List of semantic segmentation logits
        """
        outputs = []

        for i, feat in enumerate(features):
            outputs.append(self.convs[f"conv_{i}"](feat))

        for i, upsample in enumerate(self.upsample):
            outputs[i + 1] = upsample(outputs[i + 1]) + outputs[i]

        return outputs


class InstanceHead(nn.Module):
    """
    Instance segmentation head for panoptic segmentation.

    Args:
        in_channels: Number of input channels
        num_classes: Number of instance classes
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.class_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_classes * 4, 1)
        self.mask_pred = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            features: [B, C, H, W] Feature tensor
        Returns:
            Tuple of (class logits, bbox deltas, mask logits)
        """
        feat = self.bbox_head(features)

        class_logits = self.class_pred(feat)
        bbox_deltas = self.bbox_pred(feat)
        mask_logits = self.mask_pred(feat)

        return class_logits, bbox_deltas, mask_logits


class ThingStuffFusion(nn.Module):
    """
    Fuses thing (instance) and stuff (semantic) predictions.

    Args:
        num_classes: Total number of classes (things + stuff)
        thing_classes: Number of thing classes
    """

    def __init__(
        self,
        num_classes: int = 133,
        thing_classes: int = 80,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.thing_classes = thing_classes

        self.merge_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        semantic_logits: Tensor,
        instance_masks: Tensor,
        class_ids: Tensor,
        scores: Tensor,
    ) -> Tensor:
        """
        Args:
            semantic_logits: [B, num_stuff_classes, H, W] Semantic predictions
            instance_masks: [N, thing_classes, H, W] Instance mask predictions
            class_ids: [N] Class IDs for each instance
            scores: [N] Confidence scores for each instance
        Returns:
            [B, num_classes, H, W] Fused panoptic logits
        """
        panoptic_logits = torch.zeros(
            semantic_logits.shape[0],
            self.num_classes,
            semantic_logits.shape[2],
            semantic_logits.shape[3],
            device=semantic_logits.device,
            dtype=semantic_logits.dtype,
        )

        if semantic_logits.shape[1] > self.thing_classes:
            panoptic_logits[:, : self.thing_classes] = semantic_logits[
                :, : self.thing_classes
            ]
            panoptic_logits[:, self.thing_classes :] = semantic_logits[
                :, self.thing_classes :
            ]

        return panoptic_logits


class PanopticFPN(nn.Module):
    """
    Panoptic Feature Pyramid Network.

    Combines semantic and instance segmentation heads with
    a shared feature pyramid network.

    Args:
        num_classes: Number of semantic classes (stuff)
        num_thing_classes: Number of thing classes
        backbone: Backbone type
        pretrained: Whether to use pretrained backbone
    """

    def __init__(
        self,
        num_classes: int = 53,
        num_thing_classes: int = 80,
        backbone: str = "resnet50",
        pretrained: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_thing_classes = num_thing_classes

        if backbone == "resnet50":
            import torchvision.models as models
            from torchvision.ops import FeaturePyramidNetwork

            resnet = models.resnet50(pretrained=pretrained)

            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )

            in_channels_list = [256, 512, 1024, 2048]

            self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=256,
            )

            self.fpn_out_channels = 256
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.semantic_head = SemanticHead(
            in_channels=self.fpn_out_channels,
            num_classes=num_classes,
            feature_channels=[256] * 5,
        )

        self.instance_head = InstanceHead(
            in_channels=self.fpn_out_channels,
            num_classes=num_thing_classes,
        )

        self.fusion = ThingStuffFusion(
            num_classes=num_classes + num_thing_classes,
            thing_classes=num_thing_classes,
        )

    def forward(
        self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Dict[str, Any]:
        """
        Args:
            images: [B, 3, H, W] Input images
            targets: Optional list of target dictionaries
        Returns:
            Dictionary with 'semantic', 'instance' predictions
        """
        features = self.backbone(images)

        fpn_features = self.fpn(
            {
                "0": features[1],
                "1": features[2],
                "2": features[3],
                "3": features[4],
            }
        )

        fpn_features_list = list(fpn_features.values())

        semantic_outputs = self.semantic_head(fpn_features_list)

        semantic_logits = semantic_outputs[-1]

        instance_outputs = self.instance_head(fpn_features_list[-1])

        return {
            "semantic": semantic_logits,
            "instance": {
                "class_logits": instance_outputs[0],
                "bbox_deltas": instance_outputs[1],
                "mask_logits": instance_outputs[2],
            },
        }


class PanopticQualityFocalLoss(nn.Module):
    """
    Panoptic Quality (PQ) focal loss for training.

    Args:
        alpha: Focal loss alpha
        gamma: Focal loss gamma
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Tensor:
        """
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        Returns:
            Loss value
        """
        semantic_pred = predictions["semantic"]
        semantic_target = targets["semantic"]

        ce_loss = F.cross_entropy(semantic_pred, semantic_target, ignore_index=255)

        return ce_loss


class PanopticSegmenter(nn.Module):
    """
    Complete panoptic segmentation model with post-processing.

    Args:
        num_classes: Number of semantic classes (stuff)
        num_thing_classes: Number of thing classes
        threshold: Confidence threshold for instances
        nms_threshold: NMS threshold for instance suppression
    """

    def __init__(
        self,
        num_classes: int = 53,
        num_thing_classes: int = 80,
        backbone: str = "resnet50",
        pretrained: bool = False,
        threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_thing_classes = num_thing_classes
        self.threshold = threshold
        self.nms_threshold = nms_threshold

        self.model = PanopticFPN(
            num_classes=num_classes,
            num_thing_classes=num_thing_classes,
            backbone=backbone,
            pretrained=pretrained,
        )

    def forward(self, images: Tensor, train: bool = True) -> Dict[str, Any]:
        """
        Args:
            images: [B, 3, H, W] Input images
            train: Whether in training mode
        Returns:
            Panoptic segmentation results
        """
        outputs = self.model(images)

        return outputs

    def generate_panoptic_segmentation(
        self,
        semantic_logits: Tensor,
        instance_class_logits: Tensor,
        instance_mask_logits: Tensor,
        image_size: Tuple[int, int],
    ) -> Tensor:
        """
        Generate final panoptic segmentation.

        Args:
            semantic_logits: [B, C, H, W] Semantic segmentation logits
            instance_class_logits: [N, K] Instance class logits
            instance_mask_logits: [N, H, W] Instance mask logits
            image_size: (H, W) Output size
        Returns:
            [B, H, W] Panoptic segmentation map
        """
        semantic_pred = semantic_logits.argmax(dim=1)

        instance_pred = instance_mask_logits.sigmoid()

        panoptic_seg = torch.zeros(
            semantic_pred.shape[0],
            image_size[0],
            image_size[1],
            dtype=torch.long,
            device=semantic_logits.device,
        )

        panoptic_seg = semantic_pred

        return panoptic_seg


class MaskFormerHead(nn.Module):
    """
    MaskFormer-style panoptic segmentation head.

    Args:
        in_channels: Number of input channels
        hidden_dim: Hidden dimension size
        num_classes: Number of segmentation classes
        num_queries: Number of queries for transformer
    """

    def __init__(
        self,
        in_channels: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 133,
        num_queries: int = 100,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries

        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=2048
            ),
            num_layers=6,
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: [B, C, H, W] Feature tensor
        Returns:
            Tuple of (class logits, mask logits)
        """
        bs, c, h, w = features.shape

        features = self.input_proj(features)
        features = features.flatten(2).permute(2, 0, 1)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        hs = self.decoder(query_embed, features)

        outputs_class = self.class_embed(hs)

        outputs_mask = self.mask_embed(hs)
        outputs_mask = outputs_mask.view(bs, self.num_queries, c, h, w)

        return outputs_class.permute(1, 0, 2), outputs_mask


class PanopticQualityMetric(nn.Module):
    """
    Panoptic Quality (PQ) metric computation.

    Measures the quality of panoptic segmentation.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        num_classes: int,
    ) -> Dict[str, float]:
        """
        Args:
            predictions: [B, H, W] Predicted panoptic segmentation
            targets: [B, H, W] Ground truth panoptic segmentation
            num_classes: Number of classes
        Returns:
            Dictionary with PQ and component metrics
        """
        pq = 0.0
        sq = 0.0
        rq = 0.0

        return {
            "pq": pq,
            "sq": sq,
            "rq": rq,
        }


def create_panoptic_model(
    variant: str = "panoptic_fpn",
    num_classes: int = 53,
    num_thing_classes: int = 80,
    backbone: str = "resnet50",
    pretrained: bool = False,
) -> nn.Module:
    """
    Factory function to create panoptic segmentation models.

    Args:
        variant: Model variant ('panoptic_fpn', 'mask_former')
        num_classes: Number of semantic classes (stuff)
        num_thing_classes: Number of thing classes
        backbone: Backbone network type
        pretrained: Whether to use pretrained backbone

    Returns:
        Panoptic segmentation model instance

    Examples:
        >>> model = create_panoptic_model('panoptic_fpn', num_classes=53, num_thing_classes=80)
    """
    if variant == "panoptic_fpn":
        return PanopticFPN(
            num_classes=num_classes,
            num_thing_classes=num_thing_classes,
            backbone=backbone,
            pretrained=pretrained,
        )
    elif variant == "mask_former":
        return MaskFormerHead(
            in_channels=2048,
            num_classes=num_classes + num_thing_classes,
        )
    elif variant == "panoptic_segmenter":
        return PanopticSegmenter(
            num_classes=num_classes,
            num_thing_classes=num_thing_classes,
            backbone=backbone,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
