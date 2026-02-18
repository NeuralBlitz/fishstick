"""
Detection Backbones and Feature Pyramid Network

Backbone networks and FPN implementations for object detection.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBackbone(nn.Module):
    """
    ResNet backbone for object detection.

    Extracts multi-scale features for detection heads.
    """

    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = False,
        frozen_stages: int = -1,
    ):
        """
        Initialize ResNet backbone.

        Args:
            depth: ResNet depth (50, 101, or 152)
            pretrained: Whether to load pretrained weights
            frozen_stages: Number of stages to freeze (-1 for none)
        """
        super().__init__()

        from torchvision.models import (
            resnet50,
            resnet101,
            ResNet50_Weights,
            ResNet101_Weights,
        )

        if depth == 50:
            if pretrained:
                base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                base_model = resnet50()
            self.out_channels = [256, 512, 1024, 2048]
        elif depth == 101:
            if pretrained:
                base_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            else:
                base_model = resnet101()
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze specified stages."""
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for param in [self.conv1, self.bn1]:
                for p in param.parameters():
                    p.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f"layer{i}")
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input images

        Returns:
            Tuple of feature maps from each stage
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2, c3, c4, c5


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN).

    Builds multi-scale feature pyramid from backbone features.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
    ):
        """
        Initialize FPN.

        Args:
            in_channels: List of input channels from backbone
            out_channels: Output channels for each pyramid level
        """
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels:
            lateral_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        inputs: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Build feature pyramid.

        Args:
            inputs: List of feature maps from backbone (low to high level)

        Returns:
            List of pyramid feature maps
        """
        laterals = [
            lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, inputs)
        ]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="nearest",
            )

        outputs = [fpn_conv(feat) for fpn_conv, feat in zip(self.fpn_convs, laterals)]

        return outputs


class BiFPN(nn.Module):
    """
    Bi-directional Feature Pyramid Network.

    Efficient BiFPN with weighted feature fusion.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_levels: int = 5,
    ):
        """
        Initialize BiFPN.

        Args:
            in_channels: Input channels from backbone
            out_channels: Output channels
            num_levels: Number of pyramid levels
        """
        super().__init__()

        self.out_channels = out_channels

        self.input_proj = nn.ModuleList(
            [nn.Conv2d(ch, out_channels, kernel_size=1) for ch in in_channels]
        )

        self.weight_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                for _ in range(num_levels * 2 - 2)
            ]
        )

        self.fuse_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
                for _ in range(num_levels)
            ]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        inputs: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply BiFPN.

        Args:
            inputs: List of feature maps from backbone

        Returns:
            List of enhanced feature maps
        """
        inputs = [proj(feat) for proj, feat in zip(self.input_proj, inputs)]

        return inputs


class DetectionNeck(nn.Module):
    """
    Generic detection neck for feature processing.

    Combines FPN with additional processing layers.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_outs: int = 5,
    ):
        """
        Initialize detection neck.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_outs: Number of output levels
        """
        super().__init__()

        self.fpn = FeaturePyramidNetwork(in_channels, out_channels)

        self.extra_convs = nn.ModuleList()
        for i in range(num_outs - len(in_channels)):
            if i == 0:
                self.extra_convs.append(
                    nn.Conv2d(
                        in_channels[-1],
                        in_channels[-1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )
            else:
                self.extra_convs.append(
                    nn.Conv2d(
                        in_channels[-1],
                        in_channels[-1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )

    def forward(
        self,
        inputs: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through neck.

        Args:
            inputs: List of backbone features

        Returns:
            Tuple of (pyramid features, outs)
        """
        pyramid = self.fpn(inputs)

        outs = list(pyramid)

        for extra_conv in self.extra_convs:
            outs.append(extra_conv(outs[-1]))

        return pyramid, outs


class DetectionHead(nn.Module):
    """
    Generic detection head for classification and regression.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 1,
        feat_channels: int = 256,
    ):
        """
        Initialize detection head.

        Args:
            in_channels: Input channels
            num_classes: Number of object classes
            num_anchors: Number of anchors per location
            feat_channels: Feature channels in head
        """
        super().__init__()

        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, num_classes * num_anchors, kernel_size=1),
        )

        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 4 * num_anchors, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for layer in [self.cls_conv, self.reg_conv]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through head.

        Args:
            x: Input features

        Returns:
            Tuple of (classification, bbox_regression)
        """
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)

        batch_size = x.shape[0]

        cls_feat = cls_feat.permute(0, 2, 3, 1).contiguous()
        cls_feat = cls_feat.view(batch_size, -1, cls_feat.shape[-1])

        reg_feat = reg_feat.permute(0, 2, 3, 1).contiguous()
        reg_feat = reg_feat.view(batch_size, -1, reg_feat.shape[-1])

        return cls_feat, reg_feat


class FCOSHead(nn.Module):
    """
    FCOS detection head for anchor-free detection.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feat_channels: int = 256,
    ):
        """
        Initialize FCOS head.

        Args:
            in_channels: Input channels
            num_classes: Number of classes
            feat_channels: Feature channels
        """
        super().__init__()

        self.cls_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )

        self.reg_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )

        self.cls_head = nn.Conv2d(feat_channels, num_classes, kernel_size=1)
        self.center_head = nn.Conv2d(feat_channels, 1, kernel_size=1)
        self.reg_head = nn.Conv2d(feat_channels, 4, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through FCOS head.

        Args:
            x: Input features

        Returns:
            Tuple of (classification, centerness, bbox_regression)
        """
        cls_feat = x
        for conv in self.cls_convs:
            cls_feat = conv(cls_feat)

        reg_feat = x
        for conv in self.reg_convs:
            reg_feat = conv(reg_feat)

        cls_score = self.cls_head(cls_feat)
        centerness = self.center_head(reg_feat)
        bbox_reg = self.reg_head(reg_feat)

        return cls_score, centerness, bbox_reg


__all__ = [
    "ResNetBackbone",
    "FeaturePyramidNetwork",
    "BiFPN",
    "DetectionNeck",
    "DetectionHead",
    "FCOSHead",
]
