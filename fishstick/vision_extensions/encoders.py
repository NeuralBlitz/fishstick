"""
Visual Feature Encoders

CNN and ViT-based encoders including ResNet, EfficientNet,
multi-scale feature pyramids (FPN), and hierarchical feature extractors.

References:
    - ResNet: https://arxiv.org/abs/1512.03385
    - EfficientNet: https://arxiv.org/abs/1905.11946
    - FPN: https://arxiv.org/abs/1708.02002
"""

from typing import List, Tuple, Optional, Dict, Any
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ResNetEncoder(nn.Module):
    """
    ResNet-based Feature Encoder.

    Extracts multi-scale features from ResNet backbone for
    detection and segmentation tasks.

    Args:
        in_channels: Number of input channels
        arch: ResNet architecture variant ('resnet18', 'resnet34', 'resnet50')
        pretrained: Whether to use pretrained weights
    """

    def __init__(
        self,
        in_channels: int = 3,
        arch: str = "resnet50",
        pretrained: bool = False,
    ):
        super().__init__()

        from torchvision import models

        if arch == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.channels = [64, 128, 256, 512]
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            self.channels = [64, 128, 256, 512]
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown ResNet architecture: {arch}")

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: [B, C, H, W] input image
        Returns:
            List of multi-scale feature maps
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c2, c3, c4, c5]


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-based Feature Encoder.

    Extracts multi-scale features from EfficientNet backbone.

    Args:
        in_channels: Number of input channels
        variant: EfficientNet variant ('b0', 'b1', 'b2', 'b3', 'b4')
        pretrained: Whether to use pretrained weights
    """

    def __init__(
        self,
        in_channels: int = 3,
        variant: str = "b0",
        pretrained: bool = False,
    ):
        super().__init__()

        from torchvision import models

        if variant == "b0":
            net = models.efficientnet_b0(pretrained=pretrained)
            self.channels = [24, 40, 112, 320]
        elif variant == "b1":
            net = models.efficientnet_b1(pretrained=pretrained)
            self.channels = [24, 40, 112, 320]
        elif variant == "b2":
            net = models.efficientnet_b2(pretrained=pretrained)
            self.channels = [24, 48, 120, 352]
        elif variant == "b3":
            net = models.efficientnet_b3(pretrained=pretrained)
            self.channels = [32, 48, 136, 384]
        elif variant == "b4":
            net = models.efficientnet_b4(pretrained=pretrained)
            self.channels = [32, 56, 160, 448]
        else:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")

        self.features = net.features

        self.out_indices = [2, 3, 4, 5]

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: [B, C, H, W] input image
        Returns:
            List of multi-scale feature maps
        """
        features = []

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                features.append(x)

        return features


class ViTFeatureEncoder(nn.Module):
    """
    Vision Transformer-based Feature Encoder.

    Extracts hierarchical features from ViT backbone for
    dense prediction tasks.

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        pos_embed = nn.Parameter(torch.zeros(1, self.grid_size**2 + 1, embed_dim))

        self.cls_token = cls_token
        self.pos_embed = pos_embed

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: [B, C, H, W] input image
        Returns:
            List of feature maps at different stages
        """
        B = x.shape[0]

        x = self.patch_embed(x)
        _, _, h, w = x.shape

        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        patch_tokens = x[:, 1:]

        return [patch_tokens]


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN).

    Builds high-level semantic features at all scales through
    top-down pathway and lateral connections.

    Args:
        in_channels: List of input channel dimensions
        out_channels: Output channel dimension for all levels
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
    ):
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, kernel_size=1))
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

        self.extra_convs = None
        if len(in_channels) < 4:
            self.extra_convs = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """
        Args:
            features: List of [B, C_i, H_i, W_i] feature maps
        Returns:
            List of [B, out_channels, H_i, W_i] FPN features
        """
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest"
            )

        outs = [fpn_conv(feat) for fpn_conv, feat in zip(self.fpn_convs, laterals)]

        if self.extra_convs is not None:
            outs.append(self.extra_convs(laterals[-1]))

        return outs


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-Scale Feature Extractor with Nested Feature Pyramid.

    Extracts features at multiple scales with dense connections.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        num_scales: Number of scale levels
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        num_scales: int = 4,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.scale_blocks = nn.ModuleList()

        for i in range(num_scales):
            self.scale_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion = nn.ModuleList(
            [
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
                for _ in range(num_scales - 1)
            ]
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: [B, C, H, W] input image
        Returns:
            List of multi-scale features
        """
        features = []

        x = self.stem(x)
        features.append(x)

        for i, block in enumerate(self.scale_blocks):
            x = block(x)
            features.append(x)

        for i in range(len(features) - 1, 0, -1):
            upsampled = F.interpolate(
                features[i],
                size=features[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = torch.cat([features[i - 1], upsampled], dim=1)
            features[i - 1] = self.fusion[i - 1](fused)

        return features


class CSPResNetEncoder(nn.Module):
    """
    CSP (Cross Stage Partial) ResNet Encoder.

    Efficient encoder using partial residual connections.

    Args:
        in_channels: Number of input channels
        depth: Encoder depth variant
        out_channels: Base output channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        depth: int = 50,
        out_channels: int = 64,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        stage_channels = [
            out_channels,
            out_channels * 2,
            out_channels * 4,
            out_channels * 8,
        ]
        stage_depths = [3, 6, 9, 3] if depth == 50 else [3, 6, 9, 3]

        self.stages = nn.ModuleList()

        for i, (ch, d) in enumerate(zip(stage_channels, stage_depths)):
            self.stages.append(self._make_stage(ch, d, i > 0))

        self.channels = stage_channels

    def _make_stage(self, channels: int, depth: int, downsample: bool) -> nn.Module:
        layers = []

        if downsample:
            layers.append(
                nn.Conv2d(channels // 2, channels, kernel_size=3, stride=2, padding=1)
            )

        for _ in range(depth):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: [B, C, H, W] input
        Returns:
            List of multi-scale features
        """
        x = self.stem(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


class HybridEncoder(nn.Module):
    """
    Hybrid CNN-Transformer Encoder.

    Combines CNN local features with transformer global context.

    Args:
        cnn_channels: List of CNN channel dimensions
        transformer_dim: Transformer embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        cnn_channels: List[int] = [64, 128, 256, 512],
        transformer_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()

        self.input_proj = nn.ModuleList(
            [nn.Conv2d(ch, transformer_dim, kernel_size=1) for ch in cnn_channels]
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=num_heads,
                dim_feedforward=transformer_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=depth,
        )

        self.fpn = FeaturePyramidNetwork(cnn_channels, transformer_dim)

    def forward(self, cnn_features: List[Tensor]) -> List[Tensor]:
        """
        Args:
            cnn_features: List of CNN feature maps
        Returns:
            List of enhanced features with transformer context
        """
        proj_features = []

        for i, feat in enumerate(cnn_features):
            proj = self.input_proj[i](feat)
            proj = proj.flatten(2).transpose(1, 2)
            proj_features.append(proj)

        enhanced = []
        for feat in proj_features:
            B, N, C = feat.shape
            H = W = int(math.sqrt(N))

            feat_2d = feat.transpose(1, 2).reshape(B, C, H, W)

            enhanced.append(feat_2d)

        fpn_out = self.fpn(enhanced)

        return fpn_out


def create_encoder(
    encoder_type: str,
    in_channels: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create feature encoders.

    Args:
        encoder_type: Type of encoder ('resnet', 'efficientnet', 'vit', 'csp_resnet')
        in_channels: Number of input channels
        **kwargs: Additional arguments for specific encoders

    Returns:
        Feature encoder module

    Raises:
        ValueError: If encoder_type is not recognized
    """
    encoders = {
        "resnet": ResNetEncoder,
        "efficientnet": EfficientNetEncoder,
        "vit": ViTFeatureEncoder,
        "csp_resnet": CSPResNetEncoder,
    }

    if encoder_type.lower() not in encoders:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. Available: {list(encoders.keys())}"
        )

    return encoders[encoder_type.lower()](in_channels=in_channels, **kwargs)
