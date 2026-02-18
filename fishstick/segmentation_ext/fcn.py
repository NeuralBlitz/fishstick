"""
Fully Convolutional Networks (FCN) for Semantic Segmentation

Implementation of FCN-8s, FCN-16s, FCN-32s architectures with support for
skip connections and pretrained encoder backbones.

References:
    - FCN: https://arxiv.org/abs/1411.4038

Author: fishstick AI Framework
Version: 0.1.0
"""

from typing import List, Optional, Dict, Any, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .components import ConvBNReLU, UpsampleBlock


class FCNEncoder(nn.Module):
    """
    FCN-style encoder with VGG or ResNet backbone.

    Args:
        backbone: Backbone type ('vgg16', 'vgg19', 'resnet50', 'resnet101')
        pretrained: Whether to use pretrained weights
        output_stride: Output stride for feature extraction
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = False,
        output_stride: int = 32,
    ):
        super().__init__()
        self.backbone = backbone
        self.output_stride = output_stride

        if "resnet" in backbone:
            self.encoder = self._build_resnet_encoder(backbone, output_stride)
        elif "vgg" in backbone:
            self.encoder = self._build_vgg_encoder(backbone)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def _build_resnet_encoder(self, backbone: str, output_stride: int) -> nn.Module:
        """Build ResNet encoder."""
        import torchvision.models as models

        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=False)
        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=False)
        else:
            resnet = models.resnet101(pretrained=False)

        if output_stride == 8:
            resnet.layer3[0].conv2.stride = (1, 1)
            resnet.layer3[0].downsample[0].stride = (1, 1)
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)
        elif output_stride == 16:
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)

        encoder_layers = [
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        ]

        return nn.Sequential(*encoder_layers)

    def _build_vgg_encoder(self, backbone: str) -> nn.Module:
        """Build VGG encoder."""
        import torchvision.models as models

        if backbone == "vgg16":
            vgg = models.vgg16(pretrained=False).features
        else:
            vgg = models.vgg19(pretrained=False).features

        return vgg

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: [B, 3, H, W] Input image
        Returns:
            List of feature maps at different scales
        """
        features = []

        for layer in self.encoder:
            x = layer(x)
            features.append(x)

        return features


class FCN32s(nn.Module):
    """
    FCN-32s: Single stream with 32x upsampling.

    The simplest FCN variant that predicts at 1/32 resolution
    and upsamples directly to full resolution.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        backbone: Encoder backbone type
        pretrained: Whether to use pretrained encoder
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        backbone: str = "resnet50",
        pretrained: bool = False,
    ):
        super().__init__()

        self.encoder = FCNEncoder(backbone, pretrained)

        encoder_channels = self._get_encoder_channels(backbone)

        self.score = nn.Conv2d(encoder_channels[-1], num_classes, kernel_size=1)

    def _get_encoder_channels(self, backbone: str) -> List[int]:
        """Get encoder output channels."""
        if backbone == "resnet50" or backbone == "resnet101":
            return [256, 512, 1024, 2048]
        elif backbone == "vgg16":
            return [64, 128, 256, 512, 512]
        return [256, 512, 1024, 2048]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        features = self.encoder(x)
        score = self.score(features[-1])

        out = F.interpolate(
            score, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        return out


class FCN16s(nn.Module):
    """
    FCN-16s: Skip connection from pool4 for 16x upsampling.

    Combines features from pool4 (1/16 resolution) with final features
    for better detail recovery.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        backbone: Encoder backbone type
        pretrained: Whether to use pretrained encoder
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        backbone: str = "resnet50",
        pretrained: bool = False,
    ):
        super().__init__()

        self.encoder = FCNEncoder(backbone, pretrained)

        encoder_channels = self._get_encoder_channels(backbone)

        self.score_pool4 = nn.Conv2d(encoder_channels[-2], num_classes, kernel_size=1)
        self.score = nn.Conv2d(encoder_channels[-1], num_classes, kernel_size=1)

        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, bias=False
        )

    def _get_encoder_channels(self, backbone: str) -> List[int]:
        if backbone in ["resnet50", "resnet101"]:
            return [256, 512, 1024, 2048]
        return [64, 128, 256, 512, 512]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        features = self.encoder(x)

        pool4 = features[-2]
        pool5 = features[-1]

        score_pool4 = self.score_pool4(pool4)
        score = self.score(pool5)

        upscore2 = self.upscore2(score)

        upscore16 = F.interpolate(
            upscore2, size=score_pool4.shape[2:], mode="bilinear", align_corners=False
        )

        out = upscore16 + score_pool4

        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


class FCN8s(nn.Module):
    """
    FCN-8s: Full skip connections for best detail recovery.

    Combines features from pool3, pool4, and pool5 for optimal
    segmentation quality with fine-grained details.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        backbone: Encoder backbone type
        pretrained: Whether to use pretrained encoder
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        backbone: str = "resnet50",
        pretrained: bool = False,
    ):
        super().__init__()

        self.encoder = FCNEncoder(backbone, pretrained)

        encoder_channels = self._get_encoder_channels(backbone)

        self.score_pool3 = nn.Conv2d(encoder_channels[-3], num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(encoder_channels[-2], num_classes, kernel_size=1)
        self.score_pool5 = nn.Conv2d(encoder_channels[-1], num_classes, kernel_size=1)

        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, bias=False
        )
        self.upscore4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=4, bias=False
        )
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, bias=False
        )

    def _get_encoder_channels(self, backbone: str) -> List[int]:
        if backbone in ["resnet50", "resnet101"]:
            return [256, 512, 1024, 2048]
        return [64, 128, 256, 512, 512]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        features = self.encoder(x)

        pool3 = features[-3]
        pool4 = features[-2]
        pool5 = features[-1]

        score_pool3 = self.score_pool3(pool3)
        score_pool4 = self.score_pool4(pool4)
        score_pool5 = self.score_pool5(pool5)

        upscore2 = self.upscore2(score_pool5)

        upscore_pool4 = upscore2 + score_pool4
        upscore4 = self.upscore4(upscore_pool4)

        upscore_pool3 = upscore4 + score_pool3
        out = self.upscore8(upscore_pool3)

        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


class FCN(nn.Module):
    """
    Unified FCN model supporting all variants.

    Args:
        variant: FCN variant ('fcn32s', 'fcn16s', 'fcn8s')
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        backbone: Encoder backbone type
        pretrained: Whether to use pretrained encoder
    """

    def __init__(
        self,
        variant: str = "fcn8s",
        in_channels: int = 3,
        num_classes: int = 21,
        backbone: str = "resnet50",
        pretrained: bool = False,
    ):
        super().__init__()

        self.variant = variant

        if variant == "fcn32s":
            self.fcn = FCN32s(in_channels, num_classes, backbone, pretrained)
        elif variant == "fcn16s":
            self.fcn = FCN16s(in_channels, num_classes, backbone, pretrained)
        elif variant == "fcn8s":
            self.fcn = FCN8s(in_channels, num_classes, backbone, pretrained)
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        return self.fcn(x)


class FCNWithAdapters(nn.Module):
    """
    FCN with learnable adapters for transfer learning.

    Adds adaptation layers between encoder and decoder for
    better fine-tuning from pretrained models.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        backbone: Encoder backbone type
        pretrained: Whether to use pretrained encoder
        adapter_channels: Number of channels in adapter layers
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        backbone: str = "resnet50",
        pretrained: bool = False,
        adapter_channels: int = 256,
    ):
        super().__init__()

        self.encoder = FCNEncoder(backbone, pretrained)

        encoder_channels = self._get_encoder_channels(backbone)

        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, adapter_channels, 1, bias=False),
                    nn.BatchNorm2d(adapter_channels),
                    nn.ReLU(inplace=True),
                )
                for ch in encoder_channels
            ]
        )

        self.score = nn.Conv2d(adapter_channels, num_classes, kernel_size=1)

    def _get_encoder_channels(self, backbone: str) -> List[int]:
        if backbone in ["resnet50", "resnet101"]:
            return [256, 512, 1024, 2048]
        return [64, 128, 256, 512, 512]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        features = self.encoder(x)

        adapted_features = []
        for feat, adapter in zip(features, self.adapters):
            adapted_features.append(adapter(feat))

        score = self.score(adapted_features[-1])

        out = F.interpolate(
            score, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        return out


def create_fcn(
    variant: str = "fcn8s",
    num_classes: int = 21,
    in_channels: int = 3,
    backbone: str = "resnet50",
    pretrained: bool = False,
) -> nn.Module:
    """
    Factory function to create an FCN model.

    Args:
        variant: FCN variant ('fcn32s', 'fcn16s', 'fcn8s')
        num_classes: Number of segmentation classes
        in_channels: Number of input channels
        backbone: Encoder backbone type
        pretrained: Whether to use pretrained encoder

    Returns:
        FCN model instance

    Examples:
        >>> model = create_fcn('fcn8s', num_classes=21, backbone='resnet101')
        >>> x = torch.randn(1, 3, 512, 512)
        >>> output = model(x)
    """
    return FCN(variant, in_channels, num_classes, backbone, pretrained)
