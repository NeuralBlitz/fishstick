"""
DeepLab Variants for Semantic Segmentation

Advanced DeepLab architectures including:
- DeepLabV3 with ASPP
- DeepLabV3+ with encoder-decoder
- Depthwise separable convolutions
- MobileNetV3-based DeepLab

References:
    - DeepLabV3: https://arxiv.org/abs/1706.05587
    - DeepLabV3+: https://arxiv.org/abs/1802.02611

Author: fishstick AI Framework
Version: 0.1.0
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .components import ASPPModule, FeatureFusionModule, ConvBNReLU


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution (MobileNet-style).

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size
        stride: Stride
        padding: Padding
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(self.depthwise(x))
        x = self.relu(x)
        x = self.bn2(self.pointwise(x))
        x = self.relu(x)
        return x


class ResNetBlock(nn.Module):
    """
    ResNet block with optional dilation.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride
        dilation: Dilation rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)

        self.conv2 = nn.Conv2d(
            out_channels // 4,
            out_channels // 4,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels // 4)

        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class DeepLabV3Encoder(nn.Module):
    """
    DeepLabV3 encoder with ResNet backbone.

    Args:
        backbone: Backbone type ('resnet50', 'resnet101')
        output_stride: Output stride (8 or 16)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        output_stride: int = 16,
    ):
        super().__init__()

        if backbone == "resnet50":
            import torchvision.models as models

            resnet = models.resnet50(pretrained=False)
            self.channels = [256, 512, 1024, 2048]
        else:
            import torchvision.models as models

            resnet = models.resnet101(pretrained=False)
            self.channels = [256, 512, 1024, 2048]

        if output_stride == 8:
            resnet.layer3[0].conv2.stride = (1, 1)
            resnet.layer3[0].downsample[0].stride = (1, 1)
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)
        elif output_stride == 16:
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: [B, 3, H, W] Input image
        Returns:
            List of feature maps at different scales
        """
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return [x]


class DeepLabV3(nn.Module):
    """
    DeepLabV3 with Atrous Spatial Pyramid Pooling.

    Uses ASPP module with atrous convolutions at multiple rates
    for multi-scale context aggregation.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        backbone: Encoder backbone type
        output_stride: Output stride (8 or 16)
        aspp_channels: Number of channels in ASPP output
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        backbone: str = "resnet50",
        output_stride: int = 16,
        aspp_channels: int = 256,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.encoder = DeepLabV3Encoder(backbone, output_stride)

        encoder_channels = self.encoder.channels[-1]

        self.aspp = ASPPModule(
            encoder_channels,
            aspp_channels,
            atrous_rates=(6, 12, 18) if output_stride == 16 else (12, 24, 36),
            dropout_rate=dropout_rate,
        )

        self.classifier = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        features = self.encoder(x)

        x = self.aspp(features[-1])

        x = self.classifier(x)

        out = F.interpolate(x, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with encoder-decoder structure.

    Adds a decoder with ASPP and low-level features for better
    boundary delineation.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        backbone: Encoder backbone type
        output_stride: Output stride
        aspp_channels: Number of channels in ASPP
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        backbone: str = "resnet50",
        output_stride: int = 16,
        aspp_channels: int = 256,
    ):
        super().__init__()

        if backbone == "resnet50":
            import torchvision.models as models

            resnet = models.resnet50(pretrained=False)
            low_level_channels = 256
            high_level_channels = 2048
        else:
            import torchvision.models as models

            resnet = models.resnet101(pretrained=False)
            low_level_channels = 256
            high_level_channels = 2048

        if output_stride == 8:
            resnet.layer3[0].conv2.stride = (1, 1)
            resnet.layer3[0].downsample[0].stride = (1, 1)
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aspp = ASPPModule(
            high_level_channels,
            aspp_channels,
            atrous_rates=(6, 12, 18) if output_stride == 16 else (12, 24, 36),
            dropout_rate=0.5,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                aspp_channels + low_level_channels,
                aspp_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_channels, aspp_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)

        self._low_level_channels = low_level_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        stem_out = self.stem(x)

        low_level = self.layer1(stem_out)
        low_level = self.layer2(low_level)

        x = self.layer3(low_level)
        x = self.layer4(x)

        x = self.aspp(x)

        x = F.interpolate(
            x, size=low_level.shape[2:], mode="bilinear", align_corners=False
        )

        x = torch.cat([x, low_level], dim=1)

        x = self.decoder(x)

        x = self.classifier(x)

        out = F.interpolate(x, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


class MobileNetV3Encoder(nn.Module):
    """
    MobileNetV3 encoder for efficient DeepLab.

    Args:
        mode: 'large' or 'small' MobileNetV3
    """

    def __init__(self, mode: str = "large"):
        super().__init__()

        if mode == "large":
            self.channels = [16, 24, 40, 160, 160]
        else:
            self.channels = [16, 16, 24, 48, 96]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
        )

        if mode == "large":
            self.blocks = self._build_large_blocks()
        else:
            self.blocks = self._build_small_blocks()

    def _build_large_blocks(self) -> nn.Module:
        layers = [
            self._make_block(16, 16, 1, 1, 16),
            self._make_block(16, 24, 2, 2, 24),
            self._make_block(24, 40, 2, 5, 40),
            self._make_block(40, 80, 2, 5, 80),
            self._make_block(80, 160, 1, 5, 160),
        ]
        return nn.Sequential(*layers)

    def _build_small_blocks(self) -> nn.Module:
        layers = [
            self._make_block(16, 16, 2, 1, 16),
            self._make_block(16, 24, 2, 4, 24),
            self._make_block(24, 48, 2, 3, 48),
            self._make_block(48, 96, 1, 6, 96),
        ]
        return nn.Sequential(*layers)

    def _make_block(
        self, in_ch: int, out_ch: int, stride: int, kernel: int, exp_ch: int
    ) -> nn.Module:
        padding = (kernel - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_ch, exp_ch, 1, bias=False),
            nn.BatchNorm2d(exp_ch),
            nn.Hardswish(inplace=True),
            nn.Conv2d(
                exp_ch, exp_ch, kernel, stride, padding, groups=exp_ch, bias=False
            ),
            nn.BatchNorm2d(exp_ch),
            nn.Hardswish(inplace=True),
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.stem(x)

        low_level = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 1:
                low_level = x

        return x, low_level


class MobileDeepLabV3(nn.Module):
    """
    MobileNetV3-based DeepLabV3+ for efficient segmentation.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        mode: MobileNetV3 mode ('large' or 'small')
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        mode: str = "large",
    ):
        super().__init__()

        self.encoder = MobileNetV3Encoder(mode)

        aspp_channels = 256

        self.aspp = ASPPModule(
            160, aspp_channels, atrous_rates=(6, 12, 18), dropout_rate=0.5
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_channels + 24, aspp_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(aspp_channels, aspp_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.Hardswish(inplace=True),
        )

        self.classifier = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        x, low_level = self.encoder(x)

        x = self.aspp(x)

        x = F.interpolate(
            x, size=low_level.shape[2:], mode="bilinear", align_corners=False
        )

        x = torch.cat([x, low_level], dim=1)

        x = self.decoder(x)

        x = self.classifier(x)

        out = F.interpolate(x, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


class LiteDeepLabV3(nn.Module):
    """
    Lite DeepLabV3 with depthwise separable convolutions.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.encoder = nn.ModuleList(
            [
                DepthwiseSeparableConv(32, 64, stride=2),
                DepthwiseSeparableConv(64, 128, stride=2),
                DepthwiseSeparableConv(128, 256, stride=2),
                DepthwiseSeparableConv(256, 512, stride=2),
            ]
        )

        self.aspp = ASPPModule(512, 256, atrous_rates=(6, 12, 18), dropout_rate=0.5)

        self.decoder = nn.ModuleList(
            [
                DepthwiseSeparableConv(256 + 128, 256),
                DepthwiseSeparableConv(256 + 64, 256),
                DepthwiseSeparableConv(256 + 32, 256),
            ]
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        x = self.stem(x)

        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i < len(self.encoder) - 1:
                skips.append(x)

        x = self.aspp(x)

        skips = skips[::-1]

        for i, decoder_block in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            skip = skips[i] if i < len(skips) else None
            if skip is not None:
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(
                        x, size=skip.shape[2:], mode="bilinear", align_corners=False
                    )
                x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)

        x = self.classifier(x)

        out = F.interpolate(x, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


def create_deeplab(
    variant: str = "deeplabv3+",
    num_classes: int = 21,
    in_channels: int = 3,
    backbone: str = "resnet50",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create DeepLab models.

    Args:
        variant: Model variant ('deeplabv3', 'deeplabv3+', 'mobile_deeplab', 'lite_deeplab')
        num_classes: Number of segmentation classes
        in_channels: Number of input channels
        backbone: Encoder backbone type
        **kwargs: Additional model-specific arguments

    Returns:
        DeepLab model instance

    Examples:
        >>> model = create_deeplab('deeplabv3+', num_classes=21, backbone='resnet101')
    """
    if variant == "deeplabv3":
        return DeepLabV3(in_channels, num_classes, backbone)
    elif variant == "deeplabv3+":
        return DeepLabV3Plus(in_channels, num_classes, backbone)
    elif variant == "mobile_deeplab":
        return MobileDeepLabV3(in_channels, num_classes)
    elif variant == "lite_deeplab":
        return LiteDeepLabV3(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")
