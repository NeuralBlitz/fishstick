"""
U-Net Variants for Semantic Segmentation

Advanced U-Net architectures including:
- Residual U-Net with skip connections
- Attention U-Net with attention gates
- U-Net++ with nested skip pathways
- Mobile U-Net for lightweight segmentation

References:
    - U-Net: https://arxiv.org/abs/1505.04597
    - Attention U-Net: https://arxiv.org/abs/1804.03999
    - U-Net++: https://arxiv.org/abs/1807.10165

Author: fishstick AI Framework
Version: 0.1.0
"""

from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .components import ConvBNReLU, UpsampleBlock


class EncoderBlock(nn.Module):
    """
    Encoder block with double convolutions and optional pooling.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        use_bn: Whether to use batch normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = True,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBNReLU(
                in_channels, out_channels, kernel_size=3, padding=1, use_bn=use_bn
            ),
            ConvBNReLU(
                out_channels, out_channels, kernel_size=3, padding=1, use_bn=use_bn
            ),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, C, H, W] Input
        Returns:
            Tuple of (pooled features, unpooled features for skip)
        """
        conv_out = self.conv(x)
        pool_out = self.pool(conv_out)
        return pool_out, conv_out


class AttentionGate(nn.Module):
    """
    Attention gate for focusing on relevant features.

    Args:
        gate_channels: Number of channels in gate (decoder) signal
        skip_channels: Number of channels in skip (encoder) signal
        inter_channels: Number of intermediate channels
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ):
        super().__init__()

        if inter_channels is None:
            inter_channels = skip_channels // 2

        self.W_g = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(inter_channels),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(inter_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: Tensor, skip: Tensor) -> Tensor:
        """
        Args:
            gate: [B, C_g, H_g, W_g] Gate signal from decoder
            skip: [B, C_s, H_s, W_s] Skip connection from encoder
        Returns:
            [B, C_s, H_s, W_s] Attention-weighted skip features
        """
        g1 = self.W_g(gate)
        x1 = self.W_x(skip)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=False
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return skip * psi


class ResidualEncoderBlock(nn.Module):
    """
    Residual encoder block with skip connections.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, C, H, W] Input
        Returns:
            Tuple of (pooled features, skip features)
        """
        identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        out = self.relu(out)

        skip = out
        pool = self.pool(out)

        return pool, skip


class UNet(nn.Module):
    """
    Standard U-Net for Semantic Segmentation.

    Encoder-decoder architecture with skip connections.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        base_channels: Base number of channels
        depth: Number of encoder/decoder levels
        use_bn: Whether to use batch normalization
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        use_bn: bool = True,
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        channels = base_channels
        for i in range(depth):
            self.encoder.append(
                EncoderBlock(in_channels if i == 0 else channels // 2, channels, use_bn)
            )
            channels *= 2

        channels = base_channels * (2 ** (depth - 1))
        for i in range(depth - 1):
            self.decoder.append(UpsampleBlock(channels, channels // 2, scale_factor=2))
            channels //= 2

        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        skips = []

        for encoder_block in self.encoder:
            x, skip = encoder_block(x)
            skips.append(skip)

        skips = skips[:-1][::-1]

        for i, decoder_block in enumerate(self.decoder):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return self.final(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net with attention gates.

    Uses attention gates to focus on relevant features in skip connections,
    improving feature selection in medical imaging segmentation.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        base_channels: Base number of channels
        depth: Number of encoder/decoder levels
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoder = nn.ModuleList()

        channels = base_channels
        for i in range(depth):
            self.encoder.append(
                EncoderBlock(
                    in_channels if i == 0 else channels // 2,
                    channels,
                )
            )
            channels *= 2

        channels = base_channels * (2 ** (depth - 1))
        for i in range(depth - 1):
            self.attention_gates.append(AttentionGate(channels, channels // 2))
            self.decoder.append(
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2)
            )
            channels //= 2

        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        self._depth = depth

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        skips = []

        for encoder_block in self.encoder:
            x, skip = encoder_block(x)
            skips.append(skip)

        skips = skips[:-1][::-1]

        for i, (decoder_conv, attention_gate) in enumerate(
            zip(self.decoder, self.attention_gates)
        ):
            x = decoder_conv(x)

            skip = skips[i] if i < len(skips) else None
            if skip is not None:
                skip = attention_gate(x, skip)
                x = torch.cat([x, skip], dim=1)
                x = ConvBNReLU(x.shape[1], x.shape[1] // 2, kernel_size=3, padding=1)(x)

        return self.final(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ with nested and dense skip pathways.

    Uses nested U-structures with deep supervision for better
    multi-scale feature fusion.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        base_channels: Base number of channels
        deep_supervision: Whether to use deep supervision
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        deep_supervision: bool = False,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        self.stage1 = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNReLU(in_channels, base_channels, kernel_size=3, padding=1),
                    ConvBNReLU(base_channels, base_channels, kernel_size=3, padding=1),
                )
            ]
        )

        self.stage2 = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNReLU(
                        base_channels * 2, base_channels * 2, kernel_size=3, padding=1
                    ),
                    ConvBNReLU(
                        base_channels * 2, base_channels * 2, kernel_size=3, padding=1
                    ),
                )
            ]
        )

        self.stage3 = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNReLU(
                        base_channels * 4, base_channels * 4, kernel_size=3, padding=1
                    ),
                    ConvBNReLU(
                        base_channels * 4, base_channels * 4, kernel_size=3, padding=1
                    ),
                )
            ]
        )

        self.stage4 = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNReLU(
                        base_channels * 8, base_channels * 8, kernel_size=3, padding=1
                    ),
                    ConvBNReLU(
                        base_channels * 8, base_channels * 8, kernel_size=3, padding=1
                    ),
                )
            ]
        )

        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 8, kernel_size=2, stride=2
        )
        self.up3 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 4, kernel_size=2, stride=2
        )
        self.up2 = nn.ConvTranspose2d(
            base_channels * 2, base_channels * 2, kernel_size=2, stride=2
        )

        if deep_supervision:
            self.final1 = nn.Conv2d(base_channels, num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(base_channels * 2, num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(base_channels * 4, num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(base_channels * 8, num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        x1 = self.stage1[0](x)

        x2 = self.pool(x1)
        x2 = torch.cat([x2, self.up2(x1)], dim=1)
        x2 = self.stage2[0](x2)

        x3 = self.pool(x2)
        x3 = torch.cat([x3, self.up3(x2)], dim=1)
        x3 = self.stage3[0](x3)

        x4 = self.pool(x3)
        x4 = torch.cat([x4, self.up4(x3)], dim=1)
        x4 = self.stage4[0](x4)

        if self.deep_supervision:
            out1 = self.final1(x1)
            out2 = self.final2(x2)
            out3 = self.final3(x3)
            out4 = self.final4(x4)

            out1 = F.interpolate(
                out1, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            out2 = F.interpolate(
                out2, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            out3 = F.interpolate(
                out3, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            out4 = F.interpolate(
                out4, size=x.shape[2:], mode="bilinear", align_corners=False
            )

            return (out1 + out2 + out3 + out4) / 4

        x = self.up4(x4)
        x = torch.cat([x, x3], dim=1)
        x = ConvBNReLU(x.shape[1], base_channels * 4, kernel_size=3, padding=1)(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = ConvBNReLU(x.shape[1], base_channels * 2, kernel_size=3, padding=1)(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = ConvBNReLU(x.shape[1], base_channels, kernel_size=3, padding=1)(x)

        return self.final(x)


class MobileConvBlock(nn.Module):
    """
    Depthwise separable convolution block for Mobile U-Net.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for convolutions
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(self.depthwise(x))
        x = self.relu(x)
        x = self.bn2(self.pointwise(x))
        x = self.relu(x)
        return x


class MobileUNet(nn.Module):
    """
    Mobile U-Net for lightweight segmentation.

    Uses depthwise separable convolutions for efficient
    inference on mobile and embedded devices.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        base_channels: Base number of channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.encoder1 = self._make_layer(base_channels, base_channels * 2, 2)
        self.encoder2 = self._make_layer(base_channels * 2, base_channels * 4, 2)
        self.encoder3 = self._make_layer(base_channels * 4, base_channels * 8, 2)

        self.decoder3 = self._make_decode_layer(base_channels * 8, base_channels * 4)
        self.decoder2 = self._make_decode_layer(base_channels * 4, base_channels * 2)
        self.decoder1 = self._make_decode_layer(base_channels * 2, base_channels)

        self.final = nn.Sequential(
            nn.Conv2d(
                base_channels, base_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, kernel_size=1),
        )

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int
    ) -> nn.Module:
        layers = [MobileConvBlock(in_channels, out_channels, stride=2)]
        for _ in range(num_blocks - 1):
            layers.append(MobileConvBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _make_decode_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            MobileConvBlock(out_channels * 2, out_channels, stride=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        x = self.stem(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        d3 = self.decoder3(e3)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)

        return self.final(d1)


class ResidualUNet(nn.Module):
    """
    Residual U-Net with skip connections in encoder/decoder.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        base_channels: Base number of channels
        depth: Number of levels
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        channels = base_channels
        for i in range(depth):
            self.encoder.append(
                ResidualEncoderBlock(
                    in_channels if i == 0 else channels // 2,
                    channels,
                )
            )
            channels *= 2

        channels = base_channels * (2 ** (depth - 1))
        for i in range(depth - 1):
            self.decoder.append(
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2)
            )
            channels //= 2

        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input image
        Returns:
            [B, num_classes, H, W] Segmentation logits
        """
        skips = []

        for encoder_block in self.encoder:
            x, skip = encoder_block(x)
            skips.append(skip)

        skips = skips[:-1][::-1]

        for i, decoder_conv in enumerate(self.decoder):
            x = decoder_conv(x)

            skip = skips[i] if i < len(skips) else None
            if skip is not None:
                x = x + skip
                x = ConvBNReLU(x.shape[1], x.shape[1] // 2, kernel_size=3, padding=1)(x)

        return self.final(x)


def create_unet(
    variant: str = "unet",
    num_classes: int = 1,
    in_channels: int = 3,
    base_channels: int = 64,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create U-Net variants.

    Args:
        variant: Model variant ('unet', 'attention_unet', 'unet++', 'mobile_unet', 'residual_unet')
        num_classes: Number of segmentation classes
        in_channels: Number of input channels
        base_channels: Base number of channels
        **kwargs: Additional model-specific arguments

    Returns:
        U-Net model instance

    Examples:
        >>> model = create_unet('attention_unet', num_classes=1, base_channels=64)
    """
    if variant == "unet":
        return UNet(in_channels, num_classes, base_channels)
    elif variant == "attention_unet":
        return AttentionUNet(in_channels, num_classes, base_channels)
    elif variant == "unet++" or variant == "unetplusplus":
        return UNetPlusPlus(in_channels, num_classes, base_channels)
    elif variant == "mobile_unet":
        return MobileUNet(in_channels, num_classes, base_channels)
    elif variant == "residual_unet":
        return ResidualUNet(in_channels, num_classes, base_channels)
    else:
        raise ValueError(f"Unknown variant: {variant}")
