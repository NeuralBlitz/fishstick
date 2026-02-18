"""
2D Human Pose Estimation Models

Implementation of 2D pose estimation architectures including HRNet, Stacked Hourglass,
and Simple Baseline models.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """Basic residual block for HRNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class HRNetEncoder(nn.Module):
    """HRNet high-resolution representation encoder."""

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 64,
        stage_channels: Tuple[int, int, int, int] = (32, 64, 128, 256),
        stage_blocks: Tuple[int, int, int, int] = (1, 1, 4, 3),
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, stem_channels, 3, 2, 1),
            ConvBlock(stem_channels, stem_channels, 3, 1, 1),
        )

        self.stage1 = self._make_stage(
            stem_channels, stage_channels[0], stage_blocks[0]
        )
        self.stage2 = self._make_stage(
            stage_channels[0], stage_channels[1], stage_blocks[1], downsample=True
        )
        self.stage3 = self._make_stage(
            stage_channels[1], stage_channels[2], stage_blocks[2], downsample=True
        )
        self.stage4 = self._make_stage(
            stage_channels[2], stage_channels[3], stage_blocks[3], downsample=True
        )

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        downsample: bool = False,
    ) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=2 if downsample else 1)]
        layers += [
            BasicBlock(out_channels, out_channels) for _ in range(num_blocks - 1)
        ]
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return [x1, x2, x3, x4]


class HRNetDecoder(nn.Module):
    """HRNet decoder for multi-resolution feature fusion."""

    def __init__(
        self,
        stage_channels: Tuple[int, int, int, int] = (32, 64, 128, 256),
        num_joints: int = 17,
        final_channels: int = 32,
    ):
        super().__init__()
        self.num_joints = num_joints

        self.transition1 = self._make_transition(
            [stage_channels[0]], stage_channels[:2]
        )
        self.transition2 = self._make_transition(
            stage_channels[:2], stage_channels[1:3]
        )
        self.transition3 = self._make_transition(
            stage_channels[1:3], stage_channels[2:4]
        )

        self.fusion2 = self._make_fusion(stage_channels[1], stage_channels[2])
        self.fusion3 = self._make_fusion(stage_channels[2], stage_channels[3])

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(stage_channels[3], final_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Sequential(
            ConvBlock(final_channels, final_channels, 3, 1, 1),
            nn.Conv2d(final_channels, num_joints, 1),
        )

    def _make_transition(
        self, in_channels_list: List[int], out_channels_list: List[int]
    ) -> nn.ModuleList:
        transition = nn.ModuleList()
        for i, out_ch in enumerate(out_channels_list):
            if i < len(in_channels_list):
                transition.append(ConvBlock(in_channels_list[i], out_ch, 3, 1, 1))
            else:
                in_ch = in_channels_list[-1]
                layers = [
                    nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
                transition.append(nn.Sequential(*layers))
        return transition

    def _make_fusion(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            ConvBlock(in_channels + out_channels, out_channels, 3, 1, 1),
            ConvBlock(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, features: List[Tensor]) -> Tensor:
        x1, x2, x3, x4 = features

        x = self.deconv1(x4)
        x = self.final(x)

        return x


class HRNet(nn.Module):
    """
    High-Resolution Network for 2D Pose Estimation.

    Maintains high-resolution representations throughout the network.

    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        num_joints: Number of output keypoints (default: 17 for COCO)
        stem_channels: Number of channels in stem
        stage_channels: Channels for each stage
        stage_blocks: Number of blocks per stage

    Example:
        >>> model = HRNet(num_joints=17)
        >>> heatmap = model(images)  # (B, num_joints, H, W)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = 17,
        stem_channels: int = 64,
        stage_channels: Tuple[int, int, int, int] = (32, 64, 128, 256),
        stage_blocks: Tuple[int, int, int, int] = (1, 1, 4, 3),
    ):
        super().__init__()
        self.num_joints = num_joints

        self.encoder = HRNetEncoder(
            in_channels, stem_channels, stage_channels, stage_blocks
        )
        self.decoder = HRNetDecoder(stage_channels, num_joints)

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        heatmap = self.decoder(features)
        return heatmap


class SimpleBaseline2D(nn.Module):
    """
    Simple Baseline 2D Pose Estimation using ResNet backbone + deconv layers.

    Args:
        backbone: Name of backbone ('resnet50', 'resnet101')
        num_joints: Number of output keypoints
        deconv_channels: Channels for deconv layers
        deconv_kernel: Kernel size for deconv layers

    Example:
        >>> model = SimpleBaseline2D('resnet50', num_joints=17)
        >>> heatmap = model(images)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_joints: int = 17,
        deconv_channels: Tuple[int, int, int] = (256, 256, 256),
        deconv_kernel: int = 4,
    ):
        super().__init__()
        self.num_joints = num_joints

        if backbone == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=False)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            base_channels = 512
        elif backbone == "resnet34":
            resnet = torchvision.models.resnet34(pretrained=False)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            base_channels = 512
        elif backbone == "resnet50":
            resnet = torchvision.models.resnet50(pretrained=False)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            base_channels = 2048
        else:
            resnet = torchvision.models.resnet101(pretrained=False)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            base_channels = 2048

        self.deconv = nn.ModuleList()
        in_channels = base_channels
        for out_channels in deconv_channels:
            self.deconv.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels, out_channels, deconv_kernel, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channels

        self.final_layer = nn.Conv2d(deconv_channels[-1], num_joints, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)

        for deconv_layer in self.deconv:
            x = deconv_layer(x)

        return self.final_layer(x)


class StackedHourglass(nn.Module):
    """
    Stacked Hourglass Network for 2D Pose Estimation.

    Uses repeated hourglass modules with intermediate supervision.

    Args:
        num_stacks: Number of hourglass stacks
        in_channels: Number of input channels
        num_joints: Number of output keypoints
        intermediate supervision: Whether to add supervision at each stack

    Example:
        >>> model = StackedHourglass(num_stacks=2, num_joints=17)
        >>> heatmaps = model(images)  # List of heatmaps from each stack
    """

    def __init__(
        self,
        num_stacks: int = 2,
        in_channels: int = 3,
        num_joints: int = 17,
        intermediate_supervision: bool = True,
    ):
        super().__init__()
        self.num_stacks = num_stacks
        self.num_joints = num_joints

        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBlock(64, 128, 3, 1, 1)
        self.conv3 = ConvBlock(128, 128, 3, 1, 1)
        self.conv4 = ConvBlock(128, 256, 3, 1, 1)

        self.hourglasses = nn.ModuleList([Hourglass(256, 4) for _ in range(num_stacks)])

        self.conv_convs = nn.ModuleList(
            [ConvBlock(256, 256, 3, 1, 1) for _ in range(num_stacks)]
        )

        self.conv_outs = nn.ModuleList(
            [nn.Conv2d(256, num_joints, 1) for _ in range(num_stacks)]
        )

        self.intermediate_supervision = intermediate_supervision
        if intermediate_supervision:
            self.int_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvBlock(256, 256, 3, 1, 1), nn.Conv2d(256, num_joints, 1)
                    )
                    for _ in range(num_stacks - 1)
                ]
            )

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []
        for i in range(self.num_stacks):
            hg_out = self.hourglasses[i](x)
            conv_out = self.conv_convs[i](hg_out)
            out = self.conv_outs[i](conv_out)
            outputs.append(out)

            if self.intermediate_supervision and i < self.num_stacks - 1:
                x = x + self.int_convs[i](hg_out)
            elif i < self.num_stacks - 1:
                x = hg_out

        return outputs


class Hourglass(nn.Module):
    """Single Hourglass module with residual downsampling and upsampling."""

    def __init__(self, channels: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels

        self.up1 = BasicBlock(channels, channels)

        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = BasicBlock(channels, channels * 2)

        if num_levels > 1:
            self.low2 = Hourglass(channels * 2, num_levels - 1)
        else:
            self.low2 = BasicBlock(channels * 2, channels * 2)

        self.low3 = BasicBlock(channels * 2, channels)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x: Tensor) -> Tensor:
        up1 = self.up1(x)
        low1 = self.pool(x)
        low1 = self.low1(low1)

        low2 = self.low2(low1)

        low3 = self.low2(low2)
        up2 = self.up2(low3)

        return up1 + up2


class Pose2DModel(nn.Module):
    """
    Generic 2D Pose Estimation Model wrapper.

    Supports multiple backbones and provides unified interface.

    Args:
        backbone: Type of backbone ('hrnet', 'hourglass', 'simple')
        num_joints: Number of keypoints
        pretrained: Whether to use pretrained weights

    Example:
        >>> model = Pose2DModel('hrnet', num_joints=17)
        >>> heatmap = model(images)
    """

    def __init__(
        self,
        backbone: str = "hrnet",
        num_joints: int = 17,
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.backbone_type = backbone

        if backbone == "hrnet":
            self.model = HRNet(num_joints=num_joints, **kwargs)
        elif backbone == "hourglass":
            self.model = StackedHourglass(num_joints=num_joints, **kwargs)
        elif backbone == "simple":
            self.model = SimpleBaseline2D(num_joints=num_joints, **kwargs)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def get_heatmap(self, x: Tensor) -> Tensor:
        """Get predicted heatmaps."""
        heatmap = self.forward(x)
        if isinstance(heatmap, list):
            return heatmap[-1]
        return heatmap


def create_hrnet_w32(num_joints: int = 17) -> HRNet:
    """Create HRNet-W32 model."""
    return HRNet(
        num_joints=num_joints,
        stem_channels=64,
        stage_channels=(32, 64, 128, 256),
        stage_blocks=(1, 1, 4, 3),
    )


def create_hrnet_w48(num_joints: int = 17) -> HRNet:
    """Create HRNet-W48 model."""
    return HRNet(
        num_joints=num_joints,
        stem_channels=64,
        stage_channels=(48, 96, 192, 384),
        stage_blocks=(1, 1, 3, 4),
    )


import torchvision
