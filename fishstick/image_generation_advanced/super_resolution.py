"""
Super-Resolution models for image generation.

This module provides image super-resolution models:
- ESRGAN: Enhanced SRGAN for high-quality upscaling
- SRFlow: Flow-based super-resolution
- Real-ESRGAN: Real-world super-resolution
- SwinIR: Transformer-based SR
- EDVR: Video super-resolution
"""

from typing import Optional, List, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for feature extraction.

    Args:
        in_channels: Number of input channels
        hidden_channels: Number of hidden channels
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 32,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels, 3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels + hidden_channels * 2, hidden_channels, 3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels + hidden_channels * 3, hidden_channels, 3, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels + hidden_channels * 4, in_channels, 3, padding=1
        )

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with dense connections."""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))

        return x5 * 0.2 + x


class ResidualInResidualDenseBlock(nn.Module):
    """Residual in Residual Dense Block (RRDB).

    Args:
        in_channels: Number of input channels
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()

        self.rdb1 = ResidualDenseBlock(in_channels)
        self.rdb2 = ResidualDenseBlock(in_channels)
        self.rdb3 = ResidualDenseBlock(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual-in-residual structure."""
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        return out * 0.2 + x


class ESRGANGenerator(nn.Module):
    """ESRGAN Generator for image super-resolution.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_features: Number of feature channels
        num_blocks: Number of RRDB blocks
        scale: Upsampling scale factor
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 23,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(in_channels, num_features, 3, padding=1)

        self.body = nn.Sequential(
            *[ResidualInResidualDenseBlock(num_features) for _ in range(num_blocks)]
        )

        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1)

        self.upsampler = nn.ModuleList(
            [
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

        if scale == 4:
            self.upsampler.extend(
                [
                    nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        self.conv_last = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for super-resolution."""
        feat = self.conv_first(x)

        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        for layer in self.upsampler:
            if isinstance(layer, nn.Conv2d):
                feat = F.pixel_shuffle(layer(feat), 2)
            else:
                feat = layer(feat)

        out = self.conv_last(feat)

        return out


class ESRGANDiscriminator(nn.Module):
    """ESRGAN Discriminator for adversarial training.

    Args:
        in_channels: Number of input channels
        base_features: Number of base features
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_features: int = 64,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = base_features
        for i in range(2):
            layers.extend(
                [
                    nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels *= 2

        for i in range(3):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        layers.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 1),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for discrimination."""
        return self.net(x)


class SRFlow(nn.Module):
    """SRFlow: Super-Resolution using Normalizing Flows.

    Args:
        in_channels: Number of input channels
        hidden_channels: Number of hidden channels
        num_flows: Number of flow steps
        scale: Upsampling scale factor
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_flows: int = 14,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
        )

        self.flows = nn.ModuleList(
            [FlowBlock(hidden_channels) for _ in range(num_flows)]
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through SRFlow."""
        h = self.encoder(x)

        log_det = torch.zeros(x.size(0), device=x.device)
        for flow in self.flows:
            h, ld = flow(h)
            log_det = log_det + ld

        out = self.decoder(h)

        return out

    def sample(self, num_samples: int, lr_features: Tensor) -> Tensor:
        """Generate SR image from LR features."""
        b = lr_features.shape[0]

        z = torch.randn_like(lr_features)

        for flow in reversed(self.flows):
            z = flow.inverse(z)

        return self.decoder(z)


class FlowBlock(nn.Module):
    """Flow block for SRFlow.

    Args:
        channels: Number of channels
    """

    def __init__(self, channels: int = 64):
        super().__init__()

        self.actnorm = ActNorm2d(channels)
        self.inv_conv = InvConv2d(channels)
        self.coupling = AffineCoupling2d(channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through flow block."""
        log_det = torch.zeros(x.size(0), device=x.device)

        x, ld = self.actnorm(x)
        log_det = log_det + ld

        x, ld = self.inv_conv(x)
        log_det = log_det + ld

        x, ld = self.coupling(x)
        log_det = log_det + ld

        return x, log_det

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse pass through flow block."""
        z, _ = self.coupling.inverse(z)
        z, _ = self.inv_conv.inverse(z)
        z, _ = self.actnorm.inverse(z)
        return z


class ActNorm2d(nn.Module):
    """Activation Normalization for 2D inputs."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.initialized = False

    def initialize(self, x: Tensor) -> None:
        """Initialize using batch statistics."""
        with torch.no_grad():
            bias = -x.mean(dim=[0, 2, 3], keepdim=True)
            var = ((x + bias) ** 2).mean(dim=[0, 2, 3], keepdim=True)
            log_scale = -0.5 * torch.log(var + 1e-6)
            self.bias.data.copy_(bias.data)
            self.log_scale.data.copy_(log_scale.data)
            self.initialized = True

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        if not self.initialized:
            self.initialize(x)
        log_det = self.log_scale.sum() * x.shape[2] * x.shape[3]
        return (x + self.bias) * torch.exp(self.log_scale), log_det

    def inverse(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        log_det = -self.log_scale.sum() * x.shape[2] * x.shape[3]
        return (x * torch.exp(-self.log_scale)) - self.bias, log_det


class InvConv2d(nn.Module):
    """Invertible 1x1 convolution."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        weight = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        log_det = torch.slogdet(self.weight)[1] * x.shape[2] * x.shape[3]
        return F.conv2d(
            x, self.weight.view(self.num_channels, self.num_channels, 1, 1)
        ), log_det

    def inverse(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        weight_inv = torch.inverse(self.weight)
        log_det = -torch.slogdet(self.weight)[1] * x.shape[2] * x.shape[3]
        return F.conv2d(
            x, weight_inv.view(self.num_channels, self.num_channels, 1, 1)
        ), log_det


class AffineCoupling2d(nn.Module):
    """Affine coupling for 2D inputs."""

    def __init__(self, num_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.split_channels = num_channels // 2

        self.net = nn.Sequential(
            nn.Conv2d(self.split_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_channels, 3, padding=1),
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        x_a, x_b = x[:, : self.split_channels], x[:, self.split_channels :]

        params = self.net(x_a)
        log_scale, shift = (
            params[:, : self.split_channels],
            params[:, self.split_channels :],
        )
        log_scale = torch.tanh(log_scale)

        y_b = x_b * torch.exp(log_scale) + shift
        log_det = log_scale.sum(dim=[1, 2, 3])

        return torch.cat([x_a, y_b], dim=1), log_det

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        y_a, y_b = y[:, : self.split_channels], y[:, self.split_channels :]

        params = self.net(y_a)
        log_scale, shift = (
            params[:, : self.split_channels],
            params[:, self.split_channels :],
        )
        log_scale = torch.tanh(log_scale)

        x_b = (y_b - shift) * torch.exp(-log_scale)
        log_det = -log_scale.sum(dim=[1, 2, 3])

        return torch.cat([y_a, x_b], dim=1), log_det


class SwinIRBlock(nn.Module):
    """Swin Transformer block for image restoration.

    Args:
        dim: Dimension of features
        num_heads: Number of attention heads
        window_size: Window size for attention
    """

    def __init__(
        self,
        dim: int = 96,
        num_heads: int = 8,
        window_size: int = 8,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with windowed attention."""
        b, c, h, w = x.shape

        x_flat = x.flatten(2).transpose(1, 2)

        x_norm = self.norm1(x_flat)

        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_norm + attn_out

        x = x + self.mlp(self.norm2(x))

        x = x.transpose(1, 2).reshape(b, c, h, w)

        return x


class SwinIR(nn.Module):
    """SwinIR: Swin Transformer for Image Restoration.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dim: Dimension of features
        num_blocks: Number of Swin blocks
        scale: Upsampling scale factor
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 96,
        num_blocks: int = 8,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale

        self.shallow = nn.Conv2d(in_channels, dim, 3, padding=1)

        self.blocks = nn.ModuleList([SwinIRBlock(dim) for _ in range(num_blocks)])

        self.deep = nn.Conv2d(dim, dim, 3, padding=1)

        self.upsampler = nn.Sequential(
            nn.Conv2d(dim, dim * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(dim, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for SwinIR."""
        x = self.shallow(x)

        for block in self.blocks:
            x = block(x)

        x = self.deep(x)

        return self.upsampler(x)


class RealESRGAN(nn.Module):
    """Real-ESRGAN for real-world super-resolution.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_features: Number of feature channels
        scale: Upsampling scale factor
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        scale: int = 4,
    ):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(in_channels, num_features, 3, padding=1)

        self.body = nn.ModuleList(
            [ResidualInResidualDenseBlock(num_features) for _ in range(23)]
        )

        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1)

        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_last = nn.Conv2d(num_features, out_channels, 3, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for real-world SR."""
        feat = self.conv_first(x)

        body_feat = self.conv_body(
            sum(block(feat) for block in self.body) / len(self.body)
        )
        feat = feat + body_feat

        out = self.upsampler(feat)
        out = self.conv_last(out)

        return out


class EDVRNet(nn.Module):
    """EDVR: Enhanced Deformable Video Restoration.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_frames: Number of input frames
        num_features: Number of feature channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_frames: int = 5,
        num_features: int = 64,
    ):
        super().__init__()
        self.num_frames = num_frames

        self.extract = nn.Conv2d(in_channels, num_features, 3, padding=1)

        self.pcd_align = PCDAalignment(num_features, num_frames)

        self.reconstruction = nn.Sequential(
            nn.Conv2d(num_features * num_frames, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for video SR.

        Args:
            x: Input tensor of shape (B, T, C, H, W)
        """
        b, t, c, h, w = x.shape

        x = x.view(b * t, c, h, w)
        feats = self.extract(x)
        _, c, h, w = feats.shape
        feats = feats.view(b, t, c, h, w)

        aligned = self.pcd_align(feats)

        b, t, c, h, w = aligned.shape
        aligned = aligned.view(b, t * c, h, w)

        recon = self.reconstruction(aligned)

        return self.upsampler(recon)


class PCDAalignment(nn.Module):
    """Pyramid, Cascading and Deformable Alignment module.

    Args:
        num_features: Number of feature channels
        num_frames: Number of frames
    """

    def __init__(
        self,
        num_features: int = 64,
        num_frames: int = 5,
    ):
        super().__init__()

        self.center = num_frames // 2

        self.offset_conv1 = nn.Conv2d(
            num_features * num_frames, num_features, 3, padding=1
        )
        self.offset_conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)

        self.dcn = nn.Conv2d(num_features, num_features, 3, padding=1, dilation=1)

    def forward(self, feats: Tensor) -> Tensor:
        """Forward pass for alignment.

        Args:
            feats: Features of shape (B, T, C, H, W)
        """
        b, t, c, h, w = feats.shape

        ref_feat = feats[:, self.center]

        feat_stack = feats.view(b, -1, h, w)

        offset = self.offset_conv2(self.lrelu(self.offset_conv1(feat_stack)))

        aligned = []
        for i in range(t):
            aligned.append(self.dcn(feats[:, i] + offset))

        return torch.stack(aligned, dim=1)

    def relu(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, 0.2, inplace=True)


class RCANBlock(nn.Module):
    """Residual Channel Attention Block.

    Args:
        num_channels: Number of input channels
        reduction: Channel reduction ratio
    """

    def __init__(
        self,
        num_channels: int = 64,
        reduction: int = 16,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with channel attention."""
        residual = x

        out = F.relu(self.conv1(x))
        out = self.conv2(out)

        attention = self.channel_attention(out)

        out = out * attention

        return out + residual


class RCAN(nn.Module):
    """Residual Channel Attention Network for image SR.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_features: Number of feature channels
        num_blocks: Number of RCAN blocks
        scale: Upsampling scale factor
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 10,
        scale: int = 4,
    ):
        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, num_features, 3, padding=1)

        self.blocks = nn.ModuleList(
            [RCANBlock(num_features) for _ in range(num_blocks)]
        )

        self.conv_last = nn.Conv2d(num_features, num_features, 3, padding=1)

        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(num_features, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for RCAN."""
        feat = self.conv_first(x)

        body_feat = feat
        for block in self.blocks:
            body_feat = block(body_feat)

        feat = feat + self.conv_last(body_feat)

        return self.upsampler(feat)
