"""
Normalizing Flow implementations for image generation.

This module provides flow-based generative models:
- Glow: Flow-based model with invertible 1x1 convolutions
- RealNVP: Real-valued Non-Volume Preserving flows
- Flow++: Improved flow-based model with variational dequantization
"""

from typing import Optional, List, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ActNorm(nn.Module):
    """Activation Normalization layer with data-dependent initialization."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.initialized = False

    def initialize(self, x: Tensor) -> None:
        """Initialize bias and scale using batch statistics."""
        with torch.no_grad():
            bias = -x.mean(dim=[0, 2, 3], keepdim=True)
            var = ((x + bias) ** 2).mean(dim=[0, 2, 3], keepdim=True)
            log_scale = -0.5 * torch.log(var + 1e-6)
            self.bias.data.copy_(bias.data)
            self.log_scale.data.copy_(log_scale.data)
            self.initialized = True

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with log determinant computation."""
        if not self.initialized:
            self.initialize(x)
        log_det = self.log_scale.sum() * x.shape[2] * x.shape[3]
        return (x + self.bias) * torch.exp(self.log_scale), log_det

    def inverse(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass with log determinant computation."""
        log_det = -self.log_scale.sum() * x.shape[2] * x.shape[3]
        return (x * torch.exp(-self.log_scale)) - self.bias, log_det


class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 convolution for Glow."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        weight = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with log determinant computation."""
        log_det = torch.slogdet(self.weight)[1] * x.shape[2] * x.shape[3]
        return F.conv2d(
            x, self.weight.view(self.num_channels, self.num_channels, 1, 1)
        ), log_det

    def inverse(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass with log determinant computation."""
        weight_inv = torch.inverse(self.weight)
        log_det = -torch.slogdet(self.weight)[1] * x.shape[2] * x.shape[3]
        return F.conv2d(
            x, weight_inv.view(self.num_channels, self.num_channels, 1, 1)
        ), log_det


class AffineCoupling(nn.Module):
    """Affine coupling layer for flow-based models."""

    def __init__(
        self,
        num_channels: int,
        hidden_channels: int = 512,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
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
        self.use_checkpoint = use_checkpoint

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with affine transformation."""
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
        """Inverse pass with affine transformation."""
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


class FlowStep(nn.Module):
    """Single flow step with activation norm, 1x1 conv, and coupling."""

    def __init__(
        self,
        num_channels: int,
        hidden_channels: int = 512,
    ):
        super().__init__()
        self.act_norm = ActNorm(num_channels)
        self.inv_conv = Invertible1x1Conv(num_channels)
        self.coupling = AffineCoupling(num_channels, hidden_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through flow step."""
        log_det = torch.zeros(x.shape[0], device=x.device)

        x, ld = self.act_norm(x)
        log_det = log_det + ld

        x, ld = self.inv_conv(x)
        log_det = log_det + ld

        x, ld = self.coupling(x)
        log_det = log_det + ld

        return x, log_det

    def inverse(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass through flow step."""
        log_det = torch.zeros(x.shape[0], device=x.device)

        x, ld = self.coupling.inverse(x)
        log_det = log_det + ld

        x, ld = self.inv_conv.inverse(x)
        log_det = log_det + ld

        x, ld = self.act_norm.inverse(x)
        log_det = log_det + ld

        return x, log_det


class Squeeze(nn.Module):
    """Squeeze layer that increases channels while reducing spatial dimensions."""

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        return (
            x.reshape(b, c, h // 2, 2, w // 2, 2)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(b, c * 4, h // 2, w // 2)
        )

    def inverse(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        return (
            x.reshape(b, c // 4, 2, 2, h, w)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(b, c // 4, h * 2, w * 2)
        )


class Glow(nn.Module):
    """Glow: Flow-based generative model for image generation.

    Args:
        num_channels: Number of input/output channels
        hidden_channels: Hidden dimension for coupling networks
        num_levels: Number of flow levels
        num_steps: Number of flow steps per level
        image_size: Input image size
    """

    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 512,
        num_levels: int = 3,
        num_steps: int = 16,
        image_size: int = 32,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_levels = num_levels
        self.num_steps = num_steps

        self.flows = nn.ModuleList()

        current_channels = num_channels
        current_size = image_size

        for level in range(num_levels):
            self.flows.append(Squeeze())
            current_channels *= 4
            current_size //= 2

            for _ in range(num_steps):
                self.flows.append(FlowStep(current_channels, hidden_channels))

        self.final_norm = ActNorm(current_channels)

        self.prior = nn.Sequential(
            nn.Conv2d(current_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, current_channels * 2, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode image to latent space."""
        log_det = torch.zeros(x.shape[0], device=x.device)

        for flow in self.flows:
            x, ld = flow(x)
            log_det = log_det + ld

        x, ld = self.final_norm(x)
        log_det = log_det + ld

        params = self.prior(x)
        log_p = -0.5 * (params[:, ::2] ** 2 + params[:, 1::2] ** 2).sum(dim=[1, 2, 3])

        return x, log_det + log_p

    def inverse(self, z: Tensor) -> Tensor:
        """Decode latent to image space."""
        z, ld = self.final_norm.inverse(z)

        for flow in reversed(self.flows):
            z, _ = flow.inverse(z)

        return z

    def sample(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        """Generate samples from the model."""
        batch_size = num_samples
        current_channels = self.num_channels * (4**self.num_levels)

        z = torch.randn(
            batch_size,
            current_channels,
            self.image_size // (2**self.num_levels),
            self.image_size // (2**self.num_levels),
            device=device,
        )

        params = self.prior(z)
        log_p = -0.5 * (params[:, ::2] ** 2 + params[:, 1::2] ** 2)

        z, ld = self.final_norm.inverse(z)

        for flow in reversed(self.flows):
            z, _ = flow.inverse(z)

        return z


class MaskedLinear(nn.Module):
    """Masked linear layer for RealNVP."""

    def __init__(self, in_features: int, out_features: int, mask: Tensor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x * self.mask) * self.mask


class RealNVPCoupling(nn.Module):
    """Coupling layer for RealNVP with masked architecture."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 1024,
        mask: Optional[Tensor] = None,
    ):
        super().__init__()
        self.register_buffer("mask", mask)

        self.net = nn.Sequential(
            nn.Linear(in_features // 2, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, in_features),
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through coupling layer."""
        x_masked = x * self.mask
        params = self.net(x_masked)

        log_scale, shift = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale)

        x_out = x.clone()
        x_out = x_out * (1 - self.mask) + (x * torch.exp(log_scale) + shift) * self.mask

        log_det = (log_scale * (1 - self.mask)).sum(dim=-1)

        return x_out, log_det

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass through coupling layer."""
        y_masked = y * self.mask
        params = self.net(y_masked)

        log_scale, shift = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale)

        x_out = y.clone()
        x_out = x_out * self.mask + ((y - shift) * torch.exp(-log_scale)) * (
            1 - self.mask
        )

        log_det = -(log_scale * (1 - self.mask)).sum(dim=-1)

        return x_out, log_det


class RealNVP(nn.Module):
    """Real-valued Non-Volume Preserving (RealNVP) flow.

    Args:
        num_channels: Number of input channels
        image_size: Size of input images
        num_couplings: Number of coupling layers
        hidden_features: Hidden dimension for coupling networks
    """

    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 32,
        num_couplings: int = 8,
        hidden_features: int = 1024,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.image_size = image_size

        num_pixels = num_channels * image_size * image_size

        self.couplings = nn.ModuleList()
        masks = []

        for i in range(num_couplings):
            mask = torch.arange(num_pixels) % 2 == (i % 2)
            mask = mask.float().view(1, -1)

            self.couplings.append(RealNVPCoupling(num_pixels, hidden_features, mask))
            masks.append(mask)

        self.register_buffer("masks", torch.stack(masks))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through RealNVP."""
        b, c, h, w = x.shape
        x_flat = x.view(b, -1)

        log_det = torch.zeros(b, device=x.device)

        for coupling in self.couplings:
            x_flat, ld = coupling(x_flat)
            log_det = log_det + ld

        log_prob = -0.5 * (x_flat**2 + math.log(2 * math.pi)).sum(dim=-1)

        return x_flat, log_det + log_prob

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse pass through RealNVP."""
        b = z.shape[0]

        for coupling in reversed(self.couplings):
            z, _ = coupling.inverse(z)

        return z.view(b, self.num_channels, self.image_size, self.image_size)

    def sample(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        """Generate samples from the model."""
        num_pixels = self.num_channels * self.image_size * self.image_size
        z = torch.randn(num_samples, num_pixels, device=device)

        return self.inverse(z)


class VariationalDequantization(nn.Module):
    """Variational dequantization for Flow++."""

    def __init__(
        self,
        num_channels: int,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_channels, 3, padding=1),
        )

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        """Transform uniform noise to data-dependent noise."""
        h = torch.cat([x, u], dim=1)
        return u + self.conv(h)


class FlowPlusPlus(nn.Module):
    """Flow++: Improved Flow-Based Generative Model.

    Args:
        num_channels: Number of input channels
        image_size: Size of input images
        num_levels: Number of multiscale levels
        num_steps: Number of flow steps per level
    """

    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 32,
        num_levels: int = 3,
        num_steps: int = 8,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.image_size = image_size

        self.dequantization = VariationalDequantization(num_channels)

        self.flows = nn.ModuleList()

        current_channels = num_channels
        current_size = image_size

        for level in range(num_levels):
            self.flows.append(Squeeze())
            current_channels *= 4
            current_size //= 2

            for _ in range(num_steps):
                self.flows.append(FlowStep(current_channels, hidden_channels=256))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through Flow++."""
        u = torch.rand_like(x)
        x = self.dequantization(x, u)

        log_det = -torch.log(u + 1e-8).sum(dim=[1, 2, 3])

        for flow in self.flows:
            x, ld = flow(x)
            log_det = log_det + ld

        return x, log_det

    def sample(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        """Generate samples from the model."""
        current_channels = self.num_channels * (4**3)

        z = torch.randn(
            num_samples,
            current_channels,
            self.image_size // 8,
            self.image_size // 8,
            device=device,
        )

        for flow in reversed(self.flows):
            z, _ = flow.inverse(z)

        return torch.sigmoid(z)
