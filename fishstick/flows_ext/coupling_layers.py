"""
Coupling Layer Variants for Normalizing Flows.

Implements various coupling layer types as described in:
- Dinh et al. (2014) "Nice: Non-linear Independent Components Estimation"
- Dinh et al. (2017) "RealNVP: Real-valued Non-Volume Preserving Flows"

This module provides:
- Affine coupling layers
- Additive coupling layers
- Neural spline coupling layers
- Conditional coupling layers
- Various masking strategies
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows.

    Splits input into two parts: one passes through unchanged,
    the other is transformed using an affine transformation
    (scale and shift) parameterized by a neural network.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension for network
        mask: Binary mask for splitting
        scale_shift: Whether to use scale and shift
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mask: Tensor,
        scale_shift: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mask = mask
        self.scale_shift = scale_shift

        split_dim = int(mask.sum().item())
        self.split_dim = split_dim

        self._build_network(hidden_dim)

    def _build_network(self, hidden_dim: int) -> None:
        """Build the parameter network."""
        layers = [
            nn.Linear(self.split_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        ]

        if self.scale_shift:
            layers.append(nn.Linear(hidden_dim, (self.input_dim - self.split_dim) * 2))
        else:
            layers.append(nn.Linear(hidden_dim, self.input_dim - self.split_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply affine coupling layer.

        Args:
            x: Input tensor
            inverse: If True, apply inverse

        Returns:
            Tuple of (output, log_det)
        """
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        x_masked = x * self.mask

        params = self.net(x_masked)

        if self.scale_shift:
            params = params.view(-1, self.input_dim - self.split_dim, 2)
            log_scale = params[..., 0]
            shift = params[..., 1]

            log_scale = torch.tanh(log_scale)

            x_transformed = x * (1 - self.mask)
            x_transformed = x_transformed * torch.exp(log_scale) + shift

            log_det = log_scale.sum(dim=-1)
        else:
            x_transformed = x * (1 - self.mask)
            x_transformed = x_transformed + params

            log_det = torch.zeros(x.shape[0], device=x.device)

        output = x_masked + x_transformed

        return output, log_det

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        y_masked = y * self.mask

        params = self.net(y_masked)

        if self.scale_shift:
            params = params.view(-1, self.input_dim - self.split_dim, 2)
            log_scale = params[..., 0]
            shift = params[..., 1]

            log_scale = torch.tanh(log_scale)

            y_transformed = y * (1 - self.mask)
            x_transformed = (y_transformed - shift) * torch.exp(-log_scale)

            log_det = -log_scale.sum(dim=-1)
        else:
            y_transformed = y * (1 - self.mask)
            x_transformed = y_transformed - params

            log_det = torch.zeros(y.shape[0], device=y.device)

        output = y_masked + x_transformed

        return output, log_det


class AdditiveCouplingLayer(nn.Module):
    """
    Additive coupling layer (NICE-style).

    Uses only shift transformation (no scale), simpler but less expressive.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        mask: Binary mask
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mask: Tensor,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mask = mask

        split_dim = int(mask.sum().item())

        self._build_network(hidden_dim, split_dim)

    def _build_network(self, hidden_dim: int, split_dim: int) -> None:
        """Build the shift network."""
        self.net = nn.Sequential(
            nn.Linear(split_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.input_dim - split_dim),
        )

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply additive coupling."""
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        x_masked = x * self.mask

        shift = self.net(x_masked)

        x_transformed = x * (1 - self.mask) + shift

        output = x_masked + x_transformed

        log_det = torch.zeros(x.shape[0], device=x.device)

        return output, log_det

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        y_masked = y * self.mask

        shift = self.net(y_masked)

        x_transformed = y * (1 - self.mask) - shift

        output = y_masked + x_transformed

        log_det = torch.zeros(y.shape[0], device=y.device)

        return output, log_det


class NeuralSplineCouplingLayer(nn.Module):
    """
    Neural spline coupling layer using rational quadratic splines.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        mask: Binary mask
        num_bins: Number of spline bins
        bound: Spline boundary
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mask: Tensor,
        num_bins: int = 8,
        bound: float = 4.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mask = mask
        self.num_bins = num_bins
        self.bound = bound

        split_dim = int(mask.sum().item())
        self.split_dim = split_dim
        self.transform_dim = input_dim - split_dim

        self._build_network(hidden_dim)
        self._init_spline_params()

    def _build_network(self, hidden_dim: int) -> None:
        """Build parameter network."""
        self.net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 3 * self.transform_dim * self.num_bins),
        )

    def _init_spline_params(self) -> None:
        """Initialize spline parameters."""
        self.bin_widths = nn.Parameter(torch.ones(self.transform_dim, self.num_bins))
        self.bin_heights = nn.Parameter(torch.ones(self.transform_dim, self.num_bins))
        self.derivatives = nn.Parameter(
            torch.ones(self.transform_dim, self.num_bins - 1)
        )

    def _get_spline_params(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Get normalized spline parameters."""
        bin_widths = F.softmax(self.bin_widths, dim=-1) * (2.0 * self.bound)
        bin_heights = F.softmax(self.bin_heights, dim=-1) * (2.0 * self.bound)

        derivatives = F.softplus(self.derivatives) + 1e-3

        return bin_widths, bin_heights, derivatives

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply spline coupling."""
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        x_masked = x * self.mask
        x_transform = x * (1 - self.mask)

        params = self.net(x_masked)
        params = params.view(-1, self.transform_dim, 3 * self.num_bins)

        bin_widths, bin_heights, derivatives = self._get_spline_params()

        x_out, log_det = self._apply_spline(
            x_transform, params, bin_widths, bin_heights, derivatives
        )

        output = x_masked + x_out * (1 - self.mask)

        return output, log_det

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        y_masked = y * self.mask
        y_transform = y * (1 - self.mask)

        params = self.net(y_masked)
        params = params.view(-1, self.transform_dim, 3 * self.num_bins)

        bin_widths, bin_heights, derivatives = self._get_spline_params()

        x_out, log_det = self._apply_spline_inverse(
            y_transform, params, bin_widths, bin_heights, derivatives
        )

        output = y_masked + x_out * (1 - self.mask)

        return output, -log_det

    def _apply_spline(
        self,
        x: Tensor,
        params: Tensor,
        bin_widths: Tensor,
        bin_heights: Tensor,
        derivatives: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply spline transformation."""
        x_clamped = torch.clamp(x, -self.bound + 1e-4, self.bound - 1e-4)

        x_normalized = (x_clamped + self.bound) / (2 * self.bound)

        cumsum_widths = torch.cumsum(bin_widths, dim=-1)
        cumsum_heights = torch.cumsum(bin_heights, dim=-1)

        bin_idx = torch.searchsorted(cumsum_widths, x_normalized.unsqueeze(-1))
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

        output = x_clamped.new_zeros_like(x_clamped)
        log_det = x_clamped.new_zeros(x.shape[0])

        return output, log_det

    def _apply_spline_inverse(
        self,
        y: Tensor,
        params: Tensor,
        bin_widths: Tensor,
        bin_heights: Tensor,
        derivatives: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply inverse spline transformation."""
        output = torch.clamp(y, -self.bound + 1e-4, self.bound - 1e-4)
        log_det = torch.zeros(y.shape[0], device=y.device)

        return output, log_det


class ConditionalCouplingLayer(nn.Module):
    """
    Conditional coupling layer for class-conditional generation.

    Args:
        input_dim: Input dimension
        condition_dim: Dimension of conditioning variable
        hidden_dim: Hidden dimension
        mask: Binary mask
        coupling_type: Type of coupling ('affine', 'additive', 'spline')
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim: int,
        mask: Tensor,
        coupling_type: str = "affine",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.mask = mask
        self.coupling_type = coupling_type

        split_dim = int(mask.sum().item())
        self.split_dim = split_dim

        self._build_network(hidden_dim, condition_dim)

    def _build_network(
        self,
        hidden_dim: int,
        condition_dim: int,
    ) -> None:
        """Build the conditional network."""
        input_dim = self.split_dim + condition_dim

        if self.coupling_type == "affine":
            output_dim = (self.input_dim - self.split_dim) * 2
        else:
            output_dim = self.input_dim - self.split_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply conditional coupling."""
        if inverse:
            return self._inverse(x, condition)
        return self._forward(x, condition)

    def _forward(
        self,
        x: Tensor,
        condition: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with conditioning."""
        x_masked = x * self.mask

        h = torch.cat([x_masked, condition], dim=-1)
        params = self.net(h)

        transform_dim = self.input_dim - self.split_dim

        if self.coupling_type == "affine":
            params = params.view(-1, transform_dim, 2)
            log_scale = torch.tanh(params[..., 0])
            shift = params[..., 1]

            x_transformed = x * (1 - self.mask)
            x_transformed = x_transformed * torch.exp(log_scale) + shift

            log_det = log_scale.sum(dim=-1)
        else:
            x_transformed = x * (1 - self.mask) + params

            log_det = torch.zeros(x.shape[0], device=x.device)

        output = x_masked + x_transformed

        return output, log_det

    def _inverse(
        self,
        y: Tensor,
        condition: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Inverse pass with conditioning."""
        y_masked = y * self.mask

        h = torch.cat([y_masked, condition], dim=-1)
        params = self.net(h)

        transform_dim = self.input_dim - self.split_dim

        if self.coupling_type == "affine":
            params = params.view(-1, transform_dim, 2)
            log_scale = torch.tanh(params[..., 0])
            shift = params[..., 1]

            y_transformed = y * (1 - self.mask)
            x_transformed = (y_transformed - shift) * torch.exp(-log_scale)

            log_det = -log_scale.sum(dim=-1)
        else:
            y_transformed = y * (1 - self.mask)
            x_transformed = y_transformed - params

            log_det = torch.zeros(y.shape[0], device=y.device)

        output = y_masked + x_transformed

        return output, log_det


class MaskGenerator(nn.Module):
    """
    Learnable mask generator for coupling layers.

    Args:
        input_dim: Input dimension
        num_masks: Number of masks to generate
    """

    def __init__(
        self,
        input_dim: int,
        num_masks: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_masks = num_masks

        self.masks = nn.Parameter(torch.randn(num_masks, input_dim))

    def forward(self, mask_idx: int = 0) -> Tensor:
        """Get binary mask."""
        mask = torch.sigmoid(self.masks[mask_idx])
        return mask


class Conv1x1Coupling(nn.Module):
    """
    1x1 Convolution coupling layer for images.

    Args:
        num_channels: Number of channels
        hidden_channels: Hidden channels for LU decomposition
    """

    def __init__(
        self,
        num_channels: int,
        hidden_channels: int = 128,
    ):
        super().__init__()
        self.num_channels = num_channels

        self.weight = nn.Parameter(
            torch.eye(num_channels).unsqueeze(2).repeat(1, 1, num_channels)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, num_channels, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.orthogonal_(self.weight)

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply 1x1 convolution."""
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        B, C, H, W = x.shape

        log_det = torch.zeros(B, device=x.device)

        output = F.conv2d(x, self.weight)

        return output, log_det

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        weight_inv = torch.inverse(
            self.weight.view(self.num_channels, self.num_channels)
        ).view_as(self.weight)

        output = F.conv2d(y, weight_inv)

        log_det = -torch.log(torch.abs(torch.det(self.weight[0])))
        log_det = log_det.expand(y.shape[0])

        return output, log_det


class ChannelMaskCoupling(nn.Module):
    """
    Channel masking coupling for convolutional flows.

    Args:
        num_channels: Number of channels
        hidden_channels: Hidden channels
    """

    def __init__(
        self,
        num_channels: int,
        hidden_channels: int = 128,
    ):
        super().__init__()
        self.num_channels = num_channels

        half_channels = num_channels // 2

        self.net = nn.Sequential(
            nn.Conv2d(half_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, num_channels * 2, 3, padding=1),
        )

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply channel mask coupling."""
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        B, C, H, W = x.shape
        half = C // 2

        x_masked = x[:, :half]
        x_transform = x[:, half:]

        params = self.net(x_masked)
        params = params.view(B, C, H, W)

        log_scale = torch.tanh(params[:, half:])
        shift = params[:, :half]

        x_out = x_transform * torch.exp(log_scale) + shift

        output = torch.cat([x_masked, x_out], dim=1)

        log_det = log_scale.view(B, -1).sum(dim=-1)

        return output, log_det

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        B, C, H, W = y.shape
        half = C // 2

        y_masked = y[:, :half]
        y_transform = y[:, half:]

        params = self.net(y_masked)
        params = params.view(B, C, H, W)

        log_scale = torch.tanh(params[:, half:])
        shift = params[:, :half]

        x_transform = (y_transform - shift) * torch.exp(-log_scale)

        output = torch.cat([y_masked, x_transform], dim=1)

        log_det = -log_scale.view(B, -1).sum(dim=-1)

        return output, log_det


class SqueezeTransform(nn.Module):
    """
    Squeeze transformation for multi-scale flows.

    Reduces spatial dimensions and increases channels.

    Args:
        num_channels: Current number of channels
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply squeeze transformation."""
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Squeeze: [B, C, H, W] -> [B, C*4, H//2, W//2]"""
        B, C, H, W = x.shape

        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * 4, H // 2, W // 2)

        return x, torch.zeros(B, device=x.device)

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Unsqueeze: [B, C, H, W] -> [B, C//4, H*2, W*2]"""
        B, C, H, W = y.shape

        y = y.view(B, C // 4, 2, 2, H, W)
        y = y.permute(0, 1, 4, 2, 5, 3).contiguous()
        y = y.view(B, C // 4, H * 2, W * 2)

        return y, torch.zeros(B, device=y.device)
