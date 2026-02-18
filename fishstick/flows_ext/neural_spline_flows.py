"""
Neural Spline Flows Implementation.

Implements Rational Quadratic Spline (RQS) transforms for normalizing flows
as described in:
- Durkan et al. (2019) "Neural Spline Flows"
- Clipper: https://arxiv.org/abs/1906.04032

This module provides:
- Rational quadratic spline transformations
- Neural spline coupling layers
- Learnable knot positions and derivatives
- 1D and 2D spline flows
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


class RationalQuadraticSpline(nn.Module):
    """
    Rational Quadratic Spline transform for 1D data.

    Implements a monotonic rational quadratic spline that can be
    learned and inverted exactly. The spline is parameterized by
    bin widths, heights, and derivatives at knots.

    Args:
        num_bins: Number of bins in the spline
        bound: Boundary for the spline (data range)
        min_derivative: Minimum derivative at knots
    """

    def __init__(
        self,
        num_bins: int = 8,
        bound: float = 4.0,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.bound = bound
        self.min_derivative = min_derivative

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize learnable spline parameters."""
        self.bin_widths = nn.Parameter(
            torch.ones(self.num_bins) * (2.0 * self.bound / self.num_bins)
        )
        self.bin_heights = nn.Parameter(
            torch.ones(self.num_bins) * (2.0 * self.bound / self.num_bins)
        )
        self.derivatives = nn.Parameter(
            torch.ones(self.num_bins - 1) * self.min_derivative
        )

    def _compute_derivatives(self) -> Tensor:
        """Compute padded derivatives with boundary conditions."""
        left_derivatives = F.pad(
            self.derivatives, (1, 0), mode="constant", value=self.min_derivative
        )
        right_derivatives = F.pad(
            self.derivatives, (0, 1), mode="constant", value=self.min_derivative
        )
        return torch.cat([left_derivatives, right_derivatives])

    def _compute_factors(
        self,
        x: Tensor,
        bin_idx: Tensor,
        left_derivative: Tensor,
        right_derivative: Tensor,
        bin_width: Tensor,
        bin_height: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the rational quadratic factors.

        Args:
            x: Input values within bin
            bin_idx: Bin indices for each x
            left_derivative: Left derivative at knot
            right_derivative: Right derivative at knot
            bin_width: Width of bin
            bin_height: Height of bin

        Returns:
            Tuple of (a, b, c) factors
        """
        x_normalized = (x + self.bound) / bin_width

        derivative_ratio = right_derivative / (left_derivative + 1e-8)
        gamma = bin_height / (bin_width + 1e8)

        x_sq = x_normalized**2
        x_gamma = x_normalized * gamma

        a = (derivative_ratio * x_sq) - (2.0 * x_gamma) + derivative_ratio
        b = (
            (2.0 * x_gamma)
            - (derivative_ratio * x_sq)
            - (2.0 * x_normalized)
            + derivative_ratio
        )
        c = x_normalized - 1.0

        return a, b, c

    def _compute_outputs(
        self,
        x: Tensor,
        bin_idx: Tensor,
        left_derivative: Tensor,
        right_derivative: Tensor,
        bin_width: Tensor,
        bin_height: Tensor,
    ) -> Tensor:
        """Compute spline output for each bin."""
        a, b, c = self._compute_factors(
            x, bin_idx, left_derivative, right_derivative, bin_width, bin_height
        )

        discriminant = b**2 - 4 * a * c
        sqrt_disc = torch.sqrt(discriminant + 1e-8)

        x_out = (-b + sqrt_disc) / (2.0 * a + 1e8)
        x_out = x_out * bin_height - self.bound

        return x_out

    def _compute_log_derivative(
        self,
        x: Tensor,
        bin_idx: Tensor,
        left_derivative: Tensor,
        right_derivative: Tensor,
        bin_width: Tensor,
        bin_height: Tensor,
    ) -> Tensor:
        """Compute log derivative of the spline."""
        a, b, c = self._compute_factors(
            x, bin_idx, left_derivative, right_derivative, bin_width, bin_height
        )

        x_normalized = (x + self.bound) / bin_width
        derivative_ratio = right_derivative / (left_derivative + 1e-8)
        gamma = bin_height / (bin_width + 1e8)

        discriminant = b**2 - 4 * a * c
        sqrt_disc = torch.sqrt(discriminant + 1e-8)

        dx_out_dx = (b / (2.0 * a + 1e8)) + (
            (c * (b + 2.0 * a * x_normalized)) / (2.0 * a * sqrt_disc + 1e8)
        )

        dx_out_dx = dx_out_dx * (bin_height / bin_width)

        return torch.log(dx_out_dx + 1e-8)

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rational quadratic spline transform.

        Args:
            x: Input tensor [batch, dim]
            inverse: If True, apply inverse transform

        Returns:
            Tuple of (output, log_det)
        """
        if inverse:
            return self._inverse(x)

        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward transform: data -> spline -> transformed."""
        x_clamped = torch.clamp(x, -self.bound + 1e-4, self.bound - 1e-4)

        bin_widths = F.softmax(self.bin_widths, dim=0) * (2.0 * self.bound)
        bin_heights = F.softmax(self.bin_heights, dim=0) * (2.0 * self.bound)

        cumsum_widths = torch.cumsum(bin_widths, dim=0) - bin_widths[0]
        cumsum_heights = torch.cumsum(bin_heights, dim=0) - bin_heights[0]

        cumsum_widths = cumsum_widths - self.bound
        cumsum_heights = cumsum_heights - self.bound

        bin_idx = torch.searchsorted(cumsum_widths, x_clamped)
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

        bin_idx_flat = bin_idx.flatten()
        x_flat = x_clamped.flatten()

        widths = bin_widths[bin_idx_flat]
        heights = bin_heights[bin_idx_flat]

        derivatives = self._compute_derivatives()
        left_deriv = derivatives[bin_idx_flat]
        right_deriv = derivatives[bin_idx_flat + 1]

        cumsum_w = cumsum_widths[bin_idx_flat]
        cumsum_h = cumsum_heights[bin_idx_flat]

        x_in_bin = x_flat - cumsum_w

        x_out = self._compute_outputs(
            x_in_bin, bin_idx_flat, left_deriv, right_deriv, widths, heights
        )

        log_deriv = self._compute_log_derivative(
            x_in_bin, bin_idx_flat, left_deriv, right_deriv, widths, heights
        )

        x_out = x_out.reshape_as(x)
        log_deriv = log_deriv.reshape_as(x)

        return x_out, log_deriv.sum(dim=-1)

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse transform: use Newton's method."""
        y_clamped = torch.clamp(y, -self.bound + 1e-4, self.bound - 1e-4)

        bin_widths = F.softmax(self.bin_widths, dim=0) * (2.0 * self.bound)
        bin_heights = F.softmax(self.bin_heights, dim=0) * (2.0 * self.bound)

        cumsum_heights = torch.cumsum(bin_heights, dim=0) - bin_heights[0]
        cumsum_heights = cumsum_heights - self.bound

        bin_idx = torch.searchsorted(cumsum_heights, y_clamped)
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

        bin_idx_flat = bin_idx.flatten()
        y_flat = y_clamped.flatten()

        widths = bin_widths[bin_idx_flat]
        heights = bin_heights[bin_idx_flat]

        derivatives = self._compute_derivatives()
        left_deriv = derivatives[bin_idx_flat]
        right_deriv = derivatives[bin_idx_flat + 1]

        cumsum_h = cumsum_heights[bin_idx_flat]
        y_in_bin = y_flat - cumsum_h

        x_initial = y_in_bin * (widths / (heights + 1e-8))

        for _ in range(10):
            outputs, log_deriv = self._forward(x_initial.reshape_as(y) + y - y_clamped)
            x_initial = x_initial - (outputs.flatten() - y_flat) * widths / (
                heights + 1e-8
            )

        x_out = x_initial.reshape_as(y)

        log_deriv = -log_deriv

        return x_out, log_deriv.sum(dim=-1)


class NeuralSplineFlow(nn.Module):
    """
    Neural Spline Flow using coupling layers with RQS transforms.

    Implements a stack of coupling layers, each using a neural network
    to parameterize rational quadratic spline transforms.

    Args:
        input_dim: Dimension of input data
        hidden_dim: Hidden dimension for coupling network
        num_layers: Number of coupling layers
        num_bins: Number of bins in spline
        bound: Boundary for spline
        num_blocks: Number of residual blocks in coupling network
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_bins: int = 8,
        bound: float = 4.0,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers

        self.coupling_layers = nn.ModuleList()

        for i in range(num_layers):
            mask = self._create_mask(input_dim, i)
            self.coupling_layers.append(
                SplineCouplingLayer(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    mask=mask,
                    num_bins=num_bins,
                    bound=bound,
                    num_blocks=num_blocks,
                )
            )

    def _create_mask(self, dim: int, layer_idx: int) -> Tensor:
        """Create alternating mask for coupling."""
        mask = torch.zeros(dim)
        half_dim = dim // 2
        if layer_idx % 2 == 0:
            mask[:half_dim] = 1.0
        else:
            mask[half_dim:] = 1.0
        return mask

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply neural spline flow.

        Args:
            x: Input tensor [batch, input_dim]
            inverse: If True, apply inverse transform

        Returns:
            Tuple of (output, log_det)
        """
        log_det = torch.zeros(x.shape[0], device=x.device)

        layers = self.coupling_layers if not inverse else reversed(self.coupling_layers)

        for layer in layers:
            x, ld = layer(x, inverse=inverse)
            log_det = log_det + ld

        return x, log_det

    def sample(self, num_samples: int, device: str = "cpu") -> Tensor:
        """
        Generate samples from the flow.

        Args:
            num_samples: Number of samples
            device: Device to generate on

        Returns:
            Generated samples
        """
        z = torch.randn(num_samples, self.input_dim, device=device)
        x, _ = self.forward(z, inverse=True)
        return x

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute log probability of samples.

        Args:
            x: Input samples [batch, input_dim]

        Returns:
            Log probabilities
        """
        z, log_det = self.forward(x, inverse=False)
        log_prob = -0.5 * (z**2).sum(dim=-1) - 0.5 * self.input_dim * np.log(2 * np.pi)
        return log_prob + log_det


class SplineCouplingLayer(nn.Module):
    """
    Spline coupling layer using rational quadratic splines.

    Splits input into two parts: one passes through unchanged,
    the other is transformed by a spline parameterized by a neural network.

    Args:
        input_dim: Dimension of input
        hidden_dim: Hidden dimension for network
        mask: Binary mask [input_dim]
        num_bins: Number of spline bins
        bound: Spline boundary
        num_blocks: Number of residual blocks
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mask: Tensor,
        num_bins: int = 8,
        bound: float = 4.0,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mask = mask
        self.num_bins = num_bins
        self.bound = bound

        self.split_dim = int(mask.sum().item())

        self.net = self._build_network(hidden_dim, num_blocks)

        self.spline = RationalQuadraticSpline(
            num_bins=num_bins,
            bound=bound,
        )

        self._init_weights()

    def _build_network(self, hidden_dim: int, num_blocks: int) -> nn.Module:
        """Build the neural network for spline parameterization."""
        in_dim = self.split_dim

        layers = []
        for _ in range(num_blocks):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            in_dim = hidden_dim

        layers.append(
            nn.Linear(in_dim, 3 * self.num_bins * (self.input_dim - self.split_dim))
        )

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply spline coupling layer.

        Args:
            x: Input tensor
            inverse: If True, apply inverse

        Returns:
            Tuple of (output, log_det)
        """
        x_masked = x * self.mask
        x_pass = x * (1.0 - self.mask)

        params = self.net(x_masked)

        x_transform, log_det = self.spline(x_pass, inverse=inverse)

        output = x_masked + x_transform * (1.0 - self.mask)

        return output, log_det


class ConditionalNeuralSplineFlow(nn.Module):
    """
    Conditional Neural Spline Flow for conditional generation.

    Args:
        input_dim: Dimension of input data
        condition_dim: Dimension of conditioning variable
        hidden_dim: Hidden dimension
        num_layers: Number of coupling layers
        num_bins: Number of spline bins
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_bins: int = 8,
        bound: float = 4.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        self.condition_embedding = nn.Linear(condition_dim, hidden_dim)

        self.flow = NeuralSplineFlow(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_bins=num_bins,
            bound=bound,
        )

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply conditional flow."""
        return self.flow(x, inverse=inverse)

    def sample(
        self,
        num_samples: int,
        condition: Tensor,
        device: str = "cpu",
    ) -> Tensor:
        """Generate conditional samples."""
        z = torch.randn(num_samples, self.input_dim, device=device)
        x, _ = self.forward(z, condition, inverse=True)
        return x

    def log_prob(
        self,
        x: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Compute conditional log probability."""
        z, log_det = self.forward(x, condition, inverse=False)
        log_prob = -0.5 * (z**2).sum(dim=-1) - 0.5 * self.input_dim * np.log(2 * np.pi)
        return log_prob + log_det


class BatchNormFlow(nn.Module):
    """
    Batch normalization for normalizing flows.

    Applies batch normalization that can be inverted for flow operations.

    Args:
        momentum: Momentum for running statistics
        eps: Epsilon for numerical stability
    """

    def __init__(
        self,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))

        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply batch normalization."""
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)

            self.running_mean = (
                self.momentum * mean.squeeze() + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * var.squeeze() + (1 - self.momentum) * self.running_var
            )
        else:
            mean = self.running_mean.view(1, -1)
            var = self.running_var.view(1, -1)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        output = self.weight * x_normalized + self.bias

        log_det = -0.5 * torch.log(var + self.eps) + torch.log(self.weight.abs())
        log_det = log_det.sum().expand(x.shape[0])

        return output, log_det

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        mean = (
            self.running_mean.view(1, -1)
            if not self.training
            else y.mean(dim=0, keepdim=True)
        )
        var = (
            self.running_var.view(1, -1)
            if not self.training
            else y.var(dim=0, keepdim=True, unbiased=False)
        )

        y_centered = (y - self.bias) / self.weight

        x = y_centered * torch.sqrt(var + self.eps) + mean

        log_det = 0.5 * torch.log(var + self.eps) - torch.log(self.weight.abs())
        log_det = log_det.sum().expand(y.shape[0])

        return x, log_det


class ActNorm(nn.Module):
    """
    Activation Normalization layer.

    Per-channel normalization with learnable scale and bias.

    Args:
        num_channels: Number of channels
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

        self.log_scale = nn.Parameter(torch.zeros(1, num_channels))
        self.bias = nn.Parameter(torch.zeros(1, num_channels))

        self.initialized = False

    def initialize(self, x: Tensor) -> None:
        """Initialize parameters from first batch."""
        with torch.no_grad():
            bias = -x.mean(dim=0, keepdim=True)
            var = ((x + bias) ** 2).mean(dim=0, keepdim=True)
            log_scale = -0.5 * torch.log(var + 1e-6)

            self.bias.data = bias
            self.log_scale.data = log_scale

        self.initialized = True

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply activation normalization."""
        if not self.initialized:
            self.initialize(x)

        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        output = (x + self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum(dim=-1).expand(x.shape[0])
        return output, log_det

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        x = y * torch.exp(-self.log_scale) - self.bias
        log_det = -self.log_scale.sum(dim=-1).expand(y.shape[0])
        return x, log_det
