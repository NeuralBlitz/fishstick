"""
Utilities and Helper Functions for Neural Operators.

Common utilities including transforms, loss functions, visualization,
and data processing tools for operator learning.
"""

from typing import Optional, Tuple, List, Callable, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class GridGenerator:
    """Generate grids for operator training."""

    @staticmethod
    def generate_1d(
        num_points: int,
        start: float = 0.0,
        end: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Generate 1D grid of points."""
        return torch.linspace(start, end, num_points, device=device)

    @staticmethod
    def generate_2d(
        resolution: Tuple[int, int],
        start: float = 0.0,
        end: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate 2D grid of points."""
        x = torch.linspace(start, end, resolution[0], device=device)
        y = torch.linspace(start, end, resolution[1], device=device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        return xx, yy

    @staticmethod
    def generate_3d(
        resolution: Tuple[int, int, int],
        start: float = 0.0,
        end: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate 3D grid of points."""
        x = torch.linspace(start, end, resolution[0], device=device)
        y = torch.linspace(start, end, resolution[1], device=device)
        z = torch.linspace(start, end, resolution[2], device=device)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        return xx, yy, zz


class SensorSampler:
    """Sample sensors/locations from function domains."""

    @staticmethod
    def uniform(
        num_sensors: int,
        domain_dim: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Uniform random sensor locations."""
        return torch.rand(num_sensors, domain_dim, device=device)

    @staticmethod
    def chebyshev(
        num_sensors: int,
        domain_dim: int,
    ) -> Tensor:
        """Chebyshev nodes for 1D domain."""
        indices = torch.arange(1, num_sensors + 1)
        points = torch.cos((2 * indices - 1) * np.pi / (2 * num_sensors))
        points = (points + 1) / 2
        return points.view(-1, 1).repeat(1, domain_dim)

    @staticmethod
    def sobol(
        num_sensors: int,
        domain_dim: int,
        scramble: bool = True,
    ) -> Tensor:
        """Sobol sequence for quasi-random sampling."""
        try:
            from scipy.stats import qmc

            sampler = qmc.Sobol(d=domain_dim, scramble=scramble)
            samples = sampler.random(num_sensors)
            return torch.from_numpy(samples).float()
        except ImportError:
            return SensorSampler.uniform(num_sensors, None)


class FunctionGenerator:
    """Generate synthetic functions for testing operators."""

    @staticmethod
    def sinusoidal(
        grid: Tensor,
        freq: float = 1.0,
        phase: float = 0.0,
        amplitude: float = 1.0,
    ) -> Tensor:
        """Generate sinusoidal function."""
        return amplitude * torch.sin(2 * np.pi * freq * grid + phase)

    @staticmethod
    def polynomial(
        grid: Tensor,
        coeffs: List[float],
    ) -> Tensor:
        """Generate polynomial function."""
        result = torch.zeros_like(grid[..., :1])
        for i, c in enumerate(coeffs):
            result = result + c * (grid**i)
        return result

    @staticmethod
    def gaussian(
        grid: Tensor,
        mean: float = 0.5,
        std: float = 0.1,
        amplitude: float = 1.0,
    ) -> Tensor:
        """Generate Gaussian function."""
        return amplitude * torch.exp(-((grid - mean) ** 2) / (2 * std**2))

    @staticmethod
    def random_wave(
        grid: Tensor,
        num_modes: int = 5,
        seed: Optional[int] = None,
    ) -> Tensor:
        """Generate random wave function."""
        if seed is not None:
            torch.manual_seed(seed)

        result = torch.zeros_like(grid[..., :1])
        for i in range(1, num_modes + 1):
            freq = i * np.pi
            phase = torch.rand_like(grid[..., :1]) * 2 * np.pi
            amp = 1.0 / i
            result = result + amp * torch.sin(freq * grid + phase)
        return result


class PDEOperator:
    """Apply PDE operators to functions."""

    @staticmethod
    def gradient(
        u: Tensor,
        dx: float = 0.01,
        dim: int = -1,
    ) -> Tensor:
        """Compute spatial gradient."""
        return torch.gradient(u, dim=dim, spacing=dx)[0]

    @staticmethod
    def laplacian(
        u: Tensor,
        dx: float = 0.01,
        dim: int = -1,
    ) -> Tensor:
        """Compute Laplacian."""
        grad_u = PDEOperator.gradient(u, dx, dim)
        return PDEOperator.gradient(grad_u, dx, dim)

    @staticmethod
    def divergence(
        u: Tensor,
        dx: float = 0.01,
    ) -> Tensor:
        """Compute divergence for vector fields."""
        result = torch.zeros_like(u[..., :1])
        for i in range(u.size(-1)):
            result = result + PDEOperator.gradient(u[..., i : i + 1], dx, dim=-1)
        return result


class OperatorNormalizer:
    """Normalize input/output functions for operator learning."""

    def __init__(
        self,
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
        epsilon: float = 1e-8,
    ):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, data: List[Tensor]) -> "OperatorNormalizer":
        """Fit normalizer to data."""
        all_data = torch.cat([d.flatten() for d in data])
        self.mean = all_data.mean()
        self.std = all_data.std()
        return self

    def transform(self, x: Tensor) -> Tensor:
        """Normalize data."""
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Denormalize data."""
        if self.mean is None or self.std is None:
            return x
        return x * (self.std + self.epsilon) + self.mean


class BoundaryCondition:
    """Apply boundary conditions to PDE solutions."""

    @staticmethod
    def dirichlet(
        u: Tensor,
        boundary_values: Dict[str, float],
        grid: Tensor,
    ) -> Tensor:
        """Apply Dirichlet boundary conditions."""
        result = u.clone()
        if "left" in boundary_values:
            result[..., 0] = boundary_values["left"]
        if "right" in boundary_values:
            result[..., -1] = boundary_values["right"]
        return result

    @staticmethod
    def neumann(
        u: Tensor,
        flux_values: Dict[str, float],
        dx: float,
    ) -> Tensor:
        """Apply Neumann boundary conditions (flux)."""
        result = u.clone()
        if "left" in flux_values:
            grad_left = (u[..., 1] - u[..., 0]) / dx
            result[..., 0] = u[..., 0] - flux_values["left"] * dx
        if "right" in flux_values:
            grad_right = (u[..., -1] - u[..., -2]) / dx
            result[..., -1] = u[..., -1] + flux_values["right"] * dx
        return result


class QuadratureRule:
    """Numerical quadrature for integration."""

    @staticmethod
    def trapezoid(
        values: Tensor,
        dx: float = 0.01,
    ) -> Tensor:
        """Trapezoid rule."""
        return torch.sum(values, dim=-1) * dx

    @staticmethod
    def simpson(
        values: Tensor,
        dx: float = 0.01,
    ) -> Tensor:
        """Simpson's rule (requires odd number of points)."""
        if values.size(-1) % 2 == 0:
            values = values[..., :-1]

        n = values.size(-1)
        result = values[..., 0] + values[..., -1]
        result = result + 4 * values[..., 1:-1:2].sum(dim=-1)
        result = result + 2 * values[..., 2:-2:2].sum(dim=-1)
        return result * dx / 3

    @staticmethod
    def gaussian(
        values: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Gaussian quadrature."""
        return torch.sum(values * weights, dim=-1)


class LossScheduler:
    """Dynamic loss weighting for multi-task learning."""

    def __init__(
        self,
        initial_weights: Dict[str, float],
        strategy: str = "inverse",
    ):
        self.weights = initial_weights
        self.strategy = strategy
        self.step_count = 0

    def step(self, losses: Dict[str, Tensor]) -> Dict[str, float]:
        """Update loss weights based on strategy."""
        self.step_count += 1

        if self.strategy == "inverse":
            for key in losses:
                if key in self.weights:
                    self.weights[key] = 1.0 / (self.step_count + 1)
        elif self.strategy == "uncertainties":
            for key in losses:
                if key in self.weights:
                    self.weights[key] = 1.0 / (losses[key].item() ** 2 + 1e-8)

        return self.weights

    def compute_weighted_loss(self, losses: Dict[str, Tensor]) -> Tensor:
        """Compute weighted sum of losses."""
        weighted = sum(
            w * losses[key] for key, w in self.weights.items() if key in losses
        )
        return weighted


class AttentionOperator(nn.Module):
    """Attention mechanism for operator learning."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = x.size()

        if context is None:
            context = x

        Q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(context)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(context)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        )

        return self.out_proj(attn_output)


class ResidualBlock(nn.Module):
    """Residual block for operator networks."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


class OperatorEmbedding(nn.Module):
    """Embedding layer for operator inputs."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        max_positions: int = 1024,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.position_embedding = nn.Embedding(max_positions, embed_dim)
        self.projection = nn.Linear(input_dim, embed_dim)

    def forward(
        self,
        x: Tensor,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = x.size()

        if positions is None:
            positions = (
                torch.arange(seq_len, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        pos_embed = self.position_embedding(positions)
        x_embed = self.projection(x)

        return x_embed + pos_embed


class SpectralPooling1D(nn.Module):
    """Spectral pooling in frequency domain."""

    def __init__(self, num_modes: int):
        super().__init__()
        self.num_modes = num_modes

    def forward(self, x: Tensor) -> Tensor:
        x_ft = torch.fft.rfft(x, dim=-1)

        modes = min(self.num_modes, x_ft.size(-1))
        x_ft_truncated = x_ft[..., :modes]

        padding = x.size(-1) - modes
        x_ft_padded = F.pad(x_ft_truncated, (0, padding))

        return torch.fft.irfft(x_ft_padded, dim=-1, n=x.size(-1))


class DomainPadding(nn.Module):
    """Padding for periodic/non-periodic domains."""

    def __init__(
        self,
        padding_size: int,
        mode: str = "constant",
    ):
        super().__init__()
        self.padding_size = padding_size
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(x, (self.padding_size,) * 2, mode=self.mode)


class ComplexMLP(nn.Module):
    """MLP for processing complex-valued data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim * 2

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim * 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x_real, x_imag = x.real, x.imag
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        output = self.net(x_combined)
        out_real, out_imag = output.chunk(2, dim=-1)
        return torch.complex(out_real, out_imag)


class OperatorEnsemble(nn.Module):
    """Ensemble of multiple operator models."""

    def __init__(self, operators: List[nn.Module]):
        super().__init__()
        self.operators = nn.ModuleList(operators)

    def forward(self, x: Tensor) -> Tensor:
        outputs = [op(x) for op in self.operators]
        return torch.stack(outputs).mean(dim=0)

    def forward_with_uncertainty(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = [op(x) for op in self.operators]
        outputs_stack = torch.stack(outputs)
        mean = outputs_stack.mean(dim=0)
        std = outputs_stack.std(dim=0)
        return mean, std


@dataclass
class OperatorMetrics:
    """Container for operator evaluation metrics."""

    l2_error: float
    relative_error: float
    max_error: float
    inference_time: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "l2_error": self.l2_error,
            "relative_error": self.relative_error,
            "max_error": self.max_error,
            "inference_time": self.inference_time,
        }


class OperatorEvaluator:
    """Evaluate operator predictions against ground truth."""

    @staticmethod
    def compute_metrics(
        predictions: Tensor,
        targets: Tensor,
        inference_time: float,
    ) -> OperatorMetrics:
        """Compute evaluation metrics."""
        l2_error = torch.sqrt(((predictions - targets) ** 2).mean()).item()
        relative_error = (
            (torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8))
            .mean()
            .item()
        )
        max_error = torch.max(torch.abs(predictions - targets)).item()

        return OperatorMetrics(
            l2_error=l2_error,
            relative_error=relative_error,
            max_error=max_error,
            inference_time=inference_time,
        )


__all__ = [
    "GridGenerator",
    "SensorSampler",
    "FunctionGenerator",
    "PDEOperator",
    "OperatorNormalizer",
    "BoundaryCondition",
    "QuadratureRule",
    "LossScheduler",
    "AttentionOperator",
    "ResidualBlock",
    "OperatorEmbedding",
    "SpectralPooling1D",
    "DomainPadding",
    "ComplexMLP",
    "OperatorEnsemble",
    "OperatorMetrics",
    "OperatorEvaluator",
]
