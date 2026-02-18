"""
Steerable CNNs for Geometric Deep Learning.

Implements steerable convolutional networks that maintain equivariance
through learned filters that can be "steered" to any orientation.

Based on:
- Cohen & Welling (2016): Steerable CNNs
- Weiler et al. (2019): Learning 3D Shape with Geometry
- Marcos et al. (2017): Rotation Steerable CNNs
"""

from typing import Optional, Tuple, List, Callable
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class ClebschGordan(nn.Module):
    """
    Clebsch-Gordan coefficients for coupling representations.

    Used in steerable CNNs to combine features from different
    irreducible representations.
    """

    def __init__(self, l_max: int = 2):
        super().__init__()
        self.l_max = l_max
        self._compute_coefficients()

    def _compute_coefficients(self) -> None:
        """Precompute Clebsch-Gordan coefficients."""
        coeffs = {}

        for l1 in range(self.l_max + 1):
            for l2 in range(self.l_max + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.l_max) + 1):
                    key = (l1, l2, L)
                    coeffs[key] = self._compute_cg_coefficient(l1, l2, L)

        self.register_buffer("coeffs", torch.tensor(list(coeffs.keys())))
        self.coeff_dict = coeffs

    def _compute_cg_coefficient(
        self,
        j1: int,
        j2: int,
        j: int,
    ) -> float:
        """Compute Clebsch-Gordan coefficient (simplified)."""
        if abs(j1 - j2) > j or j > j1 + j2:
            return 0.0

        if (j1 + j2 - j) % 2 == 1:
            return 0.0

        return 1.0 / math.sqrt(max(1, j + 1))

    def forward(self, l1: int, l2: int, L: int) -> Tensor:
        """Get CG coefficients for coupling l1 and l2 to L."""
        key = (l1, l2, L)
        return torch.tensor(self.coeff_dict.get(key, 0.0), dtype=torch.float32)


class IrrepRepresentations(nn.Module):
    """
    Irreducible representations of SO(3).

    Handles the transformation of features under rotation.
    """

    def __init__(self, l_max: int = 2):
        super().__init__()
        self.l_max = l_max
        self.dim = (l_max + 1) ** 2

    def rotate_features(
        self,
        features: Tensor,
        angles: Tensor,
    ) -> Tensor:
        """
        Rotate features by given angles.

        Args:
            features: Spherical harmonic features [..., (l_max+1)^2]
            angles: Rotation angles [...]

        Returns:
            Rotated features
        """
        B = features.size(0)
        l_max = self.l_max

        rotated = torch.zeros_like(features)

        idx = 0
        for l in range(l_max + 1):
            dim_l = 2 * l + 1
            feat_l = features[..., idx : idx + dim_l]

            theta = angles
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)

            if l == 0:
                rotated[..., idx : idx + dim_l] = feat_l
            elif l == 1:
                rot = torch.stack(
                    [
                        torch.stack([cos_t, -sin_t], dim=-1),
                        torch.stack([sin_t, cos_t], dim=-1),
                    ],
                    dim=-2,
                )
                rotated[..., idx : idx + dim_l] = torch.einsum(
                    "...ij,...j->...i", rot, feat_l
                )
            else:
                rotated[..., idx : idx + dim_l] = feat_l

            idx += dim_l

        return rotated


class SteerableFilter(nn.Module):
    """
    Learnable steerable filter bank.

    Filters can be rotated to any orientation, enabling
    equivariance to rotations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        n_orientations: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_orientations = n_orientations

        self.base_filter = nn.Conv2d(
            in_channels,
            out_channels * n_orientations,
            kernel_size,
            padding=kernel_size // 2,
            groups=1,
            bias=False,
        )

        self.angles = torch.linspace(0, 2 * math.pi, n_orientations + 1)[:-1]

    def _create_rotation_matrix(self, angle: float) -> Tensor:
        """Create 2D rotation matrix."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        return torch.tensor(
            [
                [cos_a, -sin_a],
                [sin_a, cos_a],
            ]
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Generate steerable filters.

        Args:
            x: Input features [B, in_channels, H, W]

        Returns:
            List of oriented filters
        """
        base = self.base_filter(x)

        oriented = []
        for i, angle in enumerate(self.angles):
            theta = self._create_rotation_matrix(angle).to(x.device)

            grid = F.affine_grid(
                theta.unsqueeze(0),
                x.shape,
                align_corners=False,
            )

            oriented_filter = F.grid_sample(
                base[:, i * self.out_channels : (i + 1) * self.out_channels],
                grid,
                align_corners=False,
            )
            oriented.append(oriented_filter)

        return oriented


class SteerableConv2D(nn.Module):
    """
    Steerable 2D convolution layer.

    Maintains equivariance to rotations through steerable filters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        n_orientations: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_orientations = n_orientations

        self.filter = SteerableFilter(
            in_channels,
            out_channels,
            kernel_size,
            n_orientations,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply steerable convolution.

        Args:
            x: Input [B, in_channels, H, W]

        Returns:
            Output [B, out_channels, H, W]
        """
        oriented_filters = self.filter(x)

        out = torch.stack(oriented_filters, dim=0).mean(dim=0)

        if self.bias is not None:
            out = out + self.bias

        return out


class SteerableResBlock(nn.Module):
    """
    Steerable residual block.
    """

    def __init__(
        self,
        channels: int,
        n_orientations: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.n_orientations = n_orientations

        self.conv1 = SteerableConv2D(
            channels,
            channels,
            kernel_size=3,
            n_orientations=n_orientations,
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = SteerableConv2D(
            channels,
            channels,
            kernel_size=3,
            n_orientations=n_orientations,
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.act(out)

        return out


class SteerableCNN(nn.Module):
    """
    Complete Steerable CNN architecture.

    Equivariant to rotations in the plane.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_orientations: int = 8,
        n_blocks: int = 4,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            SteerableConv2D(
                in_channels,
                hidden_channels,
                kernel_size=3,
                n_orientations=n_orientations,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList(
            [
                SteerableResBlock(hidden_channels, n_orientations)
                for _ in range(n_blocks)
            ]
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input [B, in_channels, H, W]

        Returns:
            Class logits [B, out_channels]
        """
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        return self.classifier(x)


class EquivariantNonLinearity(nn.Module):
    """
    Pointwise non-linearity that preserves equivariance.
    """

    def __init__(self, l_max: int = 2):
        super().__init__()
        self.l_max = l_max

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply equivariant activation.

        Args:
            x: Input with spherical harmonic structure [..., (l_max+1)^2]

        Returns:
            Activated output
        """
        idx = 0
        out = []

        for l in range(self.l_max + 1):
            dim_l = 2 * l + 1
            feat_l = x[..., idx : idx + dim_l]

            norm = torch.norm(feat_l, dim=-1, keepdim=True)
            norm = F.silu(norm)

            feat_l = feat_l / (norm + 1e-8) * norm

            out.append(feat_l)
            idx += dim_l

        return torch.cat(out, dim=-1)


class GeometricFeatureAggregation(nn.Module):
    """
    Aggregate features across orientations while maintaining equivariance.
    """

    def __init__(
        self,
        channels: int,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.channels = channels
        self.aggregation = aggregation

    def forward(self, x: Tensor) -> Tensor:
        """
        Aggregate oriented features.

        Args:
            x: Features [..., n_orientations, channels]

        Returns:
            Aggregated features [..., channels]
        """
        if self.aggregation == "mean":
            return x.mean(dim=-2)
        elif self.aggregation == "max":
            return x.max(dim=-2)[0]
        elif self.aggregation == "sum":
            return x.sum(dim=-2)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class SteerableAttention(nn.Module):
    """
    Attention mechanism for steerable features.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        self.out_proj = nn.Linear(channels, channels)

    def forward(
        self,
        x: Tensor,
        orientation_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply steerable attention.

        Args:
            x: Features [..., channels]
            orientation_weights: Optional orientation attention [..., n_orientations]

        Returns:
            Attended features
        """
        B = x.size(0)

        q = self.q_proj(x).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, self.num_heads, self.head_dim)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.view(B, self.channels)

        out = self.out_proj(out)

        if orientation_weights is not None:
            out = out * orientation_weights

        return out


class SteerablePool(nn.Module):
    """
    Steerable pooling layer.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: Optional[int] = None,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x: Tensor) -> Tensor:
        """Apply pooling while preserving geometric structure."""
        return self.pool(x)


__all__ = [
    "ClebschGordan",
    "IrrepRepresentations",
    "SteerableFilter",
    "SteerableConv2D",
    "SteerableResBlock",
    "SteerableCNN",
    "EquivariantNonLinearity",
    "GeometricFeatureAggregation",
    "SteerableAttention",
    "SteerablePool",
]
