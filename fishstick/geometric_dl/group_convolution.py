"""
Group Convolutions for Geometric Deep Learning.

Implements equivariant convolutions for various symmetry groups:
- SO(3): 3D rotation group
- O(3): 3D rotation and reflection group
- SE(3): 3D rigid motions (rotations + translations)
- C_n: Cyclic groups
- D_n: Dihedral groups

Based on:
- Cohen & Welling (2016): Group Equivariant Convolutional Networks
- Weiler & Cesa (2019): General E(2)-Equivariant Steerable CNNs
- Thomas et al. (2018): Tensor Field Networks
"""

from typing import Optional, Tuple, List, Callable
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class GroupEquivariantConv(nn.Module):
    """
    Base class for group equivariant convolutions.

    Implements convolution that is equivariant to a group of symmetries.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_size: int,
        kernel_size: int = 3,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_size = group_size
        self.kernel_size = kernel_size

        hidden_dim = hidden_dim or (in_channels + out_channels) // 2

        self.group_conv = nn.Conv2d(
            in_channels * group_size,
            hidden_dim,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.SiLU()

        self.filter_net = nn.Conv2d(hidden_dim, out_channels * group_size, 1)

    def forward(
        self,
        x: Tensor,
        group_elements: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with group convolution.

        Args:
            x: Input features [B, in_channels, H, W]
            group_elements: Precomputed group element transforms

        Returns:
            Group-convolved output [B, out_channels * group_size, H, W]
        """
        B, C, H, W = x.shape

        if group_elements is None:
            group_elements = self._get_group_elements()

        x_expanded = self._expand_by_group(x, group_elements)

        out = self.group_conv(x_expanded)
        out = self.bn(out)
        out = self.act(out)
        out = self.filter_net(out)

        return out

    def _get_group_elements(self) -> Tensor:
        """Get group elements for transformation."""
        raise NotImplementedError

    def _expand_by_group(self, x: Tensor, group_elements: Tensor) -> Tensor:
        """Expand input by group elements."""
        raise NotImplementedError


class SO3EquivariantConv(nn.Module):
    """
    SO(3)-equivariant convolution for 3D point clouds/volumes.

    Maintains equivariance to 3D rotations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        l_max: int = 2,
        radial_hidden_dim: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l_max = l_max
        self.n_sh = (l_max + 1) ** 2

        self.radial_net = nn.Sequential(
            nn.Linear(self.n_sh * in_channels, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, self.n_sh * out_channels),
        )

        self.spherical_harmonics = SphericalHarmonics(l_max)

    def forward(
        self,
        features: Tensor,
        positions: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Compute SO(3)-equivariant features.

        Args:
            features: Node features [N, in_channels]
            positions: 3D coordinates [N, 3]
            edge_index: Edge connectivity [2, E]

        Returns:
            Updated features [N, out_channels]
        """
        row, col = edge_index

        rel_pos = positions[row] - positions[col]
        distances = torch.norm(rel_pos, dim=-1, keepdim=True).clamp(min=1e-8)
        directions = rel_pos / distances

        y_ij = self.spherical_harmonics(directions)

        neighbor_features = features[col]
        combined = torch.cat([y_ij, neighbor_features], dim=-1)

        messages = self.radial_net(combined)

        out = torch.zeros(
            features.size(0), self.out_channels * self.n_sh, device=features.device
        )
        out.index_add_(
            0, row, messages.view(-1, self.n_sh, self.out_channels).sum(dim=1)
        )

        return out


class O3EquivariantConv(nn.Module):
    """
    O(3)-equivariant convolution (includes reflections).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        l_max: int = 2,
        include_reflections: bool = True,
        radial_hidden_dim: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l_max = l_max
        self.include_reflections = include_reflections

        n_sh = (l_max + 1) ** 2
        if include_reflections:
            n_sh = n_sh * 2

        self.radial_net = nn.Sequential(
            nn.Linear(n_sh * in_channels, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, n_sh * out_channels),
        )

        self.spherical_harmonics = SphericalHarmonics(l_max)

    def forward(
        self,
        features: Tensor,
        positions: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Compute O(3)-equivariant features."""
        row, col = edge_index

        rel_pos = positions[row] - positions[col]
        distances = torch.norm(rel_pos, dim=-1, keepdim=True).clamp(min=1e-8)
        directions = rel_pos / distances

        y_ij = self.spherical_harmonics(directions)

        if self.include_reflections:
            y_ij_reflected = self.spherical_harmonics(-directions)
            y_ij = torch.cat([y_ij, y_ij_reflected], dim=-1)

        neighbor_features = features[col]
        combined = torch.cat([y_ij, neighbor_features], dim=-1)

        messages = self.radial_net(combined)

        n_sh = (self.l_max + 1) ** 2
        if self.include_reflections:
            n_sh = n_sh * 2

        out = torch.zeros(
            features.size(0), self.out_channels * n_sh, device=features.device
        )
        out.index_add_(0, row, messages.view(-1, n_sh, self.out_channels).sum(dim=1))

        return out


class SE3EquivariantConv(nn.Module):
    """
    SE(3)-equivariant convolution for 3D point clouds.

    Equivariant to rotations and translations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        l_max: int = 2,
        radial_hidden_dim: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l_max = l_max
        self.n_sh = (l_max + 1) ** 2

        self.radial_net = nn.Sequential(
            nn.Linear(self.n_sh * in_channels + 1, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, self.n_sh * out_channels),
        )

        self.spherical_harmonics = SphericalHarmonics(l_max)

    def forward(
        self,
        features: Tensor,
        positions: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute SE(3)-equivariant features and position updates.

        Args:
            features: Node features [N, in_channels]
            positions: 3D coordinates [N, 3]
            edge_index: Edge connectivity [2, E]

        Returns:
            updated_features: [N, out_channels]
            updated_positions: [N, 3]
        """
        row, col = edge_index

        rel_pos = positions[row] - positions[col]
        distances = torch.norm(rel_pos, dim=-1, keepdim=True).clamp(min=1e-8)
        directions = rel_pos / distances

        y_ij = self.spherical_harmonics(directions)

        neighbor_features = features[col]
        combined = torch.cat([y_ij, neighbor_features, distances], dim=-1)

        messages = self.radial_net(combined)

        out_features = torch.zeros(
            features.size(0), self.out_channels * self.n_sh, device=features.device
        )
        out_features.index_add_(
            0, row, messages.view(-1, self.n_sh, self.out_channels).sum(dim=1)
        )

        coord_net = nn.Linear(self.n_sh * out_channels, 1, bias=False)
        coord_messages = coord_net(messages.view(-1, self.n_sh * self.out_channels))
        coord_update = coord_messages * directions

        out_positions = torch.zeros_like(positions)
        out_positions.index_add_(0, row, coord_update)

        return out_features, out_positions


class CyclicGroupConv(nn.Module):
    """
    C_n group equivariant convolution (rotational symmetry).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n

        self.conv = nn.Conv2d(
            in_channels * n,
            out_channels * n,
            kernel_size,
            padding=kernel_size // 2,
            groups=n,
        )

        self.angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]

    def _rotate(self, x: Tensor, angle: float) -> Tensor:
        """Rotate feature map by angle."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        theta = torch.tensor(
            [
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
            ],
            device=x.device,
            dtype=x.dtype,
        )

        grid = F.affine_grid(
            theta.unsqueeze(0),
            x.shape,
            align_corners=False,
        )

        return F.grid_sample(
            x, grid, align_corners=False, mode="bilinear", padding_mode="border"
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply C_n equivariant convolution.

        Args:
            x: Input [B, in_channels, H, W]

        Returns:
            Output [B, out_channels * n, H, W]
        """
        B, C, H, W = x.shape

        rotated = []
        for angle in self.angles:
            rotated.append(self._rotate(x, angle))

        x_stacked = torch.cat(rotated, dim=1)

        out = self.conv(x_stacked)

        return out


class DihedralGroupConv(nn.Module):
    """
    D_n group equivariant convolution (rotational + reflectional symmetry).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n

        self.conv = nn.Conv2d(
            in_channels * n * 2,
            out_channels * n * 2,
            kernel_size,
            padding=kernel_size // 2,
            groups=n * 2,
        )

        self.angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]

    def _transform(self, x: Tensor, angle: float, flip: bool) -> Tensor:
        """Apply dihedral transformation."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        if flip:
            sin_a = -sin_a

        theta = torch.tensor(
            [
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
            ],
            device=x.device,
            dtype=x.dtype,
        )

        grid = F.affine_grid(
            theta.unsqueeze(0),
            x.shape,
            align_corners=False,
        )

        return F.grid_sample(
            x, grid, align_corners=False, mode="bilinear", padding_mode="border"
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply D_n equivariant convolution.

        Args:
            x: Input [B, in_channels, H, W]

        Returns:
            Output [B, out_channels * n * 2, H, W]
        """
        transformed = []

        for angle in self.angles:
            transformed.append(self._transform(x, angle, flip=False))

        for angle in self.angles:
            transformed.append(self._transform(x, angle, flip=True))

        x_stacked = torch.cat(transformed, dim=1)

        out = self.conv(x_stacked)

        return out


class SphericalHarmonics(nn.Module):
    """
    Spherical harmonics for encoding 3D directions.
    """

    def __init__(self, l_max: int = 2):
        super().__init__()
        self.l_max = l_max

    def forward(self, vectors: Tensor) -> Tensor:
        """
        Compute spherical harmonics up to l_max.

        Args:
            vectors: Direction vectors [..., 3]

        Returns:
            Spherical harmonics features [..., (l_max+1)^2]
        """
        x = vectors[..., 0]
        y = vectors[..., 1]
        z = vectors[..., 2]

        norm = torch.norm(vectors, dim=-1, keepdim=True).clamp(min=1e-8)
        x = x / norm.squeeze(-1)
        y = y / norm.squeeze(-1)
        z = z / norm.squeeze(-1)

        features = []

        if self.l_max >= 0:
            features.append(0.5 * torch.ones_like(x) / math.sqrt(math.pi))

        if self.l_max >= 1:
            features.append(-0.5 * math.sqrt(3 / (2 * math.pi)) * (x + 1j * y).real)
            features.append(0.5 * math.sqrt(3 / math.pi) * z)
            features.append(0.5 * math.sqrt(3 / (2 * math.pi)) * (x - 1j * y).real)

        if self.l_max >= 2:
            features.append(
                0.25 * math.sqrt(15 / (2 * math.pi)) * ((x + 1j * y) ** 2).real
            )
            features.append(
                -0.5 * math.sqrt(15 / (2 * math.pi)) * z * (x + 1j * y).real
            )
            features.append(0.25 * math.sqrt(5 / math.pi) * (2 * z**2 - x**2 - y**2))
            features.append(0.5 * math.sqrt(15 / (2 * math.pi)) * z * (x - 1j * y).real)
            features.append(
                0.25 * math.sqrt(15 / (2 * math.pi)) * ((x - 1j * y) ** 2).real
            )

        return torch.stack(features, dim=-1)


class GroupBatchNorm(nn.Module):
    """Batch normalization for group-convolved features."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(num_groups, num_channels))
        self.bias = nn.Parameter(torch.zeros(num_groups, num_channels))

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize across group dimension.

        Args:
            x: [B, num_groups * num_channels, H, W]

        Returns:
            Normalized output
        """
        B = x.size(0)
        x = x.view(B, self.num_groups, self.num_channels, -1)

        x = (x - x.mean(dim=-1, keepdim=True)) / (
            x.var(dim=-1, keepdim=True) + self.eps
        ).sqrt()

        x = x * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1)

        return x.view(B, -1, *x.shape[-1:])


__all__ = [
    "GroupEquivariantConv",
    "SO3EquivariantConv",
    "O3EquivariantConv",
    "SE3EquivariantConv",
    "CyclicGroupConv",
    "DihedralGroupConv",
    "SphericalHarmonics",
    "GroupBatchNorm",
]
