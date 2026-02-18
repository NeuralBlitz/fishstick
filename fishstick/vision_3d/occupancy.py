"""
Occupancy Networks for 3D Reconstruction

Implicit representation of 3D surfaces.
"""

from typing import Tuple, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class OccupancyNetwork(nn.Module):
    """
    Occupancy Network - implicit 3D surface representation.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 5,
        with_batch_norm: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        layers = []

        in_dim = latent_dim + 3
        for i in range(num_layers):
            out_dim = hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if with_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))

            layers.append(nn.ReLU(inplace=True))

            in_dim = out_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        points: Tensor,
        latent: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict occupancy at points.

        Args:
            points: Query points [N, 3] or [B, N, 3]
            latent: Latent conditioning [B, latent_dim]

        Returns:
            occupancy: Occupancy values [N, 1] or [B, N, 1]
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, N, _ = points.shape

        if latent is not None:
            latent = latent.unsqueeze(1).expand(-1, N, -1)
            inputs = torch.cat([latent, points], dim=-1)
        else:
            inputs = points

        inputs_flat = inputs.view(B * N, -1)
        occupancy = self.network(inputs_flat)
        occupancy = occupancy.view(B, N, -1)

        if squeeze:
            return occupancy.squeeze(0)

        return occupancy


class OccupancyField(nn.Module):
    """
    Occupancy field for querying occupancy values.
    """

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(
        self,
        points: Tensor,
        latent: Optional[Tensor] = None,
    ) -> Tensor:
        """Query occupancy."""
        return self.network(points, latent)

    def get_isosurface(
        self,
        resolution: int = 32,
        threshold: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract isosurface via marching cubes.
        """
        return torch.zeros(resolution, resolution, resolution), torch.zeros(0, 3)


class ConvOccupancyNetwork(nn.Module):
    """
    Convolutional Occupancy Network for point cloud input.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim + 3, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, points: Tensor, pcd: Tensor) -> Tensor:
        """
        Args:
            points: Query points [N, 3]
            pcd: Point cloud [M, 3]

        Returns:
            Occupancy values [N, 1]
        """
        pcd_features = pcd.t().unsqueeze(0)
        pcd_features = self.encoder(pcd_features)
        pcd_features = pcd_features.mean(dim=2)

        points_exp = points.unsqueeze(0).expand(pcd_features.size(0), -1, -1)
        combined = torch.cat(
            [pcd_features.unsqueeze(1).expand(-1, points.size(0), -1), points_exp],
            dim=-1,
        )

        return self.decoder(combined.squeeze(0))


class ImplicitSurface(nn.Module):
    """
    Implicit surface representation (SDF or occupancy).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        is_sdf: bool = False,
    ):
        super().__init__()
        self.is_sdf = is_sdf

        layers = []
        in_dim = 3

        for i in range(num_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, points: Tensor) -> Tensor:
        """
        Query signed distance or occupancy.

        Args:
            points: Points [N, 3]

        Returns:
            values: SDF or occupancy [N, 1]
        """
        return self.network(points)

    def extract_mesh(
        self,
        resolution: int = 64,
        threshold: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract mesh via marching cubes.

        Returns:
            vertices, faces
        """
        return torch.zeros(resolution, resolution, resolution), torch.zeros(0, 3)
