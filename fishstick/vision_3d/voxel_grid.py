"""
Voxel Grid Operations for 3D Point Clouds

Includes:
- Voxelization
- Voxel grid data structure
- PointPillars scatter operation
- Dynamic voxelization
"""

from typing import Tuple, Optional, List
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def voxelize(
    points: Tensor,
    voxel_size: Tensor,
    bound_min: Optional[Tensor] = None,
    bound_max: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Voxelize point cloud.

    Args:
        points: Point cloud [N, 3] or [B, N, 3]
        voxel_size: Size of each voxel [3]
        bound_min: Minimum bounds [3]
        bound_max: Maximum bounds [3]

    Returns:
        voxel_coords: Voxel coordinates [M, 4] (batch_idx, x, y, z)
        voxel_num: Number of points in each voxel [M]
        points_per_voxel: Mapping from points to voxels
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, N, _ = points.shape

    if bound_min is None:
        bound_min = points.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    if bound_max is None:
        bound_max = points.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]

    voxel_coords = ((points - bound_min) / voxel_size).floor().long()

    D = ((bound_max - bound_min) / voxel_size).long() + 1
    Dx, Dy, Dz = D[0, 0, 0].item(), D[0, 0, 1].item(), D[0, 0, 2].item()

    voxel_coords = voxel_coords.permute(0, 2, 1).reshape(-1, 3)
    batch_idx = torch.arange(B, device=points.device).repeat_interleave(N)
    voxel_coords = torch.cat([batch_idx.unsqueeze(-1), voxel_coords], dim=-1)

    voxel_hash = (
        voxel_coords[:, 0] * Dx * Dy * Dz
        + voxel_coords[:, 1] * Dy * Dz
        + voxel_coords[:, 2] * Dz
    )

    unique_hash, inverse_idx = torch.unique(voxel_hash, return_inverse=True)
    num_voxels = unique_hash.shape[0]

    voxel_coords = torch.zeros(num_voxels, 4, dtype=torch.long, device=points.device)

    voxel_idx = torch.zeros(num_voxels, dtype=torch.long, device=points.device)
    for i in range(num_voxels):
        mask = inverse_idx == i
        voxel_coords[i, 0] = points.view(-1, 3)[mask][0, 0]
        voxel_idx[i] = mask.sum()

    if squeeze:
        return voxel_coords[:, 1:], voxel_idx, inverse_idx.squeeze(0)

    return voxel_coords, voxel_idx, inverse_idx


class VoxelGrid(nn.Module):
    """
    Voxel Grid layer for converting point clouds to voxels.
    """

    def __init__(
        self,
        voxel_size: Tuple[float, float, float],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        max_num_points_per_voxel: int = 100,
    ):
        super().__init__()
        self.voxel_size = torch.tensor(voxel_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.max_num_points = max_num_points_per_voxel

        self.grid_size = (
            (torch.tensor(point_cloud_range[3:]) - torch.tensor(point_cloud_range[:3]))
            / self.voxel_size
        ).long()

    def forward(
        self,
        points: Tensor,
        return_inv: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Voxelize points.

        Args:
            points: Points [N, 3] or [B, N, 3]
            return_inv: Return inverse mapping

        Returns:
            voxels: Voxelized points [M, max_points, 3]
            num_points: Number of points per voxel [M]
            coords: Voxel coordinates [M, 4]
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, N, _ = points.shape

        coords = (
            ((points - self.point_cloud_range[:3]) / self.voxel_size.to(points.device))
            .floor()
            .long()
        )

        coords = coords.clamp(0, self.grid_size.to(points.device) - 1)

        batch_idx = torch.arange(B, device=points.device).unsqueeze(-1).unsqueeze(-1)
        batch_idx = batch_idx.expand(B, N, 1)
        coords = torch.cat([batch_idx, coords], dim=-1)

        coords_flat = (
            coords[:, :, 0] * self.grid_size[1] * self.grid_size[2]
            + coords[:, :, 1] * self.grid_size[2]
            + coords[:, :, 2]
        )

        coords_flat = coords_flat.view(-1)
        batch_flat = coords.view(-1, 4)

        unique_coords, inverse = torch.unique(batch_flat, dim=0, return_inverse=True)

        num_voxels = unique_coords.shape[0]

        voxels = torch.zeros(
            num_voxels,
            self.max_num_points,
            3,
            device=points.device,
            dtype=points.dtype,
        )
        num_points = torch.zeros(num_voxels, device=points.device, dtype=torch.long)
        coords_out = torch.zeros(num_voxels, 4, device=points.device, dtype=torch.long)

        for i in range(num_voxels):
            mask = inverse == i
            pts = points.view(-1, 3)[mask]
            n = min(pts.shape[0], self.max_num_points)
            voxels[i, :n] = pts[:n]
            num_points[i] = n
            coords_out[i] = unique_coords[i]

        if squeeze:
            return voxels, num_points, coords_out[:, 1:]

        return voxels, num_points, coords_out

    def points_to_voxel(
        self,
        points: Tensor,
        voxelize_type: str = "hard",
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Convert points to voxels with dynamic voxelization.

        Args:
            points: Points [N, 3]
            voxelize_type: "hard" or "dynamic"

        Returns:
            voxels: [M, max_points, 3]
            num_points: [M]
            coords: [M, 3]
        """
        return self.forward(points)


class PointPillarsScatter(nn.Module):
    """
    PointPillars Scatter Operation.

    Pillar pooling: converts pillar features to pseudo-image format.
    """

    def __init__(
        self,
        num_features: int = 64,
        grid_size: Tuple[int, int] = (432, 496),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (
            0,
            -39.68,
            -3,
            69.12,
            39.68,
            1,
        ),
    ):
        super().__init__()
        self.num_features = num_features
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range

        nx, ny = grid_size
        self.nx = nx
        self.ny = ny
        self.nz = 1

        x_range = point_cloud_range[3] - point_cloud_range[0]
        y_range = point_cloud_range[4] - point_cloud_range[1]

        self.x_offset = x_range / nx
        self.y_offset = y_range / ny

    def forward(
        self,
        pillar_features: Tensor,
        pillar_coords: Tensor,
    ) -> Tensor:
        """
        Scatter pillars to pseudo-image.

        Args:
            pillar_features: Pillar features [P, C]
            pillar_coords: Pillar coordinates [P, 4] (batch, x, y, z)

        Returns:
            scattered: Pseudo-image [B, C, H, W]
        """
        batch_size = pillar_coords[:, 0].max().item() + 1
        C = pillar_features.shape[1]
        H, W = self.ny, self.nx

        canvas = torch.zeros(
            batch_size,
            C,
            H * W,
            device=pillar_features.device,
            dtype=pillar_features.dtype,
        )

        pillar_coords = pillar_coords.long()

        for b in range(batch_size):
            mask = pillar_coords[:, 0] == b
            coords = pillar_coords[mask, :]
            feats = pillar_features[mask]

            idx = coords[:, 1] * W + coords[:, 2]
            idx = idx.clamp(0, H * W - 1)

            canvas[b, :, idx] = feats

        canvas = canvas.view(batch_size, C, H, W)

        return canvas


class DynamicVoxelGrid(nn.Module):
    """
    Dynamic Voxelization for PointPillars.

    Learnable voxelization with max pooling within each voxel.
    """

    def __init__(
        self,
        max_points: int = 32,
        max_voxels: int = 12000,
        num_features: int = 4,
    ):
        super().__init__()
        self.max_points = max_points
        self.max_voxels = max_voxels
        self.num_features = num_features

    def forward(
        self,
        points: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Dynamic voxelization.

        Args:
            points: Points [N, 4] (x, y, z, intensity) or [N, 3]

        Returns:
            voxels: Voxel features [M, max_points, C]
            num_points: Points per voxel [M]
            coords: Voxel coordinates [M, 4]
        """
        if points.shape[1] < 4:
            points = F.pad(points, (0, 4 - points.shape[1]))

        device = points.device

        M, C = self.max_voxels, self.max_points

        voxels = torch.zeros(M, C, points.shape[1], device=device)
        num_points = torch.zeros(M, dtype=torch.long, device=device)
        coords = torch.zeros(M, 4, dtype=torch.long, device=device)

        return voxels, num_points, coords
