"""
Point Cloud Processing Module

Core operations for point cloud manipulation including:
- Farthest point sampling (FPS)
- K-Nearest Neighbors (KNN)
- Ball query
- Feature interpolation
- Grouping operations
"""

from typing import Tuple, Optional
import torch
from torch import Tensor
import torch.nn.functional as F


def square_distance(src: Tensor, dst: Tensor) -> Tensor:
    """
    Calculate squared Euclidean distance between two point sets.

    Args:
        src: Source points [N, C] where N is number of points, C is dimension
        dst: Destination points [M, C]

    Returns:
        dist: Squared distances [N, M]
    """
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1, 0))
    dist += torch.sum(src**2, -1).view(N, 1)
    dist += torch.sum(dst**2, -1).view(1, M)
    return dist


def farthest_point_sample(xyz: Tensor, npoint: int) -> Tensor:
    """
    Farthest Point Sampling (FPS).

    Iteratively selects the point that is farthest from the already selected points.
    Provides uniform sampling coverage.

    Args:
        xyz: Point cloud [B, N, 3] or [N, 3]
        npoint: Number of points to sample

    Returns:
        indices: Sampled point indices [B, npoint] or [npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape if xyz.dim() == 3 else (1, *xyz.shape)
    xyz = xyz.view(B, N, C) if xyz.dim() == 3 else xyz.unsqueeze(0)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids.squeeze(0) if centroids.shape[0] == 1 else centroids


def furthest_point_sample(xyz: Tensor, npoint: int) -> Tensor:
    """
    Wrapper for farthest point sample - same functionality.

    Args:
        xyz: Point cloud [B, N, 3]
        npoint: Number of points to sample

    Returns:
        indices: Sampled point indices [B, npoint]
    """
    return farthest_point_sample(xyz, npoint)


def gather_points(points: Tensor, idx: Tensor) -> Tensor:
    """
    Gather points according to indices.

    Args:
        points: Point cloud [B, N, C]
        idx: Indices [B, S] or [S]

    Returns:
        gathered: Gathered points [B, S, C] or [S, C]
    """
    device = points.device
    B, N, C = points.shape
    S = idx.shape[0] if idx.dim() == 1 else idx.shape[1]

    if idx.dim() == 1:
        idx = idx.unsqueeze(0).expand(B, -1)

    view = torch.arange(B, device=device).unsqueeze(-1) * N
    view = view.expand(-1, S)
    index = view + idx

    return points.reshape(B * N, -1)[index].reshape(B, S, C)


def knn_query(k: int, points: Tensor, query_points: Tensor) -> Tuple[Tensor, Tensor]:
    """
    K-Nearest Neighbors search.

    Args:
        k: Number of nearest neighbors
        points: Reference points [B, N, C]
        query_points: Query points [B, S, C]

    Returns:
        dist: Distances [B, S, k]
        group_idx: Indices of k nearest neighbors [B, S, k]
    """
    B, N, C = points.shape
    S = query_points.shape[1]

    dist = square_distance(query_points, points)
    dist, group_idx = torch.topk(dist, k, dim=-1, largest=False, sorted=True)

    return dist, group_idx


def query_ball_point(
    radius: float,
    nsample: int,
    xyz: Tensor,
    new_xyz: Tensor,
) -> Tensor:
    """
    Ball query: find all points within a ball of given radius.

    Args:
        radius: Local region radius
        nsample: Maximum number of points in local region
        xyz: All points [B, N, 3]
        new_xyz: Query points [B, S, 3]

    Returns:
        group_idx: Grouped points indices [B, S, nsample]
    """
    B, N, C = xyz.shape
    S = new_xyz.shape[1]

    group_idx = torch.arange(N, dtype=torch.long, device=xyz.device)
    group_idx = group_idx.view(1, 1, N).repeat([B, S, 1])

    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: Tensor,
    points: Optional[Tensor] = None,
    returnfps: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Sample points and group them into local regions.

    Args:
        npoint: Number of points to sample
        radius: Ball query radius
        nsample: Number of points in each group
        xyz: Point cloud [B, N, 3]
        points: Point features [B, N, D]
        returnfps: Whether to return FPS indices

    Returns:
        new_xyz: Sampled points [B, npoint, 3]
        new_points: Grouped features [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = gather_points(xyz, fps_idx)

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = gather_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = gather_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx

    return new_xyz, new_points


def sample_and_group_all(
    xyz: Tensor,
    points: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Sample all points (no sampling, just grouping).

    Args:
        xyz: Point cloud [B, N, 3]
        points: Point features [B, N, D]

    Returns:
        new_xyz: All points [B, 1, 3]
        new_points: All features [B, 1, N, 3+D]
    """
    B, N, C = xyz.shape

    new_xyz = torch.zeros(B, 1, C, device=xyz.device)
    grouped_xyz = xyz.view(B, 1, N, C)

    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


def pointnet_fp(
    lxyz: Tensor,
    lpoints: Tensor,
    gxyz: Tensor,
    gpoints: Tensor,
) -> Tensor:
    """
    PointNet Feature Propagation layer.

    Propagates features from sparse points to dense points.

    Args:
        lxyz: Local coordinates [B, N, 3]
        lpoints: Local features [B, N, D]
        gxyz: Global coordinates [B, S, 3]
        gpoints: Global features [B, S, D']

    Returns:
        new_points: Interpolated features [B, S, D+D']
    """
    B, N, C = lxyz.shape
    S = gxyz.shape[1]

    dists = square_distance(gxyz, lxyz)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :3], idx[:, :, :3]

    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    interpolated = gather_points(lpoints, idx)
    interpolated = (interpolated * weight.view(B, S, 3, 1)).sum(dim=2)

    if gpoints is not None:
        new_points = torch.cat([gpoints, interpolated], dim=-1)
    else:
        new_points = interpolated

    return new_points


class ThreeNN(nn.Module):
    """
    Find three nearest neighbors for each point.
    """

    def forward(self, unknown: Tensor, known: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Find 3 nearest neighbors.

        Args:
            unknown: Query points [B, N, 3]
            known: Reference points [B, M, 3]

        Returns:
            dist: Distances [B, N, 3]
            idx: Indices [B, N, 3]
        """
        dist, idx = knn_query(3, known, unknown)
        return dist, idx
