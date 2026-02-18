"""
Comprehensive Point Cloud Models Module for Fishstick

This module implements state-of-the-art point cloud processing architectures
including PointNet, PointNet++, DGCNN, PointCNN, PointTransformer, and PCT.

Features:
- Point-wise MLP architectures
- Hierarchical set abstraction
- Graph-based convolutions
- Attention-based processing
- Classification and segmentation heads
- Utility functions for point cloud operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable, Union
import numpy as np
from math import pi


# =============================================================================
# Section 1: Utility Functions
# =============================================================================


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS)

    Sample npoint points from xyz using farthest point sampling algorithm.

    Args:
        xyz: (B, N, 3) point coordinates
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, num_points, device=device) * 1e10
    farthest = torch.randint(
        0, num_points, (batch_size,), dtype=torch.long, device=device
    )
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance = torch.where(mask, dist, distance)
        farthest = torch.max(distance, -1)[1]

    return centroids


def knn_point(k: int, xyz: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """
    K-Nearest Neighbor search

    Find k nearest neighbors for query points in xyz.

    Args:
        k: number of neighbors
        xyz: (B, N, C) database points
        query: (B, M, C) query points

    Returns:
        idx: (B, M, k) indices of k nearest neighbors
    """
    # Compute pairwise distances
    B, N, C = xyz.shape
    _, M, _ = query.shape

    # (B, M, N)
    dist = -2 * torch.matmul(query, xyz.permute(0, 2, 1))
    dist += torch.sum(query**2, -1).view(B, M, 1)
    dist += torch.sum(xyz**2, -1).view(B, 1, N)

    # Get k nearest neighbors
    _, idx = torch.topk(-dist, k, dim=-1)
    return idx


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Ball query - find points within radius

    Find all points within radius of each query point.

    Args:
        radius: search radius
        nsample: maximum number of points to sample
        xyz: (B, N, 3) database points
        new_xyz: (B, M, 3) query points

    Returns:
        idx: (B, M, nsample) indices of points within radius
    """
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    _, npoint, _ = new_xyz.shape

    # Compute distances
    dist = -2 * torch.matmul(new_xyz, xyz.permute(0, 2, 1))
    dist += torch.sum(new_xyz**2, -1).view(batch_size, npoint, 1)
    dist += torch.sum(xyz**2, -1).view(batch_size, 1, num_points)

    # Find points within radius
    dist = torch.sqrt(torch.clamp(dist, min=0))
    mask = dist <= radius

    # Get indices
    dist[~mask] = 1e10
    _, idx = torch.topk(-dist, nsample, dim=-1, largest=True, sorted=False)

    # Handle case where fewer points are within radius
    mask_count = mask.sum(dim=-1)
    for b in range(batch_size):
        for m in range(npoint):
            if mask_count[b, m] < nsample:
                idx[b, m, mask_count[b, m] :] = idx[b, m, 0]

    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index points using indices

    Args:
        points: (B, N, C) points
        idx: (B, M, K) or (B, M) indices

    Returns:
        new_points: (B, M, K, C) or (B, M, C) indexed points
    """
    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(batch_size, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Compute squared Euclidean distance between two sets of points

    Args:
        src: (B, N, C)
        dst: (B, M, C)

    Returns:
        dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


# =============================================================================
# Section 2: Sampling Methods
# =============================================================================


class FPSampler(nn.Module):
    """Farthest Point Sampling module"""

    def __init__(self, npoint: int):
        super().__init__()
        self.npoint = npoint

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3)
        Returns:
            new_xyz: (B, npoint, 3)
            idx: (B, npoint)
        """
        idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, idx)
        return new_xyz, idx


class RandomSampler(nn.Module):
    """Random point sampling"""

    def __init__(self, npoint: int):
        super().__init__()
        self.npoint = npoint

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3)
        Returns:
            new_xyz: (B, npoint, 3)
            idx: (B, npoint)
        """
        device = xyz.device
        batch_size, num_points, _ = xyz.shape
        idx = torch.randperm(num_points, device=device)[: self.npoint]
        idx = idx.unsqueeze(0).repeat(batch_size, 1)
        new_xyz = index_points(xyz, idx)
        return new_xyz, idx


# =============================================================================
# Section 3: Grouping Methods
# =============================================================================


class BallQuery(nn.Module):
    """Ball query grouping"""

    def __init__(self, radius: float, nsample: int):
        super().__init__()
        self.radius = radius
        self.nsample = nsample

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)
            new_xyz: (B, M, 3)
        Returns:
            idx: (B, M, nsample)
        """
        return query_ball_point(self.radius, self.nsample, xyz, new_xyz)


class KNNGroup(nn.Module):
    """KNN grouping"""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)
            new_xyz: (B, M, 3)
        Returns:
            idx: (B, M, k)
        """
        return knn_point(self.k, xyz, new_xyz)


# =============================================================================
# Section 4: Set Abstraction Layers
# =============================================================================


class SA_Layer(nn.Module):
    """
    Set Abstraction Layer (PointNet++ style)

    Performs:
    1. Sampling (FPS)
    2. Grouping (Ball query or KNN)
    3. PointNet on each group
    """

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        group_all: bool = False,
        use_knn: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_knn = use_knn

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, C) or None
        Returns:
            new_xyz: (B, npoint, 3)
            new_points: (B, npoint, mlp[-1])
        """
        xyz = xyz.contiguous()

        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            grouped_xyz = xyz.unsqueeze(1)
            if points is not None:
                grouped_points = points.unsqueeze(1)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
        else:
            # Sampling
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)

            # Grouping
            if self.use_knn:
                idx = knn_point(self.nsample, xyz, new_xyz)
            else:
                idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)

            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

        # PointNet
        grouped_points = grouped_points.permute(0, 3, 1, 2)  # (B, C, npoint, nsample)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        new_points = torch.max(grouped_points, 3)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])

        return new_xyz, new_points


class FP_Layer(nn.Module):
    """
    Feature Propagation Layer (PointNet++ style)

    Propagates features from coarse to fine point sets using interpolation.
    """

    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: Optional[torch.Tensor],
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            xyz1: (B, N, 3) - fine points
            xyz2: (B, M, 3) - coarse points
            points1: (B, N, C1) or None - fine features
            points2: (B, M, C2) - coarse features
        Returns:
            new_points: (B, N, mlp[-1])
        """
        B, N, C = xyz1.shape
        _, M, _ = xyz2.shape

        if M == 1:
            # Global feature
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # Interpolate using 3-nearest neighbors
            dists = square_distance(xyz1, xyz2)
            dists, idx = torch.topk(dists, 3, dim=-1, largest=False, sorted=False)
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points.permute(0, 2, 1)


# =============================================================================
# Section 5: Pooling Methods
# =============================================================================


class MaxPooling(nn.Module):
    """Max pooling over points"""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) or (B, C, N)
        Returns:
            pooled: (B, C)
        """
        return torch.max(x, self.dim)[0]


class AvgPooling(nn.Module):
    """Average pooling over points"""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) or (B, C, N)
        Returns:
            pooled: (B, C)
        """
        return torch.mean(x, self.dim)


class AttentionPooling(nn.Module):
    """Attention-based pooling"""

    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.q_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N)
        Returns:
            pooled: (B, C)
        """
        B, C, N = x.shape
        q = self.q_conv(x).max(dim=-1, keepdim=True)[0]  # (B, C, 1)
        k = self.k_conv(x)  # (B, C, N)
        v = self.v_conv(x)  # (B, C, N)

        attention = self.softmax(torch.bmm(q.transpose(1, 2), k))  # (B, 1, N)
        x = torch.bmm(v, attention.transpose(1, 2)).squeeze(-1)  # (B, C)

        return x


# =============================================================================
# Section 6: Convolutions
# =============================================================================


class PointConv(nn.Module):
    """
    Point Convolution

    Learns a weight function for each point based on its position
    and applies it to the features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nsample: int = 32,
        weight_hidden: List[int] = [32, 32],
    ):
        super().__init__()
        self.nsample = nsample

        # Weight network (MLP on relative positions)
        weight_layers = []
        last_dim = 3  # relative position
        for hidden_dim in weight_hidden:
            weight_layers.extend(
                [nn.Linear(last_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
            )
            last_dim = hidden_dim
        weight_layers.append(nn.Linear(last_dim, in_channels * out_channels))
        self.weight_net = nn.Sequential(*weight_layers)

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(
        self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, C_in) or None
        Returns:
            out: (B, N, C_out)
        """
        B, N, _ = xyz.shape

        # Sample and group
        idx = knn_point(self.nsample, xyz, xyz)
        grouped_xyz = index_points(xyz, idx)  # (B, N, nsample, 3)
        relative_xyz = grouped_xyz - xyz.unsqueeze(2)  # (B, N, nsample, 3)

        if points is not None:
            grouped_points = index_points(points, idx)  # (B, N, nsample, C_in)
        else:
            grouped_points = relative_xyz

        C_in = grouped_points.shape[-1]
        C_out = self.weight_net[-1].out_features // C_in

        # Compute weights
        relative_xyz_flat = relative_xyz.view(-1, 3)  # (B*N*nsample, 3)
        weights = self.weight_net(relative_xyz_flat)  # (B*N*nsample, C_in*C_out)
        weights = weights.view(B, N, self.nsample, C_in, C_out)

        # Apply convolution
        grouped_points = grouped_points.unsqueeze(-1)  # (B, N, nsample, C_in, 1)
        out = torch.sum(grouped_points * weights, dim=3)  # (B, N, nsample, C_out)
        out = torch.max(out, dim=2)[0]  # (B, N, C_out)

        out = out.permute(0, 2, 1)  # (B, C_out, N)
        out = self.bn(out)
        out = out.permute(0, 2, 1)  # (B, N, C_out)

        return out


class KPConv(nn.Module):
    """
    Kernel Point Convolution

    Uses a set of kernel points to define convolution kernels in 3D space.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kpoints: int = 15,
        radius: float = 0.5,
        nsample: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kpoints = num_kpoints
        self.radius = radius
        self.nsample = nsample

        # Initialize kernel points (sphere sampling)
        self.kernel_points = nn.Parameter(self._init_sphere_points(num_kpoints))

        # Weights for each kernel point
        self.weights = nn.Parameter(
            torch.randn(num_kpoints, in_channels, out_channels) * 0.1
        )

        self.bn = nn.BatchNorm1d(out_channels)

    def _init_sphere_points(self, n: int) -> torch.Tensor:
        """Initialize points on a unit sphere using Fibonacci sphere"""
        points = []
        phi = pi * (3.0 - np.sqrt(5.0))  # Golden angle

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])

        return torch.tensor(points, dtype=torch.float32)

    def forward(
        self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, C_in) or None
        Returns:
            out: (B, N, C_out)
        """
        B, N, _ = xyz.shape
        device = xyz.device

        # Sample and group
        idx = knn_point(self.nsample, xyz, xyz)
        grouped_xyz = index_points(xyz, idx)  # (B, N, nsample, 3)
        relative_xyz = grouped_xyz - xyz.unsqueeze(2)  # (B, N, nsample, 3)

        if points is not None:
            grouped_points = index_points(points, idx)  # (B, N, nsample, C_in)
        else:
            grouped_points = relative_xyz
            if self.in_channels != 3:
                grouped_points = F.linear(
                    grouped_points, torch.randn(3, self.in_channels, device=device)
                )

        # Normalize by radius
        relative_xyz = relative_xyz / self.radius

        # Compute correlation between points and kernel points
        # kernel_points: (K, 3)
        kernel_points = self.kernel_points.to(device)

        # (B, N, nsample, 1, 3) - (1, 1, 1, K, 3) = (B, N, nsample, K, 3)
        diff = relative_xyz.unsqueeze(3) - kernel_points.view(1, 1, 1, -1, 3)
        correlation = F.relu(1.0 - torch.norm(diff, dim=-1))  # (B, N, nsample, K)

        # Apply convolution: sum over kernel points
        # correlation: (B, N, nsample, K)
        # weights: (K, C_in, C_out)
        # grouped_points: (B, N, nsample, C_in)

        out = torch.zeros(B, N, self.out_channels, device=device)
        for k in range(self.num_kpoints):
            corr_k = correlation[..., k]  # (B, N, nsample)
            weighted_features = grouped_points * corr_k.unsqueeze(
                -1
            )  # (B, N, nsample, C_in)
            features_k = weighted_features.sum(dim=2)  # (B, N, C_in)
            out += torch.matmul(features_k, self.weights[k])  # (B, N, C_out)

        out = out.permute(0, 2, 1)
        out = self.bn(out)
        out = out.permute(0, 2, 1)

        return out


class PAConv(nn.Module):
    """
    Point Attention Convolution

    Uses attention mechanism to learn adaptive weights for point convolution.
    """

    def __init__(
        self, in_channels: int, out_channels: int, nsample: int = 32, num_heads: int = 4
    ):
        super().__init__()
        self.nsample = nsample
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        assert out_channels % num_heads == 0, (
            "out_channels must be divisible by num_heads"
        )

        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)
        )

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, C_in)
        Returns:
            out: (B, N, C_out)
        """
        B, N, C_in = points.shape

        # Sample and group
        idx = knn_point(self.nsample, xyz, xyz)
        grouped_xyz = index_points(xyz, idx)
        relative_xyz = grouped_xyz - xyz.unsqueeze(2)
        grouped_points = index_points(points, idx)

        # Compute attention
        q = self.q_proj(points)  # (B, N, C_out)
        k = self.k_proj(grouped_points)  # (B, N, nsample, C_out)
        v = self.v_proj(grouped_points)  # (B, N, nsample, C_out)
        pos_encoding = self.pos_proj(relative_xyz)  # (B, N, nsample, C_out)

        # Multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim)  # (B, N, num_heads, head_dim)
        k = k.view(B, N, self.nsample, self.num_heads, self.head_dim)
        v = v.view(B, N, self.nsample, self.num_heads, self.head_dim)
        pos_encoding = pos_encoding.view(
            B, N, self.nsample, self.num_heads, self.head_dim
        )

        # Add positional encoding to keys
        k = k + pos_encoding

        # Attention scores
        q = q.unsqueeze(2)  # (B, N, 1, num_heads, head_dim)
        attn = torch.sum(q * k, dim=-1) / np.sqrt(
            self.head_dim
        )  # (B, N, nsample, num_heads)
        attn = F.softmax(attn, dim=2)

        # Apply attention
        out = torch.sum(attn.unsqueeze(-1) * v, dim=2)  # (B, N, num_heads, head_dim)
        out = out.view(B, N, -1)

        out = out.permute(0, 2, 1)
        out = self.bn(out)
        out = out.permute(0, 2, 1)

        return out


class AdaptiveConv(nn.Module):
    """
    Adaptive Convolution

    Learns adaptive convolution kernels based on local point distribution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nsample: int = 32,
        num_adaptive_weights: int = 4,
    ):
        super().__init__()
        self.nsample = nsample
        self.num_adaptive_weights = num_adaptive_weights

        # Adaptive weight generator
        self.weight_gen = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, num_adaptive_weights),
            nn.Softmax(dim=-1),
        )

        # Base kernels
        self.kernels = nn.ParameterList(
            [
                nn.Parameter(torch.randn(in_channels, out_channels) * 0.1)
                for _ in range(num_adaptive_weights)
            ]
        )

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, C_in)
        Returns:
            out: (B, N, C_out)
        """
        B, N, C_in = points.shape

        # Sample and group
        idx = knn_point(self.nsample, xyz, xyz)
        grouped_xyz = index_points(xyz, idx)
        relative_xyz = grouped_xyz - xyz.unsqueeze(2)
        grouped_points = index_points(points, idx)

        # Generate adaptive weights based on local geometry
        # Use mean and std of relative positions as geometric descriptor
        geo_feat = torch.cat(
            [relative_xyz.mean(dim=2), relative_xyz.std(dim=2)], dim=-1
        )  # (B, N, 6)

        # Simplified: use first 3 dims for weight generation
        adaptive_weights = self.weight_gen(geo_feat[:, :, :3])  # (B, N, num_weights)

        # Aggregate features
        features = grouped_points.mean(dim=2)  # (B, N, C_in)

        # Apply adaptive convolution
        out = torch.zeros(B, N, self.kernels[0].shape[1], device=xyz.device)
        for i, kernel in enumerate(self.kernels):
            kernel_out = torch.matmul(features, kernel)
            out += adaptive_weights[:, :, i : i + 1] * kernel_out

        out = out.permute(0, 2, 1)
        out = self.bn(out)
        out = out.permute(0, 2, 1)

        return out


# =============================================================================
# Section 7: Point Processing Architectures
# =============================================================================


class PointNet(nn.Module):
    """
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

    Original PointNet with input and feature transformers.
    """

    def __init__(
        self,
        num_classes: int = 40,
        input_transform: bool = True,
        feature_transform: bool = True,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        # Input transform (T-Net)
        if input_transform:
            self.stn = self._build_tnet(3)

        # MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        # Feature transform (T-Net)
        if feature_transform:
            self.fstn = self._build_tnet(64)

        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def _build_tnet(self, k: int) -> nn.Module:
        """Build transformation network"""
        return nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, N) or (B, N, 3)
        Returns:
            global_feat: (B, 1024) global feature
            point_feat: (B, 64, N) point-wise features for segmentation
        """
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1)  # (B, 3, N)

        batch_size = x.size(0)
        num_points = x.size(2)

        # Input transform
        if self.input_transform:
            transform = self.stn(x)
            transform = transform.view(-1, 3, 3)
            x = torch.bmm(transform, x)

        # MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transform
        if self.feature_transform:
            transform_feat = self.fstn(x)
            transform_feat = transform_feat.view(-1, 64, 64)
            x = torch.bmm(transform_feat, x)
            point_feat = x
        else:
            point_feat = x

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        # Max pooling
        global_feat = torch.max(x, 2, keepdim=False)[0]

        return global_feat, point_feat


class PointNetPP(nn.Module):
    """
    PointNet++: Deep Hierarchical Feature Learning on Point Sets

    Hierarchical point set learning with set abstraction layers.
    """

    def __init__(self, num_classes: int = 40, normal_channel: bool = False):
        super().__init__()
        in_channel = 6 if normal_channel else 3

        # Set Abstraction layers
        self.sa1 = SA_Layer(512, 0.2, 32, in_channel, [64, 64, 128])
        self.sa2 = SA_Layer(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = SA_Layer(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) or (B, 3, N)
        Returns:
            logits: (B, num_classes)
        """
        if x.shape[1] == 3:
            x = x.permute(0, 2, 1)

        B, N, C = x.shape

        # Set Abstraction
        l1_xyz, l1_points = self.sa1(x)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Global feature
        x = l3_points.view(B, 1024)

        # FC layers
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


class DGCNN(nn.Module):
    """
    Dynamic Graph CNN for Learning on Point Clouds

    Uses EdgeConv to capture local geometric structure by constructing
    dynamic graphs in feature space.
    """

    def __init__(
        self,
        num_classes: int = 40,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims

        # EdgeConv layers
        self.conv1 = self._build_edgeconv(6, 64)
        self.conv2 = self._build_edgeconv(64 * 2, 64)
        self.conv3 = self._build_edgeconv(64 * 2, 128)
        self.conv4 = self._build_edgeconv(128 * 2, 256)

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, 1),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Fully connected layers
        self.linear1 = nn.Linear(emb_dims * 2, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def _build_edgeconv(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build EdgeConv block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def knn(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Find k nearest neighbors in feature space"""
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def get_graph_feature(
        self, x: torch.Tensor, k: int = 20, idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Construct graph features using EdgeConv"""
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)

        if idx is None:
            idx = self.knn(x, k)

        device = x.device
        idx_base = (
            torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)

        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, N) or (B, N, 3)
        Returns:
            logits: (B, num_classes)
        """
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)

        x = self.get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        # Global max pooling
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # Global avg pooling
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x


class PointCNN(nn.Module):
    """
    PointCNN: Convolution On X-Transformed Points

    Uses X-transformation to canonicalize point ordering for convolution.
    """

    def __init__(
        self,
        num_classes: int = 40,
        num_points: int = 1024,
        K: int = 8,
        D: int = 1,
        P: int = -1,
        C: int = 64,
    ):
        super().__init__()
        self.num_points = num_points
        self.K = K  # K-nearest neighbors
        self.D = D  # Dilation
        self.P = P  # Representative points
        self.C = C  # Channels

        # XConv blocks
        self.xconv1 = XConv(3, 32, K, D, P, C)
        self.xconv2 = XConv(32, 64, K, D, P // 2, C)
        self.xconv3 = XConv(64, 128, K, D, P // 4, C)
        self.xconv4 = XConv(128, 256, K, D, P // 8, C)
        self.xconv5 = XConv(256, 512, K, D, -1, C)

        # FC layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) or (B, 3, N)
        Returns:
            logits: (B, num_classes)
        """
        if x.shape[1] == 3:
            x = x.permute(0, 2, 1)

        B = x.shape[0]

        # XConv layers
        x = self.xconv1(x)
        x = self.xconv2(x)
        x = self.xconv3(x)
        x = self.xconv4(x)
        x = self.xconv5(x)

        # Global pooling
        x = x.max(dim=1)[0]

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class XConv(nn.Module):
    """X-Convolution block for PointCNN"""

    def __init__(
        self, in_channels: int, out_channels: int, K: int, D: int, P: int, C: int
    ):
        super().__init__()
        self.K = K
        self.D = D
        self.P = P

        # X-transformation
        self.x_trans = nn.Sequential(nn.Conv1d(K, K * K, 1), nn.BatchNorm1d(K * K))

        # Convolution
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C_in)
        Returns:
            out: (B, P, C_out)
        """
        B, N, C_in = x.shape

        # Sample representative points if needed
        if self.P > 0 and self.P < N:
            idx = farthest_point_sample(x[:, :, :3] if C_in >= 3 else x, self.P)
            p = index_points(x, idx)
        else:
            p = x
            self.P = N

        # KNN grouping
        idx = knn_point(self.K, x, p)
        grouped = index_points(x, idx)  # (B, P, K, C_in)

        # X-transformation
        grouped_flat = grouped.view(B * self.P, self.K, C_in)

        # Compute X matrix from first 3 dims (positions)
        positions = grouped[:, :, :, :3] if C_in >= 3 else grouped
        X_input = positions.view(B * self.P, self.K, -1).permute(0, 2, 1)
        if X_input.shape[1] > 1:
            X_input = X_input[:, :1, :]
        X = self.x_trans(X_input.squeeze(1)).view(B * self.P, self.K, self.K)
        X = F.softmax(X, dim=-1)

        # Apply X-transformation
        grouped_transformed = torch.bmm(X, grouped_flat)  # (B*P, K, C_in)
        grouped_transformed = grouped_transformed.max(dim=1)[0]  # (B*P, C_in)
        grouped_transformed = grouped_transformed.view(B, self.P, C_in)

        # Convolution
        out = grouped_transformed.permute(0, 2, 1)
        out = self.conv(out)
        out = out.permute(0, 2, 1)

        return out


class PointTransformer(nn.Module):
    """
    Point Transformer

    Transformer architecture adapted for point cloud processing.
    """

    def __init__(
        self,
        num_classes: int = 40,
        in_channels: int = 3,
        n_blocks: int = 4,
        n_heads: int = 4,
        d_model: int = 256,
        k: int = 16,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.k = k

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Conv1d(in_channels, d_model // 4, 1),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model // 2, 1),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model // 2 if i == 0 else d_model, d_model, n_heads, k
                )
                for i in range(n_blocks)
            ]
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) or (B, 3, N)
        Returns:
            logits: (B, num_classes)
        """
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1)

        # Input embedding
        x = self.input_embedding(x)  # (B, d_model//2, N)
        x = x.permute(0, 2, 1)  # (B, N, d_model//2)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Global pooling
        x = x.max(dim=1)[0]  # (B, d_model)

        # Classification
        x = self.fc(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with local attention"""

    def __init__(self, in_channels: int, out_channels: int, n_heads: int, k: int):
        super().__init__()
        self.k = k

        self.linear_in = nn.Linear(in_channels, out_channels)
        self.norm_in = nn.LayerNorm(out_channels)

        self.attention = PointTransformerLayer(out_channels, n_heads)
        self.norm_attn = nn.LayerNorm(out_channels)

        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.ReLU(),
            nn.Linear(out_channels * 4, out_channels),
        )
        self.norm_ffn = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        Returns:
            out: (B, N, C)
        """
        # Linear projection
        x = self.linear_in(x)
        x = self.norm_in(x)

        # Local attention
        residual = x
        idx = knn_point(self.k, x, x)
        x = self.attention(x, idx)
        x = self.norm_attn(x + residual)

        # FFN
        residual = x
        x = self.ffn(x)
        x = self.norm_ffn(x + residual)

        return x


class PointTransformerLayer(nn.Module):
    """Point Transformer Layer with vector attention"""

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, channels), nn.ReLU(), nn.Linear(channels, channels)
        )

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
            idx: (B, N, K) neighbor indices
        Returns:
            out: (B, N, C)
        """
        B, N, C = x.shape
        K = idx.shape[-1]

        # Project
        q = self.q_proj(x)  # (B, N, C)
        k = self.k_proj(x)  # (B, N, C)
        v = self.v_proj(x)  # (B, N, C)

        # Gather neighbors
        k_neighbors = index_points(k, idx)  # (B, N, K, C)
        v_neighbors = index_points(v, idx)  # (B, N, K, C)

        # Multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim)  # (B, N, num_heads, head_dim)
        k_neighbors = k_neighbors.view(B, N, K, self.num_heads, self.head_dim)
        v_neighbors = v_neighbors.view(B, N, K, self.num_heads, self.head_dim)

        # Compute attention scores
        q = q.unsqueeze(2)  # (B, N, 1, num_heads, head_dim)
        attn = torch.sum(q * k_neighbors, dim=-1) / np.sqrt(
            self.head_dim
        )  # (B, N, K, num_heads)
        attn = F.softmax(attn, dim=2)

        # Apply attention
        out = torch.sum(
            attn.unsqueeze(-1) * v_neighbors, dim=2
        )  # (B, N, num_heads, head_dim)
        out = out.view(B, N, C)

        return out


class PCT(nn.Module):
    """
    Point Cloud Transformer (PCT)

    Global transformer architecture for point cloud processing.
    """

    def __init__(
        self,
        num_classes: int = 40,
        in_channels: int = 3,
        num_points: int = 1024,
        embed_dim: int = 128,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_points = num_points
        self.embed_dim = embed_dim

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, 1),
        )

        # Positional encoding
        self.pos_encoding = nn.Sequential(
            nn.Conv1d(3, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, 1),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [PCTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )

        # Output
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) or (B, 3, N)
        Returns:
            logits: (B, num_classes)
        """
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1)

        # Input embedding
        feature = self.input_embedding(x)  # (B, embed_dim, N)
        pos = self.pos_encoding(x)  # (B, embed_dim, N)

        x = feature + pos
        x = x.permute(0, 2, 1)  # (B, N, embed_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Layer norm
        x = self.norm(x)

        # Global pooling
        x = x.max(dim=1)[0]  # (B, embed_dim)

        # Classification head
        x = self.head(x)

        return x


class PCTBlock(nn.Module):
    """PCT Transformer Block"""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        Returns:
            out: (B, N, C)
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


# =============================================================================
# Section 8: Classification Models
# =============================================================================


class PointNetCls(nn.Module):
    """PointNet for Classification"""

    def __init__(
        self,
        num_classes: int = 40,
        input_transform: bool = True,
        feature_transform: bool = True,
    ):
        super().__init__()
        self.pointnet = PointNet(num_classes, input_transform, feature_transform)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, N) or (B, N, 3)
        Returns:
            logits: (B, num_classes)
        """
        global_feat, _ = self.pointnet(x)

        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return x


class PointNetPPCls(nn.Module):
    """PointNet++ for Classification"""

    def __init__(self, num_classes: int = 40, normal_channel: bool = False):
        super().__init__()
        self.backbone = PointNetPP(num_classes, normal_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) or (B, 3, N)
        Returns:
            logits: (B, num_classes)
        """
        return self.backbone(x)


class DGCNNCls(nn.Module):
    """DGCNN for Classification"""

    def __init__(
        self,
        num_classes: int = 40,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone = DGCNN(num_classes, k, emb_dims, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) or (B, 3, N)
        Returns:
            logits: (B, num_classes)
        """
        return self.backbone(x)


# =============================================================================
# Section 9: Segmentation Models
# =============================================================================


class PointNetSeg(nn.Module):
    """PointNet for Segmentation"""

    def __init__(
        self,
        num_classes: int = 50,
        input_transform: bool = True,
        feature_transform: bool = True,
    ):
        super().__init__()
        self.pointnet = PointNet(num_classes, input_transform, feature_transform)

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, N) or (B, N, 3)
        Returns:
            logits: (B, num_classes, N)
        """
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        num_points = x.size(2)

        global_feat, point_feat = self.pointnet(x)

        # Broadcast global feature to each point
        global_feat = global_feat.unsqueeze(-1).repeat(1, 1, num_points)

        # Concatenate global and point features
        x = torch.cat([point_feat, global_feat], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x


class PointNetPPSeg(nn.Module):
    """PointNet++ for Segmentation"""

    def __init__(self, num_classes: int = 50, normal_channel: bool = False):
        super().__init__()
        in_channel = 6 if normal_channel else 3

        # Encoder
        self.sa1 = SA_Layer(1024, 0.05, 32, in_channel, [32, 32, 64])
        self.sa2 = SA_Layer(256, 0.1, 32, 64 + 3, [64, 64, 128])
        self.sa3 = SA_Layer(64, 0.2, 32, 128 + 3, [128, 128, 256])
        self.sa4 = SA_Layer(16, 0.4, 32, 256 + 3, [256, 256, 512])

        # Decoder
        self.fp4 = FP_Layer(512 + 256, [256, 256])
        self.fp3 = FP_Layer(256 + 128, [256, 256])
        self.fp2 = FP_Layer(256 + 64, [256, 128])
        self.fp1 = FP_Layer(128 + in_channel, [128, 128, 128])

        # Segmentation head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) or (B, 3, N)
        Returns:
            logits: (B, num_classes, N)
        """
        if x.shape[1] == 3:
            x = x.permute(0, 2, 1)

        # Encoder
        l1_xyz, l1_points = self.sa1(x)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Decoder
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        # Handle case where input might be larger than l1_xyz
        if x.shape[1] > l1_xyz.shape[1]:
            x_down = x[:, : l1_xyz.shape[1], :]
        else:
            x_down = x

        x = self.fp1(x_down, l1_xyz, None, l1_points)

        # Segmentation head
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)

        return x


class DGCNNSeg(nn.Module):
    """DGCNN for Segmentation"""

    def __init__(self, num_classes: int = 50, k: int = 20, emb_dims: int = 1024):
        super().__init__()
        self.k = k

        # EdgeConv layers
        self.conv1 = self._build_edgeconv(6, 64)
        self.conv2 = self._build_edgeconv(64 * 2, 64)
        self.conv3 = self._build_edgeconv(64 * 2, 128)
        self.conv4 = self._build_edgeconv(128 * 2, 256)

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, 1),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Segmentation head
        self.seg_conv1 = nn.Sequential(
            nn.Conv1d(emb_dims + 512, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.seg_conv2 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.seg_conv3 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.seg_conv4 = nn.Conv1d(128, num_classes, 1)

        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)

    def _build_edgeconv(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build EdgeConv block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def knn(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Find k nearest neighbors in feature space"""
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def get_graph_feature(
        self, x: torch.Tensor, k: int = 20, idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Construct graph features using EdgeConv"""
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)

        if idx is None:
            idx = self.knn(x, k)

        device = x.device
        idx_base = (
            torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)

        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, N) or (B, N, 3)
        Returns:
            logits: (B, num_classes, N)
        """
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = self.get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        # Concatenate with skip connections
        x = torch.cat([x, x1, x2, x3, x4], dim=1)

        # Segmentation head
        x = self.seg_conv1(x)
        x = self.dp1(x)
        x = self.seg_conv2(x)
        x = self.dp2(x)
        x = self.seg_conv3(x)
        x = self.seg_conv4(x)

        return x


# =============================================================================
# Section 10: Export
# =============================================================================

__all__ = [
    # Utilities
    "farthest_point_sample",
    "knn_point",
    "query_ball_point",
    "index_points",
    "square_distance",
    # Sampling
    "FPSampler",
    "RandomSampler",
    # Grouping
    "BallQuery",
    "KNNGroup",
    # Set Abstraction
    "SA_Layer",
    "FP_Layer",
    # Pooling
    "MaxPooling",
    "AvgPooling",
    "AttentionPooling",
    # Convolutions
    "PointConv",
    "KPConv",
    "PAConv",
    "AdaptiveConv",
    # Architectures
    "PointNet",
    "PointNetPP",
    "DGCNN",
    "PointCNN",
    "PointTransformer",
    "PCT",
    "XConv",
    "TransformerBlock",
    "PointTransformerLayer",
    "PCTBlock",
    # Classification
    "PointNetCls",
    "PointNetPPCls",
    "DGCNNCls",
    # Segmentation
    "PointNetSeg",
    "PointNetPPSeg",
    "DGCNNSeg",
]
