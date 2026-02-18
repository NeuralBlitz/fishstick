"""
Fishstick 3D Vision Module - Point Cloud Processing and Deep Learning

Comprehensive 3D vision capabilities including point cloud processing,
deep learning architectures for classification, detection, segmentation,
and 3D reconstruction.

Author: Fishstick Team
"""

import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional dependencies
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    warnings.warn("Open3D not installed. Some features will be unavailable.")

try:
    import scipy.spatial.distance as spatial_distance
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not installed. Some features will be unavailable.")

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("Scikit-learn not installed. Some features will be unavailable.")


def _ensure_tensor(x: Union[np.ndarray, torch.Tensor], device: str = 'cpu') -> torch.Tensor:
    """Ensure input is a torch tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    return x.float().to(device)


def _ensure_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Ensure input is a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# =============================================================================
# Point Cloud Processing
# =============================================================================

class PointCloudLoader:
    """Load point clouds from various file formats.
    
    Supports: PLY, PCD, LAS/LAZ, NPY, NPZ, TXT/CSV
    """
    
    def __init__(self):
        self.supported_formats = ['.ply', '.pcd', '.las', '.laz', '.npy', '.npz', '.txt', '.csv']
    
    def load(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        if suffix == '.ply':
            return self._load_ply(filepath)
        elif suffix == '.pcd':
            return self._load_pcd(filepath)
        elif suffix in ['.las', '.laz']:
            return self._load_las(filepath)
        elif suffix == '.npy':
            return self._load_npy(filepath)
        elif suffix == '.npz':
            return self._load_npz(filepath)
        elif suffix in ['.txt', '.csv']:
            return self._load_txt(filepath)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _load_ply(self, filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if HAS_OPEN3D:
            pcd = o3d.io.read_point_cloud(str(filepath))
            points = np.asarray(pcd.points)
            if pcd.has_colors():
                return points, np.asarray(pcd.colors)
            elif pcd.has_normals():
                return points, np.asarray(pcd.normals)
            return points, None
        else:
            raise NotImplementedError("PLY loading requires Open3D")
    
    def _load_pcd(self, filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if HAS_OPEN3D:
            pcd = o3d.io.read_point_cloud(str(filepath))
            points = np.asarray(pcd.points)
            return points, np.asarray(pcd.colors) if pcd.has_colors() else None
        else:
            raise NotImplementedError("PCD loading requires Open3D")
    
    def _load_las(self, filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        try:
            import laspy
            las = laspy.read(filepath)
            points = np.vstack([las.x, las.y, las.z]).T
            attributes = []
            if hasattr(las, 'intensity'):
                attributes.append(las.intensity.reshape(-1, 1))
            if hasattr(las, 'classification'):
                attributes.append(las.classification.reshape(-1, 1))
            if attributes:
                return points, np.hstack(attributes)
            return points, None
        except ImportError:
            raise ImportError("laspy is required for LAS/LAZ files")
    
    def _load_npy(self, filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        data = np.load(filepath)
        if data.ndim == 2 and data.shape[1] >= 3:
            return data[:, :3], data[:, 3:] if data.shape[1] > 3 else None
        raise ValueError(f"Unexpected numpy array shape: {data.shape}")
    
    def _load_npz(self, filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        data = np.load(filepath)
        if 'points' in data:
            points = data['points']
            attributes = data.get('colors') or data.get('normals') or data.get('labels')
            return points, attributes
        keys = list(data.keys())
        arr = data[keys[0]]
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3], arr[:, 3:] if arr.shape[1] > 3 else None
        raise ValueError("Cannot parse NPZ file structure")
    
    def _load_txt(self, filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        delimiter = ',' if filepath.suffix == '.csv' else ' '
        data = np.loadtxt(filepath, delimiter=delimiter)
        if data.ndim == 2 and data.shape[1] >= 3:
            return data[:, :3], data[:, 3:] if data.shape[1] > 3 else None
        raise ValueError(f"Unexpected text file shape: {data.shape}")
    
    def save(self, filepath: Union[str, Path], points: np.ndarray, 
             attributes: Optional[np.ndarray] = None, format: Optional[str] = None):
        filepath = Path(filepath)
        fmt = format or filepath.suffix.lower()
        
        if fmt in ['.ply', 'ply']:
            self._save_ply(filepath, points, attributes)
        elif fmt in ['.pcd', 'pcd']:
            self._save_pcd(filepath, points, attributes)
        elif fmt in ['.npy', 'npy']:
            if attributes is not None:
                np.save(filepath, np.hstack([points, attributes]))
            else:
                np.save(filepath, points)
        else:
            raise ValueError(f"Unsupported save format: {fmt}")
    
    def _save_ply(self, filepath: Path, points: np.ndarray, attributes: Optional[np.ndarray]):
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if attributes is not None and attributes.shape[1] == 3:
                pcd.colors = o3d.utility.Vector3dVector(attributes)
            o3d.io.write_point_cloud(str(filepath), pcd)
        else:
            raise NotImplementedError("PLY saving requires Open3D")
    
    def _save_pcd(self, filepath: Path, points: np.ndarray, attributes: Optional[np.ndarray]):
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if attributes is not None and attributes.shape[1] == 3:
                pcd.colors = o3d.utility.Vector3dVector(attributes)
            o3d.io.write_point_cloud(str(filepath), pcd)
        else:
            raise NotImplementedError("PCD saving requires Open3D")


class PointCloudNormalizer:
    """Normalize point clouds (center, scale, rotate)."""
    
    def center(self, points: np.ndarray) -> np.ndarray:
        centroid = np.mean(points, axis=0)
        return points - centroid
    
    def normalize_scale(self, points: np.ndarray, method: str = 'unit_sphere') -> np.ndarray:
        if method == 'unit_sphere':
            max_dist = np.max(np.linalg.norm(points, axis=1))
            return points / max_dist if max_dist > 0 else points
        elif method == 'unit_cube':
            max_range = np.max(np.ptp(points, axis=0))
            return points / max_range if max_range > 0 else points
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def center_scale(self, points: np.ndarray, method: str = 'unit_sphere') -> np.ndarray:
        centered = self.center(points)
        return self.normalize_scale(centered, method)
    
    def rotate(self, points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        return points @ rotation_matrix.T
    
    def rotate_euler(self, points: np.ndarray, angles: Tuple[float, float, float], 
                     order: str = 'xyz') -> np.ndarray:
        rx, ry, rz = angles
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        mats = {'x': Rx, 'y': Ry, 'z': Rz}
        R = np.eye(3)
        for axis in order:
            R = R @ mats[axis]
        return points @ R.T


class Voxelizer:
    """Convert point clouds to voxel grids."""
    
    def __init__(self, voxel_size: float = 0.05, grid_size: Optional[Tuple[int, int, int]] = None):
        self.voxel_size = voxel_size
        self.grid_size = grid_size
    
    def voxelize(self, points: np.ndarray, features: Optional[np.ndarray] = None,
                 method: str = 'binary') -> np.ndarray:
        points = _ensure_numpy(points)
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        
        if self.grid_size is None:
            grid_size = tuple(((max_coords - min_coords) / self.voxel_size).astype(int) + 1)
        else:
            grid_size = self.grid_size
        
        voxel_indices = ((points - min_coords) / self.voxel_size).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, np.array(grid_size) - 1)
        
        if features is not None:
            n_channels = features.shape[1]
            voxel_grid = np.zeros((n_channels, *grid_size), dtype=np.float32)
        else:
            voxel_grid = np.zeros(grid_size, dtype=np.float32)
        
        for i, (vx, vy, vz) in enumerate(voxel_indices):
            if features is not None:
                if method == 'mean':
                    voxel_grid[:, vx, vy, vz] = (voxel_grid[:, vx, vy, vz] + features[i]) / 2
                elif method == 'max':
                    voxel_grid[:, vx, vy, vz] = np.maximum(voxel_grid[:, vx, vy, vz], features[i])
                else:
                    voxel_grid[:, vx, vy, vz] = features[i]
            else:
                voxel_grid[vx, vy, vz] = 1
        
        return voxel_grid


class PointSampler:
    """Sampling strategies for point clouds."""
    
    def random_sample(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        n_points = len(points)
        if n_points <= n_samples:
            indices = np.random.choice(n_points, n_samples, replace=True)
        else:
            indices = np.random.choice(n_points, n_samples, replace=False)
        return points[indices]
    
    def farthest_point_sample(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        n_points = len(points)
        if n_points <= n_samples:
            return self.random_sample(points, n_samples)
        
        selected_indices = [np.random.randint(n_points)]
        distances = np.full(n_points, np.inf)
        
        for _ in range(1, n_samples):
            last_point = points[selected_indices[-1]]
            dists = np.sum((points - last_point) ** 2, axis=1)
            distances = np.minimum(distances, dists)
            farthest_idx = np.argmax(distances)
            selected_indices.append(farthest_idx)
        
        return points[selected_indices]
    
    def fps(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        return self.farthest_point_sample(points, n_samples)


class PointAugmentor:
    """Data augmentation for point clouds."""
    
    def __init__(self, 
                 jitter_std: float = 0.01,
                 scale_range: Tuple[float, float] = (0.8, 1.2),
                 rotation_range: Tuple[float, float, float] = (0, 0, 360),
                 dropout_ratio: float = 0.0):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.rotation_range = tuple(np.radians(r) for r in rotation_range)
        self.dropout_ratio = dropout_ratio
    
    def jitter(self, points: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*points.shape) * self.jitter_std
        return points + noise
    
    def random_scale(self, points: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return points * scale
    
    def random_rotate(self, points: np.ndarray) -> np.ndarray:
        angles = tuple(np.random.uniform(-r, r) for r in self.rotation_range)
        normalizer = PointCloudNormalizer()
        return normalizer.rotate_euler(points, angles)
    
    def random_dropout(self, points: np.ndarray) -> np.ndarray:
        if self.dropout_ratio <= 0:
            return points
        n_keep = int(len(points) * (1 - self.dropout_ratio))
        keep_indices = np.random.choice(len(points), n_keep, replace=False)
        return points[keep_indices]
    
    def augment(self, points: np.ndarray) -> np.ndarray:
        points = self.random_rotate(points)
        points = self.random_scale(points)
        points = self.jitter(points)
        points = self.random_dropout(points)
        return points


# =============================================================================
# Point Cloud Networks
# =============================================================================

class SharedMLP(nn.Module):
    """Shared Multi-Layer Perceptron for point-wise features."""
    
    def __init__(self, in_channels: int, out_channels: List[int], bn: bool = True):
        super().__init__()
        layers = []
        prev_ch = in_channels
        
        for out_ch in out_channels:
            layers.append(nn.Conv1d(prev_ch, out_ch, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            prev_ch = out_ch
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TNet(nn.Module):
    """Transformation Network for PointNet."""
    
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        
        self.mlp1 = SharedMLP(k, [64, 128, 1024])
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, k * k)
        )
        
        self.fc[-1].weight.data.fill_(0)
        self.fc[-1].bias.data = torch.eye(k).view(-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.mlp1(x)
        x = torch.max(x, 2, keepdim=False)[0]
        x = self.fc(x)
        x = x.view(batch_size, self.k, self.k)
        return x


class PointNet(nn.Module):
    """PointNet for point cloud classification.
    
    "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
    Charles R. Qi, et al., CVPR 2017
    """
    
    def __init__(self, num_classes: int = 40, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.input_transform = TNet(k=in_channels)
        self.mlp1 = SharedMLP(in_channels, [64, 64])
        self.feature_transform = TNet(k=64)
        self.mlp2 = SharedMLP(64, [128, 1024])
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        trans = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        x = self.mlp1(x)
        
        trans_feat = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)
        
        x = self.mlp2(x)
        x = torch.max(x, 2, keepdim=False)[0]
        x = self.fc(x)
        
        return x


class EdgeConv(nn.Module):
    """Edge Convolution layer for DGCNN."""
    
    def __init__(self, k: int, in_channels: int, out_channels: List[int]):
        super().__init__()
        self.k = k
        self.conv = SharedMLP(in_channels * 2, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.knn(x, self.k)
        
        batch_size, num_points, k = idx.size()
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return self.conv(feature)
    
    @staticmethod
    def knn(x: torch.Tensor, k: int) -> torch.Tensor:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx


class DGCNN(nn.Module):
    """Dynamic Graph CNN for point cloud classification.
    
    "Dynamic Graph CNN for Learning on Point Clouds"
    Yue Wang, et al., ACM TOG 2019
    """
    
    def __init__(self, num_classes: int = 40, k: int = 20, dropout: float = 0.5):
        super().__init__()
        self.k = k
        
        self.edge_conv1 = EdgeConv(k, 3, [64, 64])
        self.edge_conv2 = EdgeConv(k, 64, [64, 128])
        self.edge_conv3 = EdgeConv(k, 128, [128, 256])
        
        self.conv5 = SharedMLP(448, [1024])
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)
        
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv5(x)
        
        x1 = torch.max(x, 2, keepdim=False)[0]
        x2 = torch.mean(x, 2, keepdim=False)
        x = torch.cat((x1, x2), dim=1)
        
        x = self.fc(x)
        return x


class PointTransformer(nn.Module):
    """Point Transformer for point cloud classification.
    
    "Point Transformer"
    Hengshuang Zhao, et al., ICCV 2021
    """
    
    def __init__(self, num_classes: int = 40, in_channels: int = 3,
                 dim: List[int] = [32, 64, 128, 256, 512]):
        super().__init__()
        
        self.embedding = SharedMLP(in_channels, [dim[0]])
        
        self.transformer_blocks = nn.ModuleList()
        for i in range(len(dim) - 1):
            self.transformer_blocks.append(
                nn.TransformerEncoderLayer(dim[i], nhead=4, batch_first=True)
            )
        
        self.fc = nn.Sequential(
            nn.Linear(dim[-1], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        x = torch.max(x, dim=1)[0]
        x = self.fc(x)
        return x


class PCT(nn.Module):
    """Point Cloud Transformer.
    
    "PCT: Point Cloud Transformer"
    Meng-Hao Guo, et al., Computational Visual Media 2021
    """
    
    def __init__(self, num_classes: int = 40, in_channels: int = 3,
                 embed_dim: int = 128, depth: int = 4, num_heads: int = 4):
        super().__init__()
        
        self.input_embed = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, embed_dim, 1),
            nn.BatchNorm1d(embed_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_embed(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = torch.max(x, dim=1)[0]
        x = self.fc(x)
        return x


# =============================================================================
# 3D Object Detection
# =============================================================================

class PointPillars(nn.Module):
    """PointPillars for efficient 3D object detection.
    
    "PointPillars: Fast Encoders for Object Detection from Point Clouds"
    Alex H. Lang, et al., CVPR 2019
    """
    
    def __init__(self, num_classes: int = 3,
                 voxel_size: Tuple[float, float] = (0.16, 0.16),
                 max_points_per_pillar: int = 100,
                 max_pillars: int = 12000):
        super().__init__()
        
        self.voxel_size = voxel_size
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars
        
        self.pfn = nn.Sequential(
            nn.Linear(9, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        self.backbone = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.cls_head = nn.Conv2d(256, num_classes * 2, 1)
        self.reg_head = nn.Conv2d(256, num_classes * 7, 1)
    
    def forward(self, pillars: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = pillars.size(0)
        
        x = pillars.view(-1, 9)
        x = self.pfn(x)
        x = x.view(B, self.max_pillars, self.max_points_per_pillar, 64)
        x = torch.max(x, dim=2)[0]
        
        x = x.permute(0, 2, 1).unsqueeze(-1)
        features = self.backbone(x)
        
        cls_preds = self.cls_head(features)
        reg_preds = self.reg_head(features)
        
        return {'cls_preds': cls_preds, 'reg_preds': reg_preds}


class VoxelNet(nn.Module):
    """VoxelNet for 3D object detection.
    
    "VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection"
    Yin Zhou, Oncel Tuzel, CVPR 2018
    """
    
    def __init__(self, num_classes: int = 3, input_channels: int = 4):
        super().__init__()
        
        self.vfe1 = nn.Sequential(
            nn.Linear(input_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.vfe2 = nn.Sequential(
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(128, 64, 3, stride=(2, 1, 1), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.detection_head = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes * 7)
        )
    
    def forward(self, voxel_features: torch.Tensor) -> torch.Tensor:
        x = self.vfe1(voxel_features)
        x = self.vfe2(x)
        x = self.conv3d(x.unsqueeze(0).unsqueeze(-1))
        x = torch.max(x.view(x.size(0), -1), dim=-1)[0]
        return self.detection_head(x)


# =============================================================================
# 3D Semantic Segmentation
# =============================================================================

class RandLANet(nn.Module):
    """RandLA-Net for efficient large-scale semantic segmentation.
    
    "RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds"
    Qingyong Hu, et al., CVPR 2020
    """
    
    def __init__(self, num_classes: int = 13, d_in: int = 3):
        super().__init__()
        
        self.fc0 = nn.Linear(d_in, 8)
        
        dims = [8, 32, 128, 256, 512]
        self.encoder = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.encoder.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(inplace=True)
            ))
        
        self.seg_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc0(x)
        
        for block in self.encoder:
            x = block(x)
            if x.size(1) > 1:
                sample_indices = torch.randperm(x.size(1))[:x.size(1)//4]
                x = x[:, sample_indices, :]
        
        x = self.seg_head(x)
        return x


class KPConv(nn.Module):
    """KPConv for semantic segmentation.
    
    "KPConv: Flexible and Deformable Convolution for Point Clouds"
    Hugues Thomas, et al., ICCV 2019
    """
    
    def __init__(self, num_classes: int = 13, in_channels: int = 3):
        super().__init__()
        
        self.encoder = nn.ModuleList([
            nn.Conv1d(in_channels, 64, 1),
            nn.Conv1d(64, 128, 1),
            nn.Conv1d(128, 256, 1),
        ])
        
        self.seg_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, num_classes, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        for conv in self.encoder:
            x = F.relu(conv(x))
        x = self.seg_head(x)
        return x.permute(0, 2, 1)


class MinkowskiNet(nn.Module):
    """MinkowskiNet using sparse convolutions.
    
    "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks"
    Christopher Choy, et al., CVPR 2019
    
    Note: This is a simplified version using standard convolutions.
    """
    
    def __init__(self, num_classes: int = 13, in_channels: int = 3):
        super().__init__()
        
        self.encoder = nn.ModuleList([
            nn.Conv1d(in_channels, 32, 1),
            nn.Conv1d(32, 64, 1),
            nn.Conv1d(64, 128, 1),
        ])
        
        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, num_classes, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.unsqueeze(0).permute(0, 2, 1)
        for block in self.encoder:
            x = F.relu(block(x))
        x = self.seg_head(x)
        return x.squeeze(0).permute(1, 0)


class Cylinder3D(nn.Module):
    """Cylinder3D for semantic segmentation using cylindrical partition.
    
    "Cylinder3D: An Effective 3D Framework for Driving-scene LiDAR Semantic Segmentation"
    Xinge Zhu, et al., arXiv 2021
    """
    
    def __init__(self, num_classes: int = 19, grid_size: Tuple[int, int, int] = (480, 360, 32)):
        super().__init__()
        
        self.grid_size = grid_size
        
        self.backbone = nn.Sequential(
            nn.Conv3d(4, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        self.seg_head = nn.Conv3d(32, num_classes, 1)
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = self.backbone(grid)
        logits = self.seg_head(x)
        return logits


# =============================================================================
# 3D Reconstruction
# =============================================================================

class PointCloudReconstructor:
    """Surface reconstruction from point clouds."""
    
    def __init__(self, method: str = 'poisson'):
        self.method = method
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required for reconstruction")
    
    def reconstruct(self, points: np.ndarray, 
                    normals: Optional[np.ndarray] = None,
                    **kwargs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        elif self.method == 'poisson':
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(100)
        
        if self.method == 'poisson':
            depth = kwargs.get('depth', 9)
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        elif self.method == 'alpha_shapes':
            alpha = kwargs.get('alpha', 0.03)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        else:
            raise ValueError(f"Unknown reconstruction method: {self.method}")
        
        return mesh


class MeshGenerator:
    """Generate meshes from point clouds."""
    
    def __init__(self):
        if not HAS_OPEN3D:
            warnings.warn("Open3D not available. Some features limited.")
    
    def points_to_mesh(self, points: np.ndarray, method: str = 'delaunay', **kwargs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if method == 'convex_hull':
            mesh, _ = pcd.compute_convex_hull()
        elif method == 'delaunay':
            alpha = kwargs.get('alpha', 0.1)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        elif method == 'poisson':
            reconstructor = PointCloudReconstructor('poisson')
            mesh = reconstructor.reconstruct(points, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return mesh
    
    def marching_cubes(self, sdf: np.ndarray, level: float = 0.0,
                       spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        try:
            from skimage import measure
            vertices, faces, normals, _ = measure.marching_cubes(sdf, level=level, spacing=spacing)
            return vertices, faces, normals
        except ImportError:
            raise ImportError("scikit-image is required for marching cubes")


class ImplicitRepresentation(nn.Module):
    """Neural implicit surface representation.
    
    Represents 3D shapes as continuous signed distance functions (SDF)
    using neural networks.
    """
    
    def __init__(self, hidden_dim: int = 256, num_layers: int = 8):
        super().__init__()
        
        layers = []
        in_dim = 3
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.network(coords)
    
    def query_sdf(self, coords: np.ndarray) -> np.ndarray:
        coords_tensor = torch.from_numpy(coords).float()
        with torch.no_grad():
            sdf = self.forward(coords_tensor)
        return sdf.cpu().numpy().squeeze()
    
    def extract_mesh(self, bounds: Tuple[float, float] = (-1, 1), resolution: int = 256):
        x = np.linspace(bounds[0], bounds[1], resolution)
        y = np.linspace(bounds[0], bounds[1], resolution)
        z = np.linspace(bounds[0], bounds[1], resolution)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        
        sdf = self.query_sdf(coords).reshape(resolution, resolution, resolution)
        
        generator = MeshGenerator()
        vertices, faces, normals = generator.marching_cubes(sdf, level=0.0)
        
        return vertices, faces


# =============================================================================
# 3D Utilities
# =============================================================================

class ICPRegistration:
    """Iterative Closest Point registration."""
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def register(self, source: np.ndarray, target: np.ndarray,
                 initial_transform: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if HAS_OPEN3D:
            return self._register_open3d(source, target, initial_transform)
        else:
            return self._register_numpy(source, target, initial_transform)
    
    def _register_open3d(self, source: np.ndarray, target: np.ndarray, initial_transform):
        source_pcd = o3d.geometry.PointCloud()
        target_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source)
        target_pcd.points = o3d.utility.Vector3dVector(target)
        
        if initial_transform is None:
            initial_transform = np.eye(4)
        
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 0.05, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations,
                relative_fitness=self.tolerance,
                relative_rmse=self.tolerance
            )
        )
        
        source_aligned = np.asarray(source_pcd.transform(result.transformation).points)
        
        return {
            'transformation': result.transformation,
            'source_aligned': source_aligned,
            'rmse': result.inlier_rmse,
            'fitness': result.fitness
        }
    
    def _register_numpy(self, source: np.ndarray, target: np.ndarray, initial_transform):
        T = np.eye(4) if initial_transform is None else initial_transform.copy()
        prev_error = float('inf')
        
        for iteration in range(self.max_iterations):
            source_hom = np.hstack([source, np.ones((len(source), 1))])
            source_transformed = (T @ source_hom.T).T[:, :3]
            
            distances, indices = self._nearest_neighbors(source_transformed, target)
            T_new = self._best_fit_transform(source_transformed, target[indices])
            T = T_new @ T
            
            mean_error = np.mean(distances)
            if abs(prev_error - mean_error) < self.tolerance:
                break
            prev_error = mean_error
        
        source_aligned = (T @ np.hstack([source, np.ones((len(source), 1))]).T).T[:, :3]
        
        return {
            'transformation': T,
            'source_aligned': source_aligned,
            'rmse': mean_error,
            'fitness': 1.0
        }
    
    def _nearest_neighbors(self, source: np.ndarray, target: np.ndarray):
        if HAS_SKLEARN:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
            distances, indices = nbrs.kneighbors(source)
            return distances.squeeze(), indices.squeeze()
        else:
            distances = spatial_distance.cdist(source, target)
            indices = np.argmin(distances, axis=1)
            min_distances = distances[np.arange(len(source)), indices]
            return min_distances, indices
    
    def _best_fit_transform(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        
        AA = A - centroid_A
        BB = B - centroid_B
        
        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = centroid_B - R @ centroid_A
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T


class PointCloudVisualizer:
    """3D visualization for point clouds."""
    
    def __init__(self, window_name: str = "Point Cloud"):
        self.window_name = window_name
        if not HAS_OPEN3D:
            warnings.warn("Open3D not available. Visualization limited.")
    
    def visualize(self, points: np.ndarray, 
                  colors: Optional[np.ndarray] = None,
                  labels: Optional[np.ndarray] = None,
                  point_size: float = 2.0):
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required for visualization")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif labels is not None:
            label_colors = self._label_to_color(labels)
            pcd.colors = o3d.utility.Vector3dVector(label_colors)
        
        o3d.visualization.draw_geometries([pcd], window_name=self.window_name)
    
    @staticmethod
    def _label_to_color(labels: np.ndarray) -> np.ndarray:
        color_map = np.array([
            [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.5, 0.0],
            [0.5, 0.0, 1.0], [0.0, 0.5, 1.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.5],
        ])
        return color_map[labels % len(color_map)]


def compute_normals(points: np.ndarray, k: int = 30, normalize: bool = True) -> np.ndarray:
    """Estimate surface normals from point cloud."""
    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        normals = np.asarray(pcd.normals)
    else:
        normals = _compute_normals_numpy(points, k)
    
    if normalize:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-10)
    
    return normals


def _compute_normals_numpy(points: np.ndarray, k: int) -> np.ndarray:
    n_points = len(points)
    normals = np.zeros_like(points)
    
    if HAS_SKLEARN:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
        _, indices = nbrs.kneighbors(points)
    else:
        indices = np.tile(np.arange(n_points), (n_points, 1))
    
    for i in range(n_points):
        neighbors = points[indices[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normals[i] = eigenvectors[:, 0]
    
    return normals


def remove_outliers(points: np.ndarray, 
                    method: str = 'statistical',
                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Remove statistical outliers from point cloud."""
    if method == 'statistical':
        return _remove_statistical_outliers(points, **kwargs)
    elif method == 'radius':
        return _remove_radius_outliers(points, **kwargs)
    else:
        raise ValueError(f"Unknown outlier removal method: {method}")


def _remove_statistical_outliers(points: np.ndarray,
                                 nb_neighbors: int = 30,
                                 std_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    n_points = len(points)
    
    if HAS_SKLEARN:
        nbrs = NearestNeighbors(n_neighbors=nb_neighbors + 1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        mean_distances = distances[:, 1:].mean(axis=1)
    else:
        mean_distances = np.zeros(n_points)
        for i in range(n_points):
            dists = np.linalg.norm(points - points[i], axis=1)
            dists = np.partition(dists, nb_neighbors)[:nb_neighbors + 1]
            mean_distances[i] = dists[1:].mean()
    
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_distances < threshold
    
    return points[inlier_mask], np.where(inlier_mask)[0]


def _remove_radius_outliers(points: np.ndarray,
                            nb_points: int = 16,
                            radius: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    n_points = len(points)
    
    if HAS_SKLEARN:
        nbrs = NearestNeighbors(radius=radius).fit(points)
        counts = np.array([len(ind) for ind in nbrs.radius_neighbors(points, return_distance=False)])
    else:
        counts = np.zeros(n_points, dtype=int)
        for i in range(n_points):
            dists = np.linalg.norm(points - points[i], axis=1)
            counts[i] = np.sum(dists <= radius) - 1
    
    inlier_mask = counts >= nb_points
    return points[inlier_mask], np.where(inlier_mask)[0]


# =============================================================================
# Datasets
# =============================================================================

class PointCloudDataset(Dataset):
    """Generic point cloud dataset."""
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 split: str = 'train',
                 num_points: int = 1024,
                 load_labels: bool = True,
                 transform: Optional[Callable] = None,
                 cache: bool = False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.load_labels = load_labels
        self.transform = transform
        self.cache = cache
        
        self.file_list = self._get_file_list()
        self.loader = PointCloudLoader()
        
        if cache:
            self._cache = {}
        else:
            self._cache = None
    
    def _get_file_list(self) -> List[Path]:
        split_file = self.data_dir / f'{self.split}.txt'
        
        if split_file.exists():
            with open(split_file) as f:
                return [self.data_dir / line.strip() for line in f]
        else:
            return list(self.data_dir.glob('*.ply')) + list(self.data_dir.glob('*.pcd'))
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._cache is not None and idx in self._cache:
            return self._cache[idx].copy()
        
        filepath = self.file_list[idx]
        points, attributes = self.loader.load(filepath)
        
        sampler = PointSampler()
        if len(points) > self.num_points:
            points = sampler.fps(points, self.num_points)
        else:
            points = sampler.random_sample(points, self.num_points)
        
        normalizer = PointCloudNormalizer()
        points = normalizer.center_scale(points)
        
        sample = {'points': points, 'filename': filepath.name}
        
        if self.transform:
            sample['points'] = self.transform(sample['points'])
        
        if self._cache is not None:
            self._cache[idx] = sample.copy()
        
        return sample


class ModelNet40Loader(Dataset):
    """Loader for ModelNet40 dataset."""
    
    NUM_CLASSES = 40
    CLASS_NAMES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl',
        'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser',
        'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
        'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
        'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table',
        'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    
    def __init__(self, data_dir: Union[str, Path], split: str = 'train',
                 num_points: int = 1024):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        
        self.file_list, self.labels = self._load_file_list()
        self.sampler = PointSampler()
    
    def _load_file_list(self) -> Tuple[List[Path], List[int]]:
        file_list = []
        labels = []
        
        split_file = self.data_dir / f'{self.split}.txt'
        
        if split_file.exists():
            with open(split_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_name = line.split('_')[0]
                        label = self.CLASS_NAMES.index(class_name)
                        filepath = self.data_dir / class_name / f'{line}.txt'
                        file_list.append(filepath)
                        labels.append(label)
        
        return file_list, labels
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filepath = self.file_list[idx]
        points = np.loadtxt(filepath, delimiter=',')
        
        if len(points) > self.num_points:
            points = self.sampler.fps(points, self.num_points)
        else:
            points = self.sampler.random_sample(points, self.num_points)
        
        normalizer = PointCloudNormalizer()
        points = normalizer.center_scale(points)
        
        return {
            'points': points,
            'label': self.labels[idx],
            'filename': filepath.name
        }


class ShapeNetLoader(Dataset):
    """Loader for ShapeNet dataset with semantic part annotations."""
    
    NUM_CLASSES = 16
    NUM_PARTS = 50
    
    def __init__(self, data_dir: Union[str, Path], 
                 split: str = 'train',
                 num_points: int = 2048,
                 category: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.category = category
        
        self.file_list = self._load_file_list()
        self.sampler = PointSampler()
    
    def _load_file_list(self) -> List[Path]:
        file_list = []
        
        if self.category:
            category_dir = self.data_dir / self.category / self.split
            if category_dir.exists():
                file_list.extend(category_dir.glob('*.npy'))
        else:
            split_dir = self.data_dir / self.split
            if split_dir.exists():
                file_list.extend(split_dir.glob('**/*.npy'))
        
        return file_list
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filepath = self.file_list[idx]
        data = np.load(filepath)
        
        if data.ndim == 2:
            points = data[:, :3]
            labels = data[:, -1].astype(int) if data.shape[1] > 3 else None
        else:
            points = data
            labels = None
        
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
            if labels is not None:
                labels = labels[indices]
        
        normalizer = PointCloudNormalizer()
        points = normalizer.center_scale(points)
        
        sample = {'points': points, 'filename': filepath.name}
        if labels is not None:
            sample['labels'] = labels
        
        return sample


class KITTILoader(Dataset):
    """Loader for KITTI dataset with 3D object annotations."""
    
    NUM_CLASSES = 3
    CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
    
    def __init__(self, data_dir: Union[str, Path],
                 split: str = 'training',
                 num_points: int = 16384):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        
        velodyne_dir = self.data_dir / self.split / 'velodyne'
        self.file_list = sorted(velodyne_dir.glob('*.bin')) if velodyne_dir.exists() else []
        self.sampler = PointSampler()
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filepath = self.file_list[idx]
        
        # KITTI .bin format: (x, y, z, intensity)
        scan = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        points = scan[:, :3]
        intensity = scan[:, 3]
        
        # Sample to fixed number
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
            intensity = intensity[indices]
        
        return {
            'points': points,
            'intensity': intensity,
            'filename': filepath.stem
        }


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Processing
    'PointCloudLoader', 'PointCloudNormalizer', 'Voxelizer',
    'PointSampler', 'PointAugmentor',
    # Networks
    'PointNet', 'DGCNN', 'PointTransformer', 'PCT',
    # Detection
    'PointPillars', 'VoxelNet',
    # Segmentation
    'RandLANet', 'KPConv', 'MinkowskiNet', 'Cylinder3D',
    # Reconstruction
    'PointCloudReconstructor', 'MeshGenerator', 'ImplicitRepresentation',
    # Utilities
    'ICPRegistration', 'PointCloudVisualizer',
    'compute_normals', 'remove_outliers',
    # Datasets
    'PointCloudDataset', 'ModelNet40Loader', 'ShapeNetLoader', 'KITTILoader',
]
