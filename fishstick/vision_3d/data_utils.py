"""
Data Utilities for 3D Vision

Point cloud loading, datasets, and preprocessing.
"""

from typing import Tuple, List, Optional, Callable
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
import os


class PointCloudDataset(Dataset):
    """
    Dataset for point cloud data.
    """

    def __init__(
        self,
        point_clouds: List[Tensor],
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.point_clouds = point_clouds
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.point_clouds)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Returns:
            point_cloud: [N, 3]
            label: int
        """
        point_cloud = self.point_clouds[idx]

        if self.transform is not None:
            point_cloud = self.transform(point_cloud)

        label = self.labels[idx] if self.labels is not None else 0

        return point_cloud, label


def collate_point_cloud(batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor]:
    """
    Collate function for point cloud batches.

    Args:
        batch: List of (point_cloud, label) tuples

    Returns:
        point_clouds: Padded point clouds [B, max_N, 3]
        labels: Labels [B]
    """
    point_clouds, labels = zip(*batch)

    max_points = max(pc.shape[0] for pc in point_clouds)

    padded = []
    for pc in point_clouds:
        if pc.shape[0] < max_points:
            pad = torch.zeros(max_points - pc.shape[0], 3)
            pc = torch.cat([pc, pad], dim=0)
        padded.append(pc)

    return torch.stack(padded), torch.tensor(labels)


def read_point_cloud(path: str) -> Tensor:
    """
    Read point cloud from file.

    Supports .ply, .pcd, .xyz formats.

    Args:
        path: Path to file

    Returns:
        Point cloud [N, 3]
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".ply":
        return read_ply(path)
    elif ext == ".pcd":
        return read_pcd(path)
    elif ext == ".xyz":
        return np.loadtxt(path)[:, :3]
    else:
        raise ValueError(f"Unsupported format: {ext}")


def read_ply(path: str) -> Tensor:
    """
    Read PLY file.

    Args:
        path: Path to .ply file

    Returns:
        Point cloud [N, 3]
    """
    try:
        from plyfile import PlyData

        plydata = PlyData.read(path)
        vertex = plydata["vertex"]
        points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
        return torch.from_numpy(points).float()
    except ImportError:
        return torch.zeros(0, 3)


def read_pcd(path: str) -> Tensor:
    """
    Read PCD file.

    Args:
        path: Path to .pcd file

    Returns:
        Point cloud [N, 3]
    """
    points = []

    with open(path, "r") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line == "DATA ascii":
                in_data = True
                continue
            if in_data and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        points.append(
                            [float(parts[0]), float(parts[1]), float(parts[2])]
                        )
                    except ValueError:
                        continue

    return torch.tensor(points, dtype=torch.float32) if points else torch.zeros(0, 3)


def save_point_cloud(
    path: str, points: Tensor, colors: Optional[Tensor] = None
) -> None:
    """
    Save point cloud to file.

    Args:
        path: Output path (.ply, .xyz)
        points: Point cloud [N, 3]
        colors: Colors [N, 3] (optional)
    """
    points_np = points.cpu().numpy()

    ext = os.path.splitext(path)[1].lower()

    if ext == ".ply":
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_np)}\n")
            if colors is not None:
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            else:
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
            f.write("end_header\n")

            for i in range(len(points_np)):
                if colors is not None:
                    colors_np = colors[i].cpu().numpy()
                    f.write(
                        f"{points_np[i, 0]} {points_np[i, 1]} {points_np[i, 2]} "
                        f"{int(colors_np[0] * 255)} {int(colors_np[1] * 255)} {int(colors_np[2] * 255)}\n"
                    )
                else:
                    f.write(f"{points_np[i, 0]} {points_np[i, 1]} {points_np[i, 2]}\n")

    elif ext == ".xyz":
        np.savetxt(path, points_np, fmt="%.6f")

    else:
        raise ValueError(f"Unsupported format: {ext}")


def voxel_downsample(
    points: Tensor,
    voxel_size: float,
) -> Tensor:
    """
    Downsample point cloud via voxelization.

    Args:
        points: Point cloud [N, 3]
        voxel_size: Size of voxel

    Returns:
        Downsampled point cloud [M, 3]
    """
    indices = (points / voxel_size).long()
    unique, inverse = torch.unique(indices, dim=0, return_inverse=True)

    downsampled = torch.zeros(unique.shape[0], 3, device=points.device)

    for i in range(unique.shape[0]):
        mask = inverse == i
        downsampled[i] = points[mask].mean(dim=0)

    return downsampled
