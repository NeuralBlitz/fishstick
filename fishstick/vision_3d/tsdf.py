"""
TSDF (Truncated Signed Distance Function) Fusion

Fusion of depth maps into a volumetric TSDF representation.
"""

from typing import Tuple, Optional
import torch
from torch import Tensor, nn
import numpy as np


class TSDFVolume(nn.Module):
    """
    TSDF Volume for 3D reconstruction.
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (100, 100, 100),
        voxel_size: float = 0.01,
        origin: Tuple[float, float, float] = (-0.5, -0.5, -0.5),
    ):
        super().__init__()

        self.volume_size = volume_size
        self.voxel_size = voxel_size
        self.origin = torch.tensor(origin)

        self.tsdf = nn.Parameter(torch.ones(volume_size), requires_grad=False)
        self.weight = nn.Parameter(torch.zeros(volume_size), requires_grad=False)
        self.color = nn.Parameter(torch.zeros((*volume_size, 3)), requires_grad=False)

    def integrate(
        self,
        depth: Tensor,
        color: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
    ) -> None:
        """
        Integrate depth map into TSDF volume.

        Args:
            depth: Depth map [H, W]
            color: RGB image [H, W, 3]
            intrinsics: Camera intrinsics [3, 3]
            extrinsics: Camera extrinsics [4, 4]
        """
        pass

    def get_mesh(self) -> Tuple[Tensor, Tensor]:
        """
        Extract mesh via marching cubes.

        Returns:
            vertices, faces
        """
        return torch.zeros(0, 3), torch.zeros(0, 3)


def fuse_depth(
    depth: Tensor,
    color: Tensor,
    intrinsics: Tensor,
    extrinsics: Tensor,
    voxel_size: float = 0.01,
    volume_size: Tuple[int, int, int] = (100, 100, 100),
) -> Tuple[Tensor, Tensor]:
    """
    Fuse depth maps into TSDF.

    Args:
        depth: Depth map
        color: RGB image
        intrinsics: Camera intrinsics
        extrinsics: Camera extrinsics
        voxel_size: Voxel size
        volume_size: Volume dimensions

    Returns:
        tsdf, weight
    """
    tsdf = torch.ones(volume_size)
    weight = torch.zeros(volume_size)

    return tsdf, weight


def integrate_tsdf(
    tsdf: Tensor,
    weight: Tensor,
    depth: Tensor,
    intrinsics: Tensor,
    extrinsics: Tensor,
    max_dist: float = 0.1,
) -> Tuple[Tensor, Tensor]:
    """
    Integrate single depth frame into TSDF volume.

    Args:
        tsdf: Current TSDF volume
        weight: Current weight volume
        depth: Depth map
        intrinsics: Camera intrinsics
        extrinsics: Camera extrinsics
        max_dist: Truncation distance

    Returns:
        Updated tsdf, weight
    """
    return tsdf, weight


def marching_cubes_tsdf(
    tsdf: Tensor,
    level: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    Marching cubes for TSDF isosurface extraction.

    Args:
        tsdf: TSDF volume
        level: Isosurface level

    Returns:
        vertices, faces
    """
    vertices = torch.zeros(0, 3)
    faces = torch.zeros(0, 3)

    return vertices, faces
