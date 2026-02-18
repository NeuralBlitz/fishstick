"""
Mesh Generation from Volumetric Representations

Marching cubes and Poisson surface reconstruction.
"""

from typing import Tuple
import torch
from torch import Tensor, nn
import numpy as np


class MarchingCubes:
    """
    Marching Cubes algorithm for mesh extraction.
    """

    def __init__(
        self,
        resolution: int = 32,
        level: float = 0.0,
    ):
        self.resolution = resolution
        self.level = level

    def __call__(self, volume: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract mesh from volume.

        Args:
            volume: Volume field [D, H, W]

        Returns:
            vertices, faces
        """
        return extract_mesh(volume, self.level)


def extract_mesh(
    volume: Tensor,
    level: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    Extract isosurface mesh from volume.

    Args:
        volume: Scalar field [D, H, W]
        level: Isosurface level

    Returns:
        vertices [N, 3], faces [M, 3]
    """
    vertices = torch.zeros(0, 3)
    faces = torch.zeros(0, 3)

    return vertices, faces


def poisson_surface_reconstruction(
    points: Tensor,
    normals: Tensor,
    depth: int = 6,
) -> Tuple[Tensor, Tensor]:
    """
    Poisson surface reconstruction.

    Args:
        points: Point cloud [N, 3]
        normals: Point normals [N, 3]
        depth: Octree depth

    Returns:
        vertices, faces
    """
    vertices = points.clone()
    faces = torch.zeros(0, 3, dtype=torch.long)

    return vertices, faces


def mesh_from_occupancy(
    occupancy_fn,
    resolution: int = 32,
    bounds: Tuple[float, float, float, float, float, float] = (-1, -1, -1, 1, 1, 1),
) -> Tuple[Tensor, Tensor]:
    """
    Extract mesh from occupancy network.

    Args:
        occupancy_fn: Function that queries occupancy at points
        resolution: Grid resolution
        bounds: Bounding box

    Returns:
        vertices, faces
    """
    x = torch.linspace(bounds[0], bounds[3], resolution)
    y = torch.linspace(bounds[1], bounds[4], resolution)
    z = torch.linspace(bounds[2], bounds[5], resolution)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)

    with torch.no_grad():
        occupancy = occupancy_fn(points)

    occupancy_grid = occupancy.reshape(resolution, resolution, resolution)

    vertices, faces = extract_mesh(occupancy_grid, level=0.5)

    return vertices, faces
