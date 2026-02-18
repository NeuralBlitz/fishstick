"""
3D Visualization Utilities

Helpers for visualizing point clouds, meshes, and depth maps.
"""

from typing import Tuple, Optional, List
import torch
from torch import Tensor
import numpy as np


def visualize_point_cloud(
    points: Tensor,
    colors: Optional[Tensor] = None,
    output_path: Optional[str] = None,
) -> dict:
    """
    Visualize point cloud (returns data for visualization).

    Args:
        points: Point cloud [N, 3]
        colors: Colors [N, 3] (optional)
        output_path: Path to save (optional)

    Returns:
        Visualization data dict
    """
    points_np = points.cpu().numpy()

    data = {
        "points": points_np,
        "colors": colors.cpu().numpy() if colors is not None else None,
    }

    return data


def visualize_bounding_boxes_3d(
    boxes: Tensor,
    labels: Optional[Tensor] = None,
    scores: Optional[Tensor] = None,
) -> dict:
    """
    Visualize 3D bounding boxes.

    Args:
        boxes: Bounding boxes [N, 7]
        labels: Class labels [N]
        scores: Confidence scores [N]

    Returns:
        Visualization data dict
    """
    boxes_np = boxes.cpu().numpy()

    data = {
        "boxes": boxes_np,
        "labels": labels.cpu().numpy() if labels is not None else None,
        "scores": scores.cpu().numpy() if scores is not None else None,
    }

    return data


def visualize_depth(
    depth: Tensor,
    colormap: str = "magma",
    normalize: bool = True,
) -> Tensor:
    """
    Visualize depth map as RGB.

    Args:
        depth: Depth map [H, W] or [1, H, W]
        colormap: Colormap name
        normalize: Whether to normalize depth

    Returns:
        RGB visualization [3, H, W] or [1, 3, H, W]
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)

    depth_np = depth.cpu().numpy()

    if normalize:
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        if depth_max > depth_min:
            depth_np = (depth_np - depth_min) / (depth_max - depth_min)

    if colormap == "magma":
        vis = np.stack(
            [
                depth_np**0.5,
                depth_np**0.75,
                depth_np**0.25,
            ],
            axis=0,
        )
    elif colormap == "viridis":
        vis = np.stack(
            [
                depth_np,
                depth_np**2,
                depth_np**0.5,
            ],
            axis=0,
        )
    else:
        vis = np.stack([depth_np] * 3, axis=0)

    return torch.from_numpy(vis).float()


def create_3d_scatter(
    points: Tensor,
    colors: Optional[Tensor] = None,
    size: float = 1.0,
) -> dict:
    """
    Create 3D scatter plot data.

    Args:
        points: [N, 3]
        colors: [N, 3]
        size: Point size

    Returns:
        Scatter data dict
    """
    return {
        "type": "scatter3d",
        "points": points.cpu().numpy(),
        "colors": colors.cpu().numpy() if colors is not None else None,
        "size": size,
    }


def create_mesh_actor(
    vertices: Tensor,
    faces: Tensor,
    colors: Optional[Tensor] = None,
) -> dict:
    """
    Create mesh actor data for visualization.

    Args:
        vertices: [N, 3]
        faces: [M, 3]
        colors: [N, 3] or [M, 3]

    Returns:
        Mesh data dict
    """
    return {
        "type": "mesh",
        "vertices": vertices.cpu().numpy(),
        "faces": faces.cpu().numpy() if faces is not None else None,
        "colors": colors.cpu().numpy() if colors is not None else None,
    }
