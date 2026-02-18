"""
3D Transformations Module

Geometric transformations for point clouds and 3D data.
"""

from typing import Tuple, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
import numpy as np


def euler_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> Tensor:
    """
    Convert Euler angles to rotation matrix.

    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)

    Returns:
        Rotation matrix [3, 3]
    """
    R_x = torch.tensor(
        [
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)],
        ],
        dtype=torch.float32,
    )

    R_y = torch.tensor(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ],
        dtype=torch.float32,
    )

    R_z = torch.tensor(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    return R_z @ R_y @ R_x


def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    """
    Convert quaternion to rotation matrix.

    Args:
        quaternion: [x, y, z, w] or [w, x, y, z]

    Returns:
        Rotation matrix [3, 3]
    """
    if quaternion.shape[0] == 4:
        if quaternion[-1].abs() > 0.9:
            q = quaternion
        else:
            q = torch.cat([quaternion[1:], quaternion[:1]])
    else:
        q = quaternion

    x, y, z, w = q[0].item(), q[1].item(), q[2].item(), q[3].item()

    R = torch.tensor(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=torch.float32,
    )

    return R


def rotation_matrix_to_quaternion(R: Tensor) -> Tensor:
    """
    Convert rotation matrix to quaternion [w, x, y, z].

    Args:
        R: Rotation matrix [3, 3]

    Returns:
        Quaternion [4]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return torch.tensor([w, x, y, z], dtype=torch.float32)


def transform_matrix(
    translation: Optional[Tensor] = None,
    rotation: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Create 4x4 transformation matrix.

    Args:
        translation: [3]
        rotation: [3, 3] or quaternion
        scale: Scale factor

    Returns:
        Transformation matrix [4, 4]
    """
    T = torch.eye(4)

    if rotation is not None:
        if rotation.shape == (3, 3):
            T[:3, :3] = rotation
        else:
            T[:3, :3] = quaternion_to_rotation_matrix(rotation)

    if translation is not None:
        T[:3, 3] = translation

    if scale is not None:
        T[:3, :3] *= scale

    return T


def rotate_3d(points: Tensor, rotation: Tensor) -> Tensor:
    """
    Rotate points.

    Args:
        points: [N, 3]
        rotation: [3, 3] or quaternion [4]

    Returns:
        Rotated points [N, 3]
    """
    if rotation.shape == (4,):
        rotation = quaternion_to_rotation_matrix(rotation)

    return points @ rotation.T


def translate_3d(points: Tensor, translation: Tensor) -> Tensor:
    """
    Translate points.

    Args:
        points: [N, 3]
        translation: [3]

    Returns:
        Translated points [N, 3]
    """
    return points + translation


def scale_3d(points: Tensor, scale: float, center: Optional[Tensor] = None) -> Tensor:
    """
    Scale points.

    Args:
        points: [N, 3]
        scale: Scale factor
        center: Center of scaling

    Returns:
        Scaled points [N, 3]
    """
    if center is not None:
        points = points - center
    points = points * scale
    if center is not None:
        points = points + center
    return points


def random_rotation_3d(
    points: Tensor,
    angle_std: float = 0.1,
) -> Tensor:
    """
    Apply random rotation.

    Args:
        points: [N, 3]
        angle_std: Standard deviation of random angles

    Returns:
        Rotated points [N, 3]
    """
    roll = torch.randn(1).item() * angle_std
    pitch = torch.randn(1).item() * angle_std
    yaw = torch.randn(1).item() * angle_std

    R = euler_to_rotation_matrix(roll, pitch, yaw).to(points.device)

    return points @ R.T


def random_translate_3d(
    points: Tensor,
    translation_std: float = 0.1,
) -> Tensor:
    """
    Apply random translation.

    Args:
        points: [N, 3]
        translation_std: Standard deviation

    Returns:
        Translated points [N, 3]
    """
    translation = torch.randn(3).to(points.device) * translation_std
    return points + translation


def transform_point_cloud(
    points: Tensor,
    translation: Optional[Tensor] = None,
    rotation: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Apply full transformation to point cloud.

    Args:
        points: [N, 3]
        translation: [3]
        rotation: [3, 3]
        scale: float

    Returns:
        Transformed points [N, 3]
    """
    transformed = points.clone()

    if rotation is not None:
        transformed = rotate_3d(transformed, rotation)

    if scale is not None:
        transformed = scale_3d(transformed, scale)

    if translation is not None:
        transformed = translate_3d(transformed, translation)

    return transformed


def look_at(
    eye: Tensor,
    target: Tensor,
    up: Optional[Tensor] = None,
) -> Tensor:
    """
    Create view matrix (camera transformation).

    Args:
        eye: Camera position [3]
        target: Look-at point [3]
        up: Up vector [3]

    Returns:
        View matrix [4, 4]
    """
    if up is None:
        up = torch.tensor([0, 0, 1], dtype=torch.float32)

    z = eye - target
    z = z / torch.norm(z)

    x = torch.cross(up, z)
    x = x / torch.norm(x)

    y = torch.cross(z, x)

    view = torch.eye(4)
    view[:3, 0] = x
    view[:3, 1] = y
    view[:3, 2] = z
    view[:3, 3] = eye

    return view
