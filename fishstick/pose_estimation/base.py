"""
Base Types and Utilities for Pose Estimation

Core data structures and utility functions for pose estimation tasks.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum

import torch
from torch import Tensor
import numpy as np


class KeypointName(Enum):
    """Enumeration of standard keypoint names."""

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


@dataclass
class Keypoint:
    """
    Represents a single keypoint in 2D or 3D space.

    Attributes:
        x: X coordinate (or u for 2D image coordinates)
        y: Y coordinate (or v for 2D image coordinates)
        z: Z coordinate (optional, for 3D poses)
        visibility: Visibility flag (0=not labeled, 1=occluded, 2=visible)
        confidence: Confidence score for the keypoint detection
    """

    x: float
    y: float
    z: Optional[float] = None
    visibility: int = 2
    confidence: float = 1.0

    def to_tensor(self, include_z: bool = True) -> Tensor:
        """Convert keypoint to tensor representation."""
        if include_z and self.z is not None:
            return torch.tensor(
                [self.x, self.y, self.z, self.visibility, self.confidence]
            )
        return torch.tensor([self.x, self.y, self.visibility, self.confidence])

    @classmethod
    def from_tensor(cls, tensor: Tensor, has_z: bool = True) -> "Keypoint":
        """Create keypoint from tensor representation."""
        if has_z and len(tensor) >= 3:
            visibility = int(tensor[3].item()) if len(tensor) > 3 else 2
            confidence = float(tensor[4].item()) if len(tensor) > 4 else 1.0
            return cls(
                x=float(tensor[0]),
                y=float(tensor[1]),
                z=float(tensor[2]),
                visibility=visibility,
                confidence=confidence,
            )
        visibility = int(tensor[2].item()) if len(tensor) > 2 else 2
        confidence = float(tensor[3].item()) if len(tensor) > 3 else 1.0
        return cls(
            x=float(tensor[0]),
            y=float(tensor[1]),
            visibility=visibility,
            confidence=confidence,
        )

    def distance_to(self, other: "Keypoint", use_3d: bool = False) -> float:
        """Compute Euclidean distance to another keypoint."""
        dx = self.x - other.x
        dy = self.y - other.y
        if use_3d and self.z is not None and other.z is not None:
            dz = self.z - other.z
            return np.sqrt(dx**2 + dy**2 + dz**2)
        return np.sqrt(dx**2 + dy**2)


@dataclass
class Pose2D:
    """
    Represents a 2D pose with multiple keypoints.

    Attributes:
        keypoints: List of Keypoint objects
        skeleton: Optional skeleton connectivity
        image_id: Optional image identifier
        person_id: Optional person identifier for multi-person tracking
    """

    keypoints: List[Keypoint] = field(default_factory=list)
    skeleton: Optional[List[Tuple[int, int]]] = None
    image_id: Optional[str] = None
    person_id: Optional[int] = None

    def __len__(self) -> int:
        return len(self.keypoints)

    def to_tensor(self) -> Tensor:
        """Convert pose to tensor of shape (N, 3) or (N, 4) with confidence."""
        if not self.keypoints:
            return torch.zeros(0, 4)

        has_confidence = any(kp.confidence != 1.0 for kp in self.keypoints)
        tensors = [kp.to_tensor(include_z=False) for kp in self.keypoints]

        if has_confidence:
            return torch.stack(tensors)
        else:
            result = torch.stack(tensors)
            return result

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "Pose2D":
        """Create pose from tensor of shape (N, 3) or (N, 4)."""
        keypoints = [Keypoint.from_tensor(t, has_z=False) for t in tensor]
        return cls(keypoints=keypoints)

    def get_visible_keypoints(self) -> List[Tuple[int, Keypoint]]:
        """Get only visible keypoints with their indices."""
        return [(i, kp) for i, kp in enumerate(self.keypoints) if kp.visibility > 0]

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Get bounding box (x1, y1, x2, y2) of the pose."""
        if not self.keypoints:
            return (0, 0, 0, 0)

        visible = self.get_visible_keypoints()
        if not visible:
            return (0, 0, 0, 0)

        xs = [kp.x for _, kp in visible]
        ys = [kp.y for _, kp in visible]
        return (min(xs), min(ys), max(xs), max(ys))

    def center(self) -> Tuple[float, float]:
        """Get center of the pose bounding box."""
        x1, y1, x2, y2 = self.bounding_box()
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def scale(self) -> float:
        """Get scale of the pose (height of bounding box)."""
        _, _, _, y2 = self.bounding_box()
        return y2


@dataclass
class Pose3D:
    """
    Represents a 3D pose with multiple keypoints in 3D space.

    Attributes:
        keypoints: List of Keypoint objects (with z coordinates)
        skeleton: Optional skeleton connectivity
        camera_params: Optional camera parameters for projection
    """

    keypoints: List[Keypoint] = field(default_factory=list)
    skeleton: Optional[List[Tuple[int, int]]] = None
    camera_params: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.keypoints)

    def to_tensor(self) -> Tensor:
        """Convert pose to tensor of shape (N, 3)."""
        if not self.keypoints:
            return torch.zeros(0, 3)

        tensors = [kp.to_tensor(include_z=True)[:, :3] for kp in self.keypoints]
        return torch.cat(tensors, dim=-1).reshape(-1, 3)

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "Pose3D":
        """Create pose from tensor of shape (N, 3)."""
        keypoints = []
        for i in range(tensor.shape[0]):
            t = tensor[i]
            keypoints.append(Keypoint(x=float(t[0]), y=float(t[1]), z=float(t[2])))
        return cls(keypoints=keypoints)

    def to_2d(self, intrinsics: Optional[Tensor] = None) -> Pose2D:
        """Project 3D pose to 2D using optional camera intrinsics."""
        keypoints_2d = []
        for kp in self.keypoints:
            if intrinsics is not None and kp.z is not None and kp.z > 0:
                fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
                cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()
                x_2d = fx * kp.x / kp.z + cx
                y_2d = fy * kp.y / kp.z + cy
                keypoints_2d.append(
                    Keypoint(
                        x=x_2d,
                        y=y_2d,
                        visibility=kp.visibility,
                        confidence=kp.confidence,
                    )
                )
            else:
                keypoints_2d.append(
                    Keypoint(
                        x=kp.x,
                        y=kp.y,
                        visibility=kp.visibility,
                        confidence=kp.confidence,
                    )
                )
        return Pose2D(keypoints=keypoints_2d, skeleton=self.skeleton)

    def center_of_mass(self) -> Tuple[float, float, float]:
        """Compute center of mass of the pose."""
        if not self.keypoints:
            return (0, 0, 0)

        xs = [kp.x for kp in self.keypoints if kp.z is not None]
        ys = [kp.y for kp in self.keypoints if kp.z is not None]
        zs = [kp.z for kp in self.keypoints if kp.z is not None]

        if not xs:
            return (0, 0, 0)

        return (np.mean(xs), np.mean(ys), np.mean(zs))


@dataclass
class KeypointSet:
    """
    Container for a set of keypoints with metadata.

    Attributes:
        keypoints: Dictionary mapping keypoint names to Keypoint objects
        metadata: Additional metadata about the keypoint set
    """

    keypoints: Dict[str, Keypoint] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tensor(self, keypoint_order: List[str]) -> Tensor:
        """Convert to tensor in specified order."""
        tensors = []
        for name in keypoint_order:
            if name in self.keypoints:
                tensors.append(self.keypoints[name].to_tensor(include_z=True))
            else:
                tensors.append(torch.zeros(5))
        return torch.stack(tensors)

    def from_tensor(self, tensor: Tensor, keypoint_order: List[str]) -> "KeypointSet":
        """Create from tensor in specified order."""
        keypoints = {}
        for i, name in enumerate(keypoint_order):
            if i < tensor.shape[0]:
                keypoints[name] = Keypoint.from_tensor(tensor[i], has_z=True)
        return KeypointSet(keypoints=keypoints)


@dataclass
class PoseSkeleton:
    """
    Defines a pose skeleton structure with joint names and connectivity.

    Attributes:
        joint_names: List of joint/keypoint names
        edges: List of (parent, child) tuples defining bone connections
        flip_pairs: List of (left_idx, right_idx) pairs for horizontal flipping
       /stereo correspondence
    """

    joint_names: List[str]
    edges: List[Tuple[int, int]]
    flip_pairs: List[Tuple[int, int]] = field(default_factory=list)

    def num_joints(self) -> int:
        """Get number of joints."""
        return len(self.joint_names)


coco_keypoints = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

coco_skeleton = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

coco_flip_pairs = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]

mpii_keypoints = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "pelvis",
    "spine",
    "neck",
    "head_top",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
]

hand_keypoints = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]

animal_keypoints = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "withers",
    "tail",
    "left_front_paw",
    "right_front_paw",
    "left_back_paw",
    "right_back_paw",
]


def keypoint_to_tensor(keypoints: List[Keypoint]) -> Tensor:
    """Convert list of keypoints to tensor."""
    if not keypoints:
        return torch.zeros(0, 4)
    return torch.stack([kp.to_tensor(include_z=False) for kp in keypoints])


def tensor_to_keypoints(tensor: Tensor) -> List[Keypoint]:
    """Convert tensor to list of keypoints."""
    return [Keypoint.from_tensor(t, has_z=False) for t in tensor]


def normalize_keypoints(
    keypoints: Tensor, image_size: Tuple[int, int], format: str = "xy"
) -> Tensor:
    """
    Normalize keypoints to [0, 1] range.

    Args:
        keypoints: Tensor of shape (N, 2) or (N, 3)
        image_size: (height, width) of the image
        format: Coordinate format ('xy' or 'uv')

    Returns:
        Normalized keypoints in [0, 1]
    """
    h, w = image_size
    if format == "xy":
        normalized = keypoints.clone()
        normalized[..., 0] = normalized[..., 0] / w
        normalized[..., 1] = normalized[..., 1] / h
    else:
        normalized = keypoints.clone()
        normalized[..., 0] = normalized[..., 0] / w
        normalized[..., 1] = normalized[..., 1] / h
    return normalized


def denormalize_keypoints(keypoints: Tensor, image_size: Tuple[int, int]) -> Tensor:
    """
    Denormalize keypoints from [0, 1] to pixel coordinates.

    Args:
        keypoints: Tensor of shape (N, 2) or (N, 3) in [0, 1]
        image_size: (height, width) of the target image

    Returns:
        Denormalized keypoints in pixel coordinates
    """
    h, w = image_size
    denormalized = keypoints.clone()
    denormalized[..., 0] = denormalized[..., 0] * w
    denormalized[..., 1] = denormalized[..., 1] * h
    return denormalized


def compute_keypoint_heatmap(
    keypoints: Tensor,
    heatmap_size: Tuple[int, int],
    sigma: float = 2.0,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Generate Gaussian heatmaps for keypoints.

    Args:
        keypoints: Tensor of shape (N, 2) with (x, y) coordinates
        heatmap_size: (height, width) of the output heatmap
        sigma: Standard deviation of the Gaussian kernel
        valid_mask: Optional mask indicating valid keypoints

    Returns:
        Heatmaps of shape (N, H, W)
    """
    N = keypoints.shape[0]
    H, W = heatmap_size

    heatmaps = torch.zeros(N, H, W, device=keypoints.device, dtype=keypoints.dtype)

    if valid_mask is None:
        valid_mask = torch.ones(N, dtype=torch.bool, device=keypoints.device)

    for i in range(N):
        if not valid_mask[i]:
            continue

        x, y = keypoints[i]

        if x < 0 or x >= W or y < 0 or y >= H:
            continue

        xx, yy = torch.meshgrid(
            torch.arange(W, device=keypoints.device),
            torch.arange(H, device=keypoints.device),
            indexing="xy",
        )

        xx = xx.float() - x
        yy = yy.float() - y

        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        heatmaps[i] = gaussian

    return heatmaps


def compute_paf(
    keypoints: Tensor,
    skeleton: List[Tuple[int, int]],
    paf_size: Tuple[int, int],
    thickness: int = 1,
) -> Tensor:
    """
    Compute Part Affinity Fields for limb connections.

    Args:
        keypoints: Tensor of shape (N, 2) with (x, y) coordinates
        skeleton: List of (parent, child) index pairs
        paf_size: (height, width) of the output PAF maps
        thickness: Thickness of the PAF limb

    Returns:
        PAF maps of shape (2 * len(skeleton), H, W)
    """
    num_limbs = len(skeleton)
    H, W = paf_size

    pafs = torch.zeros(2 * num_limbs, H, W, device=keypoints.device)

    for limb_idx, (p_idx, c_idx) in enumerate(skeleton):
        if p_idx >= keypoints.shape[0] or c_idx >= keypoints.shape[0]:
            continue

        p1 = keypoints[p_idx]
        p2 = keypoints[c_idx]

        if p1[0] < 0 or p1[1] < 0 or p2[0] < 0 or p2[1] < 0:
            continue
        if p1[0] >= W or p1[1] >= H or p2[0] >= W or p2[1] >= H:
            continue

        xx, yy = torch.meshgrid(
            torch.arange(W, device=keypoints.device),
            torch.arange(H, device=keypoints.device),
            indexing="xy",
        )

        xx = xx.float()
        yy = yy.float()

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        d = torch.sqrt(dx**2 + dy**2) + 1e-8

        ux = dx / d
        uy = dy / d

        vx = yy - p1[1]
        vy = -(xx - p1[0])

        proj = vx * ux + vy * uy
        dist = torch.abs(vx * uy - vy * ux)

        limb_mask = (dist < thickness) & (proj > 0) & (proj < d)

        pafs[2 * limb_idx][limb_mask] = ux
        pafs[2 * limb_idx + 1][limb_mask] = uy

    return pafs


def flip_keypoints(
    keypoints: Tensor,
    image_width: int,
    flip_pairs: Optional[List[Tuple[int, int]]] = None,
) -> Tensor:
    """
    Flip keypoints horizontally.

    Args:
        keypoints: Tensor of shape (N, 2) with (x, y) coordinates
        image_width: Width of the image for coordinate transformation
        flip_pairs: Optional pairs of keypoints to swap (e.g., left-right)

    Returns:
        Flipped keypoints
    """
    flipped = keypoints.clone()
    flipped[..., 0] = image_width - flipped[..., 0] - 1

    if flip_pairs is not None:
        for left_idx, right_idx in flip_pairs:
            if left_idx < keypoints.shape[0] and right_idx < keypoints.shape[0]:
                flipped[left_idx], flipped[right_idx] = (
                    flipped[right_idx].clone(),
                    flipped[left_idx].clone(),
                )

    return flipped


def rotate_keypoints(keypoints: Tensor, center: Tensor, angle: float) -> Tensor:
    """
    Rotate keypoints around a center point.

    Args:
        keypoints: Tensor of shape (N, 2) with (x, y) coordinates
        center: Center of rotation (2,)
        angle: Rotation angle in radians

    Returns:
        Rotated keypoints
    """
    cos_a = torch.cos(torch.tensor(angle))
    sin_a = torch.sin(torch.tensor(angle))

    centered = keypoints - center
    rotated_x = centered[..., 0] * cos_a - centered[..., 1] * sin_a
    rotated_y = centered[..., 0] * sin_a + centered[..., 1] * cos_a

    return torch.stack([rotated_x, rotated_y], dim=-1) + center


def scale_keypoints(
    keypoints: Tensor,
    scale_factor: Union[float, Tuple[float, float]],
    center: Optional[Tensor] = None,
) -> Tensor:
    """
    Scale keypoints.

    Args:
        keypoints: Tensor of shape (N, 2) with (x, y) coordinates
        scale_factor: Single float or (scale_x, scale_y) tuple
        center: Optional center point for scaling (default: origin)

    Returns:
        Scaled keypoints
    """
    if isinstance(scale_factor, float):
        scale_factor = (scale_factor, scale_factor)

    if center is None:
        center = torch.zeros(2, device=keypoints.device, dtype=keypoints.dtype)

    centered = keypoints - center
    scaled = centered.clone()
    scaled[..., 0] = scaled[..., 0] * scale_factor[0]
    scaled[..., 1] = scaled[..., 1] * scale_factor[1]

    return scaled + center
