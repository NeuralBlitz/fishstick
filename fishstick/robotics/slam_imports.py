"""
SLAM (Simultaneous Localization and Mapping) Module for Fishstick Robotics.

Comprehensive implementations of visual, LiDAR, visual-inertial, and hybrid SLAM algorithms
including state estimation, loop closure, mapping, and optimization techniques.

References:
- ORB-SLAM2: Mur-Artal et al., 2017
- LOAM: Zhang & Singh, 2014
- VINS-Mono: Qin et al., 2018
- DSO: Engel et al., 2017
- Cartographer: Hess et al., 2016
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Callable, Any, Set
from enum import Enum, auto
from abc import ABC, abstractmethod
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from collections import defaultdict
import heapq
from copy import deepcopy


# =============================================================================
# Core SLAM Types and Data Structures
# =============================================================================


class SensorType(Enum):
    """Type of sensor used for SLAM."""
    MONO_CAMERA = auto()
    STEREO_CAMERA = auto()
    RGBD_CAMERA = auto()
    LIDAR_2D = auto()
    LIDAR_3D = auto()
    IMU = auto()
    GPS = auto()
    ODOMETRY = auto()
    MULTI = auto()


class SLAMMode(Enum):
    """SLAM operational mode."""
    MAPPING = auto()
    LOCALIZATION = auto()
    MIXED = auto()


@dataclass
class SE3Pose:
    """
    SE(3) rigid body transformation.
    
    Represents pose as rotation matrix and translation vector.
    """
    R: Tensor  # Rotation matrix [3, 3]
    t: Tensor  # Translation vector [3]
    
    def __post_init__(self):
        if self.R.dim() == 2:
            self.R = self.R.unsqueeze(0)
        if self.t.dim() == 1:
            self.t = self.t.unsqueeze(0)
    
    @property
    def T(self) -> Tensor:
        """4x4 transformation matrix."""
        batch = self.R.shape[0]
        T = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
        T[:, :3, :3] = self.R
        T[:, :3, 3] = self.t
        return T
    
    @property
    def quaternion(self) -> Tensor:
        """Convert rotation matrix to quaternion [w, x, y, z]."""
        batch = self.R.shape[0]
        quat = torch.zeros(batch, 4, device=self.R.device, dtype=self.R.dtype)
        
        # Trace-based method
        trace = self.R[:, 0, 0] + self.R[:, 1, 1] + self.R[:, 2, 2]
        
        # w is largest
        mask1 = trace > 0
        s1 = torch.sqrt(trace[mask1] + 1.0) * 2
        quat[mask1, 0] = 0.25 * s1
        quat[mask1, 1] = (self.R[mask1, 2, 1] - self.R[mask1, 1, 2]) / s1
        quat[mask1, 2] = (self.R[mask1, 0, 2] - self.R[mask1, 2, 0]) / s1
        quat[mask1, 3] = (self.R[mask1, 1, 0] - self.R[mask1, 0, 1]) / s1
        
        # x is largest
        mask2 = (~mask1) & (self.R[:, 0, 0] > self.R[:, 1, 1]) & (self.R[:, 0, 0] > self.R[:, 2, 2])
        s2 = torch.sqrt(1.0 + self.R[mask2, 0, 0] - self.R[mask2, 1, 1] - self.R[mask2, 2, 2]) * 2
        quat[mask2, 0] = (self.R[mask2, 2, 1] - self.R[mask2, 1, 2]) / s2
        quat[mask2, 1] = 0.25 * s2
        quat[mask2, 2] = (self.R[mask2, 0, 1] + self.R[mask2, 1, 0]) / s2
        quat[mask2, 3] = (self.R[mask2, 0, 2] + self.R[mask2, 2, 0]) / s2
        
        # y is largest
        mask3 = (~mask1) & (~mask2) & (self.R[:, 1, 1] > self.R[:, 2, 2])
        s3 = torch.sqrt(1.0 + self.R[mask3, 1, 1] - self.R[mask3, 0, 0] - self.R[mask3, 2, 2]) * 2
        quat[mask3, 0] = (self.R[mask3, 0, 2] - self.R[mask3, 2, 0]) / s3
        quat[mask3, 1] = (self.R[mask3, 0, 1] + self.R[mask3, 1, 0]) / s3
        quat[mask3, 2] = 0.25 * s3
        quat[mask3, 3] = (self.R[mask3, 1, 2] + self.R[mask3, 2, 1]) / s3
        
        # z is largest
        mask4 = (~mask1) & (~mask2) & (~mask3)
        s4 = torch.sqrt(1.0 + self.R[mask4, 2, 2] - self.R[mask4, 0, 0] - self.R[mask4, 1, 1]) * 2
        quat[mask4, 0] = (self.R[mask4, 1, 0] - self.R[mask4, 0, 1]) / s4
        quat[mask4, 1] = (self.R[mask4, 0, 2] + self.R[mask4, 2, 0]) / s4
        quat[mask4, 2] = (self.R[mask4, 1, 2] + self.R[mask4, 2, 1]) / s4
        quat[mask4, 3] = 0.25 * s4
        
        return quat
    
    def inverse(self) -> 'SE3Pose':
        """Compute inverse transformation."""
        R_inv = self.R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, self.t.unsqueeze(-1)).squeeze(-1)
        return SE3Pose(R=R_inv, t=t_inv)
    
    def compose(self, other: 'SE3Pose') -> 'SE3Pose':
        """Compose two SE(3) poses: self * other."""
        R_new = torch.matmul(self.R, other.R)
        t_new = torch.matmul(self.R, other.t.unsqueeze(-1)).squeeze(-1) + self.t
        return SE3Pose(R=R_new, t=t_new)
    
    @staticmethod
    def from_quaternion(quat: Tensor, translation: Tensor) -> 'SE3Pose':
        """Create from quaternion [w, x, y, z] and translation."""
        if quat.dim() == 1:
            quat = quat.unsqueeze(0)
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
        
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        R = torch.zeros(quat.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)
        
        # First row
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        
        # Second row
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - x*w)
        
        # Third row
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        
        return SE3Pose(R=R, t=translation)
    
    @staticmethod
    def identity(batch_size: int = 1, device: str = 'cpu') -> 'SE3Pose':
        """Create identity pose."""
        return SE3Pose(
            R=torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
            t=torch.zeros(batch_size, 3, device=device)
        )


@dataclass
class Landmark:
    """
    3D landmark / map point representation.
    """
    id: int
    position: Tensor  # [3]
    descriptor: Optional[Tensor] = None  # Feature descriptor
    observations: List[int] = field(default_factory=list)  # Frame IDs
    color: Optional[Tensor] = None  # RGB color [3]
    
    def __post_init__(self):
        if isinstance(self.observations, list) and len(self.observations) == 0:
            self.observations = []


@dataclass
class KeyFrame:
    """
    Keyframe representation for SLAM.
    """
    id: int
    pose: SE3Pose
    image: Optional[Tensor] = None
    timestamp: float = 0.0
    landmarks: Dict[int, Landmark] = field(default_factory=dict)
    features: Optional[Tensor] = None  # Extracted features
    descriptors: Optional[Tensor] = None  # Feature descriptors
    is_fixed: bool = False
    
    def get_landmark_positions(self) -> Tensor:
        """Get all landmark positions as tensor."""
        if not self.landmarks:
            return torch.empty(0, 3)
        return torch.stack([lm.position for lm in self.landmarks.values()])


@dataclass
class PointCloud:
    """
    Point cloud representation for LiDAR data.
    """
    points: Tensor  # [N, 3]
    intensities: Optional[Tensor] = None  # [N]
    timestamps: Optional[Tensor] = None  # [N]
    ring_ids: Optional[Tensor] = None  # LiDAR ring information [N]
    
    def transform(self, pose: SE3Pose) -> 'PointCloud':
        """Transform point cloud by SE(3) pose."""
        points_homo = torch.cat([
            self.points,
            torch.ones(self.points.shape[0], 1, device=self.points.device)
        ], dim=1)
        T = pose.T.squeeze(0)
        transformed = torch.matmul(points_homo, T.T)
        return PointCloud(
            points=transformed[:, :3],
            intensities=self.intensities,
            timestamps=self.timestamps,
            ring_ids=self.ring_ids
        )
    
    def downsample(self, voxel_size: float) -> 'PointCloud':
        """Voxel grid downsampling."""
        if len(self.points) == 0:
            return self
        
        voxel_coords = torch.floor(self.points / voxel_size).long()
        unique_coords, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
        
        # Average points in each voxel
        downsampled_points = torch.zeros(len(unique_coords), 3, device=self.points.device)
        for i in range(len(unique_coords)):
            mask = inverse_indices == i
            downsampled_points[i] = self.points[mask].mean(dim=0)
        
        return PointCloud(
            points=downsampled_points,
            intensities=None,
            timestamps=None,
            ring_ids=None
        )


@dataclass
class IMUData:
    """
    IMU measurement data.
    """
    timestamp: float
    angular_velocity: Tensor  # [3] - gyroscope
    linear_acceleration: Tensor  # [3] - accelerometer
    orientation: Optional[Tensor] = None  # Quaternion [w, x, y, z] if available


@dataclass
class SLAMState:
    """
    Current SLAM system state.
    """
    current_pose: SE3Pose
    keyframes: Dict[int, KeyFrame] = field(default_factory=dict)
    landmarks: Dict[int, Landmark] = field(default_factory=dict)
    trajectory: List[SE3Pose] = field(default_factory=list)
    timestamp: float = 0.0
    is_tracking: bool = True
    is_lost: bool = False
