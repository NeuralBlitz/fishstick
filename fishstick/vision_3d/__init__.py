"""
3D Computer Vision Module for fishstick

Comprehensive 3D vision toolkit including:
- Point cloud processing and feature extraction
- 3D object detection (PointPillars, voxel-based)
- Depth estimation (monocular, stereo)
- NeRF primitives and rendering
- 3D reconstruction (occupancy, TSDF, mesh)

Built with SE(3)-equivariant architectures and geometric deep learning principles.
"""

from typing import Tuple, List, Optional, Union, Dict, Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

# Point Cloud Processing
from fishstick.vision_3d.point_cloud import (
    farthest_point_sample,
    furthest_point_sample,
    pointnet_fp,
    gather_points,
    square_distance,
    knn_query,
    query_ball_point,
    sample_and_group,
)

# PointNet-style Networks
from fishstick.vision_3d.point_net import (
    PointNetEncoder,
    PointNetCls,
    PointNetSeg,
    PointNetDenseCls,
    TNet,
    FeatureTransformer,
)

# Voxel Operations
from fishstick.vision_3d.voxel_grid import (
    voxelize,
    VoxelGrid,
    PointPillarsScatter,
    DynamicVoxelGrid,
)

# 3D Detection
from fishstick.vision_3d.detection_3d import (
    BoundingBox3D,
    nms_3d,
    iou_3d,
    box_coder,
    box_decoder,
    convert_box_to_corners,
    convert_corners_to_box,
)

# PointPillars
from fishstick.vision_3d.point_pillars import (
    PointPillarsBackbone,
    PointPillarsHead,
    PointPillars,
    PillarFeatureNet,
)

# 3D ROI Pooling
from fishstick.vision_3d.roi_pooling_3d import (
    RoI3DPool,
    PriorBoxGenerator3D,
)

# Depth Estimation
from fishstick.vision_3d.depth_models import (
    DepthEncoder,
    DepthDecoder,
    DepthDecoderUpconv,
    MonocularDepthEstimator,
    ResNetDepthEncoder,
)

# Monodepth
from fishstick.vision_3d.monodepth import (
    MonodepthModel,
    MonodepthEncoder,
    MonodepthDecoder,
    DispResNet,
)

# Depth Losses
from fishstick.vision_3d.depth_losses import (
    DepthLoss,
    SSIMLoss,
    SmoothnessLoss,
    DisparitySmoothnessLoss,
    ReconstructionLoss,
)

# NeRF Core
from fishstick.vision_3d.nerf_core import (
    NeRF,
    NerfModel,
    PositionalEncoder,
    NeRFRenderer,
    VolumetricRenderer,
    hierarchical_sampling,
)

# Positional Encoding
from fishstick.vision_3d.positional_encoding import (
    FourierFeatures,
    GaussianFourierFeatures,
    SinusoidalPositionEncoder,
    get_nerf_positional_encoding,
)

# NeRF Losses
from fishstick.vision_3d.nerf_losses import (
    NerfLoss,
    RGBLoss,
    MSE,
    PSNR,
)

# Occupancy Networks
from fishstick.vision_3d.occupancy import (
    OccupancyNetwork,
    OccupancyField,
    ConvOccupancyNetwork,
    ImplicitSurface,
)

# TSDF Fusion
from fishstick.vision_3d.tsdf import (
    TSDFVolume,
    fuse_depth,
    integrate_tsdf,
    marching_cubes_tsdf,
)

# Mesh Generation
from fishstick.vision_3d.mesh_generation import (
    MarchingCubes,
    extract_mesh,
    poisson_surface_reconstruction,
    mesh_from_occupancy,
)

# 3D Transforms
from fishstick.vision_3d.transforms_3d import (
    rotate_3d,
    translate_3d,
    scale_3d,
    random_rotation_3d,
    random_translate_3d,
    transform_point_cloud,
    transform_matrix,
    euler_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    look_at,
)

# 3D Metrics
from fishstick.vision_3d.metrics_3d import (
    chamfer_distance,
    earth_mover_distance,
    f1_score_3d,
    iou_3d_metric,
    precision_recall_3d,
)

# 3D Visualization
from fishstick.vision_3d.visualization_3d import (
    visualize_point_cloud,
    visualize_bounding_boxes_3d,
    visualize_depth,
    create_3d_scatter,
    create_mesh_actor,
)

# Data Utils
from fishstick.vision_3d.data_utils import (
    PointCloudDataset,
    collate_point_cloud,
    read_point_cloud,
    read_ply,
    read_pcd,
    save_point_cloud,
    voxel_downsample,
)


__all__ = [
    # Point Cloud Processing
    "farthest_point_sample",
    "furthest_point_sample",
    "pointnet_fp",
    "gather_points",
    "square_distance",
    "knn_query",
    "query_ball_point",
    "sample_and_group",
    # PointNet
    "PointNetEncoder",
    "PointNetCls",
    "PointNetSeg",
    "PointNetDenseCls",
    "TNet",
    "FeatureTransformer",
    # Voxel
    "voxelize",
    "VoxelGrid",
    "PointPillarsScatter",
    "DynamicVoxelGrid",
    # 3D Detection
    "BoundingBox3D",
    "nms_3d",
    "iou_3d",
    "box_coder",
    "box_decoder",
    "convert_box_to_corners",
    "convert_corners_to_box",
    # PointPillars
    "PointPillarsBackbone",
    "PointPillarsHead",
    "PointPillars",
    "PillarFeatureNet",
    # 3D ROI
    "RoI3DPool",
    "PriorBoxGenerator3D",
    # Depth Models
    "DepthEncoder",
    "DepthDecoder",
    "DepthDecoderUpconv",
    "MonocularDepthEstimator",
    "ResNetDepthEncoder",
    # Monodepth
    "MonodepthModel",
    "MonodepthEncoder",
    "MonodepthDecoder",
    "DispResNet",
    # Depth Losses
    "DepthLoss",
    "SSIMLoss",
    "SmoothnessLoss",
    "DisparitySmoothnessLoss",
    "ReconstructionLoss",
    # NeRF Core
    "NeRF",
    "NerfModel",
    "PositionalEncoder",
    "NeRFRenderer",
    "VolumetricRenderer",
    "hierarchical_sampling",
    # Positional Encoding
    "FourierFeatures",
    "GaussianFourierFeatures",
    "SinusoidalPositionEncoder",
    "get_nerf_positional_encoding",
    # NeRF Losses
    "NerfLoss",
    "RGBLoss",
    "MSE",
    "PSNR",
    # Occupancy
    "OccupancyNetwork",
    "OccupancyField",
    "ConvOccupancyNetwork",
    "ImplicitSurface",
    # TSDF
    "TSDFVolume",
    "fuse_depth",
    "integrate_tsdf",
    "marching_cubes_tsdf",
    # Mesh
    "MarchingCubes",
    "extract_mesh",
    "poisson_surface_reconstruction",
    "mesh_from_occupancy",
    # Transforms
    "rotate_3d",
    "translate_3d",
    "scale_3d",
    "random_rotation_3d",
    "random_translate_3d",
    "transform_point_cloud",
    "transform_matrix",
    "euler_to_rotation_matrix",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "look_at",
    # Metrics
    "chamfer_distance",
    "earth_mover_distance",
    "f1_score_3d",
    "iou_3d_metric",
    "precision_recall_3d",
    # Visualization
    "visualize_point_cloud",
    "visualize_bounding_boxes_3d",
    "visualize_depth",
    "create_3d_scatter",
    "create_mesh_actor",
    # Data Utils
    "PointCloudDataset",
    "collate_point_cloud",
    "read_point_cloud",
    "read_ply",
    "read_pcd",
    "save_point_cloud",
    "voxel_downsample",
]
