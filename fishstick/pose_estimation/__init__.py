"""
Pose Estimation Module for fishstick

Comprehensive pose estimation toolkit including:
- 2D human pose estimation (HRNet, etc.)
- 3D human pose estimation (VideoPose, etc.)
- Hand pose estimation
- Animal pose estimation
- Keypoint detection

Built with geometric deep learning principles and SE(3)-equivariant architectures.

Example:
    >>> from fishstick.pose_estimation import HumanPose2D, HumanPose3D
    >>> from fishstick.pose_estimation import HandPoseEstimator
    >>> from fishstick.pose_estimation import KeypointDetector
    >>> from fishstick.pose_estimation import PoseVisualizer
"""

from typing import Tuple, List, Optional, Dict, Any, Union
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from .base import (
    Keypoint,
    Pose2D,
    Pose3D,
    KeypointSet,
    PoseSkeleton,
    coco_keypoints,
    coco_skeleton,
    mpii_keypoints,
    hand_keypoints,
    animal_keypoints,
    keypoint_to_tensor,
    tensor_to_keypoints,
    normalize_keypoints,
    denormalize_keypoints,
    compute_keypoint_heatmap,
    compute_paf,
    flip_keypoints,
    rotate_keypoints,
    scale_keypoints,
)

from .pose2d import (
    HRNet,
    HRNetEncoder,
    HRNetDecoder,
    Pose2DModel,
    SimpleBaseline2D,
    StackedHourglass,
    create_hrnet_w32,
    create_hrnet_w48,
)

from .pose2d_losses import (
    Pose2DLoss,
    MSELoss,
    SmoothL1Loss,
    OKSLoss,
    WingLoss,
    HeatmapLoss,
    PAFLoss,
    CombinedPose2DLoss,
)

from .pose2d_utils import (
    compute_pck,
    compute_auc,
    computeoks,
    decode_heatmap,
    decode_heatmap_multi,
    get_max_preds,
    transform_predictions,
    project_to_image,
    batch_nms,
)

from .pose3d import (
    HumanPose3DModel,
    VideoPose3D,
    PoseAugmenter3D,
    TemporalModel,
    SemigraphConv,
    ModelPose3D,
    GCNPose3D,
    TransformerPose3D,
    create_videopose_spatial,
    create_videopose_temporal,
)

from .pose3d_losses import (
    Pose3DLoss,
    MPJPELoss,
    PCKLoss3D,
    ProcrustesLoss,
    BoneLengthLoss,
    VelocityLoss,
    CombinedPose3DLoss,
)

from .pose3d_utils import (
    compute_similarity_transform,
    align_by_pelvis,
    compute_joint_angle,
    compute_bone_length,
    compute_velocity,
    project_3d_to_2d,
    camera_to_world,
    world_to_camera,
)

from .hand_pose import (
    HandPoseModel,
    HandPoseNet,
    MediaPipeHand,
    HandPoseEstimator,
    ManoModel,
    InterhandModel,
    create_mano_layer,
)

from .hand_pose_utils import (
    hand_keypoints_to_tensor,
    tensor_to_hand_keypoints,
    compute_hand_visibility,
    compute_finger_bending,
    compute_palm_alignment,
    hand_crop_and_resize,
)

from .animal_pose import (
    AnimalPoseModel,
    OpenMonkey,
    AnimalKeypointRCNN,
    DeepLabCutModel,
    LEAPModel,
    create_animal_pose_model,
)

from .animal_pose_utils import (
    animal_keypoints_to_tensor,
    compute_animal_pck,
    align_animal_poses,
    animal_skeleton_template,
)

from .keypoint import (
    KeypointDetector,
    KeypointRCNN,
    CenterNetKeypoint,
    HourglassKeypoint,
    KeypointRCNNHead,
    create_keypoint_rcnn_resnet,
)

from .keypoint_losses import (
    KeypointLoss,
    KeypointHeatmapLoss,
    KeypointOffsetLoss,
    KeypointLossCombined,
)

from .keypoint_utils import (
    decode_keypoints_from_heatmap,
    get_keypoint_predictions,
    compute_keypoint_iou,
    nms_keypoints,
    group_keypoints,
    match_keypoints,
)

from .pose_visualization import (
    PoseVisualizer,
    draw_skeleton,
    draw_keypoints,
    draw_heatmap,
    draw_paf,
    create_pose_animation,
    plot_pose_statistics,
)

from .pose_datasets import (
    PoseDataset,
    COCOPoseDataset,
    MPIIPoseDataset,
    HandDataset,
    AnimalPoseDataset,
    PoseDataLoader,
    collate_pose,
)


__all__ = [
    # Base types
    "Keypoint",
    "Pose2D",
    "Pose3D",
    "KeypointSet",
    "PoseSkeleton",
    "coco_keypoints",
    "coco_skeleton",
    "mpii_keypoints",
    "hand_keypoints",
    "animal_keypoints",
    "keypoint_to_tensor",
    "tensor_to_keypoints",
    "normalize_keypoints",
    "denormalize_keypoints",
    "compute_keypoint_heatmap",
    "compute_paf",
    "flip_keypoints",
    "rotate_keypoints",
    "scale_keypoints",
    # 2D Pose
    "HRNet",
    "HRNetEncoder",
    "HRNetDecoder",
    "Pose2DModel",
    "SimpleBaseline2D",
    "StackedHourglass",
    "create_hrnet_w32",
    "create_hrnet_w48",
    # 2D Losses
    "Pose2DLoss",
    "MSELoss",
    "SmoothL1Loss",
    "OKSLoss",
    "WingLoss",
    "HeatmapLoss",
    "PAFLoss",
    "CombinedPose2DLoss",
    # 2D Utils
    "compute_pck",
    "compute_auc",
    "computeoks",
    "decode_heatmap",
    "decode_heatmap_multi",
    "get_max_preds",
    "transform_predictions",
    "project_to_image",
    "batch_nms",
    # 3D Pose
    "HumanPose3DModel",
    "VideoPose3D",
    "PoseAugmenter3D",
    "TemporalModel",
    "SemigraphConv",
    "ModelPose3D",
    "GCNPose3D",
    "TransformerPose3D",
    "create_videopose_spatial",
    "create_videopose_temporal",
    # 3D Losses
    "Pose3DLoss",
    "MPJPELoss",
    "PCKLoss3D",
    "ProcrustesLoss",
    "BoneLengthLoss",
    "VelocityLoss",
    "CombinedPose3DLoss",
    # 3D Utils
    "compute_similarity_transform",
    "align_by_pelvis",
    "compute_joint_angle",
    "compute_bone_length",
    "compute_velocity",
    "project_3d_to_2d",
    "camera_to_world",
    "world_to_camera",
    # Hand Pose
    "HandPoseModel",
    "HandPoseNet",
    "MediaPipeHand",
    "HandPoseEstimator",
    "ManoModel",
    "InterhandModel",
    "create_mano_layer",
    # Hand Utils
    "hand_keypoints_to_tensor",
    "tensor_to_hand_keypoints",
    "compute_hand_visibility",
    "compute_finger_bending",
    "compute_palm_alignment",
    "hand_crop_and_resize",
    # Animal Pose
    "AnimalPoseModel",
    "OpenMonkey",
    "AnimalKeypointRCNN",
    "DeepLabCutModel",
    "LEAPModel",
    "create_animal_pose_model",
    # Animal Utils
    "animal_keypoints_to_tensor",
    "compute_animal_pck",
    "align_animal_poses",
    "animal_skeleton_template",
    # Keypoint Detection
    "KeypointDetector",
    "KeypointRCNN",
    "CenterNetKeypoint",
    "HourglassKeypoint",
    "KeypointRCNNHead",
    "create_keypoint_rcnn_resnet",
    # Keypoint Losses
    "KeypointLoss",
    "KeypointHeatmapLoss",
    "KeypointOffsetLoss",
    "KeypointLossCombined",
    # Keypoint Utils
    "decode_keypoints_from_heatmap",
    "get_keypoint_predictions",
    "compute_keypoint_iou",
    "nms_keypoints",
    "group_keypoints",
    "match_keypoints",
    # Visualization
    "PoseVisualizer",
    "draw_skeleton",
    "draw_keypoints",
    "draw_heatmap",
    "draw_paf",
    "create_pose_animation",
    "plot_pose_statistics",
    # Datasets
    "PoseDataset",
    "COCOPoseDataset",
    "MPIIPoseDataset",
    "HandDataset",
    "AnimalPoseDataset",
    "PoseDataLoader",
    "collate_pose",
]
