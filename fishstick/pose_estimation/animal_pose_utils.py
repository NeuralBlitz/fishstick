"""
Animal Pose Estimation Utilities

Utility functions for animal pose estimation including keypoint
transformations, animal-specific metrics, and preprocessing.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field

import torch
from torch import Tensor
import numpy as np


ANIMAL_KEYPOINT_SETS = {
    "horse": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "neck",
        "withers",
        "tail1",
        "tail2",
        "tail3",
        "left_shoulder",
        "left_elbow",
        "left_front_hoof",
        "right_shoulder",
        "right_elbow",
        "right_front_hoof",
        "spine",
        "left_hip",
        "left_stifle",
        "left_back_hoof",
        "right_hip",
        "right_stifle",
        "right_back_hoof",
    ],
    "dog": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "neck",
        "chest",
        "abdomen",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "left_paw",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "right_paw",
        "hip",
        "left_knee",
        "left_ankle",
        "left_paw_back",
        "right_knee",
        "right_ankle",
        "right_paw_back",
        "tail_base",
        "tail_mid",
        "tail_tip",
    ],
    "cat": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "head",
        "neck",
        "chest",
        "abdomen",
        "pelvis",
        "left_front_paw",
        "right_front_paw",
        "left_back_paw",
        "right_back_paw",
        "tail1",
        "tail2",
        "tail3",
    ],
    "bird": [
        "beak",
        "head",
        "neck",
        "chest",
        "back",
        "left_wing_tip",
        "right_wing_tip",
        "left_wing_mid",
        "right_wing_mid",
        "tail_tip",
        "tail_base",
        "left_leg",
        "right_leg",
        "left_foot",
        "right_foot",
    ],
    "generic": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "neck",
        "chest",
        "abdomen",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_paw",
        "right_paw",
        "hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_foot",
        "right_foot",
    ],
}

ANIMAL_SKELETONS = {
    "dog": [
        (0, 1),
        (1, 2),
        (0, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (7, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (7, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (7, 18),
        (18, 19),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
    ],
    "horse": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 16),
        (16, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (16, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (7, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (7, 21),
        (21, 22),
        (22, 23),
        (23, 24),
    ],
    "cat": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 14),
        (14, 17),
        (9, 11),
        (11, 15),
        (15, 18),
        (9, 12),
        (12, 16),
        (16, 19),
        (9, 13),
        (13, 20),
        (20, 21),
    ],
}

DEEPLABCUT_BODYPARTS = [
    "AnkleR",
    "KneeR",
    "HipR",
    "HipL",
    "KneeL",
    "AnkleL",
    "FootR",
    "ToesR",
    "FootL",
    "ToesL",
    "ShoulderR",
    "ElbowR",
    "WristR",
    "ShoulderL",
    "ElbowL",
    "WristL",
    "ElbowR",
    "Back",
    "Head",
    "Nose",
]

MacaqueFaceTrader_KEYPOINTS = [
    "nose",
    "eye_l",
    "eye_r",
    "ear_l",
    "ear_r",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
]


@dataclass
class AnimalPose:
    """
    Represents an animal pose with multiple keypoints.

    Attributes:
        keypoints: Tensor of shape (N, 3) containing x, y, z coordinates
        keypoint_names: List of keypoint names
        animal_type: Type of animal (e.g., "dog", "horse", "cat")
        visibility: Tensor of shape (N,) indicating keypoint visibility
        confidence: Overall confidence score
    """

    keypoints: Tensor
    keypoint_names: List[str]
    animal_type: str = "generic"
    visibility: Optional[Tensor] = None
    confidence: float = 1.0
    bbox: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self):
        if self.visibility is None:
            self.visibility = torch.ones(len(self.keypoint_names), dtype=torch.long)

        if len(self.keypoint_names) != self.keypoints.shape[0]:
            raise ValueError(
                f"Number of keypoints ({self.keypoints.shape[0]}) must match "
                f"number of keypoint names ({len(self.keypoint_names)})"
            )

    @property
    def num_keypoints(self) -> int:
        return self.keypoints.shape[0]

    def get_keypoint(self, name: str) -> Optional[Tensor]:
        """Get keypoint by name."""
        try:
            idx = self.keypoint_names.index(name)
            return self.keypoints[idx]
        except ValueError:
            return None

    def to_tensor(self) -> Tensor:
        """Convert animal pose to flattened tensor."""
        result = self.keypoints.clone()
        if self.visibility is not None:
            result = torch.cat([result, self.visibility.unsqueeze(1)], dim=1)
        return result.flatten()


def animal_keypoints_to_tensor(
    keypoints: List[Tuple[float, float, float]],
    keypoint_names: List[str],
    visibility: Optional[List[int]] = None,
) -> Tensor:
    """
    Convert animal keypoints to tensor representation.

    Args:
        keypoints: List of (x, y, z) tuples
        keypoint_names: List of keypoint names
        visibility: Optional list of visibility values

    Returns:
        Tensor of shape (N, 3) or (N, 4) with visibility
    """
    if len(keypoints) != len(keypoint_names):
        raise ValueError(
            f"Number of keypoints ({len(keypoints)}) must match "
            f"number of keypoint names ({len(keypoint_names)})"
        )

    tensor = torch.tensor(keypoints, dtype=torch.float32)

    if visibility is not None:
        vis_tensor = torch.tensor(visibility, dtype=torch.long).unsqueeze(1)
        tensor = torch.cat([tensor, vis_tensor], dim=1)

    return tensor


def tensor_to_animal_keypoints(
    tensor: Tensor,
    keypoint_names: List[str],
    has_visibility: bool = True,
) -> Tuple[List[Tuple[float, float, float]], List[int]]:
    """
    Convert tensor to animal keypoints.

    Args:
        tensor: Tensor of shape (N, 3) or (N, 4)
        keypoint_names: List of keypoint names
        has_visibility: Whether tensor contains visibility

    Returns:
        Tuple of (keypoints list, visibility list)
    """
    n_keypoints = len(keypoint_names)

    if has_visibility and tensor.shape[1] == 4:
        keypoints = tensor[:, :3].tolist()
        visibility = tensor[:, 3].long().tolist()
    else:
        keypoints = tensor[:, :3].tolist()
        visibility = [2] * n_keypoints

    return keypoints, visibility


def get_animal_skeleton(animal_type: str) -> List[Tuple[int, int]]:
    """
    Get skeleton connectivity for animal type.

    Args:
        animal_type: Type of animal

    Returns:
        List of (source, target) index pairs
    """
    return ANIMAL_SKELETONS.get(animal_type, ANIMAL_SKELETONS.get("dog"))


def animal_skeleton_template(animal_type: str) -> Dict[str, Any]:
    """
    Get full skeleton template for animal type.

    Args:
        animal_type: Type of animal

    Returns:
        Dictionary with keypoint_names, skeleton, and metadata
    """
    keypoints = ANIMAL_KEYPOINT_SETS.get(animal_type, ANIMAL_KEYPOINT_SETS["generic"])
    skeleton = get_animal_skeleton(animal_type)

    return {
        "keypoint_names": keypoints,
        "skeleton": skeleton,
        "num_keypoints": len(keypoints),
        "animal_type": animal_type,
    }


def compute_animal_pck(
    pred_keypoints: Tensor,
    gt_keypoints: Tensor,
    threshold: float = 0.2,
    normalize_by: Optional[Tensor] = None,
) -> float:
    """
    Compute Percentage of Correct Keypoints for animal pose.

    Args:
        pred_keypoints: Predicted keypoints (N, 2) or (N, 3)
        gt_keypoints: Ground truth keypoints (N, 2) or (N, 3)
        threshold: PCK threshold as fraction of body size
        normalize_by: Optional normalization factor

    Returns:
        PCK score
    """
    if pred_keypoints.shape != gt_keypoints.shape:
        raise ValueError("Keypoint shapes must match")

    distances = torch.norm(pred_keypoints - gt_keypoints, dim=1)

    if normalize_by is None:
        if gt_keypoints.shape[1] >= 2:
            size = torch.norm(gt_keypoints.max(dim=0)[0] - gt_keypoints.min(dim=0)[0])
        else:
            size = 1.0
    else:
        size = normalize_by

    threshold_px = size * threshold
    correct = (distances < threshold_px).sum().item()
    total = pred_keypoints.shape[0]

    return correct / total


def compute_animal_auc(
    pred_keypoints: Tensor,
    gt_keypoints: Tensor,
    max_threshold: float = 0.3,
    num_thresholds: int = 30,
    normalize_by: Optional[Tensor] = None,
) -> float:
    """
    Compute Area Under Curve for animal pose.

    Args:
        pred_keypoints: Predicted keypoints (N, 2) or (N, 3)
        gt_keypoints: Ground truth keypoints (N, 2) or (N, 3)
        max_threshold: Maximum threshold for AUC
        num_thresholds: Number of thresholds to evaluate
        normalize_by: Optional normalization factor

    Returns:
        AUC score
    """
    thresholds = torch.linspace(0, max_threshold, num_thresholds)

    distances = torch.norm(pred_keypoints - gt_keypoints, dim=1)

    if normalize_by is None:
        if gt_keypoints.shape[1] >= 2:
            size = torch.norm(gt_keypoints.max(dim=0)[0] - gt_keypoints.min(dim=0)[0])
        else:
            size = 1.0
    else:
        size = normalize_by

    pck_scores = []
    for thresh in thresholds:
        threshold_px = size * thresh
        pck = (distances < threshold_px).float().mean().item()
        pck_scores.append(pck)

    auc = torch.trapz(torch.tensor(pck_scores), thresholds).item()
    return auc


def align_animal_poses(
    pred_pose: AnimalPose,
    gt_pose: AnimalPose,
    method: str = "procrustes",
) -> AnimalPose:
    """
    Align predicted pose to ground truth pose.

    Args:
        pred_pose: Predicted animal pose
        gt_pose: Ground truth animal pose
        method: Alignment method ("procrustes" or "similarity")

    Returns:
        Aligned predicted pose
    """
    pred_kp = pred_pose.keypoints
    gt_kp = gt_pose.keypoints

    if method == "procrustes":
        aligned = compute_similarity_transform(pred_kp, gt_kp)
    elif method == "similarity":
        pred_center = pred_kp.mean(dim=0)
        gt_center = gt_kp.mean(dim=0)

        pred_scale = torch.norm(pred_kp - pred_center)
        gt_scale = torch.norm(gt_kp - gt_center)

        if pred_scale > 1e-6:
            scale = gt_scale / pred_scale
        else:
            scale = 1.0

        aligned = (pred_kp - pred_center) * scale + gt_center
    else:
        raise ValueError(f"Unknown alignment method: {method}")

    aligned_pose = AnimalPose(
        keypoints=aligned,
        keypoint_names=pred_pose.keypoint_names,
        animal_type=pred_pose.animal_type,
        visibility=pred_pose.visibility,
        confidence=pred_pose.confidence,
    )

    return aligned_pose


def compute_similarity_transform(
    pred: Tensor,
    gt: Tensor,
    reflective: bool = True,
) -> Tensor:
    """
    Compute similarity transform using SVD.

    Args:
        pred: Predicted keypoints (N, 3)
        gt: Ground truth keypoints (N, 3)
        reflective: Whether to allow reflection

    Returns:
        Aligned keypoints
    """
    pred_centered = pred - pred.mean(dim=0)
    gt_centered = gt - gt.mean(dim=0)

    pred_scale = torch.sqrt((pred_centered**2).sum() / pred.shape[0])
    gt_scale = torch.sqrt((gt_centered**2).sum() / gt.shape[0])

    pred_normalized = pred_centered / (pred_scale + 1e-8)
    gt_normalized = gt_centered / (gt_scale + 1e-8)

    U, S, Vt = torch.linalg.svd(pred_normalized.T @ gt_normalized)

    R = Vt.T @ U.T

    if not reflective:
        det = torch.det(R)
        if det < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

    scale = gt_scale / (pred_scale + 1e-8)
    aligned = (pred - pred.mean(dim=0)) @ R.T * scale + gt.mean(dim=0)

    return aligned


def animal_crop_and_resize(
    image: Tensor,
    keypoints: Tensor,
    crop_size: Tuple[int, int] = (384, 384),
    padding: float = 0.2,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Crop and resize animal region from image.

    Args:
        image: Image tensor of shape (C, H, W) or (B, C, H, W)
        keypoints: Keypoints tensor of shape (N, 2) or (N, 3)
        crop_size: Output crop size (width, height)
        padding: Padding ratio around bounding box

    Returns:
        Tuple of (cropped image, transform info)
    """
    if image.dim() == 3:
        is_batch = False
        image = image.unsqueeze(0)
    else:
        is_batch = True

    B, C, H, W = image.shape

    if keypoints.dim() == 3:
        keypoints = keypoints[0]

    kp = keypoints[:, :2] if keypoints.shape[1] >= 2 else keypoints

    x_min, y_min = kp.min(dim=0)[0]
    x_max, y_max = kp.max(dim=0)[0]

    width = x_max - x_min
    height = y_max - y_min

    x_min -= width * padding
    y_min -= height * padding
    x_max += width * padding
    y_max += height * padding

    x1 = int(max(0, x_min))
    y1 = int(max(0, y_min))
    x2 = int(min(W, x_max))
    y2 = int(min(H, y_max))

    cropped = image[:, :, y1:y2, x1:x2]

    crop_h, crop_w = max(y2 - y1, 1), max(x2 - x1, 1)
    scale_x = crop_size[0] / crop_w
    scale_y = crop_size[1] / crop_h

    cropped_resized = torch.nn.functional.interpolate(
        cropped,
        size=crop_size,
        mode="bilinear",
        align_corners=False,
    )

    transform = {
        "crop_box": (x1, y1, x2, y2),
        "scale": (scale_x, scale_y),
        "original_size": (crop_w, crop_h),
    }

    if not is_batch:
        cropped_resized = cropped_resized.squeeze(0)

    return cropped_resized, transform


def flip_animal_keypoints(
    keypoints: Tensor,
    keypoint_names: List[str],
    image_width: int,
    flip_horizontal: bool = True,
) -> Tensor:
    """
    Flip animal keypoints horizontally or vertically.

    Args:
        keypoints: Keypoints tensor of shape (N, 2) or (N, 3)
        keypoint_names: List of keypoint names
        image_width: Width of the image
        flip_horizontal: True for horizontal flip, False for vertical

    Returns:
        Flipped keypoints
    """
    flipped = keypoints.clone()

    if flip_horizontal:
        flipped[:, 0] = image_width - flipped[:, 0]

        left_right_pairs = [
            ("left_eye", "right_eye"),
            ("left_ear", "right_ear"),
            ("left_shoulder", "right_shoulder"),
            ("left_elbow", "right_elbow"),
            ("left_wrist", "right_wrist"),
            ("left_paw", "right_paw"),
            ("left_front_paw", "right_front_paw"),
            ("left_back_paw", "right_back_paw"),
            ("left_hip", "right_hip"),
            ("left_knee", "right_knee"),
            ("left_ankle", "right_ankle"),
            ("left_foot", "right_foot"),
        ]

        for left_name, right_name in left_right_pairs:
            if left_name in keypoint_names and right_name in keypoint_names:
                left_idx = keypoint_names.index(left_name)
                right_idx = keypoint_names.index(right_name)
                flipped[[left_idx, right_idx]] = flipped[[right_idx, left_idx]]
    else:
        flipped[:, 1] = image_width - flipped[:, 1]

    return flipped


def compute_animal_bone_length(
    keypoints: Tensor,
    skeleton: List[Tuple[int, int]],
) -> Tensor:
    """
    Compute bone lengths from keypoints and skeleton.

    Args:
        keypoints: Keypoints tensor of shape (N, 2) or (N, 3)
        skeleton: List of (source, target) index pairs

    Returns:
        Bone lengths tensor of shape (len(skeleton),)
    """
    bone_lengths = []

    for src_idx, dst_idx in skeleton:
        if src_idx < keypoints.shape[0] and dst_idx < keypoints.shape[0]:
            bone = torch.norm(keypoints[src_idx] - keypoints[dst_idx])
            bone_lengths.append(bone)
        else:
            bone_lengths.append(torch.tensor(0.0))

    return torch.stack(bone_lengths)


def interpolate_animal_pose(
    pose1: AnimalPose,
    pose2: AnimalPose,
    alpha: float,
) -> AnimalPose:
    """
    Interpolate between two animal poses.

    Args:
        pose1: First pose
        pose2: Second pose
        alpha: Interpolation factor (0 = pose1, 1 = pose2)

    Returns:
        Interpolated pose
    """
    if pose1.num_keypoints != pose2.num_keypoints:
        raise ValueError("Poses must have same number of keypoints")

    if pose1.keypoint_names != pose2.keypoint_names:
        raise ValueError("Poses must have same keypoint names")

    interpolated_kp = (1 - alpha) * pose1.keypoints + alpha * pose2.keypoints

    vis1 = (
        pose1.visibility
        if pose1.visibility is not None
        else torch.ones(pose1.num_keypoints)
    )
    vis2 = (
        pose2.visibility
        if pose2.visibility is not None
        else torch.ones(pose2.num_keypoints)
    )
    interpolated_vis = ((vis1.float() + vis2.float()) / 2).long()

    interpolated_conf = (1 - alpha) * pose1.confidence + alpha * pose2.confidence

    return AnimalPose(
        keypoints=interpolated_kp,
        keypoint_names=pose1.keypoint_names,
        animal_type=pose1.animal_type,
        visibility=interpolated_vis,
        confidence=interpolated_conf,
    )


__all__ = [
    "ANIMAL_KEYPOINT_SETS",
    "ANIMAL_SKELETONS",
    "DEEPLABCUT_BODYPARTS",
    "MacaqueFaceTrader_KEYPOINTS",
    "AnimalPose",
    "animal_keypoints_to_tensor",
    "tensor_to_animal_keypoints",
    "get_animal_skeleton",
    "animal_skeleton_template",
    "compute_animal_pck",
    "compute_animal_auc",
    "align_animal_poses",
    "compute_similarity_transform",
    "animal_crop_and_resize",
    "flip_animal_keypoints",
    "compute_animal_bone_length",
    "interpolate_animal_pose",
]
