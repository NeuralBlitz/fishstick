"""
Hand Pose Estimation Utilities

Utility functions for hand pose estimation including keypoint
transformations, hand-specific metrics, and preprocessing.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

import torch
from torch import Tensor
import numpy as np
from scipy.spatial import distance


HAND_21_KEYPOINTS = [
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

HAND_SKELETON = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

MANO_MODEL_PATH = "mano/models"
MANO_N_VERTICES = 778
MANO_N_FACES = 1538

FINGER_INDICES = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "little": [17, 18, 19, 20],
}


@dataclass
class HandPose:
    """
    Represents a hand pose with 21 keypoints.

    Attributes:
        keypoints: Tensor of shape (21, 3) containing x, y, z coordinates
        visibility: Tensor of shape (21,) indicating keypoint visibility
        hand_type: "left" or "right"
        confidence: Overall confidence score
    """

    keypoints: Tensor
    visibility: Optional[Tensor] = None
    hand_type: str = "right"
    confidence: float = 1.0

    def __post_init__(self):
        if self.visibility is None:
            self.visibility = torch.ones(21, dtype=torch.long)

    @property
    def wrist(self) -> Tensor:
        return self.keypoints[0]

    @property
    def fingers(self) -> Dict[str, Tensor]:
        return {
            name: self.keypoints[indices] for name, indices in FINGER_INDICES.items()
        }

    def to_tensor(self) -> Tensor:
        """Convert hand pose to flattened tensor."""
        result = self.keypoints.clone()
        if self.visibility is not None:
            result = torch.cat([result, self.visibility.unsqueeze(1)], dim=1)
        return result.flatten()


def hand_keypoints_to_tensor(
    keypoints: List[Tuple[float, float, float]],
    visibility: Optional[List[int]] = None,
) -> Tensor:
    """
    Convert hand keypoints to tensor representation.

    Args:
        keypoints: List of (x, y, z) tuples for 21 hand keypoints
        visibility: Optional list of visibility values

    Returns:
        Tensor of shape (21, 3) or (21, 4) with visibility
    """
    if len(keypoints) != 21:
        raise ValueError(f"Expected 21 keypoints, got {len(keypoints)}")

    tensor = torch.tensor(keypoints, dtype=torch.float32)

    if visibility is not None:
        vis_tensor = torch.tensor(visibility, dtype=torch.long).unsqueeze(1)
        tensor = torch.cat([tensor, vis_tensor], dim=1)

    return tensor


def tensor_to_hand_keypoints(
    tensor: Tensor,
    has_visibility: bool = True,
) -> Tuple[List[Tuple[float, float, float]], List[int]]:
    """
    Convert tensor to hand keypoints.

    Args:
        tensor: Tensor of shape (21, 3) or (21, 4)
        has_visibility: Whether tensor contains visibility

    Returns:
        Tuple of (keypoints list, visibility list)
    """
    if has_visibility and tensor.shape[1] == 4:
        keypoints = tensor[:, :3].tolist()
        visibility = tensor[:, 3].long().tolist()
    else:
        keypoints = tensor.tolist()
        visibility = [2] * len(keypoints)

    return keypoints, visibility


def compute_hand_visibility(
    keypoints: Tensor,
    depth_threshold: float = 0.1,
    conf_threshold: float = 0.3,
) -> Tensor:
    """
    Compute visibility for hand keypoints based on depth and confidence.

    Args:
        keypoints: Tensor of shape (21, 3) with x, y, z
        depth_threshold: Threshold for depth-based visibility
        conf_threshold: Minimum confidence for visible keypoints

    Returns:
        Tensor of shape (21,) with visibility flags
    """
    visibility = torch.ones(21, dtype=torch.long) * 2

    if keypoints.shape[1] >= 3:
        depths = keypoints[:, 2]
        if depths.max() - depths.min() > depth_threshold:
            visibility = torch.ones(21, dtype=torch.long)

    return visibility


def compute_finger_bending(
    keypoints: Tensor,
    finger_name: str,
) -> Tensor:
    """
    Compute bending angle for a specific finger.

    Args:
        keypoints: Tensor of shape (21, 3)
        finger_name: One of "thumb", "index", "middle", "ring", "little"

    Returns:
        Angle in radians
    """
    indices = FINGER_INDICES.get(finger_name)
    if indices is None:
        raise ValueError(f"Unknown finger: {finger_name}")

    if len(indices) < 3:
        raise ValueError(f"Finger {finger_name} needs at least 3 keypoints")

    p1 = keypoints[indices[0]]
    p2 = keypoints[indices[1]]
    p3 = keypoints[indices[2]]

    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

    return angle


def compute_all_finger_bending(keypoints: Tensor) -> Dict[str, Tensor]:
    """
    Compute bending angles for all fingers.

    Args:
        keypoints: Tensor of shape (21, 3)

    Returns:
        Dictionary mapping finger names to angles
    """
    return {
        finger: compute_finger_bending(keypoints, finger)
        for finger in FINGER_INDICES.keys()
    }


def compute_palm_alignment(
    keypoints: Tensor,
    reference_normal: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute palm alignment angle relative to camera.

    Args:
        keypoints: Tensor of shape (21, 3)
        reference_normal: Optional reference palm normal

    Returns:
        Alignment angle in radians
    """
    wrist = keypoints[0]
    middle_mcp = keypoints[9]
    middle_tip = keypoints[12]

    palm_center = wrist
    finger_dir = middle_tip - middle_mcp

    if reference_normal is not None:
        cross = torch.cross(finger_dir, reference_normal)
        dot = torch.dot(finger_dir, reference_normal)
        angle = torch.atan2(torch.norm(cross), dot)
    else:
        angle = torch.atan2(finger_dir[1], finger_dir[0])

    return angle


def hand_crop_and_resize(
    image: Tensor,
    keypoints: Tensor,
    crop_size: Tuple[int, int] = (256, 256),
    center: Optional[Tensor] = None,
    scale: float = 2.5,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Crop and resize hand region from image.

    Args:
        image: Image tensor of shape (C, H, W) or (H, W, C)
        keypoints: Keypoints tensor of shape (N, 2) or (N, 3)
        crop_size: Output crop size (width, height)
        center: Optional center point for crop
        scale: Scale factor for bounding box

    Returns:
        Tuple of (cropped image, transform info)
    """
    if image.dim() == 3 and image.shape[0] == 3:
        is_batch = False
        image = image.unsqueeze(0)
    else:
        is_batch = True

    B, C, H, W = image.shape

    if keypoints.dim() == 2:
        kp = keypoints[:, :2]
    else:
        kp = keypoints

    if center is None:
        center = kp.mean(dim=0)

    bbox_size = (kp.max(dim=0)[0] - kp.min(dim=0)[0]).max() * scale
    bbox_size = max(bbox_size, 50.0)

    x1 = int(center[0] - bbox_size / 2)
    y1 = int(center[1] - bbox_size / 2)
    x2 = int(center[0] + bbox_size / 2)
    y2 = int(center[1] + bbox_size / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    cropped = image[:, :, y1:y2, x1:x2]

    crop_h, crop_w = y2 - y1, x2 - x1
    scale_x = crop_size[0] / crop_w
    scale_y = crop_size[1] / crop_h

    cropped_resized = torch.nn.functional.interpolate(
        cropped,
        size=crop_size,
        mode="bilinear",
        align_corners=False,
    )

    transform = {
        "center": center,
        "bbox_size": bbox_size,
        "crop_box": (x1, y1, x2, y2),
        "scale": (scale_x, scale_y),
    }

    if not is_batch:
        cropped_resized = cropped_resized.squeeze(0)

    return cropped_resized, transform


def inverse_hand_transform(
    keypoints: Tensor,
    transform: Dict[str, Any],
) -> Tensor:
    """
    Apply inverse transform to keypoints to get original coordinates.

    Args:
        keypoints: Transformed keypoints
        transform: Transform info from hand_crop_and_resize

    Returns:
        Keypoints in original image coordinates
    """
    scale_x, scale_y = transform["scale"]
    x1, y1, _, _ = transform["crop_box"]

    if keypoints.dim() == 3:
        keypoints = keypoints.squeeze(0)

    result = keypoints.clone()
    result[:, 0] = result[:, 0] / scale_x + x1
    result[:, 1] = result[:, 1] / scale_y + y1

    return result


def compute_hand_pck(
    pred_keypoints: Tensor,
    gt_keypoints: Tensor,
    threshold: float = 0.2,
) -> float:
    """
    Compute Percentage of Correct Keypoints for hand pose.

    Args:
        pred_keypoints: Predicted keypoints (N, 3)
        gt_keypoints: Ground truth keypoints (N, 3)
        threshold: PCK threshold as fraction of hand size

    Returns:
        PCK score
    """
    if pred_keypoints.shape != gt_keypoints.shape:
        raise ValueError("Keypoint shapes must match")

    distances = torch.norm(pred_keypoints - gt_keypoints, dim=1)

    hand_size = torch.norm(gt_keypoints[0] - gt_keypoints[9])
    threshold_px = hand_size * threshold

    correct = (distances < threshold_px).sum().item()
    total = pred_keypoints.shape[0]

    return correct / total


def compute_hand_auc(
    pred_keypoints: Tensor,
    gt_keypoints: Tensor,
    max_threshold: float = 0.3,
    num_thresholds: int = 30,
) -> float:
    """
    Compute Area Under Curve for hand pose.

    Args:
        pred_keypoints: Predicted keypoints (N, 3)
        gt_keypoints: Ground truth keypoints (N, 3)
        max_threshold: Maximum threshold for AUC
        num_thresholds: Number of thresholds to evaluate

    Returns:
        AUC score
    """
    thresholds = torch.linspace(0, max_threshold, num_thresholds)

    distances = torch.norm(pred_keypoints - gt_keypoints, dim=1)
    hand_size = torch.norm(gt_keypoints[0] - gt_keypoints[9])

    pck_scores = []
    for thresh in thresholds:
        threshold_px = hand_size * thresh
        pck = (distances < threshold_px).float().mean().item()
        pck_scores.append(pck)

    auc = torch.trapz(torch.tensor(pck_scores), thresholds).item()
    return auc


def estimate_hand_side(keypoints: Tensor) -> str:
    """
    Estimate whether hand is left or right based on keypoints.

    Args:
        keypoints: Tensor of shape (21, 3)

    Returns:
        "left" or "right"
    """
    wrist = keypoints[0]
    index_mcp = keypoints[5]
    pinky_mcp = keypoints[17]

    center = (index_mcp + pinky_mcp) / 2

    if wrist[0] < center[0]:
        return "left"
    return "right"


def refine_hand_keypoints(
    keypoints: Tensor,
    heatmaps: Tensor,
    offsets: Optional[Tensor] = None,
) -> Tensor:
    """
    Refine hand keypoints using heatmaps and optional offsets.

    Args:
        keypoints: Initial keypoint predictions (N, 2)
        heatmaps: Heatmaps of shape (N, H, W)
        offsets: Optional offset predictions (N, 2, H, W)

    Returns:
        Refined keypoints
    """
    B, H, W = heatmaps.shape[1], heatmaps.shape[2], heatmaps.shape[3]

    for i in range(keypoints.shape[0]):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])

        if 0 <= x < W and 0 <= y < H:
            heatmap = heatmaps[i]

            if offsets is not None:
                offset = offsets[i, :, y, x]
                keypoints[i, 0] += offset[0]
                keypoints[i, 1] += offset[1]

    return keypoints


__all__ = [
    "HAND_21_KEYPOINTS",
    "HAND_SKELETON",
    "FINGER_INDICES",
    "HandPose",
    "hand_keypoints_to_tensor",
    "tensor_to_hand_keypoints",
    "compute_hand_visibility",
    "compute_finger_bending",
    "compute_all_finger_bending",
    "compute_palm_alignment",
    "hand_crop_and_resize",
    "inverse_hand_transform",
    "compute_hand_pck",
    "compute_hand_auc",
    "estimate_hand_side",
    "refine_hand_keypoints",
]
