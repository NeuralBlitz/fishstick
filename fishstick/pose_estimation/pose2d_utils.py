"""
Utility Functions for 2D Pose Estimation

Contains helper functions for pose evaluation, heatmap decoding, and transformations.
"""

from typing import List, Tuple, Optional, Dict, Any

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np


def compute_pck(
    pred_keypoints: Tensor,
    gt_keypoints: Tensor,
    gt_visibility: Tensor,
    thresholds: Optional[List[float]] = None,
    normalize_factor: Optional[Tensor] = None,
) -> Dict[str, float]:
    """
    Compute Percentage of Correct Keypoints (PCK).

    Args:
        pred_keypoints: Predicted keypoints (N, K, 2)
        gt_keypoints: Ground truth keypoints (N, K, 2)
        gt_visibility: Visibility flags (N, K)
        thresholds: List of distance thresholds (default: [0.05, 0.1, 0.15, 0.2, 0.25])
        normalize_factor: Normalization factor per sample (N,)

    Returns:
        Dictionary with PCK at different thresholds
    """
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]

    N, K, _ = pred_keypoints.shape
    results = {}

    dist = torch.norm(pred_keypoints - gt_keypoints, dim=-1)

    if normalize_factor is not None:
        normalize_factor = normalize_factor.view(N, 1)
        dist = dist / (normalize_factor + 1e-8)

    valid_mask = gt_visibility > 0

    for threshold in thresholds:
        correct = (dist <= threshold) & valid_mask
        pck = correct.float().sum() / (valid_mask.float().sum() + 1e-8)
        results[f"pck_{threshold}"] = pck.item()

    results["mean_dist"] = (dist * valid_mask.float()).sum() / (
        valid_mask.float().sum() + 1e-8
    )

    return results


def compute_auc(
    pred_keypoints: Tensor,
    gt_keypoints: Tensor,
    gt_visibility: Tensor,
    max_threshold: float = 0.25,
    num_points: int = 101,
    normalize_factor: Optional[Tensor] = None,
) -> float:
    """
    Compute Area Under the Curve (AUC) for PCK.

    Args:
        pred_keypoints: Predicted keypoints (N, K, 2)
        gt_keypoints: Ground truth keypoints (N, K, 2)
        gt_visibility: Visibility flags (N, K)
        max_threshold: Maximum threshold for AUC calculation
        num_points: Number of points for AUC integration
        normalize_factor: Normalization factor per sample

    Returns:
        AUC value
    """
    thresholds = torch.linspace(0, max_threshold, num_points)

    dist = torch.norm(pred_keypoints - gt_keypoints, dim=-1)

    if normalize_factor is not None:
        normalize_factor = normalize_factor.view(-1, 1)
        dist = dist / (normalize_factor + 1e-8)

    valid_mask = gt_visibility > 0

    pcks = []
    for threshold in thresholds:
        correct = (dist <= threshold) & valid_mask
        pck = correct.float().sum() / (valid_mask.float().sum() + 1e-8)
        pcks.append(pck.item())

    pcks = torch.tensor(pcks)
    auc = torch.trapz(pcks, thresholds) / max_threshold

    return auc.item()


def computeoks(
    pred_keypoints: Tensor,
    gt_keypoints: Tensor,
    gt_visibility: Tensor,
    areas: Tensor,
    sigmas: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute Object Keypoint Similarity (OKS).

    Args:
        pred_keypoints: Predicted keypoints (N, K, 2)
        gt_keypoints: Ground truth keypoints (N, K, 2)
        gt_visibility: Visibility flags (N, K)
        areas: Person areas (N,)
        sigmas: Per-keypoint standard deviations (K,)

    Returns:
        OKS values (N,)
    """
    N, K, _ = pred_keypoints.shape

    if sigmas is None:
        sigmas = torch.ones(K, device=pred_keypoints.device) * 0.1

    dist = torch.norm(pred_keypoints - gt_keypoints, dim=-1)

    k = (sigmas * 2) ** 2
    area = areas.view(N, 1)
    oks = torch.exp(-(dist**2) / (2 * k * area + 1e-8))

    valid = gt_visibility > 0
    oks_per_sample = (oks * valid.float()).sum(dim=1) / (
        valid.float().sum(dim=1) + 1e-8
    )

    return oks_per_sample


def decode_heatmap(heatmap: Tensor, softmax_temp: float = 1.0) -> Tuple[Tensor, Tensor]:
    """
    Decode keypoint locations from heatmaps.

    Args:
        heatmap: Heatmap tensor (B, K, H, W)
        softmax_temp: Temperature for softmax

    Returns:
        Tuple of (keypoints (B, K, 2), scores (B, K))
    """
    B, K, H, W = heatmap.shape

    heatmap = heatmap.view(B * K, H, W)
    heatmap = F.softmax(heatmap.view(B * K, -1) / softmax_temp, dim=1)
    heatmap = heatmap.view(B, K, H, W)

    max_vals, max_idx = torch.max(heatmap.view(B, K, -1), dim=2)
    max_idx = max_idx.unsqueeze(-1).expand(B, K, 2)

    grid_x = torch.arange(W, device=heatmap.device).float().view(1, 1, W)
    grid_y = torch.arange(H, device=heatmap.device).float().view(1, H, 1)

    coords = torch.cat(
        [
            grid_x.unsqueeze(0).unsqueeze(0).expand(B, K, H, W),
            grid_y.unsqueeze(0).unsqueeze(0).expand(B, K, H, W),
        ],
        dim=1,
    )

    preds = coords.gather(3, max_idx).squeeze(3)

    max_vals = max_vals.clamp(min=1e-8)
    scores = torch.log(max_vals + 1e-8)

    return preds, scores


def decode_heatmap_multi(
    heatmap: Tensor, max_per_key: int = 10, score_threshold: float = 0.1
) -> Tuple[Tensor, Tensor]:
    """
    Decode multiple keypoint locations from heatmaps.

    Args:
        heatmap: Heatmap tensor (B, K, H, W)
        max_per_key: Maximum detections per keypoint
        score_threshold: Minimum score threshold

    Returns:
        Tuple of (keypoints (B, K, M, 2), scores (B, K, M))
    """
    B, K, H, W = heatmap.shape

    heatmap_flat = heatmap.view(B * K, -1)
    scores, indices = torch.topk(heatmap_flat, min(max_per_key, H * W), dim=1)

    scores = scores.view(B, K, -1)
    indices = indices.view(B, K, -1)

    x_indices = indices % W
    y_indices = indices // W

    keypoints = torch.stack([x_indices.float(), y_indices.float()], dim=-1)

    mask = scores > score_threshold

    return keypoints, scores


def get_max_preds(heatmaps: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Get predictions from heatmaps by finding maximum.

    Args:
        heatmaps: Heatmap tensor (B, K, H, W)

    Returns:
        Tuple of (predictions (B, K, 2), maxvals (B, K))
    """
    B, K, H, W = heatmaps.shape

    heatmaps = heatmaps.view(B, K, -1)
    maxvals, idx = torch.max(heatmaps, dim=2)

    preds = torch.zeros(B, K, 2, device=heatmaps.device, dtype=heatmaps.dtype)
    preds[:, :, 0] = idx % W
    preds[:, :, 1] = idx // W

    maxvals = maxvals.view(B, K, 1)

    return preds, maxvals


def transform_predictions(
    keypoints: Tensor,
    center: Tensor,
    scale: Tensor,
    output_size: Tuple[int, int],
    inv: bool = False,
) -> Tensor:
    """
    Transform keypoints between image and output coordinate spaces.

    Args:
        keypoints: Keypoints in output space (N, K, 2) or (K, 2)
        center: Center of the person in image space (N, 2) or (2,)
        scale: Scale of the person (N,) or scalar
        output_size: Target output size (h, w)
        inv: If True, transform from output to image space

    Returns:
        Transformed keypoints
    """
    if keypoints.dim() == 2:
        keypoints = keypoints.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    N, K, _ = keypoints.shape
    h, w = output_size

    if not inv:
        scale = scale.view(N, 1, 1) if scale.dim() == 1 else scale.view(-1, 1, 1)
        center = center.view(N, 1, 2) if center.dim() == 2 else center.view(-1, 1, 2)

        trans = get_transformation_matrix(center, scale, output_size)

        keypoints_flat = keypoints.view(N, K, 2, 1)
        ones = torch.ones(N, K, 1, 1, device=keypoints.device)
        keypoints_homo = torch.cat([keypoints_flat, ones], dim=2)

        transformed = torch.bmm(trans, keypoints_homo)
        result = transformed[:, :, :2, 0]
    else:
        scale = scale.view(N, 1, 1) if scale.dim() == 1 else scale.view(-1, 1, 1)
        center = center.view(N, 1, 2) if center.dim() == 2 else center.view(-1, 1, 2)

        trans = get_transformation_matrix(center, scale, output_size, inv=True)

        keypoints_flat = keypoints.view(N, K, 2, 1)
        ones = torch.ones(N, K, 1, 1, device=keypoints.device)
        keypoints_homo = torch.cat([keypoints_flat, ones], dim=2)

        transformed = torch.bmm(trans, keypoints_homo)
        result = transformed[:, :, :2, 0]

    if squeeze:
        result = result.squeeze(0)

    return result


def get_transformation_matrix(
    center: Tensor, scale: Tensor, output_size: Tuple[int, int], inv: bool = False
) -> Tensor:
    """
    Get affine transformation matrix.

    Args:
        center: Center point (N, 2)
        scale: Scale (N,)
        output_size: Target size (h, w)
        inv: If True, compute inverse transformation

    Returns:
        Transformation matrix (N, 3, 3)
    """
    N = center.shape[0]
    h, w = output_size
    scale = scale.view(N, 1) if scale.dim() == 1 else scale.view(-1, 1)

    trans = torch.zeros(N, 3, 3, device=center.device)
    trans[:, 0, 0] = w / scale.squeeze(-1)
    trans[:, 1, 1] = h / scale.squeeze(-1)
    trans[:, 0, 2] = w / 2 - center[:, 0] * w / scale.squeeze(-1)
    trans[:, 1, 2] = h / 2 - center[:, 1] * h / scale.squeeze(-1)
    trans[:, 2, 2] = 1

    if inv:
        trans = torch.inverse(trans)

    return trans


def project_to_image(keypoints_3d: Tensor, camera_matrix: Tensor) -> Tensor:
    """
    Project 3D keypoints to 2D image plane.

    Args:
        keypoints_3d: 3D keypoints (N, K, 3)
        camera_matrix: Camera intrinsic matrix (3, 3) or (N, 3, 3)

    Returns:
        2D keypoints (N, K, 2)
    """
    if keypoints_3d.dim() == 2:
        keypoints_3d = keypoints_3d.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    N, K, _ = keypoints_3d.shape

    if camera_matrix.dim() == 2:
        camera_matrix = camera_matrix.unsqueeze(0).expand(N, -1, -1)

    xy = keypoints_3d[:, :, :2]
    z = keypoints_3d[:, :, 2:3].clamp(min=1e-5)

    xy_normalized = xy / z

    keypoints_2d = torch.bmm(xy_normalized, camera_matrix.transpose(1, 2))

    if squeeze:
        keypoints_2d = keypoints_2d.squeeze(0)

    return keypoints_2d


def batch_nms(
    keypoints: Tensor, scores: Tensor, threshold: float = 0.3
) -> Tuple[Tensor, Tensor]:
    """
    Apply Non-Maximum Suppression to keypoints.

    Args:
        keypoints: Keypoints (B, K, 2)
        scores: Confidence scores (B, K)
        threshold: NMS threshold

    Returns:
        Tuple of filtered (keypoints, scores)
    """
    B, K, _ = keypoints.shape

    keep_indices = []

    for b in range(B):
        kp = keypoints[b]
        sc = scores[b]

        order = torch.argsort(sc, descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order[0].item())
                break

            i = order[0]
            keep.append(i.item())

            remaining = order[1:]
            dist = torch.norm(kp[remaining] - kp[i], dim=-1)
            mask = dist > threshold
            order = remaining[mask]

        keep_indices.append(keep)

    max_keep = max(len(indices) for indices in keep_indices)

    result_keypoints = torch.zeros(B, max_keep, 2, device=keypoints.device)
    result_scores = torch.zeros(B, max_keep, device=keypoints.device)

    for b in range(B):
        indices = keep_indices[b]
        result_keypoints[b, : len(indices)] = keypoints[b, indices]
        result_scores[b, : len(indices)] = scores[b, indices]

    return result_keypoints, result_scores
