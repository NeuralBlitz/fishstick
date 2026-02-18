"""
Keypoint Detection Utilities

Utility functions for keypoint detection including heatmap decoding,
NMS, keypoint grouping, and confidence-based filtering.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np


@dataclass
class KeypointPrediction:
    """
    Represents a single keypoint prediction.

    Attributes:
        x: X coordinate
        y: Y coordinate
        confidence: Detection confidence score
        class_id: Keypoint class ID
    """

    x: float
    y: float
    confidence: float
    class_id: int = 0


def decode_keypoints_from_heatmap(
    heatmap: Tensor,
    offsetmap: Optional[Tensor] = None,
    conf_threshold: float = 0.1,
    num_keypoints: int = 17,
) -> Tuple[Tensor, Tensor]:
    """
    Decode keypoints from heatmap predictions.

    Args:
        heatmap: Heatmap of shape (B, K, H, W)
        offsetmap: Optional offset predictions of shape (B, K, 2, H, W)
        conf_threshold: Confidence threshold for keypoint detection
        num_keypoints: Number of keypoint classes

    Returns:
        Tuple of (keypoints [B, K, 2], scores [B, K])
    """
    B, K, H, W = heatmap.shape

    heatmap_flat = heatmap.view(B, K, -1)
    scores, indices = heatmap_flat.max(dim=2)

    scores = torch.sigmoid(scores)

    keypoints = torch.zeros(B, K, 2, device=heatmap.device, dtype=heatmap.dtype)

    for b in range(B):
        for k in range(K):
            idx = indices[b, k]
            y = idx // W
            x = idx % W

            keypoints[b, k, 0] = x.float()
            keypoints[b, k, 1] = y.float()

    if offsetmap is not None:
        offsetmap = offsetmap.view(B, K, 2, -1)
        for b in range(B):
            for k in range(K):
                idx = indices[b, k]
                offset = offsetmap[b, k, :, idx]
                keypoints[b, k] += offset

    mask = scores > conf_threshold

    return keypoints, scores


def get_keypoint_predictions(
    heatmap: Tensor,
    offsetmap: Optional[Tensor] = None,
    conf_threshold: float = 0.1,
    max_detections: int = 100,
) -> List[List[KeypointPrediction]]:
    """
    Get keypoint predictions from heatmap.

    Args:
        heatmap: Heatmap of shape (B, K, H, W)
        offsetmap: Optional offset predictions
        conf_threshold: Confidence threshold
        max_detections: Maximum detections per image

    Returns:
        List of predictions per image
    """
    keypoints, scores = decode_keypoints_from_heatmap(
        heatmap, offsetmap, conf_threshold
    )

    B, K, _ = keypoints.shape
    predictions = []

    for b in range(B):
        img_preds = []

        for k in range(K):
            if scores[b, k] > conf_threshold:
                pred = KeypointPrediction(
                    x=keypoints[b, k, 0].item(),
                    y=keypoints[b, k, 1].item(),
                    confidence=scores[b, k].item(),
                    class_id=k,
                )
                img_preds.append(pred)

        img_preds.sort(key=lambda x: x.confidence, reverse=True)
        predictions.append(img_preds[:max_detections])

    return predictions


def compute_keypoint_iou(
    pred_keypoints: Tensor,
    gt_keypoints: Tensor,
    threshold: float = 0.5,
) -> Tensor:
    """
    Compute IoU between predicted and ground truth keypoints.

    Args:
        pred_keypoints: Predicted keypoints (N, 2)
        gt_keypoints: Ground truth keypoints (M, 2)
        threshold: Distance threshold for matching

    Returns:
        IoU matrix of shape (N, M)
    """
    N, M = pred_keypoints.shape[0], gt_keypoints.shape[0]
    iou_matrix = torch.zeros(N, M, device=pred_keypoints.device)

    for i in range(N):
        for j in range(M):
            dist = torch.norm(pred_keypoints[i] - gt_keypoints[j])
            if dist < threshold:
                iou_matrix[i, j] = 1.0 - (dist / threshold)

    return iou_matrix


def nms_keypoints(
    keypoints: Tensor,
    scores: Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.3,
) -> Tuple[Tensor, Tensor]:
    """
    Non-maximum suppression for keypoints.

    Args:
        keypoints: Keypoints tensor of shape (N, 2)
        scores: Confidence scores of shape (N,)
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum score to keep

    Returns:
        Tuple of (filtered keypoints, filtered scores)
    """
    if keypoints.shape[0] == 0:
        return keypoints, scores

    mask = scores > score_threshold
    keypoints = keypoints[mask]
    scores = scores[mask]

    if keypoints.shape[0] == 0:
        return keypoints, scores

    order = torch.argsort(scores, descending=True)
    keypoints = keypoints[order]
    scores = scores[order]

    keep = []

    while keypoints.shape[0] > 0:
        keep.append(0)

        if keypoints.shape[0] == 1:
            break

        current = keypoints[0:1]
        rest = keypoints[1:]

        dists = torch.norm(rest - current, dim=1)
        mask = dists > iou_threshold

        keypoints = rest[mask]
        scores = scores[1:][mask]

    keep_indices = torch.tensor(keep, dtype=torch.long, device=keypoints.device)

    return keypoints[keep_indices], scores[keep_indices]


def group_keypoints(
    detections: List[Tensor],
    scores: List[Tensor],
    distance_threshold: float = 50.0,
    min_group_size: int = 3,
) -> List[Tensor]:
    """
    Group individual keypoint detections into pose instances.

    Args:
        detections: List of keypoint tensors, each (N_k, 2)
        scores: List of score tensors, each (N_k,)
        distance_threshold: Maximum distance between keypoints
        min_group_size: Minimum detections for a valid group

    Returns:
        List of grouped keypoint sets
    """
    all_keypoints = []
    all_scores = []
    all_classes = []

    for class_id, (kp, sc) in enumerate(zip(detections, scores)):
        all_keypoints.append(kp)
        all_scores.append(sc)
        all_classes.append(torch.full((kp.shape[0],), class_id, dtype=torch.long))

    all_keypoints = torch.cat(all_keypoints, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_classes = torch.cat(all_classes, dim=0)

    if all_keypoints.shape[0] == 0:
        return []

    sorted_idx = torch.argsort(all_scores, descending=True)
    all_keypoints = all_keypoints[sorted_idx]
    all_scores = all_scores[sorted_idx]
    all_classes = all_classes[sorted_idx]

    groups = []
    used = torch.zeros(all_keypoints.shape[0], dtype=torch.bool)

    for i in range(all_keypoints.shape[0]):
        if used[i]:
            continue

        group_indices = [i]
        used[i] = True

        class_i = all_classes[i]

        for j in range(i + 1, all_keypoints.shape[0]):
            if used[j]:
                continue

            if all_classes[j] != class_i:
                continue

            dist = torch.norm(all_keypoints[i] - all_keypoints[j])
            if dist < distance_threshold:
                group_indices.append(j)
                used[j] = True

        if len(group_indices) >= min_group_size:
            groups.append(all_keypoints[group_indices])

    return groups


def match_keypoints(
    pred_keypoints: Tensor,
    gt_keypoints: Tensor,
    pred_scores: Optional[Tensor] = None,
    gt_visibility: Optional[Tensor] = None,
    matching_threshold: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Match predicted keypoints to ground truth keypoints.

    Args:
        pred_keypoints: Predicted keypoints (N, 2)
        gt_keypoints: Ground truth keypoints (M, 2)
        pred_scores: Optional prediction scores (N,)
        gt_visibility: Optional ground truth visibility (M,)
        matching_threshold: Distance threshold for matching

    Returns:
        Tuple of (matched_pred_indices, matched_gt_indices, distances)
    """
    N, M = pred_keypoints.shape[0], gt_keypoints.shape[0]

    if N == 0 or M == 0:
        empty_idx = torch.tensor([], dtype=torch.long)
        return empty_idx, empty_idx, torch.tensor([], device=pred_keypoints.device)

    dist_matrix = torch.cdist(pred_keypoints, gt_keypoints)

    matched_pred = []
    matched_gt = []
    distances = []

    for gt_idx in range(M):
        if gt_visibility is not None and gt_visibility[gt_idx] == 0:
            continue

        min_dist, pred_idx = dist_matrix[:, gt_idx].min(dim=0)

        if min_dist < matching_threshold:
            matched_pred.append(pred_idx.item())
            matched_gt.append(gt_idx)
            distances.append(min_dist.item())

    if len(matched_pred) == 0:
        empty_idx = torch.tensor([], dtype=torch.long)
        return empty_idx, empty_idx, torch.tensor([], device=pred_keypoints.device)

    matched_pred = torch.tensor(matched_pred, dtype=torch.long)
    matched_gt = torch.tensor(matched_gt, dtype=torch.long)
    distances = torch.tensor(distances, device=pred_keypoints.device)

    return matched_pred, matched_gt, distances


def transform_keypoints(
    keypoints: Tensor,
    transform: Dict[str, Any],
    inverse: bool = False,
) -> Tensor:
    """
    Apply transformation to keypoints.

    Args:
        keypoints: Keypoints tensor of shape (N, 2) or (N, 3)
        transform: Transform dictionary with keys: center, scale, rotation, flip
        inverse: Whether to apply inverse transformation

    Returns:
        Transformed keypoints
    """
    transformed = keypoints.clone()

    if inverse:
        if "scale" in transform:
            scale = transform["scale"]
            transformed = transformed / scale

        if "rotation" in transform:
            angle = -transform["rotation"]
        else:
            angle = 0

        if "center" in transform:
            transformed = transformed - transform["center"]

        if angle != 0:
            cos_a = torch.cos(torch.tensor(angle))
            sin_a = torch.sin(torch.tensor(angle))
            rot = torch.tensor(
                [[cos_a, -sin_a], [sin_a, cos_a]], device=keypoints.device
            )
            transformed = transformed @ rot.T
    else:
        if angle != 0:
            cos_a = torch.cos(torch.tensor(angle))
            sin_a = torch.sin(torch.tensor(angle))
            rot = torch.tensor(
                [[cos_a, -sin_a], [sin_a, cos_a]], device=keypoints.device
            )
            transformed = transformed @ rot.T

        if "center" in transform:
            transformed = transformed + transform["center"]

        if "scale" in transform:
            scale = transform["scale"]
            transformed = transformed * scale

    return transformed


def refine_keypoints_with_gaussian(
    heatmap: Tensor,
    keypoints: Tensor,
    sigma: float = 2.0,
) -> Tensor:
    """
    Refine keypoint positions using Gaussian fitting.

    Args:
        heatmap: Heatmap tensor (K, H, W)
        keypoints: Initial keypoint positions (N, 2)
        sigma: Gaussian sigma for fitting

    Returns:
        Refined keypoint positions
    """
    K, H, W = heatmap.shape
    refined = keypoints.clone()

    for i in range(keypoints.shape[0]):
        x = int(keypoints[i, 0].item())
        y = int(keypoints[i, 1].item())

        x_start = max(0, x - 3)
        x_end = min(W, x + 4)
        y_start = max(0, y - 3)
        y_end = min(H, y + 4)

        patch = heatmap[0, y_start:y_end, x_start:x_end]

        if patch.numel() < 4:
            continue

        y_grid, x_grid = torch.meshgrid(
            torch.arange(y_start, y_end, device=heatmap.device),
            torch.arange(x_start, x_end, device=heatmap.device),
            indexing="ij",
        )

        patch_sum = patch.sum()
        if patch_sum > 1e-8:
            x_mean = (x_grid.float() * patch).sum() / patch_sum
            y_mean = (y_grid.float() * patch).sum() / patch_sum

            refined[i, 0] = x_mean
            refined[i, 1] = y_mean

    return refined


def compute_keypoint_precision_recall(
    pred_keypoints: List[Tensor],
    gt_keypoints: List[Tensor],
    thresholds: List[float] = [0.5, 0.75, 0.9],
    gt_visibility: Optional[List[Tensor]] = None,
) -> Dict[str, Any]:
    """
    Compute precision and recall for keypoint detection.

    Args:
        pred_keypoints: List of predicted keypoint sets
        gt_keypoints: List of ground truth keypoint sets
        thresholds: List of distance thresholds
        gt_visibility: Optional list of ground truth visibility

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    results = {
        "precision": {},
        "recall": {},
        "f1": {},
    }

    for thresh in thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred, gt in zip(pred_keypoints, gt_keypoints):
            vis = gt_visibility[0] if gt_visibility else None
            matched_pred, matched_gt, _ = match_keypoints(
                pred, gt, matching_threshold=thresh, gt_visibility=vis
            )

            true_positives += len(matched_pred)
            false_positives += pred.shape[0] - len(matched_pred)
            false_negatives += gt.shape[0] - len(matched_gt)

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results["precision"][thresh] = precision
        results["recall"][thresh] = recall
        results["f1"][thresh] = f1

    return results


def keypoints_to_heatmap(
    keypoints: Tensor,
    heatmap_size: Tuple[int, int],
    sigma: float = 2.0,
) -> Tensor:
    """
    Generate Gaussian heatmap from keypoints.

    Args:
        keypoints: Keypoint coordinates (N, 2)
        heatmap_size: Size of output heatmap (H, W)
        sigma: Gaussian sigma

    Returns:
        Heatmap of shape (N, H, W)
    """
    N = keypoints.shape[0]
    H, W = heatmap_size

    heatmaps = torch.zeros(N, H, W, device=keypoints.device)

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=keypoints.device),
        torch.arange(W, device=keypoints.device),
        indexing="ij",
    )

    for i in range(N):
        x, y = keypoints[i, 0], keypoints[i, 1]

        dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
        heatmaps[i] = torch.exp(-dist_sq / (2 * sigma**2))

    return heatmaps


__all__ = [
    "KeypointPrediction",
    "decode_keypoints_from_heatmap",
    "get_keypoint_predictions",
    "compute_keypoint_iou",
    "nms_keypoints",
    "group_keypoints",
    "match_keypoints",
    "transform_keypoints",
    "refine_keypoints_with_gaussian",
    "compute_keypoint_precision_recall",
    "keypoints_to_heatmap",
]
