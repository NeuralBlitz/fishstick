"""
3D Metrics Module

Evaluation metrics for 3D vision tasks.
"""

from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F


def chamfer_distance(
    points1: Tensor,
    points2: Tensor,
    bidirectional: bool = True,
) -> Tensor:
    """
    Compute Chamfer distance between two point clouds.

    Args:
        points1: First point cloud [N, 3]
        points2: Second point cloud [M, 3]
        bidirectional: Whether to compute both directions

    Returns:
        Chamfer distance
    """
    dist_matrix = torch.cdist(points1, points2)

    dist_1_to_2 = dist_matrix.min(dim=1)[0].mean()
    dist_2_to_1 = dist_matrix.min(dim=0)[0].mean()

    if bidirectional:
        return dist_1_to_2 + dist_2_to_1

    return dist_1_to_2


def earth_mover_distance(
    points1: Tensor,
    points2: Tensor,
) -> Tensor:
    """
    Approximate Earth Mover's Distance using Sinkhorn iteration.

    Args:
        points1: First point cloud [N, 3]
        points2: Second point cloud [M, 3]

    Returns:
        EMD value
    """
    dist_matrix = torch.cdist(points1, points2)

    N, M = dist_matrix.shape

    cost = dist_matrix / dist_matrix.max()

    a = torch.ones(N, device=cost.device) / N
    b = torch.ones(M, device=cost.device) / M

    for _ in range(10):
        u = a / (cost @ b + 1e-8)
        v = b / (cost.T @ u + 1e-8)

    transport = u.unsqueeze(-1) * v.unsqueeze(-2) * cost

    emd = (transport * cost).sum()

    return emd


def f1_score_3d(
    points1: Tensor,
    points2: Tensor,
    threshold: float = 0.01,
) -> Tensor:
    """
    Compute F1 score for point cloud matching.

    Args:
        points1: First point cloud [N, 3]
        points2: Second point cloud [M, 3]
        threshold: Distance threshold

    Returns:
        F1 score
    """
    dist_matrix = torch.cdist(points1, points2)

    matches1 = (dist_matrix.min(dim=1)[0] < threshold).sum()
    matches2 = (dist_matrix.min(dim=0)[0] < threshold).sum()

    precision = matches1 / points1.shape[0]
    recall = matches2 / points2.shape[0]

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1


def iou_3d_metric(
    boxes1: Tensor,
    boxes2: Tensor,
) -> Tensor:
    """
    Compute 3D IoU between bounding boxes.

    Args:
        boxes1: [N, 7] (x, y, z, l, w, h, yaw)
        boxes2: [M, 7]

    Returns:
        IoU matrix [N, M]
    """
    from fishstick.vision_3d.detection_3d import iou_3d as compute_iou

    return compute_iou(boxes1, boxes2)


def precision_recall_3d(
    pred_points: Tensor,
    gt_points: Tensor,
    threshold: float = 0.01,
) -> dict:
    """
    Compute precision and recall for point cloud.

    Args:
        pred_points: Predicted point cloud
        gt_points: Ground truth point cloud
        threshold: Distance threshold

    Returns:
        dict with precision, recall, f1
    """
    dist_matrix = torch.cdist(pred_points, gt_points)

    tp = (dist_matrix.min(dim=1)[0] < threshold).sum()
    fp = (dist_matrix.min(dim=1)[0] >= threshold).sum()
    fn = (dist_matrix.min(dim=0)[0] >= threshold).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
