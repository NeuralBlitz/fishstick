"""
Detection Metrics

Comprehensive metrics for evaluating object detection and segmentation models.

Classes:
- DetectionMetrics: Container for detection metrics

Functions:
- compute_iou: Intersection over Union
- compute_ap: Average Precision
- compute_map: mean Average Precision
- compute_confusion_matrix: Detection confusion matrix
- compute_precision_recall: Precision-Recall values
- compute_f1_curve: F1 score at different thresholds
- compute_roc_curve: ROC curve data
- compute_auc: Area Under Curve
"""

from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass

import torch
from torch import Tensor
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc as sklearn_auc,
)


@dataclass
class DetectionMetrics:
    """Container for detection metrics."""

    map50: float
    map75: float
    map: float
    precision: float
    recall: float
    f1: float


def compute_iou(
    box1: Union[Tensor, np.ndarray],
    box2: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    if isinstance(box1, Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, Tensor):
        box2 = box2.cpu().numpy()

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_ious(
    boxes1: Union[Tensor, np.ndarray],
    boxes2: Union[Tensor, np.ndarray],
) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: First set of boxes [N, 4]
        boxes2: Second set of boxes [M, 4]

    Returns:
        IoU matrix [N, M]
    """
    if isinstance(boxes1, Tensor):
        boxes1 = boxes1.cpu().numpy()
    if isinstance(boxes2, Tensor):
        boxes2 = boxes2.cpu().numpy()

    n = boxes1.shape[0]
    m = boxes2.shape[0]

    ious = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            ious[i, j] = compute_iou(boxes1[i], boxes2[j])

    return ious


def compute_ap(
    confidences: Union[Tensor, np.ndarray],
    labels: Union[Tensor, np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute Average Precision (AP) for a single class.

    Args:
        confidences: Prediction confidences
        labels: Ground truth labels (binary)
        iou_threshold: IoU threshold for matching

    Returns:
        Average Precision
    """
    if isinstance(confidences, Tensor):
        confidences = confidences.cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()

    sorted_indices = np.argsort(confidences)[::-1]
    sorted_labels = labels[sorted_indices]

    tp = sorted_labels == 1
    fp = sorted_labels == 0

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / (tp_cumsum[-1] + fp_cumsum[-1] + 1e-10)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[1], precisions, [0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0]

    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap


def compute_map(
    all_confidences: List[Union[Tensor, np.ndarray]],
    all_labels: List[Union[Tensor, np.ndarray]],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP).

    Args:
        all_confidences: List of confidence arrays per class
        all_labels: List of label arrays per class
        iou_thresholds: List of IoU thresholds

    Returns:
        Dictionary of mAP metrics
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    results = {}

    if len(all_confidences) > 0:
        aps_per_iou = []
        for iou_thresh in iou_thresholds:
            aps = []
            for conf, label in zip(all_confidences, all_labels):
                ap = compute_ap(conf, label, iou_threshold=iou_thresh)
                aps.append(ap)
            map_iou = np.mean(aps)
            aps_per_iou.append(map_iou)
            results[f"map@{int(iou_thresh * 100)}"] = map_iou

        results["map"] = np.mean(aps_per_iou)
        results["map50"] = aps_per_iou[0] if len(aps_per_iou) > 0 else 0.0
        results["map75"] = aps_per_iou[5] if len(aps_per_iou) > 5 else 0.0
    else:
        results["map"] = 0.0
        results["map50"] = 0.0
        results["map75"] = 0.0

    return results


def compute_precision_recall_curve(
    y_true: Union[Tensor, np.ndarray],
    y_score: Union[Tensor, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve.

    Args:
        y_true: True binary labels
        y_score: Prediction scores

    Returns:
        Tuple of (precision, recall, thresholds)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, Tensor):
        y_score = y_score.cpu().numpy()

    return precision_recall_curve(y_true, y_score)


def compute_roc_curve(
    y_true: Union[Tensor, np.ndarray],
    y_score: Union[Tensor, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.

    Args:
        y_true: True binary labels
        y_score: Prediction scores

    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, Tensor):
        y_score = y_score.cpu().numpy()

    return roc_curve(y_true, y_score)


def compute_auc(
    x: Union[Tensor, np.ndarray],
    y: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Area Under Curve (AUC).

    Args:
        x: X values
        y: Y values

    Returns:
        AUC score
    """
    if isinstance(x, Tensor):
        x = x.cpu().numpy()
    if isinstance(y, Tensor):
        y = y.cpu().numpy()

    return sklearn_auc(x, y)


def compute_f1_curve(
    y_true: Union[Tensor, np.ndarray],
    y_score: Union[Tensor, np.ndarray],
    num_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute F1 score at different thresholds.

    Args:
        y_true: True binary labels
        y_score: Prediction scores
        num_thresholds: Number of thresholds to evaluate

    Returns:
        Tuple of (f1_scores, thresholds, best_threshold)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, Tensor):
        y_score = y_score.cpu().numpy()

    thresholds = np.linspace(0, 1, num_thresholds)
    f1_scores = []

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return f1_scores, thresholds, best_threshold


def compute_confusion_matrix_detection(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> Dict[str, int]:
    """
    Compute confusion matrix for detection.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels

    Returns:
        Dictionary with TP, FP, FN, TN counts
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def compute_detection_metrics(
    y_true: Union[Tensor, np.ndarray],
    y_score: Union[Tensor, np.ndarray],
    threshold: float = 0.5,
) -> DetectionMetrics:
    """
    Compute all detection metrics.

    Args:
        y_true: True binary labels
        y_score: Prediction scores
        threshold: Classification threshold

    Returns:
        DetectionMetrics dataclass
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, Tensor):
        y_score = y_score.cpu().numpy()

    y_pred = (y_score >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision_val = tp / (tp + fp + 1e-10)
    recall_val = tp / (tp + fn + 1e-10)
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-10)

    map_results = compute_map([y_score], [y_true])

    return DetectionMetrics(
        map50=map_results.get("map50", 0.0),
        map75=map_results.get("map75", 0.0),
        map=map_results.get("map", 0.0),
        precision=precision_val,
        recall=recall_val,
        f1=f1_val,
    )


def compute_optimal_threshold(
    y_true: Union[Tensor, np.ndarray],
    y_score: Union[Tensor, np.ndarray],
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Compute optimal classification threshold.

    Args:
        y_true: True binary labels
        y_score: Prediction scores
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, Tensor):
        y_score = y_score.cpu().numpy()

    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.5
    best_score = 0.0

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "precision":
            score = tp / (tp + fp + 1e-10)
        elif metric == "recall":
            score = tp / (tp + fn + 1e-10)
        else:
            precision_val = tp / (tp + fp + 1e-10)
            recall_val = tp / (tp + fn + 1e-10)
            score = (
                2 * precision_val * recall_val / (precision_val + recall_val + 1e-10)
            )

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def compute_iou_from_masks(
    mask1: Union[Tensor, np.ndarray],
    mask2: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute IoU between two segmentation masks.

    Args:
        mask1: First mask
        mask2: Second mask

    Returns:
        IoU score
    """
    if isinstance(mask1, Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, Tensor):
        mask2 = mask2.cpu().numpy()

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


class DetectionMetricTracker:
    """Track detection metrics over batches."""

    def __init__(self):
        self.y_true: List[np.ndarray] = []
        self.y_score: List[np.ndarray] = []

    def update(
        self,
        y_true: Union[Tensor, np.ndarray],
        y_score: Union[Tensor, np.ndarray],
    ):
        """Update tracker with new batch."""
        if isinstance(y_true, Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_score, Tensor):
            y_score = y_score.cpu().numpy()

        self.y_true.append(y_true)
        self.y_score.append(y_score)

    def compute(self) -> Dict[str, float]:
        """Compute all detection metrics."""
        y_true_all = np.concatenate(self.y_true)
        y_score_all = np.concatenate(self.y_score)

        results = compute_map([y_score_all], [y_true_all])
        metrics = compute_detection_metrics(y_true_all, y_score_all)

        results["precision"] = metrics.precision
        results["recall"] = metrics.recall
        results["f1"] = metrics.f1

        return results

    def reset(self):
        """Reset tracker."""
        self.y_true = []
        self.y_score = []


__all__ = [
    "DetectionMetrics",
    "compute_iou",
    "compute_ious",
    "compute_ap",
    "compute_map",
    "compute_precision_recall_curve",
    "compute_roc_curve",
    "compute_auc",
    "compute_f1_curve",
    "compute_confusion_matrix_detection",
    "compute_detection_metrics",
    "compute_optimal_threshold",
    "compute_iou_from_masks",
    "DetectionMetricTracker",
]
