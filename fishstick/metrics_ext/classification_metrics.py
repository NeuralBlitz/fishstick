"""
Classification Metrics

Comprehensive metrics for evaluating classification models.

Classes:
- ClassificationMetrics: Container for all classification metrics
- F1Score: F1 score computation (macro, micro, weighted)
- BalancedAccuracy: Balanced accuracy metric
- CohensKappa: Cohen's kappa score
- MatthewsCorrelation: MCC metric
- SpecificitySensitivity: Sensitivity and specificity per class

Functions:
- compute_classification_metrics: Compute all metrics at once
- confusion_matrix: Compute confusion matrix
- per_class_metrics: Get metrics per class
"""

from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass

import torch
from torch import Tensor
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix as sk_confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    cohen_kappa: float
    mcc: float
    macro_precision: float
    macro_recall: float
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray


def confusion_matrix(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization option ('true', 'pred', 'all', None)

    Returns:
        Confusion matrix
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return sk_confusion_matrix(y_true, y_pred, normalize=normalize)


def accuracy(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return accuracy_score(y_true, y_pred)


def balanced_accuracy(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute balanced accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Balanced accuracy score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return balanced_accuracy_score(y_true, y_pred)


def f1_score_metric(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    average: str = "macro",
    zero_division: int = 0,
) -> float:
    """
    Compute F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', 'samples')
        zero_division: Zero division handling

    Returns:
        F1 score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return f1_score(y_true, y_pred, average=average, zero_division=zero_division)


def macro_f1(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    zero_division: int = 0,
) -> float:
    """
    Compute macro-averaged F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        zero_division: Zero division handling

    Returns:
        Macro F1 score
    """
    return f1_score_metric(y_true, y_pred, average="macro", zero_division=zero_division)


def micro_f1(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    zero_division: int = 0,
) -> float:
    """
    Compute micro-averaged F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        zero_division: Zero division handling

    Returns:
        Micro F1 score
    """
    return f1_score_metric(y_true, y_pred, average="micro", zero_division=zero_division)


def weighted_f1(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    zero_division: int = 0,
) -> float:
    """
    Compute weighted F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        zero_division: Zero division handling

    Returns:
        Weighted F1 score
    """
    return f1_score_metric(
        y_true, y_pred, average="weighted", zero_division=zero_division
    )


def cohen_kappa(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    weights: Optional[str] = None,
) -> float:
    """
    Compute Cohen's kappa score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        weights: Weighting type (None, 'linear', 'quadratic')

    Returns:
        Cohen's kappa score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return cohen_kappa_score(y_true, y_pred, weights=weights)


def matthews_corrcoef_metric(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Matthews correlation coefficient.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        MCC score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return matthews_corrcoef(y_true, y_pred)


def sensitivity_specificity(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    pos_label: int = 1,
) -> Tuple[float, float]:
    """
    Compute sensitivity (recall) and specificity.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label

    Returns:
        Tuple of (sensitivity, specificity)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    cm = sk_confusion_matrix(y_true, y_pred)

    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = recall_score(y_true, y_pred, average="macro", zero_division=0)
        specificity = 0.0

    return sensitivity, specificity


def per_class_precision(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    labels: Optional[np.ndarray] = None,
    zero_division: int = 0,
) -> np.ndarray:
    """
    Compute precision per class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels to compute metrics for
        zero_division: Zero division handling

    Returns:
        Array of per-class precision scores
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=zero_division
    )


def per_class_recall(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    labels: Optional[np.ndarray] = None,
    zero_division: int = 0,
) -> np.ndarray:
    """
    Compute recall per class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels to compute metrics for
        zero_division: Zero division handling

    Returns:
        Array of per-class recall scores
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=zero_division
    )


def per_class_f1(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    labels: Optional[np.ndarray] = None,
    zero_division: int = 0,
) -> np.ndarray:
    """
    Compute F1 score per class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels to compute metrics for
        zero_division: Zero division handling

    Returns:
        Array of per-class F1 scores
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=zero_division
    )


def per_class_metrics(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    labels: Optional[np.ndarray] = None,
    zero_division: int = 0,
) -> Dict[int, Dict[str, float]]:
    """
    Compute all metrics per class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels to compute metrics for
        zero_division: Zero division handling

    Returns:
        Dictionary mapping class to metric dictionary
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    precision = per_class_precision(y_true, y_pred, labels, zero_division)
    recall = per_class_recall(y_true, y_pred, labels, zero_division)
    f1 = per_class_f1(y_true, y_pred, labels, zero_division)

    results = {}
    for i, label in enumerate(labels):
        results[int(label)] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }

    return results


def compute_classification_metrics(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    zero_division: int = 0,
) -> ClassificationMetrics:
    """
    Compute all classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        zero_division: Zero division handling

    Returns:
        ClassificationMetrics dataclass
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=zero_division),
        micro_f1=f1_score(y_true, y_pred, average="micro", zero_division=zero_division),
        weighted_f1=f1_score(
            y_true, y_pred, average="weighted", zero_division=zero_division
        ),
        cohen_kappa=cohen_kappa_score(y_true, y_pred),
        mcc=matthews_corrcoef(y_true, y_pred),
        macro_precision=precision_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        ),
        macro_recall=recall_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        ),
        per_class_precision=precision_score(
            y_true, y_pred, average=None, zero_division=zero_division
        ),
        per_class_recall=recall_score(
            y_true, y_pred, average=None, zero_division=zero_division
        ),
        per_class_f1=f1_score(
            y_true, y_pred, average=None, zero_division=zero_division
        ),
    )


def compute_roc_auc(
    y_true: Union[Tensor, np.ndarray],
    y_score: Union[Tensor, np.ndarray],
    average: str = "macro",
    multi_class: str = "ovr",
) -> float:
    """
    Compute ROC AUC score.

    Args:
        y_true: True labels
        y_score: Prediction scores (probabilities or logits)
        average: Averaging method
        multi_class: Multi-class strategy ('ovr', 'ovo')

    Returns:
        ROC AUC score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, Tensor):
        y_score = y_score.cpu().numpy()

    return roc_auc_score(y_true, y_score, average=average, multi_class=multi_class)


def precision_recall_curve_data(
    y_true: Union[Tensor, np.ndarray],
    y_score: Union[Tensor, np.ndarray],
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve.

    Args:
        y_true: True labels
        y_score: Prediction scores
        pos_label: Positive class label

    Returns:
        Tuple of (precision, recall, thresholds)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, Tensor):
        y_score = y_score.cpu().numpy()

    return precision_recall_curve(y_true, y_score, pos_label=pos_label)


def average_precision(
    y_true: Union[Tensor, np.ndarray],
    y_score: Union[Tensor, np.ndarray],
    pos_label: int = 1,
) -> float:
    """
    Compute average precision (AP).

    Args:
        y_true: True labels
        y_score: Prediction scores
        pos_label: Positive class label

    Returns:
        Average precision score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, Tensor):
        y_score = y_score.cpu().numpy()

    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    return auc(recall, precision)


class ClassificationMetricTracker:
    """Track classification metrics over batches."""

    def __init__(self):
        self.y_true: List[np.ndarray] = []
        self.y_pred: List[np.ndarray] = []

    def update(
        self,
        y_true: Union[Tensor, np.ndarray],
        y_pred: Union[Tensor, np.ndarray],
    ):
        """Update tracker with new batch."""
        if isinstance(y_true, Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, Tensor):
            y_pred = y_pred.cpu().numpy()

        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def compute(self) -> ClassificationMetrics:
        """Compute all metrics."""
        y_true_all = np.concatenate(self.y_true)
        y_pred_all = np.concatenate(self.y_pred)

        return compute_classification_metrics(y_true_all, y_pred_all)

    def reset(self):
        """Reset tracker."""
        self.y_true = []
        self.y_pred = []


__all__ = [
    "ClassificationMetrics",
    "confusion_matrix",
    "accuracy",
    "balanced_accuracy",
    "f1_score_metric",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "cohen_kappa",
    "matthews_corrcoef_metric",
    "sensitivity_specificity",
    "per_class_precision",
    "per_class_recall",
    "per_class_f1",
    "per_class_metrics",
    "compute_classification_metrics",
    "compute_roc_auc",
    "precision_recall_curve_data",
    "average_precision",
    "ClassificationMetricTracker",
]
