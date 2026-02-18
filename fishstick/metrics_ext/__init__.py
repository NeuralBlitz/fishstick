"""
Metrics Extension for Fishstick

Comprehensive evaluation metrics for machine learning models.

Modules:
- base: Base classes and interfaces
- classification_metrics: Classification metrics
- regression_metrics: Regression metrics
- ranking_metrics: Ranking and retrieval metrics
- detection_metrics: Object detection metrics
- custom_metrics: Domain-specific metrics

Usage:
    from fishstick.metrics_ext import (
        compute_classification_metrics,
        compute_regression_metrics,
        compute_ranking_metrics,
        compute_detection_metrics,
        compute_fairness_metrics,
    )
"""

from typing import Dict, Union
from torch import Tensor
import numpy as np

from .base import (
    MetricBase,
    MetricTracker,
    MetricRegistry,
    MetricResult,
    MetricAggregator,
    StreamingMetricTracker,
    create_metric_tracker,
)

from .classification_metrics import (
    ClassificationMetrics,
    confusion_matrix,
    accuracy,
    balanced_accuracy,
    f1_score_metric,
    macro_f1,
    micro_f1,
    weighted_f1,
    cohen_kappa,
    matthews_corrcoef_metric,
    sensitivity_specificity,
    per_class_precision,
    per_class_recall,
    per_class_f1,
    per_class_metrics,
    compute_classification_metrics,
    compute_roc_auc,
    precision_recall_curve_data,
    average_precision,
    ClassificationMetricTracker,
)

from .regression_metrics import (
    RegressionMetrics,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    huber_loss_metric,
    r2_score,
    adjusted_r2,
    mean_absolute_percentage_error,
    symmetric_mape,
    quantile_loss,
    explained_variance,
    max_error,
    mean_squared_logarithmic_error,
    median_absolute_error,
    compute_regression_metrics,
    compute_quantile_metrics,
    RegressionMetricTracker,
)

from .ranking_metrics import (
    dcg_at_k,
    ndcg_at_k,
    average_precision_single,
    map,
    reciprocal_rank_single,
    mrr,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    f1_at_k,
    compute_ndcg,
    compute_ranking_metrics,
    rank_predictions,
    RankingMetricTracker,
    compute_rank_metrics_from_scores,
)

from .detection_metrics import (
    DetectionMetrics,
    compute_iou,
    compute_ious,
    compute_ap,
    compute_map,
    compute_precision_recall_curve,
    compute_roc_curve,
    compute_auc,
    compute_f1_curve,
    compute_confusion_matrix_detection,
    compute_detection_metrics,
    compute_optimal_threshold,
    compute_iou_from_masks,
    DetectionMetricTracker,
)

from .custom_metrics import (
    TimeSeriesMetrics,
    FairnessMetrics,
    mean_absolute_scaled_error,
    time_series_mae,
    time_series_mape,
    compute_time_series_metrics,
    bleu_score,
    simplified_rouge,
    longest_common_subsequence_length,
    fid_score,
    inception_score,
    statistical_parity_difference,
    equal_opportunity_difference,
    disparate_impact_ratio,
    compute_fairness_metrics,
    composite_score,
)


def compute_all_metrics(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    task_type: str = "classification",
    **kwargs,
) -> Dict[str, float]:
    """
    Compute all appropriate metrics for the task type.

    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: Type of task ('classification', 'regression', 'ranking', 'detection')
        **kwargs: Additional arguments for specific metrics

    Returns:
        Dictionary of all computed metrics
    """
    if task_type == "classification":
        return compute_classification_metrics(y_true, y_pred)
    elif task_type == "regression":
        return compute_regression_metrics(y_true, y_pred)
    elif task_type == "ranking":
        return compute_ranking_metrics(y_true, y_pred, **kwargs)
    elif task_type == "detection":
        return compute_detection_metrics(y_true, y_pred, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


__all__ = [
    "MetricBase",
    "MetricTracker",
    "MetricRegistry",
    "MetricResult",
    "MetricAggregator",
    "StreamingMetricTracker",
    "create_metric_tracker",
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
    "RegressionMetrics",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "huber_loss_metric",
    "r2_score",
    "adjusted_r2",
    "mean_absolute_percentage_error",
    "symmetric_mape",
    "quantile_loss",
    "explained_variance",
    "max_error",
    "mean_squared_logarithmic_error",
    "median_absolute_error",
    "compute_regression_metrics",
    "compute_quantile_metrics",
    "RegressionMetricTracker",
    "dcg_at_k",
    "ndcg_at_k",
    "average_precision_single",
    "map",
    "reciprocal_rank_single",
    "mrr",
    "hit_rate_at_k",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "compute_ndcg",
    "compute_ranking_metrics",
    "rank_predictions",
    "RankingMetricTracker",
    "compute_rank_metrics_from_scores",
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
    "TimeSeriesMetrics",
    "FairnessMetrics",
    "mean_absolute_scaled_error",
    "time_series_mae",
    "time_series_mape",
    "compute_time_series_metrics",
    "bleu_score",
    "simplified_rouge",
    "longest_common_subsequence_length",
    "fid_score",
    "inception_score",
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "disparate_impact_ratio",
    "compute_fairness_metrics",
    "composite_score",
    "compute_all_metrics",
]
