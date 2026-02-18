"""
Regression Metrics

Comprehensive metrics for evaluating regression models.

Classes:
- RegressionMetrics: Container for all regression metrics
- MetricTracker: Track metrics over batches

Functions:
- mean_absolute_error: MAE
- mean_squared_error: MSE
- root_mean_squared_error: RMSE
- huber_loss_metric: Huber loss as metric
- r2_score: R² (coefficient of determination)
- adjusted_r2: Adjusted R²
- mape: Mean Absolute Percentage Error
- smape: Symmetric MAPE
- quantile_loss: Quantile loss
- explained_variance: Explained variance score
- max_error: Maximum error
- mean_absolute_error: MAE
"""

from typing import Optional, List, Dict, Union
from dataclasses import dataclass

import torch
from torch import Tensor
import numpy as np
from sklearn.metrics import (
    mean_absolute_error as sk_mae,
    mean_squared_error as sk_mse,
    r2_score as sk_r2,
    explained_variance_score,
    max_error as sk_max_error,
    mean_absolute_percentage_error as sk_mape,
)


@dataclass
class RegressionMetrics:
    """Container for regression metrics."""

    mae: float
    mse: float
    rmse: float
    r2: float
    adjusted_r2: float
    mape: float
    smape: float
    explained_variance: float
    max_error: float


def mean_absolute_error(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Mean Absolute Error (MAE).

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        MAE score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return sk_mae(y_true, y_pred)


def mean_squared_error(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Mean Squared Error (MSE).

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        MSE score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return sk_mse(y_true, y_pred)


def root_mean_squared_error(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def huber_loss_metric(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    delta: float = 1.0,
) -> float:
    """
    Compute Huber loss as a metric.

    Args:
        y_true: True target values
        y_pred: Predicted values
        delta: Delta parameter for Huber loss

    Returns:
        Huber loss
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    linear_mask = abs_errors > delta
    quadratic_mask = ~linear_mask

    loss = np.zeros_like(errors, dtype=float)
    loss[quadratic_mask] = 0.5 * errors[quadratic_mask] ** 2
    loss[linear_mask] = delta * (abs_errors[linear_mask] - 0.5 * delta)

    return np.mean(loss)


def r2_score(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute R² Score (coefficient of determination).

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        R² score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return sk_r2(y_true, y_pred)


def adjusted_r2(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    n_features: Optional[int] = None,
) -> float:
    """
    Compute Adjusted R² Score.

    Args:
        y_true: True target values
        y_pred: Predicted values
        n_features: Number of features (if None, inferred from y_pred shape)

    Returns:
        Adjusted R² score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    r2 = sk_r2(y_true, y_pred)

    n = len(y_true)
    if n_features is None:
        n_features = 1

    adjusted = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    return adjusted


def mean_absolute_percentage_error(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        MAPE score (as percentage)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return sk_mape(y_true, y_pred) * 100


def symmetric_mape(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Symmetric MAPE (SMAPE).

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        SMAPE score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    mask = denominator != 0
    smape = np.mean(numerator[mask] / denominator[mask]) * 100

    return smape


def quantile_loss(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    quantile: float = 0.5,
) -> float:
    """
    Compute Quantile Loss.

    Args:
        y_true: True target values
        y_pred: Predicted values
        quantile: Quantile to compute

    Returns:
        Quantile loss
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    errors = y_true - y_pred
    loss = np.maximum(quantile * errors, (quantile - 1) * errors)

    return np.mean(loss)


def explained_variance(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Explained Variance Score.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Explained variance score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return explained_variance_score(y_true, y_pred)


def max_error(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Maximum Error.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Maximum error
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return sk_max_error(y_true, y_pred)


def mean_squared_logarithmic_error(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Mean Squared Logarithmic Error (MSLE).

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        MSLE score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true_safe = np.clip(y_true, 1e-10, None)
    y_pred_safe = np.clip(y_pred, 1e-10, None)

    return np.mean((np.log1p(y_true_safe) - np.log1p(y_pred_safe)) ** 2)


def median_absolute_error(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Median Absolute Error.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Median absolute error
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return np.median(np.abs(y_true - y_pred))


def compute_regression_metrics(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    n_features: Optional[int] = None,
) -> RegressionMetrics:
    """
    Compute all regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values
        n_features: Number of features for adjusted R²

    Returns:
        RegressionMetrics dataclass
    """
    if isinstance(y_true, Tensor):
        y_true_np = y_true.cpu().numpy()
    else:
        y_true_np = y_true

    if isinstance(y_pred, Tensor):
        y_pred_np = y_pred.cpu().numpy()
    else:
        y_pred_np = y_pred

    if n_features is None and hasattr(y_pred, "ndim"):
        if y_pred.ndim > 1:
            n_features = y_pred.shape[1]
        else:
            n_features = 1

    return RegressionMetrics(
        mae=mean_absolute_error(y_true_np, y_pred_np),
        mse=mean_squared_error(y_true_np, y_pred_np),
        rmse=root_mean_squared_error(y_true_np, y_pred_np),
        r2=r2_score(y_true_np, y_pred_np),
        adjusted_r2=adjusted_r2(y_true_np, y_pred_np, n_features),
        mape=mean_absolute_percentage_error(y_true_np, y_pred_np),
        smape=symmetric_mape(y_true_np, y_pred_np),
        explained_variance=explained_variance(y_true_np, y_pred_np),
        max_error=max_error(y_true_np, y_pred_np),
    )


def compute_quantile_metrics(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    quantiles: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute metrics at multiple quantiles.

    Args:
        y_true: True target values
        y_pred: Predicted values
        quantiles: List of quantiles to compute

    Returns:
        Dictionary of quantile losses
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    results = {}
    for q in quantiles:
        loss = quantile_loss(y_true, y_pred, quantile=q)
        results[f"quantile_{q}"] = loss

    return results


class RegressionMetricTracker:
    """Track regression metrics over batches."""

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

    def compute(self) -> RegressionMetrics:
        """Compute all metrics."""
        y_true_all = np.concatenate(self.y_true)
        y_pred_all = np.concatenate(self.y_pred)

        return compute_regression_metrics(y_true_all, y_pred_all)

    def reset(self):
        """Reset tracker."""
        self.y_true = []
        self.y_pred = []


__all__ = [
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
]
