"""
Climate and Weather Metrics

Evaluation metrics for climate and weather forecasting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


def compute_rmse(
    pred: Tensor,
    target: Tensor,
) -> float:
    """Root Mean Square Error.

    Args:
        pred: Predictions
        target: Ground truth

    Returns:
        RMSE value
    """
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def compute_mae(
    pred: Tensor,
    target: Tensor,
) -> float:
    """Mean Absolute Error.

    Args:
        pred: Predictions
        target: Ground truth

    Returns:
        MAE value
    """
    return torch.mean(torch.abs(pred - target)).item()


def compute_mse(
    pred: Tensor,
    target: Tensor,
) -> float:
    """Mean Square Error.

    Args:
        pred: Predictions
        target: Ground truth

    Returns:
        MSE value
    """
    return torch.mean((pred - target) ** 2).item()


def compute_correlation(
    pred: Tensor,
    target: Tensor,
) -> float:
    """Pearson correlation coefficient.

    Args:
        pred: Predictions
        target: Ground truth

    Returns:
        Correlation coefficient
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()

    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean

    correlation = (pred_centered * target_centered).sum() / (
        torch.sqrt((pred_centered**2).sum()) * torch.sqrt((target_centered**2).sum())
    )

    return correlation.item()


def compute_anomaly_correlation(
    pred: Tensor,
    target: Tensor,
    climatology: Tensor,
) -> float:
    """Anomaly correlation coefficient.

    Args:
        pred: Predictions
        target: Ground truth
        climatology: Long-term mean (climatology)

    Returns:
        Anomaly correlation
    """
    pred_anomaly = pred - climatology
    target_anomaly = target - climatology

    return compute_correlation(pred_anomaly, target_anomaly)


def compute_mean_absolute_percentage_error(
    pred: Tensor,
    target: Tensor,
    epsilon: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error.

    Args:
        pred: Predictions
        target: Ground truth
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value (as percentage)
    """
    return (
        (torch.abs(pred - target) / (torch.abs(target) + epsilon) * 100).mean().item()
    )


def compute_forecast_skill(
    pred: Tensor,
    target: Tensor,
    climatology: Tensor,
) -> float:
    """Forecast skill score (1 - RMSE_f/RMSE_c).

    Args:
        pred: Predictions
        target: Ground truth
        climatology: Climatological mean

    Returns:
        Forecast skill score
    """
    rmse_forecast = compute_rmse(pred, target)
    rmse_climatology = compute_rmse(climatology, target)

    if rmse_climatology == 0:
        return 0.0

    return 1.0 - rmse_forecast / rmse_climatology


def compute_heidke_skill_score(
    pred: Tensor,
    target: Tensor,
    threshold: float,
) -> float:
    """Heidke Skill Score for binary events.

    Args:
        pred: Predicted binary event
        target: Observed binary event
        threshold: Threshold for event detection

    Returns:
        HSS value
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    n = pred_binary.numel()

    hits = ((pred_binary == 1) & (target_binary == 1)).sum().item()
    misses = ((pred_binary == 0) & (target_binary == 1)).sum().item()
    false_alarms = ((pred_binary == 1) & (target_binary == 0)).sum().item()
    correct_negatives = ((pred_binary == 0) & (target_binary == 0)).sum().item()

    accuracy = (hits + correct_negatives) / n

    random_accuracy = (
        (hits + misses) * (hits + false_alarms)
        + (correct_negatives + misses) * (correct_negatives + false_alarms)
    ) / (n * n)

    if random_accuracy == 1:
        return 0.0

    hss = (accuracy - random_accuracy) / (1 - random_accuracy)

    return hss


def compute_equitable_threat_score(
    pred: Tensor,
    target: Tensor,
    threshold: float,
) -> float:
    """Equitable Threat Score (ETS).

    Args:
        pred: Predicted binary event
        target: Observed binary event
        threshold: Threshold for event detection

    Returns:
        ETS value
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    hits = ((pred_binary == 1) & (target_binary == 1)).sum().item()
    misses = ((pred_binary == 0) & (target_binary == 1)).sum().item()
    false_alarms = ((pred_binary == 1) & (target_binary == 0)).sum().item()

    expected = (hits + misses) * (hits + false_alarms) / (hits + misses + false_alarms)

    if (hits + misses + false_alarms) == 0 or expected == 0:
        return 0.0

    ets = (hits - expected) / (hits + misses + false_alarms - expected)

    return ets


def compute_brier_score(
    pred: Tensor,
    target: Tensor,
) -> float:
    """Brier Score for probabilistic forecasts.

    Args:
        pred: Predicted probabilities
        target: Observed binary outcome

    Returns:
        Brier Score (lower is better)
    """
    return torch.mean((pred - target) ** 2).item()


def compute_brier_skill_score(
    pred: Tensor,
    target: Tensor,
    climatology_prob: float = 0.5,
) -> float:
    """Brier Skill Score.

    Args:
        pred: Predicted probabilities
        target: Observed binary outcome
        climatology_prob: Climatological probability

    Returns:
        BSS value (1 is perfect, 0 is no skill)
    """
    bs = compute_brier_score(pred, target)

    bs_ref = (climatology_prob - target) ** 2
    bs_ref = bs_ref.mean().item()

    if bs_ref == 0:
        return 0.0

    return 1.0 - bs / bs_ref


def compute_roc_auc(
    pred: Tensor,
    target: Tensor,
) -> float:
    """ROC AUC Score.

    Args:
        pred: Predicted probabilities
        target: Observed binary outcome

    Returns:
        AUC value
    """
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    from sklearn.metrics import roc_auc_score

    return roc_auc_score(target_np, pred_np)


def compute_energy_score(
    pred: Tensor,
    target: Tensor,
    beta: float = 1.0,
) -> float:
    """Energy Score for probabilistic forecasts.

    Args:
        pred: Predicted samples (N, ...) or mean/std
        target: Single target observation
        beta: Beta parameter for energy score

    Returns:
        Energy score
    """
    if pred.dim() == 2 and pred.shape[1] == 2:
        mean = pred[:, 0]
        std = pred[:, 1]

        samples = torch.randn(100, *mean.shape, device=pred.device) * std.unsqueeze(
            0
        ) + mean.unsqueeze(0)
    else:
        samples = pred

    target_expanded = target.unsqueeze(0).expand(samples.shape[0], -1)

    diff = torch.norm(samples - target_expanded, p=2, dim=1)

    energy = diff.mean().item()

    return energy


def compute_continuous_ranked_probability_score(
    pred: Tensor,
    target: Tensor,
) -> float:
    """Continuous Ranked Probability Score (CRPS).

    Args:
        pred: Predicted distribution (mean, std)
        target: Target observation

    Returns:
        CRPS value
    """
    mean = pred[:, 0] if pred.dim() == 2 and pred.shape[1] == 2 else pred
    std = (
        pred[:, 1] if pred.dim() == 2 and pred.shape[1] == 2 else torch.ones_like(mean)
    )

    z = (target - mean) / std

    crps = std * (
        z * (2 * torch.distributions.Normal(0, 1).cdf(z) - 1)
        + 2 * torch.distributions.Normal(0, 1).log_prob(z).exp()
    )

    return crps.mean().item()


def compute_spatial_correlation(
    pred: Tensor,
    target: Tensor,
) -> float:
    """Spatial correlation.

    Args:
        pred: Predicted field (H, W)
        target: Target field (H, W)

    Returns:
        Spatial correlation
    """
    return compute_correlation(pred.unsqueeze(0), target.unsqueeze(0))


def compute_pattern_correlation(
    pred: Tensor,
    target: Tensor,
) -> float:
    """Pattern correlation (teleconnection correlation).

    Args:
        pred: Predicted field
        target: Target field

    Returns:
        Pattern correlation
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    pred_anom = pred_flat - pred_flat.mean()
    target_anom = target_flat - target_flat.mean()

    corr = (pred_anom * target_anom).sum() / (
        torch.sqrt((pred_anom**2).sum()) * torch.sqrt((target_anom**2).sum())
    )

    return corr.item()


def compute_rmse_by_variable(
    pred: Tensor,
    target: Tensor,
    variable_names: List[str],
) -> Dict[str, float]:
    """Compute RMSE for each variable.

    Args:
        pred: Predictions of shape (B, C, H, W)
        target: Target of shape (B, C, H, W)
        variable_names: List of variable names

    Returns:
        Dictionary mapping variable names to RMSE
    """
    results = {}
    for i, var in enumerate(variable_names):
        var_pred = pred[:, i]
        var_target = target[:, i]
        results[var] = compute_rmse(var_pred, var_target)

    return results


def compute_latitude_weighted_rmse(
    pred: Tensor,
    target: Tensor,
    lats: Tensor,
) -> float:
    """Latitude-weighted RMSE.

    Args:
        pred: Predictions (..., H, W)
        target: Target (..., H, W)
        lats: Latitude values (H,)

    Returns:
        Latitude-weighted RMSE
    """
    weights = torch.cos(torch.deg2rad(lats))
    weights = weights / weights.mean()

    error = (pred - target) ** 2

    weighted_error = error * weights.view(1, -1, 1)

    return torch.sqrt(weighted_error.mean()).item()


class ClimateMetrics:
    """Container for climate/weather metrics.

    Args:
        compute_latitude_weighting: Whether to use latitude weighting
    """

    def __init__(
        self,
        compute_latitude_weighting: bool = False,
    ):
        self.compute_latitude_weighting = compute_latitude_weighting
        self.history: Dict[str, List[float]] = {}

    def compute_all(
        self,
        pred: Tensor,
        target: Tensor,
        climatology: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Compute all metrics.

        Args:
            pred: Predictions
            target: Target
            climatology: Optional climatology for anomaly metrics

        Returns:
            Dictionary of metric values
        """
        metrics = {
            "rmse": compute_rmse(pred, target),
            "mae": compute_mae(pred, target),
            "mse": compute_mse(pred, target),
            "correlation": compute_correlation(pred, target),
        }

        if climatology is not None:
            metrics["anomaly_correlation"] = compute_anomaly_correlation(
                pred, target, climatology
            )
            metrics["forecast_skill"] = compute_forecast_skill(
                pred, target, climatology
            )

        return metrics

    def update_history(
        self,
        metrics: Dict[str, float],
    ):
        """Update metrics history.

        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get_averages(self) -> Dict[str, float]:
        """Get average of all metrics."""
        return {key: np.mean(vals) for key, vals in self.history.items()}
