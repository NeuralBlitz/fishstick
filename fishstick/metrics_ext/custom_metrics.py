"""
Custom Domain Metrics

Domain-specific metrics for various applications including:
- Time series forecasting metrics
- NLP metrics
- Generative model metrics
- Fairness metrics

Functions:
- mase: Mean Absolute Scaled Error
- time_series_mae: Time series specific MAE
- bleu_score: BLEU score for NLP
- simplified_rouge: Simplified ROUGE score
- fid_score: Fréchet Inception Distance
- is_score: Inception Score
- statistical_parity_difference: Fairness metric
- equal_opportunity_difference: Fairness metric
- composite_score: Combine multiple metrics
"""

from typing import Optional, List, Dict, Union, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor
import numpy as np


@dataclass
class TimeSeriesMetrics:
    """Container for time series metrics."""

    mae: float
    mse: float
    mase: float
    smape: float
    rmse: float


@dataclass
class FairnessMetrics:
    """Container for fairness metrics."""

    statistical_parity: float
    equal_opportunity: float
    disparate_impact: float


def mean_absolute_scaled_error(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    seasonality: int = 1,
) -> float:
    """
    Compute Mean Absolute Scaled Error (MASE).

    Args:
        y_true: True values
        y_pred: Predicted values
        seasonality: Seasonality period

    Returns:
        MASE score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    n = len(y_true)
    mae = np.mean(np.abs(y_true - y_pred))

    naive_errors = np.abs(y_true[seasonality:] - y_true[:-seasonality])
    naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0

    return mae / naive_mae


def time_series_mae(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute time series specific MAE.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return np.mean(np.abs(y_true - y_pred))


def time_series_mape(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    epsilon: float = 1e-10,
) -> float:
    """
    Compute MAPE for time series.

    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE score (percentage)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    mask = np.abs(y_true) > epsilon
    if np.sum(mask) == 0:
        return 0.0

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_time_series_metrics(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    seasonality: int = 1,
) -> TimeSeriesMetrics:
    """
    Compute all time series metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        seasonality: Seasonality period

    Returns:
        TimeSeriesMetrics dataclass
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()

    return TimeSeriesMetrics(
        mae=np.mean(np.abs(y_true - y_pred)),
        mse=np.mean((y_true - y_pred) ** 2),
        mase=mean_absolute_scaled_error(y_true, y_pred, seasonality),
        smape=100
        * np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
        ),
        rmse=np.sqrt(np.mean((y_true - y_pred) ** 2)),
    )


def bleu_score(
    reference: List[str],
    hypothesis: str,
    n: int = 4,
) -> float:
    """
    Compute BLEU score (simplified).

    Args:
        reference: Reference sentences
        hypothesis: Hypothesis sentence
        n: Maximum n-gram order

    Returns:
        BLEU score
    """
    ref_tokens = reference[0].split() if reference else []
    hyp_tokens = hypothesis.split()

    if not hyp_tokens:
        return 0.0

    scores = []
    for i in range(1, n + 1):
        ref_ngrams = [
            tuple(ref_tokens[j : j + i]) for j in range(len(ref_tokens) - i + 1)
        ]
        hyp_ngrams = [
            tuple(hyp_tokens[j : j + i]) for j in range(len(hyp_tokens) - i + 1)
        ]

        if not hyp_ngrams:
            scores.append(0)
            continue

        matches = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
        scores.append(matches / len(hyp_ngrams) if hyp_ngrams else 0)

    if not scores or all(s == 0 for s in scores):
        return 0.0

    brevity_penalty = min(1.0, np.exp(1 - len(ref_tokens) / (len(hyp_tokens) + 1)))

    geo_mean = np.exp(np.mean([np.log(s + 1e-10) for s in scores]))

    return brevity_penalty * geo_mean


def simplified_rouge(
    reference: str,
    hypothesis: str,
    mode: str = "L",
) -> float:
    """
    Compute simplified ROUGE score.

    Args:
        reference: Reference sentence
        hypothesis: Hypothesis sentence
        mode: ROUGE mode ('L' for longest common subsequence, '1' for unigrams, '2' for bigrams)

    Returns:
        ROUGE score
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    if mode == "1":
        ref_counts = Counter(ref_tokens)
        hyp_counts = Counter(hyp_tokens)
        overlap = sum((ref_counts & hyp_counts).values())
        return overlap / len(ref_tokens)

    elif mode == "2":
        ref_bigrams = [
            (ref_tokens[i], ref_tokens[i + 1]) for i in range(len(ref_tokens) - 1)
        ]
        hyp_bigrams = [
            (hyp_tokens[i], hyp_tokens[i + 1]) for i in range(len(hyp_tokens) - 1)
        ]

        ref_bg_set = set(ref_bigrams)
        hyp_bg_set = set(hyp_bigrams)

        overlap = len(ref_bg_set & hyp_bg_set)
        return overlap / len(ref_bg_set) if ref_bg_set else 0.0

    elif mode == "L":
        lcs_length = longest_common_subsequence_length(ref_tokens, hyp_tokens)
        return lcs_length / len(ref_tokens) if ref_tokens else 0.0

    return 0.0


def longest_common_subsequence_length(
    seq1: List[str],
    seq2: List[str],
) -> int:
    """
    Compute LCS length for ROUGE-L.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        LCS length
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def fid_score(
    real_features: Union[Tensor, np.ndarray],
    fake_features: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Fréchet Inception Distance (FID).

    Args:
        real_features: Features from real images
        fake_features: Features from generated images

    Returns:
        FID score
    """
    if isinstance(real_features, Tensor):
        real_features = real_features.cpu().numpy()
    if isinstance(fake_features, Tensor):
        fake_features = fake_features.cpu().numpy()

    real_mean = np.mean(real_features, axis=0)
    fake_mean = np.mean(fake_features, axis=0)

    real_cov = np.cov(real_features, rowvar=False)
    fake_cov = np.cov(fake_features, rowvar=False)

    mean_diff = real_mean - fake_mean
    mean_diff_squared = np.sum(mean_diff**2)

    covmean = np.sqrtm(real_cov @ fake_cov)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = mean_diff_squared + np.trace(real_cov + fake_cov - 2 * covmean)

    return float(fid)


def inception_score(
    predictions: Union[Tensor, np.ndarray],
    splits: int = 10,
    eps: float = 1e-10,
) -> Tuple[float, float]:
    """
    Compute Inception Score.

    Args:
        predictions: Class probabilities from Inception model
        splits: Number of splits for IS calculation
        eps: Small value for numerical stability

    Returns:
        Tuple of (mean, standard deviation)
    """
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()

    if predictions.ndim == 1:
        predictions = np.exp(predictions) / np.exp(predictions).sum(
            axis=1, keepdims=True
        )
    else:
        predictions = predictions / predictions.sum(axis=1, keepdims=True)

    split_scores = []
    part_size = predictions.shape[0] // splits

    for i in range(splits):
        part = predictions[i * part_size : (i + 1) * part_size]
        kl_div = part * (
            np.log(part + eps) - np.log(part.mean(axis=0, keepdims=True) + eps)
        )
        kl_div = kl_div.sum(axis=1)
        split_scores.append(np.exp(kl_div.mean()))

    return float(np.mean(split_scores)), float(np.std(split_scores))


def statistical_parity_difference(
    y_pred: Union[Tensor, np.ndarray],
    sensitive_attribute: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Statistical Parity Difference.

    Measures difference in positive prediction rates between groups.

    Args:
        y_pred: Predicted labels
        sensitive_attribute: Sensitive attribute (0 or 1)

    Returns:
        Statistical parity difference
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(sensitive_attribute, Tensor):
        sensitive_attribute = sensitive_attribute.cpu().numpy()

    group_0 = sensitive_attribute == 0
    group_1 = sensitive_attribute == 1

    if np.sum(group_0) == 0 or np.sum(group_1) == 0:
        return 0.0

    rate_0 = np.mean(y_pred[group_0])
    rate_1 = np.mean(y_pred[group_1])

    return rate_1 - rate_0


def equal_opportunity_difference(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    sensitive_attribute: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Equal Opportunity Difference.

    Measures difference in true positive rates between groups.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attribute: Sensitive attribute (0 or 1)

    Returns:
        Equal opportunity difference
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(sensitive_attribute, Tensor):
        sensitive_attribute = sensitive_attribute.cpu().numpy()

    group_0 = sensitive_attribute == 0
    group_1 = sensitive_attribute == 1

    positive_0 = y_true == 1
    positive_1 = y_true == 1

    if np.sum(group_0 & positive_0) == 0 or np.sum(group_1 & positive_1) == 0:
        return 0.0

    tpr_0 = np.mean(y_pred[group_0 & positive_0])
    tpr_1 = np.mean(y_pred[group_1 & positive_1])

    return tpr_1 - tpr_0


def disparate_impact_ratio(
    y_pred: Union[Tensor, np.ndarray],
    sensitive_attribute: Union[Tensor, np.ndarray],
    epsilon: float = 1e-10,
) -> float:
    """
    Compute Disparate Impact Ratio.

    Ratio of positive prediction rates between groups.

    Args:
        y_pred: Predicted labels
        sensitive_attribute: Sensitive attribute (0 or 1)
        epsilon: Small value for numerical stability

    Returns:
        Disparate impact ratio
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(sensitive_attribute, Tensor):
        sensitive_attribute = sensitive_attribute.cpu().numpy()

    group_0 = sensitive_attribute == 0
    group_1 = sensitive_attribute == 1

    if np.sum(group_0) == 0 or np.sum(group_1) == 0:
        return 1.0

    rate_0 = np.mean(y_pred[group_0]) + epsilon
    rate_1 = np.mean(y_pred[group_1]) + epsilon

    return rate_1 / rate_0


def compute_fairness_metrics(
    y_true: Union[Tensor, np.ndarray],
    y_pred: Union[Tensor, np.ndarray],
    sensitive_attribute: Union[Tensor, np.ndarray],
) -> FairnessMetrics:
    """
    Compute all fairness metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attribute: Sensitive attribute

    Returns:
        FairnessMetrics dataclass
    """
    return FairnessMetrics(
        statistical_parity=statistical_parity_difference(y_pred, sensitive_attribute),
        equal_opportunity=equal_opportunity_difference(
            y_true, y_pred, sensitive_attribute
        ),
        disparate_impact=disparate_impact_ratio(y_pred, sensitive_attribute),
    )


def composite_score(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute weighted composite score from multiple metrics.

    Args:
        metrics: Dictionary of metric name to value
        weights: Dictionary of metric name to weight

    Returns:
        Composite score
    """
    if weights is None:
        return np.mean(list(metrics.values()))

    total_weight = sum(weights.values())
    weighted_sum = sum(metrics.get(k, 0) * w for k, w in weights.items())

    return weighted_sum / total_weight if total_weight > 0 else 0.0


class Counter:
    """Simple counter for BLEU/ROUGE."""

    def __init__(self):
        self.counts: Dict[Tuple, int] = {}

    def __getitem__(self, key: Tuple):
        return self.counts.get(key, 0)

    def __and__(self, other: "Counter") -> "Counter":
        result = Counter()
        for key in self.counts:
            if key in other.counts:
                result.counts[key] = min(self.counts[key], other.counts[key])
        return result


__all__ = [
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
]
