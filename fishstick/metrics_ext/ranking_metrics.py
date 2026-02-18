"""
Ranking Metrics

Comprehensive metrics for evaluating ranking and retrieval models.

Classes:
- RankingMetrics: Container for ranking metrics

Functions:
- ndcg_at_k: Normalized Discounted Cumulative Gain at k
- dcg_at_k: Discounted Cumulative Gain at k
- map: Mean Average Precision
- mrr: Mean Reciprocal Rank
- hit_rate_at_k: Hit rate at k
- precision_at_k: Precision at k
- recall_at_k: Recall at k
- average_precision: Average precision for single query
- reciprocal_rank: Reciprocal rank for single query
"""

from typing import Optional, List, Dict, Union

import torch
from torch import Tensor
import numpy as np


def dcg_at_k(
    relevance: Union[Tensor, np.ndarray],
    k: Optional[int] = None,
) -> float:
    """
    Compute Discounted Cumulative Gain (DCG) at k.

    Args:
        relevance: Relevance scores
        k: Cutoff position (if None, use all)

    Returns:
        DCG score
    """
    if isinstance(relevance, Tensor):
        relevance = relevance.cpu().numpy()

    if k is None:
        k = len(relevance)
    else:
        k = min(k, len(relevance))

    gains = relevance[:k]
    discounts = np.log2(np.arange(2, k + 2))

    dcg = np.sum(gains / discounts)

    return dcg


def ndcg_at_k(
    relevance: Union[Tensor, np.ndarray],
    k: Optional[int] = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        relevance: Relevance scores
        k: Cutoff position

    Returns:
        NDCG score
    """
    if isinstance(relevance, Tensor):
        relevance = relevance.cpu().numpy()

    dcg = dcg_at_k(relevance, k)

    ideal_relevance = np.sort(relevance)[::-1]
    idcg = dcg_at_k(ideal_relevance, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision_single(
    relevance: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Average Precision for a single query.

    Args:
        relevance: Binary relevance scores

    Returns:
        Average precision
    """
    if isinstance(relevance, Tensor):
        relevance = relevance.cpu().numpy()

    num_relevant = np.sum(relevance)
    if num_relevant == 0:
        return 0.0

    precisions = []
    relevant_count = 0

    for i, rel in enumerate(relevance, 1):
        if rel == 1:
            relevant_count += 1
            precisions.append(relevant_count / i)

    return np.mean(precisions) if precisions else 0.0


def map(
    relevance_matrix: Union[Tensor, np.ndarray],
    k: Optional[int] = None,
) -> float:
    """
    Compute Mean Average Precision (MAP).

    Args:
        relevance_matrix: Matrix of relevance scores [num_queries, num_items]
        k: Cutoff position

    Returns:
        MAP score
    """
    if isinstance(relevance_matrix, Tensor):
        relevance_matrix = relevance_matrix.cpu().numpy()

    if relevance_matrix.ndim == 1:
        relevance_matrix = relevance_matrix.reshape(1, -1)

    aps = []
    for relevance in relevance_matrix:
        if k is not None:
            relevance = relevance[:k]
        ap = average_precision_single(relevance)
        aps.append(ap)

    return np.mean(aps)


def reciprocal_rank_single(
    relevance: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Reciprocal Rank for a single query.

    Args:
        relevance: Binary relevance scores

    Returns:
        Reciprocal rank
    """
    if isinstance(relevance, Tensor):
        relevance = relevance.cpu().numpy()

    for i, rel in enumerate(relevance, 1):
        if rel == 1:
            return 1.0 / i

    return 0.0


def mrr(
    relevance_matrix: Union[Tensor, np.ndarray],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        relevance_matrix: Matrix of relevance scores [num_queries, num_items]

    Returns:
        MRR score
    """
    if isinstance(relevance_matrix, Tensor):
        relevance_matrix = relevance_matrix.cpu().numpy()

    if relevance_matrix.ndim == 1:
        relevance_matrix = relevance_matrix.reshape(1, -1)

    rr_scores = []
    for relevance in relevance_matrix:
        rr = reciprocal_rank_single(relevance)
        rr_scores.append(rr)

    return np.mean(rr_scores)


def hit_rate_at_k(
    relevance_matrix: Union[Tensor, np.ndarray],
    k: int,
) -> float:
    """
    Compute Hit Rate at k.

    Args:
        relevance_matrix: Matrix of relevance scores [num_queries, num_items]
        k: Cutoff position

    Returns:
        Hit rate
    """
    if isinstance(relevance_matrix, Tensor):
        relevance_matrix = relevance_matrix.cpu().numpy()

    if relevance_matrix.ndim == 1:
        relevance_matrix = relevance_matrix.reshape(1, -1)

    hits = 0
    num_queries = relevance_matrix.shape[0]

    for relevance in relevance_matrix:
        if np.sum(relevance[:k]) > 0:
            hits += 1

    return hits / num_queries


def precision_at_k(
    relevance_matrix: Union[Tensor, np.ndarray],
    k: int,
) -> float:
    """
    Compute Precision at k.

    Args:
        relevance_matrix: Matrix of relevance scores [num_queries, num_items]
        k: Cutoff position

    Returns:
        Precision at k
    """
    if isinstance(relevance_matrix, Tensor):
        relevance_matrix = relevance_matrix.cpu().numpy()

    if relevance_matrix.ndim == 1:
        relevance_matrix = relevance_matrix.reshape(1, -1)

    precisions = []
    for relevance in relevance_matrix:
        num_relevant = np.sum(relevance[:k])
        precisions.append(num_relevant / k)

    return np.mean(precisions)


def recall_at_k(
    relevance_matrix: Union[Tensor, np.ndarray],
    k: int,
) -> float:
    """
    Compute Recall at k.

    Args:
        relevance_matrix: Matrix of relevance scores [num_queries, num_items]
        k: Cutoff position

    Returns:
        Recall at k
    """
    if isinstance(relevance_matrix, Tensor):
        relevance_matrix = relevance_matrix.cpu().numpy()

    if relevance_matrix.ndim == 1:
        relevance_matrix = relevance_matrix.reshape(1, -1)

    recalls = []
    for relevance in relevance_matrix:
        total_relevant = np.sum(relevance)
        if total_relevant > 0:
            num_relevant = np.sum(relevance[:k])
            recalls.append(num_relevant / total_relevant)
        else:
            recalls.append(0.0)

    return np.mean(recalls)


def f1_at_k(
    relevance_matrix: Union[Tensor, np.ndarray],
    k: int,
) -> float:
    """
    Compute F1 Score at k.

    Args:
        relevance_matrix: Matrix of relevance scores [num_queries, num_items]
        k: Cutoff position

    Returns:
        F1 score at k
    """
    precision = precision_at_k(relevance_matrix, k)
    recall = recall_at_k(relevance_matrix, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_ndcg(
    relevance_matrix: Union[Tensor, np.ndarray],
    k: Optional[int] = None,
) -> float:
    """
    Compute average NDCG across all queries.

    Args:
        relevance_matrix: Matrix of relevance scores [num_queries, num_items]
        k: Cutoff position

    Returns:
        Average NDCG
    """
    if isinstance(relevance_matrix, Tensor):
        relevance_matrix = relevance_matrix.cpu().numpy()

    if relevance_matrix.ndim == 1:
        relevance_matrix = relevance_matrix.reshape(1, -1)

    ndcg_scores = []
    for relevance in relevance_matrix:
        ndcg = ndcg_at_k(relevance, k)
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)


def compute_ranking_metrics(
    relevance_matrix: Union[Tensor, np.ndarray],
    k_list: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute all ranking metrics.

    Args:
        relevance_matrix: Matrix of relevance scores [num_queries, num_items]
        k_list: List of k values for evaluation

    Returns:
        Dictionary of metrics
    """
    if k_list is None:
        k_list = [1, 3, 5, 10]

    if isinstance(relevance_matrix, Tensor):
        relevance_matrix = relevance_matrix.cpu().numpy()

    metrics = {}

    metrics["map"] = map(relevance_matrix)
    metrics["mrr"] = mrr(relevance_matrix)

    for k in k_list:
        metrics[f"ndcg@{k}"] = compute_ndcg(relevance_matrix, k)
        metrics[f"precision@{k}"] = precision_at_k(relevance_matrix, k)
        metrics[f"recall@{k}"] = recall_at_k(relevance_matrix, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(relevance_matrix, k)
        metrics[f"f1@{k}"] = f1_at_k(relevance_matrix, k)

    return metrics


def rank_predictions(
    scores: Union[Tensor, np.ndarray],
    descending: bool = True,
) -> Tensor:
    """
    Rank predictions by scores.

    Args:
        scores: Prediction scores
        descending: Sort in descending order

    Returns:
        Ranked indices
    """
    if isinstance(scores, Tensor):
        return torch.argsort(scores, descending=descending)
    else:
        return np.argsort(scores)[::-1] if descending else np.argsort(scores)


class RankingMetricTracker:
    """Track ranking metrics over batches."""

    def __init__(self):
        self.relevance_matrix: List[np.ndarray] = []

    def update(
        self,
        relevance: Union[Tensor, np.ndarray],
    ):
        """Update tracker with new batch."""
        if isinstance(relevance, Tensor):
            relevance = relevance.cpu().numpy()

        self.relevance_matrix.append(relevance)

    def compute(
        self,
        k_list: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Compute all ranking metrics."""
        relevance_all = np.concatenate(self.relevance_matrix, axis=0)

        if relevance_all.ndim == 1:
            relevance_all = relevance_all.reshape(1, -1)

        return compute_ranking_metrics(relevance_all, k_list)

    def reset(self):
        """Reset tracker."""
        self.relevance_matrix = []


def compute_rank_metrics_from_scores(
    y_score: Union[Tensor, np.ndarray],
    y_true: Union[Tensor, np.ndarray],
    k_list: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute ranking metrics from prediction scores and true labels.

    Args:
        y_score: Prediction scores
        y_true: True relevance labels
        k_list: List of k values

    Returns:
        Dictionary of metrics
    """
    if isinstance(y_score, Tensor):
        y_score = y_score.cpu().numpy()
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()

    if k_list is None:
        k_list = [1, 3, 5, 10]

    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_score = y_score.reshape(1, -1)

    relevance_matrix = []
    for scores, true_rel in zip(y_score, y_true):
        sorted_idx = np.argsort(scores)[::-1]
        relevance = true_rel[sorted_idx]
        relevance_matrix.append(relevance)

    relevance_matrix = np.array(relevance_matrix)

    return compute_ranking_metrics(relevance_matrix, k_list)


__all__ = [
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
]
