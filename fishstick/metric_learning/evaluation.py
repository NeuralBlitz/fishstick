"""
Evaluation Metrics for Metric Learning

Implementation of evaluation metrics:
- Recall@k
- Normalized Mutual Information (NMI)
- F1 Score
- Mean Average Precision (MAP)
- Clustering accuracy
- Distance-based metrics
"""

from typing import Optional, List, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score,
    f1_score,
    adjusted_rand_score,
    accuracy_score,
)
from scipy.spatial.distance import cdist


def recall_at_k(
    distances: Tensor,
    labels: Tensor,
    k: int,
    exclude_self: bool = True,
) -> float:
    """Compute Recall@K metric.

    Measures the fraction of times the correct class is in the top-k nearest neighbors.

    Args:
        distances: Pairwise distance matrix (N, N)
        labels: Class labels (N,)
        k: Number of neighbors to consider
        exclude_self: Whether to exclude self from neighbors

    Returns:
        Recall@K score
    """
    if distances.dim() == 2:
        N = distances.shape[0]
        sorted_indices = torch.argsort(distances, dim=1)

        if exclude_self:
            sorted_indices = sorted_indices[:, 1:]
            k_actual = min(k, N - 1)
        else:
            k_actual = min(k, N)

        top_k = sorted_indices[:, :k_actual]

        correct = 0
        total = 0

        for i in range(N):
            neighbors = top_k[i]
            if labels[neighbors] == labels[i]:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0
    else:
        raise ValueError("distances must be a 2D tensor")


def mean_recall_at_k(
    distances: Tensor,
    labels: Tensor,
    k_list: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute Mean Recall@K for multiple K values.

    Args:
        distances: Pairwise distance matrix
        labels: Class labels
        k_list: List of K values

    Returns:
        Dictionary of Recall@K scores
    """
    if k_list is None:
        k_list = [1, 2, 4, 8, 16]

    results = {}
    for k in k_list:
        recall = recall_at_k(distances, labels, k)
        results[f"recall@{k}"] = recall

    results["mean_recall"] = np.mean(list(results.values()))
    return results


def normalized_mutual_information(
    labels_pred: Tensor,
    labels_true: Tensor,
) -> float:
    """Compute Normalized Mutual Information (NMI).

    Args:
        labels_pred: Predicted cluster labels
        labels_true: True class labels

    Returns:
        NMI score
    """
    labels_pred_np = labels_pred.cpu().numpy()
    labels_true_np = labels_true.cpu().numpy()

    nmi = normalized_mutual_info_score(labels_true_np, labels_pred_np)
    return nmi


def clustering_accuracy(
    labels_pred: Tensor,
    labels_true: Tensor,
) -> float:
    """Compute clustering accuracy using Hungarian algorithm.

    Args:
        labels_pred: Predicted cluster labels
        labels_true: True class labels

    Returns:
        Accuracy score
    """
    labels_pred_np = labels_pred.cpu().numpy()
    labels_true_np = labels_true.cpu().numpy()

    accuracy = accuracy_score(labels_true_np, labels_pred_np)
    return accuracy


def f1_score_metric(
    labels_pred: Tensor,
    labels_true: Tensor,
    average: str = "macro",
) -> float:
    """Compute F1 Score.

    Args:
        labels_pred: Predicted labels
        labels_true: True labels
        average: Averaging method

    Returns:
        F1 score
    """
    labels_pred_np = labels_pred.cpu().numpy()
    labels_true_np = labels_true.cpu().numpy()

    f1 = f1_score(labels_true_np, labels_pred_np, average=average, zero_division=0)
    return f1


def adjusted_rand_index(
    labels_pred: Tensor,
    labels_true: Tensor,
) -> float:
    """Compute Adjusted Rand Index (ARI).

    Args:
        labels_pred: Predicted cluster labels
        labels_true: True class labels

    Returns:
        ARI score
    """
    labels_pred_np = labels_pred.cpu().numpy()
    labels_true_np = labels_true.cpu().numpy()

    ari = adjusted_rand_score(labels_true_np, labels_pred_np)
    return ari


def mean_average_precision(
    distances: Tensor,
    labels: Tensor,
    exclude_self: bool = True,
) -> float:
    """Compute Mean Average Precision (MAP).

    Args:
        distances: Pairwise distance matrix
        labels: Class labels
        exclude_self: Whether to exclude self from retrieval

    Returns:
        MAP score
    """
    N = distances.shape[0]
    sorted_indices = torch.argsort(distances, dim=1)

    if exclude_self:
        sorted_indices = sorted_indices[:, 1:]

    average_precision = 0.0

    for i in range(N):
        neighbors = sorted_indices[i]
        relevant = (labels[neighbors] == labels[i]).float()

        num_relevant = relevant.sum().item()
        if num_relevant == 0:
            continue

        cumsum = torch.cumsum(relevant, dim=0)
        positions = torch.arange(1, len(relevant) + 1, device=relevant.device)

        precision_at_k = cumsum / positions
        relevant_precision = precision_at_k * relevant

        average_precision += relevant_precision.sum() / num_relevant

    return average_precision / N


def precision_at_k(
    distances: Tensor,
    labels: Tensor,
    k: int,
    exclude_self: bool = True,
) -> float:
    """Compute Precision@K.

    Args:
        distances: Pairwise distance matrix
        labels: Class labels
        k: Number of neighbors
        exclude_self: Whether to exclude self

    Returns:
        Precision@K score
    """
    N = distances.shape[0]
    sorted_indices = torch.argsort(distances, dim=1)

    if exclude_self:
        sorted_indices = sorted_indices[:, 1:]
        k_actual = min(k, N - 1)
    else:
        k_actual = min(k, N)

    top_k = sorted_indices[:, :k_actual]

    precision_sum = 0.0

    for i in range(N):
        neighbors = top_k[i]
        num_relevant = (labels[neighbors] == labels[i]).sum().item()
        precision_sum += num_relevant / k_actual

    return precision_sum / N


def compute_distance_matrix(
    features: Tensor,
    metric: str = "euclidean",
) -> Tensor:
    """Compute pairwise distance matrix.

    Args:
        features: Input features
        metric: Distance metric

    Returns:
        Distance matrix
    """
    features_np = features.cpu().numpy()
    dist = cdist(features_np, features_np, metric=metric)
    return torch.from_numpy(dist).to(features.device)


def retrieval_metrics(
    distances: Tensor,
    labels: Tensor,
    k_list: Optional[List[int]] = None,
) -> dict:
    """Compute comprehensive retrieval metrics.

    Args:
        distances: Pairwise distance matrix
        labels: Class labels
        k_list: List of K values

    Returns:
        Dictionary of metrics
    """
    if k_list is None:
        k_list = [1, 2, 4, 8, 16]

    metrics = {}

    for k in k_list:
        metrics[f"precision@{k}"] = precision_at_k(distances, labels, k)
        metrics[f"recall@{k}"] = recall_at_k(distances, labels, k)

    metrics["map"] = mean_average_precision(distances, labels)

    return metrics


def clustering_metrics(
    features: Tensor,
    labels_true: Tensor,
    labels_pred: Tensor,
) -> dict:
    """Compute clustering evaluation metrics.

    Args:
        features: Input features
        labels_true: True class labels
        labels_pred: Predicted cluster labels

    Returns:
        Dictionary of clustering metrics
    """
    metrics = {}

    metrics["nmi"] = normalized_mutual_information(labels_pred, labels_true)
    metrics["ari"] = adjusted_rand_index(labels_pred, labels_true)
    metrics["accuracy"] = clustering_accuracy(labels_pred, labels_true)

    if hasattr(features, "device"):
        features_np = features.cpu().numpy()
    else:
        features_np = features

    if hasattr(labels_true, "cpu"):
        labels_true_np = labels_true.cpu().numpy()
    else:
        labels_true_np = labels_true

    unique_true = np.unique(labels_true_np)
    centroids = []
    for label in unique_true:
        mask = labels_true_np == label
        centroid = features_np[mask].mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    if len(centroids) > 1:
        centroid_dists = cdist(centroids, centroids, metric="euclidean")
        np.fill_diagonal(centroid_dists, np.inf)
        min_inter_cluster = centroid_dists.min()
    else:
        min_inter_cluster = float("inf")

    intra_dists = []
    for i, label in enumerate(unique_true):
        mask = labels_true_np == label
        class_features = features_np[mask]
        if len(class_features) > 1:
            class_dists = cdist(class_features, class_features, metric="euclidean")
            intra_dists.append(class_dists.mean())

    avg_intra_cluster = np.mean(intra_dists) if intra_dists else 0

    metrics["avg_intra_cluster_distance"] = avg_intra_cluster
    metrics["min_inter_cluster_distance"] = min_inter_cluster
    metrics["separation_ratio"] = min_inter_cluster / (avg_intra_cluster + 1e-10)

    return metrics


def evaluate_retrieval(
    features: Tensor,
    labels: Tensor,
    k_list: Optional[List[int]] = None,
) -> dict:
    """Comprehensive retrieval evaluation.

    Args:
        features: Query/embedding features
        labels: Class labels
        k_list: List of K values for evaluation

    Returns:
        Dictionary of all metrics
    """
    distances = compute_distance_matrix(features)
    distances = distances.to(features.device)

    results = retrieval_metrics(distances, labels, k_list)

    return results


def evaluate_clustering(
    features: Tensor,
    labels_true: Tensor,
    labels_pred: Tensor,
) -> dict:
    """Comprehensive clustering evaluation.

    Args:
        features: Input features
        labels_true: True labels
        labels_pred: Predicted cluster labels

    Returns:
        Dictionary of all metrics
    """
    return clustering_metrics(features, labels_true, labels_pred)


class MetricTracker:
    """Tracker for metric learning evaluation metrics.

    Accumulates predictions and computes metrics at the end.
    """

    def __init__(self):
        self.features = []
        self.labels = []
        self.predictions = []

    def update(
        self,
        features: Tensor,
        labels: Optional[Tensor] = None,
        predictions: Optional[Tensor] = None,
    ):
        """Update tracker with new batch.

        Args:
            features: Embedding features
            labels: Ground truth labels
            predictions: Predicted labels
        """
        self.features.append(features.cpu())

        if labels is not None:
            self.labels.append(labels.cpu())

        if predictions is not None:
            self.predictions.append(predictions.cpu())

    def compute(self) -> dict:
        """Compute all metrics.

        Returns:
            Dictionary of metrics
        """
        features = torch.cat(self.features)

        if self.labels:
            labels = torch.cat(self.labels)

            distances = compute_distance_matrix(features)
            retrieval = retrieval_metrics(distances, labels)

            return retrieval
        else:
            return {}


__all__ = [
    "recall_at_k",
    "mean_recall_at_k",
    "normalized_mutual_information",
    "clustering_accuracy",
    "f1_score_metric",
    "adjusted_rand_index",
    "mean_average_precision",
    "precision_at_k",
    "compute_distance_matrix",
    "retrieval_metrics",
    "clustering_metrics",
    "evaluate_retrieval",
    "evaluate_clustering",
    "MetricTracker",
]
