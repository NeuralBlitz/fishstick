"""Batch correction for single-cell data."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class BatchCorrector(nn.Module):
    """Batch effect correction for single-cell data.

    Implements several batch correction methods:
    - Combat-style adjustment
    - Mutual nearest neighbors (MNN)
    - Harmony-style integration

    Attributes:
        method: Correction method to use
    """

    def __init__(self, method: str = "combat") -> None:
        super().__init__()
        self.method = method

    def forward(
        self,
        expression: Tensor,
        batch_labels: Tensor,
    ) -> Tensor:
        """Correct batch effects.

        Args:
            expression: Expression matrix
            batch_labels: Batch labels for each cell

        Returns:
            Batch-corrected expression
        """
        if self.method == "combat":
            return self._combat_correction(expression, batch_labels)
        elif self.method == "mnn":
            return self._mnn_correction(expression, batch_labels)
        elif self.method == "harmony":
            return self._harmony_correction(expression, batch_labels)
        else:
            return expression

    def _combat_correction(
        self,
        expression: Tensor,
        batch_labels: Tensor,
    ) -> Tensor:
        """ComBat-style batch correction.

        Args:
            expression: Expression matrix
            batch_labels: Batch labels

        Returns:
            Corrected expression
        """
        data = expression.numpy()
        batches = batch_labels.numpy()

        unique_batches = np.unique(batches)

        gene_means = data.mean(axis=0)
        gene_vars = data.var(axis=0)

        corrected = data.copy()

        for batch in unique_batches:
            batch_mask = batches == batch

            batch_mean = data[batch_mask].mean(axis=0)
            batch_var = data[batch_mask].var(axis=0)

            batch_mean_normalized = (batch_mean - gene_means) / (
                gene_vars.sqrt() + 1e-8
            )

            corrected[batch_mask] = data[batch_mask] - batch_mean_normalized

        return torch.tensor(corrected, dtype=torch.float32)

    def _mnn_correction(
        self,
        expression: Tensor,
        batch_labels: Tensor,
    ) -> Tensor:
        """Mutual nearest neighbors batch correction.

        Args:
            expression: Expression matrix
            batch_labels: Batch labels

        Returns:
            Corrected expression
        """
        from scipy.spatial.distance import cdist

        data = expression.numpy()
        batches = batch_labels.numpy()

        unique_batches = np.unique(batches)

        if len(unique_batches) < 2:
            return expression

        reference_batch = unique_batches[0]
        reference_data = data[batches == reference_batch]

        corrected = data.copy()

        for batch in unique_batches[1:]:
            batch_data = data[batches == batch]

            distances = cdist(batch_data, reference_data)

            batch_knn = np.argsort(distances, axis=1)[:, :10]
            ref_knn = np.argsort(distances, axis=0)[:10, :]

            for i in range(len(batch_data)):
                mutual_neighbors = set(batch_knn[i]) & set(
                    ref_knn[:, i % len(reference_data)]
                )

                if mutual_neighbors:
                    neighbor_indices = list(mutual_neighbors)
                    corrections = (
                        reference_data[neighbor_indices].mean(axis=0) - batch_data[i]
                    )
                    corrected[batches == batch][i] = batch_data[i] + corrections * 0.5

        return torch.tensor(corrected, dtype=torch.float32)

    def _harmony_correction(
        self,
        expression: Tensor,
        batch_labels: Tensor,
        n_clusters: int = 20,
        max_iter: int = 10,
    ) -> Tensor:
        """Harmony-style batch correction.

        Args:
            expression: Expression matrix
            batch_labels: Batch labels
            n_clusters: Number of clusters
            max_iter: Maximum iterations

        Returns:
            Corrected expression
        """
        from scipy.cluster.hierarchy import linkage, fcluster

        data = expression.numpy()
        batches = batch_labels.numpy()

        Z = linkage(data, method="ward")
        cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")

        corrected = data.copy()

        for iteration in range(max_iter):
            for cluster in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster

                batch_centroids = []
                for batch in np.unique(batches):
                    batch_mask = (batches == batch) & cluster_mask
                    if batch_mask.sum() > 0:
                        batch_centroids.append(data[batch_mask].mean(axis=0))

                if len(batch_centroids) < 2:
                    continue

                centroid_mean = np.mean(batch_centroids, axis=0)

                for batch in np.unique(batches):
                    batch_mask = (batches == batch) & cluster_mask
                    if batch_mask.sum() > 0:
                        batch_centroid = data[batch_mask].mean(axis=0)
                        correction = batch_centroid - centroid_mean

                        correction_factor = 0.1
                        corrected[batch_mask] = (
                            data[batch_mask] - correction * correction_factor
                        )

        return torch.tensor(corrected, dtype=torch.float32)


def compute_batch_entropy(batch_labels: Tensor) -> float:
    """Compute entropy of batch distribution.

    Args:
        batch_labels: Batch labels

    Returns:
        Entropy value
    """
    batches = batch_labels.numpy()
    unique, counts = np.unique(batches, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy


def compute_kBET_score(
    expression: Tensor,
    batch_labels: Tensor,
    k: int = 50,
) -> float:
    """Compute kBET batch effect metric.

    Args:
        expression: Expression matrix
        batch_labels: Batch labels
        k: Number of neighbors

    Returns:
        kBET score
    """
    from scipy.spatial.distance import cdist

    data = expression.numpy()
    batches = batch_labels.numpy()

    distances = cdist(data, data)
    knn_idx = np.argsort(distances, axis=1)[:, 1 : k + 1]

    chi_square_scores = []

    for i in range(len(data)):
        observed_batch = batches[knn_idx[i]]
        expected_freq = np.bincount(batches) / len(batches)

        observed_freq = np.bincount(observed_batch, minlength=len(expected_freq)) / k

        chi_square = np.sum(
            (observed_freq - expected_freq) ** 2 / (expected_freq + 1e-10)
        )
        chi_square_scores.append(chi_square)

    return 1.0 / (1.0 + np.mean(chi_square_scores))
