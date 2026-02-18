"""Single-cell clustering and dimensionality reduction."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class PCAEmbedder(nn.Module):
    """PCA dimensionality reduction for single-cell data.

    Attributes:
        n_components: Number of principal components
    """

    def __init__(self, n_components: int = 50) -> None:
        super().__init__()
        self.n_components = n_components

        self.mean: Optional[Tensor] = None
        self.components: Optional[Tensor] = None

    def fit(self, data: Tensor) -> "PCAEmbedder":
        """Fit PCA on data.

        Args:
            data: Expression matrix (cells x features)

        Returns:
            Self
        """
        self.mean = data.mean(dim=0)
        centered = data - self.mean

        cov = centered.T @ centered / (data.shape[0] - 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components = eigenvectors[:, : self.n_components]
        self.explained_variance = eigenvalues[: self.n_components]

        return self

    def forward(self, data: Tensor) -> Tensor:
        """Project data to PCA space.

        Args:
            data: Expression matrix

        Returns:
            PCA embedding
        """
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")

        centered = data - self.mean
        return centered @ self.components


class UMAPProjector(nn.Module):
    """UMAP projection for single-cell data.

    A simplified UMAP implementation using gradient descent.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def forward(
        self,
        data: Tensor,
        epochs: int = 100,
    ) -> Tensor:
        """Project data using UMAP.

        Args:
            data: Input data (cells x features)
            epochs: Number of optimization epochs

        Returns:
            UMAP embedding
        """
        from scipy.spatial.distance import cdist

        data_np = data.numpy()

        distances = cdist(data_np, data_np)

        knn_idx = np.argsort(distances, axis=1)[:, : self.n_neighbors]

        P = np.zeros_like(distances)
        for i in range(len(data_np)):
            for j in knn_idx[i]:
                P[i, j] = np.exp(-distances[i, j])

        P = (P + P.T) / (2 * P.shape[0])

        embedding = np.random.randn(len(data_np), self.n_components) * 0.01

        for epoch in range(epochs):
            distances_emb = cdist(embedding, embedding)
            np.fill_diagonal(distances_emb, 1)

            Q = 1.0 / (1.0 + distances_emb**2)
            np.fill_diagonal(Q, 0)
            Q = Q / Q.sum()

            grad = (
                4
                * np.sum((P - Q**2) * Q * distances_emb)
                * (embedding[:, None, :] - embedding[None, :, :])
            )

            learning_rate = 0.1 * (1 - epoch / epochs)
            embedding -= learning_rate * grad.mean(axis=1)

        return torch.tensor(embedding, dtype=torch.float32)


class SingleCellClustering(nn.Module):
    """Clustering for single-cell data.

    Supports multiple clustering methods: K-means, Leiden, Louvain.

    Attributes:
        n_clusters: Number of clusters
        method: Clustering method
    """

    def __init__(
        self,
        n_clusters: int = 10,
        method: str = "kmeans",
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.method = method

    def forward(self, embedding: Tensor) -> Tensor:
        """Cluster cells.

        Args:
            embedding: Cell embeddings

        Returns:
            Cluster labels
        """
        if self.method == "kmeans":
            return self._kmeans(embedding)
        elif self.method == "leiden":
            return self._leiden(embedding)
        else:
            return self._kmeans(embedding)

    def _kmeans(self, embedding: Tensor) -> Tensor:
        """K-means clustering.

        Args:
            embedding: Cell embeddings

        Returns:
            Cluster labels
        """
        data = embedding.numpy()

        centroids = data[np.random.choice(len(data), self.n_clusters, replace=False)]

        for _ in range(50):
            distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            for k in range(self.n_clusters):
                if np.sum(labels == k) > 0:
                    centroids[k] = data[labels == k].mean(axis=0)

        return torch.tensor(labels, dtype=torch.long)

    def _leiden(self, embedding: Tensor) -> Tensor:
        """Leiden clustering (simplified version).

        Args:
            embedding: Cell embeddings

        Returns:
            Cluster labels
        """
        from sklearn.neighbors import kneighbors_graph
        from scipy.sparse.csgraph import connected_components

        data = embedding.numpy()

        knn_graph = kneighbors_graph(data, n_neighbors=15, mode="connectivity")
        knn_graph = knn_graph + knn_graph.T
        knn_graph = (knn_graph > 0).astype(float)

        n_components, labels = connected_components(knn_graph, directed=False)

        if n_components > self.n_clusters:
            return torch.tensor(labels[: self.n_clusters], dtype=torch.long)

        return torch.tensor(labels, dtype=torch.long)


def compute_silhouette_score(
    embedding: Tensor,
    labels: Tensor,
) -> float:
    """Compute silhouette score for clustering quality.

    Args:
        embedding: Cell embeddings
        labels: Cluster labels

    Returns:
        Silhouette score
    """
    from sklearn.metrics import silhouette_score

    return silhouette_score(embedding.numpy(), labels.numpy())


def compute_cluster_markers(
    expression: Tensor,
    labels: Tensor,
    n_markers: int = 10,
) -> Dict[int, List[int]]:
    """Find marker genes for each cluster.

    Args:
        expression: Expression matrix (cells x genes)
        labels: Cluster labels
        n_markers: Number of markers per cluster

    Returns:
        Dictionary mapping cluster to marker gene indices
    """
    unique_labels = torch.unique(labels)

    markers = {}

    for label in unique_labels:
        cluster_cells = expression[labels == label]
        other_cells = expression[labels != label]

        cluster_mean = cluster_cells.mean(dim=0)
        other_mean = other_cells.mean(dim=0)

        fold_change = cluster_mean / (other_mean + 1e-8)

        top_markers = torch.argsort(fold_change, descending=True)[:n_markers]

        markers[label.item()] = top_markers.tolist()

    return markers
