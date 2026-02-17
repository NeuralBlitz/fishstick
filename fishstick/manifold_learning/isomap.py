"""
Isomap: Isometric Mapping for Manifold Learning.

Isomap computes geodesic distances along the manifold surface and
embeds the data using Multidimensional Scaling (MDS).

Based on: "A Global Geometric Framework for Nonlinear Dimensionality Reduction"
(Tenenbaum et al., Science 2000)

Key steps:
1. Construct k-nearest neighbor graph
2. Compute shortest path distances (geodesic approximation)
3. Apply MDS to recover embedding
"""

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .base import (
    ManifoldLearnerBase,
    GraphBuilder,
    compute_distance_matrix,
    EigenSolver,
)


class Isomap(ManifoldLearnerBase):
    """
    Isomap: Isometric Mapping.

    Finds low-dimensional representation that preserves geodesic
    distances along the manifold.

    Args:
        n_components: Dimension of embedding space
        n_neighbors: Number of neighbors for graph construction
        reg: Regularization parameter for matrix inversion
        metric: Distance metric ('euclidean', 'cosine', etc.)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        reg: float = 1e-3,
        metric: str = "euclidean",
    ):
        super().__init__(n_components, n_neighbors, reg)
        self.metric = metric
        self.graph_builder = None
        self.geodesic_distances_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "Isomap":
        """
        Fit Isomap to data.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            self
        """
        n_samples = X.shape[0]

        self.graph_builder = GraphBuilder(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            method="knn",
        )
        self.graph_builder.fit(X)

        adj_matrix = self.graph_builder.adjacency_matrix(X, symmetric=True)

        self.geodesic_distances_ = self._compute_geodesic_distances(
            adj_matrix, n_samples
        )

        self.embedding_ = self._mds_embedding(self.geodesic_distances_, n_samples)

        return self

    def _compute_geodesic_distances(
        self, adj_matrix: csr_matrix, n_samples: int
    ) -> Tensor:
        """
        Compute geodesic distances using Floyd-Warshall or Dijkstra.

        Uses Dijkstra for efficiency on sparse graphs.
        """
        distances = dijkstra(
            adj_matrix,
            directed=False,
            return_predecessors=False,
        )

        distances = np.nan_to_num(distances, nan=np.inf)

        return torch.from_numpy(distances).float()

    def _mds_embedding(self, distances: Tensor, n_samples: int) -> Tensor:
        """
        Apply Classical MDS to recover embedding.

        B = -0.5 * J * D^2 * J
        where J = I - (1/n) * 1*1^T

        Then eigendecomposition: B = V * Lambda * V^T
        """
        device = distances.device
        distances = distances.cpu().numpy()
        n = distances.shape[0]

        J = np.eye(n) - np.ones((n, n)) / n

        B = -0.5 * J @ (distances**2) @ J

        eigenvalues, eigenvectors = np.linalg.eigh(B)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx[: self.n_components]]
        eigenvectors = eigenvectors[:, idx[: self.n_components]]

        eigenvalues = np.maximum(eigenvalues, 0)

        embedding = eigenvectors * np.sqrt(eigenvalues)

        return torch.from_numpy(embedding).float().to(device)

    def reconstruction_error(self) -> float:
        """
        Compute residual variance as reconstruction error.

        Returns:
            Residual variance (1 - correlation between distances)
        """
        if self.geodesic_distances_ is None:
            raise RuntimeError("Model not fitted.")

        geo_dist = self.geodesic_distances_.numpy()
        embed_dist = compute_distance_matrix(self.embedding_).numpy()

        mask = (geo_dist > 0) & (geo_dist < np.inf)

        if mask.sum() == 0:
            return 0.0

        r = np.corrcoef(geo_dist[mask], embed_dist[mask])[0, 1]
        return 1 - r**2


class LandmarkIsomap(Isomap):
    """
    Landmark Isomap for large datasets.

    Uses landmark points to compute the embedding efficiently,
    reducing complexity from O(n^3) to O(n*k^2 + k^3).

    Args:
        n_landmarks: Number of landmark points (default: n_samples / 10)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        reg: float = 1e-3,
        n_landmarks: Optional[int] = None,
    ):
        super().__init__(n_components, n_neighbors, reg)
        self.n_landmarks = n_landmarks
        self.landmark_indices_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "LandmarkIsomap":
        """
        Fit Landmark Isomap to data.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            self
        """
        n_samples = X.shape[0]

        if self.n_landmarks is None:
            self.n_landmarks = max(10, n_samples // 10)

        self.n_landmarks = min(self.n_landmarks, n_samples)

        indices = torch.randperm(n_samples)[: self.n_landmarks]
        self.landmark_indices_ = indices
        X_landmarks = X[indices]

        self.graph_builder = GraphBuilder(
            n_neighbors=self.n_neighbors,
            method="knn",
        )
        self.graph_builder.fit(X_landmarks)

        adj_matrix = self.graph_builder.adjacency_matrix(X_landmarks, symmetric=True)

        landmark_distances = self._compute_geodesic_distances(
            adj_matrix, self.n_landmarks
        )

        self.embedding_ = self._landmark_mds(landmark_distances, X, indices, n_samples)

        return self

    def _landmark_mds(
        self,
        landmark_distances: Tensor,
        X: Tensor,
        landmark_indices: Tensor,
        n_samples: int,
    ) -> Tensor:
        """
        Compute embedding using landmark MDS.

        1. Compute embedding for landmarks
        2. Use triangulation for remaining points
        """
        device = X.device

        landmark_distances_np = landmark_distances.cpu().numpy()
        n_landmarks = self.n_landmarks

        J = np.eye(n_landmarks) - np.ones((n_landmarks, n_landmarks)) / n_landmarks
        B = -0.5 * J @ (landmark_distances_np**2) @ J

        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[idx[: self.n_components]], 0)
        eigenvectors = eigenvectors[:, idx[: self.n_components]]

        landmark_embedding = eigenvectors * np.sqrt(eigenvalues)

        landmark_embedding = torch.from_numpy(landmark_embedding).float().to(device)

        X_np = X.cpu().numpy()
        landmark_points = X_np[landmark_indices.cpu().numpy()]

        from sklearn.neighbors import NearestNeighbors

        nn_finder = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn_finder.fit(landmark_points)

        full_distances, neighbors = nn_finder.kneighbors(X_np)

        embedding = torch.zeros(n_samples, self.n_components, device=device)

        for i in range(n_samples):
            neighbor_embed = landmark_embedding[neighbors[i]]
            neighbor_dist = torch.from_numpy(full_distances[i]).float().to(device)

            weights = torch.exp(-neighbor_dist / neighbor_dist.mean())
            weights = weights / weights.sum()

            embedding[i] = (neighbor_embed * weights.unsqueeze(-1)).sum(dim=0)

        return embedding


class IsomapLayer(nn.Module):
    """
    Neural network layer that applies Isomap embedding.

    Uses differentiable operations where possible for integration
    into end-to-end trainable models.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
    ):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.isomap = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Isomap embedding.

        Args:
            x: Input [batch, features]

        Returns:
            Embedding [batch, n_components]
        """
        if self.isomap is None or self.training:
            isomap = Isomap(
                n_components=self.n_components,
                n_neighbors=min(self.n_neighbors, x.shape[0] - 1),
            )
            embedding = isomap.fit_transform(x)
        else:
            embedding = self.isomap.transform(x)

        return embedding

    def fit(self, x: Tensor) -> Tensor:
        """
        Fit Isomap and return embedding.

        Args:
            x: Input [batch, features]

        Returns:
            Embedding [batch, n_components]
        """
        self.isomap = Isomap(
            n_components=self.n_components,
            n_neighbors=min(self.n_neighbors, x.shape[0] - 1),
        )
        return self.isomap.fit_transform(x)


def create_isomap(
    n_components: int = 2,
    n_neighbors: int = 10,
    landmark: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create Isomap model.

    Args:
        n_components: Embedding dimension
        n_neighbors: Number of neighbors
        landmark: Use landmark Isomap for large datasets
        **kwargs: Additional arguments

    Returns:
        Isomap model
    """
    if landmark:
        return LandmarkIsomap(n_components, n_neighbors, **kwargs)
    return Isomap(n_components, n_neighbors, **kwargs)
