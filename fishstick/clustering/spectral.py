"""
Spectral clustering algorithms.

Implements:
- Standard spectral clustering
- Normalized cuts (NCut)
- Ratio cuts (RCut)
- Graph Laplacian variants
- Kernel spectral clustering
- Self-tuning spectral clustering
"""

from typing import Optional, List, Tuple, Callable
import torch
from torch import Tensor
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors

from .base import (
    ClustererBase,
    ClusterResult,
    DistanceMetric,
    relabel_clusters,
)


@dataclass
class SpectralResult(ClusterResult):
    """Result of spectral clustering."""

    affinity_matrix: Optional[Tensor] = None
    eigenvalues: Optional[Tensor] = None
    eigenvectors: Optional[Tensor] = None


class SpectralClustering(ClustererBase):
    """
    Spectral clustering using graph Laplacian.

    Clusters data using eigenvalues of the graph Laplacian matrix.
    Effective for non-convex clusters.

    Args:
        n_clusters: Number of clusters
        affinity: Affinity type ('nearest_neighbors', 'rbf', 'polynomial', 'sigmoid')
        gamma: Kernel coefficient for rbf/polynomial/sigmoid
        n_neighbors: Number of neighbors for nearest_neighbors affinity
        degree: Degree for polynomial kernel
        coef0: Constant term for polynomial/sigmoid kernel
        n_components: Number of eigenvectors to use
        random_state: Random seed
    """

    def __init__(
        self,
        n_clusters: int = 8,
        affinity: str = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        degree: float = 3.0,
        coef0: float = 1.0,
        n_components: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_clusters, random_state)
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components or n_clusters

    def fit(self, X: Tensor) -> "SpectralClustering":
        """Fit spectral clustering to data."""
        from sklearn.cluster import SpectralClustering as SklearnSpectral

        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()

        spectral = SklearnSpectral(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            gamma=self.gamma,
            n_neighbors=self.n_neighbors,
            degree=self.degree,
            coef0=self.coef0,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        spectral.fit(X_np)

        self.labels_ = torch.from_numpy(spectral.labels_).to(X.device)
        self.affinity_matrix_ = (
            torch.from_numpy(spectral.affinity_matrix_.toarray())
            if sparse.issparse(spectral.affinity_matrix_)
            else torch.from_numpy(spectral.affinity_matrix_)
            if hasattr(spectral, "affinity_matrix_")
            else None
        )
        if self.affinity_matrix_ is not None and X.is_cuda:
            self.affinity_matrix_ = self.affinity_matrix_.cuda()

        return self

    def fit_predict(self, X: Tensor) -> Tensor:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class NormalizedCutSpectral(ClustererBase):
    """
    Spectral clustering with normalized cuts.

    Uses normalized Laplacian for better cluster separation.

    Args:
        n_clusters: Number of clusters
        n_neighbors: Number of neighbors for affinity
        sigma: Sigma parameter for Gaussian kernel
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_neighbors: int = 10,
        sigma: float = 1.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_clusters, random_state)
        self.n_neighbors = n_neighbors
        self.sigma = sigma

    def fit(self, X: Tensor) -> "NormalizedCutSpectral":
        """Fit normalized cut spectral clustering."""
        n_samples = X.shape[0]
        device = X.device

        W = self._build_affinity_matrix(X)
        D = torch.sum(W, dim=1)
        D_inv_sqrt = D ** (-0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0

        D_inv_sqrt_diag = torch.diag(D_inv_sqrt)
        L_norm = torch.eye(n_samples, device=device) - D_inv_sqrt @ W @ D_inv_sqrt_diag

        eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
        eigenvectors = eigenvectors[:, : self.n_clusters]

        embeddings = eigenvectors / (
            torch.norm(eigenvectors, dim=1, keepdim=True) + 1e-10
        )

        from .kmeans import KMeans

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit(embeddings)
        self.labels_ = kmeans.labels_
        self.embeddings_ = embeddings
        self.affinity_matrix_ = W

        return self

    def _build_affinity_matrix(self, X: Tensor) -> Tensor:
        """Build affinity matrix using k-NN."""
        n_samples = X.shape[0]
        device = X.device
        X_np = X.cpu().numpy()

        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn.fit(X_np)
        distances, indices = nn.kneighbors(X_np)

        distances = distances[:, 1:]
        indices = indices[:, 1:]

        W = torch.zeros((n_samples, n_samples), device=device)
        for i in range(n_samples):
            for j, dist in zip(indices[i], distances[i]):
                W[i, j] = torch.exp(-(dist**2) / (2 * self.sigma**2))

        return (W + W.T) / 2


class RatioCutSpectral(ClustererBase):
    """
    Spectral clustering with ratio cuts.

    Uses unnormalized graph Laplacian.

    Args:
        n_clusters: Number of clusters
        n_neighbors: Number of neighbors
        sigma: Sigma for Gaussian kernel
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_neighbors: int = 10,
        sigma: float = 1.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_clusters, random_state)
        self.n_neighbors = n_neighbors
        self.sigma = sigma

    def fit(self, X: Tensor) -> "RatioCutSpectral":
        """Fit ratio cut spectral clustering."""
        n_samples = X.shape[0]
        device = X.device

        W = self._build_affinity_matrix(X)
        D = torch.sum(W, dim=1)
        D_diag = torch.diag(D)
        L = D_diag - W

        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        eigenvectors = eigenvectors[:, : self.n_clusters]

        from .kmeans import KMeans

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit(eigenvectors)
        self.labels_ = kmeans.labels_
        self.eigenvectors_ = eigenvectors

        return self

    def _build_affinity_matrix(self, X: Tensor) -> Tensor:
        """Build affinity matrix using k-NN."""
        n_samples = X.shape[0]
        device = X.device
        X_np = X.cpu().numpy()

        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn.fit(X_np)
        distances, indices = nn.kneighbors(X_np)

        distances = distances[:, 1:]
        indices = indices[:, 1:]

        W = torch.zeros((n_samples, n_samples), device=device)
        for i in range(n_samples):
            for j, dist in zip(indices[i], distances[i]):
                W[i, j] = torch.exp(-(dist**2) / (2 * self.sigma**2))

        return (W + W.T) / 2


class KernelSpectralClustering(ClustererBase):
    """
    Kernel spectral clustering.

    Performs spectral clustering in kernel-induced feature space.

    Args:
        n_clusters: Number of clusters
        kernel: Kernel type ('rbf', 'polynomial', 'sigmoid')
        gamma: RBF kernel parameter
        degree: Polynomial kernel degree
        coef0: Sigmoid/polynomial constant
    """

    def __init__(
        self,
        n_clusters: int = 8,
        kernel: str = "rbf",
        gamma: float = 1.0,
        degree: float = 3.0,
        coef0: float = 1.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_clusters, random_state)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X: Tensor) -> "KernelSpectralClustering":
        """Fit kernel spectral clustering."""
        from sklearn.kernel_approximation import Nystroem
        from sklearn.cluster import KMeans as SklearnKMeans

        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        n_samples = X_np.shape[0]

        if self.kernel == "rbf":
            K = np.exp(-self.gamma * pairwise_distances(X_np) ** 2)
        elif self.kernel == "polynomial":
            K = (self.gamma * X_np @ X_np.T + self.coef0) ** self.degree
        elif self.kernel == "sigmoid":
            K = np.tanh(self.gamma * X_np @ X_np.T + self.coef0)
        else:
            K = np.exp(-self.gamma * pairwise_distances(X_np) ** 2)

        K = torch.from_numpy(K).float()
        if X.is_cuda:
            K = K.cuda()

        D = torch.sum(K, dim=1)
        D_inv_sqrt = D ** (-0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0

        K_norm = D_inv_sqrt.unsqueeze(1) * K * D_inv_sqrt.unsqueeze(0)

        eigenvalues, eigenvectors = torch.linalg.eigh(K_norm)
        eigenvectors = eigenvectors[:, : self.n_clusters]

        skmeans = SklearnKMeans(
            n_clusters=self.n_clusters, random_state=self.random_state
        )
        skmeans.fit(eigenvectors.numpy())
        self.labels_ = torch.from_numpy(skmeans.labels_).to(X.device)

        return self


class SelfTuningSpectral(ClustererBase):
    """
    Self-tuning spectral clustering.

    Automatically adapts sigma for each point based on local density.

    Args:
        n_clusters: Number of clusters
        n_neighbors: Number of neighbors
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_neighbors: int = 10,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_clusters, random_state)
        self.n_neighbors = n_neighbors

    def fit(self, X: Tensor) -> "SelfTuningSpectral":
        """Fit self-tuning spectral clustering."""
        n_samples = X.shape[0]
        device = X.device

        W = self._build_self_tuning_affinity(X)
        D = torch.sum(W, dim=1) + 1e-10
        D_inv_sqrt = D ** (-0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0

        L_norm = torch.eye(n_samples, device=device) - D_inv_sqrt.unsqueeze(
            1
        ) * W * D_inv_sqrt.unsqueeze(0)

        eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
        eigenvectors = eigenvectors[:, 1 : self.n_clusters + 1]

        from .kmeans import KMeans

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans.fit(eigenvectors)
        self.labels_ = kmeans.labels_
        self.affinity_matrix_ = W

        return self

    def _build_self_tuning_affinity(self, X: Tensor) -> Tensor:
        """Build self-tuning affinity matrix."""
        n_samples = X.shape[0]
        device = X.device
        X_np = X.cpu().numpy()

        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn.fit(X_np)
        distances, indices = nn.kneighbors(X_np)

        distances = distances[:, 1:]
        sigma = distances.mean(dim=1, keepdim=True)

        W = torch.zeros((n_samples, n_samples), device=device)
        for i in range(n_samples):
            for j, dist in zip(indices[i], distances[i]):
                W[i, j] = torch.exp(-(dist**2) / (2 * sigma[i] * sigma[j]))

        return (W + W.T) / 2


class GraphLaplacian:
    """Compute various graph Laplacian matrices."""

    @staticmethod
    def combinatorial(X: Tensor, W: Tensor) -> Tensor:
        """Combinatorial Laplacian L = D - W."""
        D = torch.diag(W.sum(dim=1))
        return D - W

    @staticmethod
    def normalized(X: Tensor, W: Tensor) -> Tensor:
        """Normalized Laplacian L_norm = I - D^(-1/2) W D^(-1/2)."""
        D = W.sum(dim=1)
        D_inv_sqrt = D ** (-0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        D_inv_sqrt_diag = torch.diag(D_inv_sqrt)
        I = torch.eye(W.shape[0], device=W.device)
        return I - D_inv_sqrt_diag @ W @ D_inv_sqrt_diag

    @staticmethod
    def random_walk(X: Tensor, W: Tensor) -> Tensor:
        """Random walk Laplacian L_rw = I - D^(-1) W."""
        D = W.sum(dim=1)
        D[D == 0] = 1
        D_inv = torch.diag(1.0 / D)
        I = torch.eye(W.shape[0], device=W.device)
        return I - D_inv @ W


def create_spectral_clustering(
    n_clusters: int = 8,
    affinity: str = "rbf",
    **kwargs,
) -> SpectralClustering:
    """Factory function for spectral clustering."""
    return SpectralClustering(n_clusters=n_clusters, affinity=affinity, **kwargs)


def create_normalized_spectral(
    n_clusters: int = 8,
    **kwargs,
) -> NormalizedCutSpectral:
    """Factory function for normalized cut spectral clustering."""
    return NormalizedCutSpectral(n_clusters=n_clusters, **kwargs)


def pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distances."""
    from sklearn.metrics import pairwise_distances as sk_pd

    return sk_pd(X, metric=metric)
