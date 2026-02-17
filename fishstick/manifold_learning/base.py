"""
Base classes and utilities for manifold learning.

Provides common infrastructure for:
- Graph construction
- Kernel computation
- Eigenvalue solvers
- Embedding base class
"""

from typing import Optional, Tuple, Callable, List
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


@dataclass
class ManifoldEmbedding:
    """Result of a manifold learning algorithm."""

    embedding: Tensor
    reconstruction_error: Optional[float] = None
    eigenvalues: Optional[Tensor] = None
    n_neighbors: int = 0
    method: str = ""


class GraphBuilder:
    """
    Constructs neighborhood graphs for manifold learning.

    Supports k-nearest neighbors and ε-ball neighborhood selection.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        metric: str = "euclidean",
        method: str = "knn",
        epsilon: Optional[float] = None,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.method = method
        self.epsilon = epsilon

    def fit(self, X: Tensor) -> "GraphBuilder":
        """
        Fit the graph builder to data.

        Args:
            X: Input data [n_samples, n_features]
        """
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()

        if self.method == "knn":
            self.nn_ = NearestNeighbors(
                n_neighbors=self.n_neighbors + 1,
                metric=self.metric,
            )
            self.nn_.fit(X_np)
        elif self.method == "ball":
            self.nn_ = NearestNeighbors(
                n_neighbors=X_np.shape[0],
                radius=self.epsilon,
                metric=self.metric,
            )
            self.nn_.fit(X_np)

        return self

    def adjacency_matrix(self, X: Tensor, symmetric: bool = True) -> csr_matrix:
        """
        Compute adjacency matrix of the neighborhood graph.

        Args:
            X: Input data [n_samples, n_features]
            symmetric: If True, make graph symmetric

        Returns:
            Sparse adjacency matrix
        """
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        n = X_np.shape[0]

        if self.method == "knn":
            distances, indices = self.nn_.kneighbors(X_np)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        else:
            distances, indices = self.nn_.radius_neighbors(X_np)

        rows, cols, data = [], [], []

        for i in range(n):
            for j, dist in zip(indices[i], distances[i]):
                if i != j:
                    rows.append(i)
                    cols.append(j)
                    data.append(dist)

        adj = csr_matrix((data, (rows, cols)), shape=(n, n))

        if symmetric:
            adj = adj.maximum(adj.T)

        return adj

    def adjacency_matrix_torch(
        self, X: Tensor, symmetric: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute sparse adjacency matrix as torch tensors.

        Returns:
            Tuple of (indices, values) for sparse tensor construction
        """
        adj = self.adjacency_matrix(X, symmetric)
        indices = torch.from_numpy(np.array(adj.nonzero()).astype(np.int64))
        values = torch.from_numpy(adj.data.astype(np.float32))

        if X.is_cuda:
            indices = indices.cuda()
            values = values.cuda()

        return indices, values


class KernelBuilder:
    """
    Constructs kernel matrices for manifold learning.

    Supports various kernel functions:
    - Gaussian (RBF)
    - Polynomial
    - Cosine
    """

    def __init__(
        self,
        kernel_type: str = "gaussian",
        gamma: Optional[float] = None,
        n_neighbors: int = 10,
    ):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.n_neighbors = n_neighbors

    def gaussian(self, distances: Tensor, gamma: Optional[float] = None) -> Tensor:
        """Gaussian (RBF) kernel: K(x,y) = exp(-gamma * ||x-y||^2)"""
        gamma = gamma or self.gamma
        if gamma is None:
            gamma = 1.0 / distances.mean().item()
        return torch.exp(-gamma * distances.pow(2))

    def polynomial(
        self, distances: Tensor, degree: int = 3, coef0: float = 1.0
    ) -> Tensor:
        """Polynomial kernel: K(x,y) = (gamma * <x,y> + coef0)^degree"""
        gamma = self.gamma or 1.0
        return (gamma * (-distances.pow(2)) + coef0).pow(degree)

    def cosine(self, distances: Tensor) -> Tensor:
        """Cosine kernel based on distance."""
        return 1.0 / (1.0 + distances)

    def compute_kernel_matrix(
        self, X: Tensor, indices: Tensor, distances: Tensor
    ) -> Tensor:
        """
        Compute kernel matrix from precomputed distances.

        Args:
            X: Input data [n_samples, n_features]
            indices: Neighbor indices [n_samples, n_neighbors]
            distances: Neighbor distances [n_samples, n_neighbors]

        Returns:
            Kernel matrix [n_samples, n_samples]
        """
        n = X.shape[0]

        if self.kernel_type == "gaussian":
            kernel_fn = self.gaussian
        elif self.kernel_type == "polynomial":
            kernel_fn = self.polynomial
        elif self.kernel_type == "cosine":
            kernel_fn = self.cosine
        else:
            kernel_fn = self.gaussian

        kernel_matrix = torch.zeros(n, n, device=X.device, dtype=X.dtype)

        for i in range(n):
            for j_idx, j in enumerate(indices[i]):
                kernel_matrix[i, j] = kernel_fn(distances[i, j_idx : j_idx + 1])

        return kernel_matrix


class EigenSolver:
    """
    Solves eigenvalue problems for manifold learning.

    Supports:
    - Standard eigendecomposition
    - Sparse eigenvalue solvers
    - Thick-restart Lanczos
    """

    def __init__(
        self,
        which: str = "LM",
        k: int = 10,
        sigma: Optional[float] = None,
    ):
        self.which = which
        self.k = k
        self.sigma = sigma

    def solve_dense(
        self, matrix: Tensor, k: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Solve eigenvalue problem for dense matrix.

        Args:
            matrix: Square matrix [n, n]
            k: Number of eigenvalues to compute

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        k = k or self.k
        matrix_np = matrix.cpu().numpy()

        eigenvalues, eigenvectors = np.linalg.eigh(matrix_np)

        if self.which == "LM":
            idx = np.argsort(np.abs(eigenvalues))[::-1]
        elif self.which == "SM":
            idx = np.argsort(np.abs(eigenvalues))
        else:
            idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idx[:k]]
        eigenvectors = eigenvectors[:, idx[:k]]

        eigenvalues = torch.from_numpy(eigenvalues).to(matrix.device)
        eigenvectors = torch.from_numpy(eigenvectors).to(matrix.device)

        return eigenvalues, eigenvectors

    def solve_sparse(
        self, matrix: csr_matrix, k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve eigenvalue problem for sparse matrix.

        Args:
            matrix: Sparse matrix
            k: Number of eigenvalues

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        k = k or self.k

        eigenvalues, eigenvectors = eigsh(
            matrix,
            k=k,
            which=self.which,
            sigma=self.sigma,
        )

        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]


class ManifoldLearnerBase(nn.Module):
    """
    Base class for manifold learning algorithms.

    Provides common functionality for:
    - Data fitting
    - Neighbor finding
    - Embedding computation
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        reg: float = 1e-3,
    ):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.embedding_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "ManifoldLearnerBase":
        """
        Fit the manifold learning model.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            self
        """
        raise NotImplementedError

    def transform(self, X: Tensor) -> Tensor:
        """
        Transform data to embedding space.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            Embedding [n_samples, n_components]
        """
        if self.embedding_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.embedding_

    def fit_transform(self, X: Tensor) -> Tensor:
        """
        Fit and transform in one call.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            Embedding [n_samples, n_components]
        """
        self.fit(X)
        return self.embedding_


def compute_distance_matrix(X: Tensor) -> Tensor:
    """
    Compute pairwise Euclidean distance matrix.

    Args:
        X: Input data [n_samples, n_features]

    Returns:
        Distance matrix [n_samples, n_samples]
    """
    norms = torch.sum(X.pow(2), dim=1, keepdim=True)
    distances = norms + norms.T - 2 * X @ X.T
    distances = torch.clamp(distances, min=0.0)
    return distances.sqrt()


def compute_gaussian_kernel(X: Tensor, sigma: Optional[float] = None) -> Tensor:
    """
    Compute Gaussian kernel matrix.

    Args:
        X: Input data [n_samples, n_features]
        sigma: Kernel bandwidth (auto-computed if None)

    Returns:
        Kernel matrix [n_samples, n_samples]
    """
    distances = compute_distance_matrix(X)

    if sigma is None:
        sigma = distances[distances > 0].mean().item()

    return torch.exp(-distances.pow(2) / (2 * sigma**2))


def local_pca(X: Tensor, neighbors: Tensor, n_components: int) -> Tuple[Tensor, Tensor]:
    """
    Perform local PCA at each point.

    Args:
        X: Input data [n_samples, n_features]
        neighbors: Neighbor indices [n_samples, n_neighbors]
        n_components: Number of principal components

    Returns:
        Tuple of (local means, local principal components)
    """
    n_samples = X.shape[0]
    means = torch.zeros(n_samples, X.shape[1], device=X.device, dtype=X.dtype)
    components = torch.zeros(
        n_samples, n_components, X.shape[1], device=X.device, dtype=X.dtype
    )

    for i in range(n_samples):
        neighbor_data = X[neighbors[i]]
        mean_i = neighbor_data.mean(dim=0)
        means[i] = mean_i

        centered = neighbor_data - mean_i
        cov = centered.T @ centered / (centered.shape[0] - 1)

        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        k = min(n_components, len(eigvals))
        if k > 0:
            components[i, :k] = eigvecs[:, -k:]

    return means, components
