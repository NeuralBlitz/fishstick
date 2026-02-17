"""
Manifold Regularization Layers and Loss Functions.

Manifold regularization incorporates geometric structure of data
into neural network training through:
- Laplace-Beltrami operator approximation
- Graph-based regularization
- Manifold-aware loss functions

This enables learning that respects intrinsic data geometry.

Based on: "Manifold Regularization: A Geometric Framework for Learning
from Labeled and Unlabeled Data" (Belkin et al., 2006)
"""

from typing import Optional, Tuple, List, Callable
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import (
    ManifoldLearnerBase,
    GraphBuilder,
    compute_distance_matrix,
    compute_gaussian_kernel,
)


class LaplaceBeltramiOperator:
    """
    Laplace-Beltrami operator on point cloud data.

    Approximates the Laplace-Beltrami operator Δ_M f using
    graph Laplacian: L = D - W

    where W is the weight matrix and D is the degree matrix.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        sigma: Optional[float] = None,
        normalized: bool = True,
    ):
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.normalized = normalized

    def compute_laplacian(self, X: Tensor) -> Tensor:
        """
        Compute graph Laplacian approximating Laplace-Beltrami operator.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            Graph Laplacian [n_samples, n_samples]
        """
        n = X.shape[0]

        X_np = X.cpu().numpy()
        nn_finder = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn_finder.fit(X_np)
        distances, indices = nn_finder.kneighbors(X_np)

        distances = distances[:, 1:]
        indices = indices[:, 1:]

        if self.sigma is None:
            sigma = distances.mean().item()
        else:
            sigma = self.sigma

        distances = torch.from_numpy(distances).float()
        weights = torch.exp(-distances.pow(2) / (2 * sigma**2))

        W = torch.zeros(n, n, device=X.device)

        for i in range(n):
            for j_idx, j in enumerate(indices[i]):
                W[i, j] = weights[i, j_idx]
                W[j, i] = weights[i, j_idx]

        D = torch.diag(W.sum(dim=1))

        L = D - W

        if self.normalized:
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(W.sum(dim=1) + 1e-10))
            L = D_inv_sqrt @ L @ D_inv_sqrt

        return L

    def compute_gradients(self, X: Tensor, f: Tensor) -> Tensor:
        """
        Compute gradients of function f on manifold.

        Approximates ∇_M f using graph structure.

        Args:
            X: Input data [n_samples, n_features]
            f: Function values [n_samples, ...]

        Returns:
            Gradient estimates [n_samples, n_features]
        """
        L = self.compute_laplacian(X)

        grad = L @ f

        return grad


class ManifoldRegularizationLoss(nn.Module):
    """
    Manifold regularization loss term.

    L_manifold = f^T * L * f

    Encourages smooth functions on the data manifold,
    where smoothness is measured by graph Laplacian.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        sigma: Optional[float] = None,
        normalized: bool = True,
    ):
        super().__init__()
        self.lb_operator = LaplaceBeltramiOperator(n_neighbors, sigma, normalized)
        self.laplacian_: Optional[Tensor] = None

    def forward(
        self,
        embeddings: Tensor,
        X: Optional[Tensor] = None,
        weight: float = 1.0,
    ) -> Tensor:
        """
        Compute manifold regularization loss.

        Args:
            embeddings: Feature embeddings [n_samples, dim]
            X: Original data points (if different from embeddings)
            weight: Weight of regularization term

        Returns:
            Regularization loss scalar
        """
        if X is None:
            X = embeddings

        L = self.lb_operator.compute_laplacian(X)
        self.laplacian_ = L

        loss = embeddings.T @ L @ embeddings

        return weight * torch.trace(loss)

    def get_laplacian(self) -> Optional[Tensor]:
        """Get computed Laplacian matrix."""
        return self.laplacian_


class GraphBasedRegularization(nn.Module):
    """
    Graph-based regularization for semi-supervised learning.

    Implements manifold regularization from:
    "Learning with Local and Global Consistency" (Zhou et al., 2004)
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        alpha: float = 0.99,
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.alpha = alpha

    def compute_label_propagation(
        self,
        X: Tensor,
        labels: Tensor,
        n_classes: int,
    ) -> Tensor:
        """
        Propagate labels through the manifold.

        Args:
            X: Input data [n_samples, n_features]
            labels: Known labels [n_samples] (use -1 for unknown)
            n_classes: Number of classes

        Returns:
            Propagated labels [n_samples, n_classes]
        """
        n = X.shape[0]

        X_np = X.cpu().numpy()
        nn_finder = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn_finder.fit(X_np)
        _, indices = nn_finder.kneighbors(X_np)

        W = torch.zeros(n, n, device=X.device)

        for i in range(n):
            for j in indices[i]:
                W[i, j] = 1.0
                W[j, i] = 1.0

        D = torch.diag(W.sum(dim=1))
        S = D @ W @ D.pinverse()

        F = torch.zeros(n, n_classes, device=X.device)

        known_mask = labels >= 0
        for i in torch.where(known_mask)[0]:
            F[i, labels[i]] = 1.0

        F = F + self.alpha * (S @ F)

        for _ in range(100):
            F_new = self.alpha * S @ F + (1 - self.alpha) * F
            if torch.allclose(F_new, F, atol=1e-6):
                break
            F = F_new

        return F


class ManifoldRegularizationLayer(nn.Module):
    """
    Neural network layer with manifold regularization.

    Adds manifold structure awareness to any layer through
    graph-based smoothness constraints.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_neighbors: int = 10,
        reg_weight: float = 0.1,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.n_neighbors = n_neighbors
        self.reg_weight = reg_weight
        self.lb_operator = LaplaceBeltramiOperator(n_neighbors)

    def forward(
        self,
        x: Tensor,
        return_manifold_loss: bool = False,
    ) -> Tensor:
        """
        Forward pass with optional manifold loss.

        Args:
            x: Input [batch, input_dim]
            return_manifold_loss: Return manifold regularization term

        Returns:
            Output [batch, output_dim] or (output, manifold_loss)
        """
        output = self.linear(x)

        if return_manifold_loss:
            L = self.lb_operator.compute_laplacian(x)
            manifold_loss = torch.trace(output.T @ L @ output)
            return output, manifold_loss * self.reg_weight

        return output


class IntrinsicDimensionEstimator:
    """
    Estimates intrinsic dimension of manifold.

    Uses local PCA and eigenvalue analysis to estimate
    the dimensionality of the underlying manifold.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        threshold: float = 0.95,
    ):
        self.n_neighbors = n_neighbors
        self.threshold = threshold

    def fit(self, X: Tensor) -> "IntrinsicDimensionEstimator":
        """
        Estimate intrinsic dimension.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            self
        """
        self.intrinsic_dim_ = self._estimate_dimension(X)
        return self

    def _estimate_dimension(self, X: Tensor) -> int:
        """Estimate using local PCA eigenvalue ratios."""
        n_samples = X.shape[0]

        X_np = X.cpu().numpy()
        nn_finder = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn_finder.fit(X_np)
        _, indices = nn_finder.kneighbors(X_np)

        dims = []

        for i in range(n_samples):
            neighbor_idx = indices[i, 1:]
            neighbors = X_np[neighbor_idx]
            x_i = X_np[i]

            centered = neighbors - x_i

            cov = centered.T @ centered / self.n_neighbors

            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]

            if len(eigenvalues) == 0:
                continue

            eigenvalues = eigenvalues[::-1]
            cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()

            dim = np.searchsorted(cumvar, self.threshold) + 1
            dims.append(dim)

        return int(np.median(dims))


class ManifoldMetric:
    """
    Computes intrinsic distances on learned manifold.

    Uses geodesics approximated via graph distances.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
    ):
        self.n_neighbors = n_neighbors

    def fit(self, X: Tensor) -> "ManifoldMetric":
        """
        Fit metric to data.

        Args:
            X: Input data [n_samples, n_features]
        """
        X_np = X.cpu().numpy()
        nn_finder = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn_finder.fit(X_np)
        distances, indices = nn_finder.kneighbors(X_np)

        self.indices_ = indices
        self.distances_ = torch.from_numpy(distances).float()

        return self

    def geodesic_distances(self, X: Tensor) -> Tensor:
        """
        Compute approximate geodesic distances.

        Uses shortest path on nearest neighbor graph.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        n = X.shape[0]

        distances = self.distances_.numpy()
        indices = self.indices_

        rows, cols, data = [], [], []

        for i in range(n):
            for j, d in zip(indices[i], distances[i]):
                rows.append(i)
                cols.append(j)
                data.append(d)

        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        graph = graph.maximum(graph.T)

        geo_dist = dijkstra(graph, directed=False)

        return torch.from_numpy(geo_dist).float()


class ManifoldSmoothingLayer(nn.Module):
    """
    Layer that smooths features along manifold.

    Applies graph Laplacian smoothing to features.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        smoothing_weight: float = 0.5,
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.smoothing_weight = smoothing_weight

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply manifold smoothing.

        Args:
            x: Input [batch, features]

        Returns:
            Smoothed output [batch, features]
        """
        n = x.shape[0]

        X_np = x.cpu().numpy()
        nn_finder = NearestNeighbors(n_neighbors=min(self.n_neighbors, n))
        nn_finder.fit(X_np)
        _, indices = nn_finder.kneighbors(X_np)

        smoothed = x.clone()

        for i in range(n):
            neighbors = x[indices[i]]
            smoothed[i] = (1 - self.smoothing_weight) * x[
                i
            ] + self.smoothing_weight * neighbors.mean(dim=0)

        return smoothed


class ManifoldAugmentation(nn.Module):
    """
    Manifold-aware data augmentation.

    Generates new samples by interpolating along the manifold.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        n_augmentations: int = 1,
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_augmentations = n_augmentations

    def forward(self, x: Tensor) -> Tensor:
        """
        Generate manifold-augmented samples.

        Args:
            x: Input [batch, features]

        Returns:
            Augmented samples [batch * (1 + n_augmentations), features]
        """
        n = x.shape[0]

        X_np = x.cpu().numpy()
        nn_finder = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn_finder.fit(X_np)
        _, indices = nn_finder.kneighbors(X_np)
        indices = indices[:, 1:]

        augmented = []

        for _ in range(self.n_augmentations):
            for i in range(n):
                neighbor_idx = indices[i, torch.randperm(self.n_neighbors)[:2]]

                alpha = torch.rand(1).item()
                new_point = (1 - alpha) * x[i] + alpha * x[neighbor_idx[0]]

                augmented.append(new_point)

        if augmented:
            augmented = torch.stack(augmented)
            return torch.cat([x, augmented], dim=0)

        return x


def create_manifold_regularizer(
    n_neighbors: int = 10,
    sigma: Optional[float] = None,
    normalized: bool = True,
) -> ManifoldRegularizationLoss:
    """
    Factory function to create manifold regularization loss.

    Args:
        n_neighbors: Number of neighbors for graph
        sigma: Kernel bandwidth
        normalized: Use normalized Laplacian

    Returns:
        Manifold regularization loss
    """
    return ManifoldRegularizationLoss(n_neighbors, sigma, normalized)


def manifold_loss(
    embeddings: Tensor,
    X: Optional[Tensor] = None,
    n_neighbors: int = 10,
    weight: float = 1.0,
) -> Tensor:
    """
    Compute manifold regularization loss.

    Helper function for quick use.

    Args:
        embeddings: Feature embeddings
        X: Original data
        n_neighbors: Number of neighbors
        weight: Regularization weight

    Returns:
        Loss scalar
    """
    reg = ManifoldRegularizationLoss(n_neighbors)
    return reg(embeddings, X, weight)
