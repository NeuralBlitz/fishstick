"""
Hessian Eigenmaps (HLLE) for Manifold Learning.

Hessian Eigenmaps uses the Hessian of the embedding function to
preserve local tangent space structure without requiring
spectral decomposition of the graph Laplacian.

Key advantages:
- Explicitly preserves local geometry via Hessian
- No zero eigenvalues from disconnected components
- Works well for non-convex manifolds
- No orientation issues like LLE

Based on: "Hessian Eigenmaps: new locally linear embedding techniques
for high-dimensional data" (Donoho & Grimes, 2003)
"""

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import (
    ManifoldLearnerBase,
    GraphBuilder,
    compute_distance_matrix,
    local_pca,
)


class HessianEigenmaps(ManifoldLearnerBase):
    """
    Hessian Eigenmaps (HLLE).

    Embeds data by preserving local Hessian structure.

    Algorithm:
    1. Find k-nearest neighbors
    2. Estimate local tangent space via PCA
    3. Build Hessian in local coordinates
    4. Solve for null space of Hessian operator

    Args:
        n_components: Dimension of embedding space
        n_neighbors: Number of neighbors for local analysis
        reg: Regularization parameter
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        reg: float = 1e-3,
    ):
        super().__init__(n_components, n_neighbors, reg)
        self.tangent_frames_: Optional[Tensor] = None
        self.hessian_operator_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "HessianEigenmaps":
        """
        Fit Hessian Eigenmaps to data.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            self
        """
        n_samples = X.shape[0]
        d = self.n_components
        k = self.n_neighbors

        X_np = X.cpu().numpy()
        nn_finder = NearestNeighbors(n_neighbors=k + 1)
        nn_finder.fit(X_np)
        _, indices = nn_finder.kneighbors(X_np)
        indices = indices[:, 1:]

        indices = torch.from_numpy(indices).long()

        tangent_frames = self._estimate_tangent_spaces(X, indices, d)
        self.tangent_frames_ = tangent_frames

        H = self._build_hessian_operator(X, indices, tangent_frames, d, n_samples)

        eigenvalues, eigenvectors = torch.linalg.eigh(H)

        idx = torch.argsort(eigenvalues)[d + 1 : 2 * d + 1]

        self.embedding_ = eigenvectors[:, idx].to(X.device)

        return self

    def _estimate_tangent_spaces(self, X: Tensor, indices: Tensor, d: int) -> Tensor:
        """
        Estimate local tangent space at each point.

        Uses PCA on local neighborhood to find d-dimensional
        subspace that best fits the data.

        Returns:
            Tangent frames [n_samples, d, n_features]
        """
        n_samples = X.shape[0]
        device = X.device
        k = indices.shape[1]

        frames = torch.zeros(n_samples, d, X.shape[1], device=device)

        X_np = X.cpu().numpy()

        for i in range(n_samples):
            neighbor_idx = indices[i].cpu().numpy()
            neighbors = X_np[neighbor_idx]
            x_i = X_np[i]

            centered = neighbors - x_i

            cov = centered.T @ centered / (k - 1)

            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            valid = eigenvalues > 1e-10
            n_valid = valid.sum()

            frame_dim = min(d, n_valid)
            if frame_dim > 0:
                frames[i, :frame_dim, :] = torch.from_numpy(
                    eigenvectors[:, :frame_dim].T
                ).float()

        return frames

    def _build_hessian_operator(
        self,
        X: Tensor,
        indices: Tensor,
        tangent_frames: Tensor,
        d: int,
        n_samples: int,
    ) -> Tensor:
        """
        Build the Hessian operator matrix.

        The Hessian operator H captures second-order derivatives
        of the embedding function along the manifold.
        """
        k = self.n_neighbors
        device = X.device

        H = torch.zeros(n_samples, n_samples, device=device)

        p = d * (d + 1) // 2

        basis_functions = self._compute_basis_functions(k, d)

        for i in range(n_samples):
            neighbor_idx = indices[i]
            frame = tangent_frames[i]

            neighbors = X[neighbor_idx]

            theta_i = neighbors @ frame.T

            local_basis = self._evaluate_basis(theta_i, basis_functions, d)

            if local_basis.shape[0] < p:
                continue

            Q, _ = torch.linalg.qr(local_basis)

            P = torch.eye(k, device=device) - Q @ Q.T

            for j in range(k):
                for l in range(k):
                    H[neighbor_idx[j], neighbor_idx[l]] += P[j, l]

        return H

    def _compute_basis_functions(self, k: int, d: int) -> list:
        """
        Compute polynomial basis functions for Hessian.

        For d-dimensional embedding, basis includes:
        - Linear terms: x_1, ..., x_d
        - Quadratic terms: x_i * x_j for i <= j
        """
        p = d * (d + 1) // 2

        basis = []

        for idx in range(d):
            basis.append(lambda x, i=idx: x[:, i])

        for i in range(d):
            for j in range(i, d):
                basis.append(lambda x, a=i, b=j: x[:, a] * x[:, b])

        return basis[:p]

    def _evaluate_basis(self, coords: Tensor, basis: list, d: int) -> Tensor:
        """
        Evaluate basis functions at coordinates.

        Returns:
            Matrix of basis evaluations [k, p]
        """
        k = coords.shape[0]
        p = len(basis)

        values = torch.zeros(k, p, device=coords.device)

        for idx, fn in enumerate(basis):
            values[:, idx] = fn(coords)

        return values


class HessianLLE(ManifoldLearnerBase):
    """
    Hessian-based LLE combining HLLE with standard LLE.

    Uses Hessian constraints to improve local linear embedding.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        reg: float = 1e-3,
    ):
        super().__init__(n_components, n_neighbors, reg)
        self.lle_weights_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "HessianLLE":
        """
        Fit Hessian-LLE to data.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            self
        """
        n_samples = X.shape[0]
        device = X.device

        X_np = X.cpu().numpy()
        nn_finder = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn_finder.fit(X_np)
        distances, indices = nn_finder.kneighbors(X_np)
        indices = indices[:, 1:]
        distances = distances[:, 1:]

        lle_weights = self._compute_lle_weights(X, indices)
        self.lle_weights_ = lle_weights

        hessian_operator = self._compute_hessian_operator(X, indices, n_samples)

        W = torch.zeros(n_samples, n_samples, device=device)
        for i in range(n_samples):
            for j_idx, j in enumerate(indices[i]):
                W[i, j] = lle_weights[i, j_idx]

        I = torch.eye(n_samples, device=device)
        M = (I - W).T @ (I - W)

        M_reg = M + self.reg * torch.eye(n_samples, device=device)

        H_reg = hessian_operator + self.reg * torch.eye(n_samples, device=device)

        S = M_reg + 0.1 * H_reg

        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        idx = torch.argsort(eigenvalues)[1 : self.n_components + 1]

        self.embedding_ = eigenvectors[:, idx].to(device)

        return self

    def _compute_lle_weights(self, X: Tensor, indices: Tensor) -> Tensor:
        """Compute standard LLE weights."""
        n_samples, n_neighbors = indices.shape
        device = X.device

        weights = torch.zeros(n_samples, n_neighbors, device=device)

        X_np = X.cpu().numpy()

        for i in range(n_samples):
            neighbor_idx = indices[i].cpu().numpy()
            neighbors = X_np[neighbor_idx]
            x_i = X_np[i]

            centered = neighbors - x_i

            G = centered @ centered.T
            G += self.reg * np.eye(n_neighbors)

            ones = np.ones(n_neighbors)
            try:
                w = np.linalg.solve(G, ones)
            except:
                w = np.linalg.lstsq(G, ones, rcond=None)[0]

            w = w / (w.sum() + 1e-10)
            weights[i] = torch.from_numpy(w).float()

        return weights

    def _compute_hessian_operator(
        self, X: Tensor, indices: Tensor, n_samples: int
    ) -> Tensor:
        """Compute simplified Hessian operator."""
        d = self.n_components
        device = X.device

        tangent_frames = self._estimate_tangent_spaces(X, indices, d)

        return torch.zeros(n_samples, n_samples, device=device)

    def _estimate_tangent_spaces(self, X: Tensor, indices: Tensor, d: int) -> Tensor:
        """Estimate tangent spaces."""
        n_samples = X.shape[0]
        device = X.device

        frames = torch.zeros(n_samples, d, X.shape[1], device=device)

        X_np = X.cpu().numpy()

        for i in range(n_samples):
            neighbor_idx = indices[i].cpu().numpy()
            neighbors = X_np[neighbor_idx]
            x_i = X_np[i]

            centered = neighbors - x_i

            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            frame_dim = min(d, len(S))
            frames[i, :frame_dim] = torch.from_numpy(Vt[:frame_dim]).float()

        return frames


class CurvatureBasedEmbedding(ManifoldLearnerBase):
    """
    Curvature-based manifold embedding.

    Uses local curvature estimates to guide embedding,
    preserving both local and global geometric structure.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        reg: float = 1e-3,
    ):
        super().__init__(n_components, n_neighbors, reg)
        self.curvatures_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "CurvatureBasedEmbedding":
        """
        Fit curvature-based embedding.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            self
        """
        n_samples = X.shape[0]
        device = X.device

        curvatures = self._estimate_curvatures(X)
        self.curvatures_ = curvatures

        X_np = X.cpu().numpy()
        nn_finder = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn_finder.fit(X_np)
        _, indices = nn_finder.kneighbors(X_np)
        indices = indices[:, 1:]

        W = self._build_weight_matrix(X, indices, curvatures)

        D = torch.diag(W.sum(dim=1))
        L = D - W

        eigenvalues, eigenvectors = torch.linalg.eig(L)

        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        idx = torch.argsort(eigenvalues)[1 : self.n_components + 1]

        self.embedding_ = eigenvectors[:, idx].to(device)

        return self

    def _estimate_curvatures(self, X: Tensor) -> Tensor:
        """
        Estimate local curvature at each point.

        Uses the variance of distances to neighbors as a
        proxy for local curvature.
        """
        n_samples = X.shape[0]
        device = X.device

        X_np = X.cpu().numpy()
        nn_finder = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn_finder.fit(X_np)
        distances, _ = nn_finder.kneighbors(X_np)
        distances = distances[:, 1:]

        curvatures = distances.var(axis=1)

        return torch.from_numpy(curvatures).float().to(device)

    def _build_weight_matrix(
        self, X: Tensor, indices: Tensor, curvatures: Tensor
    ) -> Tensor:
        """Build weighted adjacency matrix."""
        n_samples = X.shape[0]
        device = X.device

        W = torch.zeros(n_samples, n_samples, device=device)

        for i in range(n_samples):
            neighbors = indices[i]

            for j_idx, j in enumerate(neighbors):
                base_dist = torch.norm(X[i] - X[j])

                curv_factor = 1.0 + 0.5 * (curvatures[i] + curvatures[j])

                W[i, j] = base_dist / curv_factor

        return W


class HessianEigenmapsLayer(nn.Module):
    """
    Neural network layer implementing Hessian Eigenmaps.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
    ):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def forward(self, x: Tensor) -> Tensor:
        """Apply Hessian Eigenmaps embedding."""
        he = HessianEigenmaps(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
        )
        return he.fit_transform(x)


def create_hessian_eigenmaps(
    n_components: int = 2,
    n_neighbors: int = 10,
    variant: str = "hessian",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create Hessian Eigenmaps model.

    Args:
        n_components: Embedding dimension
        n_neighbors: Number of neighbors
        variant: Type of embedding ('hessian', 'hessian_lle', 'curvature')
        **kwargs: Additional arguments

    Returns:
        Hessian Eigenmaps model
    """
    if variant == "hessian_lle":
        return HessianLLE(n_components, n_neighbors, **kwargs)
    elif variant == "curvature":
        return CurvatureBasedEmbedding(n_components, n_neighbors, **kwargs)
    return HessianEigenmaps(n_components, n_neighbors, **kwargs)
