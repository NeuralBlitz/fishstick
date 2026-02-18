"""
Locally Linear Embedding (LLE) and Variants.

LLE preserves local neighborhood relationships by representing
each point as a linear combination of its neighbors.

Variants:
- Standard LLE: Basic locally linear embedding
- Modified LLE (MLLE): Uses positive weights constraint
- Hessian LLE (HLLE): Uses Hessian-based local coordinates
- LTSA: Local Tangent Space Alignment

Based on: "Nonlinear Dimensionality Reduction by Locally Linear Embedding"
(Roweis & Saul, Science 2000)
"""

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors

from .base import (
    ManifoldLearnerBase,
    GraphBuilder,
    compute_distance_matrix,
    EigenSolver,
    local_pca,
)


class LLE(ManifoldLearnerBase):
    """
    Locally Linear Embedding (LLE).

    Finds low-dimensional representation that preserves
    local reconstruction weights.

    The algorithm:
    1. Find k nearest neighbors
    2. Compute local reconstruction weights W that minimize
       ||x_i - sum_j W_ij * x_j||^2
    3. Compute embedding Y that minimizes
       ||Y - W*Y||^2
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        reg: float = 1e-3,
        max_iter: int = 100,
    ):
        super().__init__(n_components, n_neighbors, reg)
        self.max_iter = max_iter
        self.weights_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "LLE":
        """
        Fit LLE to data.

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

        indices = torch.from_numpy(indices).long()
        distances = torch.from_numpy(distances).float()

        weights = self._compute_weights(X, indices, distances)
        self.weights_ = weights

        embedding = self._compute_embedding(weights, indices, n_samples)
        self.embedding_ = embedding.to(device)

        return self

    def _compute_weights(self, X: Tensor, indices: Tensor, distances: Tensor) -> Tensor:
        """
        Compute local reconstruction weights.

        Solves: min ||x_i - sum_j W_ij * x_j||^2
        subject to sum_j W_ij = 1
        """
        n_samples, n_neighbors = indices.shape
        device = X.device

        weights = torch.zeros(n_samples, n_neighbors, device=device)

        X_np = X.cpu().numpy()

        for i in range(n_samples):
            neighbor_idx = indices[i].cpu().numpy()
            neighbors = X_np[neighbor_idx]
            x_i = X_np[i]

            centered = neighbors - x_i

            C = centered @ centered.T
            C += self.reg * torch.eye(n_neighbors).numpy()

            ones = np.ones(n_neighbors)
            try:
                w = np.linalg.solve(C, ones)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(C, ones, rcond=None)[0]

            w = w / w.sum()
            weights[i] = torch.from_numpy(w).float()

        return weights

    def _compute_embedding(
        self, weights: Tensor, indices: Tensor, n_samples: int
    ) -> Tensor:
        """
        Compute embedding from weights.

        Solves: min ||Y - W*Y||^2 = min Y^T * (I-W)^T * (I-W) * Y
        """
        W = torch.zeros(n_samples, n_samples)

        for i in range(n_samples):
            for j_idx, j in enumerate(indices[i]):
                W[i, j] = weights[i, j_idx]

        I = torch.eye(n_samples)
        M = (I - W).T @ (I - W)

        eigenvalues, eigenvectors = torch.linalg.eigh(M)

        idx = torch.argsort(eigenvalues)[1 : self.n_components + 1]
        embedding = eigenvectors[:, idx]

        return embedding

    def reconstruction_error(self) -> float:
        """Compute reconstruction error."""
        if self.weights_ is None or self.embedding_ is None:
            raise RuntimeError("Model not fitted.")

        return float(self.weights_.abs().mean())


class ModifiedLLE(ManifoldLearnerBase):
    """
    Modified Locally Linear Embedding (MLLE).

    Uses a different weight computation that ensures positive weights,
    making it more robust for non-convex manifolds.

    Based on: "Modified Locally Linear Embedding for Face Recognition"
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        reg: float = 1e-3,
    ):
        super().__init__(n_components, n_neighbors, reg)

    def fit(self, X: Tensor) -> "ModifiedLLE":
        """
        Fit Modified LLE to data.

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

        indices = torch.from_numpy(indices).long()
        distances = torch.from_numpy(distances).float()

        weights = self._compute_weights(X, indices, distances)

        embedding = self._compute_embedding(weights, indices, n_samples)
        self.embedding_ = embedding.to(device)

        return self

    def _compute_weights(self, X: Tensor, indices: Tensor, distances: Tensor) -> Tensor:
        """
        Compute weights using modified algorithm.

        For each point, find weights that minimize reconstruction error
        while keeping weights positive.
        """
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

            eigenvalues, eigenvectors = np.linalg.eigh(G)

            k = min(3, n_neighbors)
            V = eigenvectors[:, -k:]
            E = eigenvalues[-k:]

            G_pinv = V @ np.diag(1.0 / (E + self.reg)) @ V.T

            ones = np.ones(n_neighbors)
            w = G_pinv @ ones
            w = np.maximum(w, 0)

            if w.sum() > 0:
                w = w / w.sum()

            weights[i] = torch.from_numpy(w).float()

        return weights

    def _compute_embedding(
        self, weights: Tensor, indices: Tensor, n_samples: int
    ) -> Tensor:
        """Compute embedding using standard LLE formula."""
        W = torch.zeros(n_samples, n_samples)

        for i in range(n_samples):
            for j_idx, j in enumerate(indices[i]):
                W[i, j] = weights[i, j_idx]

        I = torch.eye(n_samples)
        M = (I - W).T @ (I - W)

        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        idx = torch.argsort(eigenvalues)[1 : self.n_components + 1]

        return eigenvectors[:, idx]


class HessianLLE(ManifoldLearnerBase):
    """
    Hessian-based Locally Linear Embedding (HLLE).

    Uses local Hessian to estimate tangent space, then aligns
    tangent spaces to find global embedding.

    More robust to sampling density variations than standard LLE.

    Based on: "Hessian Eigenmaps" (Donoho & Grimes, 2003)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        reg: float = 1e-3,
    ):
        super().__init__(n_components, n_neighbors, reg)
        self.tangent_frames_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "HessianLLE":
        """
        Fit Hessian LLE to data.

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

        H = self._compute_hessian_matrix(X, indices, tangent_frames, d, n_samples)

        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        idx = torch.argsort(eigenvalues)[1 : d + 1]
        self.embedding_ = eigenvectors[:, idx].to(X.device)

        return self

    def _estimate_tangent_spaces(self, X: Tensor, indices: Tensor, d: int) -> Tensor:
        """
        Estimate local tangent spaces using PCA.

        Returns:
            Tangent frames [n_samples, d, n_features]
        """
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

            k = min(d, len(S))
            frames[i, :k] = torch.from_numpy(Vt[:k]).float()

        return frames

    def _compute_hessian_matrix(
        self,
        X: Tensor,
        indices: Tensor,
        tangent_frames: Tensor,
        d: int,
        n_samples: int,
    ) -> Tensor:
        """
        Compute Hessian matrix for HLLE.

        The Hessian is computed in local coordinate systems
        and then aligned across all points.
        """
        k = self.n_neighbors

        H = torch.zeros(n_samples, n_samples)

        for i in range(n_samples):
            neighbor_idx = indices[i]
            frame = tangent_frames[i]

            neighbor_coords = X[neighbor_idx]
            local_coords = neighbor_coords @ frame.T

            Hi = self._local_hessian(local_coords, d)

            for j_idx, j in enumerate(neighbor_idx):
                for l_idx, l in enumerate(neighbor_idx):
                    H[i, n_samples + j] += Hi[j_idx, l_idx]
                    H[n_samples + j, i] += Hi[j_idx, l_idx]

        return H

    def _local_hessian(self, local_coords: Tensor, d: int) -> Tensor:
        """
        Compute local Hessian in tangent space.

        Args:
            local_coords: Local coordinates [k, d]
            d: Intrinsic dimension

        Returns:
            Local Hessian matrix [k, k]
        """
        k = local_coords.shape[0]

        basis = torch.ones(k, d + 1)
        basis[:, 1 : d + 1] = local_coords

        try:
            Q, R = torch.linalg.qr(basis)
        except:
            return torch.eye(k)

        Hi = Q[:, d + 1 :] @ Q[:, d + 1 :].T

        return Hi


class LTSA(ManifoldLearnerBase):
    """
    Local Tangent Space Alignment (LTSA).

    Aligns local tangent space representations to find
    global embedding that preserves local geometry.

    Based on: "Principal manifolds and nonlinear dimensionality reduction
    via tangent space alignment" (Zhang & Zha, 2004)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        reg: float = 1e-3,
    ):
        super().__init__(n_components, n_neighbors, reg)
        self.tangent_coords_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "LTSA":
        """
        Fit LTSA to data.

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

        S = self._compute_alignment_matrix(X, indices, d, n_samples)

        eigenvalues, eigenvectors = torch.linalg.eigh(S)
        idx = torch.argsort(eigenvalues)[1 : d + 1]
        self.embedding_ = eigenvectors[:, idx].to(X.device)

        return self

    def _compute_alignment_matrix(
        self, X: Tensor, indices: Tensor, d: int, n_samples: int
    ) -> Tensor:
        """
        Compute alignment matrix for LTSA.

        S = sum_i (I - V_i V_i^T) (I - V_i V_i^T)^T
        where V_i contains local tangent space coordinates
        """
        S = torch.zeros(n_samples, n_samples)

        X_np = X.cpu().numpy()

        for i in range(n_samples):
            neighbor_idx = indices[i].cpu().numpy()
            neighbors = X_np[neighbor_idx]
            x_i = X_np[i]

            centered = neighbors - x_i
            k = centered.shape[0]

            U, S_vals, Vt = np.linalg.svd(centered, full_matrices=False)

            d_local = min(d, Vt.shape[0])

            if d_local == 0:
                continue

            V = Vt[:d_local].T

            one_n = np.ones((k, 1)) / k

            local_coords = centered @ V

            local_centered = local_coords - local_coords.mean(axis=0)

            if local_centered.shape[1] > 0:
                Q, R = np.linalg.qr(local_centered)

                B = np.eye(k) - one_n @ np.ones((1, k)) - Q @ Q.T

                for j_idx, j in enumerate(neighbor_idx):
                    for l_idx, l in enumerate(neighbor_idx):
                        S[j, l] += B[j_idx, l_idx]

        return S


class LLELayer(nn.Module):
    """
    Neural network layer implementing LLE embedding.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        variant: str = "standard",
    ):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.variant = variant
        self.lle = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply LLE embedding."""
        if self.variant == "standard":
            model = LLE(self.n_components, self.n_neighbors)
        elif self.variant == "modified":
            model = ModifiedLLE(self.n_components, self.n_neighbors)
        elif self.variant == "hessian":
            model = HessianLLE(self.n_components, self.n_neighbors)
        elif self.variant == "ltsa":
            model = LTSA(self.n_components, self.n_neighbors)
        else:
            model = LLE(self.n_components, self.n_neighbors)

        return model.fit_transform(x)


def create_lle(
    n_components: int = 2,
    n_neighbors: int = 10,
    variant: str = "standard",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create LLE model.

    Args:
        n_components: Embedding dimension
        n_neighbors: Number of neighbors
        variant: Type of LLE ('standard', 'modified', 'hessian', 'ltsa')
        **kwargs: Additional arguments

    Returns:
        LLE model
    """
    if variant == "modified":
        return ModifiedLLE(n_components, n_neighbors, **kwargs)
    elif variant == "hessian":
        return HessianLLE(n_components, n_neighbors, **kwargs)
    elif variant == "ltsa":
        return LTSA(n_components, n_neighbors, **kwargs)
    return LLE(n_components, n_neighbors, **kwargs)
