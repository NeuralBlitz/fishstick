"""
Diffusion Maps for Manifold Learning.

Diffusion maps embed data using the eigenvalues and eigenvectors
of a diffusion operator on the data manifold.

Key concepts:
- Diffusion distance: geometry induced by random walk
- Scale parameter: controls neighborhood size
- Markov transition matrix: defines random walk on data
- Multiscale: captures geometry at different scales

Based on: "Diffusion Maps" (Coifman et al., 2006)
"""

from typing import Optional, Tuple, List
import torch
from torch import Tensor, nn
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from .base import (
    ManifoldLearnerBase,
    GraphBuilder,
    compute_distance_matrix,
    EigenSolver,
)


class DiffusionMap(ManifoldLearnerBase):
    """
    Diffusion Maps for manifold learning.

    Embeds data using the diffusion operator:
    P(x,y) = K(x,y) / sum_z K(x,z)

    The diffusion distance D_t(x,y) approximates geodesic distance
    on the manifold, weighted by transition probability.

    Args:
        n_components: Dimension of embedding space
        n_neighbors: Number of neighbors for kernel construction
        alpha: Anisotropy parameter (0=symmetric, 1=Markov)
        epsilon: Kernel bandwidth (auto-computed if None)
        t: Diffusion time steps
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        alpha: float = 0.5,
        epsilon: Optional[float] = None,
        t: int = 1,
        reg: float = 1e-3,
    ):
        super().__init__(n_components, n_neighbors, reg)
        self.alpha = alpha
        self.epsilon = epsilon
        self.t = t
        self.transition_matrix_: Optional[Tensor] = None
        self.diffusion_distances_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "DiffusionMap":
        """
        Fit Diffusion Map to data.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            self
        """
        n_samples = X.shape[0]

        kernel = self._compute_kernel(X)

        kernel_normalized = self._normalize_kernel(kernel)

        self.transition_matrix_ = kernel_normalized

        for _ in range(self.t - 1):
            self.transition_matrix_ = self.transition_matrix_ @ kernel_normalized

        self.embedding_ = self._compute_embedding(self.transition_matrix_, n_samples)

        return self

    def _compute_kernel(self, X: Tensor) -> Tensor:
        """
        Compute Gaussian kernel matrix.

        K(x,y) = exp(-||x-y||^2 / epsilon)
        """
        distances = compute_distance_matrix(X)

        if self.epsilon is None:
            self.epsilon = distances[distances > 0].mean().item()

        kernel = torch.exp(-distances.pow(2) / self.epsilon)

        kernel = kernel * (1 - torch.eye(X.shape[0], device=X.device))

        return kernel

    def _normalize_kernel(self, kernel: Tensor) -> Tensor:
        """
        Normalize kernel to create diffusion operator.

        K_alpha = K^alpha @ D^-alpha @ K^alpha
        where D_ii = sum_j K_ij
        """
        d = kernel.sum(dim=1, keepdim=True)
        d = d.pow(self.alpha)
        d = d + self.reg

        kernel_normalized = kernel / d
        kernel_normalized = kernel_normalized / d.T

        return kernel_normalized

    def _compute_embedding(self, transition_matrix: Tensor, n_samples: int) -> Tensor:
        """
        Compute diffusion map embedding.

        Embedding coordinates: psi_t(x) = lambda_t * phi_t(x)
        where lambda_t, phi_t are eigenvalues/eigenvectors of P^t
        """
        device = transition_matrix.device

        transition_np = transition_matrix.cpu().numpy()

        eigenvalues, eigenvectors = np.linalg.eig(transition_np)

        eigenvalues = np.abs(eigenvalues)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues = eigenvalues[: self.n_components + 1]
        eigenvectors = eigenvectors[:, : self.n_components + 1]

        valid_idx = eigenvalues > 1e-10
        eigenvalues = eigenvalues[valid_idx]
        eigenvectors = eigenvectors[:, valid_idx]

        embedding = eigenvectors[:, 1:] * eigenvalues[1:]

        return (
            torch.from_numpy(embedding.real).float().to(device)[:, : self.n_components]
        )

    def diffusion_distance(self, X: Tensor) -> Tensor:
        """
        Compute diffusion distances between all pairs.

        D_t^2(x,y) = sum_k lambda_k^(2t) * (phi_k(x) - phi_k(y))^2
        """
        if self.embedding_ is None:
            raise RuntimeError("Model not fitted.")

        distances = compute_distance_matrix(self.embedding_)
        return distances


class MultiscaleDiffusionMap(DiffusionMap):
    """
    Multiscale Diffusion Maps.

    Captures geometry at multiple scales by analyzing
    the eigenvalue spectrum of the diffusion operator.

    Different scales reveal different structures:
    - Small t: local structure
    - Large t: global structure
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        epsilon: Optional[float] = None,
        time_steps: Optional[List[int]] = None,
    ):
        super().__init__(n_components, n_neighbors, epsilon=epsilon)
        self.time_steps = time_steps or [1, 2, 5, 10]
        self.scale_eigenvalues_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "MultiscaleDiffusionMap":
        """
        Fit Multiscale Diffusion Map.

        Computes eigenvalues at multiple time scales and
        analyzes the scale-space geometry.
        """
        n_samples = X.shape[0]

        kernel = self._compute_kernel(X)

        self.scale_eigenvalues_ = torch.zeros(
            len(self.time_steps), self.n_components + 1
        )

        embeddings = []

        for t_idx, t in enumerate(self.time_steps):
            normalized = self._normalize_kernel(kernel)

            for _ in range(t - 1):
                normalized = normalized @ self._normalize_kernel(kernel)

            transition_np = normalized.cpu().numpy()
            eigenvalues, eigenvectors = np.linalg.eig(transition_np)

            eigenvalues = np.abs(eigenvalues)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx[: self.n_components + 1]]

            self.scale_eigenvalues_[t_idx] = torch.from_numpy(eigenvalues).float()

            eigenvectors = eigenvectors[:, idx[1 : self.n_components + 1]]
            embedding = eigenvectors * eigenvalues[1 : self.n_components + 1]
            embeddings.append(torch.from_numpy(embedding.real).float())

        final_embedding = torch.stack(embeddings).mean(dim=0)
        self.embedding_ = final_embedding.to(X.device)

        return self

    def get_scale_analysis(self) -> dict:
        """
        Analyze the multiscale structure.

        Returns:
            Dictionary with scale information
        """
        if self.scale_eigenvalues_ is None:
            raise RuntimeError("Model not fitted.")

        return {
            "time_steps": self.time_steps,
            "eigenvalues": self.scale_eigenvalues_,
            "decay_rates": torch.diff(self.scale_eigenvalues_, dim=0),
        }


class AnisotropicDiffusionMap(DiffusionMap):
    """
    Anisotropic Diffusion Maps.

    Uses direction-aware kernel that preserves different
    geometric structures based on anisotropy parameter.

    Based on: "Geometric Diffusions as a Tool for Structure Recognition"
    (Coifman et al., 2005)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        alpha: float = 0.5,
        epsilon: Optional[float] = None,
        t: int = 1,
    ):
        super().__init__(n_components, n_neighbors, alpha, epsilon, t)
        self.local_densities_: Optional[Tensor] = None

    def _normalize_kernel(self, kernel: Tensor) -> Tensor:
        """
        Anisotropic normalization.

        The anisotropy creates preferred directions in the diffusion,
        helping to preserve different structures.
        """
        d = kernel.sum(dim=1, keepdim=True)
        d = d.pow(self.alpha)

        kernel_normalized = kernel / (d + self.reg)

        self.local_densities_ = d.squeeze()

        return kernel_normalized

    def fit(self, X: Tensor) -> "AnisotropicDiffusionMap":
        """Fit anisotropic diffusion map."""
        return super().fit(X)

    def get_density(self) -> Optional[Tensor]:
        """Get local density estimates."""
        return self.local_densities_


class KernelPCA:
    """
    Kernel PCA with diffusion map-style embedding.

    Extends standard PCA to non-linear manifolds using
    kernel trick.
    """

    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "gaussian",
        gamma: float = 1.0,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.embedding_: Optional[Tensor] = None
        self.alpha_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "KernelPCA":
        """
        Fit Kernel PCA.

        Args:
            X: Input data [n_samples, n_features]

        Returns:
            self
        """
        n = X.shape[0]

        K = self._compute_kernel(X)

        one_n = torch.ones(n, n) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        eigenvalues, eigenvectors = torch.linalg.eigh(K_centered)

        idx = torch.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx][: self.n_components]
        eigenvectors = eigenvectors[:, idx][:, : self.n_components]

        self.alpha_ = eigenvectors / torch.sqrt(eigenvalues.unsqueeze(0))

        self.embedding_ = (self.alpha_ * eigenvalues.sqrt()).to(X.device)

        return self

    def _compute_kernel(self, X: Tensor) -> Tensor:
        """Compute kernel matrix."""
        if self.kernel == "gaussian":
            distances = compute_distance_matrix(X)
            return torch.exp(-self.gamma * distances.pow(2))
        elif self.kernel == "linear":
            return X @ X.T
        else:
            return compute_distance_matrix(X)

    def transform(self, X: Tensor) -> Tensor:
        """Transform new points to kernel PCA space."""
        K_test = self._compute_kernel(X)
        return K_test @ self.alpha_


class DiffusionMapLayer(nn.Module):
    """
    Neural network layer implementing Diffusion Map embedding.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        alpha: float = 0.5,
        t: int = 1,
    ):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.t = t

    def forward(self, x: Tensor) -> Tensor:
        """Apply diffusion map embedding."""
        dm = DiffusionMap(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            alpha=self.alpha,
            t=self.t,
        )
        return dm.fit_transform(x)


def create_diffusion_map(
    n_components: int = 2,
    n_neighbors: int = 10,
    alpha: float = 0.5,
    epsilon: Optional[float] = None,
    t: int = 1,
    multiscale: bool = False,
    anisotropic: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create Diffusion Map model.

    Args:
        n_components: Embedding dimension
        n_neighbors: Number of neighbors
        alpha: Anisotropy parameter
        epsilon: Kernel bandwidth
        t: Diffusion time steps
        multiscale: Use multiscale variant
        anisotropic: Use anisotropic variant
        **kwargs: Additional arguments

    Returns:
        Diffusion Map model
    """
    if multiscale:
        return MultiscaleDiffusionMap(n_components, n_neighbors, epsilon, **kwargs)
    elif anisotropic:
        return AnisotropicDiffusionMap(n_components, n_neighbors, alpha, epsilon, t)
    return DiffusionMap(n_components, n_neighbors, alpha, epsilon, t)
