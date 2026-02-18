"""Trajectory inference for single-cell data."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class PseudotimeCalculator:
    """Calculate pseudotime for trajectory inference.

    Uses diffusion-based pseudotime calculation.
    """

    def __init__(self, n_neighbors: int = 15) -> None:
        self.n_neighbors = n_neighbors

    def compute_pseudotime(
        self,
        embedding: Tensor,
        root_cells: Optional[List[int]] = None,
    ) -> Tensor:
        """Compute pseudotime.

        Args:
            embedding: Cell embeddings
            root_cells: Indices of root cells

        Returns:
            Pseudotime values
        """
        from scipy.spatial.distance import cdist

        data = embedding.numpy()

        distances = cdist(data, data)

        knn_idx = np.argsort(distances, axis=1)[:, : self.n_neighbors]

        diffusion_matrix = np.zeros_like(distances)
        for i in range(len(data)):
            for j in knn_idx[i]:
                diffusion_matrix[i, j] = np.exp(-(distances[i, j] ** 2))

        diffusion_matrix = diffusion_matrix / diffusion_matrix.sum(
            axis=1, keepdims=True
        )

        if root_cells is None:
            root_cells = [0]

        D = np.zeros(len(data))
        D[root_cells] = 1.0

        for _ in range(100):
            D_new = diffusion_matrix @ D
            D_new = D_new / D_new.max()
            D = D_new

        return torch.tensor(D, dtype=torch.float32)


class TrajectoryInference(nn.Module):
    """Trajectory inference for single-cell data.

    Infers cell developmental trajectories using graph-based methods.

    Attributes:
        n_components: Number of components for PCA
    """

    def __init__(self, n_components: int = 50) -> None:
        super().__init__()
        self.n_components = n_components

    def forward(
        self,
        expression: Tensor,
        root_cells: Optional[List[int]] = None,
    ) -> Dict[str, Tensor]:
        """Infer cell trajectory.

        Args:
            expression: Expression matrix
            root_cells: Indices of root cells

        Returns:
            Dictionary with trajectory information
        """
        from scipy.spatial.distance import cdist

        data = expression.numpy()

        data_mean = data.mean(axis=0)
        data_centered = data - data_mean

        cov = data_centered.T @ data_centered / (data.shape[0] - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        pca_data = data_centered @ eigenvectors[:, : self.n_components]

        distances = cdist(pca_data, pca_data)

        k = 15
        knn_idx = np.argsort(distances, axis=1)[:, :k]

        transition_matrix = np.zeros_like(distances)
        sigma = 1.0
        for i in range(len(pca_data)):
            for j in knn_idx[i]:
                transition_matrix[i, j] = np.exp(
                    -(distances[i, j] ** 2) / (2 * sigma**2)
                )

        transition_matrix = transition_matrix / transition_matrix.sum(
            axis=1, keepdims=True
        )

        pseudotime_calc = PseudotimeCalculator()
        pseudotime = pseudotime_calc.compute_pseudotime(
            torch.tensor(pca_data, dtype=torch.float32),
            root_cells,
        )

        return {
            "pseudotime": pseudotime,
            "transition_matrix": torch.tensor(transition_matrix, dtype=torch.float32),
            "embedding": torch.tensor(pca_data, dtype=torch.float32),
        }


def compute_trajectory_genes(
    expression: Tensor,
    pseudotime: Tensor,
    n_genes: int = 100,
) -> Tuple[Tensor, Tensor]:
    """Find genes correlated with pseudotime.

    Args:
        expression: Expression matrix
        pseudotime: Pseudotime values
        n_genes: Number of genes to return

    Returns:
        Tuple of (gene_indices, correlations)
    """
    correlations = []

    for gene_idx in range(expression.shape[1]):
        gene_expr = expression[:, gene_idx]

        corr = torch.corrcoef(torch.stack([gene_expr, pseudotime]))[0, 1]

        correlations.append((gene_idx, corr.abs().item()))

    correlations.sort(key=lambda x: x[1], reverse=True)

    top_genes = [c[0] for c in correlations[:n_genes]]
    top_corrs = torch.tensor([c[1] for c in correlations[:n_genes]])

    return torch.tensor(top_genes), top_corrs


def compute_branch_genes(
    expression: Tensor,
    pseudotime: Tensor,
    branch_point: float,
) -> Dict[str, Tensor]:
    """Find genes that change at branch points.

    Args:
        expression: Expression matrix
        pseudotime: Pseudotime values
        branch_point: Pseudotime value of branch

    Returns:
        Dictionary of branch-specific genes
    """
    before = expression[pseudotime < branch_point]
    after = expression[pseudotime >= branch_point]

    mean_before = before.mean(dim=0)
    mean_after = after.mean(dim=0)

    fold_change = mean_after / (mean_before + 1e-8)

    up_regulated = fold_change > 1.5
    down_regulated = fold_change < 0.67

    return {
        "upregulated": torch.where(up_regulated)[0],
        "downregulated": torch.where(down_regulated)[0],
        "fold_change": fold_change,
    }
