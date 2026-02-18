"""Single-cell RNA-seq normalization methods."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class ScNormalizer(nn.Module):
    """Base normalizer for single-cell data."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, counts: Tensor) -> Tensor:
        raise NotImplementedError


class ScTransform(ScNormalizer):
    """SCTransform normalization for scRNA-seq.

    Models the relationship between variance and mean using regularized
    negative binomial regression.
    """

    def __init__(
        self,
        n_genes: int = 2000,
        min_genes: int = 200,
        min_cells: int = 3,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.min_genes = min_genes
        self.min_cells = min_cells

        self.gene_means: Optional[Tensor] = None
        self.gene_vars: Optional[Tensor] = self.register_buffer
        self.slope: Optional[Tensor] = None
        self.intercept: Optional[Tensor] = None

    def fit(self, counts: Tensor) -> "ScTransform":
        """Fit SCTransform parameters.

        Args:
            counts: Count matrix (cells x genes)

        Returns:
            Self
        """
        cell_means = counts.mean(dim=0)

        genes_detected = (counts > 0).sum(dim=0) >= self.min_cells
        genes_to_use = genes_detected.sum() >= self.min_genes

        if not genes_to_use:
            return self

        gene_means = counts.mean(dim=0)
        gene_vars = counts.var(dim=0)

        log_counts = torch.log1p(counts)
        log_means = log_counts.mean(dim=0)

        from scipy.stats import linregress

        x = log_means.numpy()
        y = np.log(gene_vars.numpy() + 1)

        slope, intercept, _, _, _ = linregress(x, y)

        self.slope = torch.tensor(slope)
        self.intercept = torch.tensor(intercept)

        return self

    def forward(self, counts: Tensor) -> Tensor:
        """Apply SCTransform normalization.

        Args:
            counts: Count matrix

        Returns:
            Normalized expression values
        """
        if self.slope is None:
            return log_normalize(counts)

        log_counts = torch.log1p(counts)

        cell_means = log_counts.mean(dim=1, keepdim=True)

        normalized = (log_counts - cell_means) / (cell_means + 1).sqrt()

        return normalized


def log_normalize(counts: Tensor, target_sum: float = 1e4) -> Tensor:
    """Log-normalize counts.

    Args:
        counts: Count matrix (cells x genes)
        target_sum: Target sum per cell

    Returns:
        Log-normalized expression
    """
    cell_sums = counts.sum(dim=1, keepdim=True)

    normalized = counts * (target_sum / cell_sums)

    log_normalized = torch.log1p(normalized)

    return log_normalized


def select_highly_variable_genes(
    counts: Tensor,
    n_genes: int = 2000,
    min_mean: float = 0.1,
    max_mean: float = 8,
) -> Tensor:
    """Select highly variable genes.

    Args:
        counts: Expression matrix
        n_genes: Number of genes to select
        min_mean: Minimum mean expression
        max_mean: Maximum mean expression

    Returns:
        Boolean mask for selected genes
    """
    gene_means = counts.mean(dim=0)
    gene_vars = counts.var(dim=0)

    mean_mask = (gene_means >= min_mean) & (gene_means <= max_mean)

    cv = gene_vars.sqrt() / (gene_means + 1e-8)

    hvg_scores = cv * gene_means

    _, top_indices = torch.topk(hvg_scores, min(n_genes, len(hvg_scores)))

    selected = torch.zeros_like(gene_means, dtype=torch.bool)
    selected[top_indices] = True
    selected = selected & mean_mask

    return selected


def compute_mitochondrial_percentage(counts: Tensor, gene_names: list) -> Tensor:
    """Compute mitochondrial gene percentage per cell.

    Args:
        counts: Count matrix
        gene_names: List of gene names

    Returns:
        Mitochondrial percentage per cell
    """
    mt_genes = [i for i, name in enumerate(gene_names) if name.startswith("MT-")]

    if not mt_genes:
        return torch.zeros(counts.shape[0])

    mt_counts = counts[:, mt_genes].sum(dim=1)
    total_counts = counts.sum(dim=1)

    return mt_counts / total_counts * 100


def compute_ribosomal_percentage(counts: Tensor, gene_names: list) -> Tensor:
    """Compute ribosomal gene percentage per cell.

    Args:
        counts: Count matrix
        gene_names: List of gene names

    Returns:
        Ribosomal percentage per cell
    """
    rb_genes = [
        i
        for i, name in enumerate(gene_names)
        if name.startswith("RPS") or name.startswith("RPL")
    ]

    if not rb_genes:
        return torch.zeros(counts.shape[0])

    rb_counts = counts[:, rb_genes].sum(dim=1)
    total_counts = counts.sum(dim=1)

    return rb_counts / total_counts * 100


def downsample_counts(
    counts: Tensor,
    target_counts: int = 10000,
) -> Tensor:
    """Downsample counts to target number.

    Args:
        counts: Count matrix (cells x genes)
        target_counts: Target total counts per cell

    Returns:
        Downsampled counts
    """
    cell_sums = counts.sum(dim=1, keepdim=True)

    scale_factors = target_counts / cell_sums.clamp(min=1)

    scaled = counts * scale_factors

    downsampled = torch.poisson(scaled)

    return downsampled


class CellCycleScorer(nn.Module):
    """Score cells for cell cycle phase."""

    def __init__(self) -> None:
        super().__init__()

        self.s_genes = [
            "MCM5",
            "PCNA",
            "TYMS",
            "FEN1",
            "MCM2",
            "MCM4",
            "RRM1",
            "RRM2",
            "UNG",
            "GINS2",
        ]
        self.g2m_genes = [
            "TOP2A",
            "BIRC5",
            "KIF2C",
            "NCAPD3",
            "NCAPG",
            "NCAPH",
            "SGO1",
            "AURKB",
            "AURKA",
            "CCNB2",
        ]

    def score(self, normalized: Tensor, gene_names: list) -> Dict[str, Tensor]:
        """Compute cell cycle scores.

        Args:
            normalized: Normalized expression matrix
            gene_names: List of gene names

        Returns:
            Dictionary of S and G2M scores
        """
        s_indices = [i for i, g in enumerate(gene_names) if g in self.s_genes]
        g2m_indices = [i for i, g in enumerate(gene_names) if g in self.g2m_genes]

        s_score = (
            normalized[:, s_indices].mean(dim=1)
            if s_indices
            else torch.zeros(normalized.shape[0])
        )
        g2m_score = (
            normalized[:, g2m_indices].mean(dim=1)
            if g2m_indices
            else torch.zeros(normalized.shape[0])
        )

        return {
            "S_score": s_score,
            "G2M_score": g2m_score,
        }
