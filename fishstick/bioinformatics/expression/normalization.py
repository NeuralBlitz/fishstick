"""Gene expression normalization methods."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


MEAN_TRANSCRIPT_LENGTHS = {
    "ACTB": 1853,
    "GAPDH": 1425,
    "RPLP0": 1120,
    "B2M": 162,
}


class ExpressionNormalizer(nn.Module):
    """Base class for expression normalization."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, counts: Tensor) -> Tensor:
        raise NotImplementedError


class TPMNormalizer(ExpressionNormalizer):
    """Transcripts Per Million normalization.

    TPM normalizes for gene length and sequencing depth,
    making gene expression levels more comparable across samples.
    """

    def __init__(
        self,
        gene_lengths: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.gene_lengths = gene_lengths or MEAN_TRANSCRIPT_LENGTHS

    def forward(self, counts: Tensor, gene_names: Optional[List[str]] = None) -> Tensor:
        """Normalize counts to TPM.

        Args:
            counts: Raw count matrix (genes x samples)
            gene_names: List of gene names

        Returns:
            TPM-normalized expression
        """
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts.shape[0])]

        lengths = torch.tensor(
            [self.gene_lengths.get(name, 1000) for name in gene_names],
            dtype=torch.float32,
        )

        rpk = counts / (lengths.unsqueeze(1) / 1000)

        per_million = rpk.sum(dim=0) / 1e6

        tpm = rpk / per_million

        return tpm


class FPKMNormalizer(ExpressionNormalizer):
    """Fragments Per Kilobase Million normalization.

    Similar to TPM but uses fragments instead of reads
    (accounts for paired-end reads).
    """

    def __init__(
        self,
        gene_lengths: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.gene_lengths = gene_lengths or MEAN_TRANSCRIPT_LENGTHS

    def forward(self, counts: Tensor, gene_names: Optional[List[str]] = None) -> Tensor:
        """Normalize counts to FPKM.

        Args:
            counts: Raw count matrix (genes x samples)
            gene_names: List of gene names

        Returns:
            FPKM-normalized expression
        """
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(counts.shape[0])]

        lengths = torch.tensor(
            [self.gene_lengths.get(name, 1000) for name in gene_names],
            dtype=torch.float32,
        )

        fpk = counts / (lengths.unsqueeze(1) / 1000)

        per_million = fpk.sum(dim=0) / 1e6

        fpkm = fpk / per_million

        return fpkm


class CPMNormalizer(ExpressionNormalizer):
    """Counts Per Million normalization.

    Simple normalization for sequencing depth only,
    does not account for gene length.
    """

    def forward(self, counts: Tensor) -> Tensor:
        """Normalize counts to CPM.

        Args:
            counts: Raw count matrix (genes x samples)

        Returns:
            CPM-normalized expression
        """
        per_million = counts.sum(dim=0) / 1e6
        return counts / per_million


class TMMNormalizer(ExpressionNormalizer):
    """Trimmed Mean of M-values normalization.

    Normalizes for composition biases between samples.
    """

    def __init__(
        self,
        ref_sample: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.ref_sample = ref_sample

    def forward(self, counts: Tensor) -> Tensor:
        """Apply TMM normalization.

        Args:
            counts: Raw count matrix

        Returns:
            TMM-normalized expression
        """
        log_ratios = torch.log2(counts[:, 1:] / counts[:, :1].clamp(min=1))

        abs_log_ratios = torch.abs(log_ratios)
        trim_mask = (abs_log_ratios > torch.quantile(abs_log_ratios, 0.3)) & (
            abs_log_ratios < torch.quantile(abs_log_ratios, 0.7)
        )

        weighted_log_ratios = (log_ratios * trim_mask).sum(dim=0) / trim_mask.sum(
            dim=0
        ).clamp(min=1)

        scale_factors = 2**weighted_log_ratios

        normalized = counts.clone()
        normalized[:, 1:] = counts[:, 1:] / scale_factors

        return normalized


def compute_size_factors(
    counts: Tensor,
    method: str = "median",
) -> Tensor:
    """Compute size factors for normalization.

    Args:
        counts: Raw count matrix
        method: Method to use ('median' or 'DESeq')

    Returns:
        Size factors for each sample
    """
    if method == "median":
        geometric_means = torch.exp(torch.log(counts.clamp(min=1)).mean(dim=0))

        ratios = counts / geometric_means.unsqueeze(0)
        size_factors = torch.median(ratios, dim=0).values

        return size_factors

    elif method == "DESeq":
        log_counts = torch.log(counts.clamp(min=1))
        log_means = log_counts.mean(dim=0)

        log_ratios = log_counts - log_means.unsqueeze(0)

        abs_ratios = torch.abs(log_ratios)
        trimmed = log_ratios[
            (abs_ratios > torch.quantile(abs_ratios, 0.25, dim=0, keepdim=True))
            & (abs_ratios < torch.quantile(abs_ratios, 0.75, dim=0, keepdim=True))
        ]

        size_factors = torch.exp(trimmed.mean(dim=0))

        return size_factors

    return torch.ones(counts.shape[1])


def voom_normalization(
    counts: Tensor,
    design: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Apply VOOM normalization from limma.

    Args:
        counts: Raw counts
        design: Experimental design matrix

    Returns:
        Tuple of (normalized expression, weights)
    """
    log_counts = torch.log2(counts.clamp(min=1))

    mean_variance = log_counts.var(dim=1)
    mean_expression = log_counts.mean(dim=1)

    from scipy.interpolate import UnivariateSpline

    smooth_var = np.polyfit(mean_expression.numpy(), mean_variance.numpy(), 3)
    fitted_var = np.polyval(smooth_var, mean_expression.numpy())

    weights = 1 / fitted_var

    normalized = log_counts
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    return normalized, weight_tensor
