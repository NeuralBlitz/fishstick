"""Differential expression analysis."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from scipy import stats


@dataclass
class DEGene:
    """Represents a differentially expressed gene.

    Attributes:
        gene_id: Gene identifier
        log_fold_change: Log2 fold change
        p_value: Statistical p-value
        adjusted_p_value: FDR-adjusted p-value
        base_mean: Mean expression across all samples
    """

    gene_id: str
    log_fold_change: float
    p_value: float
    adjusted_p_value: float
    base_mean: float

    @property
    def is_significant(self, threshold: float = 0.05) -> bool:
        return self.adjusted_p_value < threshold

    @property
    def direction(self) -> str:
        if self.log_fold_change > 0:
            return "up"
        return "down"


class DifferentialExpression(nn.Module):
    """Differential expression analysis.

    Performs differential expression analysis between two conditions
    using negative binomial modeling.

    Attributes:
        method: DE method ('ttest', 'wilcox', 'deseq2')
        min_base_mean: Minimum base mean to consider
    """

    def __init__(
        self,
        method: str = "ttest",
        min_base_mean: float = 1.0,
    ) -> None:
        super().__init__()
        self.method = method
        self.min_base_mean = min_base_mean

    def forward(
        self,
        condition1: Tensor,
        condition2: Tensor,
        gene_names: Optional[List[str]] = None,
    ) -> List[DEGene]:
        """Perform differential expression analysis.

        Args:
            condition1: Expression matrix for condition 1 (genes x samples)
            condition2: Expression matrix for condition 2
            gene_names: List of gene names

        Returns:
            List of differentially expressed genes
        """
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(condition1.shape[0])]

        results = []

        for i in range(condition1.shape[0]):
            expr1 = condition1[i].numpy()
            expr2 = condition2[i].numpy()

            mean1 = np.mean(expr1)
            mean2 = np.mean(expr2)

            if mean1 < self.min_base_mean and mean2 < self.min_base_mean:
                continue

            log_fc = np.log2((mean2 + 1) / (mean1 + 1))

            if self.method == "ttest":
                _, p_value = stats.ttest_ind(expr1, expr2)
            elif self.method == "wilcox":
                _, p_value = stats.mannwhitneyu(expr1, expr2, alternative='two-sided')
            else:
                _, p_value = stats.ttest_ind(expr1, expr2)

            base_mean = (mean1 + mean2) / 2

            results.append(DEGene(
                gene_id=gene_names[i],
                log_fold_change=log_fc,
                p_value=p_value if not np.isnan(p_value) else 1.0,
                adjusted_p_value=1.0,
                base_mean=base_mean,
            ))

        results = self._adjust_pvalues(results)

        return results

    def _adjust_pvalues(self, results: List[DEGene]) -> List[DEGene]:
        """Apply FDR correction.

        Args:
            results: DE results

        Returns:
            Results with adjusted p-values
        """
        p_values = [r.p_value for r in results]
        adjusted = self._fdr correction(p_values)

        for i, result in enumerate(results):
            result.adjusted_p_value = adjusted[i]

        return results

    @staticmethod
    def _fdr correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
        """ Benjamini-Hochberg FDR correction.

        Args:
            p_values: List of p-values
            alpha: Significance threshold

        Returns:
            Adjusted p-values
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        adjusted = np.zeros(n)
        min_adjusted = 1.0

        for i in range(n - 1, -1, -1):
            bh_value = sorted_p[i] * n / (i + 1)
            min_adjusted = min(bh_value, min_adjusted)
            adjusted[sorted_indices[i]] = min_adjusted

        return adjusted.tolist()

    def get_significant_genes(
        self,
        results: List[DEGene],
        p_threshold: float = 0.05,
        fc_threshold: float = 1.0,
    ) -> Tuple[List[DEGene], List[DEGene]]:
        """Get significant up- and down-regulated genes.

        Args:
            results: DE results
            p_threshold: P-value threshold
            fc_threshold: Log fold-change threshold

        Returns:
            Tuple of (upregulated, downregulated) genes
        """
        upregulated = []
        downregulated = []

        for gene in results:
            if gene.adjusted_p_value > p_threshold:
                continue

            if gene.log_fold_change >= fc_threshold:
                upregulated.append(gene)
            elif gene.log_fold_change <= -fc_threshold:
                downregulated.append(gene)

        return upregulated, downregulated


class DEAnalysis:
    """Complete differential expression analysis pipeline."""

    def __init__(
        self,
        normalizer: Optional[nn.Module] = None,
        de_method: str = "ttest",
    ) -> None:
        self.normalizer = normalizer
        self.de_model = DifferentialExpression(method=de_method)

    def analyze(
        self,
        counts: Tensor,
        groups: Tensor,
        gene_names: Optional[List[str]] = None,
    ) -> Dict[str, List[DEGene]]:
        """Run complete DE analysis.

        Args:
            counts: Count matrix (genes x samples)
            groups: Group labels (0 or 1)
            gene_names: Gene names

        Returns:
            Dictionary of DE results by contrast
        """
        if self.normalizer is not None:
            counts = self.normalizer.forward(counts)

        condition1 = counts[:, groups == 0]
        condition2 = counts[:, groups == 1]

        results = self.de_model(condition1, condition2, gene_names)

        return {"all_genes": results}


def compute_volcano_data(
    results: List[DEGene],
    p_threshold: float = 0.05,
    fc_threshold: float = 1.0,
) -> Dict[str, Tensor]:
    """Prepare data for volcano plot.

    Args:
        results: DE results
        p_threshold: P-value threshold
        fc_threshold: Fold-change threshold

    Returns:
        Dictionary with plotting data
    """
    log_fc = torch.tensor([r.log_fold_change for r in results])
    neg_log_p = torch.tensor([-np.log10(r.p_value + 1e-10) for r in results])

    colors = []
    for r in results:
        if r.adjusted_p_value < p_threshold:
            if r.log_fold_change > fc_threshold:
                colors.append("red")
            elif r.log_fold_change < -fc_threshold:
                colors.append("blue")
            else:
                colors.append("gray")
        else:
            colors.append("gray")

    return {
        "log_fold_change": log_fc,
        "neg_log_pvalue": neg_log_p,
        "colors": colors,
    }
