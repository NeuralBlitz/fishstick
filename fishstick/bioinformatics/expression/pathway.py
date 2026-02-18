"""Pathway enrichment analysis."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from scipy import stats


KEGG_PATHWAYS = {
    "hsa00010": ["ENO1", "ENO2", "GAPDH", "PGK1", "PKM", "LDHA"],
    "hsa00020": ["PDHA1", "PDHB", "DLAT", "DLD", "CS"],
    "hsa00190": ["ATP5F1", "COX1", "COX2", "CYTB", "ND1"],
    "hsa00240": ["CAD", "DHFR", "TYMS", "RRM1", "RRM2"],
    "hsa03010": ["RPSA", "RPS2", "RPS3", "RPL4", "RPL5"],
}


@dataclass
class EnrichmentResult:
    """Represents a pathway enrichment result.

    Attributes:
        pathway_id: KEGG or GO identifier
        pathway_name: Human-readable name
        p_value: Enrichment p-value
        genes: Genes in the pathway
        overlapping_genes: Overlapping genes
        enrichment_score: Enrichment score
    """

    pathway_id: str
    pathway_name: str
    p_value: float
    genes: List[str]
    overlapping_genes: List[str]
    enrichment_score: float


class PathwayEnrichment(nn.Module):
    """Pathway enrichment analysis.

    Performs over-representation analysis for gene sets against
    known pathways.

    Attributes:
        pathways: Dictionary of pathway_id to gene lists
        background_size: Size of gene background
    """

    def __init__(
        self,
        pathways: Optional[Dict[str, List[str]]] = None,
        background_size: int = 20000,
    ) -> None:
        super().__init__()
        self.pathways = pathways or KEGG_PATHWAYS
        self.background_size = background_size

    def forward(
        self,
        gene_set: List[str],
        p_threshold: float = 0.05,
    ) -> List[EnrichmentResult]:
        """Perform pathway enrichment analysis.

        Args:
            gene_set: List of differentially expressed genes
            p_threshold: P-value threshold

        Returns:
            List of enrichment results
        """
        gene_set_set = set(gene_set)
        n_genes = len(gene_set_set)

        results = []

        for pathway_id, pathway_genes in self.pathways.items():
            pathway_set = set(pathway_genes)
            n_pathway = len(pathway_set)

            overlap = gene_set_set & pathway_set
            n_overlap = len(overlap)

            if n_overlap == 0:
                continue

            expected = (n_pathway * n_genes) / self.background_size
            enrichment = n_overlap / max(expected, 1)

            p_value = self._hypergeometric_pvalue(
                n_overlap,
                n_pathway,
                n_genes,
                self.background_size,
            )

            if p_value < p_threshold:
                results.append(
                    EnrichmentResult(
                        pathway_id=pathway_id,
                        pathway_name=pathway_id,
                        p_value=p_value,
                        genes=pathway_genes,
                        overlapping_genes=list(overlap),
                        enrichment_score=enrichment,
                    )
                )

        results.sort(key=lambda x: x.p_value)

        return results[:20]

    def _hypergeometric_pvalue(
        self,
        k: int,
        m: int,
        n: int,
        N: int,
    ) -> float:
        """Compute hypergeometric p-value.

        Args:
            k: Number of overlapping genes
            m: Pathway size
            n: Gene set size
            N: Background size

        Returns:
            P-value
        """
        return stats.hypergeom.sf(k - 1, N, m, n)


class GSEA:
    """Gene Set Enrichment Analysis.

    Performs GSEA to identify enriched pathways based on ranked gene lists.
    """

    def __init__(
        self,
        pathways: Optional[Dict[str, List[str]]] = None,
        n_permutations: int = 1000,
    ) -> None:
        self.pathways = pathways or KEGG_PATHWAYS
        self.n_permutations = n_permutations

    def run(
        self,
        ranked_genes: List[tuple],
        min_size: int = 15,
        max_size: int = 500,
    ) -> List[EnrichmentResult]:
        """Run GSEA.

        Args:
            ranked_genes: List of (gene_name, score) tuples, sorted by score
            min_size: Minimum pathway size
            max_size: Maximum pathway size

        Returns:
            List of GSEA results
        """
        gene_to_score = {gene: score for gene, score in ranked_genes}
        genes = [gene for gene, _ in ranked_genes]

        results = []

        for pathway_id, pathway_genes in self.pathways.items():
            pathway_set = set(pathway_genes)
            pathway_size = len(pathway_set & set(genes))

            if pathway_size < min_size or pathway_size > max_size:
                continue

            es = self._compute_enrichment_score(
                genes,
                gene_to_score,
                pathway_set,
            )

            p_value = self._permutation_test(
                ranked_genes,
                pathway_set,
                es,
            )

            results.append(
                EnrichmentResult(
                    pathway_id=pathway_id,
                    pathway_name=pathway_id,
                    p_value=p_value,
                    genes=pathway_genes,
                    overlapping_genes=list(pathway_set & set(genes)),
                    enrichment_score=es,
                )
            )

        results.sort(key=lambda x: x.p_value)

        return results

    def _compute_enrichment_score(
        self,
        genes: List[str],
        gene_to_score: Dict[str, float],
        pathway_genes: Set[str],
    ) -> float:
        """Compute enrichment score.

        Args:
            genes: Ranked gene list
            gene_to_score: Gene to score mapping
            pathway_genes: Pathway gene set

        Returns:
            Enrichment score
        """
        hits = []
        no_hits = []

        for gene in genes:
            if gene in pathway_genes:
                hits.append(abs(gene_to_score.get(gene, 0)))
            else:
                no_hits.append(abs(gene_to_score.get(gene, 0)))

        if not hits:
            return 0.0

        N = len(hits) + len(no_hits)
        Nh = len(hits)

        running_sum = 0.0
        max_es = float("-inf")

        for gene in genes:
            if gene in pathway_genes:
                idx = hits.index(abs(gene_to_score.get(gene, 0)))
                running_sum += (1.0 / Nh) - (1.0 / (N - Nh))
                max_es = max(max_es, running_sum)
            else:
                running_sum -= (1.0 / Nh) - (1.0 / (N - Nh))
                max_es = max(max_es, running_sum)

        return max_es

    def _permutation_test(
        self,
        ranked_genes: List[tuple],
        pathway_genes: Set[str],
        observed_es: float,
    ) -> float:
        """Permutation test for p-value.

        Args:
            ranked_genes: Ranked gene list
            pathway_genes: Pathway genes
            observed_es: Observed enrichment score

        Returns:
            P-value
        """
        null_es = []

        for _ in range(self.n_permutations):
            shuffled = [g for g, _ in ranked_genes]
            np.random.shuffle(shuffled)

            gene_to_score = {g: ranked_genes[i][1] for i, g in enumerate(shuffled)}

            perm_es = self._compute_enrichment_score(
                [g for g, _ in ranked_genes],
                gene_to_score,
                pathway_genes,
            )
            null_es.append(perm_es)

        null_es = np.array(null_es)

        return (np.abs(null_es) >= np.abs(observed_es)).mean()


def compute_overlap_coefficient(
    set1: Set[str],
    set2: Set[str],
) -> float:
    """Compute overlap coefficient between two sets.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Overlap coefficient
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    return intersection / min(len(set1), len(set2))


def compute_jaccard_index(
    set1: Set[str],
    set2: Set[str],
) -> float:
    """Compute Jaccard index between two sets.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard index
    """
    if not set1 and not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union
