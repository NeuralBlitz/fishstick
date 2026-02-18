"""
TDA Aggregation Methods.

Provides methods for aggregating topological features
across multiple samples, batches, and scales.
"""

from typing import List, Optional, Dict, Callable
import torch
from torch import Tensor
import numpy as np


class PersistenceAggregator:
    """
    Aggregates persistence diagrams across multiple samples.

    Provides various aggregation strategies for combining
    topological features from multiple point clouds or graphs.
    """

    def __init__(
        self,
        aggregation_type: str = "mean",
    ):
        """
        Initialize aggregator.

        Args:
            aggregation_type: Type of aggregation ('mean', 'max', 'weighted')
        """
        self.aggregation_type = aggregation_type

    def aggregate(
        self,
        diagrams: List[Tensor],
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Aggregate multiple persistence diagrams.

        Args:
            diagrams: List of persistence diagrams
            weights: Optional weights for each diagram

        Returns:
            Aggregated diagram
        """
        if len(diagrams) == 0:
            return torch.zeros(1, 2)

        if self.aggregation_type == "mean":
            return self._mean_aggregate(diagrams, weights)
        elif self.aggregation_type == "max":
            return self._max_aggregate(diagrams)
        elif self.aggregation_type == "weighted":
            return self._weighted_aggregate(diagrams, weights)
        else:
            return self._mean_aggregate(diagrams, weights)

    def _mean_aggregate(
        self,
        diagrams: List[Tensor],
        weights: Optional[Tensor],
    ) -> Tensor:
        """Mean aggregation."""
        all_births = []
        all_deaths = []
        all_dims = []

        for diag in diagrams:
            if len(diag) > 0:
                all_births.append(diag[:, 0])
                all_deaths.append(diag[:, 1])
                if diag.shape[1] > 2:
                    all_dims.append(diag[:, 2])

        if len(all_births) == 0:
            return torch.zeros(1, 2)

        births = torch.cat(all_births)
        deaths = torch.cat(all_deaths)

        return torch.stack([births.mean(), deaths.mean()]).unsqueeze(0)

    def _max_aggregate(
        self,
        diagrams: List[Tensor],
    ) -> Tensor:
        """Max aggregation."""
        max_birth = float("-inf")
        max_death = float("-inf")

        for diag in diagrams:
            if len(diag) > 0:
                max_birth = max(max_birth, diag[:, 0].max().item())
                max_death = max(max_death, diag[:, 1].max().item())

        if max_birth == float("-inf"):
            max_birth = 0
        if max_death == float("-inf"):
            max_death = 0

        return torch.tensor([[max_birth, max_death]], dtype=torch.float32)

    def _weighted_aggregate(
        self,
        diagrams: List[Tensor],
        weights: Optional[Tensor],
    ) -> Tensor:
        """Weighted aggregation."""
        if weights is None:
            weights = torch.ones(len(diagrams)) / len(diagrams)

        weights = weights / weights.sum()

        weighted_births = []
        weighted_deaths = []

        for diag, w in zip(diagrams, weights):
            if len(diag) > 0:
                weighted_births.append(diag[:, 0] * w)
                weighted_deaths.append(diag[:, 1] * w)

        if len(weighted_births) == 0:
            return torch.zeros(1, 2)

        return torch.stack(
            [
                torch.cat(weighted_births).mean(),
                torch.cat(weighted_deaths).mean(),
            ]
        ).unsqueeze(0)


class TopologicalStatistics:
    """
    Computes statistical summaries of persistence diagrams.

    Provides mean, variance, and confidence intervals
    for topological features.
    """

    def __init__(self):
        pass

    def compute_statistics(
        self,
        diagrams: List[Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute statistical summaries.

        Args:
            diagrams: List of persistence diagrams

        Returns:
            Dictionary of statistics
        """
        all_births = []
        all_persistences = []

        for diag in diagrams:
            if len(diag) > 0:
                births = diag[:, 0]
                deaths = diag[:, 1]
                persistences = (deaths - births).clamp(min=0)

                all_births.extend(births.tolist())
                all_persistences.extend(persistences.tolist())

        if len(all_births) == 0:
            return {
                "mean_birth": torch.tensor(0.0),
                "std_birth": torch.tensor(0.0),
                "mean_persistence": torch.tensor(0.0),
                "std_persistence": torch.tensor(0.0),
            }

        return {
            "mean_birth": torch.tensor(np.mean(all_births)),
            "std_birth": torch.tensor(np.std(all_births)),
            "mean_persistence": torch.tensor(np.mean(all_persistences)),
            "std_persistence": torch.tensor(np.std(all_persistences)),
            "min_persistence": torch.tensor(np.min(all_persistences)),
            "max_persistence": torch.tensor(np.max(all_persistences)),
        }


class MultiScaleAggregator:
    """
    Aggregates features across multiple filtration scales.

    Combines topological features computed at different
    scales to create multi-scale representations.
    """

    def __init__(
        self,
        scales: List[float],
        aggregation_type: str = "concatenate",
    ):
        """
        Initialize multi-scale aggregator.

        Args:
            scales: List of filtration scales
            aggregation_type: How to combine scales
        """
        self.scales = scales
        self.aggregation_type = aggregation_type

    def aggregate(
        self,
        multi_scale_diagrams: Dict[float, List[Tensor]],
    ) -> Tensor:
        """
        Aggregate multi-scale diagrams.

        Args:
            multi_scale_diagrams: Dict of scale to diagrams

        Returns:
            Aggregated features
        """
        features = []

        for scale in self.scales:
            if scale in multi_scale_diagrams:
                diagrams = multi_scale_diagrams[scale]

                agg = PersistenceAggregator()
                agg_diag = agg.aggregate(diagrams)

                features.append(agg_diag.flatten())

        if len(features) == 0:
            return torch.zeros(1)

        if self.aggregation_type == "concatenate":
            return torch.cat(features)
        elif self.aggregation_type == "mean":
            return torch.stack(features).mean(dim=0)
        else:
            return torch.cat(features)


class BatchAggregator:
    """
    Aggregates topological features across a batch.

    Efficiently combines features from multiple samples
    in a batch for mini-batch training.
    """

    def __init__(
        self,
        batch_reduction: str = "mean",
    ):
        """
        Initialize batch aggregator.

        Args:
            batch_reduction: Reduction method ('mean', 'max', 'sum')
        """
        self.batch_reduction = batch_reduction

    def aggregate_batch(
        self,
        batch_diagrams: List[List[Tensor]],
    ) -> List[Tensor]:
        """
        Aggregate across batch dimension.

        Args:
            batch_diagrams: List of lists of diagrams (batch x dims)

        Returns:
            List of aggregated diagrams per dimension
        """
        if len(batch_diagrams) == 0:
            return []

        max_dims = max(len(diags) for diags in batch_diagrams)

        aggregated = []

        for dim in range(max_dims):
            dim_diagrams = []

            for sample_diagrams in batch_diagrams:
                if dim < len(sample_diagrams):
                    dim_diagrams.append(sample_diagrams[dim])

            if len(dim_diagrams) > 0:
                agg = PersistenceAggregator()
                aggregated.append(agg.aggregate(dim_diagrams))
            else:
                aggregated.append(torch.zeros(1, 2))

        return aggregated


class AttentionAggregator:
    """
    Attention-based topological feature aggregation.

    Uses learnable attention to weight features
    from different diagrams.
    """

    def __init__(
        self,
        n_samples: int,
        hidden_dim: int = 64,
    ):
        """
        Initialize attention aggregator.

        Args:
            n_samples: Number of samples to aggregate
            hidden_dim: Hidden dimension for attention
        """
        self.n_samples = n_samples
        self.hidden_dim = hidden_dim

        self.attention_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def aggregate(
        self,
        diagrams: List[Tensor],
    ) -> Tensor:
        """
        Aggregate using attention.

        Args:
            diagrams: List of persistence diagrams

        Returns:
            Aggregated features
        """
        if len(diagrams) == 0:
            return torch.zeros(1, 2)

        features = []
        for diag in diagrams:
            if len(diag) > 0:
                features.append(diag)
            else:
                features.append(torch.zeros(1, 2))

        features_tensor = torch.zeros(len(features), 2)
        for i, f in enumerate(features):
            if len(f) > 0:
                features_tensor[i] = f[0]

        attn_weights = self.attention_net(features_tensor)
        attn_weights = torch.softmax(attn_weights, dim=0)

        aggregated = (features_tensor * attn_weights).sum(dim=0, keepdim=True)

        return aggregated


class KernelAggregator:
    """
    Aggregates diagrams using kernel representations.

    Computes kernel matrix between diagrams and uses
    it for aggregation.
    """

    def __init__(
        self,
        kernel_type: str = "linear",
    ):
        """
        Initialize kernel aggregator.

        Args:
            kernel_type: Kernel type
        """
        self.kernel_type = kernel_type

    def aggregate(
        self,
        diagrams: List[Tensor],
    ) -> Tensor:
        """
        Aggregate using kernel methods.

        Args:
            diagrams: List of diagrams

        Returns:
            Aggregated kernel representation
        """
        if len(diagrams) == 0:
            return torch.zeros(1)

        from .kernels import PersistenceScaleSpaceKernel

        kernel = PersistenceScaleSpaceKernel()

        kernel_matrix = torch.zeros(len(diagrams), len(diagrams))

        for i in range(len(diagrams)):
            for j in range(len(diagrams)):
                kernel_matrix[i, j] = kernel.compute(diagrams[i], diagrams[j])

        eigenvalues = torch.linalg.eigvalsh(kernel_matrix)

        return eigenvalues


def aggregate_topological_features(
    features: List[Tensor],
    method: str = "mean",
    weights: Optional[Tensor] = None,
) -> Tensor:
    """
    Convenience function for topological feature aggregation.

    Args:
        features: List of topological feature tensors
        method: Aggregation method
        weights: Optional weights

    Returns:
        Aggregated features
    """
    agg = PersistenceAggregator(aggregation_type=method)
    return agg.aggregate(features, weights)
