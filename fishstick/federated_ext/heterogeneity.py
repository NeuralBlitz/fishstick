"""
Heterogeneous Data Handling for Federated Learning

This module provides tools for handling heterogeneous (non-IID) data:
- Dirichlet-based data partitioning
- Label skew handling
- Feature distribution skew handling
- FedBal: Balanced aggregation
- Local adaptation techniques

References:
- Hsu et al. (2019): "Measuring the Effects of Non-Identical Data Distribution in Federated Learning"
- Li et al. (2020): "Federated Optimization in Heterogeneous Networks"
- Wang et al. (2020): "Federated Learning with Non-IID Data"
- Huang et al. (2022): "FedBal: Rethinking Aggregation for Heterogeneous Data"
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


class HeterogeneityType(Enum):
    """Types of data heterogeneity."""

    IID = auto()
    LABEL_SKEW = auto()
    FEATURE_SKEW = auto()
    QUANTITY_SKEW = auto()
    DIRECTION_SKEW = auto()


@dataclass
class HeterogeneityConfig:
    """Configuration for heterogeneous data handling."""

    heterogeneity_type: HeterogeneityType = HeterogeneityType.IID
    alpha: float = 0.5
    min_samples: int = 10
    max_samples_per_client: Optional[int] = None
    shards_per_client: int = 2
    balance_factor: float = 1.0
    seed: Optional[int] = None


class BaseDataPartitioner(ABC):
    """Base class for heterogeneous data partitioning."""

    def __init__(self, config: HeterogeneityConfig):
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

    @abstractmethod
    def partition(
        self,
        dataset: Dataset,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        """Partition data across clients.

        Args:
            dataset: Dataset to partition
            num_clients: Number of clients

        Returns:
            Dictionary mapping client_id to list of sample indices
        """
        pass


class DirichletPartitioner(BaseDataPartitioner):
    """Dirichlet-based non-IID data partitioning.

    Uses Dirichlet distribution to create heterogeneous label distributions.

    Reference: Hsu et al. (2019)
    """

    def partition(
        self,
        dataset: Dataset,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        if self.config.heterogeneity_type == HeterogeneityType.IID:
            return self._partition_iid(dataset, num_clients)

        labels = self._get_labels(dataset)
        num_classes = int(np.max(labels) + 1)

        indices_per_class = self._get_indices_per_class(labels)

        partitions: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

        for class_idx in range(num_classes):
            random.shuffle(indices_per_class[class_idx])

            proportions = np.random.dirichlet([self.config.alpha] * num_clients)

            class_indices = indices_per_class[class_idx]
            splits = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            split_indices = np.split(class_indices, splits)

            for client_id, client_indices in enumerate(split_indices):
                partitions[client_id].extend(client_indices.tolist())

        for client_id in partitions:
            if len(partitions[client_id]) < self.config.min_samples:
                logger.warning(
                    f"Client {client_id} has fewer than {self.config.min_samples} samples"
                )

        logger.info(
            f"Dirichlet partition created: {num_classes} classes, alpha={self.config.alpha}"
        )
        return partitions

    def _partition_iid(
        self,
        dataset: Dataset,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        samples_per_client = len(indices) // num_clients
        remainder = len(indices) % num_clients

        partitions: Dict[int, List[int]] = {}
        start = 0

        for client_id in range(num_clients):
            end = start + samples_per_client + (1 if client_id < remainder else 0)
            partitions[client_id] = indices[start:end]
            start = end

        return partitions

    def _get_labels(self, dataset: Dataset) -> np.ndarray:
        if hasattr(dataset, "targets"):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, "labels"):
            labels = np.array(dataset.labels)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        return labels

    def _get_indices_per_class(
        self,
        labels: np.ndarray,
    ) -> List[List[int]]:
        num_classes = int(np.max(labels) + 1)
        indices_per_class = [[] for _ in range(num_classes)]

        for idx, label in enumerate(labels):
            indices_per_class[int(label)].append(idx)

        return indices_per_class


class ShardPartitioner(BaseDataPartitioner):
    """Shard-based partitioning for label skew.

    Each client receives a fixed number of shards (subsets of classes).

    Reference: Li et al. (2020)
    """

    def partition(
        self,
        dataset: Dataset,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        labels = self._get_labels(dataset)
        num_classes = int(np.max(labels) + 1)

        indices_per_class = self._get_indices_per_class(labels)

        num_shards = num_classes * self.config.shards_per_client
        shard_size = len(dataset) // num_shards

        shards = []
        for class_indices in indices_per_class:
            random.shuffle(class_indices)
            for i in range(self.config.shards_per_client):
                start = i * shard_size
                end = (
                    (i + 1) * shard_size
                    if i < self.config.shards_per_client - 1
                    else len(class_indices)
                )
                shards.append(class_indices[start:end])

        random.shuffle(shards)

        partitions: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
        shards_per_client = num_shards // num_clients

        for client_id in range(num_clients):
            start = client_id * shards_per_client
            end = (
                (client_id + 1) * shards_per_client
                if client_id < num_clients - 1
                else num_shards
            )
            for shard in shards[start:end]:
                partitions[client_id].extend(shard)

        logger.info(
            f"Shard partition created: {num_clients} clients, {self.config.shards_per_client} shards per class"
        )
        return partitions

    def _get_labels(self, dataset: Dataset) -> np.ndarray:
        if hasattr(dataset, "targets"):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, "labels"):
            labels = np.array(dataset.labels)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        return labels

    def _get_indices_per_class(
        self,
        labels: np.ndarray,
    ) -> List[List[int]]:
        num_classes = int(np.max(labels) + 1)
        indices_per_class = [[] for _ in range(num_classes)]

        for idx, label in enumerate(labels):
            indices_per_class[int(label)].append(idx)

        return indices_per_class


class QuantitySkewPartitioner(BaseDataPartitioner):
    """Quantity skew partitioning.

    Creates imbalance in the number of samples per client while
    maintaining IID distribution within each client's data.
    """

    def partition(
        self,
        dataset: Dataset,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        total_samples = len(indices)

        proportions = np.random.dirichlet([self.config.alpha] * num_clients)

        sample_counts = (proportions * total_samples).astype(int)
        sample_counts = np.maximum(sample_counts, self.config.min_samples)

        diff = sum(sample_counts) - total_samples
        if diff > 0:
            sample_counts[np.argmax(sample_counts)] -= diff

        partitions: Dict[int, List[int]] = {}
        start = 0

        for client_id in range(num_clients):
            end = start + sample_counts[client_id]
            partitions[client_id] = indices[start:end]
            start = end

        logger.info(
            f"Quantity skew partition created: {num_clients} clients with varying sample counts"
        )
        return partitions


class FeatureSkewPartitioner(BaseDataPartitioner):
    """Feature skew partitioning.

    Creates heterogeneity in feature distributions across clients
    using synthetic feature transformations.
    """

    def __init__(self, config: HeterogeneityConfig):
        super().__init__(config)
        self.feature_transforms: Dict[int, Any] = {}

    def partition(
        self,
        dataset: Dataset,
        num_clients: int,
    ) -> Dict[int, List[int]]:
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        samples_per_client = len(indices) // num_clients
        remainder = len(indices) % num_clients

        partitions: Dict[int, List[int]] = {}
        start = 0

        for client_id in range(num_clients):
            end = start + samples_per_client + (1 if client_id < remainder else 0)
            partitions[client_id] = indices[start:end]

            self.feature_transforms[client_id] = self._create_transform(client_id)
            start = end

        logger.info(
            f"Feature skew partition created: {num_clients} clients with different feature distributions"
        )
        return partitions

    def _create_transform(self, client_id: int) -> Dict[str, Any]:
        np.random.seed(self.config.seed + client_id if self.config.seed else client_id)
        return {
            "scale": np.random.uniform(0.8, 1.2),
            "shift": np.random.uniform(-0.2, 0.2),
        }

    def apply_transform(
        self,
        client_id: int,
        data: Tensor,
    ) -> Tensor:
        if client_id not in self.feature_transforms:
            return data

        transform = self.feature_transforms[client_id]
        return data * transform["scale"] + transform["shift"]


class BalancedAggregator:
    """FedBal: Balanced aggregation for heterogeneous data.

    Adjusts aggregation weights to account for data heterogeneity.

    Reference: Huang et al. (2022)
    """

    def __init__(self, config: HeterogeneityConfig):
        self.config = config
        self.client_data_stats: Dict[int, Dict[str, float]] = {}

    def compute_balanced_weights(
        self,
        client_weights: List[float],
        client_sample_counts: List[int],
    ) -> List[float]:
        """Compute balanced weights for aggregation.

        Args:
            client_weights: Original client weights (e.g., sample counts)
            client_sample_counts: Number of samples per client

        Returns:
            Balanced weights
        """
        if self.config.balance_factor == 1.0:
            total = sum(client_weights)
            return [w / total for w in client_weights]

        total_samples = sum(client_sample_counts)
        balanced_weights = []

        for original_weight, sample_count in zip(client_weights, client_sample_counts):
            client_share = sample_count / total_samples
            balanced_weight = self.config.balance_factor * client_share + (
                1 - self.config.balance_factor
            ) * (original_weight / sum(client_weights))
            balanced_weights.append(balanced_weight)

        total = sum(balanced_weights)
        return [w / total for w in balanced_weights]

    def update_client_stats(
        self,
        client_id: int,
        label_distribution: Dict[int, int],
    ) -> None:
        """Update client data statistics.

        Args:
            client_id: Client identifier
            label_distribution: Distribution of labels for this client
        """
        total = sum(label_distribution.values())
        normalized_dist = {k: v / total for k, v in label_distribution.items()}

        self.client_data_stats[client_id] = {
            "distribution": normalized_dist,
            "total_samples": total,
            "num_classes": len(label_distribution),
        }

    def compute_distribution_divergence(
        self,
        client_a: int,
        client_b: int,
    ) -> float:
        """Compute KL divergence between client distributions."""
        if (
            client_a not in self.client_data_stats
            or client_b not in self.client_data_stats
        ):
            return 0.0

        dist_a = self.client_data_stats[client_a]["distribution"]
        dist_b = self.client_data_stats[client_b]["distribution"]

        all_classes = set(dist_a.keys()) | set(dist_b.keys())
        p = np.array([dist_a.get(c, 1e-10) for c in all_classes])
        q = np.array([dist_b.get(c, 1e-10) for c in all_classes])

        p = p / p.sum()
        q = q / q.sum()

        kl_div = np.sum(p * np.log(p / q))
        return float(kl_div)


class LocalAdaptation:
    """Local adaptation techniques for heterogeneous data.

    Provides methods for adapting models to local data distributions.
    """

    def __init__(self, config: HeterogeneityConfig):
        self.config = config
        self.global_model_state: Optional[Dict[str, Tensor]] = None

    def set_global_state(self, state: Dict[str, Tensor]) -> None:
        """Set global model state."""
        self.global_model_state = {k: v.clone() for k, v in state.items()}

    def proximal_regularization(
        self,
        local_model: nn.Module,
        mu: float = 0.01,
    ) -> Tensor:
        """Compute proximal regularization loss.

        Args:
            local_model: Local model
            mu: Regularization coefficient

        Returns:
            Regularization loss
        """
        if self.global_model_state is None:
            return torch.tensor(0.0)

        loss = 0.0
        for (name, param), (global_name, global_param) in zip(
            local_model.named_parameters(), self.global_model_state.items()
        ):
            loss += torch.sum((param - global_param) ** 2)

        return mu * loss

    def knowledge_distillation(
        self,
        local_logits: Tensor,
        global_logits: Tensor,
        temperature: float = 2.0,
    ) -> Tensor:
        """Compute knowledge distillation loss.

        Args:
            local_logits: Local model predictions
            global_logits: Global model predictions
            temperature: Temperature for softening predictions

        Returns:
            Distillation loss
        """
        soft_global = torch.softmax(global_logits / temperature, dim=-1)
        soft_local = torch.log_softmax(local_logits / temperature, dim=-1)

        loss = torch.sum(-soft_global * soft_local, dim=-1)
        return loss.mean()

    def compute_adaptive_lr(
        self,
        client_sample_count: int,
        total_samples: int,
        base_lr: float = 0.01,
    ) -> float:
        """Compute adaptive learning rate based on client data size.

        Args:
            client_sample_count: Number of samples on client
            total_samples: Total samples across all clients
            base_lr: Base learning rate

        Returns:
            Adaptive learning rate
        """
        if self.config.heterogeneity_type == HeterogeneityType.IID:
            return base_lr

        client_share = client_sample_count / total_samples
        adaptive_lr = base_lr * (1 + np.log(1 / (client_share + 1e-10)))

        return min(adaptive_lr, base_lr * 10)


def create_partitioner(config: HeterogeneityConfig) -> BaseDataPartitioner:
    """Factory function to create data partitioner.

    Args:
        config: Configuration for the partitioner

    Returns:
        Instance of the appropriate partitioner

    Example:
        >>> config = HeterogeneityConfig(heterogeneity_type=HeterogeneityType.LABEL_SKEW, alpha=0.5)
        >>> partitioner = create_partitioner(config)
    """
    if config.heterogeneity_type == HeterogeneityType.IID:
        return DirichletPartitioner(config)
    elif config.heterogeneity_type == HeterogeneityType.LABEL_SKEW:
        return ShardPartitioner(config)
    elif config.heterogeneity_type == HeterogeneityType.QUANTITY_SKEW:
        return QuantitySkewPartitioner(config)
    elif config.heterogeneity_type == HeterogeneityType.FEATURE_SKEW:
        return FeatureSkewPartitioner(config)
    else:
        return DirichletPartitioner(config)


__all__ = [
    "HeterogeneityType",
    "HeterogeneityConfig",
    "BaseDataPartitioner",
    "DirichletPartitioner",
    "ShardPartitioner",
    "QuantitySkewPartitioner",
    "FeatureSkewPartitioner",
    "BalancedAggregator",
    "LocalAdaptation",
    "create_partitioner",
]
