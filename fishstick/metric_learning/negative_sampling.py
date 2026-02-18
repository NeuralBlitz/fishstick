"""
Hard Negative Sampling for Metric Learning

Implementation of various hard negative sampling strategies:
- Random negative sampling
- Semi-hard negative sampling
- Hardest negative sampling
- Distance-weighted negative sampling
- Curriculum negative sampling
"""

from typing import Optional, Tuple, List
from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class NegativeSampler(ABC):
    """Base class for negative sampling strategies."""

    @abstractmethod
    def sample(
        self,
        embeddings: Tensor,
        labels: Tensor,
        anchor_indices: Optional[Tensor] = None,
        num_negatives: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """Sample negative examples.

        Args:
            embeddings: All embeddings (batch_size, dim)
            labels: Class labels (batch_size,)
            anchor_indices: Indices of anchor samples
            num_negatives: Number of negatives to sample per anchor

        Returns:
            Tuple of (negative_indices, negative_distances)
        """
        pass


class RandomNegativeSampler(NegativeSampler):
    """Random negative sampling strategy.

    Randomly selects negative samples from different classes.
    """

    def __init__(self, replace: bool = False):
        self.replace = replace

    def sample(
        self,
        embeddings: Tensor,
        labels: Tensor,
        anchor_indices: Optional[Tensor] = None,
        num_negatives: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        if anchor_indices is None:
            anchor_indices = torch.arange(batch_size, device=device)

        labels = labels.to(device)
        embeddings = F.normalize(embeddings, dim=-1)

        negative_indices_list = []
        distances_list = []

        for anchor_idx in anchor_indices:
            anchor_label = labels[anchor_idx]

            different_class_mask = labels != anchor_label
            different_class_indices = torch.where(different_class_mask)[0]

            if len(different_class_indices) > 0:
                selected_indices = different_class_indices[
                    torch.randint(
                        len(different_class_indices),
                        (num_negatives,),
                        replacement=self.replace,
                    )
                ]
            else:
                selected_indices = torch.randint(
                    0, batch_size, (num_negatives,), device=device
                )

            negative_indices_list.append(selected_indices)

            dists = torch.cdist(
                embeddings[anchor_idx : anchor_idx + 1],
                embeddings[selected_indices],
                p=2,
            ).squeeze(0)
            distances_list.append(dists)

        negative_indices = torch.stack(negative_indices_list)
        distances = torch.stack(distances_list)

        return negative_indices, distances


class SemihardNegativeSampler(NegativeSampler):
    """Semi-hard negative sampling strategy.

    Selects negatives that are further than the positive distance but within a margin.
    """

    def __init__(
        self,
        margin: float = 1.0,
        num_negatives: int = 1,
    ):
        self.margin = margin
        self.num_negatives = num_negatives

    def sample(
        self,
        embeddings: Tensor,
        labels: Tensor,
        anchor_indices: Optional[Tensor] = None,
        num_negatives: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        if anchor_indices is None:
            anchor_indices = torch.arange(batch_size, device=device)

        labels = labels.to(device)
        embeddings = F.normalize(embeddings, dim=-1)

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        negative_indices_list = []
        distances_list = []

        for anchor_idx in anchor_indices:
            anchor_label = labels[anchor_idx]

            same_class_mask = labels == anchor_label
            same_class_indices = torch.where(same_class_mask)[0]
            same_class_indices = same_class_indices[same_class_indices != anchor_idx]

            if len(same_class_indices) > 0:
                pos_dist = dist_matrix[anchor_idx, same_class_indices].max()
            else:
                pos_dist = torch.tensor(0.0, device=device)

            different_class_mask = labels != anchor_label
            different_class_indices = torch.where(different_class_mask)[0]

            if len(different_class_indices) > 0:
                neg_dists = dist_matrix[anchor_idx, different_class_indices]

                semihard_mask = (neg_dists > pos_dist) & (
                    neg_dists < pos_dist + self.margin
                )
                semihard_indices = different_class_indices[semihard_mask]

                if len(semihard_indices) > 0:
                    selected_indices = semihard_indices[
                        torch.randperm(len(semihard_indices))[:num_negatives]
                    ]
                    if len(selected_indices) < num_negatives:
                        remaining = different_class_indices[~semihard_mask]
                        additional = remaining[
                            torch.randperm(len(remaining))[
                                : num_negatives - len(selected_indices)
                            ]
                        ]
                        selected_indices = torch.cat([selected_indices, additional])
                else:
                    available = different_class_indices[
                        torch.randperm(len(different_class_indices))[:num_negatives]
                    ]
                    selected_indices = available
            else:
                selected_indices = torch.randint(
                    0, batch_size, (num_negatives,), device=device
                )

            negative_indices_list.append(selected_indices)

            dists = torch.cdist(
                embeddings[anchor_idx : anchor_idx + 1],
                embeddings[selected_indices],
                p=2,
            ).squeeze(0)
            distances_list.append(dists)

        negative_indices = torch.stack(negative_indices_list)
        distances = torch.stack(distances_list)

        return negative_indices, distances


class HardestNegativeSampler(NegativeSampler):
    """Hardest negative sampling strategy.

    Selects the hardest (closest) negatives from different classes.
    """

    def __init__(
        self,
        num_negatives: int = 1,
        strict: bool = False,
    ):
        self.num_negatives = num_negatives
        self.strict = strict

    def sample(
        self,
        embeddings: Tensor,
        labels: Tensor,
        anchor_indices: Optional[Tensor] = None,
        num_negatives: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        if anchor_indices is None:
            anchor_indices = torch.arange(batch_size, device=device)

        labels = labels.to(device)
        embeddings = F.normalize(embeddings, dim=-1)

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        negative_indices_list = []
        distances_list = []

        for anchor_idx in anchor_indices:
            anchor_label = labels[anchor_idx]

            different_class_mask = labels != anchor_label
            different_class_indices = torch.where(different_class_mask)[0]

            if len(different_class_indices) > 0:
                neg_dists = dist_matrix[anchor_idx, different_class_indices]

                sorted_indices = different_class_indices[torch.argsort(neg_dists)]
                selected_indices = sorted_indices[:num_negatives]
            else:
                selected_indices = torch.tensor([], dtype=torch.long, device=device)
                if self.strict:
                    raise ValueError(
                        f"No negative samples available for anchor {anchor_idx}"
                    )

            if len(selected_indices) < num_negatives:
                padding = torch.randint(
                    0,
                    batch_size,
                    (num_negatives - len(selected_indices),),
                    device=device,
                )
                selected_indices = torch.cat([selected_indices, padding])

            negative_indices_list.append(selected_indices)
            distances_list.append(neg_dists[torch.argsort(neg_dists)[:num_negatives]])

        negative_indices = torch.stack(negative_indices_list)
        distances = torch.stack(distances_list)

        return negative_indices, distances


class DistanceWeightedNegativeSampler(NegativeSampler):
    """Distance-weighted negative sampling strategy.

    Samples negatives with probability proportional to their distance.
    """

    def __init__(
        self,
        num_negatives: int = 5,
        weight_by_distance: bool = True,
        temperature: float = 1.0,
    ):
        self.num_negatives = num_negatives
        self.weight_by_distance = weight_by_distance
        self.temperature = temperature

    def sample(
        self,
        embeddings: Tensor,
        labels: Tensor,
        anchor_indices: Optional[Tensor] = None,
        num_negatives: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        if anchor_indices is None:
            anchor_indices = torch.arange(batch_size, device=device)

        labels = labels.to(device)
        embeddings = F.normalize(embeddings, dim=-1)

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        negative_indices_list = []
        distances_list = []

        for anchor_idx in anchor_indices:
            anchor_label = labels[anchor_idx]

            different_class_mask = labels != anchor_label
            different_class_indices = torch.where(different_class_mask)[0]

            if len(different_class_indices) > 0:
                neg_dists = dist_matrix[anchor_idx, different_class_indices]

                if self.weight_by_distance:
                    weights = F.softmax(neg_dists / self.temperature, dim=0)
                    selected_indices = different_class_indices[
                        torch.multinomial(weights, num_negatives, replacement=False)
                    ]
                else:
                    selected_indices = different_class_indices[
                        torch.randperm(len(different_class_indices))[:num_negatives]
                    ]
            else:
                selected_indices = torch.randint(
                    0, batch_size, (num_negatives,), device=device
                )

            negative_indices_list.append(selected_indices)
            distances_list.append(dist_matrix[anchor_idx, selected_indices])

        negative_indices = torch.stack(negative_indices_list)
        distances = torch.stack(distances_list)

        return negative_indices, distances


class CurriculumNegativeSampler(NegativeSampler):
    """Curriculum negative sampling strategy.

    Gradually increases difficulty of negatives over training time.
    Starts with easy negatives and progresses to hard negatives.
    """

    def __init__(
        self,
        initial_strategy: str = "random",
        final_strategy: str = "hardest",
        num_epochs: int = 100,
        num_negatives: int = 5,
    ):
        self.initial_strategy = initial_strategy
        self.final_strategy = final_strategy
        self.num_epochs = num_epochs
        self.num_negatives = num_negatives

        self._random_sampler = RandomNegativeSampler()
        self._semihard_sampler = SemihardNegativeSampler()
        self._hardest_sampler = HardestNegativeSampler()

    def sample(
        self,
        embeddings: Tensor,
        labels: Tensor,
        anchor_indices: Optional[Tensor] = None,
        num_negatives: int = 1,
        current_epoch: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        progress = min(current_epoch / self.num_epochs, 1.0)

        if progress < 0.33:
            return self._random_sampler.sample(
                embeddings, labels, anchor_indices, num_negatives
            )
        elif progress < 0.66:
            return self._semihard_sampler.sample(
                embeddings, labels, anchor_indices, num_negatives
            )
        else:
            return self._hardest_sampler.sample(
                embeddings, labels, anchor_indices, num_negatives
            )


class InformedNegativeSampler(NegativeSampler):
    """Informed negative sampling using class statistics.

    Uses class distribution information to select more informative negatives.
    """

    def __init__(
        self,
        num_negatives: int = 5,
        alpha: float = 0.5,
    ):
        self.num_negatives = num_negatives
        self.alpha = alpha

    def sample(
        self,
        embeddings: Tensor,
        labels: Tensor,
        anchor_indices: Optional[Tensor] = None,
        num_negatives: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        if anchor_indices is None:
            anchor_indices = torch.arange(batch_size, device=device)

        labels = labels.to(device)
        embeddings = F.normalize(embeddings, dim=-1)

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        unique_labels = labels.unique()
        label_to_indices = {
            label: torch.where(labels == label)[0] for label in unique_labels
        }

        negative_indices_list = []
        distances_list = []

        for anchor_idx in anchor_indices:
            anchor_label = labels[anchor_idx]

            different_class_indices_list = [
                idx for label, idx in label_to_indices.items() if label != anchor_label
            ]

            if len(different_class_indices_list) > 0:
                class_dists = []
                for neg_indices in different_class_indices_list:
                    avg_dist = dist_matrix[anchor_idx, neg_indices].mean()
                    class_dists.append(avg_dist)

                class_dists = torch.tensor(class_dists, device=device)

                class_probs = F.softmax(class_dists / self.alpha, dim=0)

                selected_classes = torch.multinomial(
                    class_probs, num_negatives, replacement=True
                )

                selected_indices = []
                for class_idx in selected_classes:
                    class_indices = different_class_indices_list[class_idx]
                    selected = class_indices[torch.randint(len(class_indices), (1,))]
                    selected_indices.append(selected)

                selected_indices = torch.cat(selected_indices)
            else:
                selected_indices = torch.randint(
                    0, batch_size, (num_negatives,), device=device
                )

            negative_indices_list.append(selected_indices)
            distances_list.append(dist_matrix[anchor_idx, selected_indices])

        negative_indices = torch.stack(negative_indices_list)
        distances = torch.stack(distances_list)

        return negative_indices, distances


class BatchNegativeSampler(NegativeSampler):
    """Batch-level negative sampling for efficiency.

    Samples negatives from the entire batch.
    """

    def __init__(
        self,
        strategy: str = "random",
        num_negatives: int = 4,
    ):
        self.strategy = strategy.lower()
        self.num_negatives = num_negatives

    def sample(
        self,
        embeddings: Tensor,
        labels: Tensor,
        anchor_indices: Optional[Tensor] = None,
        num_negatives: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        if anchor_indices is None:
            anchor_indices = torch.arange(batch_size, device=device)

        labels = labels.to(device)
        embeddings = F.normalize(embeddings, dim=-1)

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        all_indices = torch.arange(batch_size, device=device)

        negative_indices_list = []
        distances_list = []

        for anchor_idx in anchor_indices:
            anchor_label = labels[anchor_idx]

            valid_negatives = all_indices[labels != anchor_label]

            if len(valid_negatives) > 0:
                if self.strategy == "random":
                    selected = valid_negatives[
                        torch.randperm(len(valid_negatives))[:num_negatives]
                    ]
                elif self.strategy == "hardest":
                    neg_dists = dist_matrix[anchor_idx, valid_negatives]
                    selected = valid_negatives[torch.argsort(neg_dists)[:num_negatives]]
                elif self.strategy == "semihard":
                    pos_dist = dist_matrix[
                        anchor_idx, all_indices[labels == anchor_label]
                    ].max()
                    valid = valid_negatives[
                        dist_matrix[anchor_idx, valid_negatives] > pos_dist
                    ]
                    if len(valid) > 0:
                        selected = valid[torch.randperm(len(valid))[:num_negatives]]
                    else:
                        selected = valid_negatives[
                            torch.randperm(len(valid_negatives))[:num_negatives]
                        ]
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
            else:
                selected = torch.randint(0, batch_size, (num_negatives,), device=device)

            negative_indices_list.append(selected)
            distances_list.append(dist_matrix[anchor_idx, selected])

        negative_indices = torch.stack(negative_indices_list)
        distances = torch.stack(distances_list)

        return negative_indices, distances


def create_negative_sampler(
    strategy: str,
    **kwargs,
) -> NegativeSampler:
    """Create a negative sampler based on strategy name.

    Args:
        strategy: Sampling strategy name
        **kwargs: Additional arguments for the sampler

    Returns:
        NegativeSampler instance
    """
    samplers = {
        "random": RandomNegativeSampler,
        "semihard": SemihardNegativeSampler,
        "hardest": HardestNegativeSampler,
        "distance_weighted": DistanceWeightedNegativeSampler,
        "curriculum": CurriculumNegativeSampler,
        "informed": InformedNegativeSampler,
        "batch": BatchNegativeSampler,
    }

    strategy = strategy.lower()
    if strategy not in samplers:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(samplers.keys())}"
        )

    return samplers[strategy](**kwargs)


__all__ = [
    "NegativeSampler",
    "RandomNegativeSampler",
    "SemihardNegativeSampler",
    "HardestNegativeSampler",
    "DistanceWeightedNegativeSampler",
    "CurriculumNegativeSampler",
    "InformedNegativeSampler",
    "BatchNegativeSampler",
    "create_negative_sampler",
]
