"""
Triplet Mining Strategies for Metric Learning

Implementation of various triplet mining strategies:
- Random triplet mining
- Hard negative mining (semihard, hardest)
- Distance-weighted triplet mining
- Angular triplet mining
- Batch all and batch hard strategies
"""

from typing import Optional, Tuple, List
from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TripletMiner(ABC):
    """Base class for triplet mining strategies.

    Triplet mining selects triplets (anchor, positive, negative) for training
    where:
    - Anchor and positive should be from the same class
    - Anchor and negative should be from different classes
    """

    @abstractmethod
    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Mine triplets from embeddings and labels.

        Args:
            embeddings: Embeddings of shape (batch_size, dim)
            labels: Class labels of shape (batch_size,)

        Returns:
            Tuple of (anchors, positives, negatives) indices
        """
        pass


class RandomTripletMiner(TripletMiner):
    """Random triplet mining strategy.

    Randomly selects positive and negative pairs for each anchor.
    """

    def __init__(self, num_triplets: Optional[int] = None):
        self.num_triplets = num_triplets

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        if self.num_triplets is None:
            num_triplets = batch_size
        else:
            num_triplets = self.num_triplets

        anchors = torch.randint(0, batch_size, (num_triplets,), device=device)

        positive_indices = []
        negative_indices = []

        for i, anchor_idx in enumerate(anchors):
            anchor_label = labels[anchor_idx].item()

            same_class_mask = labels == anchor_label
            same_class_indices = torch.where(same_class_mask)[0]
            same_class_indices = same_class_indices[same_class_indices != anchor_idx]

            if len(same_class_indices) > 0:
                pos_idx = same_class_indices[
                    torch.randint(len(same_class_indices), (1,))
                ].item()
            else:
                pos_idx = anchor_idx.item()

            different_class_mask = labels != anchor_label
            different_class_indices = torch.where(different_class_mask)[0]

            if len(different_class_indices) > 0:
                neg_idx = different_class_indices[
                    torch.randint(len(different_class_indices), (1,))
                ].item()
            else:
                neg_idx = (anchor_idx.item() + 1) % batch_size

            positive_indices.append(pos_idx)
            negative_indices.append(neg_idx)

        positives = torch.tensor(positive_indices, device=device)
        negatives = torch.tensor(negative_indices, device=device)

        return anchors, positives, negatives


class HardNegativeMiner(TripletMiner):
    """Hard negative mining strategy.

    Selects the hardest negative for each anchor (closest negative from different class).

    Args:
        margin: Margin for valid triplets
        mining_strategy: 'hardest', 'semihard', or 'random'
    """

    def __init__(
        self,
        margin: float = 0.1,
        mining_strategy: str = "semihard",
    ):
        self.margin = margin
        self.mining_strategy = mining_strategy.lower()

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        labels = labels.to(device)
        embeddings = F.normalize(embeddings, dim=-1)

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        anchors = torch.arange(batch_size, device=device)

        positives_list = []
        negatives_list = []

        for anchor_idx in anchors:
            anchor_label = labels[anchor_idx]

            same_class_mask = labels == anchor_label
            same_class_indices = torch.where(same_class_mask)[0]
            same_class_indices = same_class_indices[same_class_indices != anchor_idx]

            if len(same_class_indices) > 0:
                pos_dists = dist_matrix[anchor_idx, same_class_indices]
                pos_idx = same_class_indices[pos_dists.argmax()]
            else:
                pos_idx = anchor_idx

            different_class_mask = labels != anchor_label
            different_class_indices = torch.where(different_class_mask)[0]

            if len(different_class_indices) > 0:
                neg_dists = dist_matrix[anchor_idx, different_class_indices]

                if self.mining_strategy == "hardest":
                    neg_idx = different_class_indices[neg_dists.argmin()]
                elif self.mining_strategy == "semihard":
                    valid_neg_mask = neg_dists > self.margin
                    if valid_neg_mask.any():
                        valid_neg_indices = different_class_indices[valid_neg_mask]
                        neg_idx = valid_neg_indices[neg_dists[valid_neg_mask].argmax()]
                    else:
                        neg_idx = different_class_indices[neg_dists.argmin()]
                else:
                    neg_idx = different_class_indices[
                        torch.randint(len(different_class_indices), (1,))
                    ]
            else:
                neg_idx = (anchor_idx + 1) % batch_size

            positives_list.append(pos_idx)
            negatives_list.append(neg_idx)

        positives = torch.stack(positives_list)
        negatives = torch.stack(negatives_list)

        return anchors, positives, negatives


class SemihardNegativeMiner(TripletMiner):
    """Semihard negative mining strategy.

    Selects negatives that are further than the positive distance but within margin.
    """

    def __init__(self, margin: float = 1.0):
        self.margin = margin

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        labels = labels.to(device)
        embeddings = F.normalize(embeddings, dim=-1)

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        anchors = torch.arange(batch_size, device=device)

        positives_list = []
        negatives_list = []

        for anchor_idx in anchors:
            anchor_label = labels[anchor_idx]

            same_class_mask = labels == anchor_label
            same_class_indices = torch.where(same_class_mask)[0]
            same_class_indices = same_class_indices[same_class_indices != anchor_idx]

            if len(same_class_indices) > 0:
                pos_dists = dist_matrix[anchor_idx, same_class_indices]
                pos_idx = same_class_indices[pos_dists.argmax()]
                pos_dist = pos_dists.max()
            else:
                pos_idx = anchor_idx
                pos_dist = torch.tensor(0.0, device=device)

            different_class_mask = labels != anchor_label
            different_class_indices = torch.where(different_class_mask)[0]

            if len(different_class_indices) > 0:
                neg_dists = dist_matrix[anchor_idx, different_class_indices]

                semihard_mask = (neg_dist > pos_dist) & (
                    neg_dist < pos_dist + self.margin
                )
                semihard_indices = different_class_indices[semihard_mask]

                if len(semihard_indices) > 0:
                    neg_idx = semihard_indices[neg_dists[semihard_mask].argmax()]
                else:
                    neg_idx = different_class_indices[neg_dists.argmin()]
            else:
                neg_idx = (anchor_idx + 1) % batch_size

            positives_list.append(pos_idx)
            negatives_list.append(neg_idx)

        positives = torch.stack(positives_list)
        negatives = torch.stack(negatives_list)

        return anchors, positives, negatives


class DistanceWeightedMiner(TripletMiner):
    """Distance-weighted triplet mining.

    Samples negatives with probability proportional to their distance.
    """

    def __init__(
        self,
        num_negatives: int = 5,
        weight_by_distance: bool = True,
    ):
        self.num_negatives = num_negatives
        self.weight_by_distance = weight_by_distance

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        labels = labels.to(device)
        embeddings = F.normalize(embeddings, dim=-1)

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        anchors = torch.arange(batch_size, device=device)

        positives_list = []
        negatives_list = []

        for anchor_idx in anchors:
            anchor_label = labels[anchor_idx]

            same_class_mask = labels == anchor_label
            same_class_indices = torch.where(same_class_mask)[0]
            same_class_indices = same_class_indices[same_class_indices != anchor_idx]

            if len(same_class_indices) > 0:
                pos_idx = same_class_indices[
                    torch.randint(len(same_class_indices), (1,))
                ]
            else:
                pos_idx = anchor_idx.unsqueeze(0)

            different_class_mask = labels != anchor_label
            different_class_indices = torch.where(different_class_mask)[0]

            if len(different_class_indices) > 0:
                neg_dists = dist_matrix[anchor_idx, different_class_indices]

                if self.weight_by_distance:
                    probs = neg_dists / (neg_dists.sum() + 1e-8)
                    neg_indices = different_class_indices[
                        torch.multinomial(probs, self.num_negatives, replacement=False)
                    ]
                else:
                    neg_indices = different_class_indices[
                        torch.randint(
                            len(different_class_indices), (self.num_negatives,)
                        )
                    ]
            else:
                neg_indices = torch.randint(
                    0, batch_size, (self.num_negatives,), device=device
                )

            positives_list.append(pos_idx)
            negatives_list.append(neg_indices)

        positives = torch.cat(positives_list)
        negatives = torch.cat(negatives_list)

        anchors = anchors.repeat_interleave(self.num_negatives)

        return anchors, positives, negatives


class AngularTripletMiner(TripletMiner):
    """Angular triplet mining using cosine similarity.

    Mines triplets based on angular distance rather than Euclidean.
    """

    def __init__(
        self,
        margin: float = 0.0,
        mining_strategy: str = "hard",
    ):
        self.margin = margin
        self.mining_strategy = mining_strategy.lower()

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        labels = labels.to(device)
        embeddings = F.normalize(embeddings, dim=-1)

        cos_matrix = torch.mm(embeddings, embeddings.T)

        anchors = torch.arange(batch_size, device=device)

        positives_list = []
        negatives_list = []

        for anchor_idx in anchors:
            anchor_label = labels[anchor_idx]

            same_class_mask = labels == anchor_label
            same_class_indices = torch.where(same_class_mask)[0]
            same_class_indices = same_class_indices[same_class_indices != anchor_idx]

            if len(same_class_indices) > 0:
                pos_sims = cos_matrix[anchor_idx, same_class_indices]
                pos_idx = same_class_indices[pos_sims.argmax()]
            else:
                pos_idx = anchor_idx

            different_class_mask = labels != anchor_label
            different_class_indices = torch.where(different_class_mask)[0]

            if len(different_class_indices) > 0:
                neg_sims = cos_matrix[anchor_idx, different_class_indices]

                if self.mining_strategy == "hard":
                    neg_idx = different_class_indices[neg_sims.argmax()]
                elif self.mining_strategy == "semihard":
                    valid_mask = neg_sims < (
                        cos_matrix[anchor_idx, pos_idx] - self.margin
                    )
                    if valid_mask.any():
                        valid_indices = different_class_indices[valid_mask]
                        neg_idx = valid_indices[neg_sims[valid_mask].argmax()]
                    else:
                        neg_idx = different_class_indices[neg_sims.argmax()]
                else:
                    neg_idx = different_class_indices[
                        torch.randint(len(different_class_indices), (1,))
                    ]
            else:
                neg_idx = (anchor_idx + 1) % batch_size

            positives_list.append(pos_idx)
            negatives_list.append(neg_idx)

        positives = torch.stack(positives_list)
        negatives = torch.stack(negatives_list)

        return anchors, positives, negatives


class BatchAllTripletMiner(TripletMiner):
    """Batch all triplet mining strategy.

    Creates all valid triplets within a batch.
    """

    def __init__(self, margin: float = 0.0):
        self.margin = margin

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        labels = labels.to(device)

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_neq = ~labels_eq

        pos_mask = labels_eq.fill_diagonal_(False)
        neg_mask = labels_neq

        anchors = []
        positives = []
        negatives = []

        for i in range(batch_size):
            for j in range(batch_size):
                if pos_mask[i, j]:
                    for k in range(batch_size):
                        if neg_mask[i, k]:
                            anchors.append(i)
                            positives.append(j)
                            negatives.append(k)

        if len(anchors) == 0:
            anchors = torch.zeros(0, dtype=torch.long, device=device)
            positives = torch.zeros(0, dtype=torch.long, device=device)
            negatives = torch.zeros(0, dtype=torch.long, device=device)
        else:
            anchors = torch.tensor(anchors, device=device)
            positives = torch.tensor(positives, device=device)
            negatives = torch.tensor(negatives, device=device)

        return anchors, positives, negatives


class BatchHardTripletMiner(TripletMiner):
    """Batch hard triplet mining strategy.

    Creates hardest positive and hardest negative for each anchor.
    """

    def __init__(self, distance: str = "euclidean"):
        self.distance = distance.lower()

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        labels = labels.to(device)
        embeddings = F.normalize(embeddings, dim=-1)

        if self.distance == "euclidean":
            dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        elif self.distance == "cosine":
            dist_matrix = 1 - torch.mm(embeddings, embeddings.T)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).fill_diagonal_(False)
        neg_mask = ~(labels.unsqueeze(0) == labels.unsqueeze(1))

        dist_matrix = dist_matrix.masked_fill_(~pos_mask, float("inf"))
        hardest_pos_dists, hardest_pos_indices = dist_matrix.min(dim=1)

        dist_matrix = dist_matrix.masked_fill_(~neg_mask, float("-inf"))
        dist_matrix = dist_matrix.masked_fill_(
            torch.isinf(dist_matrix) & (dist_matrix > 0), float("inf")
        )
        hardest_neg_dists, hardest_neg_indices = dist_matrix.max(dim=1)

        anchors = torch.arange(batch_size, device=device)

        return anchors, hardest_pos_indices, hardest_neg_indices


class NPairMiner(TripletMiner):
    """N-Pair mining strategy.

    Creates one positive and multiple negatives per anchor.
    """

    def __init__(self, num_negatives: int = 4):
        self.num_negatives = num_negatives

    def mine(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        labels = labels.to(device)

        anchors_list = []
        positives_list = []
        negatives_list = []

        unique_labels = labels.unique()

        for label in unique_labels:
            label_mask = labels == label
            label_indices = torch.where(label_mask)[0]

            for anchor_idx in label_indices:
                other_indices = label_indices[label_indices != anchor_idx]

                if len(other_indices) > 0:
                    pos_idx = other_indices[torch.randint(len(other_indices), (1,))]
                else:
                    continue

                other_labels_mask = labels != label
                other_indices = torch.where(other_labels_mask)[0]

                if len(other_indices) >= self.num_negatives:
                    neg_indices = other_indices[
                        torch.randperm(len(other_indices))[: self.num_negatives]
                    ]
                elif len(other_indices) > 0:
                    neg_indices = other_indices.repeat(
                        (self.num_negatives // len(other_indices) + 1)
                    )[: self.num_negatives]
                else:
                    continue

                anchors_list.append(anchor_idx.unsqueeze(0).expand(self.num_negatives))
                positives_list.append(pos_idx.unsqueeze(0).expand(self.num_negatives))
                negatives_list.append(neg_indices)

        if len(anchors_list) == 0:
            return (
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        anchors = torch.cat(anchors_list)
        positives = torch.cat(positives_list)
        negatives = torch.cat(negatives_list)

        return anchors, positives, negatives


def create_triplet_miner(
    strategy: str,
    **kwargs,
) -> TripletMiner:
    """Create a triplet miner based on strategy name.

    Args:
        strategy: Mining strategy name
        **kwargs: Additional arguments for the miner

    Returns:
        TripletMiner instance
    """
    miners = {
        "random": RandomTripletMiner,
        "hard": HardNegativeMiner,
        "hardest": lambda: HardNegativeMiner(mining_strategy="hardest"),
        "semihard": SemihardNegativeMiner,
        "distance_weighted": DistanceWeightedMiner,
        "angular": AngularTripletMiner,
        "batch_all": BatchAllTripletMiner,
        "batch_hard": BatchHardTripletMiner,
        "npair": NPairMiner,
    }

    strategy = strategy.lower()
    if strategy not in miners:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(miners.keys())}"
        )

    return miners[strategy](**kwargs)


__all__ = [
    "TripletMiner",
    "RandomTripletMiner",
    "HardNegativeMiner",
    "SemihardNegativeMiner",
    "DistanceWeightedMiner",
    "AngularTripletMiner",
    "BatchAllTripletMiner",
    "BatchHardTripletMiner",
    "NPairMiner",
    "create_triplet_miner",
]
