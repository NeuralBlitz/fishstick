from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiningConfig:
    strategy: str = "hard"
    margin: float = 0.5


class HardNegativeMiner:
    def __init__(self, margin: float = 0.5):
        self.margin = margin

    def __call__(
        self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dists = F.pairwise_distance(anchor.unsqueeze(1), negatives)

        hard_mask = neg_dists > (pos_dist.unsqueeze(1) - self.margin)
        hard_negatives = negatives[hard_mask.any(dim=1)]

        return hard_negatives, hard_mask


class SemiHardNegativeMiner:
    def __init__(self, margin: float = 0.5):
        self.margin = margin

    def __call__(
        self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dists = F.pairwise_distance(anchor.unsqueeze(1), negatives)

        semi_hard_mask = (neg_dists > (pos_dist.unsqueeze(1) - self.margin)) & (
            neg_dists < (pos_dist.unsqueeze(1) + self.margin)
        )

        valid_indices = semi_hard_mask.any(dim=1)
        selected_negatives = negatives[valid_indices]

        return selected_negatives, semi_hard_mask


class CurriculumSampler:
    def __init__(
        self,
        initial_margin: float = 0.0,
        final_margin: float = 1.0,
        num_steps: int = 10000,
    ):
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.num_steps = num_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_margin(self) -> float:
        progress = min(self.current_step / self.num_steps, 1.0)
        return (
            self.initial_margin + (self.final_margin - self.initial_margin) * progress
        )

    def __call__(
        self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        margin = self.get_margin()
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dists = F.pairwise_distance(anchor.unsqueeze(1), negatives)

        curriculum_mask = neg_dists > (pos_dist.unsqueeze(1) - margin)

        valid_indices = curriculum_mask.any(dim=1)
        selected_negatives = negatives[valid_indices]

        return selected_negatives, curriculum_mask


class AllPositiveMiner:
    def __call__(
        self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return negatives, torch.ones_like(negatives, dtype=torch.bool)


class RandomNegativeMiner:
    def __init__(self, num_negatives: int = 4):
        self.num_negatives = num_negatives

    def __call__(
        self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = anchor.size(0)
        num_available = negatives.size(1)

        selected_indices = torch.randint(
            0, num_available, (batch_size, self.num_negatives), device=negatives.device
        )

        selected_negatives = negatives.gather(
            1, selected_indices.unsqueeze(2).expand(-1, -1, negatives.size(-1))
        )

        mask = torch.zeros_like(negatives, dtype=torch.bool)
        mask.scatter_(1, selected_indices, True)

        return selected_negatives, mask


class DistanceWeightedSampler:
    def __init__(self, min_distance: float = 0.5, max_distance: float = 1.5):
        self.min_distance = min_distance
        self.max_distance = max_distance

    def __call__(
        self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        neg_dists = F.pairwise_distance(anchor.unsqueeze(1), negatives)

        weights = torch.clamp(neg_dists - self.min_distance, min=0.0) * torch.clamp(
            self.max_distance - neg_dists, min=0.0
        )

        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        num_samples = min(4, negatives.size(1))
        selected_indices = torch.multinomial(
            weights.squeeze(1), num_samples, replacement=False
        )

        selected_negatives = negatives.gather(
            1, selected_indices.unsqueeze(2).expand(-1, -1, negatives.size(-1))
        )

        mask = torch.zeros_like(negatives, dtype=torch.bool)
        mask.scatter_(1, selected_indices, True)

        return selected_negatives, mask


def get_miner(strategy: str, **kwargs) -> nn.Module:
    miners = {
        "hard": HardNegativeMiner,
        "semi_hard": SemiHardNegativeMiner,
        "all": AllPositiveMiner,
        "random": RandomNegativeMiner,
        "distance_weighted": DistanceWeightedSampler,
    }

    if strategy == "curriculum":
        return CurriculumSampler(**kwargs)

    if strategy not in miners:
        raise ValueError(f"Unknown mining strategy: {strategy}")

    return miners[strategy](**kwargs)
