"""
Multi-Modal Contrastive Learning for fishstick

This module provides multi-modal contrastive learning:
- SimCLR-style contrastive learning
- CLIP-style contrastive loss
- Triplet contrastive learning
"""

from typing import Optional, List, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MultiModalSimCLR(nn.Module):
    """SimCLR-style multi-modal contrastive learning."""

    def __init__(
        self,
        embed_dim: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(
        self,
        features1: Tensor,
        features2: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        z1 = self.projection(features1)
        z2 = self.projection(features2)

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        return z1, z2


class SimCLRLoss(nn.Module):
    """NT-Xent loss for SimCLR-style contrastive learning."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        batch_size = z1.size(0)

        z = torch.cat([z1, z2], dim=0)
        similarity = z @ z.t() / self.temperature

        mask = torch.eye(batch_size * 2, device=z.device).bool()
        similarity = similarity.masked_fill(mask, float("-inf"))

        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        loss = F.cross_entropy(similarity, labels)
        return loss


class CLIPLoss(nn.Module):
    """CLIP-style contrastive loss."""

    def __init__(
        self,
        temperature: float = 0.1,
        symmetric: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
    ) -> Tensor:
        image_emb = F.normalize(image_features, dim=-1)
        text_emb = F.normalize(text_features, dim=-1)

        logits = (image_emb @ text_emb.t()) / self.temperature

        labels = torch.arange(len(logits), device=logits.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)

        if self.symmetric:
            return (loss_i + loss_t) / 2
        return loss_i


class TripletContrastiveLoss(nn.Module):
    """Triplet contrastive loss."""

    def __init__(
        self,
        margin: float = 0.3,
        distance_metric: str = "cosine",
    ):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        if self.distance_metric == "cosine":
            pos_sim = (anchor * positive).sum(dim=-1)
            neg_sim = (anchor * negative).sum(dim=-1)
            loss = F.relu(neg_sim - pos_sim + self.margin)
        else:
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)
            loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


class MoCoStyleLoss(nn.Module):
    """MoCo-style momentum contrastive loss."""

    def __init__(
        self,
        queue_size: int = 65536,
        temperature: float = 0.1,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.momentum = momentum

        self.register_buffer("queue", torch.randn(queue_size, 256))
        self.queue = F.normalize(self.queue, dim=-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: Tensor):
        batch_size = keys.size(0)

        ptr = int(self.queue_ptr)
        self.queue[ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(
        self,
        query: Tensor,
        key: Tensor,
    ) -> Tensor:
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        logits = query @ torch.cat([self.queue, key], dim=0).t() / self.temperature

        labels = torch.zeros(len(query), dtype=torch.long, device=query.device)
        loss = F.cross_entropy(logits, labels)

        self._dequeue_and_enqueue(key)

        return loss


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: Tensor,
        positives: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        query = F.normalize(query, dim=-1)
        positives = F.normalize(positives, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        all_targets = torch.cat([positives, negatives], dim=0)

        similarity = (query @ all_targets.t()) / self.temperature

        labels = torch.zeros(len(query), dtype=torch.long, device=query.device)

        return F.cross_entropy(similarity, labels)


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss for multi-modal learning."""

    def __init__(
        self,
        embed_dim: int = 256,
        lambda_offdiagonal: float = 0.005,
    ):
        super().__init__()
        self.lambda_offdiagonal = lambda_offdiagonal

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        batch_size = z1.size(0)
        dim = z1.size(-1)

        corr_matrix = (z1.T @ z2) / batch_size

        identity = torch.eye(dim, device=z1.device)
        loss = (1 - identity - self.lambda_offdiagonal * (corr_matrix**2)).sum() / dim

        return loss


class MultiModalContrastiveModel(nn.Module):
    """Complete multi-modal contrastive learning model."""

    def __init__(
        self,
        modalities: List[str],
        embed_dim: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.modalities = modalities
        self.temperature = temperature

        self.encoders = nn.ModuleDict(
            {modality: nn.Linear(embed_dim, embed_dim) for modality in modalities}
        )

        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(embed_dim, projection_dim),
                    nn.ReLU(),
                    nn.Linear(projection_dim, projection_dim),
                )
                for modality in modalities
            }
        )

    def encode(self, modality: str, features: Tensor) -> Tensor:
        encoded = self.encoders[modality](features)
        projected = self.projections[modality](encoded)
        return F.normalize(projected, dim=-1)

    def forward(
        self,
        features_dict: dict,
    ) -> dict:
        projections = {}
        for modality, features in features_dict.items():
            projections[modality] = self.encode(modality, features)
        return projections


class HardNegativeContrastiveLoss(nn.Module):
    """Contrastive loss with hard negative mining."""

    def __init__(
        self,
        temperature: float = 0.1,
        num_hard_negatives: int = 4,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        pos_sim = (anchor * positive).sum(dim=-1, keepdim=True)

        neg_sims = (anchor.unsqueeze(1) * negatives).sum(dim=-1)
        hard_neg_sims, _ = neg_sims.topk(
            min(self.num_hard_negatives, negatives.size(1)), dim=1
        )

        all_sims = torch.cat([pos_sim, hard_neg_sims], dim=1) / self.temperature

        labels = torch.zeros(len(anchor), dtype=torch.long, device=anchor.device)

        return F.cross_entropy(all_sims, labels)


class AllGatherContrastiveLoss(nn.Module):
    """Multi-GPU contrastive loss with all-gather."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        gathered = torch.cat(torch.nn.functional.all_gather(features), dim=0)

        similarity = gathered @ gathered.t() / self.temperature

        batch_size = features.size(0)
        mask = torch.eye(
            batch_size * torch.distributed.get_world_size(), device=features.device
        ).bool()
        similarity = similarity.masked_fill(mask, float("-inf"))

        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat(
            [
                labels + batch_size * i
                for i in range(torch.distributed.get_world_size())
            ],
            dim=0,
        )

        return F.cross_entropy(similarity, labels)


def create_contrastive_loss(
    loss_type: str = "simclr",
    **kwargs,
) -> nn.Module:
    """Factory function to create contrastive loss modules."""
    if loss_type == "simclr":
        return SimCLRLoss(**kwargs)
    elif loss_type == "clip":
        return CLIPLoss(**kwargs)
    elif loss_type == "triplet":
        return TripletContrastiveLoss(**kwargs)
    elif loss_type == "moco":
        return MoCoStyleLoss(**kwargs)
    elif loss_type == "info_nce":
        return InfoNCELoss(**kwargs)
    elif loss_type == "barlow_twins":
        return BarlowTwinsLoss(**kwargs)
    elif loss_type == "hard_negative":
        return HardNegativeContrastiveLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
