"""
Contrastive Learning Losses for Metric Learning

Implementation of various contrastive loss functions:
- NT-Xent (Normalized Temperature-scaled Cross Entropy)
- NPair Loss (Multi-class N-Pair)
- SupCon (Supervised Contrastive)
- ProtoNCE (Prototypical NCE)
- Circle Loss
- Triplet Margin Loss
"""

from typing import Optional, List, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Used in SimCLR and other contrastive learning methods.
    Computes loss for positive pairs and pushes apart negative pairs.

    Args:
        temperature: Temperature parameter for scaling logits
        gather_distributed: Whether to gather embeddings from all devices
        batch_size: Batch size for computing loss (for distributed)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        gather_distributed: bool = False,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.batch_size = batch_size

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute NT-Xent loss.

        Args:
            z1: First set of embeddings (batch_size, dim)
            z2: Second set of embeddings (batch_size, dim)
            labels: Optional class labels for supervised contrastive

        Returns:
            Loss value
        """
        batch_size = z1.shape[0]
        device = z1.device

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        z = torch.cat([z1, z2], dim=0)

        sim = torch.mm(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=device)
        mask = mask.fill_diagonal_(0)

        if labels is not None:
            labels = labels.repeat(2)
            mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask_pos = mask_pos.fill_diagonal_(False)
            sim = sim.masked_fill_(~mask_pos, float("-inf"))
            sim_pos = sim[mask_pos].view(2 * batch_size, -1)
            loss = -torch.logsumexp(sim_pos, dim=-1).mean()
            return loss

        sim = sim.masked_fill_(mask.bool(), float("-inf"))

        target = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=device),
                torch.arange(batch_size, device=device),
            ]
        )

        loss = F.cross_entropy(sim, target)
        return loss


class NPairLoss(nn.Module):
    """Multi-class N-Pair Loss.

    Extension of triplet loss to multiple negatives.
    Each anchor-positive pair has multiple negative samples.

    Args:
        l2_reg: L2 regularization coefficient
    """

    def __init__(self, l2_reg: float = 0.002):
        super().__init__()
        self.l2_reg = l2_reg

    def forward(
        self,
        anchors: Tensor,
        positives: Tensor,
        negatives: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute N-Pair loss.

        Args:
            anchors: Anchor embeddings (batch_size, dim)
            positives: Positive embeddings (batch_size, dim)
            negatives: Negative embeddings (batch_size, num_negatives, dim)

        Returns:
            Loss value
        """
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)

        if negatives is not None:
            negatives = F.normalize(negatives, dim=-1)
            num_negatives = negatives.shape[1]

            pos_sim = (anchors * positives).sum(dim=-1, keepdim=True)

            neg_sim = torch.bmm(negatives, anchors.unsqueeze(-1)).squeeze(-1)

            sim = torch.cat([pos_sim, neg_sim], dim=-1)

            loss = F.cross_entropy(
                sim,
                torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device),
            )
        else:
            raise ValueError("Negatives must be provided for NPairLoss")

        l2_loss = (anchors**2).sum(dim=-1).mean() + (positives**2).sum(dim=-1).mean()
        loss = loss + self.l2_reg * l2_loss

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (SupCon).

    Extension of contrastive loss to use label information.
    Pulls together embeddings of the same class while pushing apart different classes.

    Args:
        temperature: Temperature parameter for scaling
        base_temperature: Base temperature for normalization
        num_negatives: Number of negatives per positive pair
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        num_negatives: Optional[int] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_negatives = num_negatives

    def forward(
        self,
        features: Tensor,
        labels: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute supervised contrastive loss.

        Args:
            features: Features of shape (batch_size, dim)
            labels: Class labels of shape (batch_size,)
            mask: Precomputed mask of shape (batch_size, batch_size)

        Returns:
            Loss value
        """
        if labels is None and mask is None:
            raise ValueError("Either labels or mask must be provided")

        features = F.normalize(features, dim=-1)

        batch_size = features.shape[0]

        if mask is None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float()
        else:
            mask = mask.float()

        mask_pos = mask
        mask_neg = 1 - mask_pos

        sim = torch.div(torch.matmul(features, features.T), self.temperature)

        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        pos_sim = (sim * mask_pos).sum(dim=-1)

        num_positives = mask_pos.sum(dim=-1)
        num_negatives = mask_neg.sum(dim=-1)

        exp_sim = torch.exp(sim)
        exp_sim = exp_sim.masked_fill_(mask.bool(), 0)

        pos_exp = (exp_sim * mask_pos).sum(dim=-1)
        neg_exp = (exp_sim * mask_neg).sum(dim=-1)

        pos_exp = pos_exp / (num_positives + 1e-16)
        neg_exp = neg_exp / (num_negatives + 1e-16)

        log_prob = torch.log(pos_exp / (pos_exp + neg_exp + 1e-16) + 1e-16)

        loss = -log_prob

        if (num_positives == 0).any():
            loss = loss.masked_fill_(num_positives == 0, 0)

        loss = loss.mean()
        loss = loss * self.temperature / self.base_temperature

        return loss


class ProtoNCELoss(nn.Module):
    """Prototypical NCE Loss.

    Combines prototypical networks with contrastive learning.

    Args:
        temperature: Temperature parameter
        num_prototypes: Number of prototypes per class
    """

    def __init__(
        self,
        temperature: float = 0.1,
        num_prototypes: int = 5,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_prototypes = num_prototypes

    def forward(
        self,
        features: Tensor,
        prototypes: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute Prototypical NCE loss.

        Args:
            features: Query features (batch_size, dim)
            prototypes: Prototype features (num_classes, num_prototypes, dim)
            labels: Class labels (batch_size,)

        Returns:
            Loss value
        """
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)

        batch_size = features.shape[0]
        num_classes = prototypes.shape[0]

        features_exp = features.unsqueeze(1).unsqueeze(2)
        prototypes_exp = prototypes.unsqueeze(0)

        sim = (features_exp * prototypes_exp).sum(dim=-1)
        sim = sim / self.temperature

        prototypes_flat = prototypes.view(num_classes * self.num_prototypes, -1)
        labels_exp = labels.unsqueeze(1).unsqueeze(2)

        sim_all = torch.mm(features, prototypes_flat.T) / self.temperature

        labels_flat = labels.unsqueeze(1).repeat(1, self.num_prototypes).view(-1)
        pos_mask = torch.zeros(
            batch_size, num_classes * self.num_prototypes, device=features.device
        )
        pos_mask.scatter_(1, labels_flat.unsqueeze(0).expand(batch_size, -1), 1)

        pos_sim = (sim_all * pos_mask).sum(dim=-1)
        neg_sim = (sim_all * (1 - pos_mask)).sum(dim=-1) / (
            num_classes * self.num_prototypes - self.num_prototypes
        )

        loss = F.cross_entropy(sim.view(batch_size, -1), labels)
        contrastive_loss = -(pos_sim - neg_sim).mean()

        return loss + 0.1 * contrastive_loss


class CircleLoss(nn.Module):
    """Circle Loss for Metric Learning.

    A unified loss function that optimizes circular margin for both
    similarity-based and distance-based metric learning.

    Args:
        margin: Margin parameter for similarity
        gamma: Scale parameter
        num_positives: Number of positive pairs per anchor
        num_negatives: Number of negative pairs per anchor
    """

    def __init__(
        self,
        margin: float = 0.25,
        gamma: float = 256.0,
        num_positives: Optional[int] = None,
        num_negatives: Optional[int] = None,
    ):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.num_positives = num_positives
        self.num_negatives = num_negatives

    def forward(
        self,
        similarity: Tensor,
        labels: Optional[Tensor] = None,
        pos_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Circle loss.

        Args:
            similarity: Pairwise similarity matrix or similarities
            labels: Class labels for computing masks
            pos_mask: Precomputed positive mask

        Returns:
            Loss value
        """
        if labels is not None:
            labels = labels.contiguous()
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            neg_mask = 1 - pos_mask
        elif pos_mask is not None:
            neg_mask = 1 - pos_mask
        else:
            raise ValueError("Either labels or pos_mask must be provided")

        pos_sim = similarity * pos_mask
        neg_sim = similarity * neg_mask

        Np = pos_mask.sum(dim=-1)
        Nn = neg_mask.sum(dim=-1)

        neg_sim_masked = neg_sim.masked_fill_(neg_mask == 0, float("inf"))
        r_p = torch.clamp_min(
            self.margin - neg_sim_masked.max(dim=-1)[0], min=0
        ).detach()
        r_n = (pos_sim.min(dim=-1)[0] - self.margin).clamp_min_(0).detach()

        alpha_p = (r_p + similarity).masked_fill_(pos_mask == 0, 0)
        alpha_n = (r_n + similarity).masked_fill_(neg_mask == 0, 0)

        loss_pos = (alpha_p * pos_sim).sum(dim=-1) / (Np + 1e-16)
        loss_neg = (alpha_n * neg_sim).sum(dim=-1) / (Nn + 1e-16)

        loss = loss_pos + loss_neg
        loss = loss * self.gamma

        return loss.mean()


class TripletMarginLoss(nn.Module):
    """Triplet Margin Loss.

    Margin-based loss for learning embeddings where the distance between
    anchor and positive should be smaller than anchor and negative by at least margin.

    Args:
        margin: Margin between positive and negative pairs
        p: Norm degree for distance
        swap: Whether to use swap distance (min of anchor-positive and anchor-negative)
        reduction: Reduction method ('mean', 'sum', or 'none')
    """

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        swap: bool = False,
        reduction: str = "mean",
    ):
        super().__init__()
        self.margin = margin
        self.p = p
        self.swap = swap
        self.reduction = reduction

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        """Compute triplet margin loss.

        Args:
            anchor: Anchor embeddings (batch_size, dim)
            positive: Positive embeddings (batch_size, dim)
            negative: Negative embeddings (batch_size, dim)

        Returns:
            Loss value
        """
        dist_pos = F.pairwise_distance(anchor, positive, p=self.p)
        dist_neg = F.pairwise_distance(anchor, negative, p=self.p)

        if self.swap:
            dist_swap = F.pairwise_distance(positive, negative, p=self.p)
            dist_neg = torch.minimum(dist_neg, dist_swap)

        losses = F.relu(dist_pos - dist_neg + self.margin)

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses


class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity Loss.

    Combines multiple types of similarities for robust metric learning.

    Args:
        margin: Margin parameter
        beta: Weight for negative similarities
        gamma: Weight for positive similarities
    """

    def __init__(
        self,
        margin: float = 0.5,
        beta: float = 1.0,
        gamma: float = 0.5,
    ):
        super().__init__()
        self.margin = margin
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        features: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute multi-similarity loss.

        Args:
            features: Embeddings (batch_size, dim)
            labels: Class labels (batch_size,)

        Returns:
            Loss value
        """
        features = F.normalize(features, dim=-1)
        sim_matrix = torch.mm(features, features.T)

        labels = labels.contiguous()
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        neg_mask = 1 - pos_mask

        pos_sim = sim_matrix * pos_mask
        neg_sim = sim_matrix * neg_mask

        dist_pos = 1 - pos_sim
        dist_neg = sim_matrix - self.margin

        pos_loss = torch.logsumexp(self.gamma * dist_pos, dim=-1).mean()
        neg_loss = torch.logsumexp(self.beta * dist_neg, dim=-1).mean()

        return pos_loss + neg_loss


class ContrastiveLoss(nn.Module):
    """General contrastive loss with multiple modes.

    Supports unsupervised (SimCLR-style) and supervised modes.

    Args:
        temperature: Temperature parameter
        mode: 'unsupervised' or 'supervised'
        margin: Margin for supervised mode
    """

    def __init__(
        self,
        temperature: float = 0.07,
        mode: str = "unsupervised",
        margin: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.mode = mode
        self.margin = margin

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute contrastive loss.

        Args:
            z1: First view embeddings
            z2: Second view embeddings
            labels: Class labels for supervised mode

        Returns:
            Loss value
        """
        if self.mode == "unsupervised":
            return NTXentLoss(temperature=self.temperature)(z1, z2)
        elif self.mode == "supervised" and labels is not None:
            return SupConLoss(temperature=self.temperature)(
                torch.cat([z1, z2], dim=0), torch.cat([labels, labels], dim=0)
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


__all__ = [
    "NTXentLoss",
    "NPairLoss",
    "SupConLoss",
    "ProtoNCELoss",
    "CircleLoss",
    "TripletMarginLoss",
    "MultiSimilarityLoss",
    "ContrastiveLoss",
]
