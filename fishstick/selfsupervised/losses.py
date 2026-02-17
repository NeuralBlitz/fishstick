"""
Self-Supervised Learning Loss Functions

Various loss functions for self-supervised learning:
- NT-Xent (Normalized Temperature-scaled Cross Entropy)
- SimSiam Loss
- BYOL Loss
- MoCo Loss
- InfoNCE
- VICReg Loss
"""

from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class NT_XentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Used in SimCLR and other contrastive learning methods.

    Args:
        temperature: Temperature parameter for scaling logits
        gather_distributed: Whether to gather embeddings from all devices
    """

    def __init__(
        self,
        temperature: float = 0.07,
        gather_distributed: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        batch_size = z1.shape[0]
        device = z1.device

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        z = torch.cat([z1, z2], dim=0)

        sim = torch.mm(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=device)
        mask = mask.fill_diagonal_(0)

        sim = sim.masked_fill_(mask.bool(), float("-inf"))

        target = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=device),
                torch.arange(batch_size, device=device),
            ]
        )

        loss = F.cross_entropy(sim, target)
        return loss


class SimSiamLoss(nn.Module):
    """SimSiam loss with stop-gradient.

    Uses a predictor network and stop-gradient on target representations.

    Args:
        reduction: Reduction method ('mean' or 'none')
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, p1: Tensor, p2: Tensor, z1: Tensor, z2: Tensor) -> Tensor:
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)

        loss = -(p1 * z2.detach()).sum(dim=-1) - (p2 * z1.detach()).sum(dim=-1)
        loss = loss / 2

        if self.reduction == "mean":
            return loss.mean()
        return loss


class BYOLLoss(nn.Module):
    """BYOL loss between predictions and momentum-encoded targets.

    Args:
        reduction: Reduction method ('mean' or 'none')
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, p1: Tensor, p2: Tensor, z1: Tensor, z2: Tensor) -> Tensor:
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        loss = 2 - 2 * (p1 * z2).sum(dim=-1) - 2 * (p2 * z1).sum(dim=-1)
        loss = loss / 2

        if self.reduction == "mean":
            return loss.mean()
        return loss


class MoCoLoss(nn.Module):
    """MoCo (Momentum Contrast) loss.

    Args:
        temperature: Temperature for softmax
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        queue: Optional[Tensor] = None,
    ) -> Tensor:
        if queue is not None:
            k = torch.cat([k, queue], dim=0)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        logits = torch.mm(q, k.T) / self.temperature

        batch_size = q.shape[0]
        labels = torch.arange(batch_size, device=q.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class VicRegLoss(nn.Module):
    """VICReg: Variance-Invariance-Covariance Regularization Loss.

    Args:
        sim_coef: Similarity loss coefficient
        var_coef: Variance loss coefficient
        cov_coef: Covariance loss coefficient
        epsilon: Small constant for numerical stability
    """

    def __init__(
        self,
        sim_coef: float = 25.0,
        var_coef: float = 25.0,
        cov_coef: float = 1.0,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.sim_coef = sim_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef
        self.epsilon = epsilon

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        sim_loss = F.mse_loss(z1, z2)

        z = torch.cat([z1, z2], dim=0)
        batch_size = z.shape[0]
        dim = z.shape[-1]

        std_z = torch.sqrt(z.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_z))

        z_centered = z - z.mean(dim=0)
        cov_z = (z_centered.T @ z_centered) / batch_size
        cov_loss = cov_z.pow(2).sum() / dim - cov_z.diagonal().pow(2).sum() / dim

        loss = (
            self.sim_coef * sim_loss
            + self.var_coef * std_loss
            + self.cov_coef * cov_loss
        )

        return loss


class InfoNCE(nn.Module):
    """InfoNCE (Information Noise-Contrastive Estimation) loss.

    Generic contrastive loss used in many SSL methods.

    Args:
        temperature: Temperature parameter
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: Tensor,
        positive: Tensor,
        negatives: Optional[Tensor] = None,
    ) -> Tensor:
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)

        if negatives is not None:
            negatives = F.normalize(negatives, dim=-1)
            positives = torch.cat([positive, negatives], dim=0)
        else:
            positives = positive

        logits = torch.mm(query, positives.T) / self.temperature

        batch_size = query.shape[0]
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)

        return F.cross_entropy(logits, labels)


class DCLLoss(nn.Module):
    """Decoupled Contrastive Learning (DCL) Loss.

    Addresses representation collapse and improves training stability.

    Args:
        temperature: Temperature parameter
        weight_is: Weight for positive term
        weight_neg: Weight for negative term
    """

    def __init__(
        self,
        temperature: float = 0.1,
        weight_is: float = 0.5,
        weight_neg: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight_is = weight_is
        self.weight_neg = weight_neg

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        batch_size = z1.shape[0]

        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temperature

        sim_is = torch.cat(
            [sim[:batch_size, :batch_size], sim[batch_size:, batch_size:]], dim=1
        )
        sim_neg = torch.cat(
            [sim[:batch_size, batch_size:], sim[batch_size:, :batch_size]], dim=1
        )

        loss = -self.weight_is * F.log_softmax(sim_is, dim=-1).mean()
        loss += self.weight_neg * F.softmax(sim_neg.detach(), dim=-1).sum(dim=-1).mean()

        return loss
