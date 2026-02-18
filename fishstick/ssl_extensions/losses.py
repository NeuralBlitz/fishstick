"""
Advanced SSL Loss Functions

Extended loss functions for self-supervised learning:
- Contrastive losses
- Regularization losses
- Advanced SSL losses
"""

from typing import Optional, Tuple, Dict, Any, List
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from fishstick.ssl_extensions.base import stop_gradient, gather_from_all


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Args:
        temperature: Temperature for scaling similarities
        gather_distributed: Whether to gather from distributed processes
    """

    def __init__(
        self,
        temperature: float = 0.07,
        gather_distributed: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        if self.gather_distributed and torch.distributed.is_initialized():
            z1 = gather_from_all(z1)
            z2 = gather_from_all(z2)

        z = torch.cat([z1, z2], dim=0)

        sim = z @ z.T / self.temperature

        batch_size = z1.shape[0]

        mask = torch.eye(2 * batch_size, device=sim.device)
        mask = mask.fill_diagonal_(0).bool()

        sim = sim.masked_fill(mask, float("-inf"))

        labels = torch.arange(batch_size, device=sim.device)
        labels = torch.cat([labels, labels], dim=0)

        loss = F.cross_entropy(sim, labels)

        return loss


class SimSiamContrastiveLoss(nn.Module):
    """SimSiam contrastive loss.

    Args:
        use_cosine: Whether to use cosine similarity
    """

    def __init__(
        self,
        use_cosine: bool = True,
    ):
        super().__init__()
        self.use_cosine = use_cosine

    def forward(
        self,
        pred1: Tensor,
        pred2: Tensor,
        target1: Tensor,
        target2: Tensor,
    ) -> Tensor:
        if self.use_cosine:
            pred1 = F.normalize(pred1, dim=-1)
            pred2 = F.normalize(pred2, dim=-1)
            target1 = F.normalize(target1, dim=-1)
            target2 = F.normalize(target2, dim=-1)

        loss1 = -F.cosine_similarity(pred1, stop_gradient(target1), dim=-1).mean()
        loss2 = -F.cosine_similarity(pred2, stop_gradient(target2), dim=-1).mean()

        return (loss1 + loss2) / 2


class VICRegLoss(nn.Module):
    """VICReg: Variance-Invariance-Covariance Regularization loss.

    Args:
        sim_coef: Similarity loss coefficient
        var_coef: Variance loss coefficient
        cov_coef: Covariance loss coefficient
        epsilon: Epsilon for numerical stability
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

        std_z = torch.sqrt(z.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_z))

        z = z - z.mean(dim=0)
        cov_z = (z.T @ z) / z.shape[0]
        cov_loss = (
            cov_z.pow(2).sum() / z.shape[-1]
            - cov_z.diagonal().pow(2).sum() / z.shape[-1]
        )

        loss = (
            self.sim_coef * sim_loss
            + self.var_coef * std_loss
            + self.cov_coef * cov_loss
        )

        return loss


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins redundancy reduction loss.

    Args:
        embedding_dim: Embedding dimension
        lambd: Trade-off parameter
    """

    def __init__(
        self,
        embedding_dim: int,
        lambd: float = 0.005,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lambd = lambd

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        batch_size = z1.shape[0]

        c = torch.mm(z1.T, z2) / batch_size

        diag = torch.eye(self.embedding_dim, device=c.device)
        off_diag = 1 - diag

        loss_diag = (1 - c.diagonal()).pow(2).sum()
        loss_off_diag = (c * off_diag).pow(2).sum() / self.embedding_dim

        loss = loss_diag + self.lambd * loss_off_diag

        return loss


class DINOLoss(nn.Module):
    """DINO loss with teacher centering and sharpening.

    Args:
        teacher_temp: Temperature for teacher predictions
        student_temp: Temperature for student predictions
        center_momentum: Momentum for centering
    """

    def __init__(
        self,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1))

    def forward(
        self,
        student_output: Tensor,
        teacher_output: Tensor,
    ) -> Tensor:
        student_out = student_output / self.student_temp
        student_out = student_out.softmax(dim=-1)

        teacher_out = (teacher_output - self.center) / self.teacher_temp
        teacher_out = teacher_out.softmax(dim=-1)

        loss = -(teacher_out * torch.log(student_out)).sum(-1).mean()

        self._update_center(teacher_output)

        return loss

    def _update_center(self, teacher_output: Tensor):
        batch_center = teacher_output.mean(dim=0)

        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class WMSELoss(nn.Module):
    """Whitened MSE Loss for BYOL/SimSiam.

    Args:
        use_covariance: Whether to use covariance whitening
    """

    def __init__(
        self,
        use_covariance: bool = True,
    ):
        super().__init__()
        self.use_covariance = use_covariance

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        if self.use_covariance:
            z1 = self._whiten(z1)
            z2 = self._whiten(z2)

        return 2 - 2 * (z1 * z2).sum(dim=-1).mean()

    def _whiten(self, z: Tensor) -> Tensor:
        z = z - z.mean(dim=0, keepdim=True)

        std = z.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-9)

        z = z / std

        return z


class DebiasedContrastiveLoss(nn.Module):
    """Debiased contrastive loss.

    Args:
        temperature: Temperature for scaling
        num_positives: Number of positive pairs
    """

    def __init__(
        self,
        temperature: float = 0.1,
        num_positives: int = 2,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_positives = num_positives

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        batch_size = z1.shape[0]

        positives = (z1 * z2).sum(dim=-1)

        negatives = z1 @ z1.T + z2 @ z2.T

        neg_sum = negatives.sum(dim=-1) - torch.diag(negatives)

        n = negatives.shape[0]
        neg_count = n - self.num_positives

        log_prob = positives / self.temperature
        log_prob -= torch.log(torch.exp(neg_sum / self.temperature) / neg_count + 1e-9)

        loss = -log_prob.mean()

        return loss


class HardNegativeContrastiveLoss(nn.Module):
    """Hard negative contrastive loss.

    Args:
        temperature: Temperature for scaling
        margin: Margin for hard negatives
    """

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        pos_sim = (z1 * z2).sum(dim=-1)

        z_all = torch.cat([z1, z2], dim=0)

        neg_sim = z_all @ z_all.T / self.temperature

        batch_size = z1.shape[0]

        pos_mask = torch.zeros_like(neg_sim)
        pos_mask[batch_size:, :batch_size] = torch.eye(
            batch_size, device=neg_sim.device
        )
        pos_mask[:batch_size, batch_size:] = torch.eye(
            batch_size, device=neg_sim.device
        )

        neg_sim_hard = neg_sim.clone()
        neg_sim_hard[pos_mask.bool()] = float("-inf")

        hardest_neg, _ = neg_sim_hard.max(dim=-1)

        loss = F.relu(hardest_neg + self.margin - pos_sim)

        return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Args:
        temperature: Temperature for scaling
    """

    def __init__(
        self,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: Tensor,
        labels: Tensor,
    ) -> Tensor:
        features = F.normalize(features, dim=-1)

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        mask = mask - torch.diag(torch.ones(batch_size, device=features.device))

        anchor_dot_contrast = features @ features.T / self.temperature

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - mask)

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        mask_positive = mask.sum(dim=1)
        mask_positive = torch.where(
            mask_positive == 0, torch.ones_like(mask_positive), mask_positive
        )

        mean_log_prob_pos = -(mask_positive * log_prob).sum(dim=1) / mask_positive

        loss = mean_log_prob_pos.mean()

        return loss


class TripletLoss(nn.Module):
    """Triplet loss for SSL.

    Args:
        margin: Margin for triplet loss
    """

    def __init__(
        self,
        margin: float = 0.3,
    ):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)

        loss = F.relu(dist_pos - dist_neg + self.margin)

        return loss.mean()


class CenterLoss(nn.Module):
    """Center loss for SSL.

    Args:
        num_features: Number of features
        num_classes: Number of classes
        size_average: Whether to average the loss
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        size_average: bool = True,
    ):
        super().__init__()

        self.centers = nn.Parameter(torch.randn(num_classes, num_features))
        self.size_average = size_average

    def forward(
        self,
        features: Tensor,
        labels: Tensor,
    ) -> Tensor:
        batch_size = features.size(0)

        features = features.view(batch_size, -1)

        centers_batch = self.centers.index_select(0, labels.long())

        loss = (features - centers_batch).pow(2).sum() / 2.0

        if self.size_average:
            loss /= batch_size

        return loss


class ClusterLoss(nn.Module):
    """Clustering loss for SSL.

    Args:
        num_clusters: Number of clusters
        features_dim: Feature dimension
    """

    def __init__(
        self,
        num_clusters: int,
        features_dim: int,
    ):
        super().__init__()
        self.num_clusters = num_clusters

        self.centers = nn.Parameter(torch.randn(num_clusters, features_dim))

    def forward(
        self,
        features: Tensor,
        assignments: Optional[Tensor] = None,
    ) -> Tensor:
        features = F.normalize(features, dim=-1)

        if assignments is None:
            similarities = features @ F.normalize(self.centers, dim=-1).T
            assignments = similarities.argmax(dim=-1)

        centers_batch = self.centers[assignments]

        loss = (features - centers_batch).pow(2).sum(dim=-1).mean()

        return loss


class RegularizationLoss(nn.Module):
    """Regularization loss for SSL.

    Args:
        reg_type: Type of regularization ('l2', 'orthogonality', 'unit_variance')
        weight: Weight for regularization
    """

    def __init__(
        self,
        reg_type: str = "orthogonality",
        weight: float = 0.01,
    ):
        super().__init__()
        self.reg_type = reg_type
        self.weight = weight

    def forward(self, features: Tensor) -> Tensor:
        if self.reg_type == "l2":
            loss = features.pow(2).sum(dim=-1).mean()

        elif self.reg_type == "orthogonality":
            gram = features.T @ features
            identity = torch.eye(features.shape[-1], device=features.device)
            loss = (gram - identity).pow(2).sum() / features.shape[-1]

        elif self.reg_type == "unit_variance":
            mean = features.mean(dim=0)
            std = features.std(dim=0)
            loss = ((std - 1).pow(2)).mean()

        else:
            loss = torch.tensor(0.0, device=features.device)

        return self.weight * loss
