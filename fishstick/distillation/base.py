"""
Knowledge Distillation Base Classes
"""

from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Base distillation loss."""

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self, student_logits: Tensor, teacher_logits: Tensor, labels: Tensor
    ) -> Tensor:
        raise NotImplementedError


class TemperatureScaledLoss(DistillationLoss):
    """Knowledge distillation with temperature scaling."""

    def forward(
        self, student_logits: Tensor, teacher_logits: Tensor, labels: Tensor
    ) -> Tensor:
        T = self.temperature

        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        distill_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (
            T * T
        )

        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * distill_loss + (1 - self.alpha) * hard_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss."""

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        confidence = 1.0 - self.smoothing
        smooth_label = torch.full_like(logits, self.smoothing / (self.num_classes - 1))
        smooth_label.scatter_(1, labels.unsqueeze(1), confidence)

        log_probs = F.log_softmax(logits, dim=-1)
        return -(smooth_label * log_probs).sum(dim=-1).mean()


class MixupDistillationLoss(DistillationLoss):
    """Distillation with mixup augmentation."""

    def forward(
        self, student_logits: Tensor, teacher_logits: Tensor, labels: Tensor
    ) -> Tensor:
        return super().forward(student_logits, teacher_logits, labels)


class AttentionTransferLoss(nn.Module):
    """Attention transfer for knowledge distillation."""

    def __init__(self, beta: float = 1000):
        super().__init__()
        self.beta = beta

    def forward(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        student_attn = self._compute_attention(student_features)
        teacher_attn = self._compute_attention(teacher_features)

        return self.beta * F.mse_loss(student_attn, teacher_attn)

    def _compute_attention(self, features: Tensor) -> Tensor:
        return F.normalize(features.pow(2).mean(1), dim=1)


class PKDLoss(nn.Module):
    """Progressive Knowledge Distillation Loss."""

    def __init__(self, gamma: float = 0.5):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_features: Tensor,
        teacher_features: Tensor,
        labels: Tensor,
    ) -> Tensor:
        ce_loss = F.cross_entropy(student_logits, labels)

        T = 4.0
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        feat_loss = F.mse_loss(student_features, teacher_features)

        return ce_loss + self.gamma * distill_loss + (1 - self.gamma) * feat_loss


class ContrastiveDistillationLoss(nn.Module):
    """Contrastive knowledge distillation."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_emb: Tensor, teacher_emb: Tensor) -> Tensor:
        student_emb = F.normalize(student_emb, dim=1)
        teacher_emb = F.normalize(teacher_emb, dim=1)

        sim_matrix = torch.matmul(student_emb, teacher_emb.T) / self.temperature

        labels = torch.arange(len(sim_matrix), device=sim_matrix.device)

        return F.cross_entropy(sim_matrix, labels)
