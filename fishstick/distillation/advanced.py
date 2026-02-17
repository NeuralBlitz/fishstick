"""
Advanced Knowledge Distillation

Standard KD, TAKD, DML, and feature-based distillation.
"""

from typing import Optional, Dict, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """Standard knowledge distillation loss combining hard labels and soft labels.

    Args:
        temperature: Temperature for softening predictions
        alpha: Weight for hard labels (1-alpha for soft labels)
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature**2)

        if labels is not None:
            hard_loss = self.ce_loss(student_logits, labels)
            loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
            return loss, {"hard_loss": hard_loss.item(), "soft_loss": soft_loss.item()}

        return soft_loss, {"soft_loss": soft_loss.item()}


class FeatureDistillationLoss(nn.Module):
    """Feature-based knowledge distillation using intermediate features.

    Args:
        loss_type: Type of feature loss ('l2', 'cosine', 'l2_with_attention')
        temperature: Temperature for attention transfer
    """

    def __init__(self, loss_type: str = "l2", temperature: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        if self.loss_type == "l2":
            return F.mse_loss(student_features, teacher_features)

        elif self.loss_type == "cosine":
            student_norm = F.normalize(student_features, dim=-1)
            teacher_norm = F.normalize(teacher_features, dim=-1)
            return 1 - (student_norm * teacher_norm).sum(dim=-1).mean()

        elif self.loss_type == "l2_with_attention":
            student_attn = self._get_attention(student_features)
            teacher_attn = self._get_attention(teacher_features)
            return F.mse_loss(student_attn, teacher_attn)

        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _get_attention(self, features: Tensor) -> Tensor:
        return (features**2).mean(dim=-1)


class ComprehensiveDistillation(nn.Module):
    """Comprehensive knowledge distillation with multiple distillation methods.

    Args:
        temperature: Temperature for softening
        alpha: Weight for hard labels
        beta: Weight for feature distillation
        feature_loss_type: Type of feature loss
    """

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,
        beta: float = 0.1,
        feature_loss_type: str = "l2",
    ):
        super().__init__()
        self.kd_loss = KnowledgeDistillationLoss(temperature, alpha)
        self.feature_loss = FeatureDistillationLoss(feature_loss_type, temperature)
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_features: Optional[Tensor] = None,
        teacher_features: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        kd_loss, kd_metrics = self.kd_loss(student_logits, teacher_logits, labels)

        total_loss = self.alpha * kd_loss

        if student_features is not None and teacher_features is not None:
            feat_loss = self.feature_loss(student_features, teacher_features)
            total_loss = total_loss + self.beta * feat_loss
            kd_metrics["feature_loss"] = feat_loss.item()

        kd_metrics["total_loss"] = total_loss.item()
        return total_loss, kd_metrics


class TakeKD(nn.Module):
    """Teacher Assistant Knowledge Distillation (TAKD).

    Uses intermediate teacher assistants to bridge the gap between student and teacher.

    Args:
        temperature: Temperature for softening
        alpha: Weight for hard labels
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: Tensor,
        assistant_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        assistant_soft = F.softmax(assistant_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        loss = 0.0
        if labels is not None:
            ce = nn.CrossEntropyLoss()(student_logits, labels)
            loss += self.alpha * ce

        loss += F.kl_div(student_soft, assistant_soft, reduction="batchmean") * (
            self.temperature**2
        )
        loss += F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (
            self.temperature**2
        )

        return loss


class DeepMutualLearning(nn.Module):
    """Deep Mutual Learning (DML) for collaborative training of multiple networks.

    Args:
        temperature: Temperature for softening
    """

    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        logits1: Tensor,
        logits2: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        soft_log_probs1 = F.log_softmax(logits1 / self.temperature, dim=-1)
        soft_log_probs2 = F.log_softmax(logits2 / self.temperature, dim=-1)
        soft_probs1 = F.softmax(logits2 / self.temperature, dim=-1)
        soft_probs2 = F.softmax(logits1 / self.temperature, dim=-1)

        loss1 = F.kl_div(soft_log_probs1, soft_probs1, reduction="batchmean") * (
            self.temperature**2
        )
        loss2 = F.kl_div(soft_log_probs2, soft_probs2, reduction="batchmean") * (
            self.temperature**2
        )

        if labels is not None:
            ce_loss1 = F.cross_entropy(logits1, labels)
            ce_loss2 = F.cross_entropy(logits2, labels)
            loss1 = loss1 + ce_loss1
            loss2 = loss2 + ce_loss2

        return loss1, loss2


class AttentionTransfer(nn.Module):
    """Attention Transfer for knowledge distillation.

    Transfers attention maps from teacher to student.

    Args:
        stride: Stride for computing attention
    """

    def __init__(self, stride: int = 1):
        super().__init__()
        self.stride = stride

    def forward(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        student_attn = self._attention(student_features)
        teacher_attn = self._attention(teacher_features)

        if student_attn.shape != teacher_attn.shape:
            student_attn = F.adaptive_avg_pool2d(student_attn, teacher_attn.shape[-2:])

        return F.mse_loss(student_attn, teacher_attn)

    def _attention(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            return (x**2).mean(dim=1)
        return x
