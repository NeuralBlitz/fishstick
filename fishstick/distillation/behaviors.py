"""
Feature and Relation Distillation
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FeatureDistillation(nn.Module):
    """Feature-based knowledge distillation."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        return F.mse_loss(student_features, teacher_features, reduction=self.reduction)


class RelationDistillation(nn.Module):
    """Relation-based knowledge distillation."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_emb: Tensor, teacher_emb: Tensor) -> Tensor:
        student_sim = self._compute_similarity(student_emb)
        teacher_sim = self._compute_similarity(teacher_emb)

        return F.mse_loss(student_sim, teacher_sim)

    def _compute_similarity(self, emb: Tensor) -> Tensor:
        emb = F.normalize(emb, dim=1)
        sim = torch.matmul(emb, emb.T)
        return sim


class FitNetLoss(nn.Module):
    """FitNet: Distilling Feature Maps."""

    def __init__(self, hint_layer: int = 2):
        super().__init__()
        self.hint_layer = hint_layer

    def forward(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        return F.mse_loss(student_features, teacher_features)


class RKDLoss(nn.Module):
    """Relational Knowledge Distillation."""

    def __init__(self, distance: float = 1.0, angle: float = 1.0):
        super().__init__()
        self.distance = distance
        self.angle = angle

    def forward(self, student_emb: Tensor, teacher_emb: Tensor) -> Tensor:
        student_sim = self._pairwise_similarity(student_emb)
        teacher_sim = self._pairwise_similarity(teacher_emb)

        loss_dist = F.mse_loss(student_sim, teacher_sim)

        student_angle = self._angle_similarity(student_emb)
        teacher_angle = self._angle_similarity(teacher_emb)

        loss_angle = F.mse_loss(student_angle, teacher_angle)

        return self.distance * loss_dist + self.angle * loss_angle

    def _pairwise_similarity(self, emb: Tensor) -> Tensor:
        emb = F.normalize(emb, dim=1)
        return torch.matmul(emb, emb.t())

    def _angle_similarity(self, emb: Tensor) -> Tensor:
        N = emb.size(0)
        emb1 = emb.unsqueeze(0).expand(N, N, -1)
        emb2 = emb.unsqueeze(1).expand(N, N, -1)
        angle = (emb1 - emb2).pow(2).sum(-1)
        return angle


class NSTLoss(nn.Module):
    """Neural Selective Transfer."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, student_features: list, teacher_features: list) -> Tensor:
        loss = 0
        for s, t in zip(student_features, teacher_features):
            s = F.normalize(s, dim=1)
            t = F.normalize(t, dim=1)

            s_attn = self._attention(s)
            t_attn = self._attention(t)

            loss += F.mse_loss(s_attn, t_attn)

        return self.alpha * loss

    def _attention(self, x: Tensor) -> Tensor:
        return F.normalize(x.pow(2).mean(1), dim=1)
