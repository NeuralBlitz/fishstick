import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class FitNet(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        student_flat = student_features.view(student_features.size(0), -1)
        teacher_flat = teacher_features.view(teacher_features.size(0), -1)

        if student_flat.size(1) != teacher_flat.size(1):
            student_flat = F.linear(
                student_flat,
                torch.eye(student_flat.size(1), device=student_flat.device)[
                    : teacher_flat.size(1)
                ],
            )

        feat_loss = F.mse_loss(student_flat, teacher_flat)

        if (
            student_logits is not None
            and teacher_logits is not None
            and labels is not None
        ):
            hard_loss = F.cross_entropy(student_logits, labels)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            loss = (
                self.alpha * hard_loss
                + self.beta * feat_loss
                + (1 - self.alpha - self.beta) * soft_loss
            )
        else:
            loss = feat_loss

        return loss


class AttentionTransfer(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.attention_dim = attention_dim

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        student_at = self._attention(student_features)
        teacher_at = self._attention(teacher_features)

        at_loss = F.mse_loss(student_at, teacher_at)

        if (
            student_logits is not None
            and teacher_logits is not None
            and labels is not None
        ):
            hard_loss = F.cross_entropy(student_logits, labels)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            loss = (
                self.alpha * hard_loss
                + self.beta * at_loss
                + (1 - self.alpha - self.beta) * soft_loss
            )
        else:
            loss = at_loss

        return loss

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention_dim is not None:
            x = F.adaptive_avg_pool2d(x, (self.attention_dim, self.attention_dim))

        bs, c, h, w = x.size()
        attention = (x**2).view(bs, c, -1).sum(dim=2).view(bs, c, 1, 1)
        attention = attention / (c * h * w)
        return attention


class SimilarityPreserving(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        student_sim = self._pairwise_similarity(student_features)
        teacher_sim = self._pairwise_similarity(teacher_features)

        sp_loss = F.mse_loss(student_sim, teacher_sim)

        if (
            student_logits is not None
            and teacher_logits is not None
            and labels is not None
        ):
            hard_loss = F.cross_entropy(student_logits, labels)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            loss = (
                self.alpha * hard_loss
                + self.beta * sp_loss
                + (1 - self.alpha - self.beta) * soft_loss
            )
        else:
            loss = sp_loss

        return loss

    def _pairwise_similarity(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        sim = torch.mm(x, x.t())
        return sim


class RelationKnowledgeDistillation(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        distance: str = "euclidean",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.distance = distance

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        student_rel = self._relation(student_features)
        teacher_rel = self._relation(teacher_features)

        rkd_loss = F.mse_loss(student_rel, teacher_rel)

        if (
            student_logits is not None
            and teacher_logits is not None
            and labels is not None
        ):
            hard_loss = F.cross_entropy(student_logits, labels)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            loss = (
                self.alpha * hard_loss
                + self.beta * rkd_loss
                + (1 - self.alpha - self.beta) * soft_loss
            )
        else:
            loss = rkd_loss

        return loss

    def _relation(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)

        if self.distance == "euclidean":
            x = F.normalize(x, p=2, dim=1)
            dist = torch.cdist(x, x, p=2)
            return dist
        elif self.distance == "cosine":
            sim = torch.mm(x, x.t())
            return sim
        else:
            raise ValueError(f"Unknown distance: {self.distance}")


class OFD(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        margin: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        student_flat = student_features.view(student_features.size(0), -1)
        teacher_flat = teacher_features.view(teacher_features.size(0), -1)

        student_norm = F.normalize(student_flat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)

        ofd_loss = self._ofd_loss(student_norm, teacher_norm)

        if (
            student_logits is not None
            and teacher_logits is not None
            and labels is not None
        ):
            hard_loss = F.cross_entropy(student_logits, labels)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            loss = (
                self.alpha * hard_loss
                + self.beta * ofd_loss
                + (1 - self.alpha - self.beta) * soft_loss
            )
        else:
            loss = ofd_loss

        return loss

    def _ofd_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        cos = torch.mm(student, teacher.t())
        eye_mask = torch.eye(cos.size(0), device=cos.device)
        cos = cos * (1 - eye_mask) - eye_mask * self.margin

        loss = -cos.sum() / (cos.size(0) * (cos.size(0) - 1))
        return loss


class CombinedFeatureDistillation(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.4,
        methods: Optional[List[str]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if methods is None:
            methods = ["fitnet", "at", "sp"]

        self.methods = methods

        self.fitnet = FitNet()
        self.at = AttentionTransfer()
        self.sp = SimilarityPreserving()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total_loss = 0.0
        weights = {"fitnet": self.alpha, "at": self.beta, "sp": self.gamma}

        if "fitnet" in self.methods:
            total_loss += weights.get("fitnet", 0.33) * self.fitnet(
                student_features, teacher_features
            )

        if "at" in self.methods:
            total_loss += weights.get("at", 0.33) * self.at(
                student_features, teacher_features
            )

        if "sp" in self.methods:
            total_loss += weights.get("sp", 0.33) * self.sp(
                student_features, teacher_features
            )

        if (
            student_logits is not None
            and teacher_logits is not None
            and labels is not None
        ):
            hard_loss = F.cross_entropy(student_logits, labels)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            total_loss = 0.5 * hard_loss + 0.5 * soft_loss + 0.5 * total_loss

        return total_loss
