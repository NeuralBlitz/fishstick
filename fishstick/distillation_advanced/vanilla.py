import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any


class VanillaKnowledgeDistillation(nn.Module):
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = "batchmean",
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        kd_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction=self.reduction,
        ) * (T * T)

        if labels is not None and self.alpha > 0:
            student_hard = F.cross_entropy(student_logits, labels)
            loss = self.alpha * kd_loss + (1 - self.alpha) * student_hard
        else:
            loss = kd_loss

        return loss


class LabelSmoothingDistillation(nn.Module):
    def __init__(
        self,
        smoothing: float = 0.1,
        temperature: float = 4.0,
        alpha: float = 0.5,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        T = self.temperature

        smooth_labels = torch.full_like(
            student_logits, self.smoothing / (self.num_classes - 1)
        )
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - self.smoothing)

        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        hard_loss = F.kl_div(student_soft, smooth_labels, reduction="batchmean")

        loss = self.alpha * kd_loss + (1 - self.alpha) * hard_loss

        return loss


class MultiTeacherDistillation(nn.Module):
    def __init(
        self,
        teachers: List[nn.Module],
        temperature: float = 4.0,
        alpha: float = 0.5,
        teacher_weights: Optional[List[float]] = None,
        reduction: str = "batchmean",
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

        if teacher_weights is None:
            self.teacher_weights = [1.0 / len(teachers)] * len(teachers)
        else:
            total = sum(teacher_weights)
            self.teacher_weights = [w / total for w in teacher_weights]

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)

        kd_loss = 0.0
        for teacher_logits, weight in zip(teacher_logits_list, self.teacher_weights):
            teacher_soft = F.softmax(teacher_logits / T, dim=-1)
            kd_loss += (
                weight
                * F.kl_div(
                    student_soft,
                    teacher_soft,
                    reduction=self.reduction,
                )
                * (T * T)
            )

        if labels is not None and self.alpha > 0:
            student_hard = F.cross_entropy(student_logits, labels)
            loss = self.alpha * kd_loss + (1 - self.alpha) * student_hard
        else:
            loss = kd_loss

        return loss


class ProgressiveDistillation(nn.Module):
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.current_epoch < self.warmup_epochs:
            alpha = 0.0
        else:
            alpha = self.alpha

        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        if labels is not None and alpha > 0:
            student_hard = F.cross_entropy(student_logits, labels)
            loss = alpha * kd_loss + (1 - alpha) * student_hard
        else:
            loss = kd_loss

        return loss

    def step(self):
        self.current_epoch += 1


class DynamicTemperatureKD(nn.Module):
    def __init__(
        self,
        min_temp: float = 2.0,
        max_temp: float = 8.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        epoch: float = 0.0,
        total_epochs: float = 100.0,
    ) -> torch.Tensor:
        progress = epoch / total_epochs
        T = self.min_temp + (self.max_temp - self.min_temp) * progress

        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        if labels is not None and self.alpha > 0:
            student_hard = F.cross_entropy(student_logits, labels)
            loss = self.alpha * kd_loss + (1 - self.alpha) * student_hard
        else:
            loss = kd_loss

        return loss
