"""
Label Smoothing Implementations

Various label smoothing techniques for improving generalization
and calibration in classification tasks.
"""

from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.

    Applies label smoothing to softens the hard labels, reducing
    overconfidence and improving generalization.

    Args:
        smoothing: Smoothing factor in [0, 1). Higher values mean more smoothing.
        reduction: Specifies the reduction to apply: 'none', 'mean', 'sum'.
        weight: Optional per-class weights.

    Example:
        >>> loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        >>> logits = torch.randn(8, 10)
        >>> targets = torch.randint(0, 10, (8,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        loss = -smooth_targets * log_probs

        if self.weight is not None:
            weights = self.weight[targets]
            loss = loss * weights.unsqueeze(1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss.sum(dim=-1)


class ConfidenceAwareSmoothing(nn.Module):
    """
    Confidence-Aware Label Smoothing.

    Applies different smoothing levels based on the model's
    confidence in its predictions.

    Args:
        base_smoothing: Base smoothing factor.
        high_conf_threshold: Threshold above which high confidence is applied.
        low_conf_smoothing: Smoothing for low confidence predictions.
        high_conf_smoothing: Smoothing for high confidence predictions.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        base_smoothing: float = 0.1,
        high_conf_threshold: float = 0.9,
        low_conf_smoothing: float = 0.05,
        high_conf_smoothing: float = 0.15,
        reduction: str = "mean",
    ):
        super().__init__()
        self.base_smoothing = base_smoothing
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_smoothing = low_conf_smoothing
        self.high_conf_smoothing = high_conf_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()

        confidence_mask = target_probs > self.high_conf_threshold
        smoothing_values = torch.where(
            confidence_mask,
            torch.full_like(target_probs, self.high_conf_smoothing),
            torch.full_like(target_probs, self.low_conf_smoothing),
        )

        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            for i, (target, smoothing) in enumerate(zip(targets, smoothing_values)):
                smooth_targets[i].fill_(smoothing / (n_classes - 1))
                smooth_targets[i, target] = 1 - smoothing

        loss = -smooth_targets * log_probs

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss.sum(dim=-1)


class ClassWeightedSmoothing(nn.Module):
    """
    Class-Weighted Label Smoothing.

    Applies different smoothing levels for different classes based
    on their difficulty or frequency.

    Args:
        smoothing_per_class: Tensor of smoothing values per class.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        smoothing_per_class: torch.Tensor,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("smoothing_per_class", smoothing_per_class)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smoothing = self.smoothing_per_class[targets]
            for i in range(logits.size(0)):
                smooth_targets[i].fill_(smoothing[i] / (n_classes - 1))
                smooth_targets[i, targets[i]] = 1 - smoothing[i]

        loss = -smooth_targets * log_probs

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss.sum(dim=-1)


class AdaptiveSmoothing(nn.Module):
    """
    Adaptive Label Smoothing based on training dynamics.

    Automatically adjusts smoothing based on training epoch
    and per-sample difficulty.

    Args:
        initial_smoothing: Starting smoothing value.
        final_smoothing: Final smoothing value after annealing.
        warmup_epochs: Number of epochs to warmup smoothing.
        total_epochs: Total training epochs for scheduling.
        difficulty_scaling: Scale smoothing based on sample difficulty.
    """

    def __init__(
        self,
        initial_smoothing: float = 0.1,
        final_smoothing: float = 0.0,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        difficulty_scaling: bool = True,
    ):
        super().__init__()
        self.initial_smoothing = initial_smoothing
        self.final_smoothing = final_smoothing
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.difficulty_scaling = difficulty_scaling
        self.current_epoch = 0

    def step(self):
        """Increment epoch counter."""
        self.current_epoch += 1

    def _get_smoothing(self) -> float:
        if self.current_epoch < self.warmup_epochs:
            return self.initial_smoothing * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return (
                self.initial_smoothing
                + (self.final_smoothing - self.initial_smoothing) * progress
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        smoothing = self._get_smoothing()

        log_probs = F.log_softmax(logits, dim=-1)

        if self.difficulty_scaling:
            probs = F.softmax(logits, dim=-1)
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
            difficulty = 1 - target_probs
            smoothing = smoothing * (1 + difficulty)
            smoothing = smoothing.clamp(0, 0.5)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - smoothing)

        loss = -smooth_targets * log_probs
        return loss.mean()


class KnowledgeDistillationSmoothing(nn.Module):
    """
    Knowledge Distillation with Label Smoothing.

    Combines label smoothing with knowledge distillation from a teacher.

    Args:
        smoothing: Base smoothing factor.
        teacher_temperature: Temperature for softening teacher predictions.
        alpha: Weight for balancing hard labels and distillation.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        teacher_temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.teacher_temperature = teacher_temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        n_classes = student_logits.size(-1)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.teacher_temperature, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(student_log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        hard_loss = -smooth_targets * student_log_probs
        soft_loss = F.kl_div(
            student_log_probs, teacher_probs, reduction="batchmean"
        ) * (self.teacher_temperature**2)

        return self.alpha * hard_loss.mean() + (1 - self.alpha) * soft_loss


class MixupSmoothing(nn.Module):
    """
    Label Smoothing for Mixup/CutMix augmented training.

    Applies smoothing to mixed labels from data augmentation.

    Args:
        smoothing: Base smoothing factor for mixed labels.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        mixed_targets: torch.Tensor,
    ) -> torch.Tensor:
        n_classes = logits.size(-1)

        if mixed_targets.dim() == 1:
            return F.cross_entropy(logits, mixed_targets, reduction=self.reduction)

        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_targets = mixed_targets * (1 - self.smoothing)
            smooth_targets = smooth_targets + self.smoothing / n_classes

        loss = -smooth_targets * log_probs

        if self.reduction == "mean":
            return loss.sum(dim=-1).mean()
        elif self.reduction == "sum":
            return loss.sum(dim=-1).sum()
        return loss.sum(dim=-1)


__all__ = [
    "LabelSmoothingCrossEntropy",
    "ConfidenceAwareSmoothing",
    "ClassWeightedSmoothing",
    "AdaptiveSmoothing",
    "KnowledgeDistillationSmoothing",
    "MixupSmoothing",
]
