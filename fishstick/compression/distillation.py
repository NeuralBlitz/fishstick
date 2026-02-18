"""
Advanced Knowledge Distillation Methods

Includes logit matching, feature matching, attention transfer, and progressive distillation.
"""

from typing import Optional, List, Dict, Callable, Tuple, Union
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from collections import OrderedDict
import copy


class LogitDistillation(nn.Module):
    """Logit-based knowledge distillation using KL divergence.

    Standard KD that transfers knowledge through softened logits.

    Reference: Hinton et al., "Distilling the Knowledge in a Neural Network", 2015

    Args:
        temperature: Temperature for softening probability distributions
        alpha: Weight for distillation loss vs hard target loss
        reduction: Reduction method for loss
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        reduction: str = "batchmean",
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute distillation loss.

        Args:
            student_logits: Raw logits from student model
            teacher_logits: Raw logits from teacher model
            labels: Ground truth labels (optional)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)

        distill_loss = F.kl_div(soft_student, soft_targets, reduction=self.reduction)
        distill_loss = distill_loss * (self.temperature**2)

        metrics = {"distill_loss": distill_loss.item()}

        if labels is not None:
            hard_loss = F.cross_entropy(student_logits, labels)
            total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
            metrics["hard_loss"] = hard_loss.item()
            metrics["total_loss"] = total_loss.item()
            return total_loss, metrics

        return distill_loss, metrics


class FeatureDistillation(nn.Module):
    """Feature-based knowledge distillation using intermediate representations.

    Matches intermediate features between teacher and student networks.

    Args:
        loss_type: Type of feature matching loss ('mse', 'l1', 'cosine', 'smooth_l1')
        normalize: Whether to normalize features before matching
        temperature: Temperature for feature matching
    """

    def __init__(
        self,
        loss_type: str = "mse",
        normalize: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.temperature = temperature

    def forward(
        self,
        student_features: Tensor,
        teacher_features: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute feature distillation loss.

        Args:
            student_features: Features from student model
            teacher_features: Features from teacher model

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if self.normalize:
            student_features = F.normalize(student_features, dim=-1)
            teacher_features = F.normalize(teacher_features, dim=-1)

        if self.loss_type == "mse":
            loss = F.mse_loss(student_features, teacher_features)
        elif self.loss_type == "l1":
            loss = F.l1_loss(student_features, teacher_features)
        elif self.loss_type == "cosine":
            loss = (
                1
                - F.cosine_similarity(student_features, teacher_features, dim=-1).mean()
            )
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(student_features, teacher_features)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        loss = loss / self.temperature

        return loss, {"feature_loss": loss.item()}


class AttentionTransfer(nn.Module):
    """Attention transfer for knowledge distillation.

    Transfers attention maps from teacher to student.

    Reference: Zagoruyko & Komodakis, "Paying More Attention to Attention", ICLR 2017

    Args:
        attention_type: Type of attention ('spatial', 'channel', 'combined')
        normalize: Whether to normalize attention maps
    """

    def __init__(
        self,
        attention_type: str = "spatial",
        normalize: bool = True,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.normalize = normalize

    def forward(
        self,
        student_features: Tensor,
        teacher_features: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute attention transfer loss.

        Args:
            student_features: Feature maps from student (B, C, H, W)
            teacher_features: Feature maps from teacher (B, C, H', W')

        Returns:
            Tuple of (loss, metrics_dict)
        """
        student_attn = self._compute_attention(student_features)
        teacher_attn = self._compute_attention(teacher_features)

        if student_attn.shape != teacher_attn.shape:
            student_attn = F.interpolate(
                student_attn.unsqueeze(1),
                size=teacher_attn.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        loss = F.mse_loss(student_attn, teacher_attn)

        return loss, {"attention_loss": loss.item()}

    def _compute_attention(self, features: Tensor) -> Tensor:
        """Compute attention map from features."""
        if self.attention_type == "spatial":
            return (features**2).mean(dim=1)
        elif self.attention_type == "channel":
            return (features**2).mean(dim=(2, 3))
        elif self.attention_type == "combined":
            spatial = (features**2).mean(dim=1)
            channel = (features**2).mean(dim=(2, 3))
            return spatial + channel.unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")


class RelationDistillation(nn.Module):
    """Relation-based knowledge distillation.

    Transfers relational knowledge between samples.

    Reference: Park et al., "Relational Knowledge Distillation", CVPR 2019

    Args:
        distance_metric: Distance metric for relation ('euclidean', 'cosine')
        temperature: Temperature for relation matching
    """

    def __init__(
        self,
        distance_metric: str = "euclidean",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.distance_metric = distance_metric
        self.temperature = temperature

    def forward(
        self,
        student_features: Tensor,
        teacher_features: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute relation distillation loss.

        Args:
            student_features: Features from student (B, D)
            teacher_features: Features from teacher (B, D')

        Returns:
            Tuple of (loss, metrics_dict)
        """
        student_relation = self._compute_relation(student_features)
        teacher_relation = self._compute_relation(teacher_features)

        loss = F.mse_loss(student_relation, teacher_relation)

        return loss, {"relation_loss": loss.item()}

    def _compute_relation(self, features: Tensor) -> Tensor:
        """Compute pairwise relation matrix."""
        if self.distance_metric == "euclidean":
            diff = features.unsqueeze(1) - features.unsqueeze(0)
            relation = (diff**2).sum(dim=-1)
        elif self.distance_metric == "cosine":
            features_norm = F.normalize(features, dim=-1)
            relation = features_norm @ features_norm.t()
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        relation = relation / self.temperature
        return relation


class ProgressiveDistillation(nn.Module):
    """Progressive knowledge distillation with curriculum learning.

    Gradually transfers knowledge from easy to hard examples.

    Args:
        temperature: Initial temperature for distillation
        final_temperature: Final temperature
        alpha: Initial weight for distillation loss
        final_alpha: Final weight for distillation loss
        total_steps: Total number of training steps
    """

    def __init__(
        self,
        temperature: float = 4.0,
        final_temperature: float = 1.0,
        alpha: float = 0.9,
        final_alpha: float = 0.5,
        total_steps: int = 10000,
    ):
        super().__init__()
        self.temperature = temperature
        self.final_temperature = final_temperature
        self.alpha = alpha
        self.final_alpha = final_alpha
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        """Increment current step."""
        self.current_step += 1

    def get_current_params(self) -> Dict[str, float]:
        """Get current temperature and alpha based on progress."""
        progress = min(self.current_step / self.total_steps, 1.0)

        current_temp = (
            self.temperature - (self.temperature - self.final_temperature) * progress
        )
        current_alpha = self.alpha - (self.alpha - self.final_alpha) * progress

        return {"temperature": current_temp, "alpha": current_alpha}

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute progressive distillation loss."""
        params = self.get_current_params()
        temp = params["temperature"]
        alpha = params["alpha"]

        soft_targets = F.softmax(teacher_logits / temp, dim=-1)
        soft_student = F.log_softmax(student_logits / temp, dim=-1)

        distill_loss = F.kl_div(soft_student, soft_targets, reduction="batchmean") * (
            temp**2
        )

        metrics = {
            "distill_loss": distill_loss.item(),
            "temperature": temp,
            "alpha": alpha,
        }

        if labels is not None:
            hard_loss = F.cross_entropy(student_logits, labels)
            total_loss = alpha * distill_loss + (1 - alpha) * hard_loss
            metrics["hard_loss"] = hard_loss.item()
            metrics["total_loss"] = total_loss.item()
            return total_loss, metrics

        return distill_loss, metrics


class MultiStageDistillation(nn.Module):
    """Multi-stage distillation with intermediate teacher assistants.

    Uses multiple teacher assistants to bridge the gap between teacher and student.

    Reference: Mirzadeh et al., "Improved Knowledge Distillation via Teacher Assistant", AAAI 2020

    Args:
        temperature: Temperature for distillation
        alpha: Weight for distillation loss
        num_stages: Number of intermediate stages
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        num_stages: int = 2,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.num_stages = num_stages
        self.current_stage = 0

        self.logit_distill = LogitDistillation(temperature, alpha)

    def set_stage(self, stage: int):
        """Set current distillation stage."""
        self.current_stage = min(stage, self.num_stages)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        assistant_logits: Optional[List[Tensor]] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute multi-stage distillation loss."""
        if assistant_logits is None or len(assistant_logits) == 0:
            return self.logit_distill(student_logits, teacher_logits, labels)

        stage = min(self.current_stage, len(assistant_logits))
        target_logits = (
            assistant_logits[stage] if stage < len(assistant_logits) else teacher_logits
        )

        return self.logit_distill(student_logits, target_logits, labels)


class ContrastiveDistillation(nn.Module):
    """Contrastive knowledge distillation using contrastive learning.

    Transfers relational knowledge through contrastive objectives.

    Reference: Tian et al., "Contrastive Representation Distillation", ICLR 2020

    Args:
        temperature: Temperature for contrastive loss
        feature_dim: Dimension of projected features
    """

    def __init__(
        self,
        temperature: float = 0.07,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.temperature = temperature
        self.feature_dim = feature_dim

    def forward(
        self,
        student_features: Tensor,
        teacher_features: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute contrastive distillation loss."""
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features, dim=-1)

        similarity = (
            torch.matmul(student_features, teacher_features.t()) / self.temperature
        )

        batch_size = student_features.size(0)
        labels = torch.arange(batch_size, device=student_features.device)

        loss = F.cross_entropy(similarity, labels)

        return loss, {"contrastive_loss": loss.item()}


class ComprehensiveDistillation(nn.Module):
    """Comprehensive distillation combining multiple distillation methods.

    Args:
        temperature: Temperature for logit distillation
        logit_weight: Weight for logit distillation
        feature_weight: Weight for feature distillation
        attention_weight: Weight for attention transfer
        relation_weight: Weight for relation distillation
    """

    def __init__(
        self,
        temperature: float = 4.0,
        logit_weight: float = 1.0,
        feature_weight: float = 0.5,
        attention_weight: float = 0.5,
        relation_weight: float = 0.3,
    ):
        super().__init__()
        self.logit_weight = logit_weight
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight
        self.relation_weight = relation_weight

        self.logit_distill = LogitDistillation(temperature)
        self.feature_distill = FeatureDistillation()
        self.attention_transfer = AttentionTransfer()
        self.relation_distill = RelationDistillation()

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_features: Optional[List[Tensor]] = None,
        teacher_features: Optional[List[Tensor]] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute comprehensive distillation loss."""
        total_loss = torch.tensor(0.0, device=student_logits.device)
        metrics = {}

        logit_loss, logit_metrics = self.logit_distill(
            student_logits, teacher_logits, labels
        )
        total_loss = total_loss + self.logit_weight * logit_loss
        metrics.update(logit_metrics)

        if student_features is not None and teacher_features is not None:
            feature_loss = torch.tensor(0.0, device=student_logits.device)
            for sf, tf in zip(student_features, teacher_features):
                if sf.dim() == 4 and tf.dim() == 4:
                    attn_loss, _ = self.attention_transfer(sf, tf)
                    feature_loss = feature_loss + attn_loss

            if len(student_features) > 0:
                feature_loss = feature_loss / len(student_features)
                total_loss = total_loss + self.attention_weight * feature_loss
                metrics["attention_loss"] = feature_loss.item()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics


class DistillationTrainer:
    """High-level trainer for knowledge distillation.

    Args:
        teacher: Teacher model
        student: Student model
        distillation_loss: Distillation loss module
        optimizer: Optimizer for student
        device: Device to use
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        distillation_loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
    ):
        self.teacher = teacher
        self.student = student
        self.distillation_loss = distillation_loss
        self.optimizer = optimizer
        self.device = device

        self.teacher.to(device)
        self.student.to(device)
        self.teacher.eval()

        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_epoch(
        self,
        train_loader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train student for one epoch."""
        self.student.train()

        total_loss = 0.0
        total_metrics = {}
        num_batches = 0

        for batch in train_loader:
            if isinstance(batch, (tuple, list)):
                data, labels = batch[0].to(self.device), batch[1].to(self.device)
            else:
                data, labels = batch.to(self.device), None

            with torch.no_grad():
                teacher_output = self.teacher(data)

            student_output = self.student(data)

            loss, metrics = self.distillation_loss(
                student_output, teacher_output, labels
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v
            num_batches += 1

        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_metrics["loss"] = total_loss / num_batches

        return avg_metrics

    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate student model."""
        self.student.eval()

        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    data, labels = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                output = self.student(data)
                loss = F.cross_entropy(output, labels)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                total_correct += (pred == labels).sum().item()
                total_samples += labels.size(0)

        return {
            "val_loss": total_loss / len(val_loader),
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        }


__all__ = [
    "LogitDistillation",
    "FeatureDistillation",
    "AttentionTransfer",
    "RelationDistillation",
    "ProgressiveDistillation",
    "MultiStageDistillation",
    "ContrastiveDistillation",
    "ComprehensiveDistillation",
    "DistillationTrainer",
]
