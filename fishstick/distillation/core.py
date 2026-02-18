"""
Knowledge Distillation Module for Fishstick

This module provides comprehensive knowledge distillation implementations including:
- Distillation Methods: Various teacher-student knowledge transfer techniques
- Distillation Losses: Multiple loss functions for knowledge transfer
- Distillation Strategies: Different approaches to distillation
- Task-Specific: Specialized distillation for various tasks
- Data-Free: Techniques for distillation without original data
- Quantization: Model compression through distillation
- Utilities: Trainer, wrappers, and evaluation tools
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


T = TypeVar("T")
M = TypeVar("M", bound=nn.Module)
L = TypeVar("L", bound=nn.Module)


class DistillationLoss(nn.Module, ABC):
    """Base class for all distillation losses."""

    def __init__(self, temperature: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    @abstractmethod
    def forward(
        self, student_output: Tensor, teacher_output: Tensor, **kwargs
    ) -> Tensor:
        """Compute distillation loss."""
        pass


class KLDivergenceLoss(DistillationLoss):
    """
    Standard Knowledge Distillation using KL Divergence.

    Transfers knowledge from teacher to student by matching
    softened probability distributions.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = "batchmean",
    ):
        super().__init__(temperature=temperature, alpha=alpha)
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(
        self, student_output: Tensor, teacher_output: Tensor, **kwargs
    ) -> Tensor:
        """
        Compute KL divergence loss between student and teacher outputs.

        Args:
            student_output: Logits from student model
            teacher_output: Logits from teacher model
            **kwargs: Additional arguments (e.g., hard labels for alpha blending)

        Returns:
            KL divergence loss
        """
        student_soft = F.log_softmax(student_output / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_output / self.temperature, dim=-1)

        kl_loss = self.kl_loss(student_soft, teacher_soft)
        return kl_loss * (self.temperature**2)


class CosineEmbeddingLoss(DistillationLoss):
    """
    Cosine Embedding Loss for feature alignment.

    Aligns intermediate representations between teacher and student
    using cosine similarity.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        alpha: float = 0.5,
        margin: float = 0.0,
    ):
        super().__init__(temperature=temperature, alpha=alpha)
        self.margin = margin
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin)

    def forward(
        self, student_output: Tensor, teacher_output: Tensor, **kwargs
    ) -> Tensor:
        """
        Compute cosine embedding loss.

        Args:
            student_output: Student features
            teacher_output: Teacher features
            **kwargs: Must include 'target' tensor of ones

        Returns:
            Cosine embedding loss
        """
        target = kwargs.get("target", torch.ones(student_output.size(0)))
        if student_output.device != target.device:
            target = target.to(student_output.device)

        return self.cosine_loss(student_output, teacher_output, target)


class AttentionTransferLoss(DistillationLoss):
    """
    Attention Transfer Loss for intermediate layers.

    Transfers attention maps from teacher to student to capture
    spatial attention patterns.
    """

    def __init__(self, temperature: float = 1.0, alpha: float = 0.5):
        super().__init__(temperature=temperature, alpha=alpha)

    def forward(
        self, student_output: Tensor, teacher_output: Tensor, **kwargs
    ) -> Tensor:
        """
        Compute attention transfer loss.

        Args:
            student_output: Student attention maps [B, H, W] or [B, N, D]
            teacher_output: Teacher attention maps

        Returns:
            MSE loss between attention maps
        """
        student_attn = self._compute_attention(student_output)
        teacher_attn = self._compute_attention(teacher_output)

        return F.mse_loss(student_attn, teacher_attn)

    def _compute_attention(self, x: Tensor) -> Tensor:
        """Compute attention map from input tensor."""
        if x.dim() == 4:
            attention = torch.norm(x, p=2, dim=1)
            attention = attention.view(attention.size(0), -1)
        elif x.dim() == 3:
            attention = torch.norm(x, p=2, dim=-1)
        else:
            attention = x
        attention = F.normalize(attention, p=2, dim=-1)
        return attention


class FSPMatrixLoss(DistillationLoss):
    """
    Flow of Solution Procedure (FSP) Matrix Loss.

    Computes relationships between intermediate features as matrices
    and matches them between teacher and student.
    """

    def __init__(self, temperature: float = 1.0, alpha: float = 0.5):
        super().__init__(temperature=temperature, alpha=alpha)

    def forward(
        self, student_output: Tensor, teacher_output: Tensor, **kwargs
    ) -> Tensor:
        """
        Compute FSP matrix loss.

        Args:
            student_output: Tuple of (student_feat1, student_feat2)
            teacher_output: Tuple of (teacher_feat1, teacher_feat2)

        Returns:
            FSP matrix loss
        """
        if isinstance(student_output, (list, tuple)) and len(student_output) == 2:
            s_feat1, s_feat2 = student_output
        else:
            s_feat1 = student_output[:, :-1]
            s_feat2 = student_output[:, 1:]

        if isinstance(teacher_output, (list, tuple)) and len(teacher_output) == 2:
            t_feat1, t_feat2 = teacher_output
        else:
            t_feat1 = teacher_output[:, :-1]
            t_feat2 = teacher_output[:, 1:]

        s_fsp = self._compute_fsp(s_feat1, s_feat2)
        t_fsp = self._compute_fsp(t_feat1, t_feat2)

        return F.mse_loss(s_fsp, t_fsp)

    def _compute_fsp(self, feat1: Tensor, feat2: Tensor) -> Tensor:
        """Compute FSP matrix."""
        batch_size = feat1.size(0)
        feat1 = feat1.view(batch_size, -1)
        feat2 = feat2.view(batch_size, -1)
        fsp = torch.mm(feat1, feat2.t())
        fsp = fsp / (feat1.size(1) + 1e-8)
        return fsp


class RKDLoss(DistillationLoss):
    """
    Relational Knowledge Distillation Loss.

    Captures relational knowledge between sample pairs rather than
    individual predictions.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        alpha: float = 0.5,
        beta: float = 1.0,
    ):
        super().__init__(temperature=temperature, alpha=alpha)
        self.beta = beta

    def forward(
        self, student_output: Tensor, teacher_output: Tensor, **kwargs
    ) -> Tensor:
        """
        Compute relational knowledge distillation loss.

        Args:
            student_output: Student logits or features [B, D]
            teacher_output: Teacher logits or features [B, D]

        Returns:
            RKD loss
        """
        student_flat = student_output.flatten(1)
        teacher_flat = teacher_output.flatten(1)

        td = self._distance_wise(teacher_flat)
        sd = self._distance_wise(student_flat)

        loss_d = F.mse_loss(sd, td)

        ta = self._angle_wise(teacher_flat)
        sa = self._angle_wise(student_flat)

        loss_a = F.mse_loss(sa, ta)

        return loss_d + self.beta * loss_a

    def _distance_wise(self, features: Tensor) -> Tensor:
        """Compute pairwise distances."""
        batch_size = features.size(0)
        dist_matrix = torch.cdist(features, features, p=2)
        dist_matrix = dist_matrix.view(batch_size, -1)
        return dist_matrix

    def _angle_wise(self, features: Tensor) -> Tensor:
        """Compute angle-wise relations."""
        batch_size = features.size(0)
        features = F.normalize(features, p=2, dim=-1)
        similarity = torch.mm(features, features.t())
        return similarity.view(batch_size, -1)


class VanillaDistillation(nn.Module):
    """
    Basic Vanilla Knowledge Distillation.

    Simple teacher-student distillation using KL divergence
    between softened probability distributions.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        hard_label_weight: float = 0.5,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.hard_label_weight = hard_label_weight

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss()

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through teacher and student.

        Args:
            inputs: Input tensors
            hard_labels: Ground truth labels (optional)

        Returns:
            Tuple of (student_output, loss_dict)
        """
        with torch.no_grad():
            teacher_output = self.teacher(inputs)

        student_output = self.student(inputs)

        soft_loss = self.kl_loss(student_output, teacher_output)

        if hard_labels is not None:
            hard_loss = self.ce_loss(student_output, hard_labels)
            total_loss = (
                self.alpha * soft_loss
                + (1 - self.alpha) * self.hard_label_weight * hard_loss
            )
        else:
            total_loss = soft_loss

        return student_output, {
            "total_loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_labels,
        }


class LabelSmoothingDistillation(nn.Module):
    """
    Label Smoothing Distillation with Soft Labels.

    Combines label smoothing with knowledge distillation for
    improved generalization.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        smoothing: float = 0.1,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.smoothing = smoothing

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=smoothing)

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with label smoothing."""
        with torch.no_grad():
            teacher_output = self.teacher(inputs)

        student_output = self.student(inputs)

        soft_loss = self.kl_loss(student_output, teacher_output)

        if hard_labels is not None:
            hard_loss = self.ce_loss(student_output, hard_labels)
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = soft_loss

        return student_output, {
            "total_loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_labels,
        }


class IntermediateDistillation(nn.Module):
    """
    Intermediate Layer Distillation.

    Transfers knowledge from intermediate hidden layers,
    not just final outputs.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        feature_layers: List[str],
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_alpha: float = 0.5,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.feature_layers = feature_layers
        self.temperature = temperature
        self.alpha = alpha
        self.feature_alpha = feature_alpha

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss()

        self.teacher_features: Dict[str, Tensor] = {}
        self.student_features: Dict[str, Tensor] = {}

        self._register_hooks()

        for param in self.teacher.parameters():
            param.requires_grad = False

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""

        def get_teacher_features(name: str):
            def hook(module, input, output):
                self.teacher_features[name] = output

            return hook

        def get_student_features(name: str):
            def hook(module, input, output):
                self.student_features[name] = output

            return hook

        for name, module in self.teacher.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(get_teacher_features(name))

        for name, module in self.student.named_modules():
            if name in self.feature_layers:
                module.register_forward_hook(get_student_features(name))

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with intermediate feature matching."""
        self.teacher_features.clear()
        self.student_features.clear()

        with torch.no_grad():
            teacher_output = self.teacher(inputs)

        student_output = self.student(inputs)

        soft_loss = self.kl_loss(student_output, teacher_output)

        feature_loss = 0.0
        for layer_name in self.feature_layers:
            if (
                layer_name in self.teacher_features
                and layer_name in self.student_features
            ):
                t_feat = self.teacher_features[layer_name]
                s_feat = self.student_features[layer_name]

                if t_feat.shape != s_feat.shape:
                    s_feat = self._match_dimensions(s_feat, t_feat)

                feature_loss += F.mse_loss(s_feat, t_feat)

        feature_loss = feature_loss / len(self.feature_layers)

        if hard_labels is not None:
            hard_loss = self.ce_loss(student_output, hard_labels)
            total_loss = (
                self.alpha * soft_loss
                + self.feature_alpha * feature_loss
                + (1 - self.alpha - self.feature_alpha) * hard_loss
            )
        else:
            total_loss = self.alpha * soft_loss + self.feature_alpha * feature_loss

        return student_output, {
            "total_loss": total_loss,
            "soft_loss": soft_loss,
            "feature_loss": feature_loss,
            "hard_loss": hard_labels,
        }

    def _match_dimensions(self, student_feat: Tensor, teacher_feat: Tensor) -> Tensor:
        """Match student features to teacher dimensions using projection."""
        if student_feat.shape[1] != teacher_feat.shape[1]:
            proj = nn.Linear(
                student_feat.shape[1], teacher_feat.shape[1], device=student_feat.device
            ).to(student_feat.dtype)
            student_feat = proj(student_feat)
        return student_feat


class SelfDistillation(nn.Module):
    """
    Self-Knowledge Distillation.

    Uses the model itself as teacher, training earlier layers
    from later layers.
    """

    def __init__(
        self,
        student: nn.Module,
        num_stages: int = 3,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.student = student
        self.num_stages = num_stages
        self.temperature = temperature
        self.alpha = alpha

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with self-distillation."""
        outputs = self.student(inputs)

        if isinstance(outputs, tuple):
            stage_outputs = outputs
        else:
            stage_outputs = [outputs] * self.num_stages

        total_loss = 0.0
        losses_dict: Dict[str, Tensor] = {}

        for i in range(1, len(stage_outputs)):
            prev_output = stage_outputs[i - 1]
            curr_output = stage_outputs[i]

            kd_loss = self.kl_loss(prev_output, curr_output.detach())
            total_loss += kd_loss
            losses_dict[f"stage_{i}_loss"] = kd_loss

        if hard_labels is not None:
            hard_loss = self.ce_loss(stage_outputs[-1], hard_labels)
            total_loss += hard_loss
            losses_dict["hard_loss"] = hard_loss

        return stage_outputs[-1], {
            "total_loss": total_loss,
            "hard_loss": hard_labels,
            **losses_dict,
        }


class MultiTeacherDistillation(nn.Module):
    """
    Multi-Teacher Knowledge Distillation.

    Combines knowledge from multiple teacher models into a single student.
    """

    def __init__(
        self,
        teachers: List[nn.Module],
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        teacher_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.teachers = teachers
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_weights = (
            teacher_weights
            if teacher_weights
            else [1.0 / len(teachers)] * len(teachers)
        )

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss()

        for teacher in self.teachers:
            for param in teacher.parameters():
                param.requires_grad = False

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with multi-teacher distillation."""
        teacher_outputs = []
        for teacher in self.teachers:
            with torch.no_grad():
                teacher_outputs.append(teacher(inputs))

        student_output = self.student(inputs)

        avg_teacher_output = torch.zeros_like(student_output)
        for i, t_output in enumerate(teacher_outputs):
            avg_teacher_output += self.teacher_weights[i] * t_output

        soft_loss = self.kl_loss(student_output, avg_teacher_output)

        if hard_labels is not None:
            hard_loss = self.ce_loss(student_output, hard_labels)
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = soft_loss

        return student_output, {
            "total_loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_labels,
        }


class FeatureDistillation(nn.Module):
    """
    Feature-Based Distillation Strategy.

    Extracts and matches intermediate features between teacher and student.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        feature_dim: int,
        projection_dim: Optional[int] = None,
        loss_type: str = "mse",
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.projection_dim = projection_dim or feature_dim

        if projection_dim and projection_dim != feature_dim:
            self.projector = nn.Linear(feature_dim, self.projection_dim)
        else:
            self.projector = nn.Identity()

        if loss_type == "mse":
            self.feature_loss_fn = nn.MSELoss()
        elif loss_type == "cosine":
            self.feature_loss_fn = CosineEmbeddingLoss()
        else:
            self.feature_loss_fn = nn.MSELoss()

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with feature distillation."""
        with torch.no_grad():
            teacher_features = self._extract_features(self.teacher, inputs)

        student_features = self._extract_features(self.student, inputs)

        student_features = self.projector(student_features)

        feature_loss = self.feature_loss_fn(student_features, teacher_features)

        student_output = self.student(inputs)
        ce_loss = (
            nn.CrossEntropyLoss()(student_output, hard_labels) if hard_labels else None
        )

        total_loss = feature_loss + (ce_loss if ce_loss is not None else 0)

        return student_output, {
            "total_loss": total_loss,
            "feature_loss": feature_loss,
            "ce_loss": ce_loss,
        }

    def _extract_features(self, model: nn.Module, inputs: Tensor) -> Tensor:
        """Extract features from model."""
        features = model(inputs)
        if isinstance(features, tuple):
            return features[0]
        return features


class ResponseDistillation(nn.Module):
    """
    Response-Based Distillation Strategy.

    Matches final outputs/logits between teacher and student.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with response distillation."""
        with torch.no_grad():
            teacher_output = self.teacher(inputs)

        student_output = self.student(inputs)

        soft_loss = self.kl_loss(student_output, teacher_output)

        if hard_labels is not None:
            hard_loss = self.ce_loss(student_output, hard_labels)
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = soft_loss

        return student_output, {
            "total_loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_labels,
        }


class RelationDistillation(nn.Module):
    """
    Relational Distillation Strategy.

    Captures relationships between samples and transfers them.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.rkd_loss = RKDLoss(temperature=temperature)

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with relational distillation."""
        with torch.no_grad():
            teacher_features = self.teacher(inputs)

        student_features = self.student(inputs)

        rel_loss = self.rkd_loss(student_features, teacher_features)

        student_output = self.student(inputs)
        ce_loss = (
            nn.CrossEntropyLoss()(student_output, hard_labels) if hard_labels else None
        )

        total_loss = rel_loss + (ce_loss if ce_loss is not None else 0)

        return student_output, {
            "total_loss": total_loss,
            "relation_loss": rel_loss,
            "ce_loss": ce_loss,
        }


class LabelEmbeddingDistillation(nn.Module):
    """
    Label Embedding Based Distillation.

    Uses label embeddings to transfer structured knowledge.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        num_classes: int,
        embedding_dim: int,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.alpha = alpha

        self.label_embeddings = nn.Embedding(num_classes, embedding_dim)

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with label embedding distillation."""
        with torch.no_grad():
            teacher_output = self.teacher(inputs)

        student_output = self.student(inputs)

        soft_loss = self.kl_loss(student_output, teacher_output)

        if hard_labels is not None:
            label_emb = self.label_embeddings(hard_labels)

            student_emb = F.linear(
                F.normalize(student_output, dim=-1),
                F.normalize(self.label_embeddings.weight, dim=-1),
            )

            embed_loss = F.mse_loss(student_emb, label_emb)

            hard_loss = self.ce_loss(student_output, hard_labels)

            total_loss = (
                self.alpha * soft_loss
                + (1 - self.alpha) * 0.5 * hard_loss
                + (1 - self.alpha) * 0.5 * embed_loss
            )
        else:
            total_loss = soft_loss

        return student_output, {
            "total_loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_labels,
        }


class ImageClassificationDistillation(nn.Module):
    """
    Task-Specific Distillation for Image Classification.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        num_classes: int,
        temperature: float = 4.0,
        alpha: float = 0.5,
        use_attention_transfer: bool = True,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.use_attention_transfer = use_attention_transfer

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss()

        if use_attention_transfer:
            self.at_loss = AttentionTransferLoss()

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass for image classification."""
        with torch.no_grad():
            teacher_output = self.teacher(inputs)
            if isinstance(teacher_output, tuple):
                teacher_logits, teacher_features = teacher_output[0], teacher_output[1]
            else:
                teacher_logits = teacher_output
                teacher_features = None

        student_output = self.student(inputs)
        if isinstance(student_output, tuple):
            student_logits, student_features = student_output[0], student_output[1]
        else:
            student_logits = student_output
            student_features = None

        soft_loss = self.kl_loss(student_logits, teacher_logits)

        loss_dict = {"soft_loss": soft_loss}
        total_loss = self.alpha * soft_loss

        if self.use_attention_transfer and teacher_features is not None:
            if student_features is not None:
                at_loss = self.at_loss(student_features, teacher_features)
                loss_dict["attention_loss"] = at_loss
                total_loss = total_loss + (1 - self.alpha) * at_loss

        if hard_labels is not None:
            hard_loss = self.ce_loss(student_logits, hard_labels)
            loss_dict["hard_loss"] = hard_loss
            total_loss = total_loss + (1 - self.alpha) * hard_loss

        loss_dict["total_loss"] = total_loss

        return student_logits, loss_dict


class ObjectDetectionDistillation(nn.Module):
    """
    Task-Specific Distillation for Object Detection.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        distill_classification: bool = True,
        distill_box_regression: bool = True,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.distill_classification = distill_classification
        self.distill_box_regression = distill_box_regression

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.l1_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(
        self, inputs: Tensor, targets: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Forward pass for object detection."""
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)

        student_outputs = self.student(inputs)

        loss_dict: Dict[str, Tensor] = {}
        total_loss = 0.0

        if self.distill_classification:
            if isinstance(teacher_outputs, dict) and "pred_logits" in teacher_outputs:
                t_cls = teacher_outputs["pred_logits"]
                s_cls = student_outputs["pred_logits"]
                cls_loss = self.kl_loss(s_cls, t_cls)
                loss_dict["cls_loss"] = cls_loss
                total_loss = total_loss + self.alpha * cls_loss

        if self.distill_box_regression:
            if isinstance(teacher_outputs, dict) and "pred_boxes" in teacher_outputs:
                t_box = teacher_outputs["pred_boxes"]
                s_box = student_outputs["pred_boxes"]
                box_loss = self.smooth_l1_loss(s_box, t_box)
                loss_dict["box_loss"] = box_loss
                total_loss = total_loss + self.alpha * box_loss

        loss_dict["total_loss"] = total_loss

        return student_outputs, loss_dict


class SemanticSegmentationDistillation(nn.Module):
    """
    Task-Specific Distillation for Semantic Segmentation.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        num_classes: int,
        temperature: float = 4.0,
        alpha: float = 0.5,
        use_feature_distillation: bool = True,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.use_feature_distillation = use_feature_distillation

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass for semantic segmentation."""
        with torch.no_grad():
            teacher_output = self.teacher(inputs)

        student_output = self.student(inputs)

        soft_loss = self.kl_loss(
            student_output.view(-1, self.num_classes),
            teacher_output.view(-1, self.num_classes),
        )

        loss_dict = {"soft_loss": soft_loss}
        total_loss = self.alpha * soft_loss

        if hard_labels is not None:
            hard_loss = self.ce_loss(student_output, hard_labels)
            loss_dict["hard_loss"] = hard_loss
            total_loss = total_loss + (1 - self.alpha) * hard_loss

        if self.use_feature_distillation:
            teacher_features = teacher_output
            student_features = student_output
            if teacher_features.shape != student_features.shape:
                student_features = F.interpolate(
                    student_features,
                    size=teacher_features.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

        loss_dict["total_loss"] = total_loss

        return student_output, loss_dict


class NLPModelDistillation(nn.Module):
    """
    Task-Specific Distillation for NLP Models.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        distill_hidden_states: bool = True,
        distill_attention: bool = True,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.distill_hidden_states = distill_hidden_states
        self.distill_attention = distill_attention

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass for NLP model distillation."""
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids, attention_mask=attention_mask)

        student_outputs = self.student(input_ids, attention_mask=attention_mask)

        if isinstance(teacher_outputs, tuple):
            teacher_logits = teacher_outputs[0]
        else:
            teacher_logits = teacher_outputs["logits"]

        if isinstance(student_outputs, tuple):
            student_logits = student_outputs[0]
        else:
            student_logits = student_outputs["logits"]

        soft_loss = self.kl_loss(student_logits, teacher_logits)

        loss_dict = {"soft_loss": soft_loss}
        total_loss = self.alpha * soft_loss

        if self.distill_hidden_states and isinstance(teacher_outputs, tuple):
            if len(teacher_outputs) > 1:
                t_hidden = teacher_outputs[1]
                s_hidden = (
                    student_outputs[1] if isinstance(student_outputs, tuple) else None
                )
                if s_hidden is not None:
                    hidden_loss = self.mse_loss(s_hidden, t_hidden)
                    loss_dict["hidden_loss"] = hidden_loss
                    total_loss = total_loss + 0.5 * hidden_loss

        if labels is not None:
            ce_loss = nn.CrossEntropyLoss()(
                student_logits.view(-1, student_logits.size(-1)), labels.view(-1)
            )
            loss_dict["ce_loss"] = ce_loss
            total_loss = total_loss + (1 - self.alpha) * ce_loss

        loss_dict["total_loss"] = total_loss

        return student_logits, loss_dict


class SpeechRecognitionDistillation(nn.Module):
    """
    Task-Specific Distillation for Speech Recognition.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_dim: int = 80,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.feature_dim = feature_dim

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        audio_features: Tensor,
        transcript_lengths: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass for speech recognition."""
        with torch.no_grad():
            teacher_output = self.teacher(audio_features)

        student_output = self.student(audio_features)

        if teacher_output.dim() == 3:
            teacher_output = teacher_output.view(-1, teacher_output.size(-1))
        if student_output.dim() == 3:
            student_output = student_output.view(-1, student_output.size(-1))

        soft_loss = self.kl_loss(student_output, teacher_output)

        loss_dict = {"soft_loss": soft_loss}
        total_loss = self.alpha * soft_loss

        if targets is not None:
            ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)(
                F.log_softmax(student_output, dim=-1),
                targets,
                transcript_lengths or torch.tensor([student_output.size(0)]),
                torch.tensor([targets.size(0)]),
            )
            loss_dict["ctc_loss"] = ctc_loss
            total_loss = total_loss + (1 - self.alpha) * ctc_loss

        loss_dict["total_loss"] = total_loss

        return student_output, loss_dict


class DataFreeKD(nn.Module):
    """
    Data-Free Knowledge Distillation.

    Generates synthetic data for distillation when original data is unavailable.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        num_classes: int,
        latent_dim: int = 512,
        generator: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        if generator is None:
            self.generator = self._build_default_generator()
        else:
            self.generator = generator

        self.kl_loss = KLDivergenceLoss()

    def _build_default_generator(self) -> nn.Module:
        """Build default generator network."""
        return nn.Sequential(
            nn.Linear(self.latent_dim + self.num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh(),
        )

    def forward(self, num_samples: int = 100) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Generate synthetic data and compute distillation loss."""
        self.generator.train()

        noise = torch.randn(num_samples, self.latent_dim)
        class_labels = torch.randint(0, self.num_classes, (num_samples,))
        one_hot_labels = F.one_hot(class_labels, self.num_classes).float()

        synthetic_input = torch.cat([noise, one_hot_labels], dim=-1)
        synthetic_data = self.generator(synthetic_input)

        synthetic_data = synthetic_data.view(num_samples, 1, 28, 28)

        with torch.no_grad():
            teacher_output = self.teacher(synthetic_data)

        student_output = self.student(synthetic_data)

        loss = self.kl_loss(student_output, teacher_output)

        return student_output, {"distillation_loss": loss}

    def generate_batch(self, batch_size: int = 32) -> Tensor:
        """Generate a batch of synthetic data."""
        with torch.no_grad():
            noise = torch.randn(batch_size, self.latent_dim)
            class_probs = torch.ones(batch_size, self.num_classes) / self.num_classes
            class_labels = torch.multinomial(class_probs, 1).squeeze()
            one_hot_labels = F.one_hot(class_labels, self.num_classes).float()

            synthetic_input = torch.cat([noise, one_hot_labels], dim=-1)
            synthetic_data = self.generator(synthetic_input)

        return synthetic_data.view(batch_size, 1, 28, 28)


class DeepInversion(nn.Module):
    """
    Deep Inversion for Data-Free Knowledge Distillation.

    Inverts teacher to generate realistic synthetic samples.
    """

    def __init__(
        self,
        teacher: nn.Module,
        num_classes: int,
        image_size: Tuple[int, ...] = (1, 3, 224, 224),
        lr: float = 0.01,
        momentum: float = 0.9,
        decay: float = 0.0001,
        num_iterations: int = 30,
        alpha: float = 0.001,
        beta: float = 0.001,
    ):
        super().__init__()
        self.teacher = teacher
        self.num_classes = num_classes
        self.image_size = image_size
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta

        self.kl_loss = KLDivergenceLoss()

    def forward(self, batch_size: int = 32) -> Tuple[Tensor, Tensor]:
        """Generate inverted images."""
        images = torch.randn(batch_size, *self.image_size[1:], requires_grad=True)
        images = images.to(next(self.teacher.parameters()).device)

        momentum = torch.zeros_like(images)

        self.teacher.eval()

        for _ in range(self.num_iterations):
            teacher_output = self.teacher(images)

            prior_loss = self.alpha * torch.norm(images, p=2) ** 2

            total_variation = self.beta * (
                torch.sum(torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:]))
                + torch.sum(torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :]))
            )

            loss = (
                -self.kl_loss(torch.randn_like(teacher_output), teacher_output.detach())
                + prior_loss
                + total_variation
            )

            loss.backward()

            momentum = self.momentum * momentum - self.lr * images.grad
            images = images + momentum
            images.grad.zero_()

        return images, teacher_output

    def generate_inverted_images(
        self, num_samples: int = 100, batch_size: int = 32
    ) -> Tensor:
        """Generate multiple inverted images."""
        all_images = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for _ in range(num_batches):
            images, _ = self.forward(batch_size)
            all_images.append(images.detach())

        return torch.cat(all_images, dim=0)[:num_samples]


class GenerativeDistillation(nn.Module):
    """
    GAN-Based Generative Distillation.

    Uses adversarial training to generate high-quality synthetic data.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        num_classes: int,
        latent_dim: int = 128,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        self.kl_loss = KLDivergenceLoss()
        self.gan_loss = nn.BCEWithLogitsLoss()

    def _build_generator(self) -> nn.Module:
        """Build generator network."""
        return nn.Sequential(
            nn.Linear(self.latent_dim + self.num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh(),
        )

    def _build_discriminator(self) -> nn.Module:
        """Build discriminator network."""
        return nn.Sequential(
            nn.Linear(784 + self.num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, real_images: Tensor, labels: Tensor) -> Dict[str, Tensor]:
        """Forward pass with GAN-based distillation."""
        batch_size = real_images.size(0)
        device = real_images.device

        noise = torch.randn(batch_size, self.latent_dim, device=device)
        one_hot_labels = F.one_hot(labels, self.num_classes).float()

        fake_images = self.generator(torch.cat([noise, one_hot_labels], dim=-1))
        fake_images = fake_images.view(batch_size, -1)

        real_images_flat = real_images.view(batch_size, -1)
        combined = torch.cat([real_images_flat, one_hot_labels], dim=-1)
        fake_combined = torch.cat([fake_images, one_hot_labels], dim=-1)

        real_output = self.discriminator(combined)
        fake_output = self.discriminator(fake_combined.detach())

        d_loss_real = self.gan_loss(real_output, torch.ones_like(real_output))
        d_loss_fake = self.gan_loss(fake_output, torch.zeros_like(fake_output))
        d_loss = (d_loss_real + d_loss_fake) / 2

        fake_output_g = self.discriminator(fake_combined)
        g_loss = self.gan_loss(fake_output_g, torch.ones_like(fake_output_g))

        with torch.no_grad():
            teacher_output = self.teacher(real_images)

        student_output = self.student(real_images)

        distill_loss = self.kl_loss(student_output, teacher_output)

        total_g_loss = g_loss + distill_loss

        return {
            "d_loss": d_loss,
            "g_loss": total_g_loss,
            "distill_loss": distill_loss,
        }


class QuantizationAwareTraining(nn.Module):
    """
    Quantization-Aware Training with Distillation.

    Trains a quantized model using knowledge distillation.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        quantize_student: bool = True,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.quantize_student = quantize_student

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss()

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with quantization-aware training."""
        with torch.no_grad():
            teacher_output = self.teacher(inputs)

        if self.quantize_student:
            student_output = self._quantized_forward(inputs)
        else:
            student_output = self.student(inputs)

        soft_loss = self.kl_loss(student_output, teacher_output)

        if hard_labels is not None:
            hard_loss = self.ce_loss(student_output, hard_labels)
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = soft_loss

        return student_output, {
            "total_loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_labels,
        }

    def _quantized_forward(self, inputs: Tensor) -> Tensor:
        """Forward pass with quantization simulation."""
        output = self.student(inputs)

        if hasattr(self, "quantizer"):
            output = self.quantizer(output)

        return output

    def apply_quantization(self, num_bits: int = 8):
        """Apply quantization to student model."""
        self.quantizer = TensorQuantizer(num_bits=num_bits)

    def set_quantization_scheme(self, scheme: str = "dynamic"):
        """Set quantization scheme."""
        self.quantization_scheme = scheme


class PostTrainingQuantization(nn.Module):
    """
    Post-Training Quantization with Distillation.

    Performs quantization after training with knowledge transfer.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        calibration_data: Optional[DataLoader] = None,
        num_bits: int = 8,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.calibration_data = calibration_data
        self.num_bits = num_bits

        self.scale: Optional[Tensor] = None
        self.zero_point: Optional[Tensor] = None

    def calibrate(self, data_loader: DataLoader):
        """Calibrate quantization parameters."""
        self.teacher.eval()
        self.student.eval()

        outputs = []
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                outputs.append(self.student(inputs))

        all_outputs = torch.cat(outputs, dim=0)

        self.scale = (all_outputs.max() - all_outputs.min()) / (2**self.num_bits - 1)
        self.zero_point = all_outputs.min()

    def quantize(self, tensor: Tensor) -> Tensor:
        """Quantize tensor."""
        if self.scale is None or self.zero_point is None:
            raise RuntimeError("Must call calibrate() before quantize()")

        quantized = ((tensor - self.zero_point) / self.scale).round()
        quantized = torch.clamp(quantized, 0, 2**self.num_bits - 1)
        return quantized

    def dequantize(self, tensor: Tensor) -> Tensor:
        """Dequantize tensor."""
        if self.scale is None or self.zero_point is None:
            raise RuntimeError("Must call calibrate() before dequantize()")

        return tensor * self.scale + self.zero_point

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass with post-training quantization."""
        output = self.student(inputs)

        if self.training:
            return output

        quantized = self.quantize(output)
        return self.dequantize(quantized)


class MixedPrecisionDistillation(nn.Module):
    """
    Mixed Precision Distillation with Knowledge Transfer.

    Combines quantization and distillation for efficient deployment.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        fp16_weights: bool = True,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.fp16_weights = fp16_weights

        self.kl_loss = KLDivergenceLoss(temperature=temperature, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss()

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(
        self, inputs: Tensor, hard_labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward pass with mixed precision distillation."""
        with torch.no_grad():
            teacher_output = self.teacher(inputs)

        if self.fp16_weights:
            with torch.cuda.amp.autocast():
                student_output = self.student(inputs)
        else:
            student_output = self.student(inputs)

        soft_loss = self.kl_loss(student_output.float(), teacher_output.float())

        if hard_labels is not None:
            hard_loss = self.ce_loss(student_output.float(), hard_labels)
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = soft_loss

        return student_output.float(), {
            "total_loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_labels,
        }


class TensorQuantizer(nn.Module):
    """Simple tensor quantizer for quantization-aware training."""

    def __init__(self, num_bits: int = 8):
        super().__init__()
        self.num_bits = num_bits
        self.num_levels = 2**num_bits - 1

    def forward(self, x: Tensor) -> Tensor:
        """Quantize and dequantize tensor."""
        if not self.training:
            return x

        x_min = x.min()
        x_max = x.max()

        scale = (x_max - x_min) / self.num_levels
        zero_point = x_min

        quantized = ((x - zero_point) / scale).round()
        quantized = torch.clamp(quantized, 0, self.num_levels)

        dequantized = quantized * scale + zero_point

        return dequantized


class TeacherWrapper(nn.Module):
    """
    Wrapper for teacher model in distillation.

    Provides convenient interface for managing teacher models.
    """

    def __init__(self, teacher: nn.Module, freeze: bool = True):
        super().__init__()
        self.teacher = teacher
        self.freeze = freeze

        if freeze:
            for param in self.teacher.parameters():
                param.requires_grad = False

        self._register_hooks()

    def _register_hooks(self):
        """Register hooks for feature extraction."""
        self.features: Dict[str, Tensor] = {}

        def get_features(name: str):
            def hook(module, input, output):
                self.features[name] = output

            return hook

        for name, module in self.teacher.named_modules():
            if len(list(module.children())) == 0:
                module.register_forward_hook(get_features(name))

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through teacher."""
        return self.teacher(inputs)

    def get_features(self, layer_name: str) -> Optional[Tensor]:
        """Get features from specific layer."""
        return self.features.get(layer_name)

    def extract_logits(self, inputs: Tensor) -> Tensor:
        """Extract logits from teacher."""
        with torch.no_grad():
            output = self.teacher(inputs)
            if isinstance(output, tuple):
                return output[0]
            return output

    def extract_features(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Extract all registered features."""
        with torch.no_grad():
            self.teacher(inputs)
        return self.features.copy()


class TemperatureScheduler:
    """
    Scheduler for temperature in knowledge distillation.

    Implements various scheduling strategies for temperature.
    """

    def __init__(
        self,
        initial_temp: float = 4.0,
        strategy: str = "constant",
        warmup_epochs: int = 5,
        decay_rate: float = 0.9,
        min_temp: float = 1.0,
    ):
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.min_temp = min_temp
        self.epoch = 0

    def step(self, epoch: Optional[int] = None):
        """Update temperature based on strategy."""
        if epoch is not None:
            self.epoch = epoch

        if self.strategy == "constant":
            self.current_temp = self.initial_temp

        elif self.strategy == "linear_warmup":
            if self.epoch < self.warmup_epochs:
                self.current_temp = 1.0 + (self.initial_temp - 1.0) * (
                    self.epoch / self.warmup_epochs
                )
            else:
                self.current_temp = self.initial_temp

        elif self.strategy == "exponential_decay":
            if self.epoch > self.warmup_epochs:
                self.current_temp = max(
                    self.initial_temp
                    * (self.decay_rate ** (self.epoch - self.warmup_epochs)),
                    self.min_temp,
                )

        elif self.strategy == "cosine_annealing":
            if self.epoch >= self.warmup_epochs:
                self.current_temp = (
                    self.min_temp
                    + (self.initial_temp - self.min_temp)
                    * (1 + math.cos(math.pi * (self.epoch - self.warmup_epochs) / 100))
                    / 2
                )

        self.epoch += 1
        return self.current_temp

    def get_temperature(self) -> float:
        """Get current temperature."""
        return self.current_temp

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_temp = self.initial_temp
        self.epoch = 0


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    temperature: float = 4.0
    alpha: float = 0.5
    hard_label_weight: float = 0.5
    distillation_type: str = "vanilla"
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"soft": 1.0, "hard": 0.5}
    )
    use_cosine_loss: bool = False
    use_attention_transfer: bool = False
    feature_layers: List[str] = field(default_factory=list)


class DistillationTrainer:
    """
    Trainer class for knowledge distillation.

    Provides training loop and evaluation utilities.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[DistillationConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.config = config or DistillationConfig()
        self.device = device

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.optimizer = optimizer or torch.optim.Adam(
            self.student.parameters(), lr=0.001
        )

        self.temperature_scheduler = TemperatureScheduler(
            initial_temp=self.config.temperature
        )

        self.history: Dict[str, List[float]] = defaultdict(list)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.student.train()
        epoch_losses: Dict[str, float] = defaultdict(float)
        num_batches = 0

        temperature = self.temperature_scheduler.step(epoch)

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            else:
                inputs, labels = batch, None

            inputs = inputs.to(self.device)
            labels = labels.to(self.device) if labels is not None else None

            self.optimizer.zero_grad()

            with torch.no_grad():
                teacher_output = self.teacher(inputs)

            student_output = self.student(inputs)

            soft_loss = F.kl_div(
                F.log_softmax(student_output / temperature, dim=-1),
                F.softmax(teacher_output / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)

            loss_dict = {"soft_loss": soft_loss}

            if labels is not None:
                hard_loss = F.cross_entropy(student_output, labels)
                loss_dict["hard_loss"] = hard_loss

                total_loss = (
                    self.config.alpha * soft_loss + (1 - self.config.alpha) * hard_loss
                )
            else:
                total_loss = soft_loss

            total_loss.backward()
            self.optimizer.step()

            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()

            num_batches += 1

        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        epoch_losses["temperature"] = temperature
        self.history["train"].append(epoch_losses)

        return epoch_losses

    def evaluate(
        self,
        eval_loader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate student model."""
        self.student.eval()
        eval_losses: Dict[str, float] = defaultdict(float)
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in eval_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0], batch[1]
                else:
                    inputs, labels = batch, None

                inputs = inputs.to(self.device)
                labels = labels.to(self.device) if labels is not None else None

                student_output = self.student(inputs)

                if labels is not None:
                    hard_loss = F.cross_entropy(student_output, labels)
                    eval_losses["loss"] += hard_loss.item()

                    _, predicted = student_output.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

        for key in eval_losses:
            eval_losses[key] /= len(eval_loader)

        if total > 0:
            eval_losses["accuracy"] = 100.0 * correct / total

        self.history["eval"].append(eval_losses)

        return eval_losses

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        for epoch in range(num_epochs):
            train_losses = self.train_epoch(train_loader, epoch)

            if eval_loader is not None:
                eval_losses = self.evaluate(eval_loader)
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_losses.get('loss', 0):.4f} - "
                    f"Eval Loss: {eval_losses.get('loss', 0):.4f} - "
                    f"Eval Acc: {eval_losses.get('accuracy', 0):.2f}%"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_losses.get('loss', 0):.4f}"
                )

        return self.history


def distillation_evaluate(
    student: nn.Module,
    teacher: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate distillation performance.

    Args:
        student: Student model
        teacher: Teacher model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        temperature: Distillation temperature
        alpha: Weight for soft labels

    Returns:
        Dictionary with evaluation metrics
    """
    student.eval()
    teacher.eval()

    student = student.to(device)
    teacher = teacher.to(device)

    metrics: Dict[str, float] = defaultdict(float)
    num_batches = 0

    correct_student = 0
    correct_teacher = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            else:
                inputs, labels = batch, None

            inputs = inputs.to(device)
            labels = labels.to(device) if labels is not None else None

            student_output = student(inputs)
            teacher_output = teacher(inputs)

            soft_loss = F.kl_div(
                F.log_softmax(student_output / temperature, dim=-1),
                F.softmax(teacher_output / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)

            metrics["soft_loss"] += soft_loss.item()

            if labels is not None:
                hard_loss = F.cross_entropy(student_output, labels)
                metrics["hard_loss"] += hard_loss.item()

                _, predicted = student_output.max(1)
                total += labels.size(0)
                correct_student += predicted.eq(labels).sum().item()

                _, teacher_pred = teacher_output.max(1)
                correct_teacher += teacher_pred.eq(labels).sum().item()

            num_batches += 1

    for key in metrics:
        metrics[key] /= num_batches

    if total > 0:
        metrics["student_accuracy"] = 100.0 * correct_student / total
        metrics["teacher_accuracy"] = 100.0 * correct_teacher / total
        metrics["accuracy_gap"] = (
            metrics["teacher_accuracy"] - metrics["student_accuracy"]
        )

    return dict(metrics)


__all__ = [
    "DistillationLoss",
    "KLDivergenceLoss",
    "CosineEmbeddingLoss",
    "AttentionTransferLoss",
    "FSPMatrixLoss",
    "RKDLoss",
    "VanillaDistillation",
    "LabelSmoothingDistillation",
    "IntermediateDistillation",
    "SelfDistillation",
    "MultiTeacherDistillation",
    "FeatureDistillation",
    "ResponseDistillation",
    "RelationDistillation",
    "LabelEmbeddingDistillation",
    "ImageClassificationDistillation",
    "ObjectDetectionDistillation",
    "SemanticSegmentationDistillation",
    "NLPModelDistillation",
    "SpeechRecognitionDistillation",
    "DataFreeKD",
    "DeepInversion",
    "GenerativeDistillation",
    "QuantizationAwareTraining",
    "PostTrainingQuantization",
    "MixedPrecisionDistillation",
    "TensorQuantizer",
    "TeacherWrapper",
    "TemperatureScheduler",
    "DistillationConfig",
    "DistillationTrainer",
    "distillation_evaluate",
]
