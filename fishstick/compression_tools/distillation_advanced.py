"""
Advanced Knowledge Distillation Frameworks

Multi-teacher distillation, self-distillation, feature-based distillation,
attention transfer, and adaptive loss weighting.

References:
- https://arxiv.org/abs/1503.02531 (Distilling Knowledge in Neural Networks)
- https://arxiv.org/abs/1904.09149 (On the Efficacy of Knowledge Distillation)
- https://arxiv.org/abs/2006.05525 (Self-Distillation)
- https://arxiv.org/abs/1907.09699 (Feature Map Distillation)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Callable, Tuple, Union, Any
from enum import Enum
import copy
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class DistillationType(Enum):
    """Types of knowledge distillation."""

    LOGIT = "logit"
    FEATURE = "feature"
    ATTENTION = "attention"
    RELATION = "relation"
    SELF = "self"
    MULTI_TEACHER = "multi_teacher"


class KnowledgeDistiller:
    """Base knowledge distillation class.

    Implements the core distillation framework with configurable
    teacher-student architecture.

    Args:
        student_model: Student network to train
        teacher_model: Teacher network for knowledge transfer
        temperature: Temperature for softening probabilities
        alpha: Weight for distillation loss vs task loss
        beta: Weight for feature matching loss

    Example:
        >>> distiller = KnowledgeDistiller(student, teacher, temperature=4.0, alpha=0.7)
        >>> distiller.train_epoch(train_loader, optimizer)
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

        self.student_losses: List[float] = []
        self.distillation_losses: List[float] = []

        if teacher_model is not None:
            self.teacher.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False

    def compute_distillation_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
    ) -> Tensor:
        """Compute knowledge distillation loss.

        Args:
            student_logits: Student network outputs
            teacher_logits: Teacher network outputs

        Returns:
            Distillation loss
        """
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        distillation_loss = F.kl_div(
            student_log_probs,
            soft_targets,
            reduction="batchmean",
        )

        distillation_loss = distillation_loss * (self.temperature**2)

        return distillation_loss

    def compute_task_loss(
        self,
        student_logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute standard task loss."""
        return F.cross_entropy(student_logits, targets)

    def compute_total_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        targets: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute combined loss.

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        task_loss = self.compute_task_loss(student_logits, targets)

        loss_dict = {"task_loss": task_loss.item()}

        if self.teacher is not None:
            dist_loss = self.compute_distillation_loss(student_logits, teacher_logits)
            loss_dict["distillation_loss"] = dist_loss.item()

            total_loss = self.alpha * task_loss + (1 - self.alpha) * dist_loss
        else:
            total_loss = task_loss

        return total_loss, loss_dict

    def train_step(
        self,
        batch: Tuple[Tensor, Tensor],
        optimizer: Optimizer,
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: (input, target) tuple
            optimizer: Optimizer

        Returns:
            Dict of loss values
        """
        inputs, targets = batch

        optimizer.zero_grad()

        student_logits = self.student(inputs)

        if self.teacher is not None:
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
        else:
            teacher_logits = student_logits

        total_loss, loss_dict = self.compute_total_loss(
            student_logits, teacher_logits, targets
        )

        total_loss.backward()
        optimizer.step()

        self.student_losses.append(loss_dict.get("task_loss", 0.0))
        if "distillation_loss" in loss_dict:
            self.distillation_losses.append(loss_dict["distillation_loss"])

        return loss_dict

    def validate(
        self,
        val_loader: List[Tuple[Tensor, Tensor]],
    ) -> Dict[str, float]:
        """Validate student model.

        Args:
            val_loader: Validation data

        Returns:
            Dict of validation metrics
        """
        self.student.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                student_logits = self.student(inputs)

                if self.teacher is not None:
                    teacher_logits = self.teacher(inputs)
                    loss, _ = self.compute_total_loss(
                        student_logits, teacher_logits, targets
                    )
                else:
                    loss = F.cross_entropy(student_logits, targets)

                total_loss += loss.item()
                pred = student_logits.argmax(dim=1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)

        return {
            "val_loss": total_loss / max(len(val_loader), 1),
            "val_accuracy": 100.0 * correct / max(total, 1),
        }

    def train_epoch(
        self,
        train_loader: List[Tuple[Tensor, Tensor]],
        optimizer: Optimizer,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data
            optimizer: Optimizer

        Returns:
            Dict of epoch metrics
        """
        self.student.train()

        epoch_losses = []

        for batch in train_loader:
            loss_dict = self.train_step(batch, optimizer)
            epoch_losses.append(loss_dict)

        avg_losses = {
            key: sum(d[key] for d in epoch_losses) / max(len(epoch_losses), 1)
            for key in epoch_losses[0].keys()
        }

        return avg_losses


class MultiTeacherDistiller:
    """Multi-teacher knowledge distillation.

    Distills knowledge from multiple teacher networks into a single student.

    Args:
        student_model: Student network
        teacher_models: List of teacher networks
        temperature: Softmax temperature
        teacher_weights: Weights for each teacher (default: equal)
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_models: List[nn.Module],
        temperature: float = 4.0,
        teacher_weights: Optional[List[float]] = None,
    ):
        self.student = student_model
        self.teachers = teacher_models
        self.temperature = temperature
        self.teacher_weights = teacher_weights or [1.0 / len(teacher_models)] * len(
            teacher_models
        )

        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False

        self.student_losses: List[float] = []
        self.teacher_losses: List[List[float]] = [[] for _ in self.teachers]

    def compute_multi_teacher_loss(
        self,
        student_logits: Tensor,
        teacher_logits_list: List[Tensor],
    ) -> Tensor:
        """Compute loss from multiple teachers.

        Args:
            student_logits: Student outputs
            teacher_logits_list: List of teacher outputs

        Returns:
            Weighted distillation loss
        """
        total_loss = 0.0

        for i, (teacher_logits, weight) in enumerate(
            zip(teacher_logits_list, self.teacher_weights)
        ):
            soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

            dist_loss = F.kl_div(
                student_log_probs,
                soft_targets,
                reduction="batchmean",
            )

            total_loss += weight * dist_loss * (self.temperature**2)
            self.teacher_losses[i].append(dist_loss.item())

        return total_loss

    def train_step(
        self,
        batch: Tuple[Tensor, Tensor],
        optimizer: Optimizer,
        task_loss_weight: float = 0.5,
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: (input, target) tuple
            optimizer: Optimizer
            task_loss_weight: Weight for task loss

        Returns:
            Dict of loss values
        """
        inputs, targets = batch

        optimizer.zero_grad()

        student_logits = self.student(inputs)

        teacher_logits_list = []
        with torch.no_grad():
            for teacher in self.teachers:
                teacher_logits_list.append(teacher(inputs))

        task_loss = F.cross_entropy(student_logits, targets)
        dist_loss = self.compute_multi_teacher_loss(student_logits, teacher_logits_list)

        total_loss = task_loss_weight * task_loss + (1 - task_loss_weight) * dist_loss

        total_loss.backward()
        optimizer.step()

        self.student_losses.append(task_loss.item())

        return {
            "task_loss": task_loss.item(),
            "distillation_loss": dist_loss.item(),
            "total_loss": total_loss.item(),
        }

    def get_teacher_diversity(self) -> float:
        """Calculate diversity between teacher predictions."""
        if len(self.teachers) < 2:
            return 0.0

        all_predictions = []

        for teacher in self.teachers:
            preds = []
            with torch.no_grad():
                for _ in range(10):
                    dummy_input = torch.randn(1, 3, 224, 224)
                    output = teacher(dummy_input)
                    preds.append(output.argmax(dim=1).item())
            all_predictions.append(set(preds))

        diversity = 0.0
        for i in range(len(all_predictions)):
            for j in range(i + 1, len(all_predictions)):
                intersection = len(all_predictions[i] & all_predictions[j])
                union = len(all_predictions[i] | all_predictions[j])
                diversity += 1 - intersection / max(union, 1)

        return diversity / max(len(self.teachers) - 1, 1)


class SelfDistiller:
    """Self-distillation framework.

    Uses a network to distill knowledge into itself through
    intermediate layers or during multiple training rounds.

    Args:
        model: Model for self-distillation
        num_stages: Number of distillation stages
        temperature: Softmax temperature
    """

    def __init__(
        self,
        model: nn.Module,
        num_stages: int = 3,
        temperature: float = 4.0,
    ):
        self.model = model
        self.num_stages = num_stages
        self.temperature = temperature
        self.stage_models: List[nn.Module] = []
        self.stage_losses: List[List[float]] = [[] for _ in range(num_stages)]

    def _create_stage_model(self, stage: int) -> nn.Module:
        """Create model for a specific stage."""
        stage_model = copy.deepcopy(self.model)

        for i, (name, param) in enumerate(stage_model.named_parameters()):
            if i < stage * len(list(stage_model.parameters())) // self.num_stages:
                param.requires_grad = False

        return stage_model

    def distill(
        self,
        train_loader: List[Tuple[Tensor, Tensor]],
        val_loader: List[Tuple[Tensor, Tensor]],
        num_epochs: int = 10,
    ) -> Dict[int, Dict[str, float]]:
        """Perform self-distillation across stages.

        Args:
            train_loader: Training data
            val_loader: Validation data
            num_epochs: Epochs per stage

        Returns:
            Dict of stage to metrics
        """
        results = {}

        for stage in range(self.num_stages):
            print(f"Training stage {stage + 1}/{self.num_stages}")

            stage_model = self._create_stage_model(stage)
            optimizer = torch.optim.SGD(stage_model.parameters(), lr=0.01)

            for epoch in range(num_epochs):
                stage_model.train()
                for inputs, targets in train_loader:
                    optimizer.zero_grad()

                    logits = stage_model(inputs)
                    task_loss = F.cross_entropy(logits, targets)

                    if stage > 0:
                        with torch.no_grad():
                            teacher_logits = self.stage_models[stage - 1](inputs)

                        dist_loss = self._distillation_loss(logits, teacher_logits)
                        loss = 0.5 * task_loss + 0.5 * dist_loss
                    else:
                        loss = task_loss

                    loss.backward()
                    optimizer.step()

                    self.stage_losses[stage].append(loss.item())

            stage_accuracy = self._evaluate(stage_model, val_loader)
            results[stage] = {
                "loss": sum(self.stage_losses[stage])
                / max(len(self.stage_losses[stage]), 1),
                "accuracy": stage_accuracy,
            }

            self.stage_models.append(stage_model)

        return results

    def _distillation_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
    ) -> Tensor:
        """Compute distillation loss."""
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        return F.kl_div(student_log_probs, soft_targets, reduction="batchmean") * (
            self.temperature**2
        )

    def _evaluate(
        self,
        model: nn.Module,
        val_loader: List[Tuple[Tensor, Tensor]],
    ) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)

        return 100.0 * correct / max(total, 1)


class FeatureRepresentationDistiller:
    """Feature-based knowledge distillation.

    Distills knowledge by matching intermediate feature representations
    between teacher and student networks.

    Args:
        student_model: Student network
        teacher_model: Teacher network
        feature_layers: Layers to extract features from
        projection_dims: Dimension for projection layers
        temperature: Softmax temperature
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        feature_layers: List[str],
        projection_dims: Optional[List[int]] = None,
        temperature: float = 4.0,
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.feature_layers = feature_layers
        self.temperature = temperature

        self.projection_layers: Dict[str, nn.Module] = {}

        if projection_dims is None:
            projection_dims = [512] * len(feature_layers)

        for layer, dim in zip(feature_layers, projection_dims):
            self.projection_layers[layer] = nn.Linear(dim, dim)

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def extract_features(
        self,
        model: nn.Module,
        inputs: Tensor,
    ) -> Dict[str, Tensor]:
        """Extract intermediate features."""
        features = {}
        handles = []

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, Tensor):
                    features[name] = output.flatten(1)
                elif isinstance(output, tuple):
                    features[name] = output[0].flatten(1)

            return hook

        for name, module in model.named_modules():
            if name in self.feature_layers:
                handles.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            model(inputs)

        for handle in handles:
            handle.remove()

        return features

    def compute_feature_loss(
        self,
        student_features: Dict[str, Tensor],
        teacher_features: Dict[str, Tensor],
    ) -> Tensor:
        """Compute feature matching loss.

        Args:
            student_features: Student network features
            teacher_features: Teacher network features

        Returns:
            Feature distillation loss
        """
        total_loss = 0.0

        for layer in self.feature_layers:
            if layer not in student_features or layer not in teacher_features:
                continue

            student_feat = student_features[layer]
            teacher_feat = teacher_features[layer]

            if layer in self.projection_layers:
                student_feat = self.projection_layers[layer](student_feat)

            loss = F.mse_loss(student_feat, teacher_feat)
            total_loss += loss

        return total_loss

    def train_step(
        self,
        batch: Tuple[Tensor, Tensor],
        optimizer: Optimizer,
        task_loss_weight: float = 0.5,
    ) -> Dict[str, float]:
        """Single training step."""
        inputs, targets = batch

        optimizer.zero_grad()

        student_logits = self.student(inputs)
        task_loss = F.cross_entropy(student_logits, targets)

        with torch.no_grad():
            teacher_features = self.extract_features(self.teacher, inputs)
            teacher_logits = self.teacher(inputs)

        student_features = self.extract_features(self.student, inputs)

        feature_loss = self.compute_feature_loss(student_features, teacher_features)
        logit_loss = self._distillation_loss(student_logits, teacher_logits)

        total_loss = task_loss_weight * task_loss + (1 - task_loss_weight) * (
            feature_loss + logit_loss
        )

        total_loss.backward()
        optimizer.step()

        return {
            "task_loss": task_loss.item(),
            "feature_loss": feature_loss.item(),
            "logit_loss": logit_loss.item(),
            "total_loss": total_loss.item(),
        }

    def _distillation_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
    ) -> Tensor:
        """Compute logit distillation loss."""
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        return F.kl_div(student_log_probs, soft_targets, reduction="batchmean") * (
            self.temperature**2
        )


class AdaptiveDistillationLoss:
    """Adaptive loss weighting for distillation.

    Automatically adjusts weights for different distillation losses
    based on their gradient magnitudes.

    Args:
        loss_components: Names of loss components
        temperature: Temperature for loss weighting

    Example:
        >>> loss_fn = AdaptiveDistillationLoss(['task', 'feature', 'logit'])
        >>> loss = loss_fn.compute(total_loss, loss_dict)
    """

    def __init__(
        self,
        loss_components: List[str],
        temperature: float = 2.0,
    ):
        self.loss_components = loss_components
        self.temperature = temperature
        self.loss_history: Dict[str, List[float]] = {
            name: [] for name in loss_components
        }
        self.loss_weights: Dict[str, float] = {
            name: 1.0 / len(loss_components) for name in loss_components
        }

    def update_weights(
        self,
        loss_dict: Dict[str, float],
        gradient_norms: Dict[str, float],
    ):
        """Update loss weights based on losses and gradients.

        Args:
            loss_dict: Dict of loss values
            gradient_norms: Dict of gradient norms for each loss
        """
        for name in self.loss_components:
            if name in loss_dict:
                self.loss_history[name].append(loss_dict[name])

        target_ratio = 1.0 / len(self.loss_components)

        for name in self.loss_components:
            if name not in gradient_norms:
                continue

            current_ratio = gradient_norms[name] / sum(gradient_norms.values())

            adjustment = math.log(current_ratio / max(target_ratio, 1e-8))
            adjustment = torch.sigmoid(
                torch.tensor(adjustment / self.temperature)
            ).item()

            self.loss_weights[name] = (1 - self.temperature) * self.loss_weights[
                name
            ] + self.temperature * adjustment

        total_weight = sum(self.loss_weights.values())
        self.loss_weights = {
            name: weight / max(total_weight, 1e-8)
            for name, weight in self.loss_weights.items()
        }

    def compute_weighted_loss(
        self,
        loss_dict: Dict[str, float],
    ) -> Tensor:
        """Compute weighted combination of losses.

        Args:
            loss_dict: Dict of loss values

        Returns:
            Weighted total loss
        """
        total_loss = 0.0

        for name, weight in self.loss_weights.items():
            if name in loss_dict:
                total_loss += weight * loss_dict[name]

        return torch.tensor(total_loss)

    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.loss_weights.copy()


class AttentionTransfer:
    """Attention transfer distillation.

    Transfers knowledge by matching attention maps between
    teacher and student networks.

    Args:
        student_model: Student network
        teacher_model: Teacher network
        attention_layers: Layers to extract attention from
        attention_type: Type of attention (spatial, channel, scaled)
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        attention_layers: List[str],
        attention_type: str = "scaled",
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.attention_layers = attention_layers
        self.attention_type = attention_type

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_attention(
        self,
        features: Tensor,
    ) -> Tensor:
        """Compute attention map from features.

        Args:
            features: Feature tensor

        Returns:
            Attention map
        """
        if self.attention_type == "spatial":
            attention = features.abs().mean(dim=1)
            attention = F.normalize(attention, p=1, dim=(-2, -1))
        elif self.attention_type == "channel":
            attention = features.abs().mean(dim=(-2, -1))
            attention = F.normalize(attention, p=1, dim=-1)
        elif self.attention_type == "scaled":
            attention = (features**2).mean(dim=1)
            attention = F.normalize(attention, p=2, dim=(-2, -1))
        else:
            attention = features.abs()

        return attention

    def compute_attention_loss(
        self,
        student_attentions: Dict[str, Tensor],
        teacher_attentions: Dict[str, Tensor],
    ) -> Tensor:
        """Compute attention transfer loss.

        Args:
            student_attentions: Student attention maps
            teacher_attentions: Teacher attention maps

        Returns:
            Attention transfer loss
        """
        total_loss = 0.0

        for layer in self.attention_layers:
            if layer not in student_attentions or layer not in teacher_attentions:
                continue

            student_attn = student_attentions[layer]
            teacher_attn = teacher_attentions[layer]

            loss = F.mse_loss(student_attn, teacher_attn)
            total_loss += loss

        return total_loss

    def extract_attentions(
        self,
        model: nn.Module,
        inputs: Tensor,
    ) -> Dict[str, Tensor]:
        """Extract attention maps from model."""
        attentions = {}
        handles = []

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, Tensor):
                    attentions[name] = self.compute_attention(output)

            return hook

        for name, module in model.named_modules():
            if name in self.attention_layers:
                handles.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            model(inputs)

        for handle in handles:
            handle.remove()

        return attentions

    def train_step(
        self,
        batch: Tuple[Tensor, Tensor],
        optimizer: Optimizer,
        task_loss_weight: float = 0.5,
    ) -> Dict[str, float]:
        """Single training step."""
        inputs, targets = batch

        optimizer.zero_grad()

        student_logits = self.student(inputs)
        task_loss = F.cross_entropy(student_logits, targets)

        with torch.no_grad():
            teacher_attentions = self.extract_attentions(self.teacher, inputs)

        student_attentions = self.extract_attentions(self.student, inputs)

        attention_loss = self.compute_attention_loss(
            student_attentions, teacher_attentions
        )

        total_loss = (
            task_loss_weight * task_loss + (1 - task_loss_weight) * attention_loss
        )

        total_loss.backward()
        optimizer.step()

        return {
            "task_loss": task_loss.item(),
            "attention_loss": attention_loss.item(),
            "total_loss": total_loss.item(),
        }
