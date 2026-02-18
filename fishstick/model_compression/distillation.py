"""
Model Distillation Module

Task-agnostic and architecture-agnostic knowledge distillation:
- Output-based distillation (logits, soft targets)
- Feature-based distillation (intermediate representations)
- Progressive distillation (stage-wise)
- Multi-source distillation (multiple teachers)
"""

from typing import Optional, Dict, Any, List, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from copy import deepcopy


class BaseDistiller:
    """Base class for all distillation methods."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def compute_distillation_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute distillation loss."""
        raise NotImplementedError

    def forward(
        self,
        inputs: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass through both teacher and student."""
        raise NotImplementedError

    def train_step(
        self,
        inputs: Tensor,
        labels: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Single training step."""
        raise NotImplementedError


class TaskAgnosticDistiller(BaseDistiller):
    """Task-agnostic distillation - works with any output format."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_extractor: Optional[Callable[[nn.Module], Tensor]] = None,
    ):
        super().__init__(teacher, student, temperature, alpha)
        self.feature_extractor = feature_extractor
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_dist self,
        studentillation_loss(
       _logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute KL divergence between student and teacher outputs."""
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)

        loss_kd = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return loss_kd

    def compute_feature_loss(
        self,
        student_features: Tensor,
        teacher_features: Tensor,
    ) -> Tensor:
        """Compute MSE loss between student and teacher features."""
        return F.mse_loss(student_features, teacher_features)

    def forward(
        self,
        inputs: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass with teacher frozen."""
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)

        student_outputs = self.student(inputs)

        return {
            'student_logits': student_outputs,
            'teacher_logits': teacher_outputs,
            'labels': labels,
        }

    def train_step(
        self,
        inputs: Tensor,
        labels: Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Single training step."""
        optimizer.zero_grad()

        outputs = self.forward(inputs, labels)

        loss_kd = self.compute_distillation_loss(
            outputs['student_logits'],
            outputs['teacher_logits'],
        )

        if criterion is not None and labels is not None:
            loss_ce = criterion(outputs['student_logits'], labels)
            loss = self.alpha * loss_kd + (1 - self.alpha) * loss_ce
        else:
            loss = loss_kd

        loss.backward()
        optimizer.step()

        return {
            'loss': loss.item(),
            'loss_kd': loss_kd.item(),
        }


class ArchitectureAgnosticDistiller(BaseDistiller):
    """Architecture-agnostic distillation - handles different architectures."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        projection_layers: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__(teacher, student, temperature, alpha)
        self.projection_layers = projection_layers or {}
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def add_projection(
        self,
        feature_name: str,
        projection: nn.Module,
    ):
        """Add projection layer for feature alignment."""
        self.projection_layers[feature_name] = projection

    def get_teacher_features(
        self,
        inputs: Tensor,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """Extract features from teacher model."""
        features = {}
        hooks = []

        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook

        if layer_names:
            for name in layer_names:
                module = self._get_module_by_name(self.teacher, name)
                if module is not None:
                    hooks.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            _ = self.teacher(inputs)

        for hook in hooks:
            hook.remove()

        return features

    def get_student_features(
        self,
        inputs: Tensor,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """Extract features from student model."""
        features = {}
        hooks = []

        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook

        if layer_names:
            for name in layer_names:
                module = self._get_module_by_name(self.student, name)
                if module is not None:
                    hooks.append(module.register_forward_hook(hook_fn(name)))

        _ = self.student(inputs)

        for hook in hooks:
            hook.remove()

        return features

    def _get_module_by_name(
        self,
        model: nn.Module,
        name: str,
    ) -> Optional[nn.Module]:
        """Get module by dot-separated name."""
        parts = name.split('.')
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def compute_aligned_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_features: Optional[Dict[str, Tensor]] = None,
        teacher_features: Optional[Dict[str, Tensor]] = None,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute combined output and feature distillation loss."""
        loss_kd = self.compute_distillation_loss(student_logits, teacher_logits)
        loss = self.alpha * loss_kd

        if student_features and teacher_features:
            for name in student_features.keys():
                s_feat = student_features[name]
                t_feat = teacher_features[name]

                if name in self.projection_layers:
                    s_feat = self.projection_layers[name](s_feat)

                if s_feat.shape != t_feat.shape:
                    s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:]) if s_feat.dim() == 4 else s_feat
                    if s_feat.shape != t_feat.shape:
                        s_feat = F.linear(s_feat, t_feat.size(-1)) if s_feat.dim() == 2 else s_feat

                loss_feat = F.mse_loss(s_feat, t_feat)
                loss += self.beta * loss_feat

        if labels is not None:
            loss_ce = F.cross_entropy(student_logits, labels)
            loss = self.alpha * loss + (1 - self.alpha) * loss_ce

        return loss


class ProgressiveDistiller(BaseDistiller):
    """Progressive distillation - trains student in stages."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        num_stages: int = 3,
    ):
        super().__init__(teacher, student, temperature, alpha)
        self.num_stages = num_stages
        self.current_stage = 0
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def set_stage(self, stage: int):
        """Set current training stage."""
        self.current_stage = stage

    def get_stage_temperature(self) -> float:
        """Get temperature for current stage."""
        return self.temperature * (1.0 + self.current_stage)

    def compute_distillation_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute distillation loss with stage-dependent temperature."""
        temp = self.get_stage_temperature()
        soft_teacher = F.softmax(teacher_logits / temp, dim=-1)
        soft_student = F.log_softmax(student_logits / temp, dim=-1)

        loss_kd = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (temp ** 2)

        return loss_kd


class MultiSourceDistiller(BaseDistiller):
    """Multi-source distillation - uses multiple teachers."""

    def __init__(
        self,
        teachers: List[nn.Module],
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        teacher_weights: Optional[List[float]] = None,
    ):
        super().__init__(teachers[0], student, temperature, alpha)
        self.teachers = teachers
        self.teacher_weights = teacher_weights or [1.0 / len(teachers)] * len(teachers)
        self.teacher_weights = [w / sum(self.teacher_weights) for w in self.teacher_weights]

        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False

    def compute_ensemble_teacher_output(
        self,
        inputs: Tensor,
    ) -> Tensor:
        """Compute weighted ensemble of teacher outputs."""
        with torch.no_grad():
            teacher_outputs = []
            for teacher in self.teachers:
                outputs = teacher(inputs)
                teacher_outputs.append(outputs)

        weighted_outputs = sum(
            w * out for w, out in zip(self.teacher_weights, teacher_outputs)
        )
        return weighted_outputs

    def compute_distillation_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute distillation loss with ensemble teachers."""
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)

        loss_kd = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return loss_kd

    def forward(
        self,
        inputs: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass with teacher ensemble."""
        teacher_logits = self.compute_ensemble_teacher_output(inputs)
        student_logits = self.student(inputs)

        return {
            'student_logits': student_logits,
            'teacher_logits': teacher_logits,
            'labels': labels,
        }


class FeatureBasedDistiller(BaseDistiller):
    """Feature-based distillation - aligns intermediate features."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        feature_layers: Optional[List[str]] = None,
    ):
        super().__init__(teacher, student, temperature, alpha)
        self.beta = beta
        self.feature_layers = feature_layers or []
        self.teacher_features: Dict[str, Tensor] = {}
        self.student_features: Dict[str, Tensor] = {}
        self._register_hooks()
        self.teacher.eval()

    def _register_hooks(self):
        """Register hooks to capture intermediate features."""
        def make_hook(storage: Dict, name: str):
            def hook(module, input, output):
                storage[name] = output
            return hook

        for name in self.feature_layers:
            t_module = self._get_module_by_name(self.teacher, name)
            s_module = self._get_module_by_name(self.student, name)

            if t_module is not None:
                t_module.register_forward_hook(make_hook(self.teacher_features, name))
            if s_module is not None:
                s_module.register_forward_hook(make_hook(self.student_features, name))

    def _get_module_by_name(
        self,
        model: nn.Module,
        name: str,
    ) -> Optional[nn.Module]:
        """Get module by dot-separated name."""
        parts = name.split('.')
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def compute_feature_loss(self) -> Tensor:
        """Compute MSE loss for all feature layers."""
        loss = 0.0

        for name in self.feature_layers:
            if name in self.teacher_features and name in self.student_features:
                t_feat = self.teacher_features[name]
                s_feat = self.student_features[name]

                if s_feat.shape != t_feat.shape:
                    if s_feat.dim() == 4:
                        s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])
                    elif s_feat.dim() == 2:
                        s_feat = F.linear(s_feat, t_feat.size(-1))

                loss += F.mse_loss(s_feat, t_feat)

        return loss

    def forward(
        self,
        inputs: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass through both models."""
        self.teacher_features.clear()
        self.student_features.clear()

        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)

        student_outputs = self.student(inputs)

        return {
            'student_logits': student_outputs,
            'teacher_logits': teacher_outputs,
            'labels': labels,
        }


class OutputBasedDistiller(BaseDistiller):
    """Output-based distillation - matches final predictions."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        use_hard_labels: bool = True,
    ):
        super().__init__(teacher, student, temperature, alpha)
        self.use_hard_labels = use_hard_labels
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_distillation_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute distillation loss from logits."""
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)

        loss_kd = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return loss_kd

    def compute_label_smoothing_loss(
        self,
        student_logits: Tensor,
        labels: Tensor,
        smoothing: float = 0.1,
    ) -> Tensor:
        """Compute cross-entropy with label smoothing."""
        n_classes = student_logits.size(-1)
        log_probs = F.log_softmax(student_logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(smoothing / (n_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)

        return F.kl_div(log_probs, true_dist, reduction='batchmean')

    def forward(
        self,
        inputs: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass."""
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)

        student_outputs = self.student(inputs)

        return {
            'student_logits': student_outputs,
            'teacher_logits': teacher_outputs,
            'labels': labels,
        }
