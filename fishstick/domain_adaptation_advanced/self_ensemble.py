import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field


@dataclass
class SelfEnsembleConfig:
    """Configuration for self-ensemble based domain adaptation."""

    num_models: int = 3
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    ensemble_weight: float = 1.0


class SelfEnsembleModel(nn.Module):
    """Self-ensemble model for domain adaptation."""

    def __init__(
        self,
        base_model: nn.Module,
        num_ensembles: int = 3,
        config: Optional[SelfEnsembleConfig] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_ensembles = num_ensembles
        self.config = config or SelfEnsembleConfig()

        self.ensemble_models = nn.ModuleList(
            [type(base_model)() for _ in range(self.num_ensembles)]
        )

        for i, model in enumerate(self.ensemble_models):
            if i == 0:
                model.load_state_dict(base_model.state_dict())
            else:
                self._init_model_from_base(model, base_model)

    def _init_model_from_base(self, model: nn.Module, base: nn.Module):
        """Initialize ensemble model with random perturbations."""
        model.load_state_dict(base.state_dict())
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01)

    def forward(
        self, x: torch.Tensor, return_all_logits: bool = False
    ) -> Dict[str, Any]:
        logits_list = []

        for model in self.ensemble_models:
            logits = model(x)
            logits_list.append(logits)

        stacked_logits = torch.stack(logits_list, dim=0)
        avg_logits = stacked_logits.mean(dim=0)

        if return_all_logits:
            return {
                "logits": avg_logits,
                "logits_list": logits_list,
                "stacked_logits": stacked_logits,
            }

        return {"logits": avg_logits}

    def compute_consistency_loss(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """Compute consistency loss across ensemble members."""
        if len(logits_list) < 2:
            return torch.tensor(0.0, device=logits_list[0].device)

        stacked = torch.stack(logits_list, dim=0)
        mean_logits = stacked.mean(dim=0, keepdim=True)

        variance = ((stacked - mean_logits) ** 2).mean()

        return variance

    def update_ensemble(
        self,
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
        threshold: float = 0.9,
    ) -> int:
        """
        Update ensemble models using high-confidence target predictions.

        Returns:
            Number of models updated
        """
        probs = F.softmax(target_logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)

        confident_mask = max_probs >= threshold

        return confident_mask.sum().item()


@dataclass
class PseudoLabelingConfig:
    """Configuration for pseudo-labeling based domain adaptation."""

    threshold: float = 0.9
    temperature: float = 1.0
    learning_rate: float = 1e-3
    pseudo_weight: float = 1.0
    confidence_warmup_epochs: int = 5


class PseudoLabelingModel(nn.Module):
    """Pseudo-labeling model for domain adaptation."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        config: Optional[PseudoLabelingConfig] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.config = config or PseudoLabelingConfig()
        self.current_epoch = 0

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        if return_features:
            return {"features": features, "logits": logits}
        return {"logits": logits}

    def generate_pseudo_labels(
        self, x: torch.Tensor, threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate pseudo labels for target domain.

        Returns:
            Tuple of (pseudo_labels, mask, confidence)
        """
        threshold = threshold or self.config.threshold

        features = self.feature_extractor(x)
        logits = self.classifier(features)

        probs = F.softmax(logits / self.config.temperature, dim=-1)
        confidences, predictions = probs.max(dim=-1)

        mask = (confidences >= threshold).float()

        return predictions, mask, confidences

    def compute_pseudo_label_loss(
        self,
        target_logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss for pseudo-labeled samples."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=target_logits.device)

        loss = F.cross_entropy(target_logits, pseudo_labels, reduction="none")

        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return masked_loss * self.config.pseudo_weight

    def update_threshold(
        self,
        epoch: int,
        initial_threshold: float = 0.9,
        final_threshold: float = 0.99,
        warmup_epochs: int = 10,
    ):
        """Update threshold during training."""
        self.current_epoch = epoch

        if epoch < warmup_epochs:
            progress = epoch / warmup_epochs
            self.config.threshold = (
                initial_threshold + (final_threshold - initial_threshold) * progress
            )
        else:
            self.config.threshold = final_threshold


@dataclass
class MeanTeacherConfig:
    """Configuration for Mean Teacher model."""

    ema_decay: float = 0.999
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    consistency_weight: float = 1.0
    consistency_rampup: int = 5000


class MeanTeacherModel(nn.Module):
    """Mean Teacher model for domain adaptation."""

    def __init__(
        self, student_model: nn.Module, config: Optional[MeanTeacherConfig] = None
    ):
        super().__init__()
        self.student = student_model
        self.teacher = type(student_model)()
        self.teacher.load_state_dict(student_model.state_dict())

        self.config = config or MeanTeacherConfig()
        self.global_step = 0

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(
        self, x: torch.Tensor, use_teacher: bool = False, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        if use_teacher:
            return self.teacher.forward(x, return_features)
        return self.student.forward(x, return_features)

    def update_teacher(self):
        """Update teacher using exponential moving average of student."""
        decay = self.config.ema_decay

        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.student.parameters(), self.teacher.parameters()
            ):
                teacher_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)

    def compute_consistency_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        use_softmax: bool = True,
    ) -> torch.Tensor:
        """Compute consistency loss between student and teacher predictions."""
        if use_softmax:
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            return F.mse_loss(student_probs, teacher_probs)

        return F.mse_loss(student_logits, teacher_logits)

    def get_consistency_weight(self) -> float:
        """Get consistency weight with ramp-up."""
        if self.global_step >= self.config.consistency_rampup:
            return self.config.consistency_weight

        progress = self.global_step / self.config.consistency_rampup
        return self.config.consistency_weight * progress

    def update(self, loss: torch.Tensor):
        """Update student and teacher."""
        self.student.zero_grad()
        loss.backward()
        self.student.step()
        self.update_teacher()
        self.global_step += 1


@dataclass
class CollaborativeSelfEnsembleConfig:
    """Configuration for collaborative self-ensemble model."""

    num_teachers: int = 3
    hidden_dim: int = 256
    learning_rate: float = 1e-3


class CollaborativeSelfEnsemble(nn.Module):
    """Collaborative self-ensemble with multiple teachers."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        num_classes: int,
        config: Optional[CollaborativeSelfEnsembleConfig] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.config = config or CollaborativeSelfEnsembleConfig()

        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, self.config.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(self.config.hidden_dim, num_classes),
                )
                for _ in range(self.config.num_teachers)
            ]
        )

        self.student_classifier = nn.Sequential(
            nn.Linear(512, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.config.hidden_dim, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_all_logits: bool = False
    ) -> Dict[str, Any]:
        features = self.feature_extractor(x)

        teacher_logits_list = [cls(features) for cls in self.classifiers]
        student_logits = self.student_classifier(features)

        all_logits = teacher_logits_list + [student_logits]
        stacked = torch.stack(all_logits, dim=0)
        avg_logits = stacked.mean(dim=0)

        if return_all_logits:
            return {
                "features": features,
                "logits": avg_logits,
                "teacher_logits": teacher_logits_list,
                "student_logits": student_logits,
            }

        return {"logits": avg_logits}

    def compute_teacher_agreement_loss(
        self, teacher_logits_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute agreement loss between teachers."""
        if len(teacher_logits_list) < 2:
            return torch.tensor(0.0, device=teacher_logits_list[0].device)

        stacked_probs = torch.stack(
            [F.softmax(logits, dim=-1) for logits in teacher_logits_list], dim=0
        )

        mean_probs = stacked_probs.mean(dim=0)

        kl_loss = F.kl_div(
            torch.log(mean_probs + 1e-8), mean_probs, reduction="batchmean"
        )

        return kl_loss

    def compute_student_teacher_consistency_loss(
        self, student_logits: torch.Tensor, teacher_logits_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute consistency loss between student and teachers."""
        teacher_probs = torch.stack(
            [F.softmax(logits, dim=-1) for logits in teacher_logits_list], dim=0
        ).mean(dim=0)

        student_probs = F.softmax(student_logits, dim=-1)

        return F.mse_loss(student_probs, teacher_probs)


def create_self_ensemble(
    base_model: nn.Module,
    num_ensembles: int = 3,
    config: Optional[SelfEnsembleConfig] = None,
) -> SelfEnsembleModel:
    """Factory function to create self-ensemble model."""
    return SelfEnsembleModel(
        base_model=base_model, num_ensembles=num_ensembles, config=config
    )


def create_pseudo_labeling(
    feature_extractor: nn.Module,
    classifier: nn.Module,
    config: Optional[PseudoLabelingConfig] = None,
) -> PseudoLabelingModel:
    """Factory function to create pseudo-labeling model."""
    return PseudoLabelingModel(
        feature_extractor=feature_extractor, classifier=classifier, config=config
    )


def create_mean_teacher(
    student_model: nn.Module, config: Optional[MeanTeacherConfig] = None
) -> MeanTeacherModel:
    """Factory function to create mean teacher model."""
    return MeanTeacherModel(student_model=student_model, config=config)


def create_collaborative_ensemble(
    feature_extractor: nn.Module,
    num_classes: int,
    num_teachers: int = 3,
    config: Optional[CollaborativeSelfEnsembleConfig] = None,
) -> CollaborativeSelfEnsemble:
    """Factory function to create collaborative self-ensemble model."""
    return CollaborativeSelfEnsemble(
        feature_extractor=feature_extractor, num_classes=num_classes, config=config
    )
