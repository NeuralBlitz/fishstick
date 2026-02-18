"""
Transfer Learning Utilities Module for Fishstick.

This module provides comprehensive transfer learning utilities including
fine-tuning strategies, feature extraction, progressive fine-tuning,
learning without forgetting (LwF), and transferability estimation.

Example:
    >>> from fishstick.domain_adaptation.transfer_utils import TransferLearner
    >>> learner = TransferLearner(source_model=model, num_classes=10)
    >>> learner.fine_tune(target_data, epochs=50)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

T = TypeVar("T")


class FeatureExtractor(Module):
    """Feature extractor wrapper for transfer learning.

    Extracts features from a pre-trained model by removing the final
    classification layer.

    Args:
        model: Pre-trained model.
        layer_name: Name of the layer to extract features from.
    """

    def __init__(
        self,
        model: Module,
        layer_name: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.layer_name = layer_name

        if layer_name is not None:
            self._register_hook()

    def _register_hook(self) -> None:
        def hook(module: Module, input: Tensor, output: Tensor) -> None:
            self.features = output

        if self.layer_name:
            for name, module in self.model.named_modules():
                if name == self.layer_name:
                    module.register_forward_hook(hook)
                    break

    def forward(self, x: Tensor) -> Tensor:
        if self.layer_name is not None:
            _ = self.model(x)
            return self.features
        else:
            features = self.model(x)
            if isinstance(features, tuple):
                return features[0]
            return features


class FineTuner(Module):
    """Fine-tuning utility with configurable freezing strategies.

    Supports various fine-tuning strategies including:
    - Full fine-tuning
    - Feature extraction (freeze backbone)
    - Partial fine-tuning (freeze early layers)

    Args:
        model: Pre-trained model.
        strategy: Fine-tuning strategy ('full', 'freeze', 'partial').
        freeze_layers: Number of layers to freeze (for 'partial' strategy).
    """

    def __init__(
        self,
        model: Module,
        strategy: str = "partial",
        freeze_layers: int = 2,
    ):
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.freeze_layers = freeze_layers

        self._apply_strategy()

    def _apply_strategy(self) -> None:
        if self.strategy == "freeze":
            self._freeze_backbone()
        elif self.strategy == "partial":
            self._freeze_partial()
        elif self.strategy == "full":
            pass
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _freeze_backbone(self) -> None:
        for name, param in self.model.named_parameters():
            if "classifier" not in name and "fc" not in name and "head" not in name:
                param.requires_grad = False

    def _freeze_partial(self) -> None:
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append(name)

        layers_to_freeze = layers[: self.freeze_layers]
        for name, param in self.model.named_parameters():
            for layer_name in layers_to_freeze:
                if layer_name in name:
                    param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class TransferLearner(Module):
    """Complete transfer learning framework.

    Manages the entire transfer learning pipeline from pre-trained model
    to fine-tuned model for new task.

    Args:
        model: Pre-trained model.
        num_classes: Number of classes in target task.
        freeze_encoder: Whether to freeze encoder layers.
        dropout: Dropout rate for new classifier.
    """

    def __init__(
        self,
        model: Module,
        num_classes: int,
        freeze_encoder: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

        self.feature_extractor = model
        if hasattr(model, "features"):
            self.feature_extractor = model.features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._get_feature_dim(), num_classes),
        )

        if freeze_encoder:
            self._freeze_encoder()

    def _get_feature_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            if hasattr(self.feature_extractor, "forward_features"):
                features = self.feature_extractor.forward_features(dummy)
            else:
                features = self.feature_extractor(dummy)

            if isinstance(features, tuple):
                features = features[0]
            return features.view(features.size(0), -1).size(1)

    def _freeze_encoder(self) -> None:
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        features = self.feature_extractor(x)
        if isinstance(features, tuple):
            features = features[0]
        features = features.view(features.size(0), -1)
        return self.classifier(features)

    def fine_tune(
        self,
        target_data: DataLoader,
        optimizer: Optional[Optimizer] = None,
        epochs: int = 10,
        criterion: Optional[Module] = None,
    ) -> Dict[str, List[float]]:
        if optimizer is None:
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in target_data:
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0], batch[1]
                else:
                    inputs = batch

                optimizer.zero_grad()
                outputs = self.forward(inputs)

                if isinstance(batch, (list, tuple)):
                    labels = batch[1]
                    loss = criterion(outputs, labels)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                else:
                    loss = torch.tensor(0.0)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(target_data)
            accuracy = 100.0 * correct / total if total > 0 else 0.0

            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)

        return history


class ProgressiveFineTuning(Module):
    """Progressive fine-tuning with gradually increasing learning rates.

    Starts with low learning rate for early layers and gradually increases
    for later layers, preventing catastrophic forgetting.

    Args:
        model: Pre-trained model.
        num_classes: Number of target classes.
        lr_multipliers: Learning rate multipliers per layer group.
    """

    def __init__(
        self,
        model: Module,
        num_classes: int,
        lr_multipliers: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

        if lr_multipliers is None:
            lr_multipliers = {
                "layer1": 0.1,
                "layer2": 0.2,
                "layer3": 0.3,
                "layer4": 1.0,
                "classifier": 2.0,
            }
        self.lr_multipliers = lr_multipliers

        self._setup_classifier()

    def _setup_classifier(self) -> None:
        feature_dim = self._get_feature_dim()
        self.classifier = nn.Linear(feature_dim, self.num_classes)

    def _get_feature_dim(self) -> int:
        return 512

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        param_groups = []
        for name, param in self.model.named_parameters():
            multiplier = 1.0
            for key, val in self.lr_multipliers.items():
                if key in name:
                    multiplier = val
                    break
            param_groups.append(
                {
                    "params": [param],
                    "lr": base_lr * multiplier,
                }
            )

        param_groups.append(
            {
                "params": self.classifier.parameters(),
                "lr": base_lr * self.lr_multipliers.get("classifier", 1.0),
            }
        )

        return param_groups


class LwF(Module):
    """Learning without Forgetting (LwF) for transfer learning.

    Preserves knowledge from original task while learning new task by
    using soft labels from pre-trained model as distillation targets.

    Reference:
        Li & Hoiem "Learning without Forgetting" ECCV 2016

    Args:
        model: Pre-trained model.
        num_classes_new: Number of classes in new task.
        temperature: Temperature for softening predictions.
        alpha: Balance between old and new task losses.
    """

    def __init__(
        self,
        model: Module,
        num_classes_new: int,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.num_classes_new = num_classes_new
        self.temperature = temperature
        self.alpha = alpha

        self.old_model = None
        self._freeze_old_model()

    def _freeze_old_model(self) -> None:
        self.old_model = type(self.model)(self.model.num_classes)
        self.old_model.load_state_dict(self.model.state_dict())
        for param in self.old_model.parameters():
            param.requires_grad = False
        self.old_model.eval()

    def distill_loss(
        self,
        new_outputs: Tensor,
        old_outputs: Tensor,
    ) -> Tensor:
        new_log_probs = F.log_softmax(new_outputs / self.temperature, dim=1)
        old_probs = F.softmax(old_outputs / self.temperature, dim=1)

        distill = -(new_log_probs * old_probs).sum(dim=1).mean()
        return distill * (self.temperature**2)

    def forward(
        self,
        x: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        new_outputs = self.model(x)

        if self.training and self.old_model is not None:
            with torch.no_grad():
                old_outputs = self.old_model(x)
        else:
            old_outputs = torch.zeros_like(new_outputs)

        return new_outputs, old_outputs


class TransferabilityEstimator:
    """Estimates transferability between source and target domains.

    Provides metrics to estimate how well a pre-trained model will
    transfer to a target domain.

    Example:
        >>> estimator = TransferabilityEstimator()
        >>> score = estimator.estimate(source_model, target_data)
        >>> print(f"Transferability: {score:.4f}")
    """

    def __init__(self):
        self.metrics: Dict[str, float] = {}

    def compute_feature_distribution(
        self,
        model: Module,
        data: DataLoader,
    ) -> Tuple[Tensor, Tensor]:
        model.eval()
        features_list = []
        labels_list = []

        with torch.no_grad():
            for batch in data:
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0], batch[1]
                else:
                    inputs = batch
                    labels = None

                if hasattr(model, "forward_features"):
                    features = model.forward_features(inputs)
                else:
                    features = model(inputs)

                if isinstance(features, tuple):
                    features = features[0]

                features_list.append(features)
                if labels is not None:
                    labels_list.append(labels)

        features = torch.cat(features_list, dim=0)
        mean = features.mean(dim=0)
        std = features.std(dim=0)

        if labels_list:
            labels = torch.cat(labels_list, dim=0)
            return mean, std, labels
        return mean, std, None

    def estimate(
        self,
        model: Module,
        target_data: DataLoader,
    ) -> float:
        mean, std, labels = self.compute_feature_distribution(model, target_data)

        variance = torch.mean(std**2).item()
        feature_spread = torch.max(std).item() - torch.min(std).item()

        transferability_score = 1.0 / (1.0 + variance + feature_spread)

        self.metrics = {
            "variance": variance,
            "feature_spread": feature_spread,
            "transferability": transferability_score,
        }

        return transferability_score

    def compute_nic(
        self,
        source_data: DataLoader,
        target_data: DataLoader,
    ) -> float:
        source_mean, source_std, _ = self.compute_feature_distribution(
            type("DummyModel", (), {"forward_features": lambda self, x: x})(),
            source_data,
        )

        target_mean, target_std, _ = self.compute_feature_distribution(
            type("DummyModel", (), {"forward_features": lambda self, x: x})(),
            target_data,
        )

        mean_dist = torch.norm(source_mean - target_mean).item()
        return mean_dist


class DomainInspector(Module):
    """Inspects and analyzes domain characteristics for transfer learning.

    Args:
        model: Pre-trained model to inspect.
    """

    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def compute_layer_statistics(
        self,
        data: DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        self.model.eval()
        stats = {}

        def hook_fn(name: str) -> Callable:
            def hook(module: Module, input: Tensor, output: Tensor):
                if isinstance(output, Tensor):
                    stats[name] = {
                        "mean": output.mean().item(),
                        "std": output.std().item(),
                        "min": output.min().item(),
                        "max": output.max().item(),
                    }

            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            for batch in data:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                _ = self.model(inputs)
                break

        for hook in hooks:
            hook.remove()

        return stats
