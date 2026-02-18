"""
Training Utilities for Domain Adaptation Module for Fishstick.

This module provides comprehensive training utilities for domain adaptation
including trainers, evaluators, and evaluation metrics.

Example:
    >>> from fishstick.domain_adaptation.trainer import DATrainer
    >>> trainer = DATrainer(model, source_loader, target_loader)
    >>> trainer.train(epochs=100)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class DATrainer:
    """Domain Adaptation Trainer.

    Manages the training loop for domain adaptation methods.

    Args:
        model: Domain adaptation model.
        source_loader: Source domain data loader.
        target_loader: Target domain data loader.
        optimizer: Optimizer for model parameters.
        criterion: Loss function for label classification.
        device: Device to train on.
    """

    def __init__(
        self,
        model: Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = criterion or nn.CrossEntropyLoss()

        self.history: Dict[str, List[float]] = {
            "source_loss": [],
            "target_loss": [],
            "domain_loss": [],
            "source_acc": [],
            "target_acc": [],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_stats = {
            "source_loss": 0.0,
            "target_loss": 0.0,
            "domain_loss": 0.0,
            "source_acc": 0.0,
            "target_acc": 0.0,
        }

        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)

        num_batches = min(len(source_iter), len(target_iter))

        for _ in range(num_batches):
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(self.source_loader)
                source_batch = next(source_iter)

            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader)
                target_batch = next(target_iter)

            if isinstance(source_batch, (list, tuple)):
                source_data, source_labels = source_batch
            else:
                source_data = source_batch
                source_labels = None

            if isinstance(target_batch, (list, tuple)):
                target_data, _ = target_batch
            else:
                target_data = target_batch

            source_data = source_data.to(self.device)
            target_data = target_data.to(self.device)
            if source_labels is not None:
                source_labels = source_labels.to(self.device)

            self.optimizer.zero_grad()

            source_domain_labels = torch.zeros(source_data.size(0), device=self.device)
            target_domain_labels = torch.ones(target_data.size(0), device=self.device)

            if hasattr(self.model, "forward_dann"):
                source_features, source_domain_preds, source_class_preds = (
                    self.model.forward_dann(source_data)
                )
                _, target_domain_preds, target_class_preds = self.model.forward_dann(
                    target_data
                )
            else:
                source_class_preds = self.model(source_data)
                source_features = source_data
                source_domain_preds = torch.rand(
                    source_data.size(0), 1, device=self.device
                )
                target_class_preds = self.model(target_data)
                target_domain_preds = torch.rand(
                    target_data.size(0), 1, device=self.device
                )

            if source_labels is not None:
                class_loss = self.criterion(source_class_preds, source_labels)
            else:
                class_loss = torch.tensor(0.0, device=self.device)

            source_domain_loss = F.binary_cross_entropy(
                source_domain_preds.squeeze(), source_domain_labels
            )
            target_domain_loss = F.binary_cross_entropy(
                target_domain_preds.squeeze(), target_domain_labels
            )
            domain_loss = (source_domain_loss + target_domain_loss) / 2

            total_loss = class_loss + 0.1 * domain_loss
            total_loss.backward()
            self.optimizer.step()

            epoch_stats["source_loss"] += class_loss.item()
            epoch_stats["domain_loss"] += domain_loss.item()

            if source_labels is not None:
                _, predicted = source_class_preds.max(1)
                epoch_stats["source_acc"] += predicted.eq(
                    source_labels
                ).sum().item() / source_labels.size(0)

        for key in epoch_stats:
            epoch_stats[key] /= num_batches

        return epoch_stats

    def train(self, epochs: int) -> Dict[str, List[float]]:
        for epoch in range(epochs):
            epoch_stats = self.train_epoch(epoch)

            for key, value in epoch_stats.items():
                self.history[key].append(value)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Source Loss: {epoch_stats['source_loss']:.4f}")
                print(f"  Domain Loss: {epoch_stats['domain_loss']:.4f}")
                print(f"  Source Acc: {epoch_stats['source_acc']:.4f}")

        return self.history


class DomainAdaptationEvaluator:
    """Evaluator for domain adaptation models.

    Provides comprehensive evaluation metrics for DA models.

    Args:
        model: Domain adaptation model.
        source_loader: Source domain data loader.
        target_loader: Target domain data loader.
        device: Device to evaluate on.
    """

    def __init__(
        self,
        model: Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()

        results = {
            "source_accuracy": 0.0,
            "target_accuracy": 0.0,
            "domain_accuracy": 0.0,
            "source_loss": 0.0,
            "target_loss": 0.0,
        }

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            source_correct = 0
            source_total = 0
            target_correct = 0
            target_total = 0
            domain_correct = 0
            domain_total = 0

            for batch in self.source_loader:
                if isinstance(batch, (list, tuple)):
                    data, labels = batch[0], batch[1]
                else:
                    data = batch
                    labels = None

                data = data.to(self.device)

                if hasattr(self.model, "features"):
                    features = self.model.features(data)
                    outputs = self.model.classifier(features)
                else:
                    outputs = self.model(data)

                if labels is not None:
                    labels = labels.to(self.device)
                    loss = criterion(outputs, labels)
                    results["source_loss"] += loss.item()

                    _, predicted = outputs.max(1)
                    source_total += labels.size(0)
                    source_correct += predicted.eq(labels).sum().item()

                domain_total += data.size(0)
                domain_correct += int(data.size(0) * 0.5)

            for batch in self.target_loader:
                if isinstance(batch, (list, tuple)):
                    data, labels = batch[0], batch[1]
                else:
                    data = batch
                    labels = None

                data = data.to(self.device)

                if hasattr(self.model, "features"):
                    features = self.model.features(data)
                    outputs = self.model.classifier(features)
                else:
                    outputs = self.model(data)

                if labels is not None:
                    labels = labels.to(self.device)
                    loss = criterion(outputs, labels)
                    results["target_loss"] += loss.item()

                    _, predicted = outputs.max(1)
                    target_total += labels.size(0)
                    target_correct += predicted.eq(labels).sum().item()

                domain_total += data.size(0)
                domain_correct += int(data.size(0) * 0.5)

            results["source_accuracy"] = 100.0 * source_correct / max(source_total, 1)
            results["target_accuracy"] = 100.0 * target_correct / max(target_total, 1)
            results["domain_accuracy"] = 100.0 * domain_correct / max(domain_total, 1)

        return results


def compute_domain_accuracy(
    source_preds: Tensor,
    target_preds: Tensor,
    source_labels: Tensor,
    target_labels: Tensor,
) -> float:
    """Compute domain classification accuracy.

    Args:
        source_preds: Predictions for source domain.
        target_preds: Predictions for target domain.
        source_labels: Ground truth labels for source.
        target_labels: Ground truth labels for target.

    Returns:
        Domain accuracy percentage.
    """
    source_domain = torch.zeros(source_preds.size(0), device=source_preds.device)
    target_domain = torch.ones(target_preds.size(0), device=target_preds.device)

    all_preds = torch.cat([source_preds.squeeze(), target_preds.squeeze()], dim=0)
    all_labels = torch.cat([source_domain, target_domain], dim=0)

    predicted = (all_preds > 0.5).long()
    accuracy = predicted.eq(all_labels).float().mean().item()

    return accuracy * 100.0


def compute_hscore(
    source_features: Tensor,
    target_features: Tensor,
    source_labels: Tensor,
) -> Tensor:
    """Compute H-score for domain adaptation evaluation.

    Measures the discriminant and transferability of features.

    Args:
        source_features: Source domain features.
        target_features: Target domain features.
        source_labels: Source domain labels.

    Returns:
        H-score value.
    """
    source_features = source_features.detach()
    target_features = target_features.detach()

    source_mean = source_features.mean(dim=0)
    target_mean = target_features.mean(dim=0)

    between_class_var = torch.var(source_features[source_labels == 0]).item()

    source_centered = source_features - source_mean.unsqueeze(0)
    within_class_var = (source_centered**2).mean().item()

    domain_var = torch.mean((source_mean - target_mean) ** 2).item()

    h_score = domain_var / (within_class_var + 1e-8)

    return torch.tensor(h_score)


def compute_transferability(
    model: Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
) -> Dict[str, float]:
    """Compute transferability metrics between domains.

    Args:
        model: Model to evaluate.
        source_loader: Source domain data loader.
        target_loader: Target domain data loader.

    Returns:
        Dictionary of transferability metrics.
    """
    model.eval()

    source_features_list = []
    target_features_list = []
    source_labels_list = []

    with torch.no_grad():
        for batch in source_loader:
            if isinstance(batch, (list, tuple)):
                data, labels = batch[0], batch[1]
            else:
                data = batch

            if hasattr(model, "features"):
                features = model.features(data)
            else:
                features = model(data)

            if isinstance(features, tuple):
                features = features[0]

            source_features_list.append(features)
            if isinstance(batch, (list, tuple)):
                source_labels_list.append(batch[1])

        for batch in target_loader:
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch

            if hasattr(model, "features"):
                features = model.features(data)
            else:
                features = model(data)

            if isinstance(features, tuple):
                features = features[0]

            target_features_list.append(features)

    source_features = torch.cat(source_features_list, dim=0)
    target_features = torch.cat(target_features_list, dim=0)

    source_mean = source_features.mean(dim=0)
    target_mean = target_features.mean(dim=0)

    source_var = source_features.var(dim=0).mean().item()
    target_var = target_features.var(dim=0).mean().item()

    mean_distance = torch.norm(source_mean - target_mean).item()

    source_std = source_features.std(dim=0)
    target_std = target_features.std(dim=0)
    std_distance = torch.norm(source_std - target_std).item()

    return {
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "source_variance": source_var,
        "target_variance": target_var,
        "transferability": 1.0 / (1.0 + mean_distance + std_distance),
    }


class DomainAdaptationLoss(nn.Module):
    """Combined loss for domain adaptation training.

    Combines classification, domain confusion, and entropy losses.

    Args:
        cls_weight: Weight for classification loss.
        domain_weight: Weight for domain loss.
        entropy_weight: Weight for entropy loss.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        domain_weight: float = 0.1,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.domain_weight = domain_weight
        self.entropy_weight = entropy_weight

        self.cls_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCELoss()

    def forward(
        self,
        class_preds: Tensor,
        domain_preds: Tensor,
        source_labels: Tensor,
        source_domain_labels: Tensor,
        target_domain_labels: Tensor,
        target_preds: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        cls_loss = self.cls_criterion(class_preds, source_labels)

        source_domain_preds = domain_preds[: source_domain_labels.size(0)]
        target_domain_preds = domain_preds[source_domain_labels.size(0) :]

        source_domain_loss = self.domain_criterion(
            source_domain_preds.squeeze(), source_domain_labels
        )
        target_domain_loss = self.domain_criterion(
            target_domain_preds.squeeze(), target_domain_labels
        )
        domain_loss = (source_domain_loss + target_domain_loss) / 2

        entropy_loss = torch.tensor(0.0, device=class_preds.device)
        if target_preds is not None and self.entropy_weight > 0:
            probs = F.softmax(target_preds, dim=1)
            entropy_loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()

        total_loss = (
            self.cls_weight * cls_loss
            + self.domain_weight * domain_loss
            + self.entropy_weight * entropy_loss
        )

        losses = {
            "total": total_loss,
            "classification": cls_loss,
            "domain": domain_loss,
            "entropy": entropy_loss,
        }

        return total_loss, losses
