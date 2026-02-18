"""
Federated Evaluation for fishstick

This module provides evaluation metrics and tools for federated learning:
- Federated accuracy computation
- Fairness metrics across clients
- Privacy-preserving evaluation
- Personalized evaluation
- Convergence diagnostics

References:
- Li et al. (2020): "Federated Learning on Non-IID Data"
- Kairouz et al. (2019): "Advances and Open Problems in Federated Learning"
- Mohri et al. (2019): "Fairness in Federated Learning"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of federated evaluation metrics."""

    ACCURACY = auto()
    LOSS = auto()
    PRECISION = auto()
    RECALL = auto()
    F1_SCORE = auto()
    AUC = auto()


@dataclass
class EvaluationConfig:
    """Configuration for federated evaluation."""

    metrics: List[MetricType] = field(
        default_factory=lambda: [MetricType.ACCURACY, MetricType.LOSS]
    )
    batch_size: int = 32
    num_workers: int = 0
    use_cuda: bool = True
    fairness_alpha: float = 0.5
    privacy_epsilon: float = 1.0
    confidence_level: float = 0.95


@dataclass
class ClientMetrics:
    """Metrics for a single client."""

    client_id: int
    accuracy: float = 0.0
    loss: float = 0.0
    num_samples: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    predictions: List[int] = field(default_factory=list)
    targets: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "num_samples": self.num_samples,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }


@dataclass
class FederatedMetrics:
    """Aggregated federated evaluation metrics."""

    round_number: int
    global_accuracy: float = 0.0
    global_loss: float = 0.0
    client_metrics: List[ClientMetrics] = field(default_factory=list)
    fairness_score: float = 0.0
    std_deviation: float = 0.0
    convergence_measure: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "global_accuracy": self.global_accuracy,
            "global_loss": self.global_loss,
            "fairness_score": self.fairness_score,
            "std_deviation": self.std_deviation,
            "convergence_measure": self.convergence_measure,
        }


class BaseEvaluator(ABC):
    """Base class for federated evaluation."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.history: List[FederatedMetrics] = []

    @abstractmethod
    def evaluate_client(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: Optional[torch.device] = None,
    ) -> ClientMetrics:
        """Evaluate model on a client's dataset."""
        pass

    def evaluate_federated(
        self,
        model: nn.Module,
        client_datasets: Dict[int, Dataset],
        round_number: int = 0,
        device: Optional[torch.device] = None,
    ) -> FederatedMetrics:
        """Evaluate model across all clients."""
        client_metrics_list = []

        for client_id, dataset in client_datasets.items():
            metrics = self.evaluate_client(model, dataset, device)
            metrics.client_id = client_id
            client_metrics_list.append(metrics)

        return self.aggregate_metrics(client_metrics_list, round_number)

    def aggregate_metrics(
        self,
        client_metrics: List[ClientMetrics],
        round_number: int,
    ) -> FederatedMetrics:
        """Aggregate client metrics into federated metrics."""
        if not client_metrics:
            return FederatedMetrics(round_number=round_number)

        total_samples = sum(m.num_samples for m in client_metrics)

        weighted_accuracy = sum(
            m.accuracy * m.num_samples for m in client_metrics
        ) / max(total_samples, 1)

        weighted_loss = sum(m.loss * m.num_samples for m in client_metrics) / max(
            total_samples, 1
        )

        accuracies = [m.accuracy for m in client_metrics]
        fairness_score = 1.0 - np.std(accuracies) if accuracies else 0.0

        std_deviation = np.std(accuracies) if accuracies else 0.0

        convergence = self._compute_convergence()

        return FederatedMetrics(
            round_number=round_number,
            global_accuracy=weighted_accuracy,
            global_loss=weighted_loss,
            client_metrics=client_metrics,
            fairness_score=fairness_score,
            std_deviation=std_deviation,
            convergence_measure=convergence,
        )

    def _compute_convergence(self) -> float:
        """Compute convergence measure from history."""
        if len(self.history) < 2:
            return 0.0

        recent_losses = [m.global_loss for m in self.history[-5:]]
        if len(recent_losses) < 2:
            return 0.0

        loss_diff = abs(recent_losses[-1] - recent_losses[-2])
        return float(loss_diff)


class StandardEvaluator(BaseEvaluator):
    """Standard federated evaluator."""

    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.criterion = nn.CrossEntropyLoss()

    def evaluate_client(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: Optional[torch.device] = None,
    ) -> ClientMetrics:
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and self.config.use_cuda else "cpu"
            )

        model.eval()
        model.to(device)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                all_predictions.extend(predicted.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        precision, recall, f1 = self._compute_precision_recall_f1(
            all_predictions, all_targets
        )

        return ClientMetrics(
            client_id=-1,
            accuracy=accuracy,
            loss=avg_loss,
            num_samples=total,
            precision=precision,
            recall=recall,
            f1_score=f1,
            predictions=all_predictions,
            targets=all_targets,
        )

    def _compute_precision_recall_f1(
        self,
        predictions: List[int],
        targets: List[int],
    ) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 score."""
        if not predictions:
            return 0.0, 0.0, 0.0

        classes = set(targets)
        total_precision = 0.0
        total_recall = 0.0

        for cls in classes:
            tp = sum(1 for p, t in zip(predictions, targets) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(predictions, targets) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(predictions, targets) if p != cls and t == cls)

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)

            total_precision += precision
            total_recall += recall

        avg_precision = total_precision / max(len(classes), 1)
        avg_recall = total_recall / max(len(classes), 1)
        f1 = 2 * (avg_precision * avg_recall) / max(avg_precision + avg_recall, 1e-10)

        return avg_precision, avg_recall, f1


class PrivacyPreservingEvaluator(BaseEvaluator):
    """Privacy-preserving federated evaluation using local differential privacy."""

    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.epsilon = config.privacy_epsilon
        self.delta = 1e-5

    def evaluate_client(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: Optional[torch.device] = None,
    ) -> ClientMetrics:
        base_evaluator = StandardEvaluator(self.config)
        metrics = base_evaluator.evaluate_client(model, dataset, device)

        noisy_metrics = self._add_noise(metrics)

        return noisy_metrics

    def _add_noise(self, metrics: ClientMetrics) -> ClientMetrics:
        """Add Laplace noise for differential privacy."""
        sensitivity = 1.0
        scale = sensitivity / self.epsilon

        metrics.accuracy += np.random.laplace(0, scale)
        metrics.accuracy = np.clip(metrics.accuracy, 0.0, 1.0)

        metrics.loss += np.random.laplace(0, scale)

        return metrics


class PersonalizedEvaluator:
    """Personalized federated evaluation.

    Evaluates personalized models for each client after local adaptation.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate_personalized(
        self,
        models: Dict[int, nn.Module],
        client_datasets: Dict[int, Dataset],
        device: Optional[torch.device] = None,
    ) -> Dict[int, ClientMetrics]:
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and self.config.use_cuda else "cpu"
            )

        evaluator = StandardEvaluator(self.config)
        results = {}

        for client_id, model in models.items():
            if client_id in client_datasets:
                metrics = evaluator.evaluate_client(
                    model, client_datasets[client_id], device
                )
                metrics.client_id = client_id
                results[client_id] = metrics

        return results

    def compute_personalization_benefit(
        self,
        global_metrics: FederatedMetrics,
        personalized_metrics: Dict[int, ClientMetrics],
    ) -> Dict[str, float]:
        """Compute benefit of personalization over global model."""
        if not personalized_metrics:
            return {"benefit": 0.0}

        personalized_accs = [m.accuracy for m in personalized_metrics.values()]
        avg_personalized_acc = np.mean(personalized_accs)

        benefit = avg_personalized_acc - global_metrics.global_accuracy

        return {
            "benefit": benefit,
            "avg_personalized_accuracy": avg_personalized_acc,
            "global_accuracy": global_metrics.global_accuracy,
            "std_personalized": np.std(personalized_accs),
        }


class ConvergenceDiagnostics:
    """Convergence diagnostics for federated learning."""

    def __init__(self):
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
        self.client_drift_history: List[float] = []

    def update(
        self,
        loss: float,
        accuracy: float,
        client_drift: float = 0.0,
    ) -> None:
        """Update convergence diagnostics."""
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        self.client_drift_history.append(client_drift)

    def is_converged(
        self,
        window_size: int = 5,
        loss_threshold: float = 1e-4,
    ) -> bool:
        """Check if training has converged."""
        if len(self.loss_history) < window_size:
            return False

        recent_losses = self.loss_history[-window_size:]
        loss_variance = np.var(recent_losses)

        return loss_variance < loss_threshold

    def compute_convergence_rate(
        self,
    ) -> float:
        """Compute convergence rate."""
        if len(self.loss_history) < 2:
            return 0.0

        losses = np.array(self.loss_history)
        log_losses = np.log(losses + 1e-10)

        if len(log_losses) < 2:
            return 0.0

        coeffs = np.polyfit(range(len(log_losses)), log_losses, 1)
        return float(coeffs[0])

    def detect_plateau(
        self,
        window_size: int = 10,
    ) -> bool:
        """Detect if training has hit a plateau."""
        if len(self.accuracy_history) < window_size:
            return False

        recent_accs = self.accuracy_history[-window_size:]
        acc_variance = np.var(recent_accs)

        return acc_variance < 1e-6

    def get_diagnostics_report(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics report."""
        return {
            "total_rounds": len(self.loss_history),
            "current_loss": self.loss_history[-1] if self.loss_history else None,
            "current_accuracy": self.accuracy_history[-1]
            if self.accuracy_history
            else None,
            "best_loss": min(self.loss_history) if self.loss_history else None,
            "best_accuracy": max(self.accuracy_history)
            if self.accuracy_history
            else None,
            "convergence_rate": self.compute_convergence_rate(),
            "is_plateau": self.detect_plateau(),
            "is_converged": self.is_converged(),
        }


def create_evaluator(
    config: EvaluationConfig,
    privacy_preserving: bool = False,
) -> BaseEvaluator:
    """Factory function to create evaluator.

    Args:
        config: Configuration for the evaluator
        privacy_preserving: Whether to use privacy-preserving evaluation

    Returns:
        Instance of the appropriate evaluator
    """
    if privacy_preserving:
        return PrivacyPreservingEvaluator(config)
    return StandardEvaluator(config)


__all__ = [
    "MetricType",
    "EvaluationConfig",
    "ClientMetrics",
    "FederatedMetrics",
    "BaseEvaluator",
    "StandardEvaluator",
    "PrivacyPreservingEvaluator",
    "PersonalizedEvaluator",
    "ConvergenceDiagnostics",
    "create_evaluator",
]
