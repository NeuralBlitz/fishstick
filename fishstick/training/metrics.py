"""
Training Metrics

Comprehensive metrics for evaluation during training.
"""

from typing import Any, Dict, List, Optional
import torch
from torch import Tensor, nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix as sk_confusion_matrix,
)


class Metric:
    """Base metric class."""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update metric with new predictions."""
        raise NotImplementedError

    def compute(self) -> float:
        """Compute final metric value."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset metric state."""
        pass


class Accuracy(Metric):
    """Classification accuracy."""

    def __init__(self):
        super().__init__("accuracy")
        self.correct = 0
        self.total = 0

    def update(self, preds: Tensor, targets: Tensor) -> None:
        pred_labels = preds.argmax(dim=-1) if preds.dim() > 1 else preds
        self.correct += (pred_labels == targets).sum().item()
        self.total += targets.size(0)

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class Precision(Metric):
    """Precision score."""

    def __init__(self, average: str = "macro"):
        super().__init__("precision")
        self.average = average
        self.preds: List[int] = []
        self.targets: List[int] = []

    def update(self, preds: Tensor, targets: Tensor) -> None:
        pred_labels = preds.argmax(dim=-1).cpu().numpy()
        target_labels = targets.cpu().numpy()
        self.preds.extend(pred_labels)
        self.targets.extend(target_labels)

    def compute(self) -> float:
        if len(self.preds) == 0:
            return 0.0
        return precision_score(
            self.targets, self.preds, average=self.average, zero_division=0
        )

    def reset(self) -> None:
        self.preds = []
        self.targets = []


class Recall(Metric):
    """Recall score."""

    def __init__(self, average: str = "macro"):
        super().__init__("recall")
        self.average = average
        self.preds: List[int] = []
        self.targets: List[int] = []

    def update(self, preds: Tensor, targets: Tensor) -> None:
        pred_labels = preds.argmax(dim=-1).cpu().numpy()
        target_labels = targets.cpu().numpy()
        self.preds.extend(pred_labels)
        self.targets.extend(target_labels)

    def compute(self) -> float:
        if len(self.preds) == 0:
            return 0.0
        return recall_score(
            self.targets, self.preds, average=self.average, zero_division=0
        )

    def reset(self) -> None:
        self.preds = []
        self.targets = []


class F1Score(Metric):
    """F1 score."""

    def __init__(self, average: str = "macro"):
        super().__init__("f1")
        self.average = average
        self.preds: List[int] = []
        self.targets: List[int] = []

    def update(self, preds: Tensor, targets: Tensor) -> None:
        pred_labels = preds.argmax(dim=-1).cpu().numpy()
        target_labels = targets.cpu().numpy()
        self.preds.extend(pred_labels)
        self.targets.extend(target_labels)

    def compute(self) -> float:
        if len(self.preds) == 0:
            return 0.0
        return f1_score(self.targets, self.preds, average=self.average, zero_division=0)

    def reset(self) -> None:
        self.preds = []
        self.targets = []


class AUCROC(Metric):
    """AUC-ROC score."""

    def __init__(self, average: str = "macro"):
        super().__init__("aucroc")
        self.average = average
        self.probs: List[Tensor] = []
        self.targets: List[int] = []

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.dim() == 2:
            probs = torch.softmax(preds, dim=-1)
        else:
            probs = preds
        self.probs.append(probs.cpu())
        self.targets.extend(targets.cpu().numpy())

    def compute(self) -> float:
        if len(self.probs) == 0:
            return 0.0
        probs = torch.cat(self.probs, dim=0).numpy()
        targets = np.array(self.targets)

        try:
            if probs.shape[1] == 2:
                return roc_auc_score(targets, probs[:, 1], average=self.average)
            else:
                return roc_auc_score(
                    targets, probs, average=self.average, multi_class="ovr"
                )
        except ValueError:
            return 0.0

    def reset(self) -> None:
        self.probs = []
        self.targets = []


class ConfusionMatrix(Metric):
    """Confusion matrix."""

    def __init__(self, num_classes: int):
        super().__init__("confusion_matrix")
        self.num_classes = num_classes
        self.preds: List[int] = []
        self.targets: List[int] = []

    def update(self, preds: Tensor, targets: Tensor) -> None:
        pred_labels = preds.argmax(dim=-1).cpu().numpy()
        target_labels = targets.cpu().numpy()
        self.preds.extend(pred_labels)
        self.targets.extend(target_labels)

    def compute(self) -> np.ndarray:
        if len(self.preds) == 0:
            return np.zeros((self.num_classes, self.num_classes))
        return sk_confusion_matrix(
            self.targets, self.preds, labels=range(self.num_classes)
        )

    def reset(self) -> None:
        self.preds = []
        self.targets = []


class MetricTracker:
    """Track multiple metrics during training."""

    def __init__(self, metrics: Optional[List[Metric]] = None):
        self.metrics = metrics or []
        self.history: Dict[str, List[float]] = {}

    def add_metric(self, metric: Metric) -> "MetricTracker":
        self.metrics.append(metric)
        return self

    def update(self, preds: Tensor, targets: Tensor) -> None:
        for metric in self.metrics:
            metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute()
        return results

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def log(self, metrics: Dict[str, float]) -> None:
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)

    def get_history(self, name: str) -> List[float]:
        return self.history.get(name, [])


class LossMetric(Metric):
    """Track loss values."""

    def __init__(self, name: str = "loss"):
        super().__init__(name)
        self.losses: List[float] = []

    def update(self, preds: Tensor, targets: Tensor) -> None:
        pass

    def update_loss(self, loss: float) -> None:
        self.losses.append(loss)

    def compute(self) -> float:
        return sum(self.losses) / len(self.losses) if self.losses else 0.0

    def reset(self) -> None:
        self.losses = []


class TopKAccuracy(Metric):
    """Top-K accuracy."""

    def __init__(self, k: int = 5):
        super().__init__(f"top{k}_accuracy")
        self.k = k
        self.correct = 0
        self.total = 0

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.dim() == 1:
            return

        _, top_k_preds = preds.topk(self.k, dim=-1, largest=True, sorted=True)
        targets_expand = targets.view(-1, 1).expand_as(top_k_preds)
        self.correct += (top_k_preds == targets_expand).any(dim=-1).sum().item()
        self.total += targets.size(0)

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class MeanSquaredError(Metric):
    """MSE for regression."""

    def __init__(self):
        super().__init__("mse")
        self.squared_errors: List[float] = []

    def update(self, preds: Tensor, targets: Tensor) -> None:
        se = ((preds - targets) ** 2).mean().item()
        self.squared_errors.append(se)

    def compute(self) -> float:
        return (
            sum(self.squared_errors) / len(self.squared_errors)
            if self.squared_errors
            else 0.0
        )

    def reset(self) -> None:
        self.squared_errors = []


class MeanAbsoluteError(Metric):
    """MAE for regression."""

    def __init__(self):
        super().__init__("mae")
        self.absolute_errors: List[float] = []

    def update(self, preds: Tensor, targets: Tensor) -> None:
        ae = (preds - targets).abs().mean().item()
        self.absolute_errors.append(ae)

    def compute(self) -> float:
        return (
            sum(self.absolute_errors) / len(self.absolute_errors)
            if self.absolute_errors
            else 0.0
        )

    def reset(self) -> None:
        self.absolute_errors = []


class R2Score(Metric):
    """R² score for regression."""

    def __init__(self):
        super().__init__("r2")
        self.preds: List[Tensor] = []
        self.targets: List[Tensor] = []

    def update(self, preds: Tensor, targets: Tensor) -> None:
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> float:
        if len(self.preds) == 0:
            return 0.0
        preds = torch.cat(self.preds, dim=0).numpy().flatten()
        targets = torch.cat(self.targets, dim=0).numpy().flatten()

        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def reset(self) -> None:
        self.preds = []
        self.targets = []
