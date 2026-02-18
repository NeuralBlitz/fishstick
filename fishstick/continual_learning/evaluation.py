"""
Evaluation Metrics for Continual Learning.

Comprehensive metrics for evaluating continual learning performance.

Classes:
- ContinualMetrics: Container for metrics
- AverageAccuracy: Average accuracy metric
- ForgettingMeasure: Forgetting metric
- BackwardTransfer: Backward transfer metric
- ForwardTransfer: Forward transfer metric
- compute_metrics: Compute all metrics
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor
import numpy as np


@dataclass
class ContinualMetrics:
    """Container for continual learning metrics."""

    accuracy_matrix: np.ndarray
    forgetting: np.ndarray
    forward_transfer: np.ndarray
    backward_transfer: np.ndarray
    average_accuracy: float
    average_forgetting: float


class AverageAccuracy:
    """
    Average Accuracy metric.

    Computes average accuracy across all seen tasks.

    Args:
        num_tasks: Number of tasks
    """

    def __init__(self, num_tasks: int = 10):
        self.num_tasks = num_tasks
        self.accuracy_history: List[float] = []

    def compute(self, accuracy_matrix: np.ndarray) -> float:
        """
        Compute average accuracy.

        Args:
            accuracy_matrix: Matrix of accuracies [num_tasks, num_tasks]

        Returns:
            Average accuracy
        """
        num_tasks = accuracy_matrix.shape[0]

        final_accuracies = np.diag(accuracy_matrix)

        avg_acc = final_accuracies.sum() / num_tasks

        self.accuracy_history.append(avg_acc)

        return avg_acc

    def compute_per_task(self, accuracy_matrix: np.ndarray) -> np.ndarray:
        """Compute accuracy per task."""
        return np.diag(accuracy_matrix)


class ForgettingMeasure:
    """
    Forgetting Measure.

    Measures how much performance drops on previous tasks
    after learning new tasks.

    Args:
        num_tasks: Number of tasks
    """

    def __init__(self, num_tasks: int = 10):
        self.num_tasks = num_tasks

    def compute(self, accuracy_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute forgetting.

        Args:
            accuracy_matrix: Accuracy matrix [num_tasks, num_tasks]

        Returns:
            Tuple of (average_forgetting, per_task_forgetting)
        """
        num_tasks = accuracy_matrix.shape[0]

        forgetting = np.zeros(num_tasks)

        for i in range(num_tasks):
            best_acc = (
                np.max(accuracy_matrix[i, : i + 1]) if i > 0 else accuracy_matrix[0, 0]
            )
            final_acc = accuracy_matrix[i, num_tasks - 1]

            forgetting[i] = best_acc - final_acc

        avg_forgetting = forgetting.sum() / num_tasks

        return avg_forgetting, forgetting

    def compute_intransigence(
        self,
        accuracy_matrix: np.ndarray,
        threshold: float = 0.1,
    ) -> np.ndarray:
        """Compute intransigence (failure to learn)."""
        num_tasks = accuracy_matrix.shape[0]

        intransigence = np.zeros(num_tasks)

        for i in range(num_tasks):
            best_acc = accuracy_matrix[i, : i + 1].max()

            if best_acc < threshold:
                intransigence[i] = 1.0

        return intransigence


class BackwardTransfer:
    """
    Backward Transfer (BWT).

    Measures the influence of learning new tasks on
    performance of previous tasks.
    """

    def compute(self, accuracy_matrix: np.ndarray) -> float:
        """
        Compute backward transfer.

        Args:
            accuracy_matrix: Accuracy matrix

        Returns:
            BWT score (negative = forgetting)
        """
        num_tasks = accuracy_matrix.shape[0]

        bwt = 0.0

        for i in range(num_tasks):
            for j in range(i):
                bwt += accuracy_matrix[i, j] - accuracy_matrix[i, i]

        if num_tasks > 1:
            bwt /= num_tasks * (num_tasks - 1) / 2

        return bwt


class ForwardTransfer:
    """
    Forward Transfer (FWT).

    Measures how much previous task knowledge helps
    initial performance on new tasks.
    """

    def compute(self, accuracy_matrix: np.ndarray) -> float:
        """
        Compute forward transfer.

        Args:
            accuracy_matrix: Accuracy matrix

        Returns:
            FWT score
        """
        num_tasks = accuracy_matrix.shape[0]

        fwt = 0.0

        for i in range(1, num_tasks):
            for j in range(i):
                if j == 0:
                    random_baseline = 1.0 / accuracy_matrix.shape[1]
                else:
                    random_baseline = 0.0

                fwt += accuracy_matrix[i, j] - random_baseline

        if num_tasks > 1:
            fwt /= num_tasks - 1

        return fwt


class LearningCurveArea:
    """
    Area Under Learning Curve (AULC).

    Integrates performance over training.
    """

    def compute(self, accuracy_matrix: np.ndarray) -> float:
        """Compute area under learning curve."""
        num_tasks = accuracy_matrix.shape[0]

        total_area = 0.0

        for i in range(num_tasks):
            task_acc = accuracy_matrix[i, : i + 1]
            area = np.trapz(task_acc)
            total_area += area

        return total_area / num_tasks


def compute_metrics(
    accuracy_matrix: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all continual learning metrics.

    Args:
        accuracy_matrix: Matrix of accuracies [num_tasks, num_tasks]
                         where entry [i,j] is accuracy on task i after task j

    Returns:
        Dictionary of metric names and values
    """
    avg_acc = AverageAccuracy()
    avg_accuracy = avg_acc.compute(accuracy_matrix)

    forgetting_measure = ForgettingMeasure()
    avg_forgetting, per_task_forgetting = forgetting_measure.compute(accuracy_matrix)

    bwt = BackwardTransfer().compute(accuracy_matrix)
    fwt = ForwardTransfer().compute(accuracy_matrix)

    aulc = LearningCurveArea().compute(accuracy_matrix)

    metrics = {
        "average_accuracy": avg_accuracy,
        "average_forgetting": avg_forgetting,
        "backward_transfer": bwt,
        "forward_transfer": fwt,
        "aulc": aulc,
    }

    return metrics


def compute_bwt_plus(accuracy_matrix: np.ndarray) -> float:
    """
    Compute BWT+ (positive backward transfer).

    Only considers positive transfer.
    """
    num_tasks = accuracy_matrix.shape[0]

    bwt_plus = 0.0

    for i in range(num_tasks):
        for j in range(i):
            diff = accuracy_matrix[i, j] - accuracy_matrix[i, i]
            if diff > 0:
                bwt_plus += diff

    if num_tasks > 1:
        bwt_plus /= num_tasks * (num_tasks - 1) / 2

    return bwt_plus


def compute_retention(accuracy_matrix: np.ndarray) -> float:
    """
    Compute retention rate.

    Ratio of final to best performance.
    """
    num_tasks = accuracy_matrix.shape[0]

    retention = 0.0

    for i in range(num_tasks):
        best = accuracy_matrix[i, : i + 1].max()
        final = accuracy_matrix[i, -1]

        if best > 0:
            retention += final / best

    return retention / num_tasks
