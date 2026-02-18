"""
Continual Learning Trainer and Utilities.

Training loop and evaluation protocols for continual learning.

Classes:
- ContinualTrainer: Main trainer class
- TaskSequence: Task sequence manager
- EvaluationProtocol: Evaluation utilities
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from collections import defaultdict


@dataclass
class TaskInfo:
    """Information about a task."""

    task_id: int
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    num_classes: int = 0


class ContinualTrainer:
    """
    Main Continual Learning Trainer.

    Provides training loop and evaluation for continual learning scenarios.

    Args:
        model: Neural network
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device

        self.current_task = 0
        self.accuracy_matrix: List[List[float]] = []

    def train_task(
        self,
        task: TaskInfo,
        epochs: int = 10,
        lr: float = 1e-3,
        replay_buffer: Optional[Any] = None,
    ) -> Dict[str, List[float]]:
        """
        Train on a single task.

        Args:
            task: Task to train on
            epochs: Number of epochs
            lr: Learning rate
            replay_buffer: Optional replay buffer

        Returns:
            Dictionary of training metrics
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        losses = []

        for epoch in range(epochs):
            epoch_losses = []

            for batch_idx, (x, y) in enumerate(task.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                logits = self.model(x)
                loss = F.cross_entropy(logits, y)

                if replay_buffer is not None:
                    pass

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            print(
                f"Task {task.task_id}, Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}"
            )

        self.current_task = task.task_id

        return {"loss": losses}

    def evaluate_task(
        self,
        task: TaskInfo,
    ) -> float:
        """
        Evaluate on a task.

        Args:
            task: Task to evaluate

        Returns:
            Accuracy
        """
        self.model.eval()

        correct = 0
        total = 0

        test_loader = task.test_loader or task.val_loader

        if test_loader is None:
            return 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                preds = logits.argmax(dim=-1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        accuracy = correct / total if total > 0 else 0.0

        return accuracy

    def evaluate_all_tasks(
        self,
        tasks: List[TaskInfo],
    ) -> np.ndarray:
        """
        Evaluate on all tasks.

        Args:
            tasks: List of tasks

        Returns:
            Accuracy matrix
        """
        num_tasks = len(tasks)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))

        for eval_task in range(num_tasks):
            for train_task in range(eval_task + 1):
                accuracy = self.evaluate_task(tasks[eval_task])
                accuracy_matrix[eval_task, train_task] = accuracy

        self.accuracy_matrix = accuracy_matrix.tolist()

        return accuracy_matrix


class TaskSequence:
    """
    Task Sequence Manager.

    Manages sequence of tasks for continual learning.

    Args:
        tasks: List of tasks
        shuffle: Whether to shuffle order
    """

    def __init__(
        self,
        tasks: Optional[List[TaskInfo]] = None,
        shuffle: bool = False,
    ):
        self.tasks = tasks or []
        self.shuffle = shuffle

        if shuffle:
            np.random.shuffle(self.tasks)

    def add_task(self, task: TaskInfo) -> None:
        """Add a task to the sequence."""
        self.tasks.append(task)

    def get_task(self, index: int) -> TaskInfo:
        """Get task by index."""
        return self.tasks[index]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)


class EvaluationProtocol:
    """
    Evaluation Protocol for Continual Learning.

    Provides standardized evaluation procedures.

    Args:
        model: Model to evaluate
        device: Device
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device

    def evaluate_per_task(
        self,
        tasks: List[TaskInfo],
    ) -> Dict[int, float]:
        """Evaluate each task separately."""
        results = {}

        for task in tasks:
            self.model.eval()

            correct = 0
            total = 0

            loader = task.test_loader or task.val_loader

            if loader is None:
                continue

            with torch.no_grad():
                for x, y in loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    logits = self.model(x)
                    preds = logits.argmax(dim=-1)

                    correct += (preds == y).sum().item()
                    total += y.size(0)

            accuracy = correct / total if total > 0 else 0.0
            results[task.task_id] = accuracy

        return results

    def compute_forgetting(
        self,
        results: Dict[int, float],
    ) -> float:
        """Compute forgetting from results."""
        if not results:
            return 0.0

        accuracies = list(results.values())

        if len(accuracies) < 2:
            return 0.0

        max_acc = max(accuracies[:-1])
        final_acc = accuracies[-1]

        return max_acc - final_acc

    def compute_accuracy(
        self,
        results: Dict[int, float],
    ) -> float:
        """Compute average accuracy."""
        if not results:
            return 0.0

        return np.mean(list(results.values()))


class ContinualLearningBenchmark:
    """
    Benchmark Suite for Continual Learning.

    Args:
        name: Benchmark name
    """

    def __init__(self, name: str = "default"):
        self.name = name

    def splitMNIST(self, num_tasks: int = 5) -> List[TaskInfo]:
        """Create MNIST task splits."""
        tasks = []

        classes_per_task = 10 // num_tasks

        for i in range(num_tasks):
            task_id = i
            classes = list(range(i * classes_per_task, (i + 1) * classes_per_task))

            task = TaskInfo(
                task_id=task_id,
                num_classes=classes_per_task,
            )

            tasks.append(task)

        return tasks

    def splitCIFAR10(self, num_tasks: int = 5) -> List[TaskInfo]:
        """Create CIFAR-10 task splits."""
        return self.splitMNIST(num_tasks)

    def splitCIFAR100(self, num_tasks: int = 10) -> List[TaskInfo]:
        """Create CIFAR-100 task splits."""
        tasks = []

        classes_per_task = 100 // num_tasks

        for i in range(num_tasks):
            task = TaskInfo(
                task_id=i,
                num_classes=classes_per_task,
            )

            tasks.append(task)

        return tasks
