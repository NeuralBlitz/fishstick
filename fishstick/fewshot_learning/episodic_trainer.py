"""
Episodic Trainer for few-shot meta-learning.

Provides training loops for meta-learning algorithms with episodic training.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from .types import FewShotTask, EvaluationResult, TrainingState, EpisodeConfig


class EpisodicTrainer:
    """Trainer for few-shot meta-learning with episodic training.

    Args:
        model: Meta-learning model
        config: Training configuration
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        self.state = TrainingState()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_episode(
        self,
        task: FewShotTask,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Train on a single episode.

        Args:
            task: Few-shot task
            optimizer: Optimizer

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        task.support_x = task.support_x.to(self.device)
        task.support_y = task.support_y.to(self.device)
        task.query_x = task.query_x.to(self.device)
        task.query_y = task.query_y.to(self.device)

        if hasattr(self.model, "meta_train_step"):
            loss, metrics = self.model.meta_train_step([task], optimizer)
        else:
            result = self.model.adapt(task)
            optimizer.zero_grad()
            result.query_loss.backward()
            optimizer.step()
            loss = result.query_loss
            metrics = {"loss": loss.item()}

        return metrics

    def train_epoch(
        self,
        episode_generator: Any,
        optimizer: torch.optim.Optimizer,
        num_episodes: int = 100,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            episode_generator: Generator of episodes
            optimizer: Optimizer
            num_episodes: Number of episodes

        Returns:
            Average metrics for the epoch
        """
        total_metrics = {"loss": 0.0}

        for _ in range(num_episodes):
            task = episode_generator.sample_episode()
            metrics = self.train_episode(task, optimizer)

            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v

        for k in total_metrics:
            total_metrics[k] /= num_episodes

        self.state.epoch += 1

        return total_metrics

    def evaluate(
        self,
        episode_generator: Any,
        num_episodes: int = 1000,
    ) -> EvaluationResult:
        """Evaluate the model on few-shot tasks.

        Args:
            episode_generator: Generator of evaluation episodes
            num_episodes: Number of evaluation episodes

        Returns:
            EvaluationResult with accuracy and confidence interval
        """
        self.model.eval()

        accuracies = []

        with torch.no_grad():
            for _ in range(num_episodes):
                task = episode_generator.sample_episode()
                task.support_x = task.support_x.to(self.device)
                task.support_y = task.support_y.to(self.device)
                task.query_x = task.query_x.to(self.device)
                task.query_y = task.query_y.to(self.device)

                if hasattr(self.model, "forward"):
                    if hasattr(self.model, "adapt"):
                        result = self.model.adapt(task)
                        logits = result.query_logits
                    else:
                        logits, _ = self.model(
                            task.support_x,
                            task.support_y,
                            task.query_x,
                            task.n_way,
                            task.n_shot,
                        )
                else:
                    logits = self.model(task)

                preds = torch.argmax(logits, dim=1)
                acc = (preds == task.query_y).float().mean().item()
                accuracies.append(acc)

        accuracies = torch.tensor(accuracies)

        mean_acc = accuracies.mean().item()
        std_acc = accuracies.std().item()

        ci_low = mean_acc - 1.96 * std_acc / (num_episodes**0.5)
        ci_high = mean_acc + 1.96 * std_acc / (num_episodes**0.5)

        return EvaluationResult(
            accuracy=mean_acc,
            confidence_interval=(ci_low, ci_high),
            std=std_acc,
            num_samples=num_episodes,
        )

    def train(
        self,
        train_generator: Any,
        val_generator: Any,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 100,
        num_train_episodes: int = 100,
        num_val_episodes: int = 500,
    ) -> TrainingState:
        """Full training loop.

        Args:
            train_generator: Training episode generator
            val_generator: Validation episode generator
            optimizer: Optimizer
            num_epochs: Number of epochs
            num_train_episodes: Episodes per training epoch
            num_val_episodes: Episodes for validation

        Returns:
            TrainingState with training history
        """
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(
                train_generator,
                optimizer,
                num_train_episodes,
            )

            self.state.train_loss.append(train_metrics.get("loss", 0.0))

            if epoch % 5 == 0:
                val_result = self.evaluate(val_generator, num_val_episodes)
                self.state.val_accuracy.append(val_result.accuracy)

                if val_result.accuracy > self.state.best_val_accuracy:
                    self.state.best_val_accuracy = val_result.accuracy

                print(
                    f"Epoch {epoch}: Train Loss: {train_metrics.get('loss', 0):.4f}, "
                    f"Val Acc: {val_result.accuracy:.2%}"
                )

        return self.state


class FewShotEvaluator:
    """Evaluator for few-shot learning models.

    Provides comprehensive evaluation with confidence intervals.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def evaluate_single_task(self, task: FewShotTask) -> Tuple[float, Tensor]:
        """Evaluate on a single task.

        Args:
            task: Few-shot task

        Returns:
            Tuple of (accuracy, predictions)
        """
        self.model.eval()

        task.support_x = task.support_x.to(self.device)
        task.support_y = task.support_y.to(self.device)
        task.query_x = task.query_x.to(self.device)
        task.query_y = task.query_y.to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "adapt"):
                result = self.model.adapt(task)
                logits = result.query_logits
            else:
                logits, _ = self.model(
                    task.support_x,
                    task.support_y,
                    task.query_x,
                    task.n_way,
                    task.n_shot,
                )

            preds = torch.argmax(logits, dim=1)
            acc = (preds == task.query_y).float().mean().item()

        return acc, preds

    def evaluate_tasks(
        self,
        tasks: List[FewShotTask],
        compute_ci: bool = True,
        ci_level: float = 0.95,
    ) -> EvaluationResult:
        """Evaluate on multiple tasks.

        Args:
            tasks: List of few-shot tasks
            compute_ci: Whether to compute confidence interval
            ci_level: Confidence level

        Returns:
            EvaluationResult
        """
        accuracies = []

        for task in tasks:
            acc, _ = self.evaluate_single_task(task)
            accuracies.append(acc)

        accuracies = torch.tensor(accuracies)

        mean_acc = accuracies.mean().item()
        std_acc = accuracies.std().item()

        ci_low, ci_high = mean_acc, mean_acc

        if compute_ci:
            from scipy import stats

            z = stats.norm.ppf((1 + ci_level) / 2)
            margin = z * std_acc / (len(accuracies) ** 0.5)
            ci_low = mean_acc - margin
            ci_high = mean_acc + margin

        return EvaluationResult(
            accuracy=mean_acc,
            confidence_interval=(ci_low, ci_high),
            std=std_acc,
            num_samples=len(tasks),
        )


def compute_confidence_interval(
    accuracies: List[float],
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute confidence interval for accuracies.

    Args:
        accuracies: List of accuracies
        ci_level: Confidence level

    Returns:
        Tuple of (mean, ci_low, ci_high)
    """
    import numpy as np

    accuracies = np.array(accuracies)
    mean = accuracies.mean()
    std = accuracies.std()

    from scipy import stats

    z = stats.norm.ppf((1 + ci_level) / 2)
    margin = z * std / (len(accuracies) ** 0.5)

    return mean, mean - margin, mean + margin
