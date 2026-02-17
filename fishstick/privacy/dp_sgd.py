"""
Differentially Private Stochastic Gradient Descent (DP-SGD).

This module implements DP-SGD optimizer and training utilities
for training neural networks with differential privacy guarantees.

Example:
    >>> from fishstick.privacy import DPSGD, DPTrainer
    >>>
    >>> optimizer = DPSGD(model.parameters(), lr=0.01, noise_multiplier=1.0)
    >>> trainer = DPTrainer(model, optimizer, epsilon=8.0, delta=1e-5)
    >>> trainer.train(train_loader)
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from fishstick.privacy.accountant import RDPAccountant, PrivacyAccountant
from fishstick.privacy.clipping import StaticClipper, GradientClipper
from fishstick.privacy.noise import GaussianMechanism

Tensor = torch.Tensor
Module = nn.Module


@dataclass
class DPConfig:
    """Configuration for DP-SGD training.

    Attributes:
        epsilon: Target privacy budget (epsilon).
        delta: Target delta for (epsilon, delta)-DP.
        max_grad_norm: Maximum gradient norm for clipping.
        noise_multiplier: Ratio of noise std to max_grad_norm.
        minibatch_size: Size of micro-batches within each batch.
        learning_rate: Learning rate for optimizer.
    """

    epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    minibatch_size: int = 256
    learning_rate: float = 0.01


class DPSGD(optim.Optimizer):
    """Differentially Private SGD optimizer.

    Implements DP-SGD (also known as Privacy SGD) with gradient clipping
    and Gaussian noise addition.

    Reference:
        Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.

    Args:
        params: Model parameters to optimize.
        lr: Learning rate.
        max_grad_norm: Maximum gradient norm for clipping.
        noise_multiplier: Ratio of noise std to max_grad_norm.
        eps: Deprecated, use epsilon instead.
    """

    def __init__(
        self,
        params: Iterator[Tensor],
        lr: float = 0.01,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        eps: Optional[float] = None,
    ):
        if eps is not None:
            max_norm = eps
        else:
            max_norm = max_grad_norm

        defaults = dict(
            lr=lr,
            max_norm=max_norm,
            noise_multiplier=noise_multiplier,
        )
        super().__init__(params, defaults)

        self._grad_clipper = StaticClipper(max_norm=max_grad_norm)
        self._noise_mechanism = GaussianMechanism(epsilon=1.0)

    @property
    def noise_multiplier(self) -> float:
        """Get noise multiplier."""
        return self.defaults["noise_multiplier"]

    @property
    def max_grad_norm(self) -> float:
        """Get max gradient norm."""
        return self.defaults["max_norm"]

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure: Loss closure for gradient computation.

        Returns:
            Loss value if closure provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                clipped_grad, _ = self._grad_clipper.clip(grad)

                noise = (
                    torch.randn_like(clipped_grad)
                    * group["noise_multiplier"]
                    * group["max_norm"]
                )

                noisy_grad = clipped_grad + noise

                p.data.add_(other=noisy_grad, alpha=-group["lr"])

        return loss

    def clip_gradients(
        self,
        model: Module,
    ) -> float:
        """Clip all model gradients.

        Args:
            model: Model with gradients to clip.

        Returns:
            Total gradient norm before clipping.
        """
        total_norm = 0.0

        for p in model.parameters():
            if p.grad is not None:
                clipped, norm = self._grad_clipper.clip(p.grad)
                p.grad = clipped
                total_norm = max(total_norm, norm)

        return total_norm

    def add_noise_to_gradients(
        self,
        model: Module,
    ) -> None:
        """Add Gaussian noise to clipped gradients.

        Args:
            model: Model with gradients.
        """
        noise_scale = self.noise_multiplier * self.max_grad_norm

        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * noise_scale
                p.grad = p.grad + noise


class DPTrainer:
    """High-level trainer for DP-SGD.

    Provides a simplified interface for training models with differential privacy,
    handling gradient clipping, noise addition, and privacy accounting.

    Args:
        model: Model to train.
        optimizer: Optimizer (should be DPSGD).
        epsilon: Target privacy budget.
        delta: Target delta for (epsilon, delta)-DP.
        max_grad_norm: Maximum gradient norm for clipping.
        noise_multiplier: Noise multiplier for DP.
        accountant: Privacy accountant to use.

    Example:
        >>> optimizer = DPSGD(model.parameters(), lr=0.01)
        >>> trainer = DPTrainer(model, optimizer, epsilon=8.0)
        >>> history = trainer.train(train_loader, epochs=10)
    """

    def __init__(
        self,
        model: Module,
        optimizer: optim.Optimizer,
        epsilon: float = 8.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        accountant: Optional[PrivacyAccountant] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

        if accountant is None:
            self.accountant = RDPAccountant(epsilon, delta)
        else:
            self.accountant = accountant

        self._grad_clipper = StaticClipper(max_norm=max_grad_norm)
        self._step = 0

    def train_step(
        self,
        batch: Tuple[Tensor, Tensor],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Dict[str, float]:
        """Perform a single training step with DP.

        Args:
            batch: Tuple of (inputs, targets).
            loss_fn: Loss function.

        Returns:
            Dictionary with loss and privacy metrics.
        """
        inputs, targets = batch
        inputs = inputs.to(next(self.model.parameters()).device)
        targets = targets.to(next(self.model.parameters()).device)

        self.model.zero_grad()

        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)

        loss.backward()

        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                clipped, norm = self._grad_clipper.clip(p.grad)
                p.grad = clipped
                total_norm = max(total_norm, norm)

        noise_scale = self.noise_multiplier * self.max_grad_norm
        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * noise_scale
                p.grad = p.grad + noise

        self.optimizer.step()

        self._step += 1
        self.accountant.step(sample_rate=1.0, noise_multiplier=self.noise_multiplier)

        eps, _ = self.accountant.get_privacy_spent()

        return {
            "loss": loss.item(),
            "grad_norm": total_norm,
            "epsilon": eps,
        }

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> Dict[str, List[float]]:
        """Train the model with DP-SGD.

        Args:
            train_loader: Training data loader.
            epochs: Number of epochs.
            loss_fn: Loss function (default: cross entropy).
            val_loader: Validation data loader.
            device: Device to train on.

        Returns:
            Training history dictionary.
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.model.to(device)
        self.model.train()

        history = {
            "train_loss": [],
            "grad_norm": [],
            "epsilon": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            epoch_epsilon = 0.0
            n_batches = 0

            for batch in train_loader:
                metrics = self.train_step(batch, loss_fn)

                epoch_loss += metrics["loss"]
                epoch_grad_norm += metrics["grad_norm"]
                epoch_epsilon = metrics["epsilon"]
                n_batches += 1

            history["train_loss"].append(epoch_loss / n_batches)
            history["grad_norm"].append(epoch_grad_norm / n_batches)
            history["epsilon"].append(epoch_epsilon)

            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, loss_fn, device)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {history['train_loss'][-1]:.4f} - "
                f"Grad Norm: {history['grad_norm'][-1]:.4f} - "
                f"Epsilon: {history['epsilon'][-1]:.4f}"
            )

        return history

    def _evaluate(
        self,
        loader: DataLoader,
        loss_fn: Callable,
        device: str,
    ) -> Tuple[float, float]:
        """Evaluate the model.

        Args:
            loader: Data loader.
            loss_fn: Loss function.
            device: Device.

        Returns:
            Tuple of (loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += inputs.size(0)

        self.model.train()
        return total_loss / total, correct / total

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get total privacy budget spent.

        Returns:
            Tuple of (epsilon, delta) spent.
        """
        return self.accountant.get_privacy_spent()


class DPGradientDescent:
    """Functional DP-SGD implementation for custom training loops.

    Provides functional methods for clipping, noising, and aggregating
    gradients for use in custom training pipelines.

    Args:
        max_grad_norm: Maximum gradient norm for clipping.
        noise_multiplier: Ratio of noise std to max_grad_norm.

    Example:
        >>> clipper = DPGradientDescent(max_grad_norm=1.0, noise_multiplier=1.0)
        >>> loss.backward()
        >>> clipper.clip_and_noise(model.parameters())
        >>> optimizer.step()
    """

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
    ):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self._clipper = StaticClipper(max_norm=max_grad_norm)

    def clip_and_noise(
        self,
        parameters: Iterator[Tensor],
    ) -> float:
        """Clip gradients and add noise.

        Args:
            parameters: Model parameters with gradients.

        Returns:
            Total gradient norm before clipping.
        """
        total_norm = 0.0

        for p in parameters:
            if p.grad is not None:
                clipped, norm = self._grad_clipper.clip(p.grad)
                p.grad = clipped

                noise = (
                    torch.randn_like(p.grad)
                    * self.noise_multiplier
                    * self.max_grad_norm
                )
                p.grad = p.grad + noise

                total_norm = max(total_norm, norm)

        return total_norm

    def clip_only(
        self,
        parameters: Iterator[Tensor],
    ) -> float:
        """Clip gradients without adding noise (for testing).

        Args:
            parameters: Model parameters with gradients.

        Returns:
            Total gradient norm before clipping.
        """
        total_norm = 0.0

        for p in parameters:
            if p.grad is not None:
                clipped, norm = self._grad_clipper.clip(p.grad)
                p.grad = clipped
                total_norm = max(total_norm, norm)

        return total_norm


def create_dp_optimizer(
    model: Module,
    optimizer_type: str = "sgd",
    lr: float = 0.01,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    **kwargs,
) -> optim.Optimizer:
    """Factory function to create DP optimizer.

    Args:
        model: Model to optimize.
        optimizer_type: Type of optimizer ('sgd', 'adam', 'adamw').
        lr: Learning rate.
        max_grad_norm: Maximum gradient norm.
        noise_multiplier: Noise multiplier.
        **kwargs: Additional optimizer arguments.

    Returns:
        Configured optimizer.

    Example:
        >>> opt = create_dp_optimizer(model, 'sgd', lr=0.01)
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "sgd":
        return DPSGD(
            model.parameters(),
            lr=lr,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
        )
    elif optimizer_type in ("adam", "adamw"):
        return DPSGD(
            model.parameters(),
            lr=lr,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def compute_noise_batch(
    batch_size: int,
    sample_size: int,
    max_grad_norm: float,
    noise_multiplier: float,
    device: torch.device,
) -> Tensor:
    """Compute noise for a batch with proper scaling.

    Args:
        batch_size: Size of current batch.
        sample_size: Total dataset size.
        max_grad_norm: Clipping norm.
        noise_multiplier: Noise multiplier.
        device: Device to create noise on.

    Returns:
        Noise tensor.
    """
    q = batch_size / sample_size

    sigma = noise_multiplier * max_grad_norm

    noise = torch.randn(batch_size, device=device) * sigma

    return noise / q


def estimate_epsilon(
    num_steps: int,
    sample_rate: float,
    noise_multiplier: float,
    delta: float = 1e-5,
) -> float:
    """Estimate epsilon spent after training.

    Args:
        num_steps: Number of training steps.
        sample_rate: Sampling rate per step.
        noise_multiplier: Noise multiplier.
        delta: Target delta.

    Returns:
        Estimated epsilon spent.
    """
    import math

    sigma = noise_multiplier

    alpha = math.log(1 / delta) / (2 * num_steps * sample_rate * (sigma**2))

    epsilon = num_steps * sample_rate * (math.exp(1 / sigma) - 1) + math.sqrt(
        2 * num_steps * sample_rate * math.log(1 / delta)
    )

    return epsilon
