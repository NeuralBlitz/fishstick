"""
Privacy Engine - High-Level API for Differential Privacy.

This module provides a unified, high-level API for training models
with differential privacy guarantees.

Example:
    >>> from fishstick.privacy import PrivacyEngine
    >>>
    >>> engine = PrivacyEngine(model, epsilon=8.0, delta=1e-5)
    >>> engine.clip_grad_norm = 1.0
    >>> engine.noise_multiplier = 1.0
    >>>
    >>> for epoch in range(10):
    ...     engine.step(batch)
"""

from __future__ import annotations

import copy
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

from fishstick.privacy.accountant import (
    RDPAccountant,
    PrivacyAccountant,
    PrivacyBudgetTracker,
)
from fishstick.privacy.aggregation import (
    PrivateAggregator,
    DPFederatedAggregator,
    NoisyAggregator,
)
from fishstick.privacy.amplification import SubsampleAmplifier
from fishstick.privacy.clipping import StaticClipper, GradientClipper
from fishstick.privacy.dp_sgd import DPSGD, estimate_epsilon
from fishstick.privacy.noise import GaussianMechanism, NoiseMechanism
from fishstick.privacy.sampling import PoissonSampler

Tensor = torch.Tensor
Module = nn.Module


@dataclass
class PrivacyEngineConfig:
    """Configuration for PrivacyEngine.

    Attributes:
        epsilon: Target privacy budget.
        delta: Target delta for (epsilon, delta)-DP.
        max_grad_norm: Maximum gradient norm for clipping.
        noise_multiplier: Ratio of noise std to max_grad_norm.
        sample_rate: Sampling rate for mini-batch.
        accounting_mode: Privacy accounting mode ('rdp', 'basic').
        clip_per_layer: Whether to clip per layer instead of globally.
    """

    epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    sample_rate: float = 0.01
    accounting_mode: str = "rdp"
    clip_per_layer: bool = False
    secure_aggregation: bool = False


class PrivacyEngine:
    """High-level privacy engine for DP training.

    Provides a unified interface for training models with differential privacy,
    handling gradient clipping, noise addition, and privacy accounting.

    Args:
        model: Model to train.
        optimizer: Optimizer to use (will be converted to DPSGD if not already).
        config: Privacy engine configuration.
        accountant: Custom privacy accountant (optional).

    Example:
        >>> engine = PrivacyEngine(model, epsilon=8.0, delta=1e-5)
        >>> engine.noise_multiplier = 1.0
        >>> engine.max_grad_norm = 1.0
        >>>
        >>> for epoch in range(10):
        ...     for batch in train_loader:
        ...         loss = engine.step(batch, targets, loss_fn)
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optional[optim.Optimizer] = None,
        config: Optional[PrivacyEngineConfig] = None,
        accountant: Optional[PrivacyAccountant] = None,
    ):
        self.model = model
        self.config = config or PrivacyEngineConfig()

        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01)

        self.optimizer = optimizer

        self._setup_accountant(accountant)
        self._setup_clipper()
        self._setup_noise_mechanism()

        self._step_count = 0

    def _setup_accountant(self, accountant: Optional[PrivacyAccountant]) -> None:
        """Setup privacy accountant."""
        if accountant is not None:
            self.accountant = accountant
        else:
            if self.config.accounting_mode == "rdp":
                self.accountant = RDPAccountant(
                    self.config.epsilon,
                    self.config.delta,
                )
            else:
                from fishstick.privacy.accountant import BasicAccountant

                self.accountant = BasicAccountant(
                    self.config.epsilon,
                    self.config.delta,
                )

    def _setup_clipper(self) -> None:
        """Setup gradient clipper."""
        if self.config.clip_per_layer:
            from fishstick.privacy.clipping import PerLayerClipper

            self._clipper = PerLayerClipper(self.config.max_grad_norm)
        else:
            self._clipper = StaticClipper(self.config.max_grad_norm)

    def _setup_noise_mechanism(self) -> None:
        """Setup noise mechanism."""
        self._noise_mechanism = GaussianMechanism(
            epsilon=self.config.epsilon,
            delta=self.config.delta,
        )

    def step(
        self,
        batch: Tuple[Tensor, Tensor],
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> Dict[str, float]:
        """Perform a single training step with DP.

        Args:
            batch: Tuple of (inputs, targets).
            loss_fn: Loss function (default: CrossEntropyLoss).

        Returns:
            Dictionary with loss and privacy metrics.
        """
        inputs, targets = batch

        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        targets = targets.to(device)

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.model.zero_grad()

        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)

        loss.backward()

        total_norm = self._clip_gradients()

        self._add_noise()

        self.optimizer.step()

        self._step_count += 1
        self.accountant.step(
            sample_rate=self.config.sample_rate,
            noise_multiplier=self.config.noise_multiplier,
        )

        eps, delta = self.accountant.get_privacy_spent()

        return {
            "loss": loss.item(),
            "grad_norm": total_norm,
            "epsilon": eps,
            "delta": delta,
        }

    def _clip_gradients(self) -> float:
        """Clip all model gradients.

        Returns:
            Total gradient norm before clipping.
        """
        total_norm = 0.0

        for p in self.model.parameters():
            if p.grad is not None:
                clipped, norm = self._clipper.clip(p.grad)
                p.grad = clipped
                total_norm = max(total_norm, norm)

        return total_norm

    def _add_noise(self) -> None:
        """Add Gaussian noise to clipped gradients."""
        noise_scale = self.config.noise_multiplier * self.config.max_grad_norm

        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * noise_scale
                p.grad = p.grad + noise

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        callbacks: Optional[List[Callable[[Dict], None]]] = None,
    ) -> Dict[str, List[float]]:
        """Train the model with DP.

        Args:
            train_loader: Training data loader.
            epochs: Number of epochs.
            loss_fn: Loss function.
            val_loader: Validation data loader.
            device: Device to train on.
            callbacks: List of callbacks for monitoring.

        Returns:
            Training history.
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
            n_batches = 0

            for batch in train_loader:
                metrics = self.step(batch, loss_fn)

                epoch_loss += metrics["loss"]
                epoch_grad_norm += metrics["grad_norm"]
                n_batches += 1

                if callbacks:
                    for cb in callbacks:
                        cb(metrics)

            eps, _ = self.accountant.get_privacy_spent()

            history["train_loss"].append(epoch_loss / n_batches)
            history["grad_norm"].append(epoch_grad_norm / n_batches)
            history["epsilon"].append(eps)

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

    def is_target_reached(self) -> bool:
        """Check if target privacy has been reached.

        Returns:
            True if target epsilon is within budget.
        """
        eps, _ = self.get_privacy_spent()
        return eps <= self.config.epsilon

    def get_remaining_budget(self) -> float:
        """Get remaining epsilon budget.

        Returns:
            Remaining epsilon.
        """
        eps, _ = self.get_privacy_spent()
        return max(0, self.config.epsilon - eps)

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing.

        Returns:
            State dictionary.
        """
        return {
            "config": self.config,
            "step_count": self._step_count,
            "accountant_state": self.accountant,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: State dictionary.
        """
        self._step_count = state_dict["step_count"]

    def __repr__(self) -> str:
        eps, delta = self.get_privacy_spent()
        return (
            f"PrivacyEngine(\n"
            f"  epsilon: {self.config.epsilon},\n"
            f"  delta: {self.config.delta},\n"
            f"  max_grad_norm: {self.config.max_grad_norm},\n"
            f"  noise_multiplier: {self.config.noise_multiplier},\n"
            f"  steps: {self._step_count},\n"
            f"  spent: ({eps:.4f}, {delta:.2e})\n"
            f")"
        )


class FederatedPrivacyEngine(PrivacyEngine):
    """Privacy engine for federated learning.

    Extends PrivacyEngine with federated-specific features like
    client aggregation and secure aggregation.

    Args:
        model_template: Template model for clients.
        config: Privacy engine configuration.
        aggregator: Aggregation method to use.

    Example:
        >>> engine = FederatedPrivacyEngine(
        ...     model_template=Model(),
        ...     epsilon=8.0,
        ...     aggregator='dp_federated'
        ... )
        >>> global_model = engine.aggregate(client_updates)
    """

    def __init__(
        self,
        model_template: Module,
        config: Optional[PrivacyEngineConfig] = None,
        aggregator: str = "dp_federated",
    ):
        super().__init__(model_template, config=config)

        self.model_template = model_template
        self._setup_aggregator(aggregator)

    def _setup_aggregator(self, aggregator_type: str) -> None:
        """Setup client aggregator."""
        if aggregator_type == "dp_federated":
            self._aggregator = DPFederatedAggregator(
                clip_norm=self.config.max_grad_norm,
                noise_scale=self.config.noise_multiplier,
                epsilon=self.config.epsilon,
                delta=self.config.delta,
            )
        elif aggregator_type == "noisy":
            self._aggregator = NoisyAggregator(
                noise_scale=self.config.noise_multiplier,
                clip_norm=self.config.max_grad_norm,
            )
        else:
            raise ValueError(f"Unknown aggregator: {aggregator_type}")

    def aggregate(
        self,
        client_updates: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Aggregate client updates.

        Args:
            client_updates: List of model parameter dictionaries.
            client_weights: Optional weights for each client.

        Returns:
            Aggregated model parameters.
        """
        return self._aggregator.aggregate(client_updates, client_weights)

    def aggregate_models(
        self,
        client_models: List[Module],
        client_weights: Optional[List[float]] = None,
    ) -> Module:
        """Aggregate client models.

        Args:
            client_models: List of PyTorch models.
            client_weights: Optional weights.

        Returns:
            Aggregated model.
        """
        return self._aggregator.aggregate_models(client_models, client_weights)


def create_privacy_engine(
    model: Module,
    optimizer: Optional[optim.Optimizer] = None,
    epsilon: float = 8.0,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    **kwargs,
) -> PrivacyEngine:
    """Factory function to create privacy engine.

    Args:
        model: Model to train.
        optimizer: Optimizer to use.
        epsilon: Target privacy budget.
        delta: Target delta.
        max_grad_norm: Maximum gradient norm.
        noise_multiplier: Noise multiplier.
        **kwargs: Additional configuration.

    Returns:
        Configured PrivacyEngine.

    Example:
        >>> engine = create_privacy_engine(model, epsilon=8.0)
    """
    config = PrivacyEngineConfig(
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        **kwargs,
    )

    return PrivacyEngine(model, optimizer, config)


def estimate_training_epsilon(
    num_steps: int,
    sample_rate: float,
    noise_multiplier: float,
    delta: float = 1e-5,
) -> float:
    """Estimate epsilon after training.

    Args:
        num_steps: Number of training steps.
        sample_rate: Sampling rate per step.
        noise_multiplier: Noise multiplier.
        delta: Target delta.

    Returns:
        Estimated epsilon spent.
    """
    return estimate_epsilon(num_steps, sample_rate, noise_multiplier, delta)
