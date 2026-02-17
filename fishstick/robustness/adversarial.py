"""
Adversarial Robustness Tools for fishstick.

This module provides comprehensive tools for adversarial attacks, defenses,
evaluation, and certification for deep learning models.

Example:
    >>> from fishstick.robustness.adversarial import FGSM, PGD, AdversarialTraining
    >>>
    >>> # Create an attack
    >>> attack = PGD(epsilon=0.03, alpha=0.01, steps=40)
    >>>
    >>> # Generate adversarial examples
    >>> adv_x = attack(model, x, y)
    >>>
    >>> # Evaluate robustness
    >>> from fishstick.robustness.adversarial import robust_accuracy
    >>> acc = robust_accuracy(model, x, y, attack)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Type aliases
Tensor = torch.Tensor
Module = nn.Module
LossFunction = Callable[[Tensor, Tensor], Tensor]
NormType = Literal["l2", "linf"]


# =============================================================================
# Attack Utilities
# =============================================================================


def clip_perturbation(
    perturbation: Tensor,
    epsilon: float,
    norm: NormType = "linf",
) -> Tensor:
    """Clip perturbation to epsilon ball.

    Args:
        perturbation: The perturbation tensor to clip.
        epsilon: Maximum perturbation magnitude.
        norm: Norm type ('l2' or 'linf').

    Returns:
        Clipped perturbation.

    Example:
        >>> delta = torch.randn(1, 3, 32, 32)
        >>> delta = clip_perturbation(delta, epsilon=0.03, norm="linf")
    """
    if norm == "linf":
        return torch.clamp(perturbation, -epsilon, epsilon)
    elif norm == "l2":
        batch_size = perturbation.shape[0]
        perturbation_flat = perturbation.view(batch_size, -1)
        norm_flat = torch.norm(perturbation_flat, p=2, dim=1, keepdim=True)
        factor = torch.clamp(norm_flat / epsilon, min=1.0)
        perturbation_flat = perturbation_flat / factor
        return perturbation_flat.view_as(perturbation)
    else:
        raise ValueError(f"Unknown norm: {norm}")


def normalize_perturbation(
    perturbation: Tensor,
    norm: NormType = "linf",
) -> Tensor:
    """Normalize perturbation by L-inf or L2 norm.

    Args:
        perturbation: The perturbation tensor to normalize.
        norm: Norm type ('l2' or 'linf').

    Returns:
        Normalized perturbation with unit norm.

    Example:
        >>> delta = torch.randn(1, 3, 32, 32)
        >>> delta = normalize_perturbation(delta, norm="l2")
    """
    if norm == "linf":
        return torch.sign(perturbation)
    elif norm == "l2":
        batch_size = perturbation.shape[0]
        perturbation_flat = perturbation.view(batch_size, -1)
        norm_flat = torch.norm(perturbation_flat, p=2, dim=1, keepdim=True)
        norm_flat = torch.clamp(norm_flat, min=1e-12)
        perturbation_flat = perturbation_flat / norm_flat
        return perturbation_flat.view_as(perturbation)
    else:
        raise ValueError(f"Unknown norm: {norm}")


def project_perturbation(
    perturbation: Tensor,
    original: Tensor,
    epsilon: float,
    norm: NormType = "linf",
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> Tensor:
    """Project perturbation onto valid perturbation set.

    Projects perturbation to satisfy:
    1. Norm constraint: ||delta|| <= epsilon
    2. Bounds constraint: bounds[0] <= x + delta <= bounds[1]

    Args:
        perturbation: Current perturbation.
        original: Original input.
        epsilon: Maximum perturbation magnitude.
        norm: Norm type ('l2' or 'linf').
        bounds: Valid input bounds (min, max).

    Returns:
        Projected perturbation.

    Example:
        >>> delta = torch.randn(1, 3, 32, 32)
        >>> x = torch.rand(1, 3, 32, 32)
        >>> delta = project_perturbation(delta, x, epsilon=0.03)
    """
    # Clip to epsilon ball
    if norm == "linf":
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
    elif norm == "l2":
        batch_size = perturbation.shape[0]
        perturbation_flat = perturbation.view(batch_size, -1)
        norm_flat = torch.norm(perturbation_flat, p=2, dim=1, keepdim=True)
        factor = torch.clamp(norm_flat / epsilon, min=1.0)
        perturbation_flat = perturbation_flat / factor
        perturbation = perturbation_flat.view_as(perturbation)

    # Clip to valid input range
    adv = torch.clamp(original + perturbation, bounds[0], bounds[1])
    perturbation = adv - original

    return perturbation


def compute_gradient(
    model: Module,
    x: Tensor,
    y: Tensor,
    loss_fn: Optional[LossFunction] = None,
    create_graph: bool = False,
) -> Tensor:
    """Compute gradient of loss w.r.t. input.

    Args:
        model: The model to attack.
        x: Input tensor.
        y: True labels.
        loss_fn: Loss function (default: cross entropy).
        create_graph: Whether to create computation graph.

    Returns:
        Gradient tensor.
    """
    if loss_fn is None:
        loss_fn = F.cross_entropy

    x_adv = x.clone().detach().requires_grad_(True)
    model.eval()

    output = model(x_adv)
    loss = loss_fn(output, y)

    grad = torch.autograd.grad(
        loss,
        x_adv,
        retain_graph=create_graph,
        create_graph=create_graph,
    )[0]

    return grad


# =============================================================================
# Base Attack Class
# =============================================================================


class Attack(ABC):
    """Base class for adversarial attacks.

    All attacks should inherit from this class and implement the
    `forward` method.

    Attributes:
        model: The target model.
        bounds: Valid input bounds (min, max).
        targeted: Whether the attack is targeted.

    Example:
        >>> class MyAttack(Attack):
        ...     def forward(self, x, y):
        ...         # Attack implementation
        ...         return x_adv
    """

    def __init__(
        self,
        bounds: Tuple[float, float] = (0.0, 1.0),
        targeted: bool = False,
    ):
        self.bounds = bounds
        self.targeted = targeted
        self._model: Optional[Module] = None

    def __call__(
        self,
        model: Module,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Generate adversarial examples.

        Args:
            model: The model to attack.
            x: Input tensor.
            y: Target labels (or true labels for untargeted attacks).

        Returns:
            Adversarial examples.
        """
        self._model = model
        training = model.training
        model.eval()

        with torch.enable_grad():
            x_adv = self.forward(x, y)

        if training:
            model.train()

        return x_adv.detach()

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Generate adversarial examples.

        Args:
            x: Input tensor.
            y: Target labels.

        Returns:
            Adversarial examples.
        """
        pass

    def _loss_fn(
        self,
        output: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Compute loss for attack.

        Args:
            output: Model output.
            y: Target labels.

        Returns:
            Loss value.
        """
        if self.targeted:
            return -F.cross_entropy(output, y)
        else:
            return F.cross_entropy(output, y)

    @property
    def model(self) -> Module:
        """Get the model being attacked."""
        if self._model is None:
            raise RuntimeError("Model not set. Call attack(model, x, y) first.")
        return self._model


# =============================================================================
# Attack Methods
# =============================================================================


class FGSM(Attack):
    """Fast Gradient Sign Method (FGSM) attack.

    One-step attack that uses the sign of the gradient to create adversarial
    examples.

    Reference:
        Goodfellow et al., "Explaining and Harnessing Adversarial Examples",
        ICLR 2015.

    Args:
        epsilon: Maximum perturbation magnitude.
        bounds: Valid input bounds.
        targeted: Whether the attack is targeted.

    Example:
        >>> attack = FGSM(epsilon=0.03)
        >>> adv_x = attack(model, x, y)
    """

    def __init__(
        self,
        epsilon: float = 0.03,
        bounds: Tuple[float, float] = (0.0, 1.0),
        targeted: bool = False,
    ):
        super().__init__(bounds, targeted)
        self.epsilon = epsilon

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Generate adversarial examples using FGSM."""
        x = x.clone().detach().requires_grad_(True)

        output = self.model(x)
        loss = self._loss_fn(output, y)

        grad = torch.autograd.grad(loss, x)[0]

        if self.targeted:
            delta = -self.epsilon * torch.sign(grad)
        else:
            delta = self.epsilon * torch.sign(grad)

        x_adv = torch.clamp(x + delta, self.bounds[0], self.bounds[1])
        return x_adv


class PGD(Attack):
    """Projected Gradient Descent (PGD) attack.

    Multi-step iterative attack that projects perturbations onto epsilon ball.

    Reference:
        Madry et al., "Towards Deep Learning Models Resistant to Adversarial
        Attacks", ICLR 2018.

    Args:
        epsilon: Maximum perturbation magnitude.
        alpha: Step size per iteration.
        steps: Number of attack iterations.
        random_start: Whether to use random initialization.
        bounds: Valid input bounds.
        targeted: Whether the attack is targeted.
        norm: Norm type ('l2' or 'linf').

    Example:
        >>> attack = PGD(epsilon=0.03, alpha=0.01, steps=40)
        >>> adv_x = attack(model, x, y)
    """

    def __init__(
        self,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        steps: int = 40,
        random_start: bool = True,
        bounds: Tuple[float, float] = (0.0, 1.0),
        targeted: bool = False,
        norm: NormType = "linf",
    ):
        super().__init__(bounds, targeted)
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.norm = norm

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Generate adversarial examples using PGD."""
        x_adv = x.clone().detach()

        # Random initialization
        if self.random_start:
            if self.norm == "linf":
                delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            else:  # l2
                delta = torch.randn_like(x)
                delta = normalize_perturbation(delta, "l2") * self.epsilon
            x_adv = torch.clamp(x_adv + delta, self.bounds[0], self.bounds[1])

        for _ in range(self.steps):
            x_adv.requires_grad_(True)

            output = self.model(x_adv)
            loss = self._loss_fn(output, y)

            grad = torch.autograd.grad(loss, x_adv)[0]

            # Update with gradient step
            if self.targeted:
                x_adv = x_adv.detach() - self.alpha * torch.sign(grad)
            else:
                x_adv = x_adv.detach() + self.alpha * torch.sign(grad)

            # Project back to epsilon ball
            delta = x_adv - x
            delta = clip_perturbation(delta, self.epsilon, self.norm)
            x_adv = torch.clamp(x + delta, self.bounds[0], self.bounds[1])

        return x_adv


class PGDL2(PGD):
    """PGD attack with L2 norm constraint.

    Same as PGD but uses L2 norm for perturbation constraint.

    Args:
        epsilon: Maximum L2 perturbation magnitude.
        alpha: Step size per iteration.
        steps: Number of attack iterations.
        random_start: Whether to use random initialization.
        bounds: Valid input bounds.
        targeted: Whether the attack is targeted.

    Example:
        >>> attack = PGDL2(epsilon=1.0, alpha=0.2, steps=40)
        >>> adv_x = attack(model, x, y)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        alpha: float = 0.2,
        steps: int = 40,
        random_start: bool = True,
        bounds: Tuple[float, float] = (0.0, 1.0),
        targeted: bool = False,
    ):
        super().__init__(
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
            bounds=bounds,
            targeted=targeted,
            norm="l2",
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Generate adversarial examples using PGD-L2."""
        x_adv = x.clone().detach()
        batch_size = x.shape[0]

        # Random initialization on L2 sphere
        if self.random_start:
            delta = torch.randn_like(x)
            delta = delta.view(batch_size, -1)
            norm = torch.norm(delta, p=2, dim=1, keepdim=True)
            delta = (
                delta
                / norm.clamp_min(1e-12)
                * torch.empty(batch_size, 1).uniform_(0, self.epsilon).to(x.device)
            )
            delta = delta.view_as(x)
            x_adv = torch.clamp(x + delta, self.bounds[0], self.bounds[1])

        for _ in range(self.steps):
            x_adv.requires_grad_(True)

            output = self.model(x_adv)
            loss = self._loss_fn(output, y)

            grad = torch.autograd.grad(loss, x_adv)[0]

            # Normalize gradient for L2 step
            grad_norm = grad.view(batch_size, -1).norm(p=2, dim=1, keepdim=True)
            grad_normalized = grad / grad_norm.view_as(grad).clamp_min(1e-12)

            # Update with gradient step
            if self.targeted:
                x_adv = x_adv.detach() - self.alpha * grad_normalized
            else:
                x_adv = x_adv.detach() + self.alpha * grad_normalized

            # Project back to L2 ball
            delta = x_adv - x
            delta_flat = delta.view(batch_size, -1)
            norm_flat = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
            factor = torch.clamp(norm_flat / self.epsilon, min=1.0)
            delta_flat = delta_flat / factor
            delta = delta_flat.view_as(x)

            x_adv = torch.clamp(x + delta, self.bounds[0], self.bounds[1])

        return x_adv


class DeepFool(Attack):
    """DeepFool attack for minimal perturbation.

    Iteratively finds minimal perturbation to change classification.

    Reference:
        Moosavi-Dezfooli et al., "DeepFool: A Simple and Accurate Method to
        Fool Deep Neural Networks", CVPR 2016.

    Args:
        steps: Maximum number of iterations.
        overshoot: Overshoot parameter to ensure misclassification.
        bounds: Valid input bounds.

    Example:
        >>> attack = DeepFool(steps=50, overshoot=0.02)
        >>> adv_x = attack(model, x, y)
    """

    def __init__(
        self,
        steps: int = 50,
        overshoot: float = 0.02,
        bounds: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(bounds, targeted=False)
        self.steps = steps
        self.overshoot = overshoot

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Generate adversarial examples using DeepFool."""
        batch_size = x.shape[0]
        x_adv = x.clone().detach().requires_grad_(True)

        for _ in range(self.steps):
            output = self.model(x_adv)
            _, pred = torch.max(output, dim=1)

            # Check if already adversarial
            if not (pred == y).any():
                break

            # Compute gradients for all classes
            output_dim = output.shape[1]
            grads = []

            for k in range(output_dim):
                if x_adv.grad is not None:
                    x_adv.grad.zero_()

                output[0, k].backward(retain_graph=True)
                grads.append(x_adv.grad.clone())

            grads = torch.stack(grads)

            # Compute minimal perturbation for each sample
            x_adv_new = []
            for i in range(batch_size):
                if pred[i] != y[i]:
                    x_adv_new.append(x[i])
                    continue

                l = pred[i].item()
                grad_l = grads[l]

                min_dist = float("inf")
                best_pert = None

                for k in range(output_dim):
                    if k == l:
                        continue

                    w_k = grads[k] - grad_l
                    f_k = output[i, k] - output[i, l]

                    dist = abs(f_k) / (w_k[i].norm() + 1e-12)

                    if dist < min_dist:
                        min_dist = dist
                        best_pert = f_k * w_k[i] / (w_k[i].norm() ** 2 + 1e-12)

                if best_pert is not None:
                    x_adv_new.append(x[i] + (1 + self.overshoot) * best_pert)
                else:
                    x_adv_new.append(x[i])

            x_adv = torch.stack(x_adv_new).detach().requires_grad_(True)

        return torch.clamp(x_adv.detach(), self.bounds[0], self.bounds[1])


class CW(Attack):
    """Carlini & Wagner L2 attack.

    Powerful optimization-based attack that minimizes perturbation while
    ensuring misclassification.

    Reference:
        Carlini and Wagner, "Towards Evaluating the Robustness of Neural
        Networks", IEEE S&P 2017.

    Args:
        confidence: Confidence parameter for attack.
        steps: Number of optimization steps.
        lr: Learning rate for optimization.
        initial_const: Initial constant c.
        bounds: Valid input bounds.
        targeted: Whether the attack is targeted.

    Example:
        >>> attack = CW(confidence=0, steps=1000, lr=0.01)
        >>> adv_x = attack(model, x, y)
    """

    def __init__(
        self,
        confidence: float = 0.0,
        steps: int = 1000,
        lr: float = 0.01,
        initial_const: float = 0.001,
        bounds: Tuple[float, float] = (0.0, 1.0),
        targeted: bool = False,
    ):
        super().__init__(bounds, targeted)
        self.confidence = confidence
        self.steps = steps
        self.lr = lr
        self.initial_const = initial_const
        self.kappa = confidence

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Generate adversarial examples using C&W."""
        batch_size = x.shape[0]

        # Map to tanh space
        def to_tanh_space(x):
            return 0.5 * torch.log((x + 1e-8) / (1 - x + 1e-8))

        def from_tanh_space(w):
            return (torch.tanh(w) + 1) / 2 * (
                self.bounds[1] - self.bounds[0]
            ) + self.bounds[0]

        x_tanh = to_tanh_space((x - self.bounds[0]) / (self.bounds[1] - self.bounds[0]))

        w = x_tanh.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=self.lr)

        const = self.initial_const

        best_adv = x.clone()
        best_l2 = torch.full((batch_size,), float("inf")).to(x.device)

        for step in range(self.steps):
            x_adv = from_tanh_space(w)

            # Compute L2 distance
            l2_dist = torch.sum((x_adv - x) ** 2, dim=list(range(1, x.ndim)))

            # Compute f loss
            output = self.model(x_adv)

            real = torch.gather(output, 1, y.unsqueeze(1)).squeeze(1)

            if self.targeted:
                # Targeted: maximize target class
                other = torch.max(
                    output - torch.nn.functional.one_hot(y, output.shape[1]) * 1e10,
                    dim=1,
                )[0]
                f_loss = torch.clamp(other - real + self.kappa, min=0.0)
            else:
                # Untargeted: minimize true class
                other = torch.max(
                    output - torch.nn.functional.one_hot(y, output.shape[1]) * 1e10,
                    dim=1,
                )[0]
                f_loss = torch.clamp(real - other + self.kappa, min=0.0)

            loss = torch.sum(l2_dist + const * f_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update best
            _, pred = torch.max(output, dim=1)
            success = (pred == y) if self.targeted else (pred != y)

            improved = (l2_dist < best_l2) & success
            best_l2 = torch.where(improved, l2_dist, best_l2)
            best_adv = torch.where(
                improved.view(-1, *([1] * (x.ndim - 1))), x_adv, best_adv
            )

        return best_adv


class AutoAttack(Attack):
    """AutoAttack - ensemble of reliable attacks.

    Combines multiple attacks for comprehensive evaluation.

    Reference:
        Croce and Hein, "Reliable evaluation of adversarial robustness with
        an ensemble of diverse parameter-free attacks", ICML 2020.

    Args:
        epsilon: Maximum perturbation magnitude.
        norm: Norm type ('l2' or 'linf').
        steps: Number of steps for iterative attacks.
        bounds: Valid input bounds.
        version: Version of AutoAttack ('standard' or 'plus').

    Example:
        >>> attack = AutoAttack(epsilon=0.03, norm="linf")
        >>> adv_x = attack(model, x, y)
    """

    def __init__(
        self,
        epsilon: float = 0.03,
        norm: NormType = "linf",
        steps: int = 100,
        bounds: Tuple[float, float] = (0.0, 1.0),
        version: str = "standard",
    ):
        super().__init__(bounds, targeted=False)
        self.epsilon = epsilon
        self.norm = norm
        self.steps = steps
        self.version = version

        # Initialize component attacks
        self._init_attacks()

    def _init_attacks(self):
        """Initialize the ensemble of attacks."""
        if self.norm == "linf":
            alpha = self.epsilon / 4
            self.attacks = [
                (
                    "apgd-ce",
                    PGD(
                        epsilon=self.epsilon,
                        alpha=alpha,
                        steps=self.steps,
                        random_start=True,
                        bounds=self.bounds,
                        norm="linf",
                    ),
                ),
                (
                    "apgd-dlr",
                    PGD(
                        epsilon=self.epsilon,
                        alpha=alpha,
                        steps=self.steps,
                        random_start=True,
                        bounds=self.bounds,
                        norm="linf",
                    ),
                ),
            ]
        else:  # l2
            alpha = self.epsilon / 4
            self.attacks = [
                (
                    "apgd-ce-l2",
                    PGDL2(
                        epsilon=self.epsilon,
                        alpha=alpha,
                        steps=self.steps,
                        random_start=True,
                        bounds=self.bounds,
                    ),
                ),
            ]

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Generate adversarial examples using AutoAttack."""
        batch_size = x.shape[0]
        x_adv = x.clone()

        # Track which samples are still correctly classified
        remaining = torch.ones(batch_size, dtype=torch.bool).to(x.device)

        for name, attack in self.attacks:
            if not remaining.any():
                break

            # Apply attack to remaining samples
            x_rem = x_adv[remaining]
            y_rem = y[remaining]

            x_adv_rem = attack(self.model, x_rem, y_rem)

            # Check success
            with torch.no_grad():
                output = self.model(x_adv_rem)
                _, pred = torch.max(output, dim=1)
                success = pred != y_rem

            # Update adversarial examples
            x_adv_list = []
            idx = 0
            for i in range(batch_size):
                if remaining[i]:
                    x_adv_list.append(x_adv_rem[idx])
                    idx += 1
                else:
                    x_adv_list.append(x_adv[i])
            x_adv = torch.stack(x_adv_list)

            # Update remaining
            remaining_idx = 0
            new_remaining = remaining.clone()
            for i in range(batch_size):
                if remaining[i]:
                    if success[remaining_idx]:
                        new_remaining[i] = False
                    remaining_idx += 1
            remaining = new_remaining

        return x_adv


# =============================================================================
# Defense Methods
# =============================================================================


class Defense(ABC):
    """Base class for adversarial defenses."""

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Apply defense to input."""
        pass


class AdversarialTraining:
    """Adversarial training defense.

    Trains model on adversarial examples to improve robustness.

    Reference:
        Madry et al., "Towards Deep Learning Models Resistant to Adversarial
        Attacks", ICLR 2018.

    Args:
        model: Model to train.
        attack: Attack method for generating adversarial examples.
        alpha: Weight for adversarial loss (1.0 = pure AT).

    Example:
        >>> trainer = AdversarialTraining(model, PGD(epsilon=0.03, steps=10))
        >>> trainer.fit(train_loader, epochs=100)
    """

    def __init__(
        self,
        model: Module,
        attack: Attack,
        alpha: float = 1.0,
    ):
        self.model = model
        self.attack = attack
        self.alpha = alpha

    def training_step(
        self,
        x: Tensor,
        y: Tensor,
        optimizer: Optimizer,
        loss_fn: Optional[LossFunction] = None,
    ) -> Dict[str, float]:
        """Single training step with adversarial examples.

        Args:
            x: Input batch.
            y: Label batch.
            optimizer: Optimizer.
            loss_fn: Loss function.

        Returns:
            Dictionary with loss and metrics.
        """
        if loss_fn is None:
            loss_fn = F.cross_entropy

        self.model.train()
        optimizer.zero_grad()

        # Generate adversarial examples
        x_adv = self.attack(self.model, x, y)

        # Compute loss on adversarial examples
        output_adv = self.model(x_adv)
        loss = loss_fn(output_adv, y)

        # Optionally mix with clean loss
        if self.alpha < 1.0:
            output_clean = self.model(x)
            loss_clean = loss_fn(output_clean, y)
            loss = self.alpha * loss + (1 - self.alpha) * loss_clean

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = output_adv.argmax(dim=1)
            accuracy = (pred == y).float().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
        }

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[LossFunction] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> Dict[str, List[float]]:
        """Train the model with adversarial training.

        Args:
            train_loader: Training data loader.
            epochs: Number of epochs.
            optimizer: Optimizer (default: Adam).
            loss_fn: Loss function.
            val_loader: Validation data loader.
            device: Device to train on.

        Returns:
            Training history.
        """
        self.model.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                metrics = self.training_step(x, y, optimizer, loss_fn)
                epoch_loss += metrics["loss"]
                epoch_acc += metrics["accuracy"]
                n_batches += 1

            history["train_loss"].append(epoch_loss / n_batches)
            history["train_acc"].append(epoch_acc / n_batches)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, loss_fn, device)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {history['train_loss'][-1]:.4f}, "
                    f"Acc: {history['train_acc'][-1]:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {history['train_loss'][-1]:.4f}, "
                    f"Acc: {history['train_acc'][-1]:.4f}"
                )

        return history

    def evaluate(
        self,
        loader: DataLoader,
        loss_fn: Optional[LossFunction] = None,
        device: str = "cuda",
    ) -> Tuple[float, float]:
        """Evaluate the model.

        Args:
            loader: Data loader.
            loss_fn: Loss function.
            device: Device.

        Returns:
            Tuple of (loss, accuracy).
        """
        if loss_fn is None:
            loss_fn = F.cross_entropy

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                output = self.model(x)
                loss = loss_fn(output, y)

                total_loss += loss.item() * x.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)

        return total_loss / total, correct / total


class TRADES:
    """TRADES: TRade-off-inspired Adversarial DEfense via Surrogate loss.

    Balances natural accuracy and robust accuracy.

    Reference:
        Zhang et al., "Theoretically Principled Trade-off between Robustness
        and Accuracy", ICML 2019.

    Args:
        model: Model to train.
        attack: Attack for generating adversarial examples.
        beta: Trade-off parameter (default: 6.0).
        epsilon: Perturbation budget for inner maximization.
        step_size: Step size for attack generation.
        num_steps: Number of steps for attack generation.

    Example:
        >>> trainer = TRADES(model, beta=6.0, epsilon=0.03)
        >>> trainer.fit(train_loader, epochs=100)
    """

    def __init__(
        self,
        model: Module,
        beta: float = 6.0,
        epsilon: float = 0.03,
        step_size: float = 0.01,
        num_steps: int = 10,
    ):
        self.model = model
        self.beta = beta
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps

    def _generate_adversarial(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Generate adversarial examples for TRADES."""
        x_adv = x + 0.001 * torch.randn_like(x)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)

            with torch.enable_grad():
                loss_kl = F.kl_div(
                    F.log_softmax(self.model(x_adv), dim=1),
                    F.softmax(self.model(x), dim=1),
                    reduction="batchmean",
                )

            grad = torch.autograd.grad(loss_kl, x_adv)[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    def training_step(
        self,
        x: Tensor,
        y: Tensor,
        optimizer: Optimizer,
    ) -> Dict[str, float]:
        """Single TRADES training step.

        Args:
            x: Input batch.
            y: Label batch.
            optimizer: Optimizer.

        Returns:
            Dictionary with losses and metrics.
        """
        self.model.train()
        optimizer.zero_grad()

        # Generate adversarial examples
        x_adv = self._generate_adversarial(x, y)

        # Compute natural loss
        logits = self.model(x)
        loss_natural = F.cross_entropy(logits, y)

        # Compute robust loss (KL divergence)
        logits_adv = self.model(x_adv)
        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits, dim=1),
            reduction="batchmean",
        )

        # Total loss
        loss = loss_natural + self.beta * loss_robust

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            accuracy = (pred == y).float().mean().item()

        return {
            "loss": loss.item(),
            "loss_natural": loss_natural.item(),
            "loss_robust": loss_robust.item(),
            "accuracy": accuracy,
        }

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        optimizer: Optional[Optimizer] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> Dict[str, List[float]]:
        """Train with TRADES.

        Args:
            train_loader: Training data loader.
            epochs: Number of epochs.
            optimizer: Optimizer.
            val_loader: Validation data loader.
            device: Device.

        Returns:
            Training history.
        """
        self.model.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                metrics = self.training_step(x, y, optimizer)
                epoch_loss += metrics["loss"]
                epoch_acc += metrics["accuracy"]
                n_batches += 1

            history["train_loss"].append(epoch_loss / n_batches)
            history["train_acc"].append(epoch_acc / n_batches)

            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, device)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {history['train_loss'][-1]:.4f}, "
                    f"Acc: {history['train_acc'][-1]:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {history['train_loss'][-1]:.4f}, "
                    f"Acc: {history['train_acc'][-1]:.4f}"
                )

        return history

    def _evaluate(
        self,
        loader: DataLoader,
        device: str,
    ) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                output = self.model(x)
                loss = F.cross_entropy(output, y)

                total_loss += loss.item() * x.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)

        return total_loss / total, correct / total


class InputTransformation(Defense):
    """Input transformation defense.

    Applies transformations like JPEG compression, bit-depth reduction,
    or feature squeezing to remove adversarial perturbations.

    Args:
        method: Transformation method ('jpeg', 'bit_depth', 'feature_squeeze').
        quality: Quality parameter (JPEG: 1-100, bit_depth: 1-8).

    Example:
        >>> defense = InputTransformation('jpeg', quality=75)
        >>> x_clean = defense(x_adv)
    """

    def __init__(
        self,
        method: Literal["jpeg", "bit_depth", "feature_squeeze"] = "jpeg",
        quality: int = 75,
    ):
        self.method = method
        self.quality = quality

    def __call__(self, x: Tensor) -> Tensor:
        """Apply input transformation.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Transformed tensor.
        """
        if self.method == "jpeg":
            return self._jpeg_compression(x)
        elif self.method == "bit_depth":
            return self._bit_depth_reduction(x)
        elif self.method == "feature_squeeze":
            return self._feature_squeezing(x)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _jpeg_compression(self, x: Tensor) -> Tensor:
        """Apply JPEG compression."""
        try:
            from PIL import Image
            import io
        except ImportError:
            warnings.warn("PIL not available, skipping JPEG compression")
            return x

        device = x.device
        x_np = x.cpu().numpy()

        results = []
        for img in x_np:
            # Convert to PIL
            img_pil = Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))

            # JPEG compress
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            img_compressed = Image.open(buffer)

            # Convert back
            img_arr = np.array(img_compressed).astype(np.float32) / 255.0
            img_arr = img_arr.transpose(2, 0, 1)
            results.append(img_arr)

        return torch.tensor(np.stack(results), device=device)

    def _bit_depth_reduction(self, x: Tensor) -> Tensor:
        """Reduce bit depth."""
        levels = 2**self.quality
        x_quantized = torch.floor(x * levels) / levels
        return x_quantized

    def _feature_squeezing(self, x: Tensor) -> Tensor:
        """Apply feature squeezing (spatial smoothing)."""
        # Apply average pooling
        kernel_size = max(1, 9 - self.quality)
        if kernel_size > 1:
            x = F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)
        return x


class RandomizationDefense(Defense):
    """Randomization defense with random resizing and padding.

    Reference:
        Xie et al., "Mitigating Adversarial Effects Through Randomization",
        ICLR 2018.

    Args:
        resize_range: Range for random resizing (e.g., (0.9, 1.1)).
        padding: Maximum padding size.

    Example:
        >>> defense = RandomizationDefense(resize_range=(0.9, 1.1), padding=8)
        >>> x_clean = defense(x_adv)
    """

    def __init__(
        self,
        resize_range: Tuple[float, float] = (0.9, 1.1),
        padding: int = 8,
    ):
        self.resize_range = resize_range
        self.padding = padding

    def __call__(self, x: Tensor) -> Tensor:
        """Apply randomization defense.

        Args:
            x: Input tensor.

        Returns:
            Randomized tensor.
        """
        if not self.model.training:
            # Use deterministic version at test time
            return x

        batch_size, c, h, w = x.shape

        # Random resize
        scale = np.random.uniform(*self.resize_range)
        new_h, new_w = int(h * scale), int(w * scale)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # Random padding
        pad_h = np.random.randint(0, self.padding * 2)
        pad_w = np.random.randint(0, self.padding * 2)
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        pad_bottom = self.padding * 2 - pad_top
        pad_right = self.padding * 2 - pad_left

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

        # Resize back to original size
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        return x


class DistillationDefense:
    """Defensive distillation defense.

    Trains a model using soft labels from a teacher model trained with
    high temperature.

    Reference:
        Papernot et al., "Distillation as a Defense to Adversarial Perturbations
        against Deep Neural Networks", IEEE S&P 2016.

    Args:
        temperature: Temperature for distillation.
        alpha: Weight for soft loss vs hard loss.

    Example:
        >>> defense = DistillationDefense(temperature=100.0)
        >>> defense.distill(teacher, student, train_loader, epochs=50)
    """

    def __init__(
        self,
        temperature: float = 100.0,
        alpha: float = 0.7,
    ):
        self.temperature = temperature
        self.alpha = alpha

    def distill(
        self,
        teacher: Module,
        student: Module,
        train_loader: DataLoader,
        epochs: int,
        optimizer: Optional[Optimizer] = None,
        device: str = "cuda",
    ) -> Dict[str, List[float]]:
        """Perform defensive distillation.

        Args:
            teacher: Teacher model (already trained).
            student: Student model to train.
            train_loader: Training data loader.
            epochs: Number of epochs.
            optimizer: Optimizer for student.
            device: Device.

        Returns:
            Training history.
        """
        teacher.to(device).eval()
        student.to(device).train()

        if optimizer is None:
            optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                # Get soft labels from teacher
                with torch.no_grad():
                    teacher_logits = teacher(x) / self.temperature
                    soft_labels = F.softmax(teacher_logits, dim=1)

                # Student predictions
                student_logits = student(x)
                student_logits_temp = student_logits / self.temperature

                # Distillation loss
                loss_soft = F.kl_div(
                    F.log_softmax(student_logits_temp, dim=1),
                    soft_labels,
                    reduction="batchmean",
                ) * (self.temperature**2)

                # Hard loss
                loss_hard = F.cross_entropy(student_logits, y)

                # Combined loss
                loss = self.alpha * loss_soft + (1 - self.alpha) * loss_hard

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred = student_logits.argmax(dim=1)
                    acc = (pred == y).float().mean().item()

                epoch_loss += loss.item()
                epoch_acc += acc
                n_batches += 1

            history["loss"].append(epoch_loss / n_batches)
            history["accuracy"].append(epoch_acc / n_batches)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {history['loss'][-1]:.4f}, "
                f"Acc: {history['accuracy'][-1]:.4f}"
            )

        return history


# =============================================================================
# Certified Defenses
# =============================================================================


class RandomizedSmoothing:
    """Randomized smoothing for certified robustness.

    Adds Gaussian noise to inputs and certifies robustness radius.

    Reference:
        Cohen et al., "Certified Adversarial Robustness via Randomized
        Smoothing", ICML 2019.

    Args:
        base_classifier: Base classifier to smooth.
        sigma: Standard deviation of Gaussian noise.
        num_samples: Number of samples for certification.
        alpha: Confidence level.

    Example:
        >>> smoothed = RandomizedSmoothing(model, sigma=0.25)
        >>> prediction, radius = smoothed.certify(x, n0=100, n=10000)
    """

    def __init__(
        self,
        base_classifier: Module,
        sigma: float = 0.25,
        num_samples: int = 100000,
        alpha: float = 0.001,
    ):
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.num_samples = num_samples
        self.alpha = alpha

    def predict(
        self,
        x: Tensor,
        n: int = 100,
        batch_size: int = 1000,
    ) -> Tensor:
        """Predict class using smoothed classifier.

        Args:
            x: Input tensor.
            n: Number of noise samples.
            batch_size: Batch size for prediction.

        Returns:
            Predicted class.
        """
        self.base_classifier.eval()

        counts = self._sample_noise(x, n, batch_size)
        return counts.argmax(dim=1)

    def certify(
        self,
        x: Tensor,
        n0: int = 100,
        n: int = 100000,
        batch_size: int = 1000,
    ) -> Tuple[Tensor, Tensor]:
        """Certify robustness radius.

        Args:
            x: Input tensor.
            n0: Number of samples for initial prediction.
            n: Number of samples for certification.
            batch_size: Batch size for prediction.

        Returns:
            Tuple of (predicted class, certified radius).
            Returns (-1, 0) if abstains.
        """
        self.base_classifier.eval()

        # Initial prediction
        counts0 = self._sample_noise(x, n0, batch_size)
        cAHat = counts0.argmax(dim=1)

        # Certification
        counts = self._sample_noise(x, n, batch_size)

        # Get top two classes
        top2 = counts.topk(2, dim=1)[0]
        nA = top2[:, 0]
        nB = top2[:, 1]

        # Use Clopper-Pearson confidence interval
        from scipy.stats import binom_test

        batch_size_x = x.shape[0]
        radii = torch.zeros(batch_size_x)
        predictions = torch.full((batch_size_x,), -1, dtype=torch.long)

        for i in range(batch_size_x):
            # Check if cAHat is still top
            if counts[i, cAHat[i]] != nA[i]:
                continue

            # Two-sample binomial test
            p_lower = self._binom_confidence(nA[i].item(), n, self.alpha)

            if p_lower < 0.5:
                continue

            radius = self.sigma * self._normal_inverse_cdf(p_lower)

            predictions[i] = cAHat[i]
            radii[i] = radius

        return predictions, radii

    def _sample_noise(
        self,
        x: Tensor,
        n: int,
        batch_size: int,
    ) -> Tensor:
        """Sample predictions with Gaussian noise."""
        device = x.device
        num_classes = None
        counts = None

        with torch.no_grad():
            for _ in range(0, n, batch_size):
                this_batch_size = min(batch_size, n - _)

                # Repeat x and add noise
                x_repeated = x.repeat(this_batch_size, 1, 1, 1)
                noise = torch.randn_like(x_repeated) * self.sigma
                x_noisy = x_repeated + noise

                # Predict
                logits = self.base_classifier(x_noisy)

                if num_classes is None:
                    num_classes = logits.shape[1]
                    counts = torch.zeros(x.shape[0], num_classes, device=device)

                predictions = logits.argmax(dim=1)

                # Update counts
                for i in range(x.shape[0]):
                    for j in range(this_batch_size):
                        counts[i, predictions[i * this_batch_size + j]] += 1

        return counts

    def _binom_confidence(
        self,
        k: int,
        n: int,
        alpha: float,
    ) -> float:
        """Lower confidence bound for binomial proportion."""
        from scipy.stats import beta

        return beta.ppf(alpha, k, n - k + 1)

    def _normal_inverse_cdf(self, p: float) -> float:
        """Inverse CDF of standard normal."""
        import math

        return math.sqrt(2) * self._erfinv(2 * p - 1)

    def _erfinv(self, x: float) -> float:
        """Inverse error function approximation."""
        # Rational approximation
        a = 0.147

        log1mx2 = math.log(1 - x * x)

        p1 = 2 / (math.pi * a) + log1mx2 / 2
        p2 = log1mx2 / a

        return torch.sign(torch.tensor(x)) * math.sqrt(math.sqrt(p1 * p1 - p2) - p1)


class IntervalBoundPropagation:
    """Interval Bound Propagation for certified training.

    Trains models with certified robustness guarantees using interval bounds.

    Reference:
        Gowal et al., "On the Effectiveness of Interval Bound Propagation
        for Training Verifiably Robust Models", arXiv 2018.

    Args:
        model: Model to train.
        epsilon: Maximum perturbation.
        kappa: Schedule parameter for mixing natural and robust loss.

    Example:
        >>> ibp = IntervalBoundPropagation(model, epsilon=0.03)
        >>> ibp.fit(train_loader, epochs=100)
    """

    def __init__(
        self,
        model: Module,
        epsilon: float = 0.03,
        kappa: Callable[[int], float] = None,
    ):
        self.model = model
        self.epsilon = epsilon

        if kappa is None:
            # Default: linear schedule from 1.0 to 0.0
            self.kappa = lambda epoch: max(0.0, 1.0 - epoch / 100)
        else:
            self.kappa = kappa

    def _compute_bounds(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute interval bounds for robustness certification.

        Args:
            x: Input batch.
            y: Label batch.

        Returns:
            Tuple of (upper bounds, lower bounds).
        """
        # Create interval bounds
        x_lower = torch.clamp(x - self.epsilon, 0.0, 1.0)
        x_upper = torch.clamp(x + self.epsilon, 0.0, 1.0)

        # Propagate through network
        return self._ibp_forward(x_lower, x_upper)

    def _ibp_forward(
        self,
        lower: Tensor,
        upper: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with interval bounds."""
        # Simple implementation for ReLU networks
        # In practice, use specialized layers

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                weight_pos = torch.clamp(module.weight, min=0)
                weight_neg = torch.clamp(module.weight, max=0)

                new_lower = lower @ weight_pos.T + upper @ weight_neg.T + module.bias
                new_upper = upper @ weight_pos.T + lower @ weight_neg.T + module.bias

                lower, upper = new_lower, new_upper

            elif isinstance(module, nn.ReLU):
                lower = torch.clamp(lower, min=0)
                upper = torch.clamp(upper, min=0)

        return upper, lower

    def training_step(
        self,
        x: Tensor,
        y: Tensor,
        optimizer: Optimizer,
        epoch: int,
    ) -> Dict[str, float]:
        """Single IBP training step.

        Args:
            x: Input batch.
            y: Label batch.
            optimizer: Optimizer.
            epoch: Current epoch.

        Returns:
            Dictionary with losses and metrics.
        """
        self.model.train()
        optimizer.zero_grad()

        # Natural loss
        logits = self.model(x)
        loss_natural = F.cross_entropy(logits, y)

        # Robust loss (IBP)
        ub, lb = self._compute_bounds(x, y)

        # Worst-case logit difference
        batch_size = x.shape[0]
        num_classes = logits.shape[1]

        # Get lower bound for true class and upper bound for other classes
        y_onehot = F.one_hot(y, num_classes).float()

        lb_true = (lb * y_onehot).sum(dim=1)
        ub_other = ub * (1 - y_onehot) - y_onehot * 1e9
        ub_max_other = ub_other.max(dim=1)[0]

        # Robust loss: maximize the gap violation
        loss_robust = F.relu(ub_max_other - lb_true).mean()

        # Combined loss
        k = self.kappa(epoch)
        loss = k * loss_natural + (1 - k) * loss_robust

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            accuracy = (pred == y).float().mean().item()

            # Compute certified accuracy
            certified = (lb_true > ub_max_other).float().mean().item()

        return {
            "loss": loss.item(),
            "loss_natural": loss_natural.item(),
            "loss_robust": loss_robust.item(),
            "accuracy": accuracy,
            "certified": certified,
        }

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        optimizer: Optional[Optimizer] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> Dict[str, List[float]]:
        """Train with IBP.

        Args:
            train_loader: Training data loader.
            epochs: Number of epochs.
            optimizer: Optimizer.
            val_loader: Validation data loader.
            device: Device.

        Returns:
            Training history.
        """
        self.model.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        history = {
            "train_loss": [],
            "train_acc": [],
            "train_certified": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_cert = 0.0
            n_batches = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                metrics = self.training_step(x, y, optimizer, epoch)
                epoch_loss += metrics["loss"]
                epoch_acc += metrics["accuracy"]
                epoch_cert += metrics["certified"]
                n_batches += 1

            history["train_loss"].append(epoch_loss / n_batches)
            history["train_acc"].append(epoch_acc / n_batches)
            history["train_certified"].append(epoch_cert / n_batches)

            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, device)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {history['train_loss'][-1]:.4f}, "
                    f"Acc: {history['train_acc'][-1]:.4f}, "
                    f"Cert: {history['train_certified'][-1]:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {history['train_loss'][-1]:.4f}, "
                    f"Acc: {history['train_acc'][-1]:.4f}, "
                    f"Cert: {history['train_certified'][-1]:.4f}"
                )

        return history

    def _evaluate(
        self,
        loader: DataLoader,
        device: str,
    ) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                output = self.model(x)
                loss = F.cross_entropy(output, y)

                total_loss += loss.item() * x.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)

        return total_loss / total, correct / total


def compute_certified_radius(
    model: Module,
    x: Tensor,
    y: Tensor,
    sigma: float = 0.25,
    n0: int = 100,
    n: int = 100000,
) -> Tuple[Tensor, Tensor]:
    """Compute certified radius using randomized smoothing.

    Args:
        model: Base classifier.
        x: Input batch.
        y: True labels.
        sigma: Noise standard deviation.
        n0: Initial samples.
        n: Certification samples.

    Returns:
        Tuple of (predictions, certified radii).
    """
    smoother = RandomizedSmoothing(model, sigma=sigma)
    return smoother.certify(x, n0=n0, n=n)


# =============================================================================
# Robustness Evaluation
# =============================================================================


def robust_accuracy(
    model: Module,
    x: Tensor,
    y: Tensor,
    attack: Attack,
) -> float:
    """Compute accuracy under adversarial attack.

    Args:
        model: Model to evaluate.
        x: Input tensor.
        y: True labels.
        attack: Attack to apply.

    Returns:
        Robust accuracy (fraction of correctly classified adversarial examples).

    Example:
        >>> attack = PGD(epsilon=0.03, steps=40)
        >>> acc = robust_accuracy(model, x_test, y_test, attack)
        >>> print(f"Robust accuracy: {acc:.2%}")
    """
    model.eval()

    with torch.no_grad():
        x_adv = attack(model, x, y)
        output = model(x_adv)
        pred = output.argmax(dim=1)
        accuracy = (pred == y).float().mean().item()

    return accuracy


def perturbation_size(
    x: Tensor,
    x_adv: Tensor,
    norm: NormType = "linf",
) -> Dict[str, float]:
    """Compute average perturbation magnitude.

    Args:
        x: Original inputs.
        x_adv: Adversarial examples.
        norm: Norm type ('l2' or 'linf').

    Returns:
        Dictionary with statistics:
            - mean: Average perturbation.
            - max: Maximum perturbation.
            - median: Median perturbation.

    Example:
        >>> stats = perturbation_size(x, x_adv, norm="l2")
        >>> print(f"Mean L2: {stats['mean']:.4f}")
    """
    delta = (x_adv - x).flatten(1)

    if norm == "linf":
        norms = delta.abs().max(dim=1)[0]
    elif norm == "l2":
        norms = delta.norm(p=2, dim=1)
    else:
        raise ValueError(f"Unknown norm: {norm}")

    return {
        "mean": norms.mean().item(),
        "max": norms.max().item(),
        "median": norms.median().item(),
        "std": norms.std().item(),
    }


def attack_success_rate(
    model: Module,
    x: Tensor,
    y: Tensor,
    x_adv: Tensor,
    target: Optional[Tensor] = None,
) -> float:
    """Compute fraction of successful attacks.

    Args:
        model: Model to evaluate.
        x: Original inputs.
        y: True labels.
        x_adv: Adversarial examples.
        target: Target labels for targeted attacks (optional).

    Returns:
        Attack success rate (fraction where prediction changed).

    Example:
        >>> asr = attack_success_rate(model, x, y, x_adv)
        >>> print(f"Attack success rate: {asr:.2%}")
    """
    model.eval()

    with torch.no_grad():
        pred_orig = model(x).argmax(dim=1)
        pred_adv = model(x_adv).argmax(dim=1)

        # Only consider samples that were originally correct
        originally_correct = pred_orig == y

        if target is not None:
            # Targeted attack: success if prediction matches target
            success = pred_adv == target
        else:
            # Untargeted attack: success if prediction changed
            success = pred_adv != pred_orig

        # Success rate among originally correct samples
        if originally_correct.sum() > 0:
            asr = (
                success & originally_correct
            ).float().sum() / originally_correct.float().sum()
        else:
            asr = 0.0

    return asr.item()


@dataclass
class RobustnessMetrics:
    """Container for robustness evaluation metrics."""

    natural_accuracy: float = 0.0
    robust_accuracy: float = 0.0
    attack_success_rate: float = 0.0
    mean_perturbation: float = 0.0
    median_perturbation: float = 0.0
    max_perturbation: float = 0.0
    certified_accuracy: Optional[float] = None
    certified_radius: Optional[float] = None
    per_attack_results: Dict[str, Dict[str, float]] = field(default_factory=dict)


class RobustnessBenchmark:
    """Comprehensive robustness evaluation suite.

    Evaluates model robustness against multiple attacks and computes
    various metrics.

    Args:
        attacks: Dictionary mapping attack names to attack instances.
        epsilons: List of epsilon values to test.

    Example:
        >>> benchmark = RobustnessBenchmark({
        ...     "FGSM": FGSM(epsilon=0.03),
        ...     "PGD": PGD(epsilon=0.03, steps=40),
        ... })
        >>> results = benchmark.evaluate(model, test_loader)
        >>> benchmark.print_summary(results)
    """

    def __init__(
        self,
        attacks: Optional[Dict[str, Attack]] = None,
        epsilons: Optional[List[float]] = None,
    ):
        if attacks is None:
            self.attacks = {
                "FGSM": FGSM(epsilon=0.03),
                "PGD": PGD(epsilon=0.03, steps=40),
                "PGD-L2": PGDL2(epsilon=1.0, steps=40),
            }
        else:
            self.attacks = attacks

        self.epsilons = epsilons or [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]

    def evaluate(
        self,
        model: Module,
        data_loader: DataLoader,
        device: str = "cuda",
    ) -> Dict[str, RobustnessMetrics]:
        """Run comprehensive robustness evaluation.

        Args:
            model: Model to evaluate.
            data_loader: Test data loader.
            device: Device.

        Returns:
            Dictionary mapping attack names to metrics.
        """
        model.to(device).eval()

        results = {}

        for attack_name, attack in self.attacks.items():
            print(f"\nEvaluating {attack_name}...")

            all_x = []
            all_y = []
            all_x_adv = []

            for x, y in data_loader:
                x, y = x.to(device), y.to(device)

                with torch.no_grad():
                    x_adv = attack(model, x, y)

                all_x.append(x)
                all_y.append(y)
                all_x_adv.append(x_adv)

            x = torch.cat(all_x)
            y = torch.cat(all_y)
            x_adv = torch.cat(all_x_adv)

            # Compute metrics
            metrics = RobustnessMetrics()

            with torch.no_grad():
                # Natural accuracy
                pred = model(x).argmax(dim=1)
                metrics.natural_accuracy = (pred == y).float().mean().item()

                # Robust accuracy
                pred_adv = model(x_adv).argmax(dim=1)
                metrics.robust_accuracy = (pred_adv == y).float().mean().item()

            # Attack success rate
            metrics.attack_success_rate = attack_success_rate(model, x, y, x_adv)

            # Perturbation statistics
            pert_stats = perturbation_size(x, x_adv, norm="linf")
            metrics.mean_perturbation = pert_stats["mean"]
            metrics.median_perturbation = pert_stats["median"]
            metrics.max_perturbation = pert_stats["max"]

            results[attack_name] = metrics

        return results

    def evaluate_robustness_curve(
        self,
        model: Module,
        data_loader: DataLoader,
        attack_class: Type[Attack] = PGD,
        device: str = "cuda",
    ) -> Dict[float, float]:
        """Evaluate robust accuracy across epsilon values.

        Args:
            model: Model to evaluate.
            data_loader: Test data loader.
            attack_class: Attack class to use.
            device: Device.

        Returns:
            Dictionary mapping epsilon to robust accuracy.
        """
        model.to(device).eval()

        curve = {}

        for eps in self.epsilons:
            print(f"Evaluating epsilon={eps:.3f}...")

            attack = attack_class(epsilon=eps, steps=40)

            total_correct = 0
            total_samples = 0

            for x, y in data_loader:
                x, y = x.to(device), y.to(device)

                with torch.no_grad():
                    x_adv = attack(model, x, y)
                    pred = model(x_adv).argmax(dim=1)
                    total_correct += (pred == y).sum().item()
                    total_samples += y.size(0)

            curve[eps] = total_correct / total_samples

        return curve

    def print_summary(self, results: Dict[str, RobustnessMetrics]) -> None:
        """Print formatted summary of results.

        Args:
            results: Results dictionary from evaluate().
        """
        print("\n" + "=" * 80)
        print("ROBUSTNESS EVALUATION SUMMARY")
        print("=" * 80)

        for attack_name, metrics in results.items():
            print(f"\n{attack_name}:")
            print(f"  Natural Accuracy:      {metrics.natural_accuracy:.2%}")
            print(f"  Robust Accuracy:       {metrics.robust_accuracy:.2%}")
            print(f"  Attack Success Rate:   {metrics.attack_success_rate:.2%}")
            print(f"  Mean Perturbation:     {metrics.mean_perturbation:.4f}")
            print(f"  Median Perturbation:   {metrics.median_perturbation:.4f}")
            print(f"  Max Perturbation:      {metrics.max_perturbation:.4f}")

            if metrics.certified_accuracy is not None:
                print(f"  Certified Accuracy:    {metrics.certified_accuracy:.2%}")
            if metrics.certified_radius is not None:
                print(f"  Certified Radius:      {metrics.certified_radius:.4f}")

        print("\n" + "=" * 80)


# =============================================================================
# Visualization
# =============================================================================


def visualize_attack(
    x: Tensor,
    x_adv: Tensor,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """Visualize original, adversarial, and perturbation.

    Args:
        x: Original input (single image or batch).
        x_adv: Adversarial example.
        title: Optional title for the figure.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Example:
        >>> visualize_attack(x[0], x_adv[0], title="PGD Attack")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")

    # Handle batch dimension
    if x.ndim == 4:
        x = x[0]
        x_adv = x_adv[0]

    # Convert to numpy and move channel to end
    x_np = x.detach().cpu().numpy().transpose(1, 2, 0)
    x_adv_np = x_adv.detach().cpu().numpy().transpose(1, 2, 0)
    perturbation = (x_adv_np - x_np + 1) / 2  # Scale to [0, 1]

    # Clip for visualization
    x_np = np.clip(x_np, 0, 1)
    x_adv_np = np.clip(x_adv_np, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(x_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(x_adv_np)
    axes[1].set_title("Adversarial")
    axes[1].axis("off")

    axes[2].imshow(perturbation)
    axes[2].set_title("Perturbation (scaled)")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_perturbation_distribution(
    x: Tensor,
    x_adv: Tensor,
    norm: NormType = "linf",
    bins: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """Plot histogram of perturbation magnitudes.

    Args:
        x: Original inputs.
        x_adv: Adversarial examples.
        norm: Norm type.
        bins: Number of histogram bins.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Example:
        >>> plot_perturbation_distribution(x, x_adv, norm="l2")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")

    delta = (x_adv - x).flatten(1)

    if norm == "linf":
        norms = delta.abs().max(dim=1)[0].cpu().numpy()
        xlabel = "L-infinity Norm"
    elif norm == "l2":
        norms = delta.norm(p=2, dim=1).cpu().numpy()
        xlabel = "L2 Norm"
    else:
        raise ValueError(f"Unknown norm: {norm}")

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(norms, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {norm.upper()} Perturbation Magnitudes")
    ax.axvline(
        norms.mean(), color="r", linestyle="--", label=f"Mean: {norms.mean():.4f}"
    )
    ax.axvline(
        np.median(norms),
        color="g",
        linestyle="--",
        label=f"Median: {np.median(norms):.4f}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_robustness_curve(
    epsilons: List[float],
    accuracies: List[float],
    attack_name: str = "PGD",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """Plot accuracy vs epsilon curve.

    Args:
        epsilons: List of epsilon values.
        accuracies: Corresponding accuracies.
        attack_name: Name of attack for title.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Example:
        >>> epsilons = [0.0, 0.01, 0.02, 0.03]
        >>> accuracies = [0.95, 0.85, 0.70, 0.55]
        >>> plot_robustness_curve(epsilons, accuracies, "PGD")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(epsilons, accuracies, "b-o", linewidth=2, markersize=8)
    ax.set_xlabel("Epsilon (Perturbation Budget)")
    ax.set_ylabel("Robust Accuracy")
    ax.set_title(f"Robustness Curve: {attack_name}")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # Add accuracy labels
    for eps, acc in zip(epsilons, accuracies):
        ax.annotate(
            f"{acc:.2%}",
            (eps, acc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# =============================================================================
# Integration with fishstick Trainer
# =============================================================================


class AdversarialTrainerMixin:
    """Mixin to add adversarial training to fishstick Trainer.

    Example:
        >>> from fishstick.trainer import Trainer
        >>>
        >>> class RobustTrainer(Trainer, AdversarialTrainerMixin):
        ...     pass
        >>>
        >>> trainer = RobustTrainer(model)
        >>> trainer.fit_adversarial(train_loader, attack=PGD(epsilon=0.03))
    """

    def fit_adversarial(
        self,
        train_loader: DataLoader,
        attack: Attack,
        epochs: int = 10,
        alpha: float = 1.0,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[LossFunction] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> Dict[str, List[float]]:
        """Train model with adversarial training.

        Args:
            train_loader: Training data loader.
            attack: Attack to use for generating adversarial examples.
            epochs: Number of epochs.
            alpha: Weight for adversarial loss.
            optimizer: Optimizer.
            loss_fn: Loss function.
            val_loader: Validation data loader.
            device: Device.

        Returns:
            Training history.
        """
        adversarial_trainer = AdversarialTraining(
            self.model,
            attack,
            alpha=alpha,
        )

        return adversarial_trainer.fit(
            train_loader,
            epochs,
            optimizer,
            loss_fn,
            val_loader,
            device,
        )

    def fit_trades(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        beta: float = 6.0,
        epsilon: float = 0.03,
        step_size: float = 0.01,
        num_steps: int = 10,
        optimizer: Optional[Optimizer] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> Dict[str, List[float]]:
        """Train model with TRADES.

        Args:
            train_loader: Training data loader.
            epochs: Number of epochs.
            beta: TRADES trade-off parameter.
            epsilon: Perturbation budget.
            step_size: Step size for attack.
            num_steps: Number of attack steps.
            optimizer: Optimizer.
            val_loader: Validation data loader.
            device: Device.

        Returns:
            Training history.
        """
        trades_trainer = TRADES(
            self.model,
            beta=beta,
            epsilon=epsilon,
            step_size=step_size,
            num_steps=num_steps,
        )

        return trades_trainer.fit(
            train_loader,
            epochs,
            optimizer,
            val_loader,
            device,
        )

    def evaluate_robustness(
        self,
        data_loader: DataLoader,
        attacks: Optional[Dict[str, Attack]] = None,
        device: str = "cuda",
    ) -> Dict[str, RobustnessMetrics]:
        """Evaluate model robustness.

        Args:
            data_loader: Test data loader.
            attacks: Dictionary of attacks to evaluate.
            device: Device.

        Returns:
            Dictionary of metrics per attack.
        """
        benchmark = RobustnessBenchmark(attacks)
        return benchmark.evaluate(self.model, data_loader, device)


# =============================================================================
# Utility Classes
# =============================================================================


class AttackEnsemble:
    """Ensemble multiple attacks.

    Tries multiple attacks and returns the best adversarial example.

    Args:
        attacks: List of attacks to ensemble.
        mode: Selection mode ('best' or 'worst').

    Example:
        >>> ensemble = AttackEnsemble([FGSM(0.03), PGD(0.03, steps=40)])
        >>> x_adv = ensemble(model, x, y)
    """

    def __init__(
        self,
        attacks: List[Attack],
        mode: Literal["best", "worst"] = "best",
    ):
        self.attacks = attacks
        self.mode = mode

    def __call__(
        self,
        model: Module,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Generate adversarial examples using ensemble.

        Args:
            model: Model to attack.
            x: Input tensor.
            y: True labels.

        Returns:
            Adversarial examples.
        """
        batch_size = x.shape[0]
        device = x.device

        # Store all adversarial examples
        all_adv = []
        all_scores = []

        for attack in self.attacks:
            x_adv = attack(model, x, y)

            with torch.no_grad():
                output = model(x_adv)
                loss = F.cross_entropy(output, y, reduction="none")

            all_adv.append(x_adv)
            all_scores.append(loss)

        # Stack and select
        all_adv = torch.stack(all_adv)  # (num_attacks, batch_size, ...)
        all_scores = torch.stack(all_scores)  # (num_attacks, batch_size)

        if self.mode == "best":
            # Select attack with highest loss (most successful)
            best_idx = all_scores.argmax(dim=0)
        else:
            # Select attack with lowest loss (least detectable)
            best_idx = all_scores.argmin(dim=0)

        # Gather best adversarial examples
        x_best = torch.zeros_like(x)
        for i in range(batch_size):
            x_best[i] = all_adv[best_idx[i], i]

        return x_best


class AdaptiveAttack:
    """Adaptive attack that adjusts parameters based on success.

    Automatically adjusts attack strength if initial attack fails.

    Args:
        base_attack: Base attack class.
        epsilon_range: Range of epsilon values to try.
        steps_range: Range of step counts to try.

    Example:
        >>> adaptive = AdaptiveAttack(PGD, epsilon_range=[0.01, 0.03, 0.05])
        >>> x_adv = adaptive(model, x, y)
    """

    def __init__(
        self,
        base_attack: Type[Attack],
        epsilon_range: List[float],
        steps_range: Optional[List[int]] = None,
    ):
        self.base_attack = base_attack
        self.epsilon_range = epsilon_range
        self.steps_range = steps_range or [20, 40, 100]

    def __call__(
        self,
        model: Module,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Generate adversarial examples with adaptive strength.

        Args:
            model: Model to attack.
            x: Input tensor.
            y: True labels.

        Returns:
            Adversarial examples.
        """
        x_adv = x.clone()

        for eps in self.epsilon_range:
            for steps in self.steps_range:
                attack = self.base_attack(epsilon=eps, steps=steps)
                x_try = attack(model, x, y)

                with torch.no_grad():
                    pred = model(x_try).argmax(dim=1)
                    success = (pred != y).all()

                if success:
                    return x_try

                # Keep best so far
                with torch.no_grad():
                    pred_current = model(x_adv).argmax(dim=1)
                    pred_try = model(x_try).argmax(dim=1)

                    better = (pred_try != y) & (pred_current == y)
                    x_adv = torch.where(
                        better.view(-1, *([1] * (x.ndim - 1))), x_try, x_adv
                    )

        return x_adv


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_robustness_check(
    model: Module,
    x: Tensor,
    y: Tensor,
    epsilon: float = 0.03,
    device: str = "cuda",
) -> Dict[str, float]:
    """Quick robustness check with standard attacks.

    Args:
        model: Model to evaluate.
        x: Input batch.
        y: Label batch.
        epsilon: Perturbation budget.
        device: Device.

    Returns:
        Dictionary with results for different attacks.
    """
    model.to(device).eval()
    x, y = x.to(device), y.to(device)

    results = {}

    # Natural accuracy
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        results["natural"] = (pred == y).float().mean().item()

    # FGSM
    fgsm = FGSM(epsilon=epsilon)
    results["fgsm"] = robust_accuracy(model, x, y, fgsm)

    # PGD
    pgd = PGD(epsilon=epsilon, steps=40)
    results["pgd"] = robust_accuracy(model, x, y, pgd)

    # PGD-L2
    pgd_l2 = PGDL2(epsilon=epsilon * 10, steps=40)
    results["pgd_l2"] = robust_accuracy(model, x, y, pgd_l2)

    return results


__all__ = [
    # Attack utilities
    "clip_perturbation",
    "normalize_perturbation",
    "project_perturbation",
    "compute_gradient",
    # Attacks
    "Attack",
    "FGSM",
    "PGD",
    "PGDL2",
    "DeepFool",
    "CW",
    "AutoAttack",
    # Defenses
    "Defense",
    "AdversarialTraining",
    "TRADES",
    "InputTransformation",
    "RandomizationDefense",
    "DistillationDefense",
    # Certified defenses
    "RandomizedSmoothing",
    "IntervalBoundPropagation",
    "compute_certified_radius",
    # Evaluation
    "robust_accuracy",
    "perturbation_size",
    "attack_success_rate",
    "RobustnessMetrics",
    "RobustnessBenchmark",
    # Visualization
    "visualize_attack",
    "plot_perturbation_distribution",
    "plot_robustness_curve",
    # Integration
    "AdversarialTrainerMixin",
    # Utilities
    "AttackEnsemble",
    "AdaptiveAttack",
    "quick_robustness_check",
]
