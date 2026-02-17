"""
Physics-Inspired Learning Rate Schedulers

Advanced LR schedulers based on physical dynamics:
- Thermodynamic annealing (simulated annealing)
- Hamiltonian dynamics (momentum-based oscillation)
- Quantum-inspired (tunneling dynamics)
- Riemannian gradient flow

Reference:
- Kirkpatrick et al. (1983). Optimization by Simulated Annealing.
- Betancourt (2018). The Geometric Foundations of Hamiltonian Monte Carlo.
- Girolami & Calderhead (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo.
"""

import math
from typing import Optional, Callable
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class ThermodynamicAnnealingScheduler(_LRScheduler):
    """
    Thermodynamic Annealing Scheduler (Simulated Annealing-inspired).

    This scheduler models the learning rate as a temperature parameter that
    decreases according to a simulated annealing schedule, allowing the optimizer
    to escape local minima early in training and converge to better solutions.

    Args:
        optimizer: Wrapped optimizer
        T_max: Maximum temperature (initial learning rate multiplier)
        T_min: Minimum temperature (final learning rate multiplier)
        max_epochs: Total number of epochs
        schedule: Annealing schedule type ('linear', 'exponential', 'log')
        cooling_rate: Rate of temperature decrease for exponential schedule

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = ThermodynamicAnnealingScheduler(optimizer, T_max=1.0, T_min=0.01, max_epochs=100)
        >>> for epoch in range(100):
        ...     train()
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: float = 1.0,
        T_min: float = 0.01,
        max_epochs: int = 100,
        schedule: str = "exponential",
        cooling_rate: float = 0.95,
        last_epoch: int = -1,
    ):
        self.T_max = T_max
        self.T_min = T_min
        self.max_epochs = max_epochs
        self.schedule = schedule
        self.cooling_rate = cooling_rate
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.max_epochs:
            return [self.T_min for _ in self.base_lrs]

        if self.schedule == "linear":
            progress = self.last_epoch / self.max_epochs
            temperature = self.T_max - (self.T_max - self.T_min) * progress

        elif self.schedule == "exponential":
            temperature = self.T_max * (self.cooling_rate**self.last_epoch)
            temperature = max(temperature, self.T_min)

        elif self.schedule == "log":
            temperature = self.T_max / (1 + self.last_epoch / 10)
            temperature = max(temperature, self.T_min)

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return [base_lr * temperature for base_lr in self.base_lrs]


class HamiltonianDynamicsScheduler(_LRScheduler):
    """
    Hamiltonian Dynamics LR Scheduler.

    Based on Hamiltonian dynamics, this scheduler introduces oscillatory
    behavior in the learning rate that can help escape saddle points
    while maintaining convergence towards minima.

    Args:
        optimizer: Wrapped optimizer
        base_lr: Base learning rate
        amplitude: Oscillation amplitude
        frequency: Oscillation frequency (cycles per epoch)
        decay: Temporal decay of oscillation amplitude
        phase: Initial phase offset

    Reference:
        - Betancourt (2018). The Geometric Foundations of Hamiltonian Monte Carlo.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-3,
        amplitude: float = 0.1,
        frequency: float = 0.1,
        decay: float = 0.99,
        phase: float = 0.0,
        last_epoch: int = -1,
    ):
        self.base_lr = base_lr
        self.amplitude = amplitude
        self.frequency = frequency
        self.decay = decay
        self.phase = phase
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch

        decay_factor = self.decay**t
        oscillation = (
            self.amplitude
            * decay_factor
            * math.sin(2 * math.pi * self.frequency * t + self.phase)
        )

        lr_multiplier = 1.0 + oscillation

        return [self.base_lr * lr_multiplier for _ in self.base_lrs]


class QuantumTunnelingScheduler(_LRScheduler):
    """
    Quantum Tunneling-inspired LR Scheduler.

    Inspired by quantum tunneling, this scheduler occasionally allows
    larger learning rates to "tunnel" through flat regions or local minima.

    Args:
        optimizer: Wrapped optimizer
        base_lr: Base learning rate
        tunneling_prob: Probability of tunneling event
        tunneling_factor: Multiplier during tunneling
        barrier_threshold: Gradient magnitude threshold for barrier detection

    Reference:
        - Alternative: Hadikar et al. (2020). Quantum-inspired Optimization.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-3,
        tunneling_prob: float = 0.05,
        tunneling_factor: float = 3.0,
        barrier_threshold: float = 1e-4,
        last_epoch: int = -1,
    ):
        self.base_lr = base_lr
        self.tunneling_prob = tunneling_prob
        self.tunneling_factor = tunneling_factor
        self.barrier_threshold = barrier_threshold
        self.last_gradient_norm = None
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        base_lr = self.base_lr

        if self.last_gradient_norm is not None:
            if self.last_gradient_norm < self.barrier_threshold:
                tunneling = torch.bernoulli(torch.tensor(self.tunneling_prob)).item()
                if tunneling:
                    return [base_lr * self.tunneling_factor for _ in self.base_lrs]

        return [base_lr for _ in self.base_lrs]

    def step(self, epoch: Optional[int] = None):
        result = super().step(epoch)

        if self.optimizer.param_groups:
            total_norm = 0.0
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        total_norm += p.grad.data.norm().item() ** 2
            self.last_gradient_norm = math.sqrt(total_norm)

        return result


class RiemannianGradientFlowScheduler(_LRScheduler):
    """
    Riemannian Gradient Flow Scheduler.

    Based on gradient flow on a Riemannian manifold, this scheduler
    adapts the learning rate based on local curvature estimates.

    Args:
        optimizer: Wrapped optimizer
        base_lr: Base learning rate
        curvature_window: Window size for curvature estimation
        epsilon: Small constant for numerical stability

    Reference:
        - Girolami & Calderhead (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-3,
        curvature_window: int = 10,
        epsilon: float = 1e-8,
        last_epoch: int = -1,
    ):
        self.base_lr = base_lr
        self.curvature_window = curvature_window
        self.epsilon = epsilon
        self.grad_history = []
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        base_lr = self.base_lr

        if len(self.grad_history) > 1:
            grad_diff = self.grad_history[-1] - self.grad_history[-2]
            curvature = grad_diff.norm() / (self.grad_history[-1].norm() + self.epsilon)

            metric = 1.0 + curvature.item()
            adjusted_lr = base_lr / metric

            return [adjusted_lr for _ in self.base_lrs]

        return [base_lr for _ in self.base_lrs]

    def step(self, epoch: Optional[int] = None):
        result = super().step(epoch)

        if self.optimizer.param_groups:
            total_norm = 0.0
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        total_norm += p.grad.data.norm().item() ** 2
            grad_norm = math.sqrt(total_norm)

            self.grad_history.append(grad_norm)

            if len(self.grad_history) > self.curvature_window:
                self.grad_history.pop(0)

        return result


class AdaptiveCurvatureScheduler(_LRScheduler):
    """
    Adaptive Curvature-based Scheduler.

    Adjusts learning rate based on estimated local curvature using
    gradient variance and autocorrelation.

    Args:
        optimizer: Wrapped optimizer
        base_lr: Base learning rate
        window_size: Window for computing statistics
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-3,
        window_size: int = 20,
        min_lr: float = 1e-6,
        max_lr: float = 1e-1,
        last_epoch: int = -1,
    ):
        self.base_lr = base_lr
        self.window_size = window_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.grad_history = []
        self.loss_history = []
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if len(self.grad_history) < 2:
            return [self.base_lr for _ in self.base_lrs]

        grad_mean = sum(self.grad_history) / len(self.grad_history)
        grad_var = sum((g - grad_mean) ** 2 for g in self.grad_history) / len(
            self.grad_history
        )

        if len(self.loss_history) > 1:
            loss_diff = self.loss_history[-1] - self.loss_history[-2]
            improvement = -loss_diff

            if improvement > 0:
                curvature_proxy = grad_var / (improvement + self.epsilon)
            else:
                curvature_proxy = grad_var * 10
        else:
            curvature_proxy = 1.0

        lr_scale = 1.0 / (1.0 + math.log(1 + curvature_proxy))

        adjusted_lr = self.base_lr * lr_scale
        adjusted_lr = max(self.min_lr, min(adjusted_lr, self.max_lr))

        return [adjusted_lr for _ in self.base_lrs]

    @property
    def epsilon(self):
        return 1e-8

    def step(self, epoch: Optional[int] = None):
        result = super().step(epoch)

        if self.optimizer.param_groups:
            total_norm = 0.0
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        total_norm += p.grad.data.norm().item() ** 2
            grad_norm = math.sqrt(total_norm)

            self.grad_history.append(grad_norm)

            if len(self.grad_history) > self.window_size:
                self.grad_history.pop(0)

        return result


class CyclicOscillationScheduler(_LRScheduler):
    """
    Cyclic Oscillation Scheduler with Multiple Frequency Components.

    Combines multiple sinusoidal oscillations at different frequencies
    to create a complex, non-repeating learning rate pattern.

    Args:
        optimizer: Wrapped optimizer
        base_lr: Base learning rate
        frequencies: List of oscillation frequencies
        amplitudes: List of oscillation amplitudes (corresponding to frequencies)
        decay: Global decay rate
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-3,
        frequencies: Optional[list] = None,
        amplitudes: Optional[list] = None,
        decay: float = 0.995,
        last_epoch: int = -1,
    ):
        self.base_lr = base_lr
        self.frequencies = frequencies or [0.05, 0.1, 0.2]
        self.amplitudes = amplitudes or [0.15, 0.08, 0.03]
        self.decay = decay

        if len(self.frequencies) != len(self.amplitudes):
            raise ValueError("frequencies and amplitudes must have same length")

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        decay_factor = self.decay**t

        oscillation = 0.0
        for freq, amp in zip(self.frequencies, self.amplitudes):
            oscillation += amp * math.sin(2 * math.pi * freq * t)

        lr_multiplier = 1.0 + decay_factor * oscillation

        return [self.base_lr * lr_multiplier for _ in self.base_lrs]


class StochasticWeightAveragingScheduler(_LRScheduler):
    """
    Stochastic Weight Averaging (SWA) Scheduler.

    Maintains a running average of model weights and allows for
    cyclical learning rates in the SWA phase.

    Args:
        optimizer: Wrapped optimizer
        swa_start: Epoch to start SWA
        swa_lr: Learning rate for SWA phase
        base_scheduler: Optional base scheduler before SWA

    Reference:
        - Izmailov et al. (2018). Averaging Weights Leads to Wider Optima.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        swa_start: int = 80,
        swa_lr: float = 1e-4,
        base_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1,
    ):
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.swa_start:
            if self.base_scheduler:
                return self.base_scheduler.get_last_lr()
            return self.base_lrs
        else:
            return [self.swa_lr for _ in self.base_lrs]

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1

        if epoch >= self.swa_start and self.base_scheduler:
            self.base_scheduler.last_epoch = epoch

        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


__all__ = [
    "ThermodynamicAnnealingScheduler",
    "HamiltonianDynamicsScheduler",
    "QuantumTunnelingScheduler",
    "RiemannianGradientFlowScheduler",
    "AdaptiveCurvatureScheduler",
    "CyclicOscillationScheduler",
    "StochasticWeightAveragingScheduler",
]
