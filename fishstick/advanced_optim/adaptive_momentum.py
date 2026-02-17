"""
Adaptive Momentum Methods

Advanced momentum-based optimization with adaptive damping and eigenvalue-based scaling:
- Heavy-ball method with spectral radius control
- Nesterov momentum with adaptive damping
- Adaptive eigenvalue-based scaling
- Variance-reduced momentum methods

Reference:
- Polyak (1964). Some methods of speeding up the convergence of iteration methods.
- Nesterov (1983). A method for solving the convex programming problem with convergence rate O(1/k^2).
- Hazan et al. (2018). Towards Definite Learning without Forgetfulness.
"""

import math
from typing import Optional, Dict, List
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
import numpy as np
from collections import deque


class AdaptiveMomentumOptimizer(Optimizer):
    """
    Adaptive Momentum Optimizer with Eigenvalue-Based Scaling.

    Adjusts momentum based on local curvature estimates using
    eigenvalue information from gradient covariance.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.001)
        base_momentum: Base momentum coefficient (default: 0.9)
        beta1: Exponential decay rate for momentum (default: 0.9)
        beta2: Exponential decay rate for eigenvalue estimation (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 1e-5)

    Reference:
        - Chen et al. (2019). Closing the Generalization Gap of Adaptive Gradient Methods.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        base_momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 1e-5,
    ):
        defaults = dict(
            lr=lr,
            base_momentum=base_momentum,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data

                if group["weight_decay"] > 0:
                    grad = grad.add(param.data, alpha=group["weight_decay"])

                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param.data)
                    state["exp_avg_sq"] = torch.zeros_like(param.data)
                    state["momentum"] = torch.zeros_like(param.data)

                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                momentum = state["momentum"]

                beta1 = group["beta1"]
                beta2 = group["beta2"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2

                eigenvalue_proxy = exp_avg_sq_corrected.sqrt().add_(group["epsilon"])
                adaptive_beta = 1 - (1 / eigenvalue_proxy.log().abs().add_(1))
                adaptive_beta = adaptive_beta.clamp(0.5, 0.999)

                momentum.mul_(adaptive_beta).add_(exp_avg_corrected)

                param.data.add_(momentum, alpha=-group["lr"])

        return loss


class HeavyBallOptimizer(Optimizer):
    """
    Heavy-Ball Method with Spectral Radius Control.

    The classical heavy-ball method with added spectral radius control
    for improved stability and convergence.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.001)
        momentum: Momentum coefficient (default: 0.9)
        spectral_radius: Target spectral radius (default: 0.99)
        damping: Damping factor for momentum (default: 0.1)
        weight_decay: Weight decay (default: 1e-4)

    Reference:
        - Polyak (1964). Some methods of speeding up the convergence of iteration methods.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        momentum: float = 0.9,
        spectral_radius: float = 0.99,
        damping: float = 0.1,
        weight_decay: float = 1e-4,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            spectral_radius=spectral_radius,
            damping=damping,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data

                if group["weight_decay"] > 0:
                    grad = grad.add(param.data, alpha=group["weight_decay"])

                state = self.state[param]

                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(param.data)

                momentum = state["momentum"]

                grad_norm = grad.norm()
                if grad_norm > 0:
                    spectral_lr = group["lr"] * (
                        group["spectral_radius"] / (1 + grad_norm)
                    )
                else:
                    spectral_lr = group["lr"]

                momentum.mul_(group["momentum"]).add_(
                    grad, alpha=-(1 - group["damping"]) * spectral_lr
                )
                param.data.add_(momentum, alpha=1 - group["damping"])

        return loss


class NesterovMomentumOptimizer(Optimizer):
    """
    Nesterov Momentum with Adaptive Damping.

    Nesterov accelerated gradient with damping that adapts based on
    local gradient variability.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.001)
        momentum: Base momentum coefficient (default: 0.9)
        adaptive_damping: Whether to use adaptive damping (default: True)
        min_damping: Minimum damping coefficient (default: 0.1)
        max_damping: Maximum damping coefficient (default: 0.9)
        window_size: Window for gradient variability estimation (default: 10)
        weight_decay: Weight decay (default: 1e-4)

    Reference:
        - Nesterov (1983). A method for solving the convex programming problem.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        momentum: float = 0.9,
        adaptive_damping: bool = True,
        min_damping: float = 0.1,
        max_damping: float = 0.9,
        window_size: int = 10,
        weight_decay: float = 1e-4,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            adaptive_damping=adaptive_damping,
            min_damping=min_damping,
            max_damping=max_damping,
            window_size=window_size,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            grad_variance_sum = 0.0
            grad_count = 0

            if group["adaptive_damping"]:
                for param in group["params"]:
                    if param.grad is not None:
                        state = self.state[param]
                        if "grad_history" not in state:
                            state["grad_history"] = deque(maxlen=group["window_size"])

                        grad_norm = param.grad.data.norm().item()
                        state["grad_history"].append(grad_norm)
                        grad_count += 1

                        if len(state["grad_history"]) > 1:
                            history = list(state["grad_history"])
                            grad_variance_sum += np.var(history)

                if grad_count > 0 and grad_variance_sum > 0:
                    avg_variance = grad_variance_sum / grad_count
                    damping = group["min_damping"] + (
                        group["max_damping"] - group["min_damping"]
                    ) * min(1.0, avg_variance)
                else:
                    damping = group["momentum"]
            else:
                damping = group["momentum"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data

                if group["weight_decay"] > 0:
                    grad = grad.add(param.data, alpha=group["weight_decay"])

                state = self.state[param]

                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(param.data)

                momentum = state["momentum"]

                momentum.mul_(damping).add_(grad)

                param.data.add_(grad, alpha=-group["lr"])
                param.data.add_(momentum, alpha=-group["lr"])

        return loss


class VarianceReducedMomentum(Optimizer):
    """
    Variance-Reduced Momentum Optimizer.

    Combines momentum with variance reduction techniques for more
    stable convergence in stochastic optimization.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.001)
        momentum: Momentum coefficient (default: 0.9)
        variance_reduction: Variance reduction strength (default: 0.5)
        weight_decay: Weight decay (default: 1e-4)

    Reference:
        - Johnson & Zhang (2013). Accelerating Stochastic Gradient Descent.
        - Defazio et al. (2014). SAGA: A Fast Incremental Gradient Method.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        momentum: float = 0.9,
        variance_reduction: float = 0.5,
        weight_decay: float = 1e-4,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            variance_reduction=variance_reduction,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.reference_grads = {}

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data

                if group["weight_decay"] > 0:
                    grad = grad.add(param.data, alpha=group["weight_decay"])

                state = self.state[param]

                if id(param) not in self.reference_grads:
                    self.reference_grads[id(param)] = torch.zeros_like(param.data)

                reference_grad = self.reference_grads[id(param)]

                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(param.data)

                momentum = state["momentum"]

                variance_corrected = (
                    grad - reference_grad + state.get("reference_mean", grad)
                )

                momentum.mul_(group["momentum"]).add_(variance_corrected)

                reference_grad.copy_(grad)

                param.data.add_(momentum, alpha=-group["lr"])

                state["reference_mean"] = (
                    state.get("reference_mean", torch.zeros_like(param.data))
                    .mul_(0.99)
                    .add_(grad, alpha=0.01)
                )

        return loss


class PadagradMomentum(Optimizer):
    """
    Parameter-wise Adaptive Gradient with Momentum.

    Adapts learning rates per parameter based on gradient history
    while maintaining momentum.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.01)
        momentum: Momentum coefficient (default: 0.9)
        eps: Small constant for numerical stability (default: 1e-10)
        weight_decay: Weight decay (default: 1e-4)

    Reference:
        - Duchi et al. (2011). Adaptive Subgradient Methods.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        eps: float = 1e-10,
        weight_decay: float = 1e-4,
    ):
        defaults = dict(lr=lr, momentum=momentum, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data

                if group["weight_decay"] > 0:
                    grad = grad.add(param.data, alpha=group["weight_decay"])

                state = self.state[param]

                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(param.data)
                    state["sum_squares"] = torch.zeros_like(param.data)

                momentum = state["momentum"]
                sum_squares = state["sum_squares"]

                grad_squared = grad.pow(2)
                sum_squares.add_(grad_squared)

                rms = sum_squares.sqrt().add_(group["eps"])

                momentum.mul_(group["momentum"]).add_(grad / rms)

                param.data.add_(momentum, alpha=-group["lr"])

        return loss


class ElasticMomentum(Optimizer):
    """
    Elastic Momentum Optimizer.

    Combines multiple momentum estimates with an elastic mechanism
    that balances exploration and exploitation.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.001)
        fast_momentum: Fast momentum coefficient (default: 0.95)
        slow_momentum: Slow momentum coefficient (default: 0.8)
        elasticity: Balance between fast and slow (default: 0.5)
        weight_decay: Weight decay (default: 1e-4)
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        fast_momentum: float = 0.95,
        slow_momentum: float = 0.8,
        elasticity: float = 0.5,
        weight_decay: float = 1e-4,
    ):
        defaults = dict(
            lr=lr,
            fast_momentum=fast_momentum,
            slow_momentum=slow_momentum,
            elasticity=elasticity,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data

                if group["weight_decay"] > 0:
                    grad = grad.add(param.data, alpha=group["weight_decay"])

                state = self.state[param]

                if len(state) == 0:
                    state["fast_momentum"] = torch.zeros_like(param.data)
                    state["slow_momentum"] = torch.zeros_like(param.data)

                fast = state["fast_momentum"]
                slow = state["slow_momentum"]

                fast.mul_(group["fast_momentum"]).add_(grad)
                slow.mul_(group["slow_momentum"]).add_(grad)

                combined = fast.mul_(group["elasticity"]) + slow.mul_(
                    1 - group["elasticity"]
                )

                param.data.add_(combined, alpha=-group["lr"])

        return loss


__all__ = [
    "AdaptiveMomentumOptimizer",
    "HeavyBallOptimizer",
    "NesterovMomentumOptimizer",
    "VarianceReducedMomentum",
    "PadagradMomentum",
    "ElasticMomentum",
]
