"""
Second-Order Optimization Methods

Advanced optimizers utilizing curvature information for improved convergence:
- K-FAC: Kronecker-Factored Approximate Curvature
- ESGD: Equilibrium SGD
- Shampoo: Preconditioned Gradient Descent

Reference:
- Martens & Grosse (2015). Optimizing Neural Networks with Kronecker-factored Approximate Curvature.
- Zhang et al. (2019). A Simple Bayesian Approach to Learning Curves.
- Anil et al. (2019). Scalable Second-Order Optimization for Deep Learning.
"""

from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
import math
from collections import defaultdict


class KFACOptimizer(Optimizer):
    """
    Kronecker-Factored Approximate Curvature (K-FAC) Optimizer.

    K-FAC computes an approximation to the Fisher information matrix using
    Kronecker products, allowing for efficient preconditioning of gradients.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.001)
        momentum: Momentum factor (default: 0.9)
        weight_decay: Weight decay (L2 penalty) (default: 1e-5)
        stat_decay: Decay factor for running statistics (default: 0.95)
        damping: Damping factor for numerical stability (default: 0.001)
        grad_averaging: Whether to use gradient averaging (default: True)
        N: Number of data points for Fisher estimation (default: 1024)

    Example:
        >>> model = nn.Linear(10, 2)
        >>> optimizer = KFACOptimizer(model.parameters(), lr=0.001)
        >>> for input, target in data:
        ...     output = model(input)
        ...     loss = loss_fn(output, target)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 1e-5,
        stat_decay: float = 0.95,
        damping: float = 0.001,
        grad_averaging: bool = True,
        N: int = 1024,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            stat_decay=stat_decay,
            damping=damping,
            grad_averaging=grad_averaging,
            N=N,
        )
        super().__init__(params, defaults)

    def _compute_fisher(
        self, param: nn.Parameter, inputs: List[Tensor], outputs: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Compute Fisher information matrix approximation for a parameter."""
        param_shape = param.shape

        if len(param_shape) == 1:
            return None, None

        if len(param_shape) == 2:
            in_features, out_features = param_shape

            grad_out = torch.stack([o.grad for o in outputs if o.grad is not None])
            if len(grad_out) == 0:
                return None, None

            grad_out_mean = grad_out.mean(0)

            act = torch.stack([i for i in inputs if i.requires_grad])
            if len(act) == 0:
                return None, None
            act_mean = act.mean(0)

            a_aT = torch.matmul(act_mean.unsqueeze(1), act_mean.unsqueeze(2)).squeeze()
            g_gT = torch.matmul(
                grad_out_mean.unsqueeze(1), grad_out_mean.unsqueeze(2)
            ).squeeze()

            return a_aT, g_gT

        return None, None

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            group_params = []

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(param.grad)

                    if param.ndim == 2:
                        state["A"] = torch.eye(param.shape[0], device=param.device)
                        state["G"] = torch.eye(param.shape[1], device=param.device)

                state["step"] += 1

                d = group["damping"]
                momentum = group["momentum"]
                stat_decay = group["stat_decay"]

                grad = param.grad.data

                if group["weight_decay"] > 0:
                    grad = grad.add(param.data, alpha=group["weight_decay"])

                if param.ndim == 2:
                    A = state["A"]
                    G = state["G"]

                    A.mul_(stat_decay)
                    G.mul_(stat_decay)

                    out_features, in_features = param.shape

                    if grad.shape == param.shape:
                        grad_reshaped = grad.view(out_features, in_features)

                        g = grad_reshaped.mean(dim=0)
                        a = grad_reshaped.mean(dim=1)

                        if g.shape[0] == G.shape[0] and a.shape[0] == A.shape[0]:
                            A_new = torch.outer(a, a)
                            G_new = torch.outer(g, g)

                            A.add_(A_new, alpha=(1 - stat_decay))
                            G.add_(G_new, alpha=(1 - stat_decay))

                    A_inv = torch.linalg.inv(
                        A + d * torch.eye(A.shape[0], device=A.device)
                    )
                    G_inv = torch.linalg.inv(
                        G + d * torch.eye(G.shape[0], device=A.device)
                    )

                    precond_grad = torch.matmul(torch.matmul(A_inv, grad), G_inv)
                else:
                    precond_grad = grad

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(precond_grad)

                param.data.add_(buf, alpha=-group["lr"])

        return loss


class ESGDOptimizer(Optimizer):
    """
    Equilibrium Stochastic Gradient Descent (ESGD) Optimizer.

    ESGD uses a global equilibrium condition to adaptively adjust learning rates
    based on the ratio of gradient variance to expected improvement.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.01)
        beta: Momentum coefficient (default: 0.9)
        rho: Equilibrium threshold (default: 0.5)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay (default: 1e-4)

    Reference:
        - Zhang et al. (2019). A Simple Bayesian Approach to Learning Curves.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        beta: float = 0.9,
        rho: float = 0.5,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ):
        defaults = dict(lr=lr, beta=beta, rho=rho, eps=eps, weight_decay=weight_decay)
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
                    state["exp_sq_avg"] = torch.zeros_like(param.data)
                    state["delta"] = torch.zeros_like(param.data)

                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_sq_avg = state["exp_sq_avg"]
                delta = state["delta"]

                beta = group["beta"]
                rho = group["rho"]
                eps = group["eps"]

                exp_avg.mul_(beta).add_(grad, alpha=1 - beta)
                exp_sq_avg.mul_(beta).add_(grad.pow(2), alpha=1 - beta)

                variance = exp_sq_avg - exp_avg.pow(2)

                lr = group["lr"]
                equilibrium = rho * (exp_sq_avg.sqrt() + eps) / (variance.sqrt() + eps)
                lr_tensor = lr * torch.clamp(equilibrium, min=0.01, max=100)
                lr_scalar = lr_tensor.item() if lr_tensor.numel() == 1 else lr

                delta.mul_(beta).add_(grad, alpha=lr_scalar)
                param.data.add_(delta, alpha=-1)

        return loss


class ShampooOptimizer(Optimizer):
    """
    Shampoo (Preconditioned Gradient Descent) Optimizer.

    Shampoo maintains per-layer preconditioning matrices computed from
    Kronecker products of activations and gradients.

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.001)
        momentum: Momentum factor (default: 0.9)
        weight_decay: Weight decay (default: 1e-4)
        beta: Exponential moving average coefficient (default: 0.5)
        gamma: Matrix exponent (default: 0.5)
        damping: Damping for numerical stability (default: 1e-6)

    Reference:
        - Anil et al. (2019). Scalable Second-Order Optimization for Deep Learning.
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        beta: float = 0.5,
        gamma: float = 0.5,
        damping: float = 1e-6,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            beta=beta,
            gamma=gamma,
            damping=damping,
        )
        super().__init__(params, defaults)

    def _compute_preconditioner(self, grad: Tensor, param: Tensor) -> Dict[str, Tensor]:
        """Compute preconditioning matrices for a parameter."""
        if grad.ndim < 2:
            return {}

        param_shape = param.shape
        precond = {}

        if len(param_shape) == 4:
            out_c, in_c, kh, kw = param_shape

            grad_reshaped = grad.reshape(out_c, in_c * kh * kw)
            grad_outer = grad_reshaped @ grad_reshaped.T
            precond["O"] = grad_outer
            precond["I"] = (
                grad.reshape(out_c * kh * kw, in_c)
                @ grad.reshape(out_c * kh * kw, in_c).T
            )

        elif len(param_shape) == 2:
            out_f, in_f = param_shape
            grad_outer = grad @ grad.T
            precond["O"] = grad_outer

            grad_outer_in = grad.T @ grad
            precond["I"] = grad_outer_in

        return precond

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
                    state["momentum"] = torch.zeros_like(param)

                    param_shape = param.shape
                    if len(param_shape) == 4:
                        out_c, in_c, kh, kw = param_shape
                        state["precond_O"] = torch.eye(out_c, device=param.device)
                        state["precond_I"] = torch.eye(
                            in_c * kh * kw, device=param.device
                        )
                    elif len(param_shape) == 2:
                        out_f, in_f = param_shape
                        state["precond_O"] = torch.eye(out_f, device=param.device)
                        state["precond_I"] = torch.eye(in_f, device=param.device)

                state["step"] += 1

                beta = group["beta"]
                gamma = group["gamma"]
                damping = group["damping"]

                if len(param.shape) == 4:
                    out_c, in_c, kh, kw = param.shape
                    grad_reshaped = grad.reshape(out_c, in_c * kh * kw)

                    grad_outer_o = grad_reshaped @ grad_reshaped.T
                    grad_outer_i = (
                        grad.reshape(out_c * kh * kw, in_c)
                        @ grad.reshape(out_c * kh * kw, in_c).T
                    )

                    precond_o = state["precond_O"]
                    precond_i = state["precond_I"]

                    precond_o.mul_(beta).add_(grad_outer_o, alpha=1 - beta)
                    precond_i.mul_(beta).add_(grad_outer_i, alpha=1 - beta)

                    precond_o_inv = torch.linalg.inv(
                        precond_o + damping * torch.eye(out_c, device=param.device),
                    )
                    precond_i_inv = torch.linalg.inv(
                        precond_i
                        + damping * torch.eye(in_c * kh * kw, device=param.device),
                    )

                    precond_grad = precond_o_inv @ grad_reshaped @ precond_i_inv
                    precond_grad = precond_grad.reshape(out_c, in_c, kh, kw)

                elif len(param.shape) == 2:
                    precond_o = state["precond_O"]
                    precond_o.mul_(beta).add_(grad @ grad.T, alpha=1 - beta)

                    precond_o_inv = torch.linalg.inv(
                        precond_o
                        + damping * torch.eye(precond_o.shape[0], device=param.device),
                    )
                    precond_grad = precond_o_inv @ grad

                else:
                    precond_grad = grad

                momentum = state["momentum"]
                momentum.mul_(group["momentum"]).add_(precond_grad)

                param.data.add_(momentum, alpha=-group["lr"])

        return loss


class SecondOrderInfo:
    """Container for second-order optimizer statistics and utilities."""

    @staticmethod
    def compute_gradient_spectrum(grad: Tensor, top_k: int = 10) -> Tensor:
        """
        Compute the eigenvalue spectrum of the gradient's second moment matrix.

        Args:
            grad: Gradient tensor
            top_k: Number of top eigenvalues to return

        Returns:
            Tensor of top eigenvalues
        """
        grad_flat = grad.flatten()
        n = min(grad_flat.shape[0], 1000)
        indices = torch.randperm(grad_flat.shape[0])[:n]
        samples = grad_flat[indices]

        matrix = samples.unsqueeze(1) * samples.unsqueeze(0)
        eigenvalues = torch.linalg.eigvalsh(matrix)

        return eigenvalues[-top_k:]

    @staticmethod
    def compute_condition_number(grad: Tensor) -> float:
        """
        Estimate the condition number of the gradient covariance.

        Args:
            grad: Gradient tensor

        Returns:
            Estimated condition number
        """
        grad_flat = grad.flatten()
        n = min(grad_flat.shape[0], 500)
        indices = torch.randperm(grad_flat.shape[0])[:n]
        samples = grad_flat[indices]

        matrix = samples.unsqueeze(1) * samples.unsqueeze(0)
        eigenvalues = torch.linalg.eigvalsh(matrix)

        if eigenvalues[0] > 0:
            return eigenvalues[-1].item() / eigenvalues[0].item()
        return float("inf")

    @staticmethod
    def adaptive_damping(grad: Tensor, base_damping: float = 1e-3) -> float:
        """
        Compute adaptive damping based on gradient curvature.

        Args:
            grad: Gradient tensor
            base_damping: Base damping value

        Returns:
            Adaptive damping factor
        """
        cond = SecondOrderInfo.compute_condition_number(grad)
        return base_damping * (1 + math.log(1 + cond))


__all__ = [
    "KFACOptimizer",
    "ESGDOptimizer",
    "ShampooOptimizer",
    "SecondOrderInfo",
]
