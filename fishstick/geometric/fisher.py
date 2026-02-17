"""Geometric module for Fisher information and natural gradients."""

from typing import Optional, Callable
import torch
from torch import Tensor, nn
import numpy as np

from ..core.types import MetricTensor


class FisherInformationMetric:
    """
    Fisher Information Metric g_ij(θ).

    Defines Riemannian structure on parameter manifold:
        g_ij(θ) = E_{x~p(x|θ)}[∂_i log p(x|θ) · ∂_j log p(x|θ)]

    The Fisher metric:
    1. Measures information content near θ
    2. Defines natural gradient direction
    3. Enables geodesic-based optimization
    """

    def __init__(self, damping: float = 1e-4, ema_decay: float = 0.99):
        self.damping = damping
        self.ema_decay = ema_decay
        self._fisher_ema: Optional[Tensor] = None

    def compute(self, log_probs: Tensor, params: Tensor) -> MetricTensor:
        """
        Compute Fisher Information Matrix from log probabilities.

        G = E[∇ log p · (∇ log p)^T]
        """
        params = params.detach().requires_grad_(True)
        score = torch.autograd.grad(log_probs.sum(), params, retain_graph=True)[0]

        if score.dim() == 1:
            G = torch.outer(score, score)
        else:
            G = score.T @ score / score.shape[0]

        return MetricTensor(G + self.damping * torch.eye(G.shape[0]))

    def monte_carlo_estimate(
        self, model: nn.Module, data: Tensor, n_samples: int = 100
    ) -> MetricTensor:
        """
        Monte Carlo estimation of Fisher matrix.
        """
        params = torch.cat([p.flatten() for p in model.parameters()])
        dim = params.shape[0]

        scores = []
        model.eval()

        for _ in range(n_samples):
            model.zero_grad()
            output = model(data)
            log_prob = torch.log_softmax(output, dim=-1).sum()

            score = torch.autograd.grad(
                log_prob, model.parameters(), retain_graph=False, create_graph=False
            )
            flat_score = torch.cat([s.flatten() for s in score])
            scores.append(flat_score)

        scores = torch.stack(scores)
        G = (scores.T @ scores) / n_samples

        return MetricTensor(G + self.damping * torch.eye(dim))

    def update_ema(self, grad: Tensor) -> None:
        """Update exponential moving average of Fisher."""
        if self._fisher_ema is None:
            self._fisher_ema = torch.outer(grad, grad)
        else:
            self._fisher_ema = self.ema_decay * self._fisher_ema + (
                1 - self.ema_decay
            ) * torch.outer(grad, grad)

    def get_inverse(self) -> Optional[Tensor]:
        """Get inverse of EMA Fisher matrix."""
        if self._fisher_ema is None:
            return None
        return torch.linalg.inv(
            self._fisher_ema + self.damping * torch.eye(self._fisher_ema.shape[0])
        )


class NaturalGradient:
    """
    Natural Gradient Descent.

    Update rule: θ ← θ - η G^{-1} ∇_θ L

    The natural gradient:
    1. Follows geodesics on statistical manifold
    2. Achieves faster convergence in curved spaces
    3. Is invariant to parameter reparameterization
    """

    def __init__(
        self,
        params: list,
        lr: float = 0.01,
        damping: float = 1e-4,
        ema_decay: float = 0.99,
    ):
        self.params = list(params)
        self.lr = lr
        self.fisher = FisherInformationMetric(damping, ema_decay)
        self._flat_params = self._flatten_params()

    def _flatten_params(self) -> Tensor:
        return torch.cat([p.detach().flatten() for p in self.params])

    def _unflatten_grad(self, flat_grad: Tensor) -> list:
        grads = []
        idx = 0
        for p in self.params:
            size = p.numel()
            grads.append(flat_grad[idx : idx + size].view_as(p))
            idx += size
        return grads

    def step(self, loss: Tensor) -> None:
        """
        Perform natural gradient step.

        θ ← θ - η G^{-1} ∇L
        """
        grads = torch.autograd.grad(loss, self.params, retain_graph=False)
        flat_grad = torch.cat([g.flatten() for g in grads])

        self.fisher.update_ema(flat_grad)
        G_inv = self.fisher.get_inverse()

        if G_inv is None:
            natural_grad = flat_grad
        else:
            natural_grad = G_inv @ flat_grad

        with torch.no_grad():
            grads = self._unflatten_grad(natural_grad)
            for p, g in zip(self.params, grads):
                p -= self.lr * g


class NaturalGradientOptimizer(torch.optim.Optimizer):
    """
    PyTorch optimizer implementing natural gradient descent.
    """

    def __init__(
        self, params, lr: float = 0.01, damping: float = 1e-4, ema_decay: float = 0.99
    ):
        defaults = dict(lr=lr, damping=damping, ema_decay=ema_decay)
        super().__init__(params, defaults)

        self._fisher_ema = None
        self._param_shapes = None

    def _get_flat_params(self) -> Tensor:
        return torch.cat(
            [p.flatten() for group in self.param_groups for p in group["params"]]
        )

    def _set_flat_params(self, flat_params: Tensor) -> None:
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                size = p.numel()
                p.data.copy_(flat_params[idx : idx + size].view_as(p))
                idx += size

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        flat_grad = torch.cat(
            [
                p.grad.flatten()
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
        )

        if flat_grad.dim() == 0:
            return loss

        if self._fisher_ema is None:
            self._fisher_ema = torch.outer(flat_grad, flat_grad)
        else:
            ema_decay = self.param_groups[0]["ema_decay"]
            damping = self.param_groups[0]["damping"]
            self._fisher_ema = ema_decay * self._fisher_ema + (
                1 - ema_decay
            ) * torch.outer(flat_grad, flat_grad)

        G_inv = torch.linalg.inv(
            self._fisher_ema
            + damping * torch.eye(self._fisher_ema.shape[0], device=flat_grad.device)
        )

        natural_grad = G_inv @ flat_grad

        lr = self.param_groups[0]["lr"]
        flat_params = self._get_flat_params()
        self._set_flat_params(flat_params - lr * natural_grad)

        return loss


class KFAC(NaturalGradientOptimizer):
    """
    K-FAC: Kronecker-Factored Approximate Curvature.

    Approximates Fisher as Kronecker product:
        F ≈ A ⊗ G
    where A is activation covariance and G is gradient covariance.

    This enables efficient inversion: F^{-1} ≈ A^{-1} ⊗ G^{-1}
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        damping: float = 1e-4,
        ema_decay: float = 0.99,
        factor_decay: float = 0.95,
    ):
        super().__init__(params, lr, damping, ema_decay)
        self.factor_decay = factor_decay
        self._A_factors = {}
        self._G_factors = {}

    def _update_factors(self, layer: nn.Module, activation: Tensor, grad: Tensor):
        """Update Kronecker factors for a layer."""
        layer_id = id(layer)

        A = activation.T @ activation / activation.shape[0]
        G = grad.T @ grad / grad.shape[0]

        if layer_id not in self._A_factors:
            self._A_factors[layer_id] = A
            self._G_factors[layer_id] = G
        else:
            self._A_factors[layer_id] = (
                self.factor_decay * self._A_factors[layer_id]
                + (1 - self.factor_decay) * A
            )
            self._G_factors[layer_id] = (
                self.factor_decay * self._G_factors[layer_id]
                + (1 - self.factor_decay) * G
            )

    def _get_kfac_preconditioned_grad(self, layer: nn.Module, grad: Tensor) -> Tensor:
        """Compute K-FAC preconditioned gradient."""
        layer_id = id(layer)

        if layer_id not in self._A_factors:
            return grad

        A_inv = torch.linalg.inv(
            self._A_factors[layer_id]
            + self.param_groups[0]["damping"]
            * torch.eye(self._A_factors[layer_id].shape[0])
        )
        G_inv = torch.linalg.inv(
            self._G_factors[layer_id]
            + self.param_groups[0]["damping"]
            * torch.eye(self._G_factors[layer_id].shape[0])
        )

        return G_inv @ grad @ A_inv
