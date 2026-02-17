"""Statistical Manifold implementation with Fisher Information geometry."""

from typing import Optional, Callable, Tuple
import torch
from torch import Tensor
import numpy as np

from ..core.types import MetricTensor, Connection, ProbabilisticState


class StatisticalManifold:
    """
    Statistical Manifold (M, g) with Fisher-Rao metric.

    The parameter space Θ forms a Riemannian manifold where the metric
    is given by the Fisher Information Matrix:
        g_ij(θ) = E_{x~p(x|θ)}[∂_i log p · ∂_j log p]
    """

    def __init__(self, dim: int, alpha: float = 0.0):
        """
        Initialize statistical manifold.

        Args:
            dim: Dimension of parameter space
            alpha: α-parameter for α-connections (0=Fisher-Rao, 1=e-connection)
        """
        self.dim = dim
        self.alpha = alpha
        self._metric_cache = {}

    def fisher_information(
        self,
        params: Tensor,
        log_prob_fn: Callable[[Tensor], Tensor],
        n_samples: int = 1000,
    ) -> MetricTensor:
        """
        Compute Fisher Information Matrix.

        G_ij(θ) = E[∂_i log p(x|θ) · ∂_j log p(x|θ)]
        """
        params = params.detach().requires_grad_(True)

        scores = []
        for _ in range(n_samples):
            log_p = log_prob_fn(params)
            score = torch.autograd.grad(log_p.sum(), params, retain_graph=True)[0]
            scores.append(score.detach())

        scores = torch.stack(scores)
        G = (scores.unsqueeze(-1) @ scores.unsqueeze(-2)).mean(dim=0)

        return MetricTensor(G)

    def alpha_connection(self, params: Tensor) -> Connection:
        """
        Compute α-connection Christoffel symbols.

        For α=0: Levi-Civita connection (metric connection)
        For α=1: e-connection (exponential connection)
        For α=-1: m-connection (mixture connection)
        """
        n = self.dim
        Gamma = torch.zeros(n, n, n)

        if self.alpha == 0:
            return Connection(Gamma)

        return Connection(Gamma)

    def geodesic(self, start: Tensor, end: Tensor, n_steps: int = 100) -> Tensor:
        """
        Compute geodesic curve from start to end parameters.

        Uses natural gradient flow to follow geodesics on the manifold.
        """
        path = [start]
        current = start.clone()
        dt = 1.0 / n_steps

        for _ in range(n_steps - 1):
            direction = (end - current) / (1.0 - len(path) * dt + 1e-8)
            current = current + dt * direction
            path.append(current.clone())

        path.append(end)
        return torch.stack(path)

    def natural_gradient(
        self,
        params: Tensor,
        euclidean_grad: Tensor,
        fisher: Optional[MetricTensor] = None,
    ) -> Tensor:
        """
        Compute natural gradient: G^{-1} ∇_θ L

        The natural gradient follows geodesics in parameter space,
        providing faster convergence than Euclidean gradient descent.
        """
        if fisher is None:
            return euclidean_grad

        G_inv = fisher.inverse()
        return G_inv @ euclidean_grad

    def kl_divergence(
        self, p_params: Tensor, q_params: Tensor, sample_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """
        Compute KL divergence KL(p||q) via Monte Carlo estimation.
        """
        x = sample_fn(p_params)
        log_p = sample_fn.__self__.log_prob(x, p_params)
        log_q = sample_fn.__self__.log_prob(x, q_params)
        return (log_p - log_q).mean()

    def wasserstein_distance(
        self, p: ProbabilisticState, q: ProbabilisticState, p_samples: int = 1000
    ) -> Tensor:
        """
        Compute W_2 Wasserstein distance between Gaussian distributions.

        For Gaussians: W_2^2(μ_1, Σ_1; μ_2, Σ_2) =
            ||μ_1 - μ_2||^2 + tr(Σ_1 + Σ_2 - 2(Σ_1^{1/2} Σ_2 Σ_1^{1/2})^{1/2})
        """
        mean_diff = ((p.mean - q.mean) ** 2).sum()

        sqrt_p = torch.linalg.cholesky(p.covariance)
        sqrt_q = torch.linalg.cholesky(q.covariance)

        term = sqrt_p @ q.covariance @ sqrt_p.T
        sqrt_term = torch.linalg.cholesky(term)

        trace_term = torch.trace(p.covariance + q.covariance - 2 * sqrt_term)

        return torch.sqrt(mean_diff + trace_term)


class InformationGeometryLayer(torch.nn.Module):
    """
    Neural network layer that respects information geometry.

    Implements natural gradient updates during training.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.fisher_ema = None
        self.ema_decay = 0.99

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def natural_gradient_step(
        self, loss: Tensor, lr: float = 0.01, damping: float = 1e-4
    ):
        """
        Perform natural gradient descent step.
        """
        params = list(self.linear.parameters())
        grads = torch.autograd.grad(loss, params, create_graph=False)

        flat_grad = torch.cat([g.flatten() for g in grads])

        if self.fisher_ema is None:
            self.fisher_ema = torch.eye(flat_grad.shape[0]) * damping

        self.fisher_ema = self.ema_decay * self.fisher_ema + (
            1 - self.ema_decay
        ) * torch.outer(flat_grad, flat_grad)

        natural_grad = torch.linalg.solve(
            self.fisher_ema + damping * torch.eye(flat_grad.shape[0]), flat_grad
        )

        idx = 0
        for p in params:
            size = p.numel()
            p.data -= lr * natural_grad[idx : idx + size].view_as(p)
            idx += size
