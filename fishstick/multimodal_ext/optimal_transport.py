"""
Optimal Transport for Modality Alignment in fishstick

This module provides optimal transport-based alignment:
- Wasserstein distance computation
- Sinkhorn iterations
- OT-based modality alignment
- Entropic regularization
"""

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SinkhornDistance(nn.Module):
    """Sinkhorn distance for optimal transport."""

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        C = self._cost_matrix(x, y)

        if x.size(0) == y.size(0):
            n = x.size(0)
            a = torch.ones(n, device=x.device) / n
            b = torch.ones(n, device=x.device) / n
        else:
            raise ValueError("Sinkhorn requires equal batch sizes")

        K = torch.exp(-C / self.epsilon)

        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(self.max_iter):
            u = a / (K @ v + 1e-8)
            v = b / (K.T @ u + 1e-8)

        transport_plan = torch.diag(u) @ K @ torch.diag(v)

        distance = (transport_plan * C).sum()

        if self.reduction == "mean":
            distance = distance / x.size(0)

        return distance, transport_plan, C

    def _cost_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.sum((x - y) ** 2, dim=-1)


class WassersteinDistance(nn.Module):
    """Wasserstein distance computation."""

    def __init__(
        self,
        p: float = 2.0,
    ):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        C = self._cost_matrix(x, y)
        distance = C ** (1 / self.p)
        return distance.mean()

    def _cost_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.sum((x - y) ** 2, dim=-1)


class OptimalTransportAlignment(nn.Module):
    """Optimal transport-based modality alignment."""

    def __init__(
        self,
        embed_dim: int = 256,
        epsilon: float = 0.1,
        max_iter: int = 100,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sinkhorn = SinkhornDistance(epsilon, max_iter)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        distance, transport_plan, cost = self.sinkhorn(source, target)

        aligned_source = self._transport(source, transport_plan)

        return aligned_source, distance, transport_plan

    def _transport(self, x: Tensor, transport_plan: Tensor) -> Tensor:
        return transport_plan @ x


class EntropicOT(nn.Module):
    """Entropic regularized optimal transport."""

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 100,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        C = self._compute_cost(source, target)

        n = source.size(0)
        a = torch.ones(n, device=source.device) / n
        b = torch.ones(n, device=target.device) / n

        log_K = -C / self.epsilon

        u = torch.zeros(n, device=source.device)
        v = torch.zeros(n, device=source.device)

        for _ in range(self.max_iter):
            log_u = torch.log(a + 1e-8) - torch.logsumexp(log_K + v.unsqueeze(1), dim=0)
            u = torch.exp(log_u)

            log_v = torch.log(b + 1e-8) - torch.logsumexp(
                log_K.T + u.unsqueeze(1), dim=0
            )
            v = torch.exp(log_v)

        transport_plan = torch.exp(log_K + v.unsqueeze(1) + u.unsqueeze(0))

        return transport_plan, C

    def _compute_cost(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.sum((x - y) ** 2, dim=-1)


class UnbalancedOT(nn.Module):
    """Unbalanced optimal transport for modality alignment."""

    def __init__(
        self,
        epsilon: float = 0.1,
        tau: float = 1.0,
        max_iter: int = 100,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.tau = tau
        self.max_iter = max_iter

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        C = self._compute_cost(source, target)

        n = source.size(0)
        m = target.size(0)

        a = torch.ones(n, device=source.device)
        b = torch.ones(m, device=target.device)

        K = torch.exp(-C / self.epsilon)

        for _ in range(self.max_iter):
            q = (a.unsqueeze(1) * K * b.unsqueeze(0)).sum(dim=0)
            b = b * (q / (self.tau + q)) ** 0.5

            p = (a.unsqueeze(1) * K * b.unsqueeze(0)).sum(dim=1)
            a = a * (p / (self.tau + p)) ** 0.5

        transport_plan = a.unsqueeze(1) * K * b.unsqueeze(0)

        loss = (transport_plan * C).sum()
        return loss

    def _compute_cost(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.sum((x - y) ** 2, dim=-1)


class POTAlignment(nn.Module):
    """Partial optimal transport for modality alignment."""

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 100,
        portion: float = 0.5,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.portion = portion

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        C = self._compute_cost(source, target)

        n = source.size(0)
        m = target.size(0)

        a = torch.ones(n, device=source.device) / n
        b = torch.ones(m, device=target.device) * (n * self.portion) / m

        K = torch.exp(-C / self.epsilon)

        for _ in range(self.max_iter):
            b = b / (K.T @ (a / (K @ b + 1e-8)) + 1e-8)

        transport_plan = a.unsqueeze(1) * K * b.unsqueeze(0)

        loss = (transport_plan * C).sum()
        return loss

    def _compute_cost(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.sum((x - y) ** 2, dim=-1)


class GromovWasserstein(nn.Module):
    """Gromov-Wasserstein distance for structural alignment."""

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 100,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        C_s = self._distance_matrix(source)
        C_t = self._distance_matrix(target)

        n = source.size(0)
        a = torch.ones(n, device=source.device) / n
        b = torch.ones(n, device=source.device) / n

        C1 = C_s.unsqueeze(2).expand(n, n, n)
        C2 = C_t.unsqueeze(1).expand(n, n, n)

        cost = torch.abs(C1 - C2)

        transport_plan = a.unsqueeze(1) * b.unsqueeze(0)

        for _ in range(self.max_iter):
            marginals = cost * transport_plan.unsqueeze(-1)
            gw_dist = (marginals * cost).sum()

        return gw_dist

    def _distance_matrix(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        y = x.unsqueeze(0)
        return torch.sum((x - y) ** 2, dim=-1)


class Sinkhornknopp(nn.Module):
    """Sinkhorn-Knopp algorithm for matrix scaling."""

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 100,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        C = self._compute_cost(source, target)

        n = source.size(0)
        m = target.size(0)

        a = torch.ones(n, device=source.device) / n
        b = torch.ones(m, device=target.device) / m

        K = torch.exp(-C / self.epsilon)

        for _ in range(self.max_iter):
            u = a / (K @ b + 1e-8)
            v = b / (K.T @ u + 1e-8)

        transport_plan = torch.diag(u) @ K @ torch.diag(v)

        return transport_plan

    def _compute_cost(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.sum((x - y) ** 2, dim=-1)


class NeuralOptimalTransport(nn.Module):
    """Neural optimal transport module."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.cost_network = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        source_expanded = source.unsqueeze(1).expand(-1, target.size(0), -1)
        target_expanded = target.unsqueeze(0).expand(source.size(0), -1, -1)

        cost_input = torch.cat([source_expanded, target_expanded], dim=-1)
        C = self.cost_network(cost_input).squeeze(-1)

        n = source.size(0)
        a = torch.ones(n, device=source.device) / n
        b = torch.ones(n, device=source.device) / n

        K = torch.exp(-C)

        for _ in range(100):
            u = a / (K @ b + 1e-8)
            v = b / (K.T @ u + 1e-8)

        transport_plan = torch.diag(u) @ K @ torch.diag(v)

        return transport_plan, C


def create_ot_module(
    ot_type: str = "sinkhorn",
    **kwargs,
) -> nn.Module:
    """Factory function to create optimal transport modules."""
    if ot_type == "sinkhorn":
        return SinkhornDistance(**kwargs)
    elif ot_type == "wasserstein":
        return WassersteinDistance(**kwargs)
    elif ot_type == "entropic":
        return EntropicOT(**kwargs)
    elif ot_type == "unbalanced":
        return UnbalancedOT(**kwargs)
    elif ot_type == "partial":
        return POTAlignment(**kwargs)
    elif ot_type == "gromov":
        return GromovWasserstein(**kwargs)
    elif ot_type == "neural":
        return NeuralOptimalTransport(**kwargs)
    else:
        raise ValueError(f"Unknown OT type: {ot_type}")
