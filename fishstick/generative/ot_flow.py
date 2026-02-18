"""
Optimal Transport Flow Matching.

Implements flow matching with optimal transport geometry as described in:
Tong et al. (2023) "Flow Matching in Latent Space"
Fatras et al. (2021) "Learning with OT Flow"

Key features:
- Sinkhorn divergence for computing optimal transport
- Entropic optimal transport
- Minimum cost flow matching
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def sinkhorn(
    a: Tensor,
    b: Tensor,
    M: Tensor,
    epsilon: float = 0.1,
    num_iterations: int = 100,
) -> Tuple[Tensor, Tensor]:
    """
    Sinkhorn algorithm for entropic optimal transport.

    Args:
        a: Source distribution weights [batch_size]
        b: Target distribution weights [batch_size]
        M: Cost matrix [batch_size, batch_size]
        epsilon: Entropic regularization
        num_iterations: Number of Sinkhorn iterations

    Returns:
        Transport plan, log
    """
    n = a.shape[0]
    m = b.shape[0]

    K = torch.exp(-M / epsilon)

    u = torch.ones(n, device=M.device)
    v = torch.ones(m, device=M.device)

    for _ in range(num_iterations):
        u = a / (K @ v + 1e-8)
        v = b / (K.T @ u + 1e-8)

    P = torch.diag(u) @ K @ torch.diag(v)

    return P, None


def sinkhorn_divergence(
    x0: Tensor,
    x1: Tensor,
    epsilon: float = 0.1,
    num_iterations: int = 100,
) -> Tensor:
    """
    Compute Sinkhorn divergence between two point clouds.

    Args:
        x0: Source points [batch_size, dim]
        x1: Target points [batch_size, dim]
        epsilon: Entropic regularization
        num_iterations: Number of iterations

    Returns:
        Sinkhorn divergence
    """
    batch_size = x0.shape[0]

    a = torch.ones(batch_size, device=x0.device) / batch_size
    b = torch.ones(batch_size, device=x1.device) / batch_size

    M = torch.cdist(x0, x1) ** 2

    P, _ = sinkhorn(a, b, M, epsilon, num_iterations)

    cost = (P * M).sum()

    return cost


class OptimalTransportFlow(nn.Module):
    """
    Flow matching with optimal transport coupling.

    Uses optimal transport to find better correspondences between
    noise and data for improved flow matching.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        epsilon: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.epsilon = epsilon

        from .flow_matching import FlowMatchingNetwork

        self.velocity_net = FlowMatchingNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict velocity field."""
        return self.velocity_net(x, t)

    def compute_ot_coupling(
        self,
        x0: Tensor,
        x1: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute optimal transport coupling between source and target.

        Args:
            x0: Source samples (noise)
            x1: Target samples (data)

        Returns:
            Coupled source and target
        """
        batch_size = x0.shape[0]

        a = torch.ones(batch_size, device=x0.device) / batch_size
        b = torch.ones(batch_size, device=x1.device) / batch_size

        M = torch.cdist(x0, x1) ** 2

        P, _ = sinkhorn(a, b, M, self.epsilon)

        x0_coupled = P @ x1

        x1_coupled = x1

        return x0_coupled, x1_coupled

    def training_loss(
        self,
        x0: Tensor,
        x1: Tensor,
        use_ot: bool = True,
    ) -> Tensor:
        """
        Compute flow matching loss with optional OT coupling.

        Args:
            x0: Noise samples
            x1: Data samples
            use_ot: Whether to use optimal transport coupling

        Returns:
            Loss
        """
        if use_ot:
            x0_coupled, x1_coupled = self.compute_ot_coupling(x0, x1)
        else:
            x0_coupled, x1_coupled = x0, x1

        batch_size = x0.shape[0]

        t = torch.rand(batch_size, device=x0.device)

        xt = (1 - t.view(-1, 1)) * x0_coupled + t.view(-1, 1) * x1_coupled

        target_velocity = x1_coupled - x0_coupled

        predicted_velocity = self.velocity_net(xt, t)

        loss = ((predicted_velocity - target_velocity) ** 2).mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_steps: int = 100,
        device: str = "cpu",
    ) -> Tensor:
        """Generate samples."""
        x = torch.randn(num_samples, self.input_dim, device=device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((num_samples,), i / num_steps, device=device)
            velocity = self.velocity_net(x, t)
            x = x + velocity * dt

        return x


class SinkhornDivergence(nn.Module):
    """
    Sinkhorn divergence loss for training generative models.

    Provides an alternative to Wasserstein distance for comparing distributions.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        num_iterations: int = 100,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_iterations = num_iterations

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        y0: Optional[Tensor] = None,
        y1: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute Sinkhorn divergence.

        Args:
            x0: First samples
            x1: Second samples
            y0: Optional labels for first samples
            y1: Optional labels for second samples

        Returns:
            Sinkhorn divergence
        """
        if y0 is not None and y1 is not None:
            x0 = x0[y0]
            x1 = x1[y1]

        return sinkhorn_divergence(x0, x1, self.epsilon, self.num_iterations)

    def conditional_sinkhorn(
        self,
        x0: Tensor,
        x1: Tensor,
        labels: Tensor,
        num_classes: int,
    ) -> Tensor:
        """
        Compute class-conditional Sinkhorn divergence.

        Args:
            x0: Source samples
            x1: Target samples
            labels: Class labels
            num_classes: Number of classes

        Returns:
            Conditional Sinkhorn divergence
        """
        total_loss = 0

        for c in range(num_classes):
            mask = labels == c

            if mask.sum() > 0:
                loss = sinkhorn_divergence(
                    x0[mask], x1[mask], self.epsilon, self.num_iterations
                )
                total_loss += loss

        return total_loss / num_classes


class OTFlowMatching(nn.Module):
    """
    Complete OT-Flow model combining optimal transport with flow matching.

    Uses entropic regularization and computes optimal transport for training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        epsilon: float = 0.1,
        lambda_reg: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg

        from .flow_matching import FlowMatchingNetwork

        self.velocity_net = FlowMatchingNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict velocity."""
        return self.velocity_net(x, t)

    def training_loss(
        self,
        x0: Tensor,
        x1: Tensor,
    ) -> Tensor:
        """
        Compute OT-Flow matching loss.

        Args:
            x0: Noise samples
            x1: Data samples

        Returns:
            Combined loss
        """
        batch_size = x0.shape[0]

        a = torch.ones(batch_size, device=x0.device) / batch_size
        b = torch.ones(batch_size, device=x1.device) / batch_size

        M = torch.cdist(x0, x1) ** 2

        P, _ = sinkhorn(a, b, M, self.epsilon)

        x0_coupled = P @ x1

        t = torch.rand(batch_size, device=x0.device)

        xt = (1 - t.view(-1, 1)) * x0_coupled + t.view(-1, 1) * x1

        target_velocity = x1 - x0_coupled

        predicted_velocity = self.velocity_net(xt, t)

        flow_loss = ((predicted_velocity - target_velocity) ** 2).mean()

        ot_cost = (P * M).sum()

        total_loss = flow_loss + self.lambda_reg * ot_cost

        return total_loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_steps: int = 100,
        device: str = "cpu",
    ) -> Tensor:
        """Generate samples."""
        x = torch.randn(num_samples, self.input_dim, device=device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((num_samples,), i / num_steps, device=device)
            velocity = self.velocity_net(x, t)
            x = x + velocity * dt

        return x


class ConditionalOTFlow(nn.Module):
    """
    Conditional OT-Flow for class-conditional generation.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(num_classes, hidden_dim)

        from .flow_matching import FlowMatchingNetwork

        self.velocity_net = FlowMatchingNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        class_labels: Tensor,
    ) -> Tensor:
        """Predict velocity with conditioning."""
        condition = self.embedding(class_labels)
        return self.velocity_net(x, t, condition)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        class_labels: Tensor,
        num_steps: int = 100,
        device: str = "cpu",
    ) -> Tensor:
        """Generate class-conditional samples."""
        x = torch.randn(num_samples, self.input_dim, device=device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((num_samples,), i / num_steps, device=device)

            condition = self.embedding(class_labels)
            velocity = self.velocity_net(x, t, condition)

            x = x + velocity * dt

        return x
