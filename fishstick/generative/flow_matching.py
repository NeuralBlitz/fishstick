"""
Flow Matching Implementation.

Implements Conditional Flow Matching as described in:
Lipman et al. (2023) "Flow Matching for Generative Modeling"
Albergo & Vanden-Eijnden (2022) "Building Normalizing Flows with Stochastic Interpolants"

Flow matching provides an alternative to diffusion models with:
- Straight-line interpolation in probability space
- No need for learned variance schedules
- Potentially faster sampling
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


class FlowMatchingNetwork(nn.Module):
    """
    Neural network that predicts velocity field for flow matching.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        for i in range(num_layers):
            in_dim = input_dim * 2 if i == 0 else hidden_dim
            out_dim = hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        condition: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict velocity at position x and time t.

        Args:
            x: Position [batch, input_dim]
            t: Time [batch]
            condition: Optional conditioning

        Returns:
            Velocity [batch, input_dim]
        """
        t = t.view(-1, 1).expand(-1, self.input_dim)

        if condition is not None:
            h = torch.cat([x, t, condition], dim=-1)
        else:
            h = torch.cat([x, t], dim=-1)

        velocity = self.net(h)

        return velocity


class FlowMatching(nn.Module):
    """
    Flow Matching model for generative modeling.

    Learns to predict velocity fields that transform noise to data.

    Args:
        input_dim: Dimension of data
        hidden_dim: Hidden dimension for network
        num_layers: Number of layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.velocity_net = FlowMatchingNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        condition: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict velocity field.

        Args:
            x: Current position
            t: Time
            condition: Optional conditioning

        Returns:
            Velocity
        """
        return self.velocity_net(x, t, condition)

    def training_loss(
        self,
        x0: Tensor,
        x1: Tensor,
    ) -> Tensor:
        """
        Compute flow matching loss.

        Args:
            x0: Noise samples from prior
            x1: Data samples

        Returns:
            Loss
        """
        batch_size = x0.shape[0]

        t = torch.rand(batch_size, device=x0.device)

        xt = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1

        target_velocity = x1 - x0

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
        """
        Generate samples using Euler integration.

        Args:
            num_samples: Number of samples
            num_steps: Number of integration steps
            device: Device

        Returns:
            Generated samples
        """
        x = torch.randn(num_samples, self.input_dim, device=device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((num_samples,), i / num_steps, device=device)

            velocity = self.velocity_net(x, t)

            x = x + velocity * dt

        return x

    @torch.no_grad()
    def sample_heun(
        self,
        num_samples: int,
        num_steps: int = 100,
        device: str = "cpu",
    ) -> Tensor:
        """
        Generate samples using Heun's method (more accurate).

        Args:
            num_samples: Number of samples
            num_steps: Number of steps
            device: Device

        Returns:
            Generated samples
        """
        x = torch.randn(num_samples, self.input_dim, device=device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((num_samples,), i / num_steps, device=device)

            v1 = self.velocity_net(x, t)

            x_temp = x + v1 * dt

            if i < num_steps - 1:
                t_next = torch.full((num_samples,), (i + 1) / num_steps, device=device)
                v2 = self.velocity_net(x_temp, t_next)

                x = x + (v1 + v2) * dt / 2
            else:
                x = x_temp

        return x


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching for class-conditional generation.
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
        """
        Predict velocity with class conditioning.

        Args:
            x: Position
            t: Time
            class_labels: Class labels

        Returns:
            Velocity
        """
        condition = self.embedding(class_labels)

        return self.velocity_net(x, t, condition)

    def training_loss(
        self,
        x0: Tensor,
        x1: Tensor,
        class_labels: Tensor,
    ) -> Tensor:
        """
        Compute conditional flow matching loss.

        Args:
            x0: Noise samples
            x1: Data samples
            class_labels: Class labels

        Returns:
            Loss
        """
        batch_size = x0.shape[0]

        t = torch.rand(batch_size, device=x0.device)

        xt = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1

        target_velocity = x1 - x0

        condition = self.embedding(class_labels)
        predicted_velocity = self.velocity_net(xt, t, condition)

        loss = ((predicted_velocity - target_velocity) ** 2).mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        class_labels: Tensor,
        num_steps: int = 100,
        device: str = "cpu",
    ) -> Tensor:
        """
        Generate class-conditional samples.

        Args:
            num_samples: Number of samples
            class_labels: Class labels
            num_steps: Number of steps
            device: Device

        Returns:
            Generated samples
        """
        x = torch.randn(num_samples, self.input_dim, device=device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((num_samples,), i / num_steps, device=device)

            condition = self.embedding(class_labels)
            velocity = self.velocity_net(x, t, condition)

            x = x + velocity * dt

        return x


class RectifiedFlow(nn.Module):
    """
    Rectified Flow (Flow Matching with optimal transport).

    Uses optimal transport interpolation for more efficient training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.velocity_net = FlowMatchingNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict velocity."""
        return self.velocity_net(x, t)

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

    def compute_coupling(
        self,
        x0: Tensor,
        x1: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute optimal transport coupling.

        Args:
            x0: Source samples (noise)
            x1: Target samples (data)

        Returns:
            Coupled source and target
        """
        cost_matrix = torch.cdist(x0, x1)

        indices = torch.argmin(cost_matrix, dim=1)

        x0_coupled = x0
        x1_coupled = x1[indices]

        return x0_coupled, x1_coupled

    def training_loss_ot(
        self,
        x0: Tensor,
        x1: Tensor,
    ) -> Tensor:
        """
        Compute flow matching loss with optimal transport.

        Args:
            x0: Noise samples
            x1: Data samples

        Returns:
            Loss
        """
        x0_coupled, x1_coupled = self.compute_coupling(x0, x1)

        return self.training_loss(x0_coupled, x1_coupled)
