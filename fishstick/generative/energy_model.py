"""
Energy-Based Models (EBMs).

Implements energy-based generative models as described in:
LeCun et al. (2006) "A Tutorial on Energy-Based Learning"
Du & Mordatch (2019) "Implicit Generation and Modeling with Energy-Based Models"

Energy-based models learn an energy function E(x) where:
- Low energy = high probability (data)
- High energy = low probability (noise)
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class EnergyBasedModel(nn.Module):
    """
    Base class for Energy-Based Models.

    Learns an energy function that assigns low energy to data
    and high energy to other points.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute energy of input.

        Args:
            x: Input [batch, input_dim]

        Returns:
            Energy values [batch, 1]
        """
        return self.net(x)

    def energy(self, x: Tensor) -> Tensor:
        """Get energy as scalar."""
        return self.forward(x).squeeze(-1)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 100,
        step_size: float = 0.1,
        noise_scale: float = 0.1,
    ) -> Tensor:
        """
        Generate samples using Langevin dynamics.

        Args:
            shape: Sample shape
            num_steps: Number of Langevin steps
            step_size: Step size for Langevin dynamics
            noise_scale: Scale of noise for Langevin dynamics

        Returns:
            Generated samples
        """
        batch_size = shape[0]

        x = torch.randn(shape, device=next(self.parameters()).device)

        for _ in range(num_steps):
            noise = torch.randn_like(x) * noise_scale

            energy = self.energy(x)

            x_grad = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=x,
                create_graph=True,
            )[0]

            x = x - step_size * x_grad + noise

        return x

    def contrastive_divergence(
        self,
        x_data: Tensor,
        k: int = 1,
        step_size: float = 0.1,
    ) -> Tensor:
        """
        Compute contrastive divergence loss.

        Args:
            x_data: Data samples
            k: Number of Gibbs sampling steps
            step_size: Step size for Langevin

        Returns:
            CD-k loss
        """
        batch_size = x_data.shape[0]
        x_model = x_data.clone().detach()

        for _ in range(k):
            noise = torch.randn_like(x_model) * 0.1

            energy = self.energy(x_model)
            x_grad = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=x_model,
                create_graph=False,
            )[0]

            x_model = x_model - step_size * x_grad + noise

        pos_energy = self.energy(x_data).mean()
        neg_energy = self.energy(x_model).mean()

        return pos_energy - neg_energy


class ConvEnergyModel(nn.Module):
    """
    Convolutional Energy-Based Model for images.

    Uses convolutional architecture for energy computation on images.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(hidden_channels * 8 * 4 * 4, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute energy of image.

        Args:
            x: Image [batch, channels, height, width]

        Returns:
            Energy [batch, 1]
        """
        return self.net(x)

    def energy(self, x: Tensor) -> Tensor:
        """Get energy as scalar."""
        return self.forward(x).squeeze(-1)


class EBMLoss(nn.Module):
    """
    Energy-Based Model loss using contrastive divergence.
    """

    def __init__(
        self,
        lambda_reg: float = 0.1,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(
        self,
        energy_pos: Tensor,
        energy_neg: Tensor,
    ) -> Tensor:
        """
        Compute EBM loss.

        Args:
            energy_pos: Energy of positive (data) samples
            energy_neg: Energy of negative (model) samples

        Returns:
            Loss
        """
        loss = F.relu(energy_pos - energy_neg + 0.1).mean()

        return loss


class JointDistributionEBM(nn.Module):
    """
    Joint energy-based model for joint p(x, y) modeling.

    Can be used for classification as energy-based model.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Compute joint energy E(x, y).

        Args:
            x: Input
            y: Class labels (one-hot)

        Returns:
            Joint energy
        """
        xy = torch.cat([x, y], dim=-1)
        return self.net(xy)

    def classify(self, x: Tensor) -> Tensor:
        """
        Classify by finding minimum energy class.

        Args:
            x: Input

        Returns:
            Predicted class probabilities
        """
        batch_size = x.shape[0]

        y_one_hot = torch.eye(self.num_classes, device=x.device).unsqueeze(0)
        y_one_hot = y_one_hot.expand(batch_size, -1, -1)

        x_expanded = x.unsqueeze(1).expand(-1, self.num_classes, -1)

        xy = torch.cat([x_expanded, y_one_hot], dim=-1)

        energies = self.net(xy).squeeze(-1)

        probs = F.softmax(-energies, dim=-1)

        return probs
