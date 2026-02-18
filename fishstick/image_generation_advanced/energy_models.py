"""
Energy-Based Model (EBM) implementations for image generation.

This module provides energy-based generative models:
- EBM: Base energy-based model with energy function
- Convolutional EBM: EBM with convolutional architecture
- Diffusion EBM: EBM with diffusion process for sampling
- Langevin Dynamics: Sampling method for EBMs
"""

from typing import Optional, List, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EnergyFunction(nn.Module):
    """Base energy function for EBMs."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute energy for input."""
        return self.net(x)


class ConvolutionalEnergyFunction(nn.Module):
    """Convolutional energy function for image EBMs.

    Args:
        num_channels: Number of input channels
        hidden_channels: Number of hidden channels
        image_size: Size of input images
    """

    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 64,
        image_size: int = 32,
    ):
        super().__init__()
        self.image_size = image_size

        layers = []
        current_channels = num_channels

        for i in range(4):
            layers.extend(
                [
                    nn.Conv2d(current_channels, hidden_channels * (2**i), 4, 2, 1),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = hidden_channels * (2**i)

        self.conv = nn.Sequential(*layers)

        final_size = image_size // (2**4)
        self.fc = nn.Sequential(
            nn.Linear(current_channels * final_size * final_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute energy for input image."""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class ConditionalEnergyFunction(nn.Module):
    """Conditional energy function for class-conditioned generation.

    Args:
        num_channels: Number of input channels
        num_classes: Number of conditioning classes
        hidden_channels: Number of hidden channels
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 10,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.image_net = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.label_embed = nn.Embedding(num_classes, hidden_channels)

        self.combined_net = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        """Compute conditional energy for input and labels."""
        h_img = self.image_net(x)

        label_embed = self.label_embed(labels)
        label_embed = (
            label_embed.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, x.shape[2], x.shape[3])
        )

        h = torch.cat([h_img, label_embed], dim=1)
        h = self.combined_net(h)
        h = h.view(h.size(0), -1)

        return self.fc(h)


class EBM(nn.Module):
    """Energy-Based Model with contrastive divergence training.

    Args:
        energy_fn: Energy function network
        sigma: Noise standard deviation for Langevin dynamics
        num_steps: Number of Langevin steps for sampling
    """

    def __init__(
        self,
        energy_fn: Optional[nn.Module] = None,
        input_dim: int = 3072,
        hidden_dim: int = 512,
        sigma: float = 0.01,
        num_steps: int = 10,
    ):
        super().__init__()

        if energy_fn is None:
            self.energy_fn = ConvolutionalEnergyFunction(num_channels=3, image_size=32)
        else:
            self.energy_fn = energy_fn

        self.sigma = sigma
        self.num_steps = num_steps

    def forward(self, x: Tensor) -> Tensor:
        """Compute energy for input."""
        return self.energy_fn(x).squeeze(-1)

    def langevin_dynamics(
        self,
        x: Tensor,
        lr: float = 0.01,
        noise_std: Optional[float] = None,
    ) -> Tensor:
        """Run Langevin dynamics for sampling.

        Args:
            x: Initial noise tensor
            lr: Learning rate for Langevin updates
            noise_std: Standard deviation of noise (defaults to self.sigma)
        """
        if noise_std is None:
            noise_std = self.sigma

        x = x.clone().detach().requires_grad_(True)

        for _ in range(self.num_steps):
            energy = self.energy_fn(x)
            grad = torch.autograd.grad(energy.sum(), x, retain_graph=True)[0]

            x = x - lr * grad + noise_std * torch.randn_like(x)

        return x.detach()

    def sample(
        self,
        num_samples: int,
        shape: Tuple[int, ...],
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """Generate samples using Langevin dynamics.

        Args:
            num_samples: Number of samples to generate
            shape: Shape of each sample (excluding batch)
            device: Device to generate samples on
        """
        x = torch.randn(num_samples, *shape, device=device)
        return self.langevin_dynamics(x)

    def contrastive_divergence_loss(
        self,
        x_pos: Tensor,
        k: int = 10,
        lr: float = 0.01,
    ) -> torch.Tensor:
        """Compute contrastive divergence loss.

        Args:
            x_pos: Positive samples (real data)
            k: Number of Langevin steps
            lr: Learning rate for Langevin updates
        """
        batch_size = x_pos.size(0)

        x_neg = x_pos.clone().detach()
        x_neg.requires_grad_(True)

        for _ in range(k):
            energy = self.energy_fn(x_neg)
            grad = torch.autograd.grad(energy.sum(), x_neg, retain_graph=True)[0]
            x_neg = x_neg - lr * grad + self.sigma * torch.randn_like(x_neg)

        e_pos = self.energy_fn(x_pos).mean()
        e_neg = self.energy_fn(x_neg.detach()).mean()

        return e_pos - e_neg


class ConditionalEBM(nn.Module):
    """Conditional Energy-Based Model for class-conditional generation.

    Args:
        num_classes: Number of conditioning classes
        num_channels: Number of input channels
        hidden_channels: Number of hidden channels
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.energy_fn = ConditionalEnergyFunction(
            num_channels=num_channels,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
        )

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        """Compute conditional energy."""
        return self.energy_fn(x, labels).squeeze(-1)

    def conditional_langevin_dynamics(
        self,
        labels: Tensor,
        shape: Tuple[int, int, int, int],
        lr: float = 0.01,
    ) -> Tensor:
        """Run conditional Langevin dynamics.

        Args:
            labels: Class labels for conditioning
            shape: Shape of samples to generate
            lr: Learning rate for Langevin updates
        """
        batch_size = labels.size(0)
        x = torch.randn(batch_size, *shape[1:], device=labels.device)
        x = x.clone().requires_grad_(True)

        for _ in range(10):
            energy = self.energy_fn(x, labels)
            grad = torch.autograd.grad(energy.sum(), x, retain_graph=True)[0]
            x = x - lr * grad + 0.01 * torch.randn_like(x)

        return x.detach()

    def sample(
        self,
        labels: Tensor,
        shape: Tuple[int, int, int, int],
    ) -> Tensor:
        """Generate conditional samples.

        Args:
            labels: Class labels for conditioning
            shape: Shape of samples to generate
        """
        return self.conditional_langevin_dynamics(labels, shape)


class MCMCScheduler:
    """MCMC scheduler for adaptive Langevin dynamics.

    Args:
        initial_steps: Initial number of Langevin steps
        final_steps: Final number of Langevin steps
        max_iterations: Maximum number of training iterations
    """

    def __init__(
        self,
        initial_steps: int = 10,
        final_steps: int = 60,
        max_iterations: int = 100000,
    ):
        self.initial_steps = initial_steps
        self.final_steps = final_steps
        self.max_iterations = max_iterations

    def get_steps(self, iteration: int) -> int:
        """Get number of steps for current iteration."""
        progress = min(iteration / self.max_iterations, 1.0)
        return int(
            self.initial_steps + (self.final_steps - self.initial_steps) * progress
        )


class DenoisingEBM(nn.Module):
    """Denoising Energy-Based Model for improved sampling.

    Args:
        num_channels: Number of input channels
        hidden_channels: Number of hidden channels
    """

    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 128,
    ):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.net = nn.Sequential(
            nn.Conv2d(num_channels + num_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_channels, 3, padding=1),
        )

        self.energy_net = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1),
        )

    def denoise(self, x_noisy: Tensor, t: Tensor) -> Tensor:
        """Denoise noisy input at timestep t."""
        time_emb = self.get_time_embedding(t)
        time_emb = self.time_embed(time_emb)

        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        time_emb = time_emb.expand(-1, -1, x_noisy.shape[2], x_noisy.shape[3])

        h = torch.cat([x_noisy, time_emb], dim=1)
        denoised = self.net(h)

        return denoised

    def energy(self, x: Tensor) -> Tensor:
        """Compute energy of input."""
        return self.energy_net(x).squeeze(-1)

    def get_time_embedding(self, t: Tensor) -> Tensor:
        """Create sinusoidal time embedding."""
        half_dim = 32
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute energy at timestep t."""
        return self.energy(x).squeeze(-1)


class GradientPenaltyEBM(nn.Module):
    """EBM with gradient penalty for improved training stability.

    Args:
        energy_fn: Energy function network
        penalty_weight: Weight for gradient penalty
    """

    def __init__(
        self,
        energy_fn: Optional[nn.Module] = None,
        num_channels: int = 3,
        image_size: int = 32,
        penalty_weight: float = 10.0,
    ):
        super().__init__()

        if energy_fn is None:
            self.energy_fn = ConvolutionalEnergyFunction(
                num_channels=num_channels,
                image_size=image_size,
            )
        else:
            self.energy_fn = energy_fn

        self.penalty_weight = penalty_weight

    def forward(self, x: Tensor) -> Tensor:
        """Compute energy for input."""
        return self.energy_fn(x).squeeze(-1)

    def gradient_penalty(self, x_real: Tensor, x_fake: Tensor) -> Tensor:
        """Compute gradient penalty."""
        batch_size = x_real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=x_real.device)

        x_interpolated = alpha * x_real + (1 - alpha) * x_fake
        x_interpolated.requires_grad_(True)

        energy = self.energy_fn(x_interpolated)

        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=x_interpolated,
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_norm = grad.view(batch_size, -1).norm(dim=1)
        penalty = ((grad_norm - 1) ** 2).mean()

        return penalty

    def wasserstein_loss(self, x_real: Tensor, x_fake: Tensor) -> torch.Tensor:
        """Compute Wasserstein loss."""
        energy_real = self.energy_fn(x_real).mean()
        energy_fake = self.energy_fn(x_fake).mean()

        return energy_fake - energy_real

    def compute_loss(
        self,
        x_real: Tensor,
        x_fake: Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss with gradient penalty."""
        wasserstein = self.wasserstein_loss(x_real, x_fake)
        gp = self.gradient_penalty(x_real, x_fake)

        loss = wasserstein + self.penalty_weight * gp

        return loss, {
            "wasserstein": wasserstein,
            "gradient_penalty": gp,
        }


class JointEBM(nn.Module):
    """Joint Energy-Based Model for joint generation and classification.

    Args:
        num_classes: Number of classes
        num_channels: Number of input channels
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.energy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass computing energy and class logits."""
        features = self.feature_extractor(x)

        energy = self.energy_head(features)
        logits = self.class_head(features)

        if return_features:
            return energy.squeeze(-1), logits, features

        return energy.squeeze(-1), logits
