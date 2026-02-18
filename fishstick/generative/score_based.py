"""
Score-Based Generative Models.

Implements score-based generative modeling using SDEs and ODEs as described in:
Song & Ermon (2019) "Generative Modeling by Estimating Gradients of the Data Distribution"
Song et al. (2021) "Score-Based Generative Modeling through Stochastic Differential Equations"

This includes both the stochastic (SDE) and deterministic (ODE) sampling approaches.
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


class ScoreNetwork(nn.Module):
    """
    Network that estimates the score function ∇_x log p(x).

    The score network takes noisy data and predicts the gradient of the
    log probability density (score function).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels

        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(self._make_res_block(ch, out_ch, dropout))
                ch = out_ch
            if i < len(channel_mults) - 1:
                self.down_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))

        self.mid_block = nn.Sequential(
            self._make_res_block(ch, ch, dropout),
            self._make_res_block(ch, ch, dropout),
        )

        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(self._make_res_block(ch, out_ch, dropout))
                ch = out_ch
            if i < len(channel_mults) - 1:
                self.up_blocks.append(
                    nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
                )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def _make_res_block(self, in_ch: int, out_ch: int, dropout: float) -> nn.Module:
        """Create a residual block with group normalization."""
        return nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.Dropout(dropout),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """
        Predict the score at noise level sigma.

        Args:
            x: Noisy input [batch, channels, height, width]
            sigma: Noise level [batch] or scalar

        Returns:
            Predicted score [batch, channels, height, width]
        """
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0).expand(x.shape[0])

        t = sigma.log().unsqueeze(-1)
        t_emb = self.time_mlp(t)

        h = self.conv_in(x)

        hs = []
        for block in self.down_blocks:
            if isinstance(block, nn.Conv2d):
                h = block(h)
            else:
                h = block(h + t_emb.view(-1, t_emb.size(-1), 1, 1))
            hs.append(h)

        h = self.mid_block(h + t_emb.view(-1, t_emb.size(-1), 1, 1))

        for block in self.up_blocks:
            if isinstance(block, nn.ConvTranspose2d):
                h = block(h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h + t_emb.view(-1, t_emb.size(-1), 1, 1))

        h = self.conv_out(h)
        return h


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for score network."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ScoreBasedModel(nn.Module):
    """
    Score-based generative model using stochastic differential equations.

    Combines multiple noise scales to train a unified score network.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        num_scales: int = 1000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_scales = num_scales

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.sigmas = torch.exp(
            torch.linspace(np.log(sigma_min), np.log(sigma_max), num_scales)
        )

        self.score_net = ScoreNetwork(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
        )

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """
        Predict score for given input and noise level.

        Args:
            x: Noisy input
            sigma: Noise level

        Returns:
            Predicted score
        """
        return self.score_net(x, sigma)

    def training_loss(self, x: Tensor) -> Tensor:
        """
        Compute denoising score matching loss.

        Args:
            x: Clean data samples

        Returns:
            Score matching loss
        """
        batch_size = x.shape[0]

        sigma = torch.rand(batch_size, device=x.device)
        sigma = sigma * (self.sigma_max - self.sigma_min) + self.sigma_min

        noise = torch.randn_like(x)
        noisy_x = x + noise * sigma.view(-1, 1, 1, 1)

        predicted_score = self.score_net(noisy_x, sigma)

        target_score = -noise / sigma.view(-1, 1, 1, 1)

        loss = 0.5 * (
            (predicted_score - target_score) ** 2 * sigma.view(-1, 1, 1, 1) ** 2
        )
        return loss.mean()

    @torch.no_grad()
    def sample_ode(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 100,
        device: str = "cpu",
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
    ) -> Tensor:
        """
        Generate samples using ODE (deterministic) sampling.

        Uses probability flow ODE for deterministic generation.

        Args:
            shape: Sample shape
            num_steps: Number of integration steps
            device: Device
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level

        Returns:
            Generated samples
        """
        batch_size = shape[0]
        sigma_min = sigma_min or self.sigma_min
        sigma_max = sigma_max or self.sigma_max

        x = torch.randn(shape, device=device) * sigma_max

        dt = -sigma_max / num_steps

        for i in range(num_steps):
            sigma = sigma_max * (1 - i / num_steps) + sigma_min * (i / num_steps)
            sigma_tensor = torch.full((batch_size,), sigma, device=device)

            score = self.score_net(x, sigma_tensor)

            drift = -0.5 * sigma**2 * score

            if i < num_steps - 1:
                x = x + drift * dt
            else:
                x = x + drift * dt

        return x

    @torch.no_grad()
    def sample_sde(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 100,
        device: str = "cpu",
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        snr: float = 0.16,
    ) -> Tensor:
        """
        Generate samples using SDE (stochastic) sampling.

        Uses Langevin dynamics for stochastic generation.

        Args:
            shape: Sample shape
            num_steps: Number of integration steps
            device: Device
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            snr: Signal-to-noise ratio for Langevin step

        Returns:
            Generated samples
        """
        batch_size = shape[0]
        sigma_min = sigma_min or self.sigma_min
        sigma_max = sigma_max or self.sigma_max

        x = torch.randn(shape, device=device) * sigma_max

        dt = -sigma_max / num_steps

        for i in range(num_steps):
            sigma = sigma_max * (1 - i / num_steps) + sigma_min * (i / num_steps)
            sigma_tensor = torch.full((batch_size,), sigma, device=device)

            score = self.score_net(x, sigma_tensor)

            drift = -0.5 * sigma**2 * score
            diffusion = sigma * np.sqrt(-dt)

            noise = torch.randn_like(x)
            x = x + drift * dt + diffusion * noise

        return x


class AnnealedLangevinDynamics:
    """
    Annealed Langevin Dynamics (ALD) sampler.

    Progressive denoising through multiple noise scales with Langevin dynamics.
    """

    def __init__(
        self,
        model: ScoreBasedModel,
        num_steps: int = 100,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        snr: float = 0.16,
        mcmc_steps: int = 10,
    ):
        self.model = model
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.snr = snr
        self.mcmc_steps = mcmc_steps

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        device: str = "cpu",
    ) -> Tensor:
        """
        Generate samples using annealed Langevin dynamics.

        Args:
            shape: Sample shape
            device: Device

        Returns:
            Generated samples
        """
        batch_size = shape[0]

        sigmas = torch.exp(
            torch.linspace(
                np.log(self.sigma_max), np.log(self.sigma_min), self.num_steps
            )
        )

        x = torch.randn(shape, device=device) * self.sigma_max

        for sigma in sigmas:
            step_size = (self.snr * sigma) ** 2 * 2

            for _ in range(self.mcmc_steps):
                noise = torch.randn_like(x)

                score = self.model(x, torch.full((batch_size,), sigma, device=device))

                x = x + 0.5 * step_size * score + np.sqrt(step_size) * noise

        return x


class ScoreSDE(nn.Module):
    """
    Complete Score-based SDE model.

    Implements both training and sampling for score-based generative modeling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        sde_type: str = "ve",
        num_timesteps: int = 1000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sde_type = sde_type
        self.num_timesteps = num_timesteps

        self.score_net = ScoreNetwork(
            in_channels=in_channels,
            base_channels=base_channels,
        )

        self._setup_sde()

    def _setup_sde(self) -> None:
        """Set up SDE parameters based on type."""
        if self.sde_type == "ve":
            self.sigma_min = 0.01
            self.sigma_max = 1.0
        elif self.sde_type == "vp":
            self.beta_min = 0.1
            self.beta_max = 20.0

    def marginal_prob(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get marginal probability parameters at time t.

        Returns mean and std of the perturbation kernel.
        """
        if self.sde_type == "ve":
            mean = torch.zeros_like(t)
            std = t
        elif self.sde_type == "vp":
            beta = self.beta_min + t * (self.beta_max - self.beta_min)
            mean = torch.exp(-0.5 * (beta * t).cumsum(dim=0))
            std = torch.sqrt(1 - torch.exp(-(beta * t).cumsum(dim=0)))
        else:
            raise ValueError(f"Unknown SDE type: {self.sde_type}")

        while mean.dim() < 3:
            mean = mean.unsqueeze(-1)
            std = std.unsqueeze(-1)

        return mean, std

    def sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute drift and diffusion of the SDE.

        Returns:
            drift, diffusion
        """
        if self.sde_type == "ve":
            drift = torch.zeros_like(x)
            diffusion = 2 * t
        elif self.sde_type == "vp":
            beta = self.beta_min + t * (self.beta_max - self.beta_min)
            drift = -0.5 * beta.view(-1, 1, 1) * x
            diffusion = torch.sqrt(beta).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown SDE type: {self.sde_type}")

        return drift, diffusion

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict score at time t."""
        return self.score_net(x, t)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 1000,
        device: str = "cpu",
    ) -> Tensor:
        """Generate samples using reverse SDE."""
        batch_size = shape[0]

        x = torch.randn(shape, device=device)

        dt = -1.0 / num_steps
        t = torch.ones(batch_size, device=device)

        for _ in range(num_steps):
            t = (t + dt).clamp(min=0)

            drift, diffusion = self.sde(x, t)

            score = self.forward(x, t)

            x = (
                x
                - drift * dt
                + diffusion.view(-1, 1, 1, 1) * np.sqrt(-dt) * torch.randn_like(x)
            )
            x = x - diffusion.view(-1, 1, 1, 1) ** 2 * score * dt

        return x

    def training_loss(self, x: Tensor) -> Tensor:
        """Compute score matching loss."""
        batch_size = x.shape[0]
        t = torch.rand(batch_size, device=x.device)

        mean, std = self.marginal_prob(t)

        noise = torch.randn_like(x)
        perturbed_x = mean.view(-1, 1, 1, 1) * x + std.view(-1, 1, 1, 1) * noise

        predicted_score = self.forward(perturbed_x, t)

        target_score = -noise / (std.view(-1, 1, 1, 1) + 1e-5)

        loss = 0.5 * ((predicted_score - target_score) ** 2).mean()
        return loss
