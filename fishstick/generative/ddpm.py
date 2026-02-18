"""
Denoising Diffusion Probabilistic Models (DDPM).

Implements the diffusion process for image generation as described in:
Ho, Jain, and Abbeel (2020) "Denoising Diffusion Probabilistic Models"

The forward process gradually adds Gaussian noise to data, and the reverse
process learns to denoise to recover the original data distribution.
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


class DiffusionScheduler:
    """
    Manages the noise schedule for the diffusion process.

    Implements linear and cosine schedules for beta values.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule

        if schedule == "linear":
            self.betas = self._linear_schedule()
        elif schedule == "cosine":
            self.betas = self._cosine_schedule()
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def _linear_schedule(self) -> Tensor:
        """Linear schedule from beta_start to beta_end."""
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)

    def _cosine_schedule(self) -> Tensor:
        """Cosine schedule as proposed in Nichol & Dhariwal (2021)."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """
        Add noise to data at timestep t.

        Args:
            x0: Original data [batch, ...]
            noise: Noise to add [batch, ...]
            t: Timestep [batch]

        Returns:
            Noisy data at timestep t
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t]

        while len(sqrt_alpha_prod.shape) < len(x0.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        eta: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform one denoising step.

        Args:
            model_output: Predicted noise from model
            timestep: Current timestep
            sample: Current noisy sample
            eta: Stochasticity parameter (0 = deterministic)

        Returns:
            pred_prev_sample: Previous sample (less noisy)
            pred_original_sample: Predicted original sample
            variance: Computed variance
        """
        t = timestep
        pred_original_sample = (
            sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
        ) / self.sqrt_alphas_cumprod[t]

        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        pred_prev_sample = (
            self.posterior_mean_coef1[t] * pred_original_sample
            + self.posterior_mean_coef2[t] * sample
        )

        variance = self.posterior_variance[t]
        if eta > 0:
            noise = torch.randn_like(model_output)
            variance = variance * eta
            pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise

        return pred_prev_sample, pred_original_sample, variance

    def get_variance(self, timestep: int) -> Tensor:
        """Get variance at given timestep."""
        return self.posterior_variance[timestep]


class ResidualBlock(nn.Module):
    """Residual block with group normalization."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

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


class UNet(nn.Module):
    """
    U-Net architecture for diffusion models.

    Predicts noise to be subtracted from noisy images.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (4, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        ch = base_channels
        mult_list = list(channel_mults)
        for i, mult in enumerate(mult_list):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(ch, dropout))
                ch = out_ch
            if i < len(channel_mults) - 1:
                self.down_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))

        self.mid_block = nn.Sequential(
            ResidualBlock(ch, dropout),
            ResidualBlock(ch, dropout),
        )

        rev_mults = list(reversed(mult_list))
        for i, mult in enumerate(rev_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(ch, dropout))
                ch = out_ch
            if i < len(rev_mults) - 1:
                self.up_blocks.append(
                    nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
                )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input image [batch, channels, height, width]
            t: Timestep [batch]

        Returns:
            Predicted noise [batch, channels, height, width]
        """
        t_emb = self.time_mlp(t)

        h = self.conv_in(x)

        hs = []
        for block in self.down_blocks:
            h = block(h)
            hs.append(h)

        h = self.mid_block(h)

        for block in self.up_blocks:
            if isinstance(block, nn.ConvTranspose2d):
                h = block(h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h)

        h = self.conv_out(h)
        return h


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.

    Implements both training and sampling procedures for DDPM.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base number of channels
        channel_mults: Channel multipliers for each resolution
        num_res_blocks: Number of residual blocks per resolution
        num_timesteps: Number of diffusion timesteps
        beta_schedule: Beta schedule type
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps

        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            schedule=beta_schedule,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Training forward pass.

        Args:
            x: Clean images [batch, channels, height, width]

        Returns:
            Predicted noise, actual noise
        """
        batch_size = x.shape[0]
        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=x.device, dtype=torch.long
        )
        noise = torch.randn_like(x)
        noisy_x = self.scheduler.add_noise(x, noise, t)

        predicted_noise = self.model(noisy_x, t)
        return predicted_noise, noise

    def sample(
        self,
        shape: Tuple[int, ...],
        device: str = "cpu",
        return_intermediates: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Generate samples using the reverse diffusion process.

        Args:
            shape: Shape of samples to generate [batch, channels, height, width]
            device: Device to generate on
            return_intermediates: Whether to return all intermediate steps

        Returns:
            Generated samples, optionally with intermediate steps
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        intermediates = [] if return_intermediates else None

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            with torch.no_grad():
                predicted_noise = self.model(x, t_batch)
                x, _, _ = self.scheduler.step(predicted_noise, t, x)

            if return_intermediates and (t % 100 == 0 or t == 0):
                intermediates.append(x.clone())

        if return_intermediates:
            return intermediates
        return x

    def training_loss(self, x: Tensor) -> Tensor:
        """
        Compute the simplified training loss.

        Args:
            x: Clean images

        Returns:
            MSE loss between predicted and true noise
        """
        predicted_noise, noise = self(x)
        return F.mse_loss(predicted_noise, noise)

    def get_loss_weight(self, t: Tensor) -> Tensor:
        """
        Get loss weighting (higher weight for middle timesteps).

        Args:
            t: Timesteps

        Returns:
            Loss weights
        """
        return self.scheduler.sqrt_recipm1_alphas_cumprod[t]
