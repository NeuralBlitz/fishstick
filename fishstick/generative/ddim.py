"""
Denoising Diffusion Implicit Models (DDIM).

Implements the faster sampling variant of DDPM as described in:
Song, Meng, and Ermon (2021) "Denoising Diffusion Implicit Models"

DDIM enables much faster sampling by using a non-Markovian reverse process
that produces high-quality samples in fewer steps.
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from .ddpm import DiffusionScheduler, UNet, TimeEmbedding


class DDIMScheduler:
    """
    Scheduler for DDIM sampling.

    Implements the non-Markovian reverse process for faster sampling.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
        eta: float = 0.0,
    ):
        self.num_timesteps = num_timesteps
        self.eta = eta

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / num_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.final_alpha_cumprod = torch.tensor(1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def set_timesteps(self, num_steps: int, device: str = "cpu") -> Tensor:
        """
        Set the sampling timesteps for DDIM.

        Args:
            num_steps: Number of sampling steps
            device: Device for timesteps

        Returns:
            Timestep array
        """
        step_ratio = self.num_timesteps // num_steps
        timesteps = (np.arange(0, num_steps) * step_ratio).round()[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(device)
        return timesteps

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        eta: Optional[float] = None,
    ) -> Tensor:
        """
        Perform one DDIM sampling step.

        Args:
            model_output: Predicted noise from model
            timestep: Current timestep
            sample: Current sample
            eta: Stochasticity parameter

        Returns:
            Previous sample
        """
        if eta is None:
            eta = self.eta

        prev_timestep = timestep - self.num_timesteps // len(self.sqrt_alphas_cumprod)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        variance = (
            (1 - alpha_prod_t_prev)
            / (1 - alpha_prod_t)
            * (1 - alpha_prod_t / alpha_prod_t_prev)
        )

        std = variance**0.5
        if eta > 0:
            noise = torch.randn_like(model_output)
            variance_noise = std * noise
        else:
            variance_noise = 0

        pred_sample_direction = (
            1 - alpha_prod_t_prev - eta * variance
        ) ** 0.5 * model_output

        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )

        if eta > 0:
            prev_sample = prev_sample + variance_noise

        return prev_sample


class DDIM(nn.Module):
    """
    Denoising Diffusion Implicit Model.

    Faster sampling version of DDPM using non-Markovian reverse process.

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

        self.scheduler = DDIMScheduler(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
        )

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """
        Predict noise from noisy input.

        Args:
            x: Noisy input
            t: Timestep (if None, computed from x)

        Returns:
            Predicted noise
        """
        if t is None:
            batch_size = x.shape[0]
            t = torch.randint(
                0, self.num_timesteps, (batch_size,), device=x.device, dtype=torch.long
            )
        return self.model(x, t)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        device: str = "cpu",
        eta: float = 0.0,
        return_intermediates: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Generate samples using DDIM sampling.

        Uses fewer steps than DDPM for much faster generation.

        Args:
            shape: Shape of samples [batch, channels, height, width]
            num_steps: Number of sampling steps (typically 20-100)
            device: Device to generate on
            eta: Stochasticity parameter (0 = deterministic)
            return_intermediates: Whether to return intermediate steps

        Returns:
            Generated samples
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        timesteps = self.scheduler.set_timesteps(num_steps, device)

        intermediates = [] if return_intermediates else None

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            model_output = self.model(x, t_batch)
            x = self.scheduler.step(model_output, t, x, eta)

            if return_intermediates and (
                i % (num_steps // 10) == 0 or i == num_steps - 1
            ):
                intermediates.append(x.clone())

        if return_intermediates:
            return intermediates
        return x

    @torch.no_grad()
    def interpolate(
        self,
        x0: Tensor,
        x1: Tensor,
        alpha: float = 0.5,
        num_steps: int = 50,
    ) -> Tensor:
        """
        Interpolate between two images using DDIM.

        Args:
            x0: First image
            x1: Second image
            alpha: Interpolation coefficient
            num_steps: Number of sampling steps

        Returns:
            Interpolated image
        """
        assert x0.shape == x1.shape

        t = torch.full(
            (x0.shape[0],), self.num_timesteps - 1, device=x0.device, dtype=torch.long
        )

        noise0 = self.scheduler.add_noise(x0, torch.randn_like(x0), t)
        noise1 = self.scheduler.add_noise(x1, torch.randn_like(x1), t)

        x_interp = (1 - alpha) * noise0 + alpha * noise1

        timesteps = self.scheduler.set_timesteps(num_steps, x0.device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((x0.shape[0],), t, device=x0.device, dtype=torch.long)
            model_output = self.model(x_interp, t_batch)
            x_interp = self.scheduler.step(model_output, t, x_interp)

        return x_interp

    def training_loss(self, x: Tensor) -> Tensor:
        """
        Compute the training loss.

        Args:
            x: Clean images

        Returns:
            MSE loss
        """
        batch_size = x.shape[0]
        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=x.device, dtype=torch.long
        )
        noise = torch.randn_like(x)

        sqrt_alpha_prod = self.scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_prod = self.scheduler.sqrt_one_minus_alphas_cumprod[t]

        while len(sqrt_alpha_prod.shape) < len(x.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_x = sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise

        predicted_noise = self.model(noisy_x, t)
        return F.mse_loss(predicted_noise, noise)


class ClassifierFreeGD(nn.Module):
    """
    Classifier-Free Guidance Diffusion.

    Enables conditional generation without a classifier.
    """

    def __init__(
        self,
        model: nn.Module,
        guidance_scale: float = 7.5,
    ):
        super().__init__()
        self.model = model
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[Tensor] = None,
        num_steps: int = 50,
        device: str = "cpu",
    ) -> Tensor:
        """
        Generate samples with classifier-free guidance.

        Args:
            shape: Sample shape
            condition: Conditioning tensor
            num_steps: Number of sampling steps
            device: Device

        Returns:
            Generated samples
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        timesteps = self.model.scheduler.set_timesteps(num_steps, device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if condition is not None:
                cond_input = torch.cat([x, condition], dim=1)
                uncond_input = torch.cat([x, torch.zeros_like(condition)], dim=1)

                model_output_cond = self.model.model(cond_input, t_batch)
                model_output_uncond = self.model.model(uncond_input, t_batch)

                model_output = model_output_uncond + self.guidance_scale * (
                    model_output_cond - model_output_uncond
                )
            else:
                model_output = self.model.model(x, t_batch)

            x = self.model.scheduler.step(model_output, t, x)

        return x
