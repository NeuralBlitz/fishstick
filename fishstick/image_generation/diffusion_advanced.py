"""
Advanced Diffusion Models for image generation.

This module provides implementations of:
- DDPM: Denoising Diffusion Probabilistic Models
- DDIM: Denoising Diffusion Implicit Models (faster sampling)
- Latent Diffusion: Diffusion in latent space
- Classifier-Free Guidance: Improved sampling with guidance
- Consistency Models: Single-step generation models
"""

from typing import Optional, List, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding for diffusion models."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = h + self.time_emb(self.act(t_emb))[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block for diffusion models."""

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.reshape(b, c, h * w)

        qkv = self.qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, self.head_dim, h * w)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w)

        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * (self.head_dim**-0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(b, c, h * w)

        return x + self.proj(out).reshape(b, c, h, w)


class UNetDiffusion(nn.Module):
    """U-Net architecture for diffusion models."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dims: List[int] = [128, 256, 512],
        time_emb_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        self.time_emb = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i, (ch_in, ch_out) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(ch_in, ch_out, time_emb_dim, dropout),
                        ResidualBlock(ch_out, ch_out, time_emb_dim, dropout),
                    ]
                )
            )
            self.attn_blocks.append(AttentionBlock(ch_out, num_heads))

        self.middle_block = nn.ModuleList(
            [
                ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim, dropout),
                AttentionBlock(hidden_dims[-1], num_heads),
                ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim, dropout),
            ]
        )

        self.up_blocks = nn.ModuleList()
        for i, (ch_in, ch_out) in enumerate(
            zip(hidden_dims[::-1][:-1], hidden_dims[::-1][1:])
        ):
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(ch_in + ch_out, ch_out, time_emb_dim, dropout),
                        ResidualBlock(ch_out, ch_out, time_emb_dim, dropout),
                    ]
                )
            )
            self.attn_blocks.append(AttentionBlock(ch_out, num_heads))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[0], out_channels, 3, padding=1),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_emb(t)

        x = self.conv_in(x)

        residuals = []
        for i, (res1, res2) in enumerate(self.down_blocks):
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = self.attn_blocks[i](x)
            residuals.append(x)
            x = F.max_pool2d(x, 2)

        for block in self.middle_block:
            if isinstance(block, AttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb)

        for i, (res1, res2) in enumerate(self.up_blocks):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, residuals.pop()], dim=1)
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = self.attn_blocks[len(self.down_blocks) + i](x)

        return self.conv_out(x)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model.

    Args:
        in_channels: Number of input channels
        hidden_dims: List of hidden dimensions for U-Net
        num_timesteps: Number of diffusion timesteps
        beta_schedule: Schedule for noise variance
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [128, 256, 512],
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif beta_schedule == "cosine":
            t = torch.arange(num_timesteps + 1)
            alphas_cumprod = (
                torch.cos(((t / num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        else:
            betas = torch.linspace(0.0001, 0.02, num_timesteps)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod)
        )

        self.model = UNetDiffusion(in_channels, in_channels, hidden_dims)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        return self.model(noisy_x, t)

    def _extract(self, coeffs: Tensor, t: Tensor, x_shape: Tuple) -> Tensor:
        batch_size = t.shape[0]
        out = coeffs.to(t.device).gather(0, t)
        return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))

    @torch.no_grad()
    def sample(self, shape: Tuple, device: torch.device) -> Tensor:
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            predicted_noise = self.model(x, t_batch)

            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                x
                - (1 - alpha_t).sqrt() / (1 - alpha_cumprod_t).sqrt() * predicted_noise
            ) / alpha_t.sqrt()
            x = x + (1 - alpha_t).sqrt() * noise

        return x


class DDIM(nn.Module):
    """Denoising Diffusion Implicit Model for faster sampling.

    Args:
        in_channels: Number of input channels
        hidden_dims: List of hidden dimensions for U-Net
        num_timesteps: Number of diffusion timesteps
        eta: Parameter for variance (0 = deterministic)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [128, 256, 512],
        num_timesteps: int = 1000,
        eta: float = 0.0,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.eta = eta

        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("final_alpha_cumprod", torch.tensor(1.0))

        self.model = UNetDiffusion(in_channels, in_channels, hidden_dims)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        noise = torch.randn_like(x)
        return self.model(x, t)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        device: torch.device,
        num_steps: int = 50,
    ) -> Tensor:
        x = torch.randn(shape, device=device)
        step_size = self.num_timesteps // num_steps

        for i in reversed(range(num_steps)):
            t = i * step_size
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            predicted_noise = self.model(x, t_batch)

            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_s = self.alphas_cumprod[max(t - step_size, 0)]

            predicted_x0 = (
                x - (1 - alpha_cumprod_t).sqrt() * predicted_noise
            ) / alpha_cumprod_t.sqrt()

            direction_pointing_to_xt = (1 - alpha_cumprod_s).sqrt() * predicted_noise

            if i > 0:
                x = predicted_x0 + direction_pointing_to_xt
            else:
                x = predicted_x0

        return x


class LatentDiffusion(nn.Module):
    """Latent Diffusion Model operating in latent space.

    Args:
        encoder: VAE encoder for compressing images to latent space
        decoder: VAE decoder for reconstructing images from latent space
        diffusion: DDPM/DDIM model in latent space
        latent_channels: Number of channels in latent space
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        diffusion: nn.Module,
        latent_channels: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion = diffusion
        self.latent_channels = latent_channels

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        z = self.encode(x)
        return self.diffusion(z, t)

    @torch.no_grad()
    def sample(self, shape: Tuple, device: torch.device) -> Tensor:
        z = self.diffusion.sample(shape, device)
        return self.decode(z)


class StableDiffusion(nn.Module):
    """Stable Diffusion model with text conditioning.

    Args:
        vae: VAE for image encoding/decoding
        unet: U-Net for noise prediction
        text_encoder: Text encoder for conditioning
        num_timesteps: Number of diffusion steps
    """

    def __init__(
        self,
        vae: nn.Module,
        unet: nn.Module,
        text_encoder: nn.Module,
        num_timesteps: int = 1000,
    ):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.num_timesteps = num_timesteps

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
    ) -> Tensor:
        batch_size = len(prompts)
        device = next(self.parameters()).device

        text_embeddings = self.text_encoder(prompts)

        latents = torch.randn(batch_size, 4, height // 8, width // 8, device=device)

        for t in reversed(range(num_inference_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            noise_pred = self.unet(latents, t_batch, text_embeddings)

            if guidance_scale > 1:
                noise_pred_uncond = self.unet(
                    latents, t_batch, torch.zeros_like(text_embeddings)
                )
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred - noise_pred_uncond
                )

            latents = self._step(latents, noise_pred, t)

        images = self.vae.decode(latents)
        return images

    def _step(self, latents: Tensor, noise_pred: Tensor, t: int) -> Tensor:
        alpha_cumprod = 1 - t / self.num_timesteps
        alpha_t = alpha_cumprod

        pred_x0 = (
            latents - (1 - alpha_cumprod).sqrt() * noise_pred
        ) / alpha_cumprod.sqrt()

        return pred_x0


class ClassifierFreeGD(nn.Module):
    """Classifier-Free Guidance for improved generation quality.

    Args:
        model: Base diffusion model
        unconditional_prob: Probability of dropping conditioning
    """

    def __init__(
        self,
        model: nn.Module,
        unconditional_prob: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.unconditional_prob = unconditional_prob

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        condition: Optional[Tensor] = None,
        guidance_scale: float = 7.5,
    ) -> Tensor:
        if condition is not None and self.training:
            mask = torch.rand(x.shape[0], device=x.device) < self.unconditional_prob
            condition[mask] = 0

        noise_pred_cond = self.model(x, t, condition)

        if condition is None or not self.training:
            return noise_pred_cond

        noise_pred_uncond = self.model(x, t, torch.zeros_like(condition))

        return noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )


class ConsistencyModel(nn.Module):
    """Consistency Models for single-step or few-step generation.

    Args:
        in_channels: Number of input channels
        hidden_dims: Hidden dimensions for the model
        out_channels: Number of output channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [128, 256, 512],
        out_channels: int = 3,
    ):
        super().__init__()
        self.time_emb_dim = 256

        self.time_emb = nn.Sequential(
            TimeEmbedding(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        self.conv_in = nn.Conv2d(in_channels + 1, hidden_dims[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        for i, (ch_in, ch_out) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.down_blocks.append(ResidualBlock(ch_in, ch_out, self.time_emb_dim))

        self.up_blocks = nn.ModuleList()
        for i, (ch_in, ch_out) in enumerate(
            zip(hidden_dims[::-1][:-1], hidden_dims[::-1][1:])
        ):
            self.up_blocks.append(
                ResidualBlock(ch_in + ch_out, ch_out, self.time_emb_dim)
            )

        self.conv_out = nn.Conv2d(hidden_dims[0], out_channels, 3, padding=1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_emb(t)
        t_scaled = t_emb / 1000.0

        x = torch.cat([x, t_scaled.unsqueeze(-1).unsqueeze(-1)], dim=1)
        x = self.conv_in(x)

        residuals = []
        for block in self.down_blocks:
            x = block(x, t_emb)
            residuals.append(x)
            x = F.max_pool2d(x, 2)

        for block in self.up_blocks:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x, t_emb)

        return self.conv_out(x)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        device: torch.device,
        num_steps: int = 1,
    ) -> Tensor:
        x = torch.randn(shape, device=device)

        sigma_max = 80.0
        sigma_min = 0.002

        for i in range(num_steps):
            t = sigma_max * ((sigma_min / sigma_max) ** (i / num_steps))
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.float32)

            x = x - self.forward(x, t_tensor) * (t - sigma_min / sigma_max) / t

        return x
