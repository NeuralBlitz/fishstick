"""
Latent Diffusion Models (LDM).

Implements latent space diffusion as described in:
Rombach et al. (2022) "High-Resolution Image Synthesis with Latent Diffusion Models"

LDMs perform diffusion in a learned latent space, enabling efficient high-resolution
image generation with dramatically reduced compute requirements.
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from .ddpm import DiffusionScheduler, UNet
from .ddim import DDIMScheduler


class AutoencoderKL(nn.Module):
    """
    Variational Autoencoder with latent space for LDM.

    Learns a compact latent representation suitable for diffusion.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        hidden_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        self.encoder = self._build_encoder(
            in_channels, hidden_channels, channel_mults, num_res_blocks
        )
        self.decoder = self._build_decoder(
            latent_channels, hidden_channels, channel_mults, num_res_blocks
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    def _build_encoder(
        self,
        in_channels: int,
        hidden_channels: int,
        channel_mults: Tuple[int, ...],
        num_res_blocks: int,
    ) -> nn.Module:
        """Build encoder network."""
        modules = [nn.Conv2d(in_channels, hidden_channels, 3, padding=1)]

        in_ch = hidden_channels
        for i, mult in enumerate(channel_mults):
            out_ch = hidden_channels * mult
            for _ in range(num_res_blocks):
                modules.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch
            if i < len(channel_mults) - 1:
                modules.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))

        modules.extend(
            [
                nn.GroupNorm(32, in_ch),
                nn.SiLU(),
                nn.Conv2d(in_ch, 2 * latent_channels, 3, padding=1),
            ]
        )

        return nn.Sequential(*modules)

    def _build_decoder(
        self,
        latent_channels: int,
        hidden_channels: int,
        channel_mults: Tuple[int, ...],
        num_res_blocks: int,
    ) -> nn.Module:
        """Build decoder network."""
        modules = [
            nn.Conv2d(
                latent_channels, hidden_channels * channel_mults[-1], 3, padding=1
            )
        ]

        in_ch = hidden_channels * channel_mults[-1]
        for i in reversed(range(len(channel_mults))):
            mult = channel_mults[i]
            out_ch = hidden_channels * mult

            for _ in range(num_res_blocks + 1):
                modules.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch

            if i > 0:
                modules.append(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
                )

        modules.extend(
            [
                nn.GroupNorm(32, in_ch),
                nn.SiLU(),
                nn.Conv2d(in_ch, in_channels, 3, padding=1),
            ]
        )

        return nn.Sequential(*modules)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input to latent distribution.

        Args:
            x: Input [B, C, H, W]

        Returns:
            mean, logvar of latent distribution
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = moments.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent to output.

        Args:
            z: Latent [B, latent_channels, H, W]

        Returns:
            Reconstructed output
        """
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full encode-decode pass.

        Args:
            x: Input

        Returns:
            reconstructed, mean, logvar
        """
        mean, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)

        reconstructed = self.decode(z)
        return reconstructed, mean, logvar


class ResidualBlock(nn.Module):
    """Residual block with group normalization."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """Self-attention block for latent diffusion."""

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)

        qkv = self.qkv(h).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)

        h = (attn @ v).reshape(B, H * W, C)
        h = self.proj(h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)

        return x + h


class UNetLatent(nn.Module):
    """
    U-Net for latent diffusion.

    Modified for working in latent space with cross-attention conditioning.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 320,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (4, 2),
        context_dim: int = 768,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(ch, out_ch))
                if i in attention_resolutions:
                    self.down_blocks.append(
                        AttentionBlock(out_ch, num_heads=out_ch // 64)
                    )
                ch = out_ch
            if i < len(channel_mults) - 1:
                self.down_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))

        self.mid_block = nn.Sequential(
            ResidualBlock(ch, ch),
            AttentionBlock(ch, num_heads=ch // 64),
            ResidualBlock(ch, ch),
        )

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(ch, out_ch))
                if i in attention_resolutions:
                    self.up_blocks.append(
                        AttentionBlock(out_ch, num_heads=out_ch // 64)
                    )
                ch = out_ch
            if i < len(channel_mults) - 1:
                self.up_blocks.append(
                    nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
                )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with optional context conditioning.

        Args:
            x: Latent input
            timesteps: Diffusion timesteps
            context: Conditioning context

        Returns:
            Predicted noise
        """
        t_emb = self.time_embed(self._get_timestep_embedding(timesteps))

        h = self.conv_in(x)
        hs = []

        for module in self.down_blocks:
            h = module(h)
            hs.append(h)

        h = self.mid_block(h)

        for module in self.up_blocks:
            if isinstance(module, nn.ConvTranspose2d):
                h = module(h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h)

        h = self.conv_out(h)
        return h

    def _get_timestep_embedding(self, timesteps: Tensor) -> Tensor:
        half_dim = self.base_channels // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model.

    Performs diffusion in a learned latent space for efficient generation.

    Args:
        in_channels: Input channels (typically 3 for RGB)
        latent_channels: Latent space channels
        base_channels: Base number of channels
        num_timesteps: Number of diffusion steps
        beta_schedule: Noise schedule
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 320,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        context_dim: int = 768,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_timesteps = num_timesteps

        self.autoencoder = AutoencoderKL(
            in_channels=in_channels,
            latent_channels=latent_channels,
        )

        self.unet = UNetLatent(
            in_channels=latent_channels,
            out_channels=latent_channels,
            base_channels=base_channels,
            context_dim=context_dim,
        )

        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            schedule=beta_schedule,
        )

        self.vq_model = None

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode image to latent space.

        Args:
            x: Input image

        Returns:
            Latent representation
        """
        with torch.no_grad():
            mean, logvar = self.autoencoder.encode(x)
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent to image.

        Args:
            z: Latent representation

        Returns:
            Reconstructed image
        """
        with torch.no_grad():
            return self.autoencoder.decode(z)

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass (predict noise in latent space).

        Args:
            x: Input in latent space
            t: Timesteps

        Returns:
            Predicted noise
        """
        if t is None:
            batch_size = x.shape[0]
            t = torch.randint(
                0, self.num_timesteps, (batch_size,), device=x.device, dtype=torch.long
            )
        return self.unet(x, t)

    def training_loss(self, x: Tensor) -> Tensor:
        """
        Compute training loss.

        Args:
            x: Clean images

        Returns:
            Noise prediction loss
        """
        z = self.encode(x)

        batch_size = z.shape[0]
        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=z.device, dtype=torch.long
        )

        noise = torch.randn_like(z)
        noisy_z = self.scheduler.add_noise(z, noise, t)

        predicted_noise = self.forward(noisy_z, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        context: Optional[Tensor] = None,
        num_steps: int = 50,
        device: str = "cpu",
    ) -> Tensor:
        """
        Generate samples.

        Args:
            shape: Sample shape (excluding latent channels)
            context: Conditioning context
            num_steps: Number of sampling steps
            device: Device

        Returns:
            Generated images
        """
        latent_shape = (
            shape[0],
            self.latent_channels,
            shape[2] // 8,
            shape[3] // 8,
        )

        z = torch.randn(latent_shape, device=device)

        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps, device=device
        ).long()

        scheduler = DDIMScheduler(num_timesteps=self.num_timesteps)
        scheduler.eta = 0.0

        for i, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0).expand(shape[0])

            predicted_noise = self.unet(z, t_batch, context)

            z = scheduler.step(predicted_noise, t, z)

        images = self.decode(z)
        return images


class VQModel(nn.Module):
    """
    Vector Quantized Autoencoder.

    Alternative to VAE for LDM using discrete latent codes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 3,
        num_embeddings: int = 8192,
        embedding_dim: int = 256,
        hidden_channels: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.encoder = self._build_encoder(in_channels, hidden_channels)
        self.decoder = self._build_decoder(hidden_channels)

        self.quantize = VectorQuantizer(num_embeddings, embedding_dim)
        self.quant_conv = nn.Conv2d(hidden_channels, embedding_dim, 1)
        self.post_quant_conv = nn.Conv2d(embedding_dim, hidden_channels, 1)

    def _build_encoder(self, in_channels: int, hidden_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
        )

    def _build_decoder(self, hidden_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_channels, self.in_channels, 4, stride=2, padding=1
            ),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.encoder(x)
        h = self.quant_conv(h)

        quant, indices, commit_loss = self.quantize(h)

        quant = self.post_quant_conv(quant)
        reconstructed = self.decoder(quant)

        return reconstructed, commit_loss, indices


class VectorQuantizer(nn.Module):
    """Vector quantizer using codebook."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, C, H, W = z.shape
        z_flattened = z.permute(0, 2, 3, 1).reshape(-1, C)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        indices = torch.argmin(d, dim=1)
        quantized = self.embedding(indices).reshape(B, H, W, C).permute(0, 3, 1, 2)

        commit_loss = F.mse_loss(quantized.detach(), z)
        quantized = z + (quantized - z).detach()

        return quantized, indices, commit_loss
