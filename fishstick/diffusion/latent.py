from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                ResBlock(in_channels if i == 0 else out_channels, out_channels)
            )
        self.blocks = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                ResBlock(in_channels if i == 0 else out_channels, out_channels)
            )
        self.blocks = nn.Sequential(*layers)
        self.upsample = nn.ConvTranspose2d(
            out_channels, out_channels, 4, stride=2, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.upsample(x)
        return x


class VAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        channel_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        self.conv_in = nn.Conv2d(in_channels, channel_mult[0] * 64, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        channels = channel_mult[0] * 64
        for i, mult in enumerate(channel_mult):
            out_channels = mult * 64
            self.down_blocks.append(DownBlock(channels, out_channels, num_res_blocks))
            channels = out_channels

        self.mid_block = nn.Sequential(
            ResBlock(channels, channels),
            ResBlock(channels, channels),
        )

        self.norm_out = nn.GroupNorm(32, channels)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(channels, latent_channels * 2, 3, padding=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.conv_in(x)

        for down_block in self.down_blocks:
            h = down_block(h)

        h = self.mid_block(h)

        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)

        mean, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)

        z = mean + std * torch.randn_like(std)

        return z, (mean, logvar)


class VAEDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        channel_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.out_channels = out_channels

        channels = channel_mult[-1] * 64

        self.conv_in = nn.Conv2d(latent_channels, channels, 3, padding=1)

        self.mid_block = nn.Sequential(
            ResBlock(channels, channels),
            ResBlock(channels, channels),
        )

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_channels = mult * 64
            self.up_blocks.append(UpBlock(channels, out_channels, num_res_blocks))
            channels = out_channels

        self.norm_out = nn.GroupNorm(32, channels)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid_block(h)

        for up_block in self.up_blocks:
            h = up_block(h)

        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)

        return h


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        b, c, h, w = x.shape

        x_flat = x.flatten(2).transpose(1, 2)
        context_flat = context.unsqueeze(1).expand(-1, x_flat.shape[1], -1)

        q = self.to_q(x_flat)
        k = self.to_k(context_flat)
        v = self.to_v(context_flat)

        q = q.reshape(b, -1, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.reshape(b, -1, self.num_heads, self.dim_head).transpose(1, 2)
        v = v.reshape(b, -1, self.num_heads, self.dim_head).transpose(1, 2)

        scale = self.dim_head**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, -1, c)
        out = self.to_out(out)

        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        unet: nn.Module,
        vae_encoder: VAEEncoder,
        vae_decoder: VAEDecoder,
        scheduler,
        latent_scale_factor: float = 0.18215,
    ):
        super().__init__()
        self.unet = unet
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.scheduler = scheduler
        self.latent_scale_factor = latent_scale_factor

    def encode_image(self, x: Tensor) -> Tensor:
        z, _ = self.vae_encoder(x)
        return z / self.latent_scale_factor

    def decode_latent(self, z: Tensor) -> Tensor:
        z = z * self.latent_scale_factor
        return self.vae_decoder(z)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        text_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        z = self.encode_image(x)

        noise = torch.randn_like(z)
        z_noisy = self.scheduler.add_noise(z, noise, timesteps)

        model_output = self.unet(z_noisy, timesteps, context=text_embeddings)

        return model_output


class TextToImagePipeline:
    def __init__(
        self,
        unet: nn.Module,
        vae_encoder: VAEEncoder,
        vae_decoder: VAEDecoder,
        scheduler,
        text_encoder: Optional[nn.Module] = None,
        tokenizer: Optional[object] = None,
        latent_scale_factor: float = 0.18215,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
    ):
        self.unet = unet
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.latent_scale_factor = latent_scale_factor
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def encode_prompt(self, prompt: str, device: str = "cpu") -> Tensor:
        if self.tokenizer is not None and self.text_encoder is not None:
            tokens = self.tokenizer(
                prompt, return_tensors="pt", padding="max_length", max_length=77
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            embeddings = self.text_encoder(**tokens)[0]
            return embeddings
        else:
            return torch.zeros(1, 77, 768, device=device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_images: int = 1,
        device: str = "cpu",
    ) -> Tensor:
        batch_size = num_images
        text_embeddings = self.encode_prompt(prompt, device)
        unconditional_embeddings = torch.zeros_like(text_embeddings)

        latent_height = height // 8
        latent_width = width // 8

        latents = torch.randn(
            batch_size,
            4,
            latent_height,
            latent_width,
            device=device,
            dtype=text_embeddings.dtype,
        )

        self.scheduler.set_timesteps(self.num_inference_steps, device)

        for t in self.timesteps:
            latent_model_input = (
                torch.cat([latents] * 2) if self.guidance_scale > 1 else latents
            )

            text_embeds_cat = torch.cat([unconditional_embeddings, text_embeddings])

            noise_pred = self.unet(latent_model_input, t, context=text_embeds_cat)

            if self.guidance_scale > 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(noise_pred, t, latents)

        latents = latents * self.latent_scale_factor
        images = self.vae_decoder(latents)
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)

        return images
