"""
Text-to-Image generation models.

This module provides text-conditioned image generation:
- CLIP-based generation: Using CLIP for text-image alignment
- Diffusion-based T2I: Diffusion models conditioned on text
- Multi-modal conditioning: Combines text and other modalities
- Prompt-based generation: Flexible prompt engineering
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


class TextEncoder(nn.Module):
    """Text encoder for text-to-image models.

    Args:
        vocab_size: Size of vocabulary
        embed_dim: Dimension of text embeddings
        max_length: Maximum sequence length
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        max_length: int = 77,
        num_layers: int = 12,
        num_heads: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Encode text tokens to embeddings.

        Args:
            tokens: Token IDs of shape (batch, seq_len)
            mask: Attention mask of shape (batch, seq_len)
        """
        b, seq_len = tokens.shape

        positions = (
            torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(b, -1)
        )

        embeddings = self.token_embedding(tokens) + self.position_embedding(positions)

        output = self.transformer(
            embeddings, src_key_padding_mask=~mask if mask is not None else None
        )

        return self.layer_norm(output)


class CLIPTextEncoder(nn.Module):
    """CLIP-compatible text encoder.

    Args:
        vocab_size: Size of vocabulary
        embed_dim: Dimension of output embeddings
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        max_length: int = 77,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_length, embed_dim) * 0.02
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=12,
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """Encode text to CLIP embeddings."""
        b, seq_len = tokens.shape

        x = self.token_embedding(tokens) + self.position_embedding[:, :seq_len]

        x = self.transformer(x)

        x = self.layer_norm(x)

        return x


class TextConditioning(nn.Module):
    """Text conditioning module for diffusion models.

    Args:
        text_dim: Dimension of text embeddings
        time_dim: Dimension of time embeddings
        hidden_dim: Hidden dimension for cross-attention
    """

    def __init__(
        self,
        text_dim: int = 512,
        time_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.text_projection = nn.Linear(text_dim, hidden_dim)

        self.alphas = nn.Parameter(torch.ones(1))

    def forward(
        self,
        time_emb: Tensor,
        text_emb: Tensor,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Process conditioning inputs.

        Args:
            time_emb: Time embedding
            text_emb: Text embeddings from encoder
            text_mask: Mask for text tokens
        """
        time_h = self.time_mlp(time_emb)

        text_h = self.text_projection(text_emb)

        time_h = time_h.unsqueeze(1)

        fused = time_h + text_h

        return fused


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning.

    Args:
        query_dim: Dimension of query
        context_dim: Dimension of context (text)
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        query_dim: int = 512,
        context_dim: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)

        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply cross-attention.

        Args:
            x: Query tensor (image features)
            context: Context tensor (text embeddings)
            mask: Optional attention mask
        """
        b, c, h, w = x.shape

        x_flat = x.flatten(2).transpose(1, 2)

        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, -1, c)

        return self.to_out(out).transpose(1, 2).reshape(b, c, h, w)


class TextToImageDiffusion(nn.Module):
    """Text-to-image diffusion model.

    Args:
        num_channels: Number of image channels
        text_dim: Dimension of text embeddings
        latent_dim: Dimension of latent representation
        num_steps: Number of diffusion steps
    """

    def __init__(
        self,
        num_channels: int = 3,
        text_dim: int = 512,
        latent_dim: int = 4,
        num_steps: int = 1000,
    ):
        super().__init__()
        self.num_steps = num_steps

        self.text_encoder = CLIPTextEncoder(embed_dim=text_dim)

        time_dim = 256
        self.time_embed = TimeEmbedding(time_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, latent_dim * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_dim * 4, latent_dim * 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_dim * 8, latent_dim * 16, 3, stride=2, padding=1),
            nn.SiLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 16, latent_dim * 8, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(latent_dim * 8, latent_dim * 4, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_dim * 4, latent_dim * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_dim * 4, num_channels, 3, padding=1),
        )

        self.unet = self._build_unet(latent_dim * 16, text_dim, time_dim)

    def _build_unet(
        self,
        channels: int,
        text_dim: int,
        time_dim: int,
    ) -> nn.Module:
        """Build U-Net with cross-attention."""
        return nn.ModuleDict(
            {
                "down": nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(channels, channels, 3, padding=1),
                            nn.SiLU(),
                            nn.Conv2d(channels, channels, 3, padding=1),
                            nn.SiLU(),
                        ),
                        nn.Sequential(
                            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                            nn.SiLU(),
                            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
                            nn.SiLU(),
                        ),
                    ]
                ),
                "up": nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                channels * 2, channels, 4, stride=2, padding=1
                            ),
                            nn.SiLU(),
                            nn.Conv2d(channels, channels, 3, padding=1),
                            nn.SiLU(),
                        ),
                        nn.Sequential(
                            nn.Conv2d(channels, channels, 3, padding=1),
                            nn.SiLU(),
                        ),
                    ]
                ),
                "cross_attn": CrossAttention(channels, text_dim),
            }
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        text_tokens: Tensor,
    ) -> Tensor:
        """Denoise image given text conditioning.

        Args:
            x: Noisy image
            t: Timestep
            text_tokens: Text token IDs
        """
        time_emb = self.time_embed(t)
        text_emb = self.text_encoder(text_tokens)

        h = self.encoder(x)

        for down_block in self.unet["down"]:
            h = down_block(h)

        h = self.unet["cross_attn"](h, text_emb)

        for up_block in self.unet["up"]:
            h = up_block(h)

        return self.decoder(h)

    @torch.no_grad()
    def sample(
        self,
        text_tokens: Tensor,
        shape: Tuple[int, int, int, int],
        num_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Tensor:
        """Generate image from text.

        Args:
            text_tokens: Text token IDs
            shape: Shape of generated image (B, C, H, W)
            num_steps: Number of sampling steps
            guidance_scale: Classifier-free guidance scale
        """
        b = shape[0]
        x = torch.randn(shape, device=text_tokens.device)

        text_emb = self.text_encoder(text_tokens)

        for i in reversed(range(num_steps)):
            t = torch.full((b,), i * self.num_steps // num_steps, device=x.device)

            time_emb = self.time_embed(t)

            noise_pred = self._denoise_step(x, t, text_emb, guidance_scale)

            alpha = 1 - i / num_steps
            x = x - noise_pred * (1 - alpha) ** 0.5

        return x

    def _denoise_step(
        self,
        x: Tensor,
        t: Tensor,
        text_emb: Tensor,
        guidance: float,
    ) -> Tensor:
        """Single denoising step with guidance."""
        time_emb = self.time_embed(t)

        h = self.encoder(x)

        for down_block in self.unet["down"]:
            h = down_block(h)

        h = self.unet["cross_attn"](h, text_emb)

        for up_block in self.unet["up"]:
            h = up_block(h)

        noise_pred = self.decoder(h)

        return noise_pred


class PromptEmbedder(nn.Module):
    """Prompt embedder for text-to-image models.

    Args:
        embed_dim: Dimension of embeddings
        max_length: Maximum prompt length
    """

    def __init__(
        self,
        embed_dim: int = 768,
        max_length: int = 77,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, prompt_embeds: Tensor) -> Tensor:
        """Process prompt embeddings."""
        return self.fc(prompt_embeds)


class ClassifierFreeGuidance(nn.Module):
    """Classifier-free guidance for text-to-image models.

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

    @torch.no_grad()
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        text_tokens: Tensor,
        guidance_scale: float = 7.5,
    ) -> Tensor:
        """Generate with classifier-free guidance.

        Args:
            x: Noisy image
            t: Timestep
            text_tokens: Text tokens
            guidance_scale: Guidance scale
        """
        batch_size = x.shape[0]

        mask = torch.rand(batch_size, device=x.device) < self.unconditional_prob
        text_tokens_uncond = text_tokens.clone()
        text_tokens_uncond[mask] = 0

        cond_output = self.model(x, t, text_tokens)
        uncond_output = self.model(x, t, text_tokens_uncond)

        return uncond_output + guidance_scale * (cond_output - uncond_output)


class MultiPromptGenerator(nn.Module):
    """Multi-prompt text-to-image generator.

    Args:
        text_encoder: Text encoder model
        diffusion_model: Diffusion model
        num_prompts: Number of prompts to combine
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        diffusion_model: nn.Module,
        num_prompts: int = 2,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.diffusion = diffusion_model
        self.num_prompts = num_prompts

        self.weight_net = nn.Sequential(
            nn.Linear(512 * num_prompts, 256),
            nn.ReLU(),
            nn.Linear(256, num_prompts),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        prompts: List[Tensor],
        shape: Tuple[int, int, int, int],
    ) -> Tensor:
        """Generate image from multiple prompts.

        Args:
            prompts: List of text token tensors
            shape: Output shape
        """
        text_embs = [self.text_encoder(p) for p in prompts]

        combined_emb = torch.cat(text_embs, dim=-1)
        weights = self.weight_net(combined_emb)

        weighted_emb = sum(w * emb for w, emb in zip(weights.unbind(-1), text_embs))

        return self.diffusion.sample(weighted_emb.unsqueeze(1), shape)


class ImageVariationGenerator(nn.Module):
    """Image variation generator using CLIP.

    Args:
        image_dim: Dimension of image features
        text_dim: Dimension of text embeddings
    """

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 512,
    ):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(image_dim, text_dim),
            nn.Tanh(),
        )

        self.text_projection = nn.Linear(text_dim, text_dim)

    def forward(
        self,
        images: Tensor,
        num_variations: int = 4,
    ) -> Tensor:
        """Generate variations of input images.

        Args:
            images: Input images
            num_variations: Number of variations to generate
        """
        image_features = self.image_encoder(images)

        text_emb = self.text_projection(image_features)

        return text_emb.repeat(num_variations, 1)


class StableDiffusionXL(nn.Module):
    """Stable Diffusion XL model for high-quality text-to-image.

    Args:
        latent_dim: Dimension of latent space
        text_dim: Dimension of text embeddings
        guidance_scale: Default guidance scale
    """

    def __init__(
        self,
        latent_dim: int = 4,
        text_dim: int = 768,
        guidance_scale: float = 7.5,
    ):
        super().__init__()
        self.guidance_scale = guidance_scale

        self.text_encoder = CLIPTextEncoder(embed_dim=text_dim)

        self.unet = self._build_unet(latent_dim, text_dim)

        self.vae = VAEEncoderDecoder(latent_dim)

    def _build_unet(
        self,
        latent_dim: int,
        text_dim: int,
    ) -> nn.Module:
        """Build U-Net for latent diffusion."""
        channels = latent_dim * 16

        return nn.ModuleDict(
            {
                "down": nn.ModuleList(
                    [
                        nn.Conv2d(channels, channels, 3, padding=1),
                        nn.SiLU(),
                        nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                        nn.SiLU(),
                    ]
                ),
                "mid": nn.Sequential(
                    nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
                    nn.SiLU(),
                ),
                "up": nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            channels * 2, channels, 4, stride=2, padding=1
                        ),
                        nn.SiLU(),
                        nn.Conv2d(channels, latent_dim, 3, padding=1),
                    ]
                ),
            }
        )

    @torch.no_grad()
    def forward(
        self,
        text: Tensor,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 30,
    ) -> Tensor:
        """Generate image from text.

        Args:
            text: Text tokens
            height: Output height
            width: Output width
            num_steps: Number of diffusion steps
        """
        latent_h, latent_w = height // 8, width // 8

        x = torch.randn(1, 4, latent_h, latent_w, device=text.device)

        text_emb = self.text_encoder(text)

        for _ in range(num_steps):
            x = self._denoise(x, text_emb)

        return self.vae.decode(x)

    def _denoise(self, x: Tensor, text_emb: Tensor) -> Tensor:
        """Denoise single step."""
        h = x
        for layer in self.unet["down"]:
            h = layer(h)

        h = self.unet["mid"](h)

        for layer in self.unet["up"]:
            h = layer(h)

        return h


class VAEEncoderDecoder(nn.Module):
    """VAE encoder-decoder for latent space.

    Args:
        latent_dim: Dimension of latent space
    """

    def __init__(self, latent_dim: int = 4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, latent_dim * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_dim * 4, latent_dim * 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_dim * 8, latent_dim * 8, 3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim * 8, latent_dim * 8, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(latent_dim * 8, latent_dim * 4, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_dim * 4, 3, 3, padding=1),
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encode image to latent."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to image."""
        return self.decoder(z)
