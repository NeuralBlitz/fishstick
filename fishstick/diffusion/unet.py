import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: Optional[int] = None,
        dropout: float = 0.0,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_noise: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.dropout = dropout
        self.use_conv = use_conv
        self.use_noise = use_noise

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        if emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, self.out_channels),
            )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )

        if channels != self.out_channels or use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: Tensor, emb: Optional[Tensor] = None) -> Tensor:
        h = self.in_layers(x)

        if emb is not None:
            emb_out = self.emb_layers(emb)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out.unsqueeze(-1)
            h = h + emb_out

        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            self.num_heads = channels // num_head_channels
            self.num_head_channels = num_head_channels

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)

        if use_new_attention_order:
            self.attention = Attention(self.num_heads, has_proj=True)
        else:
            self.attention = Attention(self.num_heads, has_proj=True)

        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, *spatial = x.shape
        h = self.norm(x)
        qkv = self.qkv(h.reshape(b, c, -1))
        q, k, v = qkv.chunk(3, dim=1)

        out = self.attention(q, k, v)
        out = self.proj_out(out)
        return (x + out.reshape(b, c, *spatial)).reshape(b, c, *spatial)


class Attention(nn.Module):
    def __init__(self, num_heads: int, has_proj: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.has_proj = has_proj

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        b, c, length = q.shape

        q = q.reshape(b, self.num_heads, c // self.num_heads, length)
        k = k.reshape(b, self.num_heads, c // self.num_heads, length)
        v = v.reshape(b, self.num_heads, c // self.num_heads, length)

        scale = 1 / math.sqrt(c // self.num_heads)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(b, c, length)

        if self.has_proj:
            out = F.conv1d(out, out, padding=length // 2)

        return out


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, time_embed_dim: int, act: str = "silu"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_embed_dim),
            nn.SiLU() if act == "silu" else nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(t)


class TimestepEmbedSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor, emb: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool = True, padding: int = 1):
        super().__init__()
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=padding)
        else:
            self.op = nn.AvgPool2d(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        conv_padding: int = 1,
        mode: str = "nearest",
    ):
        super().__init__()
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=conv_padding)
        else:
            self.conv = None
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if self.conv:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)
            x = self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)
        return x


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (4, 2, 1),
        dropout: float = 0.0,
        channel_mult: tuple = (1, 2, 4, 4),
        num_heads: int = 8,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        use_checkpoint: bool = False,
        num_attention_blocks: Optional[int] = None,
        image_size: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_spatial_transformer = use_spatial_transformer
        self.transformer_depth = transformer_depth
        self.context_dim = context_dim

        time_embed_dim = model_channels * 4
        self.time_embed = TimestepEmbedding(model_channels, time_embed_dim)

        self.input_blocks = nn.ModuleList()

        self.input_blocks.append(
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, 3, padding=1)
            )
        )

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch * mult, time_embed_dim, dropout, ch * mult * 2)]
                ch = ch * mult
                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads,
                                num_head_channels=ch // num_heads,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, use_conv=True))
                )
                input_block_chans.append(ch)
                ds *= 2

        if use_spatial_transformer and context_dim is not None:
            self.transformer_blocks = nn.ModuleList()
            for _ in range(len(self.input_blocks)):
                self.transformer_blocks.append(
                    nn.ModuleList([nn.Module() for _ in range(transformer_depth)])
                )

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, ch * 2),
            AttentionBlock(ch, num_heads=num_heads, num_head_channels=ch // num_heads),
            ResBlock(ch, time_embed_dim, dropout, ch * 2),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        ch * mult,
                    )
                ]
                ch = ch * mult

                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads,
                                num_head_channels=ch // num_heads,
                            )
                        )

                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, use_conv=True))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        assert (y is None) == (self.context_dim is None)
        if self.context_dim is not None and y is not None:
            context = y

        t_emb = self._get_timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        h = x
        for module in self.input_blocks:
            h = module(h, emb)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = module(h, emb)

        return self.out(h)

    def _get_timestep_embedding(self, timesteps: Tensor, embedding_dim: int) -> Tensor:
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
        )
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def classifier_free_guidance(
    model: UNetModel,
    x: Tensor,
    timesteps: Tensor,
    context: Optional[Tensor] = None,
    y: Optional[Tensor] = None,
    guidance_scale: float = 7.5,
    unconditional_context: Optional[Tensor] = None,
    unconditional_embeddings: Optional[Tensor] = None,
) -> Tensor:
    """
    Apply classifier-free guidance to the model output.

    Args:
        model: The UNet model
        x: Noisy latent
        timesteps: Diffusion timesteps
        context: Conditional context (e.g., text embeddings)
        y: Optional additional conditioning
        guidance_scale: Guidance scale (higher = more guidance)
        unconditional_context: Context for unconditional generation
        unconditional_embeddings: Embeddings for unconditional generation

    Returns:
        Guided prediction
    """
    if guidance_scale == 1.0:
        return model(x, timesteps, context, y)

    if unconditional_context is None:
        unconditional_context = (
            torch.zeros_like(context) if context is not None else None
        )
    if unconditional_embeddings is None:
        unconditional_embeddings = torch.zeros_like(y) if y is not None else None

    model_output_uncond = model(
        x, timesteps, unconditional_context, unconditional_embeddings
    )
    model_output_cond = model(x, timesteps, context, y)

    return model_output_uncond + guidance_scale * (
        model_output_cond - model_output_uncond
    )
