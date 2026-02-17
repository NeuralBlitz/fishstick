"""
Advanced Vision Transformers

Vision Transformer (ViT), DeiT, Swin Transformer, and Conv-transformer (CvT) implementations.
"""

from typing import Optional, Tuple, List
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with optional overlapping patches."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        overlap: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size**2

        if overlap:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size // 2,
                padding=patch_size // 4,
            )
        else:
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
            )

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, h, w


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with optional relative position bias."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer (ViT)

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: Use bias in QKV
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        x, H, W = self.patch_embed(x)
        B, N, C = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        return self.head(x)


class DeiT(nn.Module):
    """Data-efficient Image Transformer (DeiT)

    Adds distillation token support to ViT.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_distillation: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_distillation = use_distillation

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            n_tokens = n_patches + 2
        else:
            self.dist_token = None
            n_tokens = n_patches + 1

        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        if use_distillation:
            self.head_dist = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if use_distillation:
            trunc_normal_(self.dist_token, std=0.02)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x, H, W = self.patch_embed(x)
        B, N, C = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.use_distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x_cls = x[:, 0]
        if self.use_distillation:
            x_dist = x[:, 1]
            return self.head(x_cls), self.head_dist(x_dist)
        return (self.head(x_cls),)


class ShiftedWindowAttention(nn.Module):
    """Shifted Window Self-Attention for Swin Transformer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 3,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = (embed_dim // num_heads) ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, L, C = x.shape
        assert L == H * W, "Input size mismatch"

        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        qkv = (
            self.qkv(x_windows)
            .reshape(-1, self.window_size**2, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self._get_relative_position_bias()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(-1, self.window_size, self.window_size, C)
        )
        x = x.view(-1, Hp, Wp, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def window_partition(self, x: Tensor, window_size: int) -> Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size, window_size, C)
        )
        return windows

    def _get_relative_position_bias(self) -> Tensor:
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return (
            self.relative_position_bias_table[relative_position_index.view(-1)]
            .view(
                self.window_size * self.window_size,
                self.window_size * self.window_size,
                -1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with shifted window attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ShiftedWindowAttention(
            embed_dim, num_heads, window_size, shift_size, qkv_bias, attn_drop, drop
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, drop=drop)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depths: Number of blocks per stage
        num_heads: Number of attention heads per stage
        window_size: Window size for attention
        mlp_ratio: MLP hidden dim ratio
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            dim = int(embed_dim * 2**i)
            layer = nn.ModuleList(
                [
                    SwinTransformerBlock(
                        dim,
                        num_heads[i],
                        window_size,
                        window_size // 2 if i > 0 else 0,
                        mlp_ratio,
                        qkv_bias,
                        drop_rate,
                        attn_drop_rate,
                    )
                    for _ in range(depths[i])
                ]
            )
            if i < self.num_layers - 1:
                downsample = nn.Linear(dim, int(embed_dim * 2 ** (i + 1)))
                layer = nn.ModuleDict({"blocks": layer, "downsample": downsample})
            self.layers.append(layer)

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)), num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            for block in layer["blocks"]:
                x = block(x, H, W)

        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


class CvT(nn.Module):
    """Conv-transformer (CvT) with hierarchical structure and conv embeddings."""

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: Tuple[int, ...] = (64, 192, 384),
        num_heads: Tuple[int, ...] = (1, 3, 6),
        mlp_ratios: Tuple[int, ...] = (4, 4, 4),
        qkv_bias: bool = False,
        depths: Tuple[int, ...] = (1, 2, 10),
        sr_ratios: Tuple[int, ...] = (8, 4, 2),
    ):
        super().__init__()
        self.num_classes = num_classes

        self.stages = nn.ModuleList()
        for i in range(3):
            if i == 0:
                patch_embed = nn.Sequential(
                    nn.Conv2d(
                        in_chans, embed_dims[i] // 2, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(embed_dims[i] // 2),
                    nn.GELU(),
                    nn.Conv2d(
                        embed_dims[i] // 2,
                        embed_dims[i],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(embed_dims[i]),
                )
                num_patches = (img_size // 4) ** 2
            else:
                patch_embed = nn.Sequential(
                    nn.Conv2d(
                        embed_dims[i - 1],
                        embed_dims[i],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(embed_dims[i]),
                )
                num_patches = (img_size // (4 * 2**i)) ** 2

            cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[i]))
            pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims[i]))

            blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        embed_dims[i], num_heads[i], mlp_ratios[i], qkv_bias
                    )
                    for _ in range(depths[i])
                ]
            )
            norm = nn.LayerNorm(embed_dims[i])

            stage = nn.ModuleDict(
                {
                    "patch_embed": patch_embed,
                    "cls_token": cls_token,
                    "pos_embed": pos_embed,
                    "blocks": blocks,
                    "norm": norm,
                }
            )
            self.stages.append(stage)

        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage["patch_embed"](x)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)

            cls_tokens = stage["cls_token"].expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            x = x + stage["pos_embed"]

            for block in stage["blocks"]:
                x = block(x)

            x = stage["norm"](x)
            x = x[:, 0]

        return self.head(x)


def create_vit(pretrained: bool = False, **kwargs) -> ViT:
    """Create Vision Transformer model."""
    model = ViT(**kwargs)
    return model


def create_deit(pretrained: bool = False, **kwargs) -> DeiT:
    """Create DeiT model."""
    model = DeiT(**kwargs)
    return model


def create_swin(pretrained: bool = False, **kwargs) -> SwinTransformer:
    """Create Swin Transformer model."""
    model = SwinTransformer(**kwargs)
    return model
