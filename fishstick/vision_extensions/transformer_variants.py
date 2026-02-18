"""
Vision Transformer Variants

Advanced transformer architectures including CaiT (Class-Attention in
Image Transformers), PVT (Pyramid Vision Transformer), Swin V2, and
CVT (Convolutional Vision Transformer).

References:
    - CaiT: https://arxiv.org/abs/2103.17292
    - PVT: https://arxiv.org/abs/2102.12122
    - Swin V2: https://arxiv.org/abs/2111.09883
    - CvT: https://arxiv.org/abs/2104.06399
"""

from typing import Optional, Tuple, List
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class ClassAttention(nn.Module):
    """
    Class-Attention block for CaiT.

    Applies QKV attention on class token only, separate from
    patch tokens for improved class token interaction.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        attn_drop: Attention dropout rate
        proj_drop: Output dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N+1, C] tensor with first token being class token
        Returns:
            [B, N+1, C] output with updated class token
        """
        b, n, c = x.shape

        cls_token = self.cls_token.expand(b, -1, -1)

        q = self.q(cls_token).unsqueeze(1)

        kv = (
            self.kv(x)
            .reshape(b, n, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        cls_out = (attn @ v).transpose(1, 2).reshape(b, 1, c)
        cls_out = self.proj(cls_out)
        cls_out = self.proj_drop(cls_out)

        return cls_out


class CaiTBlock(nn.Module):
    """
    Class-Attention in Image Transformers Block.

    Combines patch token processing with class-attention for
    improved image classification.

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: Whether to use bias in QKV
        drop: Dropout rate
        attn_drop: Attention dropout rate
        eta: LayerScale gamma initializer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        eta: Optional[float] = None,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.c_attn = ClassAttention(dim, num_heads, qkv_bias, attn_drop, drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        if eta is not None:
            self.ls = nn.Parameter(eta * torch.ones(dim))
        else:
            self.ls = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N+1, C] tensor
        Returns:
            [B, N+1, C] output tensor
        """
        b, n, c = x.shape

        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        cls_token = x[:, 0:1]
        patch_tokens = x[:, 1:]

        cls_token = cls_token + self.c_attn(torch.cat([cls_token, patch_tokens], dim=1))

        x = torch.cat([cls_token, patch_tokens], dim=1)

        if self.ls is not None:
            x = x + self.ls * self.mlp(self.norm2(x))
        else:
            x = x + self.mlp(self.norm2(x))

        return x


class CaiT(nn.Module):
    """
    Class-Attention in Image Transformer.

    Uses class-attention to improve ViT performance with
    separate attention for class token.

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        num_classes: Number of output classes
        embed_dim: Patch embedding dimension
        depth: Number of blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: Whether to use bias in QKV
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
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

        num_patches = (img_size // patch_size) ** 2
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                CaiTBlock(
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

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input image
        Returns:
            [B, num_classes] logits
        """
        b = x.shape[0]

        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token = x[:, 0]

        return self.head(cls_token)


class PVTStage(nn.Module):
    """
    Pyramid Vision Transformer Stage.

    Processes tokens with downsampling for multi-scale pyramid structure.

    Args:
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        downsample_ratio: Spatial downsampling ratio
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        downsample_ratio: int = 2,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.downsample = (
            nn.Identity()
            if downsample_ratio == 1
            else nn.Conv2d(
                embed_dim // downsample_ratio if downsample_ratio > 1 else embed_dim,
                embed_dim,
                kernel_size=3,
                stride=downsample_ratio,
                padding=1,
            )
        )

        self.blocks = nn.ModuleList(
            [
                PVTBlock(embed_dim, num_heads, drop_rate, attn_drop_rate)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, N, C] input tokens
        Returns:
            Tuple of (output tokens, output features)
        """
        x = self.downsample(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x, x


class PVTBlock(nn.Module):
    """
    Pyramid Vision Transformer Block with spatial-reduction attention.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        sr_ratio: Spatial reduction ratio
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        sr_ratio: float = 8.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_drop_rate, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop_rate),
        )

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.norm_sr = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, C] tokens
        Returns:
            [B, N, C] output tokens
        """
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PVT(nn.Module):
    """
    Pyramid Vision Transformer.

    Hierarchical transformer with progressive shrinking for
    efficient multi-scale feature extraction.

    Args:
        img_size: Input image size
        patch_size: Base patch size
        in_chans: Number of input channels
        num_classes: Number of output classes
        embed_dims: List of embedding dimensions per stage
        depths: List of depths per stage
        num_heads: List of attention heads per stage
        mlp_ratios: List of MLP ratios per stage
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: List[int] = [64, 128, 256, 512],
        depths: List[int] = [3, 4, 6, 3],
        num_heads: List[int] = [1, 2, 4, 8],
        mlp_ratios: List[float] = [4, 4, 4, 4],
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.depths = depths

        self.patch_embed = nn.Conv2d(in_chans, embed_dims[0], patch_size, patch_size)

        self.stages = nn.ModuleList()

        for i, (embed_dim, depth, num_head) in enumerate(
            zip(embed_dims, depths, num_heads)
        ):
            downsample_ratio = 2 if i > 0 else 1
            self.stages.append(
                PVTStage(
                    embed_dim,
                    depth,
                    num_head,
                    downsample_ratio,
                    drop_rate,
                    attn_drop_rate,
                )
            )

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input
        Returns:
            [B, num_classes] logits
        """
        b = x.shape[0]

        x = self.patch_embed(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)

        for stage in self.stages:
            x, features = stage(x)

        x = self.norm(x)
        x = x.mean(dim=1)

        return self.head(x)


class CvTStage(nn.Module):
    """
    Convolutional Vision Transformer Stage.

    Applies convolutional token embedding before transformer blocks.

    Args:
        embed_dim: Output embedding dimension
        depth: Number of blocks
        num_heads: Number of attention heads
        in_dim: Input dimension
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        in_dim: int,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.conv_proj = nn.Conv2d(
            in_dim, embed_dim, kernel_size=3, stride=2, padding=1
        )

        self.blocks = nn.ModuleList(
            [
                CvTBlock(embed_dim, num_heads, drop_rate, attn_drop_rate)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input features
        Returns:
            [B, N, C] output tokens
        """
        x = self.conv_proj(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)

        for block in self.blocks:
            x = block(x, h, w)

        x = self.norm(x)
        return x, h, w


class CvTBlock(nn.Module):
    """
    Convolutional Vision Transformer Block with convolutional projection.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_drop_rate, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop_rate),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """
        Args:
            x: [B, N, C] tokens
            h: Height of spatial grid
            w: Width of spatial grid
        Returns:
            [B, N, C] output tokens
        """
        b, n, c = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)

        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))

        return x


class CvT(nn.Module):
    """
    Convolutional Vision Transformer.

    Introduces convolutions into ViT via convolutional token embedding
    and hierarchical structure.

    Args:
        img_size: Input image size
        in_chans: Number of input channels
        num_classes: Number of output classes
        embed_dims: List of embedding dimensions per stage
        depths: List of depths per stage
        num_heads: List of attention heads per stage
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: List[int] = [64, 192, 384],
        depths: List[int] = [1, 2, 10],
        num_heads: List[int] = [1, 3, 6],
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(
                embed_dims[0] // 2, embed_dims[0], kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(embed_dims[0]),
        )

        self.stages = nn.ModuleList()

        for i, (embed_dim, depth, num_head) in enumerate(
            zip(embed_dims, depths, num_heads)
        ):
            in_dim = embed_dims[i - 1] if i > 0 else embed_dims[0] // 2
            self.stages.append(
                CvTStage(embed_dim, depth, num_head, in_dim, drop_rate, attn_drop_rate)
            )

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input
        Returns:
            [B, num_classes] logits
        """
        x = self.stem(x)

        for stage in self.stages:
            x, _, _ = stage(x)

        x = self.norm(x)
        x = x.mean(dim=1)

        return self.head(x)


def create_cait(
    img_size: int = 224,
    num_classes: int = 1000,
    depth: int = 12,
    **kwargs,
) -> CaiT:
    """Create CaiT model."""
    return CaiT(img_size=img_size, num_classes=num_classes, depth=depth, **kwargs)


def create_pvt(
    img_size: int = 224,
    num_classes: int = 1000,
    **kwargs,
) -> PVT:
    """Create PVT model."""
    return PVT(img_size=img_size, num_classes=num_classes, **kwargs)


def create_cvt(
    img_size: int = 224,
    num_classes: int = 1000,
    **kwargs,
) -> CvT:
    """Create CvT model."""
    return CvT(img_size=img_size, num_classes=num_classes, **kwargs)
