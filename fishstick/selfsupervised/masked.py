"""
Masked Autoencoder Implementations

Self-supervised masked image modeling:
- MAE: Masked Autoencoders Are Scalable Vision Learners
- SimMIM: Simple Visual Representation Learning via Masked Image Modeling
"""

from typing import Optional, Tuple, List
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Splits image into patches and projects them to embedding space.

    Args:
        img_size: Size of input image
        patch_size: Size of each patch
        in_chans: Number of input channels
        embed_dim: Dimension of patch embeddings
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        B, C, H, W = x.shape
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, h, w


class MaskedAutoencoder(nn.Module):
    """Base class for Masked Autoencoder architectures."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_depth: int = 8,
        decoder_heads: int = 12,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=True
        )
        self.encoder = TransformerEncoder(
            embed_dim,
            encoder_depth,
            encoder_heads,
            int(embed_dim * mlp_ratio),
        )
        self.decoder = TransformerDecoder(
            embed_dim,
            decoder_depth,
            decoder_heads,
            int(embed_dim * mlp_ratio),
            patch_size**2 * in_chans,
        )

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)

    def forward_encoder(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.patch_embed(x)[0]
        x = x + self.pos_embed

        if mask is not None:
            x = x * (1 - mask.unsqueeze(-1))

        x = self.encoder(x)
        return x

    def forward_decoder(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        latent = self.forward_encoder(x, mask)
        pred = self.forward_decoder(latent)
        return pred, None


class MAE(MaskedAutoencoder):
    """Masked Autoencoders Are Scalable Vision Learners.

    Pre-training method that masks random patches of the input image
    and learns to reconstruct the missing patches.

    Args:
        img_size: Size of input image
        patch_size: Size of each patch
        embed_dim: Dimension of patch embeddings
        encoder_depth: Number of transformer blocks in encoder
        encoder_heads: Number of attention heads in encoder
        decoder_depth: Number of transformer blocks in decoder
        decoder_heads: Number of attention heads in decoder
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        mask_ratio: Ratio of patches to mask
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_depth: int = 8,
        decoder_heads: int = 12,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            encoder_depth,
            encoder_heads,
            decoder_depth,
            decoder_heads,
            mlp_ratio,
            mask_ratio,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        n_patches = x.shape[1]

        mask = self._generate_mask(n_patches, B, x.device, x.dtype)

        pos_embed = self.pos_embed.repeat(B, 1, 1)
        x = x + pos_embed

        x = x * (1 - mask.unsqueeze(-1))

        visible_indices = (1 - mask).nonzero(as_tuple=True)[1]
        visible_tokens = x[:, visible_indices]

        visible_tokens = self.encoder(visible_tokens)

        n_visible = visible_indices.shape[0]
        full_tokens = torch.zeros(B, n_patches, self.embed_dim, device=x.device)
        full_tokens[:, visible_indices] = visible_tokens
        full_tokens = full_tokens + self.mask_token * mask.unsqueeze(-1)

        pred = self.decoder(full_tokens, H, W)

        return pred, mask, visible_tokens

    def _generate_mask(
        self, n_patches: int, B: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        n_keep = int(n_patches * (1 - self.mask_ratio))
        noise = torch.rand(B, n_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones(B, n_patches, device=device, dtype=dtype)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def training_step(self, x: Tensor) -> Tuple[Tensor, dict]:
        pred, mask, _ = self(x)
        target = self._get_target(x)

        loss = F.mse_loss(pred, target, reduction="none")
        loss = (loss.mean(dim=-1) * mask).sum() / mask.sum()

        return loss, {"loss": loss.item(), "mask_ratio": self.mask_ratio}


class SimMIM(MaskedAutoencoder):
    """Simple Visual Representation Learning via Masked Image Modeling.

    A simple masked image modeling framework that uses a lightweight decoder
    and predicts raw pixels of masked patches.

    Args:
        img_size: Size of input image
        patch_size: Size of each patch
        embed_dim: Dimension of patch embeddings
        encoder_depth: Number of transformer blocks in encoder
        encoder_heads: Number of attention heads in encoder
        mask_ratio: Ratio of patches to mask
        patch_size_pred: Size of patches to predict (same as patch_size)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.6,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            encoder_depth,
            encoder_heads,
            0,
            0,
            mlp_ratio,
            mask_ratio,
        )

        self.decoder = nn.Linear(embed_dim, patch_size**2 * in_chans)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        n_patches = x.shape[1]

        mask = self._generate_mask(n_patches, B, x.device, x.dtype)
        mask = mask.unsqueeze(-1).bool()

        x = x + self.pos_embed

        x_masked = x * (~mask)
        x_masked = self.encoder(x_masked)

        pred = self.decoder(x_masked)

        return pred, mask.squeeze(-1)

    def _generate_mask(
        self, n_patches: int, B: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        n_keep = int(n_patches * (1 - self.mask_ratio))
        noise = torch.rand(B, n_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones(B, n_patches, device=device, dtype=dtype)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def training_step(self, x: Tensor) -> Tuple[Tensor, dict]:
        pred, mask = self(x)
        target = self._get_target(x)

        loss = F.mse_loss(pred, target, reduction="none")
        loss = (loss.mean(dim=-1) * mask).sum() / mask.sum()

        return loss, {"loss": loss.item()}


class TransformerEncoder(nn.Module):
    """Transformer Encoder for masked autoencoders."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_dim, drop_rate)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Transformer Decoder for masked autoencoders."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        output_dim: int,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_dim, drop_rate)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, output_dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        x = self.norm(x)
        x = self.decoder(x)
        x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with self-attention and MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=drop_rate, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MAEViT(nn.Module):
    """Vision Transformer adapted for MAE pre-training.

    Complete ViT architecture with MAE modifications.

    Args:
        img_size: Size of input image
        patch_size: Size of each patch
        in_chans: Number of input channels
        num_classes: Number of output classes (for fine-tuning)
        embed_dim: Dimension of patch embeddings
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        drop_rate: Dropout rate
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
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, embed_dim), requires_grad=True
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, int(embed_dim * mlp_ratio), drop_rate
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)[0]
        B = x.shape[0]

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        return self.head(x)
