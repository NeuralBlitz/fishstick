"""
Autoregressive image models for pixel-by-pixel generation.

This module provides implementations of:
- PixelCNN: Convolutional neural network for autoregressive pixel generation
- PixelCNN++: Improved version with better architecture
- PixelTransformer: Transformer-based autoregressive model
- ImageGPT: GPT-style transformer for image generation
- MaskGIT: Masked Generative Image Transformer
- ParallelPixelCNN: Fast parallel sampling variant
"""

from typing import Optional, List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedConv2d(nn.Module):
    """Masked convolution for autoregressive generation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        mask: Type of mask ('A' for first layer, 'B' for others)
        padding: Padding for convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        mask: str = "A",
        padding: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mask = mask
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

        self._register_mask()

    def _register_mask(self):
        mask = torch.ones(1, 1, self.kernel_size, self.kernel_size)

        if self.mask == "A":
            mask[:, :, self.kernel_size // 2 + 1 :, :] = 0
            mask[:, :, self.kernel_size // 2, self.kernel_size // 2 + 1 :] = 0

        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        self.conv.weight.data.mul_(self.mask)
        return self.conv(x)


class GatedPixelCNN(nn.Module):
    """Gated PixelCNN with residual connections.

    Args:
        in_channels: Number of input channels
        hidden_channels: Number of hidden channels
        num_layers: Number of gated conv layers
        num_classes: Number of pixel value classes (256 for 8-bit)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        num_layers: int = 15,
        num_classes: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.input_conv = MaskedConv2d(in_channels * 2, hidden_channels, 7, "A", 3)

        self.gated_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.gated_convs.append(GatedResidualBlock(hidden_channels))

        self.output_conv = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(hidden_channels, hidden_channels, 1, "B"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels * num_classes, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape

        x = self.input_conv(x)

        for gated_conv in self.gated_convs:
            x = gated_conv(x)

        x = self.output_conv(x)
        x = x.view(batch_size, channels, self.num_classes, height, width)
        x = x.permute(0, 1, 3, 4, 2)

        return x


class GatedResidualBlock(nn.Module):
    """Gated residual block with vertical and horizontal convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.v_conv = nn.Sequential(
            MaskedConv2d(channels, channels, 3, "B", 1),
            nn.Conv2d(channels, channels * 2, 1),
        )

        self.v_to_h = nn.Conv2d(channels * 2, channels * 2, 1)

        self.h_conv = nn.Sequential(
            MaskedConv2d(channels, channels, 3, "B", 1),
            nn.Conv2d(channels, channels * 2, 1),
        )

        self.v_to_h_shortcut = nn.Conv2d(channels, channels * 2, 1)

        self.output = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        v_out = self.v_conv(x)
        v_out = self.v_to_h(v_out)

        v_shortcut = self.v_to_h_shortcut(x)

        h_out = self.h_conv(x)

        out = v_out + v_shortcut + h_out

        forget_gate, update_gate = out.chunk(2, dim=1)
        forget_gate = torch.sigmoid(forget_gate)
        update_gate = torch.sigmoid(update_gate)

        v_out = torch.tanh(v_out)
        out = forget_gate * v_out + update_gate * torch.tanh(h_out)

        return self.output(out)


class PixelCNN(GatedPixelCNN):
    """PixelCNN for autoregressive image generation.

    Args:
        in_channels: Number of input channels
        hidden_channels: Number of hidden channels
        num_layers: Number of gated conv layers
        num_classes: Number of pixel value classes
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        num_layers: int = 15,
        num_classes: int = 256,
    ):
        super().__init__(in_channels, hidden_channels, num_layers, num_classes)

    def generate(
        self, num_samples: int, height: int, width: int, device: torch.device
    ) -> Tensor:
        """Generate images autoregressively."""
        images = torch.zeros(
            num_samples, self.in_channels, height, width, device=device
        )

        for i in range(height):
            for j in range(width):
                for c in range(self.in_channels):
                    with torch.no_grad():
                        logits = self.forward(images)
                        probs = F.softmax(logits[:, c, i, j, :], dim=-1)
                        samples = torch.multinomial(probs, 1).float() / 255.0
                        images[:, c, i, j] = samples.squeeze(-1)

        return images


class PixelCNNpp(nn.Module):
    """Improved PixelCNN++ with better architecture.

    Args:
        in_channels: Number of input channels
        num_classes: Number of pixel value classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = MaskedConv2d(in_channels * 2, 256, 7, "A", 3)

        self.residual_blocks = nn.ModuleList(
            [PixelCNNppResidualBlock(256, dropout) for _ in range(20)]
        )

        self.conv_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, in_channels * num_classes, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.conv_out(x)


class PixelCNNppResidualBlock(nn.Module):
    """Residual block for PixelCNN++."""

    def __init__(self, channels: int, dropout: float):
        super().__init__()

        self.conv1 = MaskedConv2d(channels, channels, 3, "B", 1)
        self.conv2 = MaskedConv2d(channels, channels, 3, "B", 1)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm1 = nn.LayerNorm(channels)
        self.layer_norm2 = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm2(x)
        x = x.permute(0, 3, 1, 2)

        return x + residual


class PixelTransformer(nn.Module):
    """Transformer-based autoregressive model for images.

    Args:
        image_size: Size of input images
        patch_size: Size of image patches
        num_classes: Number of pixel value classes
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_ratio: MLP expansion ratio
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 256,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes * num_patches)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.head(x[:, 1:])

        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ImageGPT(nn.Module):
    """GPT-style transformer for image generation.

    Args:
        vocab_size: Size of pixel value vocabulary
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_length: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)

        self.blocks = nn.ModuleList(
            [GPTBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.shape

        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=x.device))
        x = token_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.lm_head(x)

    def generate(self, num_samples: int, seq_len: int, device: torch.device) -> Tensor:
        generated = torch.zeros(num_samples, seq_len, dtype=torch.long, device=device)

        for i in range(seq_len):
            with torch.no_grad():
                logits = self.forward(generated[:, : i + 1])
                probs = F.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated[:, i] = next_token.squeeze(-1)

        return generated


class GPTBlock(nn.Module):
    """GPT transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class MaskGIT(nn.Module):
    """Masked Generative Image Transformer.

    Args:
        image_size: Size of images
        patch_size: Size of patches
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        num_classes: Number of token classes
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        num_classes: int = 1024,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, 4.0) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        if mask is not None:
            mask_tokens = self.mask_token.expand(batch_size, x.shape[1], -1)
            x = torch.where(mask.unsqueeze(-1), mask_tokens, x)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        device: torch.device,
        num_iterations: int = 12,
    ) -> Tensor:
        num_patches = (self.image_size // self.patch_size) ** 2
        mask = torch.ones(batch_size, num_patches, device=device, dtype=torch.bool)

        tokens = torch.zeros(batch_size, num_patches, dtype=torch.long, device=device)

        for _ in range(num_iterations):
            mask_ratio = 1.0 - (mask.sum(dim=1) / num_patches)

            logits = self.forward(
                torch.zeros(
                    batch_size, 3, self.image_size, self.image_size, device=device
                ),
                mask,
            )

            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, self.num_classes), 1).view(
                batch_size, num_patches
            )

            unknown_indices = mask.nonzero(as_tuple=True)

            if len(unknown_indices[1]) > 0:
                tokens[unknown_indices[0], unknown_indices[1]] = sampled[
                    unknown_indices[0], unknown_indices[1]
                ]
                mask[unknown_indices[0], unknown_indices[1]] = False

        return tokens


class ParallelPixelCNN(GatedPixelCNN):
    """Parallel PixelCNN for fast generation.

    Optimized version that processes all pixels in parallel.

    Args:
        in_channels: Number of input channels
        hidden_channels: Number of hidden channels
        num_layers: Number of layers
        num_classes: Number of pixel value classes
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        num_layers: int = 15,
        num_classes: int = 256,
    ):
        super().__init__(in_channels, hidden_channels, num_layers, num_classes)

    def generate(
        self, num_samples: int, height: int, width: int, device: torch.device
    ) -> Tensor:
        """Generate images in parallel (single forward pass with sampling)."""
        images = torch.zeros(
            num_samples, self.in_channels, height, width, device=device
        )

        with torch.no_grad():
            logits = self.forward(images)
            probs = F.softmax(logits, dim=-1)
            images = torch.multinomial(
                probs.view(-1, self.in_channels * height * width, self.num_classes), 1
            )
            images = (
                images.view(num_samples, self.in_channels, height, width).float()
                / 255.0
            )

        return images
