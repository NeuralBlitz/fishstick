"""
Masked Prediction Methods for SSL

Extended masked prediction implementations:
- Masked Image Modeling (MIM) framework
- data2vec-style masked prediction
- Token labeling for ViT
- Audio/video masked prediction
"""

from typing import Optional, Tuple, Dict, Any, List, Callable
import math
import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from fishstick.ssl_extensions.base import (
    Patchify,
    Unpatchify,
    PositionalEmbedding2D,
    DropPath,
    stop_gradient,
)


class MaskGenerator(nn.Module):
    """Generate random masks for masked prediction.
    
    Args:
        mask_ratio: Ratio of patches to mask
        min_masks: Minimum number of masks
        mode: Masking mode ('random', 'block', 'grid')
        block_size: Size of blocks for block masking
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.75,
        min_masks: int = 1,
        mode: str = 'random',
        block_size: int = 2,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.min_masks = min_masks
        self.mode = mode
        self.block_size = block_size
        
    def forward(self, B: int, L: int, H: int, W: int) -> Tuple[Tensor, Tensor]:
        num_patches = H * W
        num_masks = max(self.min_masks, int(num_patches * self.mask_ratio))
        
        if self.mode == 'random':
            mask = self._random_mask(B, L, H, W, num_masks)
        elif self.mode == 'block':
            mask = self._block_mask(B, L, H, W)
        elif self.mode == 'grid':
            mask = self._grid_mask(B, L, H, W)
        else:
            mask = self._random_mask(B, L, H, W, num_masks)
            
        mask = mask.reshape(B, H, W)
        return mask
        
    def _random_mask(self, B: int, L: int, H: int, W: int, num_masks: int) -> Tensor:
        mask = torch.zeros(B, L, dtype=torch.bool, device='cpu')
        
        for i in range(B):
            indices = torch.randperm(L)[:num_masks]
            mask[i, indices] = True
            
        return mask
    
    def _block_mask(self, B: int, L: int, H: int, W: int) -> Tensor:
        mask = torch.zeros(B, L, dtype=torch.bool)
        
        for i in range(B):
            num_blocks_h = H // self.block_size
            num_blocks_w = W // self.block_size
            num_block_patches = self.block_size * self.block_size
            
            block_h = random.randint(0, num_blocks_h - 1)
            block_w = random.randint(0, num_blocks_w - 1)
            
            for bh in range(block_h, min(block_h + 1, num_blocks_h)):
                for bw in range(block_w, min(block_w + 1, num_blocks_w)):
                    for p in range(num_block_patches):
                        patch_idx = bh * self.block_size * W + bw * self.block_size + p
                        if patch_idx < L:
                            mask[i, patch_idx] = True
                            
        return mask
    
    def _grid_mask(self, B: int, L: int, H: int, W: int) -> Tensor:
        mask = torch.zeros(B, L, dtype=torch.bool)
        
        stride = int(math.sqrt(L / (H * W)))
        
        for i in range(B):
            for h in range(0, H, stride):
                for w in range(0, W, stride):
                    if h + stride <= H and w + stride <= W:
                        patch_idx = h * W + w
                        if patch_idx < L:
                            mask[i, patch_idx] = True
                            
        return mask


class MaskedImageModeling(nn.Module):
    """Masked Image Modeling framework for SSL.
    
    General framework for masked prediction with:
    - Patch-based masking
    - Optional encoder-decoder architecture
    - Multiple decoder types
    
    Args:
        encoder: Backbone encoder (ViT, etc.)
        decoder: Optional decoder network
        embed_dim: Embedding dimension
        patch_size: Size of patches
        num_patches: Number of patches per side
        mask_ratio: Ratio of patches to mask
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Number of decoder layers
        decoder_num_heads: Number of decoder attention heads
        use_mask_token: Whether to use learnable mask tokens
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
        embed_dim: int = 768,
        patch_size: int = 16,
        num_patches: int = 14,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        use_mask_token: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.use_mask_token = use_mask_token
        
        self.patch_embed = Patchify(
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
        )
        
        num_patches = num_patches * num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        self.mask_generator = MaskGenerator(mask_ratio=mask_ratio)
        
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            
        if decoder is not None:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, decoder_embed_dim)
            )
            self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
            
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        B = x.shape[0]
        
        x = self.patch_embed(x)
        
        if mask is None:
            mask = self.mask_generator(B, x.shape[1], self.num_patches, self.num_patches)
            mask = mask.to(x.device)
            
        x = x + self.pos_embed
        
        if self.use_mask_token:
            mask_tokens = self.mask_token.expand(B, x.shape[1], -1)
            x = x * (1 - mask.unsqueeze(-1).float()) + mask_tokens * mask.unsqueeze(-1).float()
            
        encoder_out = self.encoder(x)
        
        if self.decoder is not None:
            decoder_out = self._decode(encoder_out, mask)
            return encoder_out, decoder_out, mask
            
        return encoder_out, encoder_out, mask
        
    def _decode(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        
        x = self.decoder(x)
        x = self.decoder_norm(x)
        
        return x


class Data2VecMIM(nn.Module):
    """data2vec-style masked prediction for multimodal learning.
    
    Args:
        encoder: Backbone encoder
        feature_dim: Dimension of features
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        mask_ratio: Ratio to mask
        teacher_temp: Temperature for teacher predictions
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.5,
        teacher_temp: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.mask_ratio = mask_ratio
        self.teacher_temp = teacher_temp
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        
        self.teacher = copy.deepcopy(encoder)
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B = L = x.shape[0], x.shape[1]
        
        mask = torch.rand(B, L) < self.mask_ratio
        
        x_masked = x.clone()
        mask_tokens = self.mask_token.expand(B, L, -1)
        x_masked[mask] = mask_tokens[mask]
        
        student_out = self.encoder(x_masked)
        
        with torch.no_grad():
            teacher_out = self.teacher(x)
            
        loss = self._compute_loss(student_out, teacher_out, mask)
        
        return loss, student_out, teacher_out
        
    def _compute_loss(
        self,
        student_out: Tensor,
        teacher_out: Tensor,
        mask: Tensor,
    ) -> Tensor:
        student_out = F.normalize(student_out, dim=-1)
        teacher_out = F.normalize(teacher_out, dim=-1)
        
        student_out = student_out[mask]
        teacher_out = teacher_out[mask]
        
        loss = -F.cosine_similarity(student_out, teacher_out, dim=-1).mean()
        
        return loss


class TokenLabeling(nn.Module):
    """Token labeling for ViT-based SSL.
    
    Args:
        encoder: ViT encoder
        num_classes: Number of token classes
        feature_dim: Feature dimension
        use_dalle_upsample: Whether to use DALL-E upsampling
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 8192,
        feature_dim: int = 768,
        use_dalle_upsample: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        self.head = nn.Linear(feature_dim, num_classes)
        
        if use_dalle_upsample:
            self.upsample = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 4),
                nn.GELU(),
                nn.Linear(feature_dim * 4, feature_dim * 4),
                nn.GELU(),
                nn.Linear(feature_dim * 4, num_classes),
            )
        else:
            self.upsample = None
            
    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        features = self.encoder(x)
        
        logits = self.head(features)
        
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_classes), target.view(-1), reduction='mean')
            return loss
            
        return logits


class AudioMaskedPrediction(nn.Module):
    """Masked prediction for audio spectrograms.
    
    Args:
        encoder: Audio encoder network
        feature_dim: Feature dimension
        patch_size: Time-frequency patch size
        mask_ratio: Masking ratio
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int = 768,
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.65,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, C, F, T = x.shape
        
        x = self._patchify(x)
        
        mask = self._generate_mask(B, x.shape[1])
        
        x_masked = self._apply_mask(x, mask)
        
        features = self.encoder(x_masked)
        
        return features, x, mask
        
    def _patchify(self, x: Tensor) -> Tensor:
        B, C, F, T = x.shape
        pf, pt = self.patch_size
        
        x = x.reshape(B, C, F // pf, pf, T // pt, pt)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, (F // pf) * (T // pt), C * pf * pt)
        
        return x
        
    def _generate_mask(self, B: int, L: int) -> Tensor:
        num_masks = int(L * self.mask_ratio)
        mask = torch.zeros(B, L, dtype=torch.bool)
        
        for i in range(B):
            indices = torch.randperm(L)[:num_masks]
            mask[i, indices] = True
            
        return mask
        
    def _apply_mask(self, x: Tensor, mask: Tensor) -> Tensor:
        mask_tokens = self.mask_token.expand(x.shape[0], x.shape[1], -1)
        x_masked = x * (~mask).unsqueeze(-1).float() + mask_tokens * mask.unsqueeze(-1).float()
        
        return x_masked


class VideoMaskedPrediction(nn.Module):
    """Masked prediction for video sequences.
    
    Args:
        encoder: Video encoder (3D CNN or Video Transformer)
        feature_dim: Feature dimension
        mask_ratio: Masking ratio
        tube_size: Size of temporal tubes for masking
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int = 768,
        mask_ratio: float = 0.9,
        tube_size: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.mask_ratio = mask_ratio
        self.tube_size = tube_size
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, C, T, H, W = x.shape
        
        x = self._create_tubes(x)
        
        mask = self._generate_mask(B, x.shape[1])
        
        x_masked = self._apply_mask(x, mask)
        
        features = self.encoder(x_masked)
        
        return features, x, mask
        
    def _create_tubes(self, x: Tensor) -> Tensor:
        B, C, T, H, W = x.shape
        
        x = x.reshape(B, C, T // self.tube_size, self.tube_size, H, W)
        x = x.permute(0, 2, 3, 4, 5, 1)
        x = x.reshape(B, T // self.tube_size, H * W * self.tube_size, C)
        
        return x
        
    def _generate_mask(self, B: int, L: int) -> Tensor:
        num_masks = int(L * self.mask_ratio)
        mask = torch.zeros(B, L, dtype=torch.bool)
        
        for i in range(B):
            indices = torch.randperm(L)[:num_masks]
            mask[i, indices] = True
            
        return mask
        
    def _apply_mask(self, x: Tensor, mask: Tensor) -> Tensor:
        mask_tokens = self.mask_token.expand(x.shape[0], x.shape[1], -1)
        x_masked = x * (~mask).unsqueeze(-1).float() + mask_tokens * mask.unsqueeze(-1).float()
        
        return x_masked
