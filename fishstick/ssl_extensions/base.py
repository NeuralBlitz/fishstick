"""
SSL Extensions Base Module

Core utilities and base classes for self-supervised learning extensions.
Provides common infrastructure for momentum encoders, memory banks,
and training utilities used across SSL methods.
"""

from typing import Optional, Tuple, Dict, Any, List, Callable, Union
from abc import ABC, abstractmethod
import copy
import math
from collections import deque

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributed import all_gather, get_world_size, get_rank
import numpy as np


class MomentumUpdater(nn.Module):
    """Momentum encoder updater using exponential moving average.

    Maintains a momentum encoder that is updated as a moving average
    of the online encoder. Used in BYOL, MoCo, and other SSL methods.

    Args:
        momentum: Momentum coefficient for EMA update (default: 0.999)
        warmup_epochs: Number of warmup epochs before full momentum
        base_momentum: Initial momentum during warmup
    """

    def __init__(
        self,
        momentum: float = 0.999,
        warmup_epochs: int = 0,
        base_momentum: float = 0.996,
    ):
        super().__init__()
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.base_momentum = base_momentum
        self.current_epoch = 0

    def update_momentum(
        self, online_encoder: nn.Module, momentum_encoder: nn.Module
    ) -> float:
        """Update momentum encoder parameters.

        Args:
            online_encoder: The online encoder to copy from
            momentum_encoder: The momentum encoder to update

        Returns:
            Current momentum value
        """
        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            momentum = self.base_momentum + (self.momentum - self.base_momentum) * (
                self.current_epoch / self.warmup_epochs
            )
        else:
            momentum = self.momentum

        with torch.no_grad():
            for online_param, momentum_param in zip(
                online_encoder.parameters(), momentum_encoder.parameters()
            ):
                momentum_param.data.mul_(momentum).add_(
                    online_param.data, alpha=1 - momentum
                )

        return momentum

    def step(self, epoch: int):
        """Update current epoch for warmup.

        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch


class MemoryBank(nn.Module):
    """Memory bank for storing negative examples.

    Maintains a queue of embeddings for contrastive learning.
    Used in MoCo and other methods requiring negative samples.

    Args:
        size: Maximum size of memory bank
        dim: Dimension of embeddings
        temperature: Temperature for softmax normalization
    """

    def __init__(
        self,
        size: int = 65536,
        dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.size = size
        self.dim = dim
        self.temperature = temperature

        self.register_buffer("bank", torch.randn(size, dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.bank = F.normalize(self.bank, dim=1)

    def update(self, embeddings: Tensor):
        """Update memory bank with new embeddings.

        Args:
            embeddings: New embeddings to add (batch_size, dim)
        """
        batch_size = embeddings.shape[0]

        embeddings = F.normalize(embeddings, dim=1)

        ptr = int(self.ptr)
        if ptr + batch_size <= self.size:
            self.bank[ptr : ptr + batch_size] = embeddings.detach()
        else:
            remaining = self.size - ptr
            self.bank[ptr:] = embeddings[:remaining].detach()
            self.bank[: batch_size - remaining] = embeddings[remaining:].detach()

        new_ptr = (ptr + batch_size) % self.size
        self.ptr[0] = new_ptr

    def get(self, num_samples: Optional[int] = None) -> Tensor:
        """Get random samples from memory bank.

        Args:
            num_samples: Number of samples to retrieve (default: batch_size)

        Returns:
            Sampled embeddings from bank
        """
        if num_samples is None:
            num_samples = self.size

        indices = torch.randint(0, self.size, (num_samples,), device=self.bank.device)
        return self.bank[indices]


class EMAUpdater(nn.Module):
    """Exponential Moving Average updater for model parameters.

    Args:
        decay: Decay rate for EMA
        device: Device to store EMA parameters
    """

    def __init__(self, decay: float = 0.999, device: Optional[torch.device] = None):
        super().__init__()
        self.decay = decay
        self.device = device

    def update(self, source: nn.Module, target: nn.Module):
        """Update target model with EMA of source.

        Args:
            source: Source model (online)
            target: Target model to update (EMA)
        """
        with torch.no_grad():
            for src_param, tgt_param in zip(source.parameters(), target.parameters()):
                if self.device is not None and src_param.device != self.device:
                    src_param = src_param.to(self.device)
                tgt_param.data.mul_(self.decay).add_(
                    src_param.data, alpha=1 - self.decay
                )


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all GPUs (DDP compatible).

    Used for distributed training of contrastive methods.
    """

    @staticmethod
    def forward(ctx, input: Tensor) -> List[Tensor]:
        ctx.save_for_backward(input)
        world_size = get_world_size()
        if world_size == 1:
            return [input]

        gathered = all_gather(input)
        return gathered

    @staticmethod
    def backward(ctx, grads: List[Tensor]) -> Tensor:
        rank = get_rank()
        return grads[rank].contiguous()


def gather_from_all(input: Tensor) -> Tensor:
    """Gather tensors from all processes in distributed training.

    Args:
        input: Input tensor

    Returns:
        Gathered tensor from all processes
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return input

    world_size = get_world_size()
    if world_size == 1:
        return input

    gathered = torch.cat(GatherLayer.apply(input), dim=0)
    return gathered


class StopGradient(torch.autograd.Function):
    """Stop gradient computation for target networks.

    Used in BYOL, SimSiam to prevent gradient flow to target.
    """

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        return input

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return torch.zeros_like(grad_output)


def stop_gradient(x: Tensor) -> Tensor:
    """Stop gradient computation.

    Args:
        x: Input tensor

    Returns:
        Tensor with gradient stopped
    """
    return StopGradient.apply(x)


class SSLScheduler:
    """Learning rate scheduler specifically designed for SSL training.

    Implements warmup + cosine decay schedule common in SSL methods.

    Args:
        optimizer: Optimizer to schedule
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr: Minimum learning rate after decay
        base_lr: Base learning rate
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
        base_lr: float = 0.05,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        """Update learning rate.

        Args:
            epoch: Current epoch (optional, uses internal counter if None)
        """
        if epoch is not None:
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.current_epoch += 1

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class BatchNorm1dSync(nn.BatchNorm1d):
    """Synchronized BatchNorm1d for distributed training.

    Args:
        num_features: Number of features
        eps: Epsilon for numerical stability
        momentum: Momentum for running stats
        affine: Whether to learn affine parameters
        track_running_stats: Whether to track running stats
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        if self.training and torch.distributed.is_initialized():
            world_size = get_world_size()
            if world_size > 1:
                return sync_batch_norm(
                    input,
                    self.running_mean,
                    self.running_var,
                    self.weight,
                    self.bias,
                    self.eps,
                    self.momentum,
                    self.training,
                )
        return super().forward(input)


def sync_batch_norm(
    input: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    momentum: float,
    training: bool,
) -> Tensor:
    """Synchronized batch normalization for distributed training.

    Args:
        input: Input tensor
        running_mean: Running mean
        running_var: Running variance
        weight: Scale parameter
        bias: Shift parameter
        eps: Epsilon for numerical stability
        momentum: Momentum for stats
        training: Whether in training mode

    Returns:
        Normalized tensor
    """
    if training:
        world_size = get_world_size()

        mean = input.mean(dim=(0, 2, 3) if input.dim() == 4 else 0)
        var = input.var(dim=(0, 2, 3) if input.dim() == 4 else 0, unbiased=False)

        mean = gather_from_all(mean)
        var = gather_from_all(var)

        mean = mean.mean(dim=0)
        var = var.mean(dim=0)

        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
            running_var.mul_(1 - momentum).add_(var, alpha=momentum)
    else:
        mean = running_mean
        var = running_var

    normalized = (input - mean) / torch.sqrt(var + eps)

    if weight is not None and bias is not None:
        normalized = normalized * weight + bias

    return normalized


class L2Normalize(nn.Module):
    """L2 normalization layer.

    Args:
        dim: Dimension to normalize along
        eps: Epsilon for numerical stability
    """

    def __init__(self, dim: int = -1, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=self.dim, eps=self.eps)


class MultiCropWrapper(nn.Module):
    """Multi-crop augmentation wrapper for SSL.

    Handles multiple views with different resolutions for methods
    like SwAV that use multi-crop augmentation.

    Args:
        backbone: Base encoder network
        head: Projection head
        crop_size: Size for global crops
        num_crops: Number of global crops
        num_small_crops: Number of local crops
        small_crop_size: Size for local crops
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        crop_size: int = 224,
        num_crops: int = 2,
        num_small_crops: int = 4,
        small_crop_size: int = 96,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.num_small_crops = num_small_crops
        self.small_crop_size = small_crop_size

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """Process multi-crop inputs.

        Args:
            x: List of crop tensors with potentially different sizes

        Returns:
            List of projected representations
        """
        outputs = []

        for crop in x:
            features = self.backbone(crop)
            projected = self.head(features)
            outputs.append(projected)

        return outputs


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Args:
        drop_prob: Probability of dropping path
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Patchify(nn.Module):
    """Convert image to patches for ViT-based SSL.

    Args:
        patch_size: Size of patches
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Unpatchify(nn.Module):
    """Convert patches back to image for decoder.

    Args:
        patch_size: Size of patches
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.ConvTranspose2d(
            embed_dim, in_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, L, D = x.shape
        x = x.reshape(B, H // self.patch_size, W // self.patch_size, D)
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> Tensor:
    """Generate 2D sin-cos positional embeddings.

    Args:
        embed_dim: Embedding dimension
        grid_size: Size of the grid

    Returns:
        Positional embeddings (grid_size * grid_size, embed_dim)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> Tensor:
    """Generate 2D sin-cos pos embed from grid.

    Args:
        embed_dim: Embedding dimension
        grid: Grid array (2, 1, H, W)

    Returns:
        Positional embeddings
    """
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return torch.from_numpy(emb).float()


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> Tensor:
    """Generate 1D sin-cos positional embeddings.

    Args:
        embed_dim: Embedding dimension
        pos: Position array

    Returns:
        Positional embeddings
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return torch.from_numpy(emb).float()


class PositionalEmbedding2D(nn.Module):
    """2D positional embeddings for ViT.

    Args:
        embed_dim: Embedding dimension
        grid_size: Size of the grid
        temperature: Temperature for positional encoding
    """

    def __init__(
        self,
        embed_dim: int,
        grid_size: int = 14,
        temperature: float = 10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.temperature = temperature

        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
        self.register_buffer("pos_embed", pos_embed)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        pos_embed = self.pos_embed

        if L != pos_embed.shape[0]:
            H = W = int(math.sqrt(L))
            pos_embed = get_2d_sincos_pos_embed(D, H)
            pos_embed = pos_embed.to(x.device)

        x = x + pos_embed.unsqueeze(0)
        return x
