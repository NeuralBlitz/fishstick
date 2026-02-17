"""
Model Compression Module

Pruning, quantization, and knowledge distillation for model compression.
"""

from typing import Optional, List, Tuple, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MagnitudePruner:
    """Magnitude-based weight pruner.

    Args:
        model: Model to prune
        sparsity: Target sparsity ratio (0-1)
    """

    def __init__(self, model: nn.Module, sparsity: float = 0.5):
        self.model = model
        self.sparsity = sparsity
        self.masks = {}

    def step(self):
        """Apply pruning based on weight magnitudes."""
        for name, param in self.model.named_parameters():
            if "weight" in name:
                threshold = torch.quantile(param.abs().flatten(), self.sparsity)
                mask = param.abs() > threshold
                self.masks[name] = mask
                param.data *= mask.float()

    def apply_masks(self):
        """Apply stored masks to parameters."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name].float()


class LotteryTicketPruner:
    """Lottery Ticket Hypothesis pruner.

    Finds sparse subnetworks that train from scratch.

    Args:
        model: Model to prune
        sparsity: Target sparsity
        prune_epochs: Number of pruning epochs
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.9,
        prune_epochs: int = 10,
    ):
        self.model = model
        self.sparsity = sparsity
        self.prune_epochs = prune_epochs
        self.masks = {}

    def get_initial_weights(self) -> Dict:
        """Save initial weights for resetting."""
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def prune(self, epoch: int):
        """Prune weights based on current magnitudes."""
        for name, param in self.model.named_parameters():
            if "weight" in name and param.requires_grad:
                target_sparsity = self.sparsity * (epoch + 1) / self.prune_epochs
                threshold = torch.quantile(param.abs().flatten(), target_sparsity)
                mask = param.abs() > threshold
                self.masks[name] = mask

    def reset_to_init(self, initial_weights: Dict):
        """Reset model to initial weights with pruning mask."""
        for name, param in self.model.named_parameters():
            if name in initial_weights:
                param.data = (
                    initial_weights[name].clone()
                    * self.masks.get(name, torch.ones_like(param)).float()
                )


class DynamicPruner:
    """Dynamic sparsity pruner that updates during training.

    Args:
        model: Model to prune
        target_sparsity: Final target sparsity
        prune_freq: Pruning frequency (steps)
    """

    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        prune_freq: int = 100,
    ):
        self.model = model
        self.target_sparsity = target_sparsity
        self.prune_freq = prune_freq
        self.step = 0

    def step(self):
        """Update pruning masks."""
        self.step += 1
        if self.step % self.prune_freq == 0:
            current_sparsity = self.target_sparsity * (self.step / 1000)
            for name, param in self.model.named_parameters():
                if "weight" in name:
                    threshold = torch.quantile(
                        param.abs().flatten(), min(current_sparsity, 0.99)
                    )
                    mask = param.abs() > threshold
                    param.data *= mask.float()


class Quantizer:
    """Post-training quantization wrapper.

    Args:
        model: Model to quantize
        dtype: Target dtype for quantization
    """

    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.qint8):
        self.model = model
        self.dtype = dtype

    def quantize(self):
        """Quantize model weights."""
        for name, param in self.model.named_parameters():
            if "weight" in name:
                param.data = torch.quantize_per_tensor(
                    param.data, scale=0.1, zero_point=0, dtype=self.dtype
                )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with quantization."""
        return self.model(x)


class BitPruner:
    """Gradient-based pruning for finding important weights.

    Args:
        model: Model to prune
        prune_ratio: Ratio of weights to prune
    """

    def __init__(self, model: nn.Module, prune_ratio: float = 0.3):
        self.model = model
        self.prune_ratio = prune_ratio
        self.importance_scores = {}

    def compute_importance(self):
        """Compute importance scores based on gradients."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.importance_scores[name] = param.data.abs() * param.grad.abs()

    def prune(self):
        """Prune weights based on importance scores."""
        for name, param in self.model.named_parameters():
            if name in self.importance_scores:
                threshold = torch.quantile(
                    self.importance_scores[name].flatten(), self.prune_ratio
                )
                mask = self.importance_scores[name] > threshold
                param.data *= mask.float()


class FilterPruner:
    """Filter-level pruning for convolutional networks.

    Args:
        model: Model to prune
        prune_ratio: Ratio of filters to prune per layer
    """

    def __init__(self, model: nn.Module, prune_ratio: float = 0.3):
        self.model = model
        self.prune_ratio = prune_ratio

    def compute_filter_norms(self) -> Dict[str, Tensor]:
        """Compute L2 norms of filters."""
        norms = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight
                filter_norms = weight.view(weight.size(0), -1).norm(dim=1)
                norms[name] = filter_norms
        return norms

    def prune(self):
        """Prune filters with smallest norms."""
        norms = self.compute_filter_norms()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name in norms:
                threshold = torch.quantile(norms[name], self.prune_ratio)
                mask = norms[name] > threshold
                module.weight.data = module.weight.data[mask]


class SlimmingPruner:
    """Network Slimming: Prune channels based on batch norm scales.

    Args:
        model: Model to prune
        prune_ratio: Ratio of channels to prune
    """

    def __init__(self, model: nn.Module, prune_ratio: float = 0.5):
        self.model = model
        self.prune_ratio = prune_ratio

    def get_bn_scales(self) -> Tensor:
        """Get batch norm gamma values (channel importance)."""
        scales = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                scales.append(module.weight.data.abs())
        return torch.cat(scales)

    def prune(self):
        """Prune channels based on BN scales."""
        scales = self.get_bn_scales()
        threshold = torch.quantile(scales, self.prune_ratio)

        idx = 0
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                mask = module.weight.data.abs() > threshold
                module.weight.data *= mask.float()
                module.bias.data *= mask.float()


class MagnitudePruner:
    """Alias for backward compatibility."""

    pass
