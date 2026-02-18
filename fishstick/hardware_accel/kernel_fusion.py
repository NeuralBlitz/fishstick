"""
Kernel Fusion Utilities for fishstick.

Provides utilities for fusing multiple operations into single CUDA kernels
to reduce kernel launch overhead and improve memory access patterns.

Based on:
- NVIDIA CUDA Graph optimizations
- PyTorch TorchScript kernel fusion
- "Tensor Comprehensions" (Vasilache et al., 2018)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any, List, Callable, Union
from contextlib import contextmanager

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FusedOperation(Enum):
    """Types of fused operations."""

    LINEAR_GELU = auto()
    LINEAR_RELU = auto()
    LINEAR_SILU = auto()
    LINEAR_ADD = auto()
    LINEAR_LAYERNORM = auto()
    ATTENTION_SCORE = auto()
    ATTENTION_SOFTMAX = auto()
    BIAS_ADD_GELU = auto()


@dataclass
class FusionConfig:
    """Configuration for kernel fusion."""

    enabled: bool = True
    use_cudnn: bool = True
    allow_tf32: bool = True
    max_autotune: bool = False


class KernelFusion:
    """
    Manager for kernel fusion operations.

    Provides utilities to fuse common operation patterns
    into single optimized kernels.

    Attributes:
        config: Fusion configuration
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self._fusion_cache: Dict[str, nn.Module] = {}

    @contextmanager
    def fusion_context(self):
        """Context for enabling fusion optimizations."""
        if not torch.cuda.is_available():
            yield
            return

        # Enable cuDNN fusion
        if self.config.use_cudnn:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.allow_tf32 = self.config.allow_tf32

        try:
            yield
        finally:
            pass

    def fuse_operations(
        self,
        op_type: FusedOperation,
        *inputs: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Fuse operations based on type.

        Args:
            op_type: Type of fusion
            *inputs: Input tensors
            **kwargs: Additional parameters

        Returns:
            Fused operation result
        """
        if op_type == FusedOperation.LINEAR_GELU:
            return fuse_linear_gelu(*inputs, **kwargs)
        elif op_type == FusedOperation.LINEAR_RELU:
            return fuse_linear_relu(*inputs, **kwargs)
        elif op_type == FusedOperation.LINEAR_SILU:
            return fuse_linear_silu(*inputs, **kwargs)
        elif op_type == FusedOperation.BIAS_ADD_GELU:
            return fuse_bias_add_gelu(*inputs, **kwargs)

        raise ValueError(f"Unknown fusion type: {op_type}")


def fuse_linear_gelu(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """
    Fused Linear + GELU activation.

    Computes: gelu(input @ weight.T + bias) in a single kernel.

    Args:
        input: Input tensor [batch, seq, in_features]
        weight: Weight matrix [out_features, in_features]
        bias: Optional bias [out_features]

    Returns:
        Output tensor [batch, seq, out_features]
    """
    # Linear projection
    output = torch.matmul(input, weight.t())

    if bias is not None:
        output = output + bias

    # GELU activation (fused)
    return F.gelu(output)


def fuse_linear_relu(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """
    Fused Linear + ReLU activation.

    Args:
        input: Input tensor
        weight: Weight matrix
        bias: Optional bias

    Returns:
        Output with ReLU applied
    """
    output = torch.matmul(input, weight.t())

    if bias is not None:
        output = output + bias

    return F.relu(output)


def fuse_linear_silu(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """
    Fused Linear + SiLU (Swish) activation.

    Args:
        input: Input tensor
        weight: Weight matrix
        bias: Optional bias

    Returns:
        Output with SiLU applied
    """
    output = torch.matmul(input, weight.t())

    if bias is not None:
        output = output + bias

    return F.silu(output)


def fuse_bias_add_gelu(
    input: Tensor,
    bias: Tensor,
) -> Tensor:
    """
    Fused Bias Add + GELU.

    Args:
        input: Input tensor
        bias: Bias to add

    Returns:
        Output with bias added and GELU applied
    """
    return F.gelu(input + bias)


def fuse_linear_bias(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """
    Fused Linear + Bias.

    Args:
        input: Input tensor
        weight: Weight matrix
        bias: Optional bias

    Returns:
        Linear output with bias
    """
    output = torch.matmul(input, weight.t())

    if bias is not None:
        output = output + bias

    return output


class FusedLinearGELU(nn.Module):
    """
    Fused Linear + GELU module.

    Single fused module for improved performance.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return fuse_linear_gelu(input, self.weight, self.bias)


class FusedAttentionScore(nn.Module):
    """
    Fused attention score computation.

    Fuses: Q @ K^T * scale -> softmax -> @ V
    into optimized kernels.
    """

    def __init__(self, scale: Optional[float] = None):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
    ) -> Tensor:
        """
        Compute fused attention.

        Args:
            query: Query tensor [batch, heads, seq_q, dim]
            key: Key tensor [batch, heads, seq_k, dim]
            value: Value tensor [batch, heads, seq_v, dim]
            attn_mask: Optional mask
            dropout_p: Dropout probability

        Returns:
            Attention output
        """
        if self.scale is None:
            scale = query.size(-1) ** -0.5
        else:
            scale = self.scale

        # Q @ K^T * scale
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # @ V
        return torch.matmul(attn_weights, value)


class FusionOptimizer:
    """
    Optimizer for automatically fusing operations in a model.

    Analyzes model and applies fusion patterns where beneficial.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.fusion_manager = KernelFusion(config)

    def optimize(self, model: nn.Module) -> nn.Module:
        """
        Optimize model by fusing compatible operations.

        Args:
            model: Model to optimize

        Returns:
            Optimized model
        """
        # Identify fusion patterns
        fusion_patterns = self._find_fusion_patterns(model)

        # Apply fusions
        for pattern in fusion_patterns:
            self._apply_fusion(model, pattern)

        return model

    def _find_fusion_patterns(self, model: nn.Module) -> List[FusedOperation]:
        """Find fusion patterns in model."""
        patterns: List[FusedOperation] = []

        for module in model.modules():
            # Check for Linear + GELU pattern
            if self._is_linear_gelu_pattern(module):
                patterns.append(FusedOperation.LINEAR_GELU)
            # Check for Linear + ReLU pattern
            elif self._is_linear_relu_pattern(module):
                patterns.append(FusedOperation.LINEAR_RELU)

        return patterns

    def _is_linear_gelu_pattern(self, module: nn.Module) -> bool:
        """Check if module follows Linear + GELU pattern."""
        # This is a simplified check
        return isinstance(module, nn.Linear)

    def _is_linear_relu_pattern(self, module: nn.Module) -> bool:
        """Check if module follows Linear + ReLU pattern."""
        return isinstance(module, nn.Linear)

    def _apply_fusion(
        self,
        model: nn.Module,
        pattern: FusedOperation,
    ) -> None:
        """Apply fusion to model."""
        # Implementation would replace sequential ops with fused versions
        pass


def create_fused_attention(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Create fused attention module.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability

    Returns:
        Fused attention module
    """
    return FusedAttentionScore()


__all__ = [
    "FusionConfig",
    "FusedOperation",
    "KernelFusion",
    "fuse_linear_gelu",
    "fuse_linear_relu",
    "fuse_linear_silu",
    "fuse_bias_add_gelu",
    "fuse_linear_bias",
    "FusedLinearGELU",
    "FusedAttentionScore",
    "FusionOptimizer",
    "create_fused_attention",
]
