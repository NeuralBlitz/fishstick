"""
Gradient Checkpointing Utilities for fishstick.

Provides memory-efficient training via activation checkpointing,
reducing memory usage by recomputing activations during backward pass.

Based on:
- "Training Deep Nets with Sublinear Memory Cost" (Chen et al., 2016)
- PyTorch gradient checkpointing implementation
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List, Callable, Union, Sequence
from contextlib import contextmanager
import functools
import threading

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class CheckpointedModule(nn.Module):
    """
    Wrapper module that applies gradient checkpointing to forward pass.

    Reduces memory usage by trading compute for memory during training.
    Useful for training large models that would otherwise exceed GPU memory.

    Attributes:
        module: Module to checkpoint
        use_reentrant: Use reentrant checkpointing
    """

    def __init__(
        self,
        module: nn.Module,
        use_reentrant: bool = True,
        checkpoint_policy: Optional[Callable[..., Any]] = None,
    ):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant
        self.checkpoint_policy = checkpoint_policy

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward with gradient checkpointing."""
        if self.training:
            if self.checkpoint_policy is not None:
                return checkpoint(
                    self.module,
                    *args,
                    use_reentrant=self.use_reentrant,
                    preserve_rng_state=True,
                )
            return checkpoint(
                self.module,
                *args,
                use_reentrant=self.use_reentrant,
                **kwargs,
            )
        return self.module(*args, **kwargs)


def checkpoint(
    fn: Callable[..., Any],
    *inputs: Any,
    use_reentrant: bool = True,
    preserve_rng_state: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Gradient checkpointing wrapper.

    Args:
        fn: Function to checkpoint
        *inputs: Input tensors
        use_reentrant: Use reentrant checkpointing
        preserve_rng_state: Preserve random state
        **kwargs: Additional arguments

    Returns:
        Function output with checkpointed gradients
    """
    return torch.utils.checkpoint.checkpoint(
        fn,
        *inputs,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
        **kwargs,
    )


def checkpoint_sequential(
    modules: Sequence[nn.Module],
    segments: int,
    *inputs: Tensor,
    use_reentrant: bool = True,
    preserve_rng_state: bool = True,
) -> Any:
    """
    Checkpoint a sequential module in segments.

    Divides the sequential module into segments and applies
    checkpointing to each segment individually.

    Args:
        modules: Sequence of modules
        segments: Number of segments to divide into
        *inputs: Input tensors
        use_reentrant: Use reentrant checkpointing
        preserve_rng_state: Preserve random state

    Returns:
        Output tensor
    """
    return torch.utils.checkpoint.checkpoint_sequential(
        modules,
        segments,
        *inputs,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
    )


def create_checkpoint_function(
    recompute_fn: Callable[[], Any],
    *args: Any,
    use_reentrant: bool = True,
) -> Callable[[], Any]:
    """
    Create a custom checkpoint function.

    Allows custom recomputation logic for gradient checkpointing.

    Args:
        recompute_fn: Function to recompute activations
        *args: Arguments for recomputation
        use_reentrant: Use reentrant checkpointing

    Returns:
        Checkpointed function
    """
    return functools.partial(
        torch.utils.checkpoint.checkpoint,
        recompute_fn,
        *args,
        use_reentrant=use_reentrant,
    )


class GradientCheckpointingScheduler:
    """
    Scheduler for dynamic gradient checkpointing.

    Can adjust checkpoint frequency during training based on
    memory pressure or training progress.
    """

    def __init__(
        self,
        initial_segments: int = 1,
        max_segments: int = 8,
        warmup_epochs: int = 0,
        increase_every: int = 1,
    ):
        self.initial_segments = initial_segments
        self.max_segments = max_segments
        self.warmup_epochs = warmup_epochs
        self.increase_every = increase_every

        self.current_segments = initial_segments
        self.epoch = 0

    def step(self, epoch: Optional[int] = None) -> None:
        """Update checkpoint segments."""
        if epoch is not None:
            self.epoch = epoch

        if self.epoch < self.warmup_epochs:
            return

        if (self.epoch - self.warmup_epochs) % self.increase_every == 0:
            self.current_segments = min(
                self.current_segments + 1,
                self.max_segments,
            )

    def get_segments(self) -> int:
        """Get current number of checkpoint segments."""
        return self.current_segments


class CheckpointOptions:
    """Options for gradient checkpointing."""

    def __init__(
        self,
        use_reentrant: bool = True,
        preserve_rng_state: bool = True,
        debug: bool = False,
        full_graph: bool = True,
    ):
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state
        self.debug = debug
        self.full_graph = full_graph


class CheckpointManager:
    """
    Manager for multiple gradient checkpointed modules.

    Provides centralized management of checkpointed modules
    and memory tracking.
    """

    def __init__(self):
        self._checkpoints: Dict[str, nn.Module] = {}
        self._options: Dict[str, CheckpointOptions] = {}
        self._lock = threading.Lock()
        self._memory_saved = 0

    def register(
        self,
        name: str,
        module: nn.Module,
        options: Optional[CheckpointOptions] = None,
    ) -> CheckpointedModule:
        """
        Register a module for gradient checkpointing.

        Args:
            name: Unique identifier
            module: Module to checkpoint
            options: Checkpoint options

        Returns:
            CheckpointedModule wrapper
        """
        with self._lock:
            options = options or CheckpointOptions()
            checkpointed = CheckpointedModule(
                module,
                use_reentrant=options.use_reentrant,
            )
            self._checkpoints[name] = checkpointed
            self._options[name] = options

            # Estimate memory savings
            self._memory_saved += self._estimate_savings(module)

            return checkpointed

    def _estimate_savings(self, module: nn.Module) -> int:
        """Estimate memory savings from checkpointing."""
        total_params = sum(p.numel() for p in module.parameters())
        # Rough estimate: 4 bytes per parameter for activations
        return total_params * 4

    def get(self, name: str) -> Optional[CheckpointedModule]:
        """Get checkpointed module by name."""
        return self._checkpoints.get(name)  # type: ignore

    def remove(self, name: str) -> None:
        """Remove checkpointed module."""
        with self._lock:
            if name in self._checkpoints:
                del self._checkpoints[name]
                del self._options[name]

    def get_memory_saved(self) -> int:
        """Get estimated memory saved in bytes."""
        return self._memory_saved

    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        return {
            "num_checkpoints": len(self._checkpoints),
            "memory_saved_mb": self._memory_saved / (1024**2),
            "checkpoints": list(self._checkpoints.keys()),
        }


def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_modules: Optional[List[str]] = None,
    segments: int = 1,
) -> nn.Module:
    """
    Apply gradient checkpointing to model.

    Args:
        model: Model to modify
        checkpoint_modules: List of module names to checkpoint (None = all)
        segments: Number of segments for sequential

    Returns:
        Modified model with checkpointing
    """
    if checkpoint_modules is None:
        # Checkpoint entire model
        if isinstance(model, nn.Sequential):
            return checkpoint_sequential(model, segments)
        else:
            return CheckpointedModule(model)

    # Checkpoint specific modules
    for name, module in model.named_modules():
        if any(cm in name for cm in checkpoint_modules):
            if isinstance(module, nn.Sequential):
                setattr(model, name, checkpoint_sequential(module, segments))
            else:
                setattr(model, name, CheckpointedModule(module))

    return model


class MemoryEfficientAttentionWithCheckpointing(nn.Module):
    """
    Memory-efficient attention with gradient checkpointing.

    Combines memory-efficient attention with checkpointing
    for maximum memory savings.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_checkpointing = use_checkpointing

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass with optional checkpointing."""

        def attention_fn() -> Tuple[Tensor, Optional[Tensor]]:
            return self.attn(
                query,
                key,
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        if self.use_checkpointing and self.training:
            return checkpoint(attention_fn)

        return attention_fn()


__all__ = [
    "CheckpointedModule",
    "checkpoint",
    "checkpoint_sequential",
    "create_checkpoint_function",
    "GradientCheckpointingScheduler",
    "CheckpointOptions",
    "CheckpointManager",
    "apply_gradient_checkpointing",
    "MemoryEfficientAttentionWithCheckpointing",
]
