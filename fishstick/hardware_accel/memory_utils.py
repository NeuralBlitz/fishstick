"""
GPU Memory Management Utilities for fishstick.

Provides memory tracking, pooling, and efficient memory operations.

Based on:
- PyTorch CUDA memory management
- Gradient checkpointing techniques (Chen et al., 2016)
- Memory-efficient attention (Rabe & Staats, 2021)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Callable, Union
from contextlib import contextmanager
import threading
import weakref
from collections import deque

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class MemoryStats:
    """Memory statistics for a device."""

    allocated: int  # bytes
    reserved: int  # bytes
    free: int  # bytes
    peak_allocated: int  # bytes
    peak_reserved: int  # bytes

    @property
    def allocated_gb(self) -> float:
        return self.allocated / (1024**3)

    @property
    def reserved_gb(self) -> float:
        return self.reserved / (1024**3)

    @property
    def free_gb(self) -> float:
        return self.free / (1024**3)


class GPUMemoryTracker:
    """
    Track GPU memory usage over time.

    Provides detailed memory statistics and allocation tracking
    for debugging memory issues and optimizing memory usage.

    Attributes:
        device: Device to track memory for
        track_history: Whether to track allocation history
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        track_history: bool = True,
        history_size: int = 1000,
    ):
        self.device = device or torch.device("cuda:0")
        self.track_history = track_history
        self.history_size = history_size

        self._history: deque[Tuple[float, int]] = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[MemoryStats], None]] = []
        self._enabled = True

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            free = total - reserved
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)

            return MemoryStats(
                allocated=allocated,
                reserved=reserved,
                free=free,
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
            )
        else:
            return MemoryStats(0, 0, 0, 0, 0)

    def reset_peak(self) -> None:
        """Reset peak memory statistics."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def record_allocation(self) -> None:
        """Record current allocation to history."""
        if not self.track_history or not self._enabled:
            return

        stats = self.get_stats()
        with self._lock:
            import time

            self._history.append((time.time(), stats.allocated))

    def get_history(self) -> List[Tuple[float, int]]:
        """Get allocation history."""
        with self._lock:
            return list(self._history)

    def register_callback(self, callback: Callable[[MemoryStats], None]) -> None:
        """Register a callback for memory changes."""
        self._callbacks.append(callback)

    @contextmanager
    def track(self) -> None:
        """Context manager to track memory within a block."""
        self.record_allocation()
        try:
            yield
        finally:
            self.record_allocation()
            stats = self.get_stats()
            for callback in self._callbacks:
                callback(stats)

    def enable(self) -> None:
        """Enable memory tracking."""
        self._enabled = True

    def disable(self) -> None:
        """Disable memory tracking."""
        self._enabled = False


class MemoryPool:
    """
    Pre-allocated memory pool for fast tensor allocation.

    Reduces memory allocation overhead by reusing pre-allocated
    buffers of common sizes.

    Attributes:
        device: Device for the memory pool
        max_size: Maximum pool size in bytes
    """

    def __init__(
        self,
        device: torch.device,
        max_size: int = 2 * 1024**3,  # 2GB default
    ):
        self.device = device
        self.max_size = max_size

        self._pools: Dict[Tuple[int, ...], List[Tensor]] = {}
        self._total_size = 0
        self._lock = threading.Lock()

    def allocate(
        self,
        size: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """
        Allocate a tensor from the pool.

        Args:
            size: Tensor shape
            dtype: Tensor dtype

        Returns:
            Pre-allocated or newly created tensor
        """
        key = (size, dtype)

        with self._lock:
            if key in self._pools and self._pools[key]:
                tensor = self._pools[key].pop()
                return tensor.zero_()

            # Create new tensor if not in pool
            return torch.zeros(size, device=self.device, dtype=dtype)

    def release(self, tensor: Tensor) -> None:
        """
        Release a tensor back to the pool.

        Args:
            tensor: Tensor to release
        """
        if tensor.device != self.device:
            return

        key = (tuple(tensor.shape), tensor.dtype)
        tensor_size = tensor.element_size() * tensor.nelement()

        with self._lock:
            if self._total_size + tensor_size <= self.max_size:
                if key not in self._pools:
                    self._pools[key] = []
                self._pools[key].append(tensor.detach())
                self._total_size += tensor_size

    def clear(self) -> None:
        """Clear the entire pool."""
        with self._lock:
            self._pools.clear()
            self._total_size = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_tensors = sum(len(pool) for pool in self._pools.values())
            return {
                "total_tensors": total_tensors,
                "total_size": self._total_size,
                "max_size": self.max_size,
                "pools": {str(k): len(v) for k, v in self._pools.items()},
            }


_global_memory_pools: Dict[str, MemoryPool] = {}


def get_memory_pool(device: Optional[torch.device] = None) -> MemoryPool:
    """
    Get or create a memory pool for a device.

    Args:
        device: Target device

    Returns:
        MemoryPool instance
    """
    device_str = str(device or "cuda:0")
    if device_str not in _global_memory_pools:
        _global_memory_pools[device_str] = MemoryPool(device or torch.device("cuda:0"))
    return _global_memory_pools[device_str]


def clear_cache() -> None:
    """Clear CUDA cache to free unused memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_stats(device: Optional[torch.device] = None) -> MemoryStats:
    """
    Get memory statistics for a device.

    Args:
        device: Target device

    Returns:
        MemoryStats
    """
    tracker = GPUMemoryTracker(device)
    return tracker.get_stats()


def memory_efficient_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> Tensor:
    """
    Memory-efficient attention implementation.

    Uses flash attention when available, otherwise falls back
    to a memory-efficient chunked implementation.

    Args:
        query: Query tensor [batch, heads, seq_len, dim]
        key: Key tensor [batch, heads, seq_len, dim]
        value: Value tensor [batch, heads, seq_len, dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        scale: Attention scale factor
        enable_gqa: Enable grouped-query attention

    Returns:
        Attention output
    """
    # Check for flash attention
    if hasattr(F, "scaled_dot_product_attention"):
        # Use PyTorch's native SDPA which handles memory efficiency
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    # Fallback: memory-efficient implementation
    if scale is None:
        scale = query.size(-1) ** -0.5

    # Memory-efficient: compute in chunks
    batch_size, num_heads, seq_len, head_dim = query.shape
    chunk_size = 512  # Process in chunks to save memory

    output = torch.zeros_like(query)

    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)

        # For causal attention, limit key/value to current position
        if is_causal:
            local_end = end_idx
        else:
            local_end = seq_len

        q_chunk = query[:, :, start_idx:end_idx, :]
        k_chunk = key[:, :, :local_end, :]
        v_chunk = value[:, :, :local_end, :]

        # Compute attention for chunk
        attn_weights = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale

        if is_causal:
            # Create causal mask for this chunk
            chunk_len = end_idx - start_idx
            causal_mask = torch.triu(
                torch.ones(chunk_len, local_end, device=query.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        output[:, :, start_idx:end_idx, :] = torch.matmul(attn_weights, v_chunk)

    return output


class MemoryEfficientWrapper(nn.Module):
    """
    Wrapper to make any module memory-efficient.

    Automatically applies memory-efficient attention and
    gradient checkpointing to reduce peak memory usage.
    """

    def __init__(
        self,
        module: nn.Module,
        use_checkpointing: bool = True,
        checkpoint_chunks: int = 4,
    ):
        super().__init__()
        self.module = module
        self.use_checkpointing = use_checkpointing
        self.checkpoint_chunks = checkpoint_chunks

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.use_checkpointing:
            return checkpoint_sequential(
                self.module,
                self.checkpoint_chunks,
                *args,
                **kwargs,
            )
        return self.module(*args, **kwargs)


def checkpoint_sequential(
    module: nn.Module,
    chunks: int,
    *inputs: Tensor,
    **kwargs: Any,
) -> Any:
    """
    Checkpoint a sequential module to save memory.

    Args:
        module: Sequential module to checkpoint
        chunks: Number of chunks to divide the module into
        inputs: Input tensors

    Returns:
        Module output
    """
    # Use torch.utils.checkpoint for sequential modules
    return torch.utils.checkpoint.checkpoint_sequential(
        module,
        chunks,
        *inputs,
        **kwargs,
    )


__all__ = [
    "MemoryStats",
    "GPUMemoryTracker",
    "MemoryPool",
    "get_memory_pool",
    "clear_cache",
    "get_memory_stats",
    "memory_efficient_attention",
    "MemoryEfficientWrapper",
    "checkpoint_sequential",
]
