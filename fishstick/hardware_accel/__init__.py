"""
Hardware Acceleration Module for fishstick.

Provides comprehensive hardware acceleration utilities including:
- GPU optimization and CUDA graph support
- TPU compatibility layers
- Mixed precision training
- Memory-efficient implementations
- Kernel fusion utilities
- Gradient checkpointing
- Device management

Modules:
- device_manager: Device detection and management
- memory_utils: GPU memory management utilities
- gpu_optimizer: CUDA graph optimization, kernel caching
- mixed_precision: Mixed precision training utilities
- tpu_compat: TPU compatibility layer
- xla_ops: XLA operations wrapper
- gradient_checkpointing: Activation checkpointing
- offload_utils: CPU/GPU offloading utilities
- kernel_fusion: CUDA kernel fusion utilities
- custom_kernels: Custom CUDA kernel implementations

Key Classes:
- DeviceManager: Unified device management
- GPUMemoryTracker: Memory usage tracking
- MixedPrecisionTrainer: Mixed precision training wrapper
- GradientCheckpointing: Memory-efficient training
- KernelFusion: Fused kernel operations
- TPUModel: TPU-compatible model wrapper
"""

from typing import (
    Optional,
    Tuple,
    Dict,
    Any,
    List,
    Union,
    Callable,
    Literal,
)

import torch
from torch import Tensor, nn

# Device Management
from .device_manager import (
    DeviceManager,
    get_optimal_device,
    is_cuda_available,
    is_tpu_available,
    get_device_properties,
    DeviceType,
)

# Memory Utilities
from .memory_utils import (
    GPUMemoryTracker,
    MemoryPool,
    clear_cache,
    get_memory_stats,
    MemoryStats,
    memory_efficient_attention,
)

# GPU Optimization
from .gpu_optimizer import (
    CUDAGraphOptimizer,
    cached_kernel,
    optimize_model,
    BenchmarkResult,
)

# Mixed Precision
from .mixed_precision import (
    MixedPrecisionTrainer,
    AMPGradScaler,
    Precision,
    create_amp_context,
    convert_to_fp16,
    convert_to_bf16,
)

# TPU Compatibility
from .tpu_compat import (
    TPUModel,
    TPUStrategy,
    is_tpu,
    tpu_mesh,
    configure_tpu,
)

# XLA Operations
from .xla_ops import (
    XLAOperation,
    xla_compile,
    xla_mark_step,
    xla_optimization_barrier,
)

# Gradient Checkpointing
from .gradient_checkpointing import (
    CheckpointedModule,
    checkpoint,
    checkpoint_sequential,
    create_checkpoint_function,
)

# Offload Utilities
from .offload_utils import (
    CPUOffload,
    OffloadableModule,
    offload_to_cpu,
    offload_to_gpu,
)

# Kernel Fusion
from .kernel_fusion import (
    KernelFusion,
    fuse_linear_gelu,
    fuse_linear_bias,
    FusedOperation,
)

# Custom Kernels
from .custom_kernels import (
    custom_matmul,
    custom_layer_norm,
    custom_softmax,
    custom_attention,
    fused_optimizer_kernel,
)


__all__ = [
    # Device Management
    "DeviceManager",
    "get_optimal_device",
    "is_cuda_available",
    "is_tpu_available",
    "get_device_properties",
    "DeviceType",
    # Memory Utilities
    "GPUMemoryTracker",
    "MemoryPool",
    "clear_cache",
    "get_memory_stats",
    "MemoryStats",
    "memory_efficient_attention",
    # GPU Optimization
    "CUDAGraphOptimizer",
    "cached_kernel",
    "optimize_model",
    "BenchmarkResult",
    # Mixed Precision
    "MixedPrecisionTrainer",
    "AMPGradScaler",
    "Precision",
    "create_amp_context",
    "convert_to_fp16",
    "convert_to_bf16",
    # TPU Compatibility
    "TPUModel",
    "TPUStrategy",
    "is_tpu",
    "tpu_mesh",
    "configure_tpu",
    # XLA Operations
    "XLAOperation",
    "xla_compile",
    "xla_mark_step",
    "xla_optimization_barrier",
    # Gradient Checkpointing
    "CheckpointedModule",
    "checkpoint",
    "checkpoint_sequential",
    "create_checkpoint_function",
    # Offload Utilities
    "CPUOffload",
    "OffloadableModule",
    "offload_to_cpu",
    "offload_to_gpu",
    # Kernel Fusion
    "KernelFusion",
    "fuse_linear_gelu",
    "fuse_linear_bias",
    "FusedOperation",
    # Custom Kernels
    "custom_matmul",
    "custom_layer_norm",
    "custom_softmax",
    "custom_attention",
    "fused_optimizer_kernel",
]


__version__ = "0.1.0"
