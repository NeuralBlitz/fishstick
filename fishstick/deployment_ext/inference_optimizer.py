"""
Inference Optimization Utilities for fishstick

Provides inference optimization utilities including graph optimization,
operator fusion, memory optimization, caching strategies, and batch
inference optimization.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import time

import torch
from torch import nn
from torch import Tensor


class OptimizationPass(Enum):
    """Available optimization passes."""

    FUSION = "fusion"
    CONSTANT_FOLDING = "constant_folding"
    LAYERNORM_FUSION = "layernorm_fusion"
    LINEAR_FUSION = "linear_fusion"
    ATTENTION_FUSION = "attention_fusion"


@dataclass
class InferenceConfig:
    """Configuration for inference optimization."""

    use_cuda: bool = False
    use_cudnn_benchmark: bool = True
    num_threads: int = 1
    enable_amp: bool = False
    compile_mode: Optional[str] = None
    memory_format: str = "channels_last"


@dataclass
class OptimizationResult:
    """Results from inference optimization."""

    original_latency_ms: float
    optimized_latency_ms: float
    speedup: float
    memory_mb: float


class InferenceOptimizer:
    """
    Comprehensive inference optimizer with multiple optimization strategies.

    Example:
        >>> optimizer = InferenceOptimizer()
        >>> optimized = optimizer.optimize(model)
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self._setup()

    def _setup(self) -> None:
        """Setup inference environment."""
        if self.config.use_cuda and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = self.config.use_cudnn_benchmark

        if self.config.num_threads > 1:
            torch.set_num_threads(self.config.num_threads)

    def optimize(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Apply all optimizations to model.

        Args:
            model: PyTorch model

        Returns:
            Optimized model
        """
        model.eval()

        if self.config.use_cuda:
            model = model.cuda()

        if self.config.memory_format == "channels_last":
            model = model.to(memory_format=torch.channels_last)

        if self.config.compile_mode:
            model = torch.compile(model, mode=self.config.compile_mode)

        return model

    def optimize_for_inference(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Optimize model for inference using torch.jit.

        Args:
            model: PyTorch model

        Returns:
            Optimized model
        """
        model.eval()

        if hasattr(torch.jit, "optimize_for_inference"):
            if self.config.use_cuda:
                model = model.cuda()

            optimized = torch.jit.optimize_for_inference(model)
            return optimized

        return model


class GraphOptimizer:
    """
    Graph-level optimization utilities.

    Provides operator fusion, constant folding, and other graph
    transformations.

    Example:
        >>> optimizer = GraphOptimizer()
        >>> optimized = optimizer.fuse_layers(model)
    """

    def __init__(self):
        self.fusion_passes = [
            OptimizationPass.FUSION,
            OptimizationPass.CONSTANT_FOLDING,
        ]

    def fuse_layers(
        self,
        model: nn.Module,
        pass_names: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Fuse compatible layers in the model.

        Args:
            model: PyTorch model
            pass_names: Specific fusion passes to apply

        Returns:
            Model with fused layers
        """
        if pass_names is None:
            pass_names = ["aten::linear", "aten::batch_norm"]

        if hasattr(torch.jit, "fuse_modules"):
            model = torch.jit.fuse_modules(model, pass_names)

        return model

    def apply_fusion_passes(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Apply graph optimization passes.

        Args:
            model: PyTorch model

        Returns:
            Optimized model
        """
        if hasattr(model, "graph"):
            torch._C._jit_pass_inline(model.graph)

        return model

    def fold_constants(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Fold constant operations.

        Args:
            model: PyTorch model

        Returns:
            Model with folded constants
        """
        if hasattr(model, "graph"):
            torch._C._jit_pass_constant_propagation(model.graph)

        return model


class OperatorFusion:
    """
    Operator fusion utilities for common patterns.

    Fuses multiple operations into single kernels for better performance.

    Example:
        >>> fusion = OperatorFusion()
        >>> model = fusion.fuse_linear_bn(model)
    """

    @staticmethod
    def fuse_linear_bn(model: nn.Module) -> nn.Module:
        """
        Fuse Linear and BatchNorm1d layers.

        Args:
            model: PyTorch model

        Returns:
            Model with fused layers
        """
        for name, module in list(model.named_children()):
            if isinstance(module, nn.BatchNorm1d):
                parent_name = name.replace(".bn", "")
                if hasattr(model, parent_name):
                    linear = getattr(model, parent_name)

                    if isinstance(linear, nn.Linear):
                        OperatorFusion._fuse_linear_bn_impl(linear, module)

        return model

    @staticmethod
    def _fuse_linear_bn_impl(linear: nn.Linear, bn: nn.BatchNorm1d) -> None:
        """Implement Linear-BN fusion."""
        bn.eval()
        linear.eval()

        if bn.affine:
            bn_std = torch.sqrt(bn.running_var + bn.eps)
            weight = linear.weight.data / bn_std.view(1, -1)
            bias = (linear.bias.data - bn.running_mean) / bn_std * bn.weight + bn.bias

            linear.weight.data = weight * bn.weight.view(1, -1)
            if bias is not None:
                linear.bias.data = bias
        else:
            bn_std = torch.sqrt(bn.running_var + bn.eps)
            weight = linear.weight.data / bn_std.view(1, -1)
            bias = (linear.bias.data - bn.running_mean) / bn_std

            linear.weight.data = weight
            if bias is not None:
                linear.bias.data = bias

    @staticmethod
    def fuse_conv_bn(model: nn.Module) -> nn.Module:
        """
        Fuse Conv and BatchNorm layers.

        Args:
            model: PyTorch model

        Returns:
            Model with fused layers
        """
        for mod_type in [nn.BatchNorm2d, nn.BatchNorm3d]:
            for name, module in list(model.named_children()):
                if isinstance(module, mod_type):
                    parent_name = name.replace(".bn", "")
                    if hasattr(model, parent_name):
                        conv = getattr(model, parent_name)

                        if isinstance(conv, (nn.Conv2d, nn.Conv3d)):
                            OperatorFusion._fuse_conv_bn_impl(conv, module)

        return model

    @staticmethod
    def _fuse_conv_bn_impl(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
        """Implement Conv-BN fusion."""
        bn.eval()
        conv.eval()

        if bn.affine:
            bn_std = torch.sqrt(bn.running_var + bn.eps)

            conv.weight.data = conv.weight.data * (
                bn.weight.view(-1, 1, 1, 1) / bn_std.view(-1, 1, 1, 1)
            )

            if conv.bias is not None:
                conv.bias.data = (
                    conv.bias.data - bn.running_mean.view(-1)
                ) * bn.weight / bn_std + bn.bias
            else:
                conv.bias = nn.Parameter(
                    (-bn.running_mean.view(-1)) * bn.weight / bn_std + bn.bias
                )
        else:
            bn_std = torch.sqrt(bn.running_var + bn.eps)
            conv.weight.data = conv.weight.data / bn_std.view(-1, 1, 1, 1)

            if conv.bias is not None:
                conv.bias.data = conv.bias.data - bn.running_mean.view(-1) / bn_std
            else:
                conv.bias = nn.Parameter(-bn.running_mean.view(-1) / bn_std)


class MemoryOptimizer:
    """
    Memory optimization utilities.

    Provides gradient checkpointing, memory-efficient attention,
    and other memory-saving techniques.
    """

    @staticmethod
    def enable_gradient_checkpointing(
        model: nn.Module,
        checkpoint_ratio: float = 0.5,
    ) -> None:
        """
        Enable gradient checkpointing to save memory.

        Args:
            model: PyTorch model
            checkpoint_ratio: Ratio of layers to checkpoint
        """
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    @staticmethod
    def clear_cache() -> None:
        """Clear CUDA cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """
        Get current memory usage statistics.

        Returns:
            Memory statistics in MB
        """
        stats = {}

        if torch.cuda.is_available():
            stats["allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            stats["reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            stats["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

        return stats

    @staticmethod
    def reset_peak_stats() -> None:
        """Reset peak memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


class CachedInference:
    """
    Inference caching utilities.

    Provides result caching and computation reuse.
    """

    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Tensor] = {}

    def get_cache_key(self, inputs: Tuple[Tensor, ...]) -> str:
        """Generate cache key from inputs."""
        key_parts = []
        for inp in inputs:
            key_parts.append(str(inp.shape))
            key_parts.append(str(inp.sum().item()))
        return "|".join(key_parts)

    def cached_inference(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Run inference with caching.

        Args:
            model: PyTorch model
            inputs: Input tensors

        Returns:
            Model outputs
        """
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        cache_key = self.get_cache_key(inputs)

        if cache_key in self.cache:
            return self.cache[cache_key]

        model.eval()
        with torch.no_grad():
            outputs = model(*inputs)

        if len(self.cache) >= self.max_cache_size:
            self.cache.clear()

        self.cache[cache_key] = outputs

        return outputs

    def clear_cache(self) -> None:
        """Clear inference cache."""
        self.cache.clear()


class BatchInferenceOptimizer:
    """
    Batch inference optimization.

    Optimizes inference by batching multiple requests.
    """

    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        timeout_ms: float = 10.0,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_inputs: List[Tensor] = []
        self.last_batch_time = time.time()

    def add_input(self, input_tensor: Tensor) -> None:
        """Add input to batch queue."""
        self.pending_inputs.append(input_tensor)

    def should_process(self) -> bool:
        """Check if batch should be processed."""
        if len(self.pending_inputs) >= self.max_batch_size:
            return True

        elapsed = (time.time() - self.last_batch_time) * 1000
        if elapsed >= self.timeout_ms and len(self.pending_inputs) > 0:
            return True

        return False

    def process_batch(self) -> List[Tensor]:
        """
        Process pending inputs as batch.

        Returns:
            List of outputs
        """
        if not self.pending_inputs:
            return []

        batch = torch.cat(self.pending_inputs, dim=0)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)

        if isinstance(outputs, Tensor):
            outputs = torch.split(outputs, 1, dim=0)
            outputs = [o.squeeze(0) for o in outputs]

        self.pending_inputs = []
        self.last_batch_time = time.time()

        return outputs


def optimize_for_inference(
    model: nn.Module,
    use_cuda: bool = False,
) -> nn.Module:
    """
    Convenience function to optimize model for inference.

    Args:
        model: PyTorch model
        use_cuda: Whether to use CUDA

    Returns:
        Optimized model
    """
    config = InferenceConfig(use_cuda=use_cuda)
    optimizer = InferenceOptimizer(config)
    return optimizer.optimize(model)


def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10,
    use_cuda: bool = False,
) -> Dict[str, float]:
    """
    Benchmark model inference.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        use_cuda: Whether to use CUDA

    Returns:
        Benchmark results
    """
    import time

    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        dummy_input = torch.randn(*input_shape, device="cuda")
    else:
        dummy_input = torch.randn(*input_shape)

    model.eval()

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

        if use_cuda:
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()

            with torch.no_grad():
                _ = model(dummy_input)

            if use_cuda:
                torch.cuda.synchronize()

            latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
    }


def apply_memory_optimizations(model: nn.Module) -> nn.Module:
    """
    Apply memory optimizations to model.

    Args:
        model: PyTorch model

    Returns:
        Optimized model
    """
    optimizer = GraphOptimizer()
    model = optimizer.fuse_layers(model)

    return model
