"""
GPU Optimization Utilities for fishstick.

Provides CUDA graph optimization, kernel caching, and performance tuning.

Based on:
- NVIDIA CUDA Graphs (NCCL)
- PyTorch CUDA graph capture
- Kernel fusion optimizations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Callable, Union
from contextlib import contextmanager
import threading
import time
import functools

import torch
from torch import Tensor, nn
from torch.utils.cpp_extension import load_inline


@dataclass
class BenchmarkResult:
    """Results from benchmarking a model or operation."""

    operation_name: str
    avg_time_ms: float
    std_time_ms: float
    throughput: float
    memory_used_mb: float
    num_runs: int

    def __str__(self) -> str:
        return (
            f"{self.operation_name}: "
            f"{self.avg_time_ms:.2f}±{self.std_time_ms:.2f}ms, "
            f"{self.throughput:.1f} ops/s, "
            f"{self.memory_used_mb:.1f}MB"
        )


class CUDAGraphOptimizer:
    """
    Optimize models using CUDA Graphs.

    CUDA Graphs capture a entire CUDA kernel launch sequence
    and replay it, reducing kernel launch overhead.

    Attributes:
        device: Target device for optimization
        warmup_runs: Number of warmup runs before capture
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        warmup_runs: int = 10,
    ):
        self.device = device or torch.device("cuda:0")
        self.warmup_runs = warmup_runs

        self._graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self._static_inputs: Dict[str, Tuple[Tensor, ...]] = {}
        self._static_outputs: Dict[str, Tuple[Tensor, ...]] = {}
        self._graph_modules: Dict[str, nn.Module] = {}
        self._lock = threading.Lock()

    def capture(
        self,
        name: str,
        model: nn.Module,
        sample_input: Tuple[Tensor, ...],
    ) -> None:
        """
        Capture a CUDA graph for a model.

        Args:
            name: Name identifier for the graph
            model: Model to capture
            sample_input: Sample input for warmup and capture
        """
        model = model.to(self.device)
        model.eval()

        # Move inputs to device
        static_inputs = tuple(
            x.to(self.device).clone() if isinstance(x, Tensor) else x
            for x in sample_input
        )

        # Warmup runs
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = model(*static_inputs)

        torch.cuda.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()

        # Create static tensors for outputs
        with torch.no_grad():
            static_outputs = model(*static_inputs)

            # Handle single tensor or tuple output
            if isinstance(static_outputs, Tensor):
                static_outputs = (static_outputs,)

            # Clone outputs to use as static buffers
            static_outputs = tuple(o.clone().detach() for o in static_outputs)

        # Re-record with graph capture
        with torch.no_grad():
            # Use static inputs for recording
            for i, x in enumerate(sample_input):
                if isinstance(x, Tensor):
                    static_inputs[i].copy_(x.to(self.device))

            graph.capture_begin()
            outputs = model(*static_inputs)
            graph.capture_end()

        # Store graph and static buffers
        with self._lock:
            self._graphs[name] = graph
            self._static_inputs[name] = static_inputs
            self._static_outputs[name] = static_outputs
            self._graph_modules[name] = model

        # Clear cache after capture
        torch.cuda.empty_cache()

    def run(self, name: str, input_tensor: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """
        Run an optimized CUDA graph.

        Args:
            name: Graph name
            input_tensor: Input tensors

        Returns:
            Model output
        """
        if name not in self._graphs:
            raise ValueError(f"Graph '{name}' not found. Call capture() first.")

        static_inputs = self._static_inputs[name]
        static_outputs = self._static_outputs[name]

        # Copy new inputs to static buffers
        for i, x in enumerate(input_tensor):
            if isinstance(x, Tensor):
                static_inputs[i].copy_(x)

        # Replay graph
        self._graphs[name].replay()

        return static_outputs

    def clear(self, name: Optional[str] = None) -> None:
        """
        Clear captured graphs.

        Args:
            name: Specific graph to clear, or None to clear all
        """
        with self._lock:
            if name is None:
                self._graphs.clear()
                self._static_inputs.clear()
                self._static_outputs.clear()
                self._graph_modules.clear()
            elif name in self._graphs:
                del self._graphs[name]
                del self._static_inputs[name]
                del self._static_outputs[name]
                del self._graph_modules[name]

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "num_graphs": len(self._graphs),
            "graph_names": list(self._graphs.keys()),
        }


def cached_kernel(
    fn: Callable[..., Tensor],
    cache_size: int = 10,
) -> Callable[..., Tensor]:
    """
    Decorator to cache kernel results based on input shapes.

    Useful for operations with deterministic outputs for the
    same input shapes (e.g., certain initialization operations).

    Args:
        fn: Function to cache
        cache_size: Maximum number of cached results

    Returns:
        Wrapped function with caching
    """
    cache: Dict[Tuple[Tuple[int, ...], torch.dtype], Tensor] = {}
    cache_order: List[Tuple[Tuple[int, ...], torch.dtype]] = []
    lock = threading.Lock()

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Tensor:
        # Extract shape and dtype from first tensor argument
        shape_dtype: Optional[Tuple[Tuple[int, ...], torch.dtype]] = None

        for arg in args:
            if isinstance(arg, Tensor):
                shape_dtype = (tuple(arg.shape), arg.dtype)
                break

        if shape_dtype is None:
            return fn(*args, **kwargs)

        with lock:
            if shape_dtype in cache:
                return cache[shape_dtype]

        result = fn(*args, **kwargs)

        with lock:
            if len(cache) >= cache_size:
                oldest = cache_order.pop(0)
                del cache[oldest]

            cache[shape_dtype] = result
            cache_order.append(shape_dtype)

        return result

    return wrapper


def optimize_model(
    model: nn.Module,
    example_input: Tuple[Tensor, ...],
    optimization_level: int = 2,
    use_cudnn: bool = True,
    use_math_precise: bool = False,
    benchmark: bool = True,
) -> nn.Module:
    """
    Apply various optimizations to a model.

    Args:
        model: Model to optimize
        example_input: Example input for optimization
        optimization_level: Level of optimization (0-3)
        use_cudnn: Enable cuDNN
        use_math_precise: Use mathematically precise operations
        benchmark: Run benchmark to find best algorithm

    Returns:
        Optimized model
    """
    model = model.eval()

    # Set cuDNN settings
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = use_cudnn

        if use_cudnn and benchmark:
            torch.backends.cudnn.benchmark = True

        if use_math_precise:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Apply torch compile for optimization level >= 2
    if optimization_level >= 2 and hasattr(torch, "compile"):
        backend = "inductor" if optimization_level >= 3 else "eager"
        model = torch.compile(model, backend=backend)

    # Enable gradient checkpointing for level 3
    if optimization_level >= 3:
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()

    return model


class KernelCache:
    """
    Cache for frequently used kernels with shape-based lookup.
    """

    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
        self._lock = threading.Lock()
        self._access_count: Dict[str, int] = {}

    def get(
        self,
        key: str,
        factory: Callable[[], Any],
    ) -> Any:
        """
        Get or create a cached kernel.

        Args:
            key: Cache key
            factory: Factory function to create kernel if not cached

        Returns:
            Cached or newly created kernel
        """
        with self._lock:
            if key in self._cache:
                self._access_count[key] = self._access_count.get(key, 0) + 1
                return self._cache[key]

            # Create new kernel
            kernel = factory()

            # Evict if cache is full
            if len(self._cache) >= self._max_size:
                self._evict_lru()

            self._cache[key] = kernel
            self._access_count[key] = 1

            return kernel

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_count:
            return

        lru_key = min(self._access_count, key=self._access_count.get)
        del self._cache[lru_key]
        del self._access_count[lru_key]

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "total_accesses": sum(self._access_count.values()),
            }


_global_kernel_cache = KernelCache()


def benchmark_operation(
    fn: Callable[..., Any],
    *args: Any,
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """
    Benchmark a function or model.

    Args:
        fn: Function to benchmark
        *args: Arguments to pass to function
        num_runs: Number of benchmark iterations
        warmup_runs: Number of warmup runs
        device: Device to benchmark on
        **kwargs: Keyword arguments

    Returns:
        BenchmarkResult
    """
    device = device or torch.device("cuda:0")

    # Move tensors to device
    args = tuple(x.to(device) if isinstance(x, Tensor) else x for x in args)

    # Warmup
    for _ in range(warmup_runs):
        _ = fn(*args, **kwargs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times: List[float] = []
    memory_start = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0

    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = fn(*args, **kwargs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    memory_used = (
        torch.cuda.memory_allocated(device) - memory_start
        if device.type == "cuda"
        else 0
    )

    import statistics

    return BenchmarkResult(
        operation_name=fn.__name__ if hasattr(fn, "__name__") else str(fn),
        avg_time_ms=statistics.mean(times),
        std_time_ms=statistics.stdev(times) if len(times) > 1 else 0,
        throughput=num_runs / (sum(times) / 1000),  # ops per second
        memory_used_mb=memory_used / (1024**2),
        num_runs=num_runs,
    )


__all__ = [
    "CUDAGraphOptimizer",
    "cached_kernel",
    "optimize_model",
    "BenchmarkResult",
    "KernelCache",
    "benchmark_operation",
]
