"""
Model Speedup and Inference Optimization Utilities

Layer fusion, operator optimization, memory-efficient inference,
model benchmarking, and runtime profiling utilities.

References:
- https://pytorch.org/docs/stable/generated/torch.jit.fusion.html
- https://pytorch.org/tutorials/intermediate/source_code_for_profiling_torch.html
- https://github.com/pytorch/ao/blob/main/torch/ao/quantization/fx/fuse.py
"""

from __future__ import annotations

from typing import Optional, List, Dict, Callable, Tuple, Union, Any, Literal
from enum import Enum
import time
import copy
import warnings

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity


class ModelSpeedupProfiler:
    """Profiler for model speed and memory analysis.

    Provides detailed profiling of model inference including
    layer-wise timing and memory usage.

    Args:
        model: Model to profile
        device: Device for profiling

    Example:
        >>> profiler = ModelSpeedupProfiler(model)
        >>> results = profiler.profile(input_data)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.device = device
        self.layer_times: Dict[str, float] = {}
        self.layer_memory: Dict[str, float] = {}
        self.total_time: float = 0.0
        self.total_memory: float = 0.0

    def profile(
        self,
        input_data: Tensor,
        warmup_runs: int = 10,
        num_runs: int = 100,
    ) -> Dict[str, Any]:
        """Profile model inference.

        Args:
            input_data: Input tensor for profiling
            warmup_runs: Number of warmup iterations
            num_runs: Number of profiling iterations

        Returns:
            Dict of profiling results
        """
        model = self.model.to(self.device)
        model.eval()
        input_data = input_data.to(self.device)

        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(input_data)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(input_data)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        self.total_time = (end_time - start_time) / num_runs

        results = {
            "total_time_ms": self.total_time * 1000,
            "throughput_samples_per_sec": num_runs / (end_time - start_time),
            "layer_times": self._profile_layers(input_data),
            "memory_usage_mb": self._get_memory_usage(),
        }

        return results

    def _profile_layers(
        self,
        input_data: Tensor,
    ) -> Dict[str, float]:
        """Profile individual layer execution times."""
        layer_times = {}

        def hook_fn(name):
            def hook(module, input, output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                layer_times[name] = time.perf_counter()

            return hook

        handles = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                handles.append(module.register_forward_pre_hook(hook_fn(name)))

        with torch.no_grad():
            self.model(input_data)

        for handle in handles:
            handle.remove()

        return layer_times

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def profile_with_profiler(
        self,
        input_data: Tensor,
    ) -> Dict[str, Any]:
        """Profile using PyTorch profiler.

        Args:
            input_data: Input tensor

        Returns:
            Profiler results
        """
        model = self.model.to(self.device)
        model.eval()
        input_data = input_data.to(self.device)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            if torch.cuda.is_available()
            else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad():
                model(input_data)

        return {
            "cpu_time_total": prof.key_averages().cpu_time_total,
            "cuda_time_total": prof.key_averages().cuda_time_total,
            "cpu_time_total_ms": prof.key_averages().cpu_time_total / 1000,
            "cuda_time_total_ms": prof.key_averages().cuda_time_total / 1000,
            "table": prof.key_averages().table(sort_by="cpu_time_total", row_limit=10),
        }


class LayerFuser:
    """Layer fusion utilities for model optimization.

    Fuses consecutive layers for faster inference.

    Args:
        model: Model to fuse
        fuse_layers: Whether to fuse conv+bn, linear+activation, etc.
    """

    def __init__(
        self,
        model: nn.Module,
        fuse_layers: bool = True,
    ):
        self.model = model
        self.fuse_layers = fuse_layers
        self.fused_layers: Dict[str, nn.Module] = {}

    def fuse_conv_bn(
        self,
        model: Optional[nn.Module] = None,
    ) -> nn.Module:
        """Fuse Conv2d + BatchNorm2d layers.

        Args:
            model: Model to fuse (uses self.model if None)

        Returns:
            Model with fused layers
        """
        if model is None:
            model = self.model

        model = copy.deepcopy(model)

        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                    prev_module = (
                        parent.get_submodule(child_name) if child_name else None
                    )
                else:
                    prev_module = None
                    parent = None

                if isinstance(prev_module, nn.Conv2d):
                    fused = self._fuse_conv_bn(prev_module, module)
                    if parent:
                        setattr(parent, child_name, fused)

        return model

    def _fuse_conv_bn(
        self,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
    ) -> nn.Module:
        """Fuse a Conv2d and BatchNorm2d into a single module."""
        fused = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
        )

        w_conv = conv.weight.clone()
        w_bn = bn.weight / torch.sqrt(bn.running_var + bn.eps)

        fused.weight.data = w_conv * w_bn.view(-1, 1, 1, 1)

        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.out_channels)

        b_bn = bn.bias - bn.weight * bn.running_mean / torch.sqrt(
            bn.running_var + bn.eps
        )

        fused.bias.data = w_bn * b_conv + b_bn

        return fused

    def fuse_linear_relu(
        self,
        model: Optional[nn.Module] = None,
    ) -> nn.Module:
        """Fuse Linear + ReLU layers."""
        if model is None:
            model = self.model

        model = copy.deepcopy(model)

        for name, module in model.named_children():
            if isinstance(module, nn.ReLU):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                    prev_module = (
                        parent.get_submodule(child_name) if child_name else None
                    )
                else:
                    prev_module = None
                    parent = None

                if isinstance(prev_module, nn.Linear):
                    weight = prev_module.weight
                    bias = prev_module.bias

                    new_linear = nn.Linear(
                        weight.shape[1],
                        weight.shape[0],
                        bias is not None,
                    )
                    new_linear.weight = weight
                    new_linear.bias = bias

                    relu_inplace = nn.ReLU(inplace=True)

                    fused = nn.Sequential(new_linear, relu_inplace)

                    if parent:
                        setattr(parent, child_name, fused)

        return model

    def auto_fuse(
        self,
        model: Optional[nn.Module] = None,
    ) -> nn.Module:
        """Automatically fuse known patterns.

        Args:
            model: Model to fuse

        Returns:
            Fused model
        """
        fused_model = self.fuse_conv_bn(model)
        fused_model = self.fuse_linear_relu(fused_model)

        return fused_model


class InferenceOptimizer:
    """Inference optimization wrapper.

    Applies various optimizations for faster inference including
    JIT compilation, optimization passes, and memory optimization.

    Args:
        model: Model to optimize
        optimization_level: Optimization level (0-3)
    """

    def __init__(
        self,
        model: nn.Module,
        optimization_level: int = 2,
    ):
        self.model = model
        self.optimization_level = optimization_level
        self.optimized_model: Optional[nn.Module] = None

    def optimize(
        self,
        example_input: Tensor,
    ) -> nn.Module:
        """Apply inference optimizations.

        Args:
            example_input: Example input for tracing

        Returns:
            Optimized model
        """
        self.model.eval()

        with torch.no_grad():
            if self.optimization_level >= 1:
                self.optimized_model = torch.jit.trace(self.model, example_input)
            else:
                self.optimized_model = copy.deepcopy(self.model)

            if self.optimization_level >= 2:
                self.optimized_model = torch.jit.optimize_for_inference(
                    self.optimized_model
                )

            if self.optimization_level >= 3:
                self.optimized_model = self._apply_custom_optimizations(
                    self.optimized_model
                )

        return self.optimized_model

    def _apply_custom_optimizations(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """Apply custom optimization passes."""
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass

        return model

    def optimize_for_mobile(self) -> nn.Module:
        """Optimize for mobile deployment."""
        if self.optimized_model is None:
            raise RuntimeError("Must call optimize() first")

        self.optimized_model = torch.jit.optimize_for_inference(
            self.optimized_model,
            optimization_for_inference=True,
        )

        return self.optimized_model


class ModelBenchmarker:
    """Benchmark utility for model comparison.

    Compares models on latency, throughput, and accuracy.

    Args:
        models: Dict of model_name -> model
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.models = models
        self.device = device
        self.results: Dict[str, Dict[str, float]] = {}

    def benchmark_latency(
        self,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark inference latency.

        Args:
            input_shape: Input tensor shape
            num_runs: Number of benchmark iterations
            warmup_runs: Warmup iterations

        Returns:
            Dict of model_name -> latency in ms
        """
        latencies = {}

        for name, model in self.models.items():
            model = model.to(self.device)
            model.eval()

            dummy_input = torch.randn(input_shape).to(self.device)

            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = model(dummy_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(dummy_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()

            latencies[name] = (end - start) / num_runs * 1000

        self.results["latency"] = latencies
        return latencies

    def benchmark_throughput(
        self,
        input_shape: Tuple[int, ...],
        batch_size: int = 32,
        duration_seconds: float = 10.0,
    ) -> Dict[str, float]:
        """Benchmark throughput.

        Args:
            input_shape: Input shape (without batch dimension)
            batch_size: Batch size
            duration_seconds: Benchmark duration

        Returns:
            Dict of model_name -> samples/sec
        """
        throughputs = {}

        for name, model in self.models.items():
            model = model.to(self.device)
            model.eval()

            dummy_input = torch.randn(batch_size, *input_shape).to(self.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            num_samples = 0

            while time.perf_counter() - start < duration_seconds:
                with torch.no_grad():
                    _ = model(dummy_input)
                num_samples += batch_size

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            throughputs[name] = num_samples / elapsed

        self.results["throughput"] = throughputs
        return throughputs

    def benchmark_memory(
        self,
        input_shape: Tuple[int, ...],
    ) -> Dict[str, float]:
        """Benchmark memory usage.

        Args:
            input_shape: Input shape

        Returns:
            Dict of model_name -> memory in MB
        """
        if not torch.cuda.is_available():
            return {}

        memory_usages = {}

        for name, model in self.models.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model = model.to(self.device)
            model.eval()

            dummy_input = torch.randn(input_shape).to(self.device)

            with torch.no_grad():
                _ = model(dummy_input)

            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            memory_usages[name] = memory_mb

        self.results["memory"] = memory_usages
        return memory_usages

    def run_all_benchmarks(
        self,
        input_shape: Tuple[int, ...],
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, float]]:
        """Run all benchmarks.

        Args:
            input_shape: Input shape
            batch_size: Batch size

        Returns:
            All benchmark results
        """
        self.benchmark_latency(input_shape)
        self.benchmark_throughput(input_shape, batch_size)
        self.benchmark_memory(input_shape)

        return self.results


class MemoryEfficientForward:
    """Memory-efficient forward pass utilities.

    Provides utilities for reducing memory footprint during inference.

    Args:
        model: Model to optimize
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        self.model = model

    def gradient_checkpointing_forward(
        self,
        x: Tensor,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Forward pass with gradient checkpointing.

        Reduces memory usage by recomputing activations during backward pass.

        Args:
            x: Input tensor
            use_reentrant: Use reentrant checkpointing API

        Returns:
            Model output
        """
        if hasattr(torch.utils, "checkpoint"):
            return torch.utils.checkpoint.checkpoint(
                self.model,
                x,
                use_reentrant=use_reentrant,
            )
        else:
            with torch.no_grad():
                return self.model(x)

    def chunk_forward(
        self,
        x: Tensor,
        chunk_size: int = 32,
    ) -> Tensor:
        """Process input in chunks to save memory.

        Args:
            x: Input tensor
            chunk_size: Size of chunks to process

        Returns:
            Concatenated outputs
        """
        self.model.eval()

        dim_size = x.shape[0]
        outputs = []

        for i in range(0, dim_size, chunk_size):
            chunk = x[i : i + chunk_size]
            with torch.no_grad():
                output = self.model(chunk)
            outputs.append(output)

        return torch.cat(outputs, dim=0)

    def clear_cache(self):
        """Clear model cache and free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()

    def get_optimal_batch_size(
        self,
        input_shape: Tuple[int, ...],
        target_memory_mb: float = 1024.0,
    ) -> int:
        """Estimate optimal batch size for target memory.

        Args:
            input_shape: Input tensor shape
            target_memory_mb: Target memory usage in MB

        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return 32

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        dummy = torch.randn(1, *input_shape[1:]).cuda()

        with torch.no_grad():
            _ = self.model(dummy)

        memory_per_sample = torch.cuda.max_memory_allocated() / 1024 / 1024

        available_memory = target_memory_mb * 0.8

        optimal_batch = max(1, int(available_memory / max(memory_per_sample, 0.1)))

        return optimal_batch


class OperatorFusion:
    """Operator fusion for common patterns.

    Fuses multiple operations into single kernels for speedup.

    Args:
        model: Model to fuse
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        self.model = model
        self.fused_ops: List[Callable] = []

    def fuse_add_relu(
        self,
        x: Tensor,
        add_tensor: Tensor,
    ) -> Tensor:
        """Fuse Add + ReLU into single operation."""
        return F.relu(x + add_tensor)

    def fuse_mul_add(
        self,
        x: Tensor,
        mul_tensor: Tensor,
        add_tensor: Tensor,
    ) -> Tensor:
        """Fuse Mul + Add into single operation."""
        return x * mul_tensor + add_tensor

    def fuse_conv_bias(
        self,
        conv: nn.Conv2d,
        x: Tensor,
    ) -> Tensor:
        """Fuse conv with bias addition."""
        return F.conv2d(
            x,
            conv.weight,
            conv.bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )

    def detect_fusable_patterns(
        self,
    ) -> List[Tuple[str, str]]:
        """Detect fusable patterns in model.

        Returns:
            List of (layer1, layer2) pairs that can be fused
        """
        fusable = []

        prev_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and prev_layer == "conv":
                fusable.append((prev_layer, name))
            elif isinstance(module, nn.Conv2d):
                prev_layer = "conv"
            elif isinstance(module, nn.BatchNorm2d) and prev_layer == "conv":
                fusable.append((name, "batchnorm"))
                prev_layer = None
            else:
                prev_layer = None

        return fusable


class JITOptimizer:
    """JIT compilation optimization utilities.

    Provides utilities for JIT compilation and optimization.

    Args:
        model: Model to optimize
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        self.model = model
        self.jitted_model: Optional[nn.Module] = None

    def trace_model(
        self,
        example_input: Tensor,
    ) -> nn.Module:
        """Trace model using torch.jit.trace.

        Args:
            example_input: Example input for tracing

        Returns:
            Traced model
        """
        self.model.eval()
        self.jitted_model = torch.jit.trace(self.model, example_input)
        return self.jitted_model

    def script_model(
        self,
    ) -> nn.Module:
        """Script model using torch.jit.script.

        Returns:
            Scripted model
        """
        self.model.eval()
        self.jitted_model = torch.jit.script(self.model)
        return self.jitted_model

    def optimize_traced_model(
        self,
    ) -> nn.Module:
        """Apply optimizations to traced model."""
        if self.jitted_model is None:
            raise RuntimeError("Must trace or script model first")

        self.jitted_model = torch.jit.optimize_for_inference(self.jitted_model)
        return self.jitted_model

    def freeze_model(self) -> nn.Module:
        """Freeze model by inlining parameters."""
        if self.jitted_model is None:
            raise RuntimeError("Must trace or script model first")

        self.jitted_model = torch.jit.freeze(self.jitted_model)
        return self.jitted_model


class ActivationCheckpointing:
    """Activation checkpointing for memory-efficient training.

    Reduces memory usage by recomputing activations during backward pass.

    Args:
        model: Model to apply checkpointing to
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        self.model = model

    def apply_checkpointing(
        self,
        layers: List[nn.Module],
        checkpoint_ratio: float = 0.5,
    ):
        """Apply activation checkpointing to specified layers.

        Args:
            layers: Layers to apply checkpointing to
            checkpoint_ratio: Ratio of layers to checkpoint
        """
        num_to_checkpoint = int(len(layers) * checkpoint_ratio)

        for i, layer in enumerate(layers[:num_to_checkpoint]):
            layer = self._wrap_with_checkpoint(layer)

    def _wrap_with_checkpoint(
        self,
        module: nn.Module,
    ) -> nn.Module:
        """Wrap module with checkpoint."""
        if hasattr(torch.utils.checkpoint, "checkpoint"):
            original_forward = module.forward

            def checkpoint_forward(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(
                    original_forward, *args, **kwargs, use_reentrant=False
                )

            module.forward = checkpoint_forward

        return module
