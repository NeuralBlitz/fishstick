"""
Model Benchmarking Utilities for fishstick

Provides comprehensive benchmarking utilities including latency, throughput,
memory profiling, accuracy validation, and comparison tools.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import time
import json

import torch
from torch import nn
from torch import Tensor


class BenchmarkMode(Enum):
    """Benchmark modes."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ACCURACY = "accuracy"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    num_runs: int = 100
    warmup_runs: int = 10
    batch_size: int = 1
    use_cuda: bool = False
    profile_memory: bool = True
    precision: str = "fp32"


@dataclass
class BenchmarkResult:
    """Benchmark results."""

    mode: BenchmarkMode
    num_runs: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    memory_mb: Optional[float] = None
    throughput: Optional[float] = None


@dataclass
class ComparisonResult:
    """Model comparison results."""

    models: List[str]
    metrics: Dict[str, Dict[str, float]]


class LatencyBenchmark:
    """
    Latency benchmarking for model inference.

    Measures inference latency with statistical analysis.

    Example:
        >>> benchmark = LatencyBenchmark()
        >>> result = benchmark.benchmark(model, input_shape)
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

    def benchmark(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> BenchmarkResult:
        """
        Benchmark model latency.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape

        Returns:
            Benchmark results
        """
        if self.config.use_cuda and torch.cuda.is_available():
            model = model.cuda()
            dummy_input = torch.randn(*input_shape, device="cuda")
        else:
            dummy_input = torch.randn(*input_shape)

        model.eval()

        with torch.no_grad():
            for _ in range(self.config.warmup_runs):
                _ = model(dummy_input)

            if self.config.use_cuda:
                torch.cuda.synchronize()

            latencies = []
            memory_used = []

            for _ in range(self.config.num_runs):
                if self.config.profile_memory and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                start = time.perf_counter()

                with torch.no_grad():
                    _ = model(dummy_input)

                if self.config.use_cuda:
                    torch.cuda.synchronize()

                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

                if self.config.profile_memory and torch.cuda.is_available():
                    memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    memory_used.append(memory_mb)

        latencies.sort()

        import statistics

        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        return BenchmarkResult(
            mode=BenchmarkMode.LATENCY,
            num_runs=self.config.num_runs,
            mean_ms=mean_latency,
            min_ms=min(latencies),
            max_ms=max(latencies),
            p50_ms=latencies[len(latencies) // 2],
            p95_ms=latencies[int(len(latencies) * 0.95)],
            p99_ms=latencies[int(len(latencies) * 0.99)],
            std_ms=std_latency,
            memory_mb=sum(memory_used) / len(memory_used) if memory_used else None,
        )

    def benchmark_variable_batch(
        self,
        model: nn.Module,
        input_shapes: List[Tuple[int, ...]],
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark model with different batch sizes.

        Args:
            model: PyTorch model
            input_shapes: List of input shapes

        Returns:
            Results for each input shape
        """
        results = {}

        for i, shape in enumerate(input_shapes):
            result = self.benchmark(model, shape)
            results[f"batch_{i + 1}"] = result

        return results


class ThroughputBenchmark:
    """
    Throughput benchmarking.

    Measures samples processed per second.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

    def benchmark(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> BenchmarkResult:
        """
        Benchmark model throughput.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape

        Returns:
            Benchmark results
        """
        if self.config.use_cuda and torch.cuda.is_available():
            model = model.cuda()
            dummy_input = torch.randn(
                self.config.batch_size, *input_shape[1:], device="cuda"
            )
        else:
            dummy_input = torch.randn(self.config.batch_size, *input_shape[1:])

        model.eval()

        total_samples = self.config.num_runs * self.config.batch_size

        with torch.no_grad():
            for _ in range(self.config.warmup_runs):
                _ = model(dummy_input)

            if self.config.use_cuda:
                torch.cuda.synchronize()

            start = time.perf_counter()

            for _ in range(self.config.num_runs):
                with torch.no_grad():
                    _ = model(dummy_input)

            if self.config.use_cuda:
                torch.cuda.synchronize()

            elapsed_time = time.perf_counter() - start

        throughput = total_samples / elapsed_time

        return BenchmarkResult(
            mode=BenchmarkMode.THROUGHPUT,
            num_runs=self.config.num_runs,
            mean_ms=(elapsed_time * 1000) / self.config.num_runs,
            min_ms=0,
            max_ms=0,
            p50_ms=0,
            p95_ms=0,
            p99_ms=0,
            std_ms=0,
            throughput=throughput,
        )


class MemoryProfiler:
    """
    Memory profiling for models.

    Profiles memory usage and allocation patterns.
    """

    @staticmethod
    def profile_model(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        use_cuda: bool = True,
    ) -> Dict[str, float]:
        """
        Profile model memory usage.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            use_cuda: Whether to profile CUDA memory

        Returns:
            Memory profile
        """
        if use_cuda and torch.cuda.is_available():
            model = model.cuda()
            dummy_input = torch.randn(*input_shape, device="cuda")

            torch.cuda.reset_peak_memory_stats()

            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)

            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            current_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved_memory_mb = torch.cuda.memory_reserved() / (1024 * 1024)

            return {
                "peak_mb": peak_memory_mb,
                "current_mb": current_memory_mb,
                "reserved_mb": reserved_memory_mb,
            }
        else:
            import sys

            param_size = sum(
                p.nelement() * p.element_size() for p in model.parameters()
            )
            buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())

            return {
                "parameters_mb": param_size / (1024 * 1024),
                "buffers_mb": buffer_size / (1024 * 1024),
                "total_mb": (param_size + buffer_size) / (1024 * 1024),
            }

    @staticmethod
    def profile_layer_memory(
        model: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> Dict[str, Dict[str, float]]:
        """
        Profile memory usage per layer.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape

        Returns:
            Layer-wise memory profile
        """
        layer_memory = {}

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                param_size = sum(
                    p.nelement() * p.element_size() for p in module.parameters()
                )
                layer_memory[name] = {
                    "params_mb": param_size / (1024 * 1024),
                    "num_params": sum(p.numel() for p in module.parameters()),
                }

        return layer_memory


class AccuracyValidator:
    """
    Accuracy validation for deployed models.

    Compares model outputs to reference outputs.
    """

    def __init__(
        self,
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ):
        self.rtol = rtol
        self.atol = atol

    def validate(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        test_inputs: List[Tensor],
    ) -> Dict[str, Any]:
        """
        Validate model accuracy against reference.

        Args:
            model: Model to validate
            reference_model: Reference model
            test_inputs: Test inputs

        Returns:
            Validation results
        """
        model.eval()
        reference_model.eval()

        max_diff = 0.0
        all_close = True
        num_samples = len(test_inputs)

        with torch.no_grad():
            for i, inp in enumerate(test_inputs):
                model_out = model(inp)
                ref_out = reference_model(inp)

                if isinstance(model_out, Tensor):
                    model_out = [model_out]
                if isinstance(ref_out, Tensor):
                    ref_out = [ref_out]

                for mo, ro in zip(model_out, ref_out):
                    diff = torch.max(torch.abs(mo - ro)).item()
                    max_diff = max(max_diff, diff)

                    if not torch.allclose(mo, ro, rtol=self.rtol, atol=self.atol):
                        all_close = False

        return {
            "valid": all_close,
            "max_difference": max_diff,
            "num_samples": num_samples,
            "rtol": self.rtol,
            "atol": self.atol,
        }

    def validate_with_labels(
        self,
        model: nn.Module,
        test_data: List[Tuple[Tensor, Tensor]],
    ) -> Dict[str, Any]:
        """
        Validate with labeled test data.

        Args:
            model: Model to validate
            test_data: List of (input, label) tuples

        Returns:
            Validation results with metrics
        """
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_data:
                outputs = model(inputs)

                if outputs.dim() > 1 and outputs.size(-1) > 1:
                    predictions = outputs.argmax(dim=-1)
                else:
                    predictions = (outputs > 0.5).float()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }


class ModelComparator:
    """
    Compare multiple models.

    Compares models on various metrics.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

    def compare(
        self,
        models: Dict[str, nn.Module],
        input_shape: Tuple[int, ...],
    ) -> ComparisonResult:
        """
        Compare multiple models.

        Args:
            models: Dictionary of model_name -> model
            input_shape: Input shape for benchmarking

        Returns:
            Comparison results
        """
        latency_bench = LatencyBenchmark(self.config)

        metrics = {}

        for name, model in models.items():
            result = latency_bench.benchmark(model, input_shape)

            metrics[name] = {
                "mean_ms": result.mean_ms,
                "min_ms": result.min_ms,
                "max_ms": result.max_ms,
                "p95_ms": result.p95_ms,
                "p99_ms": result.p99_ms,
                "std_ms": result.std_ms,
            }

            if result.memory_mb:
                metrics[name]["memory_mb"] = result.memory_mb

        return ComparisonResult(
            models=list(models.keys()),
            metrics=metrics,
        )

    def print_comparison(self, results: ComparisonResult) -> None:
        """Print comparison results."""
        print(f"\n{'Model':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Memory (MB)':<12}")
        print("-" * 60)

        for model_name, metrics in results.metrics.items():
            mem = (
                f"{metrics.get('memory_mb', 0):.1f}"
                if "memory_mb" in metrics
                else "N/A"
            )
            print(
                f"{model_name:<20} {metrics['mean_ms']:<12.2f} "
                f"{metrics['p95_ms']:<12.2f} {mem:<12}"
            )


class ModelBenchmarker:
    """
    Unified benchmarking interface.

    Provides comprehensive benchmarking with all metrics.

    Example:
        >>> benchmarker = ModelBenchmarker()
        >>> results = benchmarker.benchmark_full(model, input_shape)
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.latency_bench = LatencyBenchmark(self.config)
        self.throughput_bench = ThroughputBenchmark(self.config)
        self.memory_profiler = MemoryProfiler()

    def benchmark_full(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        """
        Run full benchmark suite.

        Args:
            model: PyTorch model
            input_shape: Input shape

        Returns:
            Full benchmark results
        """
        latency_result = self.latency_bench.benchmark(model, input_shape)
        throughput_result = self.throughput_bench.benchmark(model, input_shape)

        memory_profile = self.memory_profiler.profile_model(
            model, input_shape, use_cuda=self.config.use_cuda
        )

        return {
            "latency": {
                "mean_ms": latency_result.mean_ms,
                "min_ms": latency_result.min_ms,
                "max_ms": latency_result.max_ms,
                "p50_ms": latency_result.p50_ms,
                "p95_ms": latency_result.p95_ms,
                "p99_ms": latency_result.p99_ms,
                "std_ms": latency_result.std_ms,
            },
            "throughput": {
                "samples_per_sec": throughput_result.throughput,
            },
            "memory": memory_profile,
            "config": {
                "num_runs": self.config.num_runs,
                "warmup_runs": self.config.warmup_runs,
                "batch_size": self.config.batch_size,
            },
        }

    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str,
    ) -> None:
        """
        Export benchmark results to JSON.

        Args:
            results: Benchmark results
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results exported to {output_path}")


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    use_cuda: bool = False,
) -> Dict[str, float]:
    """
    Convenience function to benchmark a model.

    Args:
        model: PyTorch model
        input_shape: Input shape
        num_runs: Number of benchmark runs
        use_cuda: Whether to use CUDA

    Returns:
        Benchmark results
    """
    config = BenchmarkConfig(num_runs=num_runs, use_cuda=use_cuda)
    benchmarker = ModelBenchmarker(config)
    results = benchmarker.benchmark_full(model, input_shape)
    return results


def compare_models(
    models: Dict[str, nn.Module],
    input_shape: Tuple[int, ...],
) -> ComparisonResult:
    """
    Convenience function to compare multiple models.

    Args:
        models: Dictionary of model_name -> model
        input_shape: Input shape

    Returns:
        Comparison results
    """
    comparator = ModelComparator()
    return comparator.compare(models, input_shape)
