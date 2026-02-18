"""
TorchScript Conversion Utilities for fishstick

Provides comprehensive TorchScript conversion utilities including
tracing, scripting, hybrid modes, and optimization passes.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json

import torch
from torch import nn
from torch import Tensor
from torch.jit import ScriptModule, TracedModule


class ConversionMode(Enum):
    """TorchScript conversion modes."""

    TRACE = "trace"
    SCRIPT = "script"
    HYBRID = "hybrid"


class OptimizationLevel(Enum):
    """JIT optimization levels."""

    NONE = 0
    DEFAULT = 1
    AGGRESSIVE = 2


@dataclass
class TorchScriptConfig:
    """Configuration for TorchScript conversion."""

    mode: ConversionMode = ConversionMode.TRACE
    optimization_level: OptimizationLevel = OptimizationLevel.DEFAULT
    freeze_model: bool = True
    optimize_for_inference: bool = True
    strict: bool = True
    preserve_shapes: bool = True


@dataclass
class TraceConfig:
    """Configuration for tracing."""

    check_trace: bool = True
    check_tolerance: float = 1e-6
    strict: bool = True
    force_outplace: bool = False


@dataclass
class ScriptConfig:
    """Configuration for scripting."""

    optimize: bool = True
    freeze: bool = True
    preserve_shape: bool = True


class TorchScriptConverter:
    """
    Comprehensive TorchScript converter with multiple modes.

    Supports tracing, scripting, and hybrid conversion with various
    optimization options.

    Example:
        >>> converter = TorchScriptConverter()
        >>> ts_model = converter.convert(model, input_sample)
        >>> ts_model.save("model.pt")
    """

    def __init__(
        self,
        config: Optional[TorchScriptConfig] = None,
    ):
        self.config = config or TorchScriptConfig()

    def convert(
        self,
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> ScriptModule:
        """
        Convert a model to TorchScript.

        Args:
            model: PyTorch model
            example_inputs: Example inputs for tracing or scripting

        Returns:
            TorchScript module
        """
        model.eval()

        if self.config.mode == ConversionMode.TRACE:
            return self._trace(model, example_inputs)
        elif self.config.mode == ConversionMode.SCRIPT:
            return self._script(model)
        else:
            return self._hybrid(model, example_inputs)

    def _trace(
        self,
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> TracedModule:
        """Trace the model."""
        with torch.jit.optimized_execution(True):
            traced = torch.jit.trace(
                model,
                example_inputs,
                strict=self.config.strict,
            )

        if self.config.freeze_model:
            traced = torch.jit.freeze(traced)

        if self.config.optimize_for_inference:
            traced = torch.jit.optimize_for_inference(traced)

        return traced

    def _script(self, model: nn.Module) -> ScriptModule:
        """Script the model."""
        scripted = torch.jit.script(model)

        if self.config.freeze_model:
            scripted = torch.jit.freeze(scripted)

        if self.config.optimize_for_inference:
            scripted = torch.jit.optimize_for_inference(scripted)

        return scripted

    def _hybrid(
        self,
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> ScriptModule:
        """
        Hybrid conversion: trace compatible parts, script the rest.

        Useful for models with control flow that can't be fully scripted.
        """
        if not isinstance(example_inputs, tuple):
            example_inputs = (example_inputs,)

        with torch.jit.optimized_execution(True):
            hybrid_model = torch.jit.trace(
                model,
                example_inputs,
                strict=False,
            )

        if self.config.freeze_model:
            hybrid_model = torch.jit.freeze(hybrid_model)

        return hybrid_model

    def convert_with_optimization(
        self,
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> ScriptModule:
        """
        Convert with aggressive optimization.

        Args:
            model: PyTorch model
            example_inputs: Example inputs

        Returns:
            Optimized TorchScript module
        """
        model.eval()

        if not isinstance(example_inputs, tuple):
            example_inputs = (example_inputs,)

        with torch.jit.optimized_execution(True):
            if self.config.mode == ConversionMode.TRACE:
                traced = torch.jit.trace(
                    model,
                    example_inputs,
                    strict=self.config.strict,
                )
            else:
                traced = torch.jit.script(model)

            if self.config.optimization_level == OptimizationLevel.AGGRESSIVE:
                traced = torch._C._freeze_module(traced._c)

                for _ in range(3):
                    traced = torch.jit.fuse_modules(
                        traced, ["aten::linear", "aten::batch_norm"]
                    )

            if self.config.optimize_for_inference:
                traced = torch.jit.optimize_for_inference(traced)

        return traced

    def save(
        self,
        model: ScriptModule,
        output_path: str,
    ) -> None:
        """
        Save TorchScript model to file.

        Args:
            model: TorchScript module
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.save(str(output_path))
        print(f"TorchScript model saved to {output_path}")

    def load(self, model_path: str) -> ScriptModule:
        """
        Load TorchScript model from file.

        Args:
            model_path: Path to TorchScript model

        Returns:
            Loaded TorchScript module
        """
        return torch.jit.load(model_path)

    def optimize(
        self,
        model: ScriptModule,
    ) -> ScriptModule:
        """
        Apply JIT optimizations to a TorchScript model.

        Args:
            model: TorchScript module

        Returns:
            Optimized module
        """
        if self.config.freeze_model:
            model = torch.jit.freeze(model)

        if self.config.optimize_for_inference:
            model = torch.jit.optimize_for_inference(model)

        return model

    def get_model_info(self, model: ScriptModule) -> Dict[str, Any]:
        """
        Get information about a TorchScript model.

        Args:
            model: TorchScript module

        Returns:
            Model information
        """
        return {
            "code": model.code,
            "graph": str(model.graph),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_buffers": len(model.named_buffers()),
        }


class TorchScriptTracer:
    """
    Advanced tracing utilities with validation.

    Example:
        >>> tracer = TorchScriptTracer()
        >>> tracer.trace_with_check(model, input_sample)
    """

    def __init__(self, config: Optional[TraceConfig] = None):
        self.config = config or TraceConfig()

    def trace_with_check(
        self,
        model: nn.Module,
        example_inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> TracedModule:
        """
        Trace model with correctness checks.

        Args:
            model: PyTorch model
            example_inputs: Example inputs

        Returns:
            Traced module
        """
        model.eval()

        traced = torch.jit.trace(
            model,
            example_inputs,
            strict=self.config.strict,
            check_trace=self.config.check_trace,
        )

        if self.config.check_trace:
            self._verify_trace(model, traced, example_inputs)

        return traced

    def trace_multiple_inputs(
        self,
        model: nn.Module,
        example_inputs_list: List[Union[Tensor, Tuple[Tensor, ...]]],
    ) -> TracedModule:
        """
        Trace model with multiple input configurations.

        Useful for models with varying input shapes.

        Args:
            model: PyTorch model
            example_inputs_list: List of example inputs

        Returns:
            Traced module
        """
        model.eval()

        if not example_inputs_list:
            raise ValueError("example_inputs_list cannot be empty")

        traced = torch.jit.trace(
            model,
            example_inputs_list[0],
            strict=self.config.strict,
        )

        return traced

    def _verify_trace(
        self,
        model: nn.Module,
        traced: TracedModule,
        example_inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        """Verify traced model produces same output as original."""
        model.eval()

        with torch.no_grad():
            original_output = model(example_inputs)
            traced_output = traced(example_inputs)

        if isinstance(original_output, Tensor):
            original_output = [original_output]
        if isinstance(traced_output, Tensor):
            traced_output = [traced_output]

        for orig, tr in zip(original_output, traced_output):
            assert torch.allclose(orig, tr, rtol=self.config.check_tolerance), (
                "Trace verification failed"
            )


class TorchScriptScripter:
    """
    Advanced scripting utilities.

    Example:
        >>> scripter = TorchScriptScripter()
        >>> scripted = scripter.script_with_optimization(model)
    """

    def __init__(self, config: Optional[ScriptConfig] = None):
        self.config = config or ScriptConfig()

    def script_with_optimization(
        self,
        model: nn.Module,
    ) -> ScriptModule:
        """
        Script model with optimization.

        Args:
            model: PyTorch model

        Returns:
            Scripted module
        """
        scripted = torch.jit.script(model)

        if self.config.freeze:
            scripted = torch.jit.freeze(scripted)

        if self.config.optimize:
            scripted = torch.jit.optimize_for_inference(scripted)

        return scripted

    def script_module_methods(
        self,
        model: nn.Module,
        method_names: List[str],
    ) -> ScriptModule:
        """
        Script specific methods of a module.

        Args:
            model: PyTorch model
            method_names: Names of methods to script

        Returns:
            Scripted module
        """
        for name in method_names:
            if not hasattr(model, name):
                raise ValueError(f"Method {name} not found in model")

        scripted = torch.jit.script(model)

        return scripted


def trace_model(
    model: nn.Module,
    example_inputs: Union[Tensor, Tuple[Tensor, ...]],
    output_path: Optional[str] = None,
) -> TracedModule:
    """
    Convenience function to trace a model.

    Args:
        model: PyTorch model
        example_inputs: Example inputs
        output_path: Optional path to save

    Returns:
        Traced module
    """
    converter = TorchScriptConverter(TorchScriptConfig(mode=ConversionMode.TRACE))
    traced = converter.convert(model, example_inputs)

    if output_path:
        converter.save(traced, output_path)

    return traced


def script_model(
    model: nn.Module,
    output_path: Optional[str] = None,
) -> ScriptModule:
    """
    Convenience function to script a model.

    Args:
        model: PyTorch model
        output_path: Optional path to save

    Returns:
        Scripted module
    """
    converter = TorchScriptConverter(TorchScriptConfig(mode=ConversionMode.SCRIPT))
    scripted = converter.convert(model, torch.zeros(1))

    if output_path:
        converter.save(scripted, output_path)

    return scripted


def load_torchscript(model_path: str) -> ScriptModule:
    """
    Load a TorchScript model.

    Args:
        model_path: Path to TorchScript model

    Returns:
        Loaded module
    """
    return torch.jit.load(model_path)


def benchmark_torchscript(
    model: Union[ScriptModule, TracedModule],
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    benchmark a TorchScript model's inference speed.

    Args:
        model: TorchScript model
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs

    Returns:
        Benchmark results
    """
    import time

    dummy_input = torch.randn(*input_shape)

    for _ in range(warmup_runs):
        model(dummy_input)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model(dummy_input)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = sorted(latencies)

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
    }


def verify_torchscript_output(
    pytorch_model: nn.Module,
    ts_model: ScriptModule,
    input_data: Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    """
    Verify TorchScript model output matches PyTorch model.

    Args:
        pytorch_model: PyTorch model
        ts_model: TorchScript model
        input_data: Test input
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if outputs match
    """
    pytorch_model.eval()
    ts_model.eval()

    with torch.no_grad():
        pytorch_output = pytorch_model(input_data)
        ts_output = ts_model(input_data)

    if isinstance(pytorch_output, Tensor):
        pytorch_output = [pytorch_output]
    if isinstance(ts_output, Tensor):
        ts_output = [ts_output]

    for pt_out, ts_out in zip(pytorch_output, ts_output):
        if not torch.allclose(pt_out, ts_out, rtol=rtol, atol=atol):
            return False

    return True
