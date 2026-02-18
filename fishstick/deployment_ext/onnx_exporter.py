"""
Enhanced ONNX Export Utilities for fishstick

Provides advanced ONNX export functionality including custom operators,
batch exports, runtime inference wrappers, and optimization utilities.
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


class ONNXOpsetVersion(Enum):
    """Supported ONNX opset versions."""

    OPSET_11 = 11
    OPSET_12 = 12
    OPSET_13 = 13
    OPSET_14 = 14
    OPSET_15 = 15
    OPSET_16 = 16
    OPSET_17 = 17
    OPSET_18 = 18


class OptimizationLevel(Enum):
    """ONNX optimization levels."""

    NONE = 0
    BASIC = 1
    EXTENDED = 2
    ALL = 3


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export."""

    opset_version: int = 14
    dynamic_axes: Optional[Dict[str, Any]] = None
    verbose: bool = False
    export_params: bool = True
    do_constant_folding: bool = True
    keep_initializers_as_inputs: bool = False
    custom_opsets: Optional[Dict[str, int]] = None
    aten_export: bool = False
    dispatch_tracer: bool = True


@dataclass
class ONNXRuntimeConfig:
    """Configuration for ONNX Runtime inference."""

    execution_providers: List[str] = field(
        default_factory=lambda: ["CPUExecutionProvider"]
    )
    graph_optimization_level: int = 3
    intra_op_num_threads: int = 1
    inter_op_num_threads: int = 1
    execution_mode: str = "sequential"


class ONNXExporter:
    """
    Enhanced ONNX Exporter with advanced features.

    Provides batch export, custom operators, runtime inference wrapper,
    and validation utilities.

    Example:
        >>> exporter = ONNXExporter()
        >>> exporter.export(model, "model.onnx", input_shape=(1, 3, 224, 224))
        >>> result = exporter.infer("model.onnx", torch.randn(1, 3, 224, 224))
    """

    def __init__(self, config: Optional[ONNXExportConfig] = None):
        self.config = config or ONNXExportConfig()

    def export(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_batch_size: bool = True,
    ) -> str:
        """
        Export a PyTorch model to ONNX format.

        Args:
            model: PyTorch model to export
            output_path: Path to save the ONNX model
            input_shape: Input tensor shape (excluding batch dimension)
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_batch_size: Whether to support dynamic batch size

        Returns:
            Path to exported model
        """
        model.eval()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(1, *input_shape)

        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        dynamic_axes = self.config.dynamic_axes or {}

        if dynamic_batch_size:
            if not dynamic_axes:
                dynamic_axes = {}
            for name in input_names:
                if name not in dynamic_axes:
                    dynamic_axes[name] = {0: "batch_size"}
            for name in output_names:
                if name not in dynamic_axes:
                    dynamic_axes[name] = {0: "batch_size"}

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=self.config.export_params,
            opset_version=self.config.opset_version,
            do_constant_folding=self.config.do_constant_folding,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=self.config.verbose,
            keep_initializers_as_inputs=self.config.keep_initializers_as_inputs,
            custom_opsets=self.config.custom_opsets,
            aten=self.config.aten_export,
        )

        print(f"Model exported to {output_path}")
        return str(output_path)

    def export_batch(
        self,
        model: nn.Module,
        output_dir: str,
        input_shapes: List[Tuple[int, ...]],
        model_names: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Export model for multiple input shapes.

        Useful for creating models optimized for different batch sizes.

        Args:
            model: PyTorch model to export
            output_dir: Directory to save models
            input_shapes: List of input shapes to export
            model_names: Optional names for each export

        Returns:
            List of exported model paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if model_names is None:
            model_names = [f"model_bs{i}" for i in range(len(input_shapes))]

        exported_paths = []

        for shape, name in zip(input_shapes, model_names):
            output_path = output_dir / f"{name}.onnx"
            self.export(model, str(output_path), shape)
            exported_paths.append(str(output_path))

        return exported_paths

    def validate(self, model_path: str) -> Dict[str, Any]:
        """
        Validate an ONNX model.

        Args:
            model_path: Path to ONNX model

        Returns:
            Validation results including graph info, inputs, outputs
        """
        try:
            import onnx

            model = onnx.load(model_path)
            onnx.checker.check_model(model)

            graph = model.graph

            inputs = []
            for inp in graph.input:
                shape = [
                    dim.dim_value if dim.dim_value > 0 else "dynamic"
                    for dim in inp.type.tensor_type.shape.dim
                ]
                inputs.append(
                    {
                        "name": inp.name,
                        "shape": shape,
                        "dtype": inp.type.tensor_type.elem_type,
                    }
                )

            outputs = []
            for out in graph.output:
                shape = [
                    dim.dim_value if dim.dim_value > 0 else "dynamic"
                    for dim in out.type.tensor_type.shape.dim
                ]
                outputs.append(
                    {
                        "name": out.name,
                        "shape": shape,
                        "dtype": out.type.tensor_type.elem_type,
                    }
                )

            return {
                "valid": True,
                "graph_name": graph.name,
                "ir_version": model.ir_version,
                "producer_name": model.producer_name,
                "producer_version": model.producer_version,
                "inputs": inputs,
                "outputs": outputs,
                "nodes_count": len(graph.node),
            }

        except ImportError:
            return {"valid": False, "error": "onnx package not installed"}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def optimize(
        self,
        input_path: str,
        output_path: str,
        level: OptimizationLevel = OptimizationLevel.ALL,
    ) -> str:
        """
        Optimize an ONNX model.

        Args:
            input_path: Path to input ONNX model
            output_path: Path to save optimized model
            level: Optimization level

        Returns:
            Path to optimized model
        """
        try:
            from onnxruntime.transformers import optimizer
            from onnx import load, save

            optimized = optimizer.optimize_model(
                input_path,
                optimization_level=level.value,
            )

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            save(optimized, str(output_path))
            print(f"Optimized model saved to {output_path}")

            return str(output_path)

        except ImportError:
            raise ImportError("onnxruntime transformers required for optimization")

    def infer(
        self,
        model_path: str,
        input_data: Tensor,
        config: Optional[ONNXRuntimeConfig] = None,
    ) -> List[Tensor]:
        """
        Run inference using ONNX Runtime.

        Args:
            model_path: Path to ONNX model
            input_data: Input tensor
            config: Runtime configuration

        Returns:
            List of output tensors
        """
        config = config or ONNXRuntimeConfig()

        try:
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                if config.graph_optimization_level >= 3
                else ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
            session_options.intra_op_num_threads = config.intra_op_num_threads
            session_options.inter_op_num_threads = config.inter_op_num_threads

            if config.execution_mode == "parallel":
                session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=self._get_providers(config.execution_providers),
            )

            input_name = session.get_inputs()[0].name
            output_names = [o.name for o in session.get_outputs()]

            outputs = session.run(output_names, {input_name: input_data.numpy()})

            return [torch.from_numpy(out) for out in outputs]

        except ImportError:
            raise ImportError("onnxruntime required for inference")

    def _get_providers(self, provider_names: List[str]) -> List[str]:
        """Map provider names to ONNX Runtime providers."""
        available = ort.get_available_providers()

        provider_map = {
            "CUDAExecutionProvider": "CUDAExecutionProvider",
            "CPUExecutionProvider": "CPUExecutionProvider",
            "TensorrtExecutionProvider": "TensorrtExecutionProvider",
            "OpenVINOExecutionProvider": "OpenVINOExecutionProvider",
        }

        providers = []
        for name in provider_names:
            if provider_map.get(name) in available:
                providers.append(provider_map[name])

        return providers if providers else ["CPUExecutionProvider"]

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get detailed information about an ONNX model.

        Args:
            model_path: Path to ONNX model

        Returns:
            Model information dictionary
        """
        validation = self.validate(model_path)

        if not validation.get("valid"):
            return validation

        return {
            "path": model_path,
            "graph_name": validation["graph_name"],
            "inputs": validation["inputs"],
            "outputs": validation["outputs"],
            "nodes": validation["nodes_count"],
            "ir_version": validation["ir_version"],
            "producer": f"{validation['producer_name']} {validation['producer_version']}",
        }


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    **kwargs,
) -> str:
    """
    Convenience function to export a model to ONNX.

    Args:
        model: PyTorch model
        output_path: Output path
        input_shape: Input shape
        **kwargs: Additional arguments for ONNXExporter

    Returns:
        Path to exported model
    """
    exporter = ONNXExporter()
    return exporter.export(model, output_path, input_shape, **kwargs)


def create_dynamic_quantized_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
) -> str:
    """
    Export a dynamically quantized model to ONNX.

    Applies dynamic quantization before export for smaller model size.

    Args:
        model: PyTorch model
        output_path: Output path
        input_shape: Input shape

    Returns:
        Path to exported model
    """
    model.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=torch.qint8,
    )

    return export_to_onnx(quantized_model, output_path, input_shape)


def benchmark_onnx_model(
    model_path: str,
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    Benchmark an ONNX model's inference speed.

    Args:
        model_path: Path to ONNX model
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs

    Returns:
        Benchmark results
    """
    import time

    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime required for benchmarking"}

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    dummy_input = torch.randn(*input_shape).numpy()

    for _ in range(warmup_runs):
        session.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = sorted(latencies)

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "throughput_samples_per_sec": 1000 / (sum(latencies) / len(latencies)),
    }


def compare_onnx_models(
    model_paths: List[str],
    input_shape: Tuple[int, ...],
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple ONNX models.

    Args:
        model_paths: List of ONNX model paths
        input_shape: Input tensor shape

    Returns:
        Comparison results
    """
    results = {}

    for path in model_paths:
        model_name = Path(path).stem
        results[model_name] = benchmark_onnx_model(path, input_shape)

    return results


def verify_onnx_output(
    pytorch_model: nn.Module,
    onnx_model_path: str,
    input_data: Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """
    Verify ONNX model output matches PyTorch model.

    Args:
        pytorch_model: PyTorch model
        onnx_model_path: Path to ONNX model
        input_data: Test input
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Verification results
    """
    pytorch_model.eval()

    with torch.no_grad():
        pytorch_output = pytorch_model(input_data)

    exporter = ONNXExporter()
    onnx_output = exporter.infer(onnx_model_path, input_data)

    if isinstance(pytorch_output, Tensor):
        pytorch_output = [pytorch_output]

    all_passed = True
    details = []

    for i, (pt_out, onnx_out) in enumerate(zip(pytorch_output, onnx_output)):
        is_close = torch.allclose(pt_out, onnx_out, rtol=rtol, atol=atol)
        max_diff = float(torch.max(torch.abs(pt_out - onnx_out)).item())

        if not is_close:
            all_passed = False

        details.append(
            {
                "output_index": i,
                "matches": is_close,
                "max_difference": max_diff,
                "pytorch_shape": list(pt_out.shape),
                "onnx_shape": list(onnx_out.shape),
            }
        )

    return {
        "verified": all_passed,
        "details": details,
    }
