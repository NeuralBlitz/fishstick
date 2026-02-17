"""
Fishstick Model Serving and Deployment Utilities

Comprehensive model serving infrastructure for deploying ML models at scale.
Supports multiple export formats, serving backends, optimization techniques,
and monitoring capabilities.

Key Features:
- Multi-format model export (TorchScript, ONNX, TensorRT, CoreML, OpenVINO)
- Multiple serving backends (TorchServe, Triton, FastAPI, Flask, gRPC)
- Advanced optimization (quantization, pruning, batching, graph optimization)
- Request handling (preprocessing, validation, batching, postprocessing)
- Comprehensive monitoring (latency, throughput, memory, errors)
- Auto-scaling and load balancing capabilities
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum, auto
import time
import threading
import queue
import asyncio
from collections import deque
import warnings
import json
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.quantization import (
    quantize_dynamic,
    QuantStub,
    DeQuantStub,
    prepare,
    convert,
)

# Type variables for generic types
T = TypeVar("T")
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")

# Configure logging
logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported model export formats."""

    TORCHSCRIPT = auto()
    ONNX = auto()
    TENSORRT = auto()
    COREML = auto()
    OPENVINO = auto()


class ServingBackend(Enum):
    """Supported serving backends."""

    TORCHSERVE = auto()
    TRITON = auto()
    FASTAPI = auto()
    FLASK = auto()
    GRPC = auto()


class QuantizationMode(Enum):
    """Quantization modes for model optimization."""

    DYNAMIC = auto()
    STATIC = auto()
    QAT = auto()  # Quantization-Aware Training


@dataclass
class ModelConfig:
    """Configuration for model serving.

    Attributes:
        model_name: Name identifier for the model
        version: Model version string
        batch_size: Maximum batch size for inference
        max_latency_ms: Maximum acceptable latency in milliseconds
        input_shape: Expected input tensor shape
        output_shape: Expected output tensor shape
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        precision: Model precision ('fp32', 'fp16', 'int8')
    """

    model_name: str
    version: str = "1.0.0"
    batch_size: int = 1
    max_latency_ms: float = 100.0
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    device: str = "cpu"
    precision: str = "fp32"


@dataclass
class InferenceMetrics:
    """Metrics collected during inference.

    Attributes:
        latency_ms: Inference latency in milliseconds
        throughput_qps: Queries per second
        memory_mb: Memory usage in megabytes
        batch_size: Actual batch size used
        timestamp: When the inference occurred
    """

    latency_ms: float
    throughput_qps: float
    memory_mb: float
    batch_size: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HealthStatus:
    """Model health check status.

    Attributes:
        is_healthy: Whether the model is healthy
        last_check: Timestamp of last health check
        error_message: Error message if unhealthy
        latency_ms: Health check latency
    """

    is_healthy: bool
    last_check: datetime
    error_message: Optional[str] = None
    latency_ms: float = 0.0


# =============================================================================
# Model Exporters
# =============================================================================


class BaseExporter(ABC):
    """Abstract base class for model exporters."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def export(self, model: nn.Module, output_path: Union[str, Path]) -> Path:
        """Export model to specified format.

        Args:
            model: PyTorch model to export
            output_path: Path to save exported model

        Returns:
            Path to exported model file
        """
        pass

    @abstractmethod
    def validate(self, model_path: Union[str, Path]) -> bool:
        """Validate exported model.

        Args:
            model_path: Path to exported model

        Returns:
            True if model is valid, False otherwise
        """
        pass


class TorchScriptExporter(BaseExporter):
    """Export PyTorch models to TorchScript format.

    TorchScript provides a way to create serializable and optimizable
    models from PyTorch code.

    Example:
        >>> model = MyModel()
        >>> exporter = TorchScriptExporter(config)
        >>> exporter.export(model, "model.pt")
    """

    def __init__(
        self,
        config: ModelConfig,
        method: str = "trace",
        optimize: bool = True,
        example_inputs: Optional[torch.Tensor] = None,
    ):
        """Initialize TorchScript exporter.

        Args:
            config: Model configuration
            method: Export method ('trace' or 'script')
            optimize: Whether to optimize the exported model
            example_inputs: Example inputs for tracing
        """
        super().__init__(config)
        self.method = method
        self.optimize = optimize
        self.example_inputs = example_inputs

    def export(self, model: nn.Module, output_path: Union[str, Path]) -> Path:
        """Export model to TorchScript format.

        Args:
            model: PyTorch model to export
            output_path: Path to save the TorchScript model

        Returns:
            Path to exported .pt or .pth file
        """
        model.eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.method == "trace":
            if self.example_inputs is None:
                # Create dummy input based on config
                shape = self.config.input_shape or (1, 3, 224, 224)
                self.example_inputs = torch.randn(*shape)

            scripted_model = torch.jit.trace(model, self.example_inputs)
        elif self.method == "script":
            scripted_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if self.optimize:
            scripted_model = torch.jit.optimize_for_inference(scripted_model)

        scripted_model.save(str(output_path))
        logger.info(f"Model exported to TorchScript: {output_path}")

        return output_path

    def validate(self, model_path: Union[str, Path]) -> bool:
        """Validate TorchScript model.

        Args:
            model_path: Path to TorchScript model

        Returns:
            True if model loads successfully
        """
        try:
            model = torch.jit.load(str(model_path))
            # Run a forward pass to verify
            if self.example_inputs is not None:
                with torch.no_grad():
                    _ = model(self.example_inputs)
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


class ONNXExporter(BaseExporter):
    """Export PyTorch models to ONNX format.

    ONNX (Open Neural Network Exchange) enables interoperability
    between different deep learning frameworks.

    Example:
        >>> exporter = ONNXExporter(config, opset_version=13)
        >>> exporter.export(model, "model.onnx")
    """

    def __init__(
        self,
        config: ModelConfig,
        opset_version: int = 13,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        simplify: bool = True,
    ):
        """Initialize ONNX exporter.

        Args:
            config: Model configuration
            opset_version: ONNX opset version
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes configuration
            simplify: Whether to simplify ONNX graph
        """
        super().__init__(config)
        self.opset_version = opset_version
        self.input_names = input_names or ["input"]
        self.output_names = output_names or ["output"]
        self.dynamic_axes = dynamic_axes
        self.simplify = simplify

    def export(
        self,
        model: nn.Module,
        output_path: Union[str, Path],
        example_inputs: Optional[torch.Tensor] = None,
    ) -> Path:
        """Export model to ONNX format.

        Args:
            model: PyTorch model to export
            output_path: Path to save ONNX model
            example_inputs: Example input tensor

        Returns:
            Path to exported .onnx file
        """
        try:
            import onnx
        except ImportError:
            raise ImportError("onnx package required. Install with: pip install onnx")

        model.eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if example_inputs is None:
            shape = self.config.input_shape or (1, 3, 224, 224)
            example_inputs = torch.randn(*shape)

        # Export to ONNX
        torch.onnx.export(
            model,
            example_inputs,
            str(output_path),
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
        )

        # Simplify if requested
        if self.simplify:
            try:
                import onnxsim

                onnx_model = onnx.load(str(output_path))
                onnx_model, check = onnxsim.simplify(onnx_model)
                onnx.save(onnx_model, str(output_path))
            except ImportError:
                logger.warning("onnx-simplifier not available, skipping simplification")

        logger.info(f"Model exported to ONNX: {output_path}")
        return output_path

    def validate(self, model_path: Union[str, Path]) -> bool:
        """Validate ONNX model.

        Args:
            model_path: Path to ONNX model

        Returns:
            True if model is valid
        """
        try:
            import onnx

            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)
            return True
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            return False


class TensorRTExporter(BaseExporter):
    """Export models to NVIDIA TensorRT format for GPU optimization.

    TensorRT provides high-performance deep learning inference
    on NVIDIA GPUs with layer fusion, precision calibration, and
    kernel auto-tuning.

    Example:
        >>> exporter = TensorRTExporter(config, fp16_mode=True)
        >>> exporter.export(model, "model.trt")
    """

    def __init__(
        self,
        config: ModelConfig,
        fp16_mode: bool = True,
        int8_mode: bool = False,
        max_workspace_size: int = 1 << 30,  # 1GB
        max_batch_size: int = 32,
    ):
        """Initialize TensorRT exporter.

        Args:
            config: Model configuration
            fp16_mode: Enable FP16 precision
            int8_mode: Enable INT8 precision (requires calibration)
            max_workspace_size: Maximum workspace size in bytes
            max_batch_size: Maximum batch size for inference
        """
        super().__init__(config)
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size

    def export(
        self,
        model: nn.Module,
        output_path: Union[str, Path],
        onnx_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Export model to TensorRT engine.

        First exports to ONNX, then converts to TensorRT engine.

        Args:
            model: PyTorch model to export
            output_path: Path to save TensorRT engine
            onnx_path: Optional path to pre-exported ONNX model

        Returns:
            Path to exported .trt or .engine file
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("tensorrt package required")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # First export to ONNX if needed
        if onnx_path is None:
            onnx_path = output_path.with_suffix(".onnx")
            onnx_exporter = ONNXExporter(self.config)
            onnx_exporter.export(model, onnx_path)

        # Build TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size

        if self.fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        if self.int8_mode:
            config.set_flag(trt.BuilderFlag.INT8)

        # Build engine
        engine = builder.build_engine(network, config)

        # Serialize and save
        with open(output_path, "wb") as f:
            f.write(engine.serialize())

        logger.info(f"Model exported to TensorRT: {output_path}")
        return output_path

    def validate(self, model_path: Union[str, Path]) -> bool:
        """Validate TensorRT engine.

        Args:
            model_path: Path to TensorRT engine

        Returns:
            True if engine loads successfully
        """
        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)

            with open(model_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            return engine is not None
        except Exception as e:
            logger.error(f"TensorRT validation failed: {e}")
            return False


class CoreMLExporter(BaseExporter):
    """Export models to Apple CoreML format for iOS/macOS deployment.

    CoreML provides optimized on-device machine learning for
    Apple platforms with hardware acceleration.

    Example:
        >>> exporter = CoreMLExporter(config, compute_units="ALL")
        >>> exporter.export(model, "model.mlpackage")
    """

    def __init__(
        self,
        config: ModelConfig,
        compute_units: str = "ALL",
        minimum_deployment_target: Optional[str] = None,
    ):
        """Initialize CoreML exporter.

        Args:
            config: Model configuration
            compute_units: Compute units ('CPU_ONLY', 'CPU_AND_GPU', 'ALL')
            minimum_deployment_target: Minimum iOS/macOS version
        """
        super().__init__(config)
        self.compute_units = compute_units
        self.minimum_deployment_target = minimum_deployment_target

    def export(
        self,
        model: nn.Module,
        output_path: Union[str, Path],
        example_inputs: Optional[torch.Tensor] = None,
    ) -> Path:
        """Export model to CoreML format.

        Args:
            model: PyTorch model to export
            output_path: Path to save CoreML model
            example_inputs: Example input tensor

        Returns:
            Path to exported .mlpackage or .mlmodel file
        """
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError(
                "coremltools required. Install with: pip install coremltools"
            )

        model.eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if example_inputs is None:
            shape = self.config.input_shape or (1, 3, 224, 224)
            example_inputs = torch.randn(*shape)

        # Trace model
        traced_model = torch.jit.trace(model, example_inputs)

        # Convert to CoreML
        compute_units = getattr(ct.ComputeUnit, self.compute_units)

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="input", shape=example_inputs.shape)],
            compute_units=compute_units,
            minimum_deployment_target=self.minimum_deployment_target,
        )

        # Save model
        if output_path.suffix == ".mlpackage":
            mlmodel.save(str(output_path))
        else:
            mlmodel.save(str(output_path.with_suffix(".mlmodel")))

        logger.info(f"Model exported to CoreML: {output_path}")
        return output_path

    def validate(self, model_path: Union[str, Path]) -> bool:
        """Validate CoreML model.

        Args:
            model_path: Path to CoreML model

        Returns:
            True if model is valid
        """
        try:
            import coremltools as ct

            _ = ct.models.MLModel(str(model_path))
            return True
        except Exception as e:
            logger.error(f"CoreML validation failed: {e}")
            return False


class OpenVINOExporter(BaseExporter):
    """Export models to Intel OpenVINO format for optimized inference.

    OpenVINO optimizes and accelerates deep learning inference
    on Intel hardware (CPU, GPU, VPU).

    Example:
        >>> exporter = OpenVINOExporter(config)
        >>> exporter.export(model, "model.xml")
    """

    def __init__(
        self,
        config: ModelConfig,
        data_type: str = "FP32",
        mean_values: Optional[List[float]] = None,
        scale_values: Optional[List[float]] = None,
    ):
        """Initialize OpenVINO exporter.

        Args:
            config: Model configuration
            data_type: Data type ('FP32', 'FP16', 'INT8')
            mean_values: Mean values for input normalization
            scale_values: Scale values for input normalization
        """
        super().__init__(config)
        self.data_type = data_type
        self.mean_values = mean_values
        self.scale_values = scale_values

    def export(
        self,
        model: nn.Module,
        output_path: Union[str, Path],
        onnx_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Export model to OpenVINO IR format.

        Args:
            model: PyTorch model to export
            output_path: Path to save OpenVINO model
            onnx_path: Optional path to pre-exported ONNX model

        Returns:
            Path to exported .xml file (weights in .bin)
        """
        try:
            from openvino.tools import mo
            from openvino.runtime import serialize
        except ImportError:
            raise ImportError("openvino-dev package required")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # First export to ONNX
        if onnx_path is None:
            onnx_path = output_path.with_suffix(".onnx")
            onnx_exporter = ONNXExporter(self.config)
            onnx_exporter.export(model, onnx_path)

        # Convert to OpenVINO IR
        ov_model = mo.convert_model(
            str(onnx_path),
            data_type=self.data_type,
            mean_values=self.mean_values,
            scale_values=self.scale_values,
        )

        # Serialize
        serialize(ov_model, str(output_path))

        logger.info(f"Model exported to OpenVINO: {output_path}")
        return output_path

    def validate(self, model_path: Union[str, Path]) -> bool:
        """Validate OpenVINO model.

        Args:
            model_path: Path to OpenVINO .xml file

        Returns:
            True if model is valid
        """
        try:
            from openvino.runtime import Core

            core = Core()
            model = core.read_model(str(model_path))
            return model is not None
        except Exception as e:
            logger.error(f"OpenVINO validation failed: {e}")
            return False


# =============================================================================
# Serving Backends
# =============================================================================


class BaseServingBackend(ABC):
    """Abstract base class for serving backends."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.is_running = False

    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load model into serving backend.

        Args:
            model_path: Path to model file
        """
        pass

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference on inputs.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor
        """
        pass

    @abstractmethod
    def start(self, **kwargs) -> None:
        """Start the serving backend."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the serving backend."""
        pass


class TorchServeWrapper(BaseServingBackend):
    """Wrapper for TorchServe serving backend.

    TorchServe is a flexible and easy-to-use tool for serving
    PyTorch models with built-in metrics and logging.

    Example:
        >>> config = ModelConfig("my_model")
        >>> server = TorchServeWrapper(config)
        >>> server.start(model_store="/models")
    """

    def __init__(
        self,
        config: ModelConfig,
        handler: Optional[str] = None,
        extra_files: Optional[List[str]] = None,
    ):
        """Initialize TorchServe wrapper.

        Args:
            config: Model configuration
            handler: Custom handler file path
            extra_files: Additional files to include
        """
        super().__init__(config)
        self.handler = handler
        self.extra_files = extra_files or []
        self.process: Optional[Any] = None

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load model for TorchServe.

        Args:
            model_path: Path to model archive (.mar file)
        """
        self.model_path = Path(model_path)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference via TorchServe API.

        Args:
            inputs: Input tensor

        Returns:
            Output from TorchServe inference
        """
        import requests

        # Convert tensor to bytes
        input_bytes = inputs.numpy().tobytes()

        response = requests.post(
            f"http://localhost:8080/predictions/{self.config.model_name}",
            data=input_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )

        response.raise_for_status()
        result = np.frombuffer(response.content, dtype=np.float32)
        return torch.from_numpy(result)

    def create_archive(
        self,
        model_file: Union[str, Path],
        output_path: Union[str, Path],
        version: str = "1.0",
    ) -> Path:
        """Create TorchServe model archive.

        Args:
            model_file: Path to model file
            output_path: Path to save archive
            version: Model version

        Returns:
            Path to .mar archive
        """
        try:
            from model_archiver.model_packaging import package_model
        except ImportError:
            raise ImportError("torch-model-archiver required")

        args = type(
            "Args",
            (),
            {
                "model_name": self.config.model_name,
                "version": version,
                "model_file": str(model_file),
                "handler": self.handler or "image_classifier",
                "extra_files": ",".join(self.extra_files) if self.extra_files else None,
                "export_path": str(Path(output_path).parent),
                "force": True,
            },
        )()

        package_model(args)
        return Path(output_path)

    def start(
        self,
        model_store: str = "/tmp/model_store",
        port: int = 8080,
        management_port: int = 8081,
        metrics_port: int = 8082,
        **kwargs,
    ) -> None:
        """Start TorchServe.

        Args:
            model_store: Directory containing model archives
            port: Inference API port
            management_port: Management API port
            metrics_port: Metrics API port
        """
        import subprocess

        cmd = [
            "torchserve",
            "--start",
            "--model-store",
            model_store,
            "--ts-config",
            kwargs.get("config_file", ""),
            "--port",
            str(port),
            "--management-port",
            str(management_port),
            "--metrics-port",
            str(metrics_port),
        ]

        if self.model_path:
            cmd.extend(["--models", f"{self.config.model_name}={self.model_path}"])

        self.process = subprocess.Popen(cmd)
        self.is_running = True

        # Wait for server to start
        time.sleep(5)
        logger.info(f"TorchServe started on port {port}")

    def stop(self) -> None:
        """Stop TorchServe."""
        import subprocess

        if self.process:
            subprocess.run(["torchserve", "--stop"])
            self.process = None

        self.is_running = False
        logger.info("TorchServe stopped")


class TritonInferenceServer(BaseServingBackend):
    """NVIDIA Triton Inference Server wrapper.

    Triton provides optimized inference for multiple frameworks
    with dynamic batching, model ensemble, and GPU acceleration.

    Example:
        >>> config = ModelConfig("my_model")
        >>> server = TritonInferenceServer(config)
        >>> server.start(model_repository="/models")
    """

    def __init__(
        self,
        config: ModelConfig,
        max_batch_size: int = 32,
        dynamic_batching: bool = True,
        instance_count: int = 1,
        gpus: Optional[List[int]] = None,
    ):
        """Initialize Triton inference server.

        Args:
            config: Model configuration
            max_batch_size: Maximum batch size
            dynamic_batching: Enable dynamic batching
            instance_count: Number of model instances
            gpus: List of GPU IDs to use
        """
        super().__init__(config)
        self.max_batch_size = max_batch_size
        self.dynamic_batching = dynamic_batching
        self.instance_count = instance_count
        self.gpus = gpus or [0]
        self.client: Optional[Any] = None

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load model into Triton.

        Args:
            model_path: Path to model directory in Triton format
        """
        self.model_path = Path(model_path)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference via Triton client.

        Args:
            inputs: Input tensor

        Returns:
            Inference results
        """
        if self.client is None:
            raise RuntimeError("Triton client not initialized. Start server first.")

        try:
            from tritonclient.http import InferenceServerClient, InferInput
        except ImportError:
            raise ImportError("tritonclient required")

        # Prepare input
        input_data = inputs.numpy()
        infer_input = InferInput("input", input_data.shape, "FP32")
        infer_input.set_data_from_numpy(input_data)

        # Run inference
        response = self.client.infer(self.config.model_name, [infer_input])

        output = response.as_numpy("output")
        return torch.from_numpy(output)

    def start(
        self,
        model_repository: str,
        http_port: int = 8000,
        grpc_port: int = 8001,
        metrics_port: int = 8002,
        **kwargs,
    ) -> None:
        """Start Triton Inference Server.

        Args:
            model_repository: Path to model repository
            http_port: HTTP service port
            grpc_port: gRPC service port
            metrics_port: Metrics port
        """
        import subprocess

        cmd = [
            "tritonserver",
            "--model-repository",
            model_repository,
            "--http-port",
            str(http_port),
            "--grpc-port",
            str(grpc_port),
            "--metrics-port",
            str(metrics_port),
        ]

        if self.gpus:
            cmd.extend(["--gpus", ",".join(map(str, self.gpus))])

        self.process = subprocess.Popen(cmd)
        self.is_running = True

        # Initialize client
        try:
            from tritonclient.http import InferenceServerClient

            self.client = InferenceServerClient(url=f"localhost:{http_port}")
        except ImportError:
            logger.warning("tritonclient not available")

        time.sleep(10)  # Wait for server
        logger.info(f"Triton started on ports HTTP:{http_port}, gRPC:{grpc_port}")

    def stop(self) -> None:
        """Stop Triton server."""
        if hasattr(self, "process") and self.process:
            self.process.terminate()
            self.process = None

        self.is_running = False
        logger.info("Triton server stopped")


class FastAPIServer(BaseServingBackend):
    """FastAPI-based custom serving backend.

    Provides a lightweight, high-performance REST API for
    model serving with automatic documentation.

    Example:
        >>> config = ModelConfig("my_model")
        >>> server = FastAPIServer(config)
        >>> server.start(host="0.0.0.0", port=8000)
    """

    def __init__(
        self,
        config: ModelConfig,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        batching: bool = False,
        max_batch_time_ms: float = 10.0,
    ):
        """Initialize FastAPI server.

        Args:
            config: Model configuration
            preprocess_fn: Custom preprocessing function
            postprocess_fn: Custom postprocessing function
            batching: Enable request batching
            max_batch_time_ms: Maximum time to wait for batch
        """
        super().__init__(config)
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.batching = batching
        self.max_batch_time_ms = max_batch_time_ms
        self.app: Optional[Any] = None
        self.server: Optional[Any] = None

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load model for serving.

        Args:
            model_path: Path to model file
        """
        # Load based on file extension
        path = Path(model_path)

        if path.suffix in [".pt", ".pth"]:
            self.model = torch.jit.load(str(model_path))
        elif path.suffix == ".onnx":
            import onnxruntime as ort

            self.model = ort.InferenceSession(str(model_path))
        else:
            self.model = torch.load(str(model_path), map_location=self.config.device)

        self.model.eval()

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run model inference.

        Args:
            inputs: Input tensor

        Returns:
            Model predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            if hasattr(self.model, "run"):
                # ONNX Runtime
                outputs = self.model.run(None, {"input": inputs.numpy()})
                return torch.from_numpy(outputs[0])
            else:
                return self.model(inputs)

    def start(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
        """Start FastAPI server.

        Args:
            host: Host address
            port: Port number
        """
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("FastAPI server starting...")
            yield
            # Shutdown
            logger.info("FastAPI server shutting down...")

        self.app = FastAPI(
            title=f"Fishstick Model Server - {self.config.model_name}",
            version=self.config.version,
            lifespan=lifespan,
        )

        class PredictionRequest(BaseModel):
            inputs: List[List[float]]

        class PredictionResponse(BaseModel):
            outputs: List[List[float]]
            latency_ms: float

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model": self.config.model_name,
                "version": self.config.version,
            }

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            start_time = time.time()

            try:
                # Convert to tensor
                inputs = torch.tensor(request.inputs, dtype=torch.float32)

                # Preprocess
                if self.preprocess_fn:
                    inputs = self.preprocess_fn(inputs)

                # Inference
                outputs = self.predict(inputs)

                # Postprocess
                if self.postprocess_fn:
                    outputs = self.postprocess_fn(outputs)

                latency_ms = (time.time() - start_time) * 1000

                return PredictionResponse(
                    outputs=outputs.tolist(), latency_ms=latency_ms
                )

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/metrics")
        async def metrics():
            return {
                "model": self.config.model_name,
                "device": self.config.device,
                "batching_enabled": self.batching,
            }

        # Run in thread
        import threading

        def run_server():
            uvicorn.run(self.app, host=host, port=port, log_level="info")

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True

        logger.info(f"FastAPI server started at http://{host}:{port}")

    def stop(self) -> None:
        """Stop FastAPI server."""
        self.is_running = False
        logger.info("FastAPI server stopped")


class FlaskServer(BaseServingBackend):
    """Flask-based lightweight serving backend.

    Simple and easy-to-deploy REST API for model serving.

    Example:
        >>> config = ModelConfig("my_model")
        >>> server = FlaskServer(config)
        >>> server.start(host="0.0.0.0", port=5000)
    """

    def __init__(
        self,
        config: ModelConfig,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
    ):
        """Initialize Flask server.

        Args:
            config: Model configuration
            preprocess_fn: Custom preprocessing function
            postprocess_fn: Custom postprocessing function
        """
        super().__init__(config)
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.app: Optional[Any] = None

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load model for serving.

        Args:
            model_path: Path to model file
        """
        path = Path(model_path)

        if path.suffix in [".pt", ".pth"]:
            self.model = torch.jit.load(str(model_path))
        else:
            self.model = torch.load(str(model_path), map_location=self.config.device)

        self.model.eval()

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run model inference.

        Args:
            inputs: Input tensor

        Returns:
            Model predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            return self.model(inputs)

    def start(self, host: str = "0.0.0.0", port: int = 5000, **kwargs) -> None:
        """Start Flask server.

        Args:
            host: Host address
            port: Port number
        """
        from flask import Flask, request, jsonify

        self.app = Flask(__name__)

        @self.app.route("/health", methods=["GET"])
        def health_check():
            return jsonify(
                {
                    "status": "healthy",
                    "model": self.config.model_name,
                    "version": self.config.version,
                }
            )

        @self.app.route("/predict", methods=["POST"])
        def predict():
            start_time = time.time()

            try:
                data = request.get_json()
                inputs = torch.tensor(data["inputs"], dtype=torch.float32)

                # Preprocess
                if self.preprocess_fn:
                    inputs = self.preprocess_fn(inputs)

                # Inference
                outputs = self.predict(inputs)

                # Postprocess
                if self.postprocess_fn:
                    outputs = self.postprocess_fn(outputs)

                latency_ms = (time.time() - start_time) * 1000

                return jsonify({"outputs": outputs.tolist(), "latency_ms": latency_ms})

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/metrics", methods=["GET"])
        def metrics():
            return jsonify(
                {"model": self.config.model_name, "device": self.config.device}
            )

        # Run in thread
        def run_server():
            self.app.run(host=host, port=port, threaded=True)

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True

        logger.info(f"Flask server started at http://{host}:{port}")

    def stop(self) -> None:
        """Stop Flask server."""
        self.is_running = False
        logger.info("Flask server stopped")


class gRPCServer(BaseServingBackend):
    """High-performance gRPC serving backend.

    gRPC provides efficient binary protocol for low-latency,
    high-throughput model serving.

    Example:
        >>> config = ModelConfig("my_model")
        >>> server = gRPCServer(config)
        >>> server.start(port=50051)
    """

    def __init__(
        self, config: ModelConfig, max_workers: int = 10, max_concurrent_rpcs: int = 100
    ):
        """Initialize gRPC server.

        Args:
            config: Model configuration
            max_workers: Maximum worker threads
            max_concurrent_rpcs: Maximum concurrent RPCs
        """
        super().__init__(config)
        self.max_workers = max_workers
        self.max_concurrent_rpcs = max_concurrent_rpcs
        self.server: Optional[Any] = None

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load model for serving.

        Args:
            model_path: Path to model file
        """
        path = Path(model_path)

        if path.suffix in [".pt", ".pth"]:
            self.model = torch.jit.load(str(model_path))
        else:
            self.model = torch.load(str(model_path), map_location=self.config.device)

        self.model.eval()

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run model inference.

        Args:
            inputs: Input tensor

        Returns:
            Model predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            return self.model(inputs)

    def start(self, port: int = 50051, **kwargs) -> None:
        """Start gRPC server.

        Args:
            port: Port number
        """
        try:
            import grpc
            from grpc import reflection
        except ImportError:
            raise ImportError("grpcio required. Install with: pip install grpcio")

        # Create gRPC server
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            maximum_concurrent_rpcs=self.max_concurrent_rpcs,
        )

        # Add servicer (simplified implementation)
        # In practice, you'd define a proper protobuf service

        self.server.add_insecure_port(f"[::]:{port}")
        self.server.start()
        self.is_running = True

        logger.info(f"gRPC server started on port {port}")

    def stop(self) -> None:
        """Stop gRPC server."""
        if self.server:
            self.server.stop(0)

        self.is_running = False
        logger.info("gRPC server stopped")


# =============================================================================
# Optimization for Serving
# =============================================================================


class ModelQuantizer:
    """Quantize models for efficient inference.

    Supports dynamic, static, and quantization-aware training (QAT)
    quantization modes to reduce model size and improve inference speed.

    Example:
        >>> quantizer = ModelQuantizer(mode="dynamic")
        >>> quantized_model = quantizer.quantize(model)
    """

    def __init__(
        self,
        mode: Union[str, QuantizationMode] = QuantizationMode.DYNAMIC,
        dtype: torch.dtype = torch.qint8,
        backend: str = "fbgemm",
    ):
        """Initialize model quantizer.

        Args:
            mode: Quantization mode ('dynamic', 'static', 'qat')
            dtype: Quantization data type
            backend: Quantization backend ('fbgemm', 'qnnpack')
        """
        if isinstance(mode, str):
            mode = QuantizationMode[mode.upper()]

        self.mode = mode
        self.dtype = dtype
        self.backend = backend

    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.utils.data.DataLoader] = None,
    ) -> nn.Module:
        """Quantize a model.

        Args:
            model: Model to quantize
            calibration_data: Calibration data for static quantization

        Returns:
            Quantized model
        """
        model.eval()

        if self.mode == QuantizationMode.DYNAMIC:
            # Dynamic quantization (easiest, runtime quantization)
            quantized_model = quantize_dynamic(
                model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=self.dtype
            )

        elif self.mode == QuantizationMode.STATIC:
            # Static quantization (requires calibration)
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")

            model.qconfig = torch.quantization.get_default_qconfig(self.backend)
            model_prepared = prepare(model, inplace=False)

            # Calibrate
            with torch.no_grad():
                for batch in calibration_data:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    model_prepared(inputs)

            quantized_model = convert(model_prepared, inplace=False)

        elif self.mode == QuantizationMode.QAT:
            # Quantization-aware training
            model.qconfig = torch.quantization.get_default_qat_qconfig(self.backend)
            model_prepared = prepare(model, inplace=False)
            quantized_model = convert(model_prepared, inplace=False)

        else:
            raise ValueError(f"Unknown quantization mode: {self.mode}")

        logger.info(f"Model quantized using {self.mode.name} mode")
        return quantized_model

    def benchmark(
        self, model: nn.Module, inputs: torch.Tensor, iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark quantized model performance.

        Args:
            model: Model to benchmark
            inputs: Input tensor
            iterations: Number of iterations

        Returns:
            Dictionary with latency and throughput metrics
        """
        model.eval()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(inputs)

        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                _ = model(inputs)

        elapsed = time.time() - start_time
        latency_ms = (elapsed / iterations) * 1000
        throughput_qps = iterations / elapsed

        return {
            "latency_ms": latency_ms,
            "throughput_qps": throughput_qps,
            "mode": self.mode.name,
        }


class ModelPruner:
    """Prune models to remove redundant weights and reduce size.

    Supports magnitude-based, structured, and unstructured pruning.

    Example:
        >>> pruner = ModelPruner(method="magnitude", amount=0.3)
        >>> pruned_model = pruner.prune(model)
    """

    def __init__(
        self, method: str = "magnitude", amount: float = 0.3, structured: bool = False
    ):
        """Initialize model pruner.

        Args:
            method: Pruning method ('magnitude', 'l1', 'random')
            amount: Fraction of weights to prune (0-1)
            structured: Whether to use structured pruning
        """
        self.method = method
        self.amount = amount
        self.structured = structured

    def prune(self, model: nn.Module) -> nn.Module:
        """Prune a model.

        Args:
            model: Model to prune

        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune
        except ImportError:
            raise ImportError("PyTorch pruning utilities not available")

        model.eval()

        # Select pruning method
        if self.method == "magnitude":
            pruning_method = prune.L1Unstructured
        elif self.method == "random":
            pruning_method = prune.RandomUnstructured
        elif self.method == "l1":
            pruning_method = prune.L1Unstructured
        else:
            raise ValueError(f"Unknown pruning method: {self.method}")

        # Apply pruning to all applicable layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                if self.structured:
                    # Structured pruning
                    prune.ln_structured(
                        module, name="weight", amount=self.amount, n=1, dim=0
                    )
                else:
                    # Unstructured pruning
                    prune_method = pruning_method(amount=self.amount)
                    prune_method.apply(module, "weight")

        # Make pruning permanent
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                prune.remove(module, "weight")

        logger.info(f"Model pruned with {self.method} method, amount={self.amount}")
        return model

    def get_sparsity(self, model: nn.Module) -> float:
        """Calculate model sparsity.

        Args:
            model: Model to analyze

        Returns:
            Fraction of zero weights
        """
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0


class BatchOptimizer:
    """Optimize batch processing for efficient inference.

    Implements dynamic batching and batch size optimization
    to maximize throughput while meeting latency constraints.

    Example:
        >>> optimizer = BatchOptimizer(max_batch_size=32, target_latency_ms=50)
        >>> optimizer.process_requests(request_queue, model)
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        target_latency_ms: float = 100.0,
        dynamic_batching: bool = True,
        timeout_ms: float = 10.0,
    ):
        """Initialize batch optimizer.

        Args:
            max_batch_size: Maximum batch size
            target_latency_ms: Target latency constraint
            dynamic_batching: Enable dynamic batch sizing
            timeout_ms: Timeout for forming a batch
        """
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        self.dynamic_batching = dynamic_batching
        self.timeout_ms = timeout_ms
        self.request_queue: queue.Queue = queue.Queue()
        self.results: Dict[str, Any] = {}

    def add_request(self, request_id: str, inputs: torch.Tensor) -> None:
        """Add a request to the batch queue.

        Args:
            request_id: Unique request identifier
            inputs: Input tensor
        """
        self.request_queue.put((request_id, inputs))

    def get_optimal_batch_size(
        self, model: nn.Module, input_shape: Tuple[int, ...]
    ) -> int:
        """Determine optimal batch size through benchmarking.

        Args:
            model: Model to benchmark
            input_shape: Shape of input tensors

        Returns:
            Optimal batch size
        """
        best_batch_size = 1
        best_throughput = 0.0

        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            if batch_size > self.max_batch_size:
                break

            # Create dummy batch
            shape = (batch_size,) + input_shape[1:]
            dummy_input = torch.randn(*shape)

            # Benchmark
            start_time = time.time()
            iterations = 10

            for _ in range(iterations):
                with torch.no_grad():
                    _ = model(dummy_input)

            elapsed = time.time() - start_time
            latency_ms = (elapsed / iterations) * 1000
            throughput_qps = (batch_size * iterations) / elapsed

            # Check latency constraint
            if latency_ms <= self.target_latency_ms:
                if throughput_qps > best_throughput:
                    best_throughput = throughput_qps
                    best_batch_size = batch_size

        logger.info(f"Optimal batch size: {best_batch_size}")
        return best_batch_size

    def process_batch(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Process a batch of requests.

        Args:
            model: Model for inference

        Returns:
            Dictionary mapping request IDs to results
        """
        batch = []
        request_ids = []

        # Collect requests
        start_time = time.time()
        while (
            len(batch) < self.max_batch_size
            and (time.time() - start_time) * 1000 < self.timeout_ms
        ):
            try:
                req_id, inputs = self.request_queue.get(timeout=0.001)
                batch.append(inputs)
                request_ids.append(req_id)
            except queue.Empty:
                if batch:
                    break

        if not batch:
            return {}

        # Stack into batch tensor
        batch_tensor = torch.stack(batch)

        # Run inference
        with torch.no_grad():
            outputs = model(batch_tensor)

        # Split results
        results = {}
        for i, req_id in enumerate(request_ids):
            results[req_id] = outputs[i]

        return results


class GraphOptimizer:
    """Optimize computation graph for inference.

    Applies graph-level optimizations like operator fusion,
    dead code elimination, and layout optimization.

    Example:
        >>> optimizer = GraphOptimizer()
        >>> optimized_model = optimizer.optimize(model)
    """

    def __init__(
        self,
        fuse_operations: bool = True,
        eliminate_dead_code: bool = True,
        optimize_layout: bool = True,
    ):
        """Initialize graph optimizer.

        Args:
            fuse_operations: Fuse compatible operations
            eliminate_dead_code: Remove unused operations
            optimize_layout: Optimize tensor layouts
        """
        self.fuse_operations = fuse_operations
        self.eliminate_dead_code = eliminate_dead_code
        self.optimize_layout = optimize_layout

    def optimize(self, model: nn.Module) -> nn.Module:
        """Optimize model computation graph.

        Args:
            model: Model to optimize

        Returns:
            Optimized model
        """
        model.eval()

        if self.fuse_operations:
            # Fuse batch norm and convolutions
            model = torch.quantization.fuse_modules(
                model,
                [["conv", "bn", "relu"]] if hasattr(model, "conv") else [],
                inplace=False,
            )

        if self.optimize_layout:
            # Set to channels_last memory format for better GPU performance
            model = model.to(memory_format=torch.channels_last)

        logger.info("Model graph optimized")
        return model

    def optimize_for_inference(self, model: nn.Module) -> torch.jit.ScriptModule:
        """Optimize model specifically for inference.

        Args:
            model: Model to optimize

        Returns:
            TorchScript optimized model
        """
        model.eval()

        # Convert to TorchScript
        example_input = torch.randn(1, 3, 224, 224)
        scripted = torch.jit.trace(model, example_input)

        # Optimize for inference
        optimized = torch.jit.optimize_for_inference(scripted)

        logger.info("Model optimized for inference")
        return optimized


# =============================================================================
# Request Handling
# =============================================================================


class RequestPreprocessor:
    """Preprocess incoming requests for model inference.

    Handles normalization, resizing, type conversion, and
    format standardization.

    Example:
        >>> preprocessor = RequestPreprocessor(
        ...     normalize=True,
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225]
        ... )
        >>> processed = preprocessor.process(raw_input)
    """

    def __init__(
        self,
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        target_shape: Optional[Tuple[int, ...]] = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize request preprocessor.

        Args:
            normalize: Whether to normalize inputs
            mean: Mean values for normalization
            std: Standard deviation values
            target_shape: Target tensor shape
            dtype: Target data type
            device: Target device
        """
        self.normalize = normalize
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None
        self.target_shape = target_shape
        self.dtype = dtype
        self.device = device

    def process(self, inputs: Any) -> torch.Tensor:
        """Process raw inputs into model-ready tensors.

        Args:
            inputs: Raw input data (numpy array, list, etc.)

        Returns:
            Preprocessed tensor
        """
        # Convert to tensor
        if isinstance(inputs, np.ndarray):
            tensor = torch.from_numpy(inputs)
        elif isinstance(inputs, list):
            tensor = torch.tensor(inputs)
        elif isinstance(inputs, torch.Tensor):
            tensor = inputs
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")

        # Ensure correct dtype and device
        tensor = tensor.to(dtype=self.dtype, device=self.device)

        # Reshape if needed
        if self.target_shape and tensor.shape != self.target_shape:
            tensor = self._reshape(tensor, self.target_shape)

        # Normalize
        if self.normalize and self.mean is not None and self.std is not None:
            mean = self.mean.to(tensor.device)
            std = self.std.to(tensor.device)
            tensor = (tensor - mean) / std

        return tensor

    def _reshape(
        self, tensor: torch.Tensor, target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Reshape tensor to target shape.

        Args:
            tensor: Input tensor
            target_shape: Target shape

        Returns:
            Reshaped tensor
        """
        # Handle batch dimension
        if tensor.dim() == len(target_shape) - 1:
            tensor = tensor.unsqueeze(0)

        # Use interpolation for spatial dimensions
        if len(target_shape) >= 4 and tensor.dim() >= 4:
            tensor = torch.nn.functional.interpolate(
                tensor, size=target_shape[2:], mode="bilinear", align_corners=False
            )

        return tensor


class RequestValidator:
    """Validate incoming requests before processing.

    Checks input format, shape, values, and constraints.

    Example:
        >>> validator = RequestValidator(
        ...     expected_shape=(1, 3, 224, 224),
        ...     value_range=(0, 1)
        ... )
        >>> is_valid, error = validator.validate(inputs)
    """

    def __init__(
        self,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[torch.dtype] = None,
        value_range: Optional[Tuple[float, float]] = None,
        allow_batch: bool = True,
        max_batch_size: int = 64,
    ):
        """Initialize request validator.

        Args:
            expected_shape: Expected input shape (without batch)
            expected_dtype: Expected data type
            value_range: Valid value range (min, max)
            allow_batch: Whether batch inputs are allowed
            max_batch_size: Maximum allowed batch size
        """
        self.expected_shape = expected_shape
        self.expected_dtype = expected_dtype
        self.value_range = value_range
        self.allow_batch = allow_batch
        self.max_batch_size = max_batch_size

    def validate(self, inputs: torch.Tensor) -> Tuple[bool, Optional[str]]:
        """Validate input tensor.

        Args:
            inputs: Input tensor to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check type
        if not isinstance(inputs, torch.Tensor):
            return False, f"Expected torch.Tensor, got {type(inputs)}"

        # Check dtype
        if self.expected_dtype and inputs.dtype != self.expected_dtype:
            return False, f"Expected dtype {self.expected_dtype}, got {inputs.dtype}"

        # Check shape
        if self.expected_shape:
            expected = self.expected_shape
            actual = inputs.shape

            if self.allow_batch:
                # Skip batch dimension
                expected = expected[1:] if len(expected) > 1 else expected
                actual = actual[1:] if len(actual) > 1 else actual

            if actual != expected:
                return False, f"Expected shape {expected}, got {actual}"

            # Check batch size
            if self.allow_batch and inputs.shape[0] > self.max_batch_size:
                return (
                    False,
                    f"Batch size {inputs.shape[0]} exceeds maximum {self.max_batch_size}",
                )

        # Check value range
        if self.value_range:
            min_val, max_val = self.value_range
            if inputs.min() < min_val or inputs.max() > max_val:
                return False, f"Values outside range [{min_val}, {max_val}]"

        # Check for NaN/Inf
        if torch.isnan(inputs).any():
            return False, "Input contains NaN values"

        if torch.isinf(inputs).any():
            return False, "Input contains Inf values"

        return True, None


class RequestBatcher:
    """Batch multiple requests for efficient processing.

    Groups individual requests into batches to maximize
    throughput and GPU utilization.

    Example:
        >>> batcher = RequestBatcher(max_batch_size=32, timeout_ms=10)
        >>> batcher.add_request(req_id_1, input_1)
        >>> batcher.add_request(req_id_2, input_2)
        >>> results = batcher.get_batch()
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        timeout_ms: float = 10.0,
        padding_value: float = 0.0,
    ):
        """Initialize request batcher.

        Args:
            max_batch_size: Maximum batch size
            timeout_ms: Maximum time to wait for batch
            padding_value: Value for padding variable-length inputs
        """
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.padding_value = padding_value
        self.request_queue: queue.Queue = queue.Queue()
        self.pending_requests: List[Tuple[str, torch.Tensor]] = []

    def add_request(self, request_id: str, inputs: torch.Tensor) -> None:
        """Add a request to the batch queue.

        Args:
            request_id: Unique request identifier
            inputs: Input tensor
        """
        self.request_queue.put((request_id, inputs))

    def get_batch(self) -> Tuple[List[str], Optional[torch.Tensor]]:
        """Get a batch of requests.

        Returns:
            Tuple of (request_ids, batch_tensor)
        """
        self.pending_requests = []
        start_time = time.time()

        while (
            len(self.pending_requests) < self.max_batch_size
            and (time.time() - start_time) * 1000 < self.timeout_ms
        ):
            try:
                req = self.request_queue.get(timeout=0.001)
                self.pending_requests.append(req)
            except queue.Empty:
                if self.pending_requests:
                    break

        if not self.pending_requests:
            return [], None

        request_ids = [req[0] for req in self.pending_requests]
        tensors = [req[1] for req in self.pending_requests]

        # Stack tensors (assume same shape for now)
        batch_tensor = torch.stack(tensors)

        return request_ids, batch_tensor

    def split_results(self, batch_outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split batched outputs back to individual results.

        Args:
            batch_outputs: Batched model outputs

        Returns:
            Dictionary mapping request IDs to outputs
        """
        results = {}
        for i, (req_id, _) in enumerate(self.pending_requests):
            results[req_id] = batch_outputs[i]

        return results


class ResponsePostprocessor:
    """Postprocess model outputs for response.

    Handles formatting, filtering, thresholding, and
    conversion to appropriate output formats.

    Example:
        >>> postprocessor = ResponsePostprocessor(
        ...     apply_softmax=True,
        ...     top_k=5
        ... )
        >>> response = postprocessor.process(model_output)
    """

    def __init__(
        self,
        apply_softmax: bool = False,
        apply_sigmoid: bool = False,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        output_format: str = "json",
    ):
        """Initialize response postprocessor.

        Args:
            apply_softmax: Apply softmax activation
            apply_sigmoid: Apply sigmoid activation
            top_k: Return top-k predictions
            threshold: Threshold for binary/multilabel outputs
            output_format: Output format ('json', 'numpy', 'list')
        """
        self.apply_softmax = apply_softmax
        self.apply_sigmoid = apply_sigmoid
        self.top_k = top_k
        self.threshold = threshold
        self.output_format = output_format

    def process(self, outputs: torch.Tensor) -> Any:
        """Process model outputs.

        Args:
            outputs: Raw model outputs

        Returns:
            Processed outputs in specified format
        """
        # Apply activation
        if self.apply_softmax:
            outputs = torch.softmax(outputs, dim=-1)
        elif self.apply_sigmoid:
            outputs = torch.sigmoid(outputs)

        # Apply threshold
        if self.threshold is not None:
            outputs = (outputs > self.threshold).float()

        # Get top-k
        if self.top_k is not None:
            values, indices = torch.topk(outputs, k=min(self.top_k, outputs.shape[-1]))
            outputs = {"values": values, "indices": indices}

        # Convert to output format
        if self.output_format == "json":
            return self._to_json(outputs)
        elif self.output_format == "numpy":
            return self._to_numpy(outputs)
        elif self.output_format == "list":
            return self._to_list(outputs)
        else:
            return outputs

    def _to_json(self, outputs: Any) -> Dict:
        """Convert outputs to JSON-serializable format."""
        if isinstance(outputs, dict):
            return {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.items()
            }
        elif isinstance(outputs, torch.Tensor):
            return {"outputs": outputs.tolist()}
        else:
            return {"outputs": outputs}

    def _to_numpy(self, outputs: Any) -> np.ndarray:
        """Convert outputs to numpy array."""
        if isinstance(outputs, torch.Tensor):
            return outputs.detach().cpu().numpy()
        elif isinstance(outputs, dict):
            return {k: v.detach().cpu().numpy() for k, v in outputs.items()}
        return np.array(outputs)

    def _to_list(self, outputs: Any) -> List:
        """Convert outputs to list."""
        if isinstance(outputs, torch.Tensor):
            return outputs.tolist()
        elif isinstance(outputs, dict):
            return {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.items()
            }
        return list(outputs)


# =============================================================================
# Monitoring
# =============================================================================


class LatencyMonitor:
    """Monitor inference latency metrics.

    Tracks latency statistics including mean, p50, p95, p99 percentiles.

    Example:
        >>> monitor = LatencyMonitor(window_size=1000)
        >>> with monitor.track():
        ...     outputs = model(inputs)
        >>> stats = monitor.get_statistics()
    """

    def __init__(self, window_size: int = 10000):
        """Initialize latency monitor.

        Args:
            window_size: Number of recent measurements to track
        """
        self.window_size = window_size
        self.latencies: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            self.latencies.append(latency_ms)

    def track(self):
        """Context manager for tracking latency.

        Returns:
            LatencyTracker context manager
        """
        return LatencyTracker(self)

    def get_statistics(self) -> Dict[str, float]:
        """Get latency statistics.

        Returns:
            Dictionary with latency statistics
        """
        with self._lock:
            if not self.latencies:
                return {
                    "mean_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "count": 0,
                }

            latencies_array = np.array(self.latencies)

            return {
                "mean_ms": float(np.mean(latencies_array)),
                "p50_ms": float(np.percentile(latencies_array, 50)),
                "p95_ms": float(np.percentile(latencies_array, 95)),
                "p99_ms": float(np.percentile(latencies_array, 99)),
                "min_ms": float(np.min(latencies_array)),
                "max_ms": float(np.max(latencies_array)),
                "count": len(self.latencies),
            }


class LatencyTracker:
    """Context manager for tracking latency."""

    def __init__(self, monitor: LatencyMonitor):
        self.monitor = monitor
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed_ms = (time.time() - self.start_time) * 1000
            self.monitor.record(elapsed_ms)


class ThroughputMonitor:
    """Monitor request throughput.

    Tracks requests per second (QPS) over time windows.

    Example:
        >>> monitor = ThroughputMonitor(window_seconds=60)
        >>> monitor.record_request()
        >>> qps = monitor.get_qps()
    """

    def __init__(self, window_seconds: float = 60.0):
        """Initialize throughput monitor.

        Args:
            window_seconds: Time window for QPS calculation
        """
        self.window_seconds = window_seconds
        self.request_times: deque = deque()
        self._lock = threading.Lock()

    def record_request(self) -> None:
        """Record a completed request."""
        with self._lock:
            now = time.time()
            self.request_times.append(now)
            self._cleanup_old_requests(now)

    def _cleanup_old_requests(self, current_time: float) -> None:
        """Remove requests outside the time window."""
        cutoff = current_time - self.window_seconds
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()

    def get_qps(self) -> float:
        """Get current queries per second.

        Returns:
            Requests per second
        """
        with self._lock:
            now = time.time()
            self._cleanup_old_requests(now)

            if len(self.request_times) < 2:
                return 0.0

            time_span = self.request_times[-1] - self.request_times[0]
            if time_span <= 0:
                return 0.0

            return len(self.request_times) / time_span

    def get_statistics(self) -> Dict[str, float]:
        """Get throughput statistics.

        Returns:
            Dictionary with QPS and request count
        """
        with self._lock:
            now = time.time()
            self._cleanup_old_requests(now)

            return {
                "qps": self.get_qps(),
                "total_requests": len(self.request_times),
                "window_seconds": self.window_seconds,
            }


class MemoryMonitor:
    """Monitor memory usage during inference.

    Tracks GPU and CPU memory consumption.

    Example:
        >>> monitor = MemoryMonitor()
        >>> usage = monitor.get_memory_usage()
    """

    def __init__(self, track_gpu: bool = True, track_cpu: bool = True):
        """Initialize memory monitor.

        Args:
            track_gpu: Whether to track GPU memory
            track_cpu: Whether to track CPU memory
        """
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.track_cpu = track_cpu

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage.

        Returns:
            Dictionary with memory statistics in MB
        """
        usage = {}

        if self.track_cpu:
            import psutil

            process = psutil.Process()
            usage["cpu_memory_mb"] = process.memory_info().rss / (1024 * 1024)

        if self.track_gpu:
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                usage[f"gpu_{i}_allocated_mb"] = allocated
                usage[f"gpu_{i}_reserved_mb"] = reserved

        return usage

    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage since last reset.

        Returns:
            Dictionary with peak memory in MB
        """
        usage = {}

        if self.track_gpu:
            for i in range(torch.cuda.device_count()):
                peak = torch.cuda.max_memory_allocated(i) / (1024 * 1024)
                usage[f"gpu_{i}_peak_mb"] = peak

        return usage

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.track_gpu:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)


class ErrorTracker:
    """Track errors and exceptions during serving.

    Logs and categorizes errors for monitoring and debugging.

    Example:
        >>> tracker = ErrorTracker()
        >>> try:
        ...     result = model(inputs)
        ... except Exception as e:
        ...     tracker.record_error(e, context="inference")
    """

    def __init__(self, max_errors: int = 1000):
        """Initialize error tracker.

        Args:
            max_errors: Maximum number of errors to store
        """
        self.max_errors = max_errors
        self.errors: deque = deque(maxlen=max_errors)
        self.error_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def record_error(
        self, error: Exception, context: str = "", request_id: Optional[str] = None
    ) -> None:
        """Record an error.

        Args:
            error: Exception that occurred
            context: Context where error occurred
            request_id: Associated request ID
        """
        with self._lock:
            error_info = {
                "timestamp": datetime.now(),
                "type": type(error).__name__,
                "message": str(error),
                "context": context,
                "request_id": request_id,
            }

            self.errors.append(error_info)

            # Count by type
            error_type = type(error).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

            # Log error
            logger.error(
                f"Error in {context}: {error}", extra={"request_id": request_id}
            )

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics.

        Returns:
            Dictionary with error counts and recent errors
        """
        with self._lock:
            return {
                "total_errors": len(self.errors),
                "error_counts": self.error_counts.copy(),
                "recent_errors": list(self.errors)[-10:],
            }

    def clear(self) -> None:
        """Clear all recorded errors."""
        with self._lock:
            self.errors.clear()
            self.error_counts.clear()


# =============================================================================
# Scaling
# =============================================================================


class LoadBalancer:
    """Distribute requests across multiple model instances.

    Implements various load balancing strategies for scaling.

    Example:
        >>> lb = LoadBalancer(strategy="round_robin")
        >>> lb.add_instance("instance_1", model_1)
        >>> lb.add_instance("instance_2", model_2)
        >>> result = lb.predict(inputs)
    """

    def __init__(self, strategy: str = "round_robin"):
        """Initialize load balancer.

        Args:
            strategy: Load balancing strategy ('round_robin', 'least_loaded', 'random')
        """
        self.strategy = strategy
        self.instances: Dict[str, nn.Module] = {}
        self.instance_loads: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def add_instance(self, instance_id: str, model: nn.Module) -> None:
        """Add a model instance to the pool.

        Args:
            instance_id: Unique instance identifier
            model: Model instance
        """
        with self._lock:
            self.instances[instance_id] = model
            self.instance_loads[instance_id] = 0

    def remove_instance(self, instance_id: str) -> None:
        """Remove a model instance.

        Args:
            instance_id: Instance to remove
        """
        with self._lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                del self.instance_loads[instance_id]

    def _select_instance(self) -> str:
        """Select an instance using the configured strategy.

        Returns:
            Selected instance ID
        """
        with self._lock:
            if not self.instances:
                raise RuntimeError("No instances available")

            instance_ids = list(self.instances.keys())

            if self.strategy == "round_robin":
                idx = self._counter % len(instance_ids)
                self._counter += 1
                return instance_ids[idx]

            elif self.strategy == "least_loaded":
                return min(self.instance_loads, key=self.instance_loads.get)

            elif self.strategy == "random":
                import random

                return random.choice(instance_ids)

            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Route prediction to selected instance.

        Args:
            inputs: Input tensor

        Returns:
            Model predictions
        """
        instance_id = self._select_instance()

        with self._lock:
            self.instance_loads[instance_id] += 1
            model = self.instances[instance_id]

        try:
            with torch.no_grad():
                outputs = model(inputs)
            return outputs
        finally:
            with self._lock:
                self.instance_loads[instance_id] -= 1


class AutoScaler:
    """Automatically scale model instances based on load.

    Scales up or down based on metrics like QPS and latency.

    Example:
        >>> scaler = AutoScaler(
        ...     min_instances=1,
        ...     max_instances=10,
        ...     target_qps=100
        ... )
        >>> scaler.start_monitoring()
    """

    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        target_qps: float = 100.0,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_seconds: float = 60.0,
    ):
        """Initialize auto-scaler.

        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            target_qps: Target queries per second per instance
            scale_up_threshold: QPS ratio threshold to scale up
            scale_down_threshold: QPS ratio threshold to scale down
            cooldown_seconds: Minimum time between scaling actions
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_qps = target_qps
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds

        self.current_instances = min_instances
        self.last_scale_time = time.time()
        self.throughput_monitor = ThroughputMonitor()

    def evaluate_scaling(self) -> Optional[str]:
        """Evaluate whether to scale up, down, or stay.

        Returns:
            'up', 'down', or None
        """
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_scale_time < self.cooldown_seconds:
            return None

        # Calculate load ratio
        current_qps = self.throughput_monitor.get_qps()
        capacity = self.current_instances * self.target_qps
        load_ratio = current_qps / capacity if capacity > 0 else 0

        if (
            load_ratio > self.scale_up_threshold
            and self.current_instances < self.max_instances
        ):
            return "up"
        elif (
            load_ratio < self.scale_down_threshold
            and self.current_instances > self.min_instances
        ):
            return "down"

        return None

    def scale_up(self) -> None:
        """Scale up by adding an instance."""
        if self.current_instances < self.max_instances:
            self.current_instances += 1
            self.last_scale_time = time.time()
            logger.info(f"Scaled up to {self.current_instances} instances")

    def scale_down(self) -> None:
        """Scale down by removing an instance."""
        if self.current_instances > self.min_instances:
            self.current_instances -= 1
            self.last_scale_time = time.time()
            logger.info(f"Scaled down to {self.current_instances} instances")


class ModelSharding:
    """Shard large models across multiple devices.

    Distributes model layers across GPUs or machines.

    Example:
        >>> sharder = ModelSharding(devices=[0, 1, 2, 3])
        >>> sharded_model = sharder.shard(model)
    """

    def __init__(self, devices: List[Union[int, str]]):
        """Initialize model sharding.

        Args:
            devices: List of device IDs or names
        """
        self.devices = devices

    def shard(self, model: nn.Module) -> nn.Module:
        """Shard model across devices.

        Args:
            model: Model to shard

        Returns:
            Sharded model
        """
        if len(self.devices) == 1:
            return model.to(self.devices[0])

        # Simple layer-wise sharding
        layers = list(model.children())
        layers_per_device = len(layers) // len(self.devices)

        for i, device in enumerate(self.devices):
            start_idx = i * layers_per_device
            end_idx = (
                (i + 1) * layers_per_device
                if i < len(self.devices) - 1
                else len(layers)
            )

            for layer in layers[start_idx:end_idx]:
                layer.to(device)

        logger.info(f"Model sharded across {len(self.devices)} devices")
        return model


class PipelineParallel:
    """Implement pipeline parallelism for model inference.

    Splits model into stages processed on different devices.

    Example:
        >>> pipeline = PipelineParallel(num_stages=4)
        >>> pipeline_model = pipeline.create_pipeline(model)
    """

    def __init__(
        self, num_stages: int = 2, devices: Optional[List[Union[int, str]]] = None
    ):
        """Initialize pipeline parallelism.

        Args:
            num_stages: Number of pipeline stages
            devices: Devices for each stage
        """
        self.num_stages = num_stages
        self.devices = devices or list(range(num_stages))
        self.stages: List[nn.Module] = []

    def create_pipeline(self, model: nn.Module) -> "PipelineModel":
        """Create a pipelined version of the model.

        Args:
            model: Model to parallelize

        Returns:
            PipelineModel wrapper
        """
        # Split model into stages
        all_layers = list(model.children())
        layers_per_stage = len(all_layers) // self.num_stages

        for i in range(self.num_stages):
            start_idx = i * layers_per_stage
            end_idx = (
                (i + 1) * layers_per_stage
                if i < self.num_stages - 1
                else len(all_layers)
            )

            stage_layers = all_layers[start_idx:end_idx]
            stage = nn.Sequential(*stage_layers).to(self.devices[i])
            self.stages.append(stage)

        return PipelineModel(self.stages, self.devices)


class PipelineModel(nn.Module):
    """Wrapper for pipelined model execution."""

    def __init__(self, stages: List[nn.Module], devices: List[Union[int, str]]):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.devices = devices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through pipeline.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        for stage, device in zip(self.stages, self.devices):
            x = x.to(device)
            x = stage(x)

        return x


# =============================================================================
# Utilities
# =============================================================================


def warmup_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    iterations: int = 10,
    device: str = "cuda",
) -> None:
    """Pre-warm model for consistent latency.

    Runs warmup iterations to initialize caches and
    achieve stable inference performance.

    Args:
        model: Model to warmup
        input_shape: Shape of dummy inputs
        iterations: Number of warmup iterations
        device: Device to run warmup on

    Example:
        >>> model = MyModel().cuda()
        >>> warmup_model(model, input_shape=(1, 3, 224, 224), iterations=20)
    """
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)

    logger.info(f"Warming up model with {iterations} iterations...")

    with torch.no_grad():
        for i in range(iterations):
            _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()

    logger.info("Model warmup complete")


def benchmark_serving(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
    iterations: int = 100,
    warmup: int = 10,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """Benchmark model serving performance across batch sizes.

    Tests latency and throughput for different configurations.

    Args:
        model: Model to benchmark
        input_shape: Shape of inputs (without batch dimension)
        batch_sizes: List of batch sizes to test
        iterations: Number of iterations per batch size
        warmup: Warmup iterations
        device: Device to benchmark on

    Returns:
        Dictionary with benchmark results for each batch size

    Example:
        >>> model = MyModel().cuda()
        >>> results = benchmark_serving(
        ...     model,
        ...     batch_sizes=[1, 4, 16],
        ...     iterations=100
        ... )
        >>> print(results[4]['throughput_qps'])
    """
    model.eval()
    model = model.to(device)

    results = {}

    for batch_size in batch_sizes:
        shape = (batch_size,) + input_shape
        dummy_input = torch.randn(*shape).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        latencies = []

        for _ in range(iterations):
            start_time = time.time()

            with torch.no_grad():
                _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()

            latencies.append((time.time() - start_time) * 1000)

        # Calculate statistics
        latencies_array = np.array(latencies)
        mean_latency = float(np.mean(latencies_array))
        p99_latency = float(np.percentile(latencies_array, 99))
        throughput_qps = (batch_size * iterations) / (sum(latencies) / 1000)

        results[batch_size] = {
            "mean_latency_ms": mean_latency,
            "p99_latency_ms": p99_latency,
            "throughput_qps": throughput_qps,
            "iterations": iterations,
        }

        logger.info(
            f"Batch {batch_size}: "
            f"latency={mean_latency:.2f}ms, "
            f"throughput={throughput_qps:.2f} QPS"
        )

    return results


def health_check(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    timeout_ms: float = 5000.0,
    device: str = "cuda",
) -> HealthStatus:
    """Verify model health and availability.

    Performs a test inference to check model functionality
    and responsiveness.

    Args:
        model: Model to check
        input_shape: Shape of test input
        timeout_ms: Maximum acceptable latency
        device: Device to run check on

    Returns:
        HealthStatus with check results

    Example:
        >>> status = health_check(model)
        >>> if status.is_healthy:
        ...     print(f"Model healthy, latency: {status.latency_ms:.2f}ms")
    """
    start_time = time.time()
    check_time = datetime.now()

    try:
        model.eval()
        dummy_input = torch.randn(*input_shape).to(device)

        with torch.no_grad():
            _ = model(dummy_input)

        if device == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.time() - start_time) * 1000

        if latency_ms > timeout_ms:
            return HealthStatus(
                is_healthy=False,
                last_check=check_time,
                error_message=f"Latency {latency_ms:.2f}ms exceeds timeout {timeout_ms}ms",
                latency_ms=latency_ms,
            )

        return HealthStatus(
            is_healthy=True, last_check=check_time, latency_ms=latency_ms
        )

    except Exception as e:
        return HealthStatus(
            is_healthy=False,
            last_check=check_time,
            error_message=str(e),
            latency_ms=(time.time() - start_time) * 1000,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ExportFormat",
    "ServingBackend",
    "QuantizationMode",
    # Data classes
    "ModelConfig",
    "InferenceMetrics",
    "HealthStatus",
    # Exporters
    "BaseExporter",
    "TorchScriptExporter",
    "ONNXExporter",
    "TensorRTExporter",
    "CoreMLExporter",
    "OpenVINOExporter",
    # Serving Backends
    "BaseServingBackend",
    "TorchServeWrapper",
    "TritonInferenceServer",
    "FastAPIServer",
    "FlaskServer",
    "gRPCServer",
    # Optimization
    "ModelQuantizer",
    "ModelPruner",
    "BatchOptimizer",
    "GraphOptimizer",
    # Request Handling
    "RequestPreprocessor",
    "RequestValidator",
    "RequestBatcher",
    "ResponsePostprocessor",
    # Monitoring
    "LatencyMonitor",
    "ThroughputMonitor",
    "MemoryMonitor",
    "ErrorTracker",
    # Scaling
    "LoadBalancer",
    "AutoScaler",
    "ModelSharding",
    "PipelineParallel",
    "PipelineModel",
    # Utilities
    "warmup_model",
    "benchmark_serving",
    "health_check",
]
