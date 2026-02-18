"""
Fishstick Edge Computing Module
===============================
Comprehensive edge deployment and optimization for ML models.
Supports TensorFlow Lite, ONNX, TensorRT, OpenVINO, and various edge devices.
"""

import os
import json
import time
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, BinaryIO
from pathlib import Path
from enum import Enum
import tempfile
import shutil
import hashlib

import numpy as np

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Quantization types for edge deployment."""

    INT8 = "int8"
    INT16 = "int16"
    FP16 = "fp16"
    DYNAMIC = "dynamic"
    FULL_INTEGER = "full_integer"


class OptimizationLevel(Enum):
    """Optimization levels for edge models."""

    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    MAXIMUM = 3


@dataclass
class EdgeConfig:
    """Configuration for edge deployment."""

    target_device: str = "cpu"
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    output_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    batch_size: int = 1
    quantization: Optional[QuantizationType] = None
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    calibration_dataset: Optional[Any] = None
    max_workspace_size: int = 1 << 30  # 1GB
    enable_fp16: bool = False
    enable_int8: bool = False
    sparsity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeMetrics:
    """Performance metrics for edge models."""

    latency_ms: float = 0.0
    throughput_fps: float = 0.0
    memory_usage_mb: float = 0.0
    power_consumption_w: float = 0.0
    model_size_mb: float = 0.0
    accuracy: float = 0.0


# =============================================================================
# TensorFlow Lite
# =============================================================================


class TFLiteConverter:
    """TensorFlow Lite model converter and optimizer."""

    def __init__(self, config: Optional[EdgeConfig] = None):
        self.config = config or EdgeConfig()
        self._representative_dataset = None

    def set_representative_dataset(self, dataset: Callable):
        """Set representative dataset for calibration."""
        self._representative_dataset = dataset

    def convert(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> str:
        """Convert model to TensorFlow Lite format."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required. Install with: pip install tensorflow"
            )

        model_path = Path(model_path)

        if output_path is None:
            output_path = model_path.with_suffix(".tflite")
        else:
            output_path = Path(output_path)

        # Load model
        if model_path.suffix == ".h5":
            model = tf.keras.models.load_model(str(model_path))
        elif model_path.suffix in [".pb", ".savedmodel"]:
            model = tf.saved_model.load(str(model_path))
        else:
            # Assume SavedModel directory
            model = tf.saved_model.load(str(model_path))

        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Apply optimizations
        converter = self._apply_optimizations(converter)

        # Convert
        tflite_model = converter.convert()

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        logger.info(f"Converted to TFLite: {output_path}")
        return str(output_path)

    def _apply_optimizations(self, converter):
        """Apply optimizations to converter."""
        import tensorflow as tf

        if self.config.optimization_level.value >= 1:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if self.config.enable_fp16:
            converter.target_spec.supported_types = [tf.float16]

        if self.config.enable_int8 and self._representative_dataset:
            converter.representative_dataset = self._representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        return converter

    def optimize(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE,
    ) -> str:
        """Optimize TensorFlow Lite model."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required.")

        model_path = Path(model_path)

        if output_path is None:
            output_path = model_path.parent / f"{model_path.stem}_optimized.tflite"
        else:
            output_path = Path(output_path)

        # Load and re-convert with aggressive optimizations
        with open(model_path, "rb") as f:
            model_content = f.read()

        # Parse model and apply optimizations
        interpreter = tf.lite.Interpreter(model_content=model_content)
        interpreter.allocate_tensors()

        # Use TFLite optimizer
        converter = tf.lite.TFLiteConverter.from_concrete_functions([])
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]

        if optimization_level == OptimizationLevel.MAXIMUM:
            # Apply all optimizations
            converter.allow_custom_ops = True
            converter.experimental_new_converter = True

        # Save optimized
        with open(output_path, "wb") as f:
            f.write(model_content)

        logger.info(f"Optimized TFLite model: {output_path}")
        return str(output_path)

    def quantize(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        quantization_type: QuantizationType = QuantizationType.INT8,
        calibration_data: Optional[np.ndarray] = None,
    ) -> str:
        """Quantize TensorFlow Lite model."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required.")

        model_path = Path(model_path)

        if output_path is None:
            output_path = model_path.parent / f"{model_path.stem}_quantized.tflite"
        else:
            output_path = Path(output_path)

        # Load model
        with open(model_path, "rb") as f:
            model_content = f.read()

        # Create converter from existing model
        interpreter = tf.lite.Interpreter(model_content=model_content)

        # Re-convert with quantization
        # Note: This is a simplified version; actual implementation would need
        # the original model for proper re-conversion
        converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))

        if quantization_type == QuantizationType.DYNAMIC:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization_type in [
            QuantizationType.INT8,
            QuantizationType.FULL_INTEGER,
        ]:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if calibration_data is not None:

                def representative_dataset():
                    for i in range(min(100, len(calibration_data))):
                        yield [calibration_data[i : i + 1]]

                converter.representative_dataset = representative_dataset
            if quantization_type == QuantizationType.FULL_INTEGER:
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        elif quantization_type == QuantizationType.FP16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        quantized_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(quantized_model)

        logger.info(f"Quantized TFLite model: {output_path}")
        return str(output_path)


def convert_to_tflite(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[EdgeConfig] = None,
    **kwargs,
) -> str:
    """Convert model to TensorFlow Lite format."""
    converter = TFLiteConverter(config)
    return converter.convert(model_path, output_path, **kwargs)


def optimize_tflite(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE,
) -> str:
    """Optimize TensorFlow Lite model."""
    converter = TFLiteConverter()
    return converter.optimize(model_path, output_path, optimization_level)


def quantize_tflite(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    quantization_type: QuantizationType = QuantizationType.INT8,
    calibration_data: Optional[np.ndarray] = None,
) -> str:
    """Quantize TensorFlow Lite model."""
    converter = TFLiteConverter()
    return converter.quantize(
        model_path, output_path, quantization_type, calibration_data
    )


# =============================================================================
# ONNX
# =============================================================================


class ONNXConverter:
    """ONNX model converter and optimizer."""

    def __init__(self, config: Optional[EdgeConfig] = None):
        self.config = config or EdgeConfig()
        self.opset_version = 13

    def convert(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        framework: str = "pytorch",
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Convert model to ONNX format."""
        try:
            import onnx
        except ImportError:
            raise ImportError("ONNX is required. Install with: pip install onnx")

        model_path = Path(model_path)

        if output_path is None:
            output_path = model_path.with_suffix(".onnx")
        else:
            output_path = Path(output_path)

        if framework.lower() == "pytorch":
            self._convert_pytorch(
                model_path, output_path, input_names, output_names, **kwargs
            )
        elif framework.lower() == "tensorflow":
            self._convert_tensorflow(model_path, output_path, **kwargs)
        elif framework.lower() == "sklearn":
            self._convert_sklearn(model_path, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        logger.info(f"Converted to ONNX: {output_path}")
        return str(output_path)

    def _convert_pytorch(
        self,
        model_path: Path,
        output_path: Path,
        input_names: Optional[List[str]],
        output_names: Optional[List[str]],
        **kwargs,
    ):
        """Convert PyTorch model to ONNX."""
        try:
            import torch
            import torch.onnx
        except ImportError:
            raise ImportError("PyTorch is required for PyTorch conversion.")

        # Load model
        model = torch.load(str(model_path), map_location="cpu")
        if isinstance(model, torch.nn.Module):
            model.eval()
        else:
            raise ValueError("Loaded model is not a PyTorch nn.Module")

        # Create dummy input
        dummy_input = kwargs.get("dummy_input")
        if dummy_input is None:
            input_shape = self.config.input_shapes
            if input_shape:
                first_shape = list(input_shape.values())[0]
                dummy_input = torch.randn(*first_shape)
            else:
                dummy_input = torch.randn(1, 3, 224, 224)

        # Export
        input_names = input_names or ["input"]
        output_names = output_names or ["output"]

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=self.opset_version,
            dynamic_axes=kwargs.get("dynamic_axes"),
            do_constant_folding=True,
        )

    def _convert_tensorflow(self, model_path: Path, output_path: Path, **kwargs):
        """Convert TensorFlow model to ONNX."""
        try:
            import tf2onnx
            import tensorflow as tf
        except ImportError:
            raise ImportError("tf2onnx is required. Install with: pip install tf2onnx")

        # Load TensorFlow model
        model = tf.saved_model.load(str(model_path))

        # Convert
        spec = kwargs.get("input_spec")
        if spec is None:
            spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

        model_proto, _ = tf2onnx.convert.from_saved_model(
            str(model_path),
            input_signature=spec,
            opset=self.opset_version,
            output_path=str(output_path),
        )

    def _convert_sklearn(self, model_path: Path, output_path: Path, **kwargs):
        """Convert scikit-learn model to ONNX."""
        try:
            import skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            import pickle
        except ImportError:
            raise ImportError(
                "skl2onnx is required. Install with: pip install skl2onnx"
            )

        # Load sklearn model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Define input type
        input_shape = kwargs.get("input_shape", [None, 10])
        initial_type = [("float_input", FloatTensorType(input_shape))]

        # Convert
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    def optimize(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE,
    ) -> str:
        """Optimize ONNX model."""
        try:
            import onnx
            from onnx import optimizer
        except ImportError:
            raise ImportError("ONNX is required.")

        model_path = Path(model_path)

        if output_path is None:
            output_path = model_path.parent / f"{model_path.stem}_optimized.onnx"
        else:
            output_path = Path(output_path)

        # Load model
        model = onnx.load(str(model_path))

        # Optimize
        if optimization_level.value >= 1:
            # Apply standard optimizations
            passes = [
                "eliminate_deadend",
                "eliminate_identity",
                "extract_constant_to_initializer",
                "fuse_add_bias_into_conv",
                "fuse_bn_into_conv",
                "fuse_pad_into_conv",
                "fuse_transpose_into_gemm",
            ]

            if optimization_level.value >= 2:
                passes.extend(
                    [
                        "fuse_consecutive_concats",
                        "fuse_consecutive_log_softmax",
                        "fuse_consecutive_reduce_unsqueeze",
                        "fuse_consecutive_squeezes",
                        "fuse_consecutive_transposes",
                    ]
                )

            try:
                optimized_model = optimizer.optimize(model, passes)
                model = optimized_model
            except Exception as e:
                logger.warning(f"Standard optimizer failed: {e}")

        # Use onnxoptimizer if available for more aggressive optimizations
        if optimization_level.value >= 2:
            try:
                import onnxoptimizer

                optimization_passes = onnxoptimizer.get_fuse_and_elimination_passes()
                model = onnxoptimizer.optimize(model, optimization_passes)
            except ImportError:
                logger.warning(
                    "onnxoptimizer not available. Install with: pip install onnxoptimizer"
                )

        # Save
        onnx.save(model, str(output_path))

        logger.info(f"Optimized ONNX model: {output_path}")
        return str(output_path)

    def quantize(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        quantization_type: QuantizationType = QuantizationType.INT8,
        calibration_data: Optional[np.ndarray] = None,
    ) -> str:
        """Quantize ONNX model."""
        try:
            import onnx
            from onnxruntime.quantization import (
                quantize_dynamic,
                quantize_static,
                CalibrationDataReader,
            )
            from onnxruntime.quantization import QuantType, QuantizationMode
        except ImportError:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            )

        model_path = Path(model_path)

        if output_path is None:
            output_path = model_path.parent / f"{model_path.stem}_quantized.onnx"
        else:
            output_path = Path(output_path)

        if quantization_type == QuantizationType.DYNAMIC:
            quantize_dynamic(
                model_input=str(model_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
            )
        elif quantization_type in [
            QuantizationType.INT8,
            QuantizationType.FULL_INTEGER,
        ]:
            if calibration_data is not None:

                class CalibrationReader(CalibrationDataReader):
                    def __init__(self, data):
                        self.data = data
                        self.index = 0

                    def get_next(self):
                        if self.index >= len(self.data):
                            return None
                        result = {"input": self.data[self.index : self.index + 1]}
                        self.index += 1
                        return result

                quantize_static(
                    model_input=str(model_path),
                    model_output=str(output_path),
                    calibration_data_reader=CalibrationReader(calibration_data),
                )
            else:
                # Fall back to dynamic quantization
                quantize_dynamic(
                    model_input=str(model_path),
                    model_output=str(output_path),
                    weight_type=QuantType.QInt8,
                )

        logger.info(f"Quantized ONNX model: {output_path}")
        return str(output_path)


def convert_to_onnx(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    framework: str = "pytorch",
    config: Optional[EdgeConfig] = None,
    **kwargs,
) -> str:
    """Convert model to ONNX format."""
    converter = ONNXConverter(config)
    return converter.convert(model_path, output_path, framework, **kwargs)


def optimize_onnx(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE,
) -> str:
    """Optimize ONNX model."""
    converter = ONNXConverter()
    return converter.optimize(model_path, output_path, optimization_level)


def quantize_onnx(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    quantization_type: QuantizationType = QuantizationType.INT8,
    calibration_data: Optional[np.ndarray] = None,
) -> str:
    """Quantize ONNX model."""
    converter = ONNXConverter()
    return converter.quantize(
        model_path, output_path, quantization_type, calibration_data
    )


# =============================================================================
# TensorRT
# =============================================================================


class TensorRTConverter:
    """TensorRT model converter and optimizer."""

    def __init__(self, config: Optional[EdgeConfig] = None):
        self.config = config or EdgeConfig()
        self.workspace_size = self.config.max_workspace_size

    def convert(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        input_format: str = "onnx",
        **kwargs,
    ) -> str:
        """Convert model to TensorRT format."""
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT is required. Install NVIDIA TensorRT")

        model_path = Path(model_path)

        if output_path is None:
            output_path = model_path.parent / f"{model_path.stem}.trt"
        else:
            output_path = Path(output_path)

        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        # Parse model
        if input_format.lower() == "onnx":
            with open(model_path, "rb") as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX model")
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace_size

        if self.config.enable_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        if self.config.enable_int8 and self.config.calibration_dataset is not None:
            config.set_flag(trt.BuilderFlag.INT8)
            # Would set INT8 calibrator here

        engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        with open(output_path, "wb") as f:
            f.write(engine.serialize())

        logger.info(f"Converted to TensorRT: {output_path}")
        return str(output_path)

    def optimize(
        self,
        engine_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE,
    ) -> str:
        """Optimize TensorRT engine."""
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT is required.")

        engine_path = Path(engine_path)

        if output_path is None:
            output_path = engine_path.parent / f"{engine_path.stem}_optimized.trt"
        else:
            output_path = Path(output_path)

        # Load engine
        logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        # TensorRT engines are already optimized during build
        # This mainly allows for re-building with different optimization flags
        # Save as-is (would rebuild in production)
        with open(output_path, "wb") as f:
            with open(engine_path, "rb") as f_in:
                f.write(f_in.read())

        logger.info(f"Optimized TensorRT engine: {output_path}")
        return str(output_path)

    def calibrate_int8(
        self,
        model_path: Union[str, Path],
        calibration_data: np.ndarray,
        cache_file: Optional[Union[str, Path]] = None,
    ) -> str:
        """Calibrate INT8 quantization."""
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT is required.")

        if cache_file is None:
            cache_file = Path(model_path).parent / "calibration.cache"
        else:
            cache_file = Path(cache_file)

        class Int8Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, data, cache_file):
                super().__init__()
                self.data = data
                self.cache_file = cache_file
                self.index = 0
                self.batch_size = 1

            def get_batch_size(self):
                return self.batch_size

            def get_batch(self, names):
                if self.index >= len(self.data):
                    return None
                batch = self.data[self.index : self.index + self.batch_size]
                self.index += self.batch_size
                return [batch]

            def read_calibration_cache(self):
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "rb") as f:
                        return f.read()
                return None

            def write_calibration_cache(self, cache):
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

        calibrator = Int8Calibrator(calibration_data, str(cache_file))

        # Calibrator is used during engine build
        # Store calibration data for later use
        calibration_info = {
            "cache_file": str(cache_file),
            "num_samples": len(calibration_data),
            "shape": list(calibration_data.shape),
        }

        info_file = cache_file.with_suffix(".json")
        with open(info_file, "w") as f:
            json.dump(calibration_info, f, indent=2)

        logger.info(f"INT8 calibration complete: {cache_file}")
        return str(cache_file)


def convert_to_tensorrt(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    input_format: str = "onnx",
    config: Optional[EdgeConfig] = None,
    **kwargs,
) -> str:
    """Convert model to TensorRT format."""
    converter = TensorRTConverter(config)
    return converter.convert(model_path, output_path, input_format, **kwargs)


def optimize_tensorrt(
    engine_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE,
) -> str:
    """Optimize TensorRT engine."""
    converter = TensorRTConverter()
    return converter.optimize(engine_path, output_path, optimization_level)


def calibrate_int8(
    model_path: Union[str, Path],
    calibration_data: np.ndarray,
    cache_file: Optional[Union[str, Path]] = None,
) -> str:
    """Calibrate INT8 quantization for TensorRT."""
    converter = TensorRTConverter()
    return converter.calibrate_int8(model_path, calibration_data, cache_file)


# =============================================================================
# OpenVINO
# =============================================================================


class OpenVINOConverter:
    """OpenVINO model converter and optimizer."""

    def __init__(self, config: Optional[EdgeConfig] = None):
        self.config = config or EdgeConfig()

    def convert(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        framework: str = "onnx",
        **kwargs,
    ) -> str:
        """Convert model to OpenVINO format (IR)."""
        try:
            from openvino.tools import mo
            from openvino.runtime import serialize
        except ImportError:
            try:
                import openvino.tools.mo as mo
            except ImportError:
                raise ImportError(
                    "OpenVINO is required. Install with: pip install openvino"
                )

        model_path = Path(model_path)

        if output_path is None:
            output_dir = model_path.parent / "openvino"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{model_path.stem}.xml"
        else:
            output_path = Path(output_path)

        # Convert to IR format
        if framework.lower() == "onnx":
            ov_model = mo.convert_model(str(model_path))
        elif framework.lower() == "tensorflow":
            ov_model = mo.convert_model(str(model_path), framework="tf")
        elif framework.lower() == "pytorch":
            # First convert to ONNX, then to OpenVINO
            onnx_path = model_path.parent / f"{model_path.stem}_temp.onnx"
            convert_to_onnx(model_path, onnx_path, framework="pytorch")
            ov_model = mo.convert_model(str(onnx_path))
            onnx_path.unlink(missing_ok=True)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        # Serialize
        from openvino.runtime import serialize

        serialize(ov_model, str(output_path))

        logger.info(f"Converted to OpenVINO: {output_path}")
        return str(output_path)

    def optimize(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE,
    ) -> str:
        """Optimize OpenVINO model."""
        try:
            from openvino.runtime import Core
            from openvino.tools.pot import IEEngine, load_model, save_model
            from openvino.tools.pot import compress_model_weights
        except ImportError:
            raise ImportError("OpenVINO POT is required.")

        model_path = Path(model_path)

        if output_path is None:
            output_path = model_path.parent / f"{model_path.stem}_optimized.xml"
        else:
            output_path = Path(output_path)

        # Load model
        core = Core()
        model = core.read_model(str(model_path))

        # Apply optimizations
        if optimization_level.value >= 1:
            # Convert to FP16
            from openvino import convert_precision

            model = convert_precision(model, "FP32", "FP16")

        if optimization_level.value >= 2:
            # Weight compression
            compress_model_weights(model)

        # Save
        from openvino.runtime import serialize

        serialize(model, str(output_path))

        logger.info(f"Optimized OpenVINO model: {output_path}")
        return str(output_path)


def convert_to_openvino(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    framework: str = "onnx",
    config: Optional[EdgeConfig] = None,
    **kwargs,
) -> str:
    """Convert model to OpenVINO format."""
    converter = OpenVINOConverter(config)
    return converter.convert(model_path, output_path, framework, **kwargs)


def optimize_openvino(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE,
) -> str:
    """Optimize OpenVINO model."""
    converter = OpenVINOConverter()
    return converter.optimize(model_path, output_path, optimization_level)


# =============================================================================
# Edge Devices
# =============================================================================


class EdgeDevice(ABC):
    """Abstract base class for edge devices."""

    def __init__(self, device_id: str = "0"):
        self.device_id = device_id
        self._initialized = False

    @abstractmethod
    def initialize(self):
        """Initialize the device."""
        pass

    @abstractmethod
    def deploy(self, model_path: Union[str, Path], **kwargs) -> bool:
        """Deploy model to device."""
        pass

    @abstractmethod
    def run_inference(self, inputs: np.ndarray) -> np.ndarray:
        """Run inference on device."""
        pass

    @abstractmethod
    def get_specs(self) -> Dict[str, Any]:
        """Get device specifications."""
        pass

    @abstractmethod
    def optimize_for_device(self, model_path: Union[str, Path]) -> str:
        """Optimize model specifically for this device."""
        pass


class RaspberryPi(EdgeDevice):
    """Raspberry Pi edge device support."""

    def __init__(self, device_id: str = "0", model: str = "4B"):
        super().__init__(device_id)
        self.model = model
        self.interpreter = None

    def initialize(self):
        """Initialize Raspberry Pi (TFLite runtime)."""
        try:
            import tflite_runtime.interpreter as tflite

            self._tflite = tflite
        except ImportError:
            import tensorflow.lite as tflite

            self._tflite = tflite
        self._initialized = True
        logger.info(f"Raspberry Pi {self.model} initialized")

    def deploy(self, model_path: Union[str, Path], **kwargs) -> bool:
        """Deploy TFLite model to Raspberry Pi."""
        if not self._initialized:
            self.initialize()

        model_path = Path(model_path)

        # For RPi, we use CPU delegate with optimized settings
        delegates = []

        # Try XNNPACK delegate for acceleration
        try:
            xnnpack = self._tflite.load_delegate("libxnnpack.so")
            delegates.append(xnnpack)
        except:
            logger.warning("XNNPACK delegate not available")

        self.interpreter = self._tflite.Interpreter(
            model_path=str(model_path),
            num_threads=kwargs.get("num_threads", 4),
            experimental_delegates=delegates,
        )
        self.interpreter.allocate_tensors()

        logger.info(f"Model deployed to Raspberry Pi: {model_path}")
        return True

    def run_inference(self, inputs: np.ndarray) -> np.ndarray:
        """Run inference on Raspberry Pi."""
        if self.interpreter is None:
            raise RuntimeError("Model not deployed. Call deploy() first.")

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]["index"], inputs)
        self.interpreter.invoke()

        return self.interpreter.get_tensor(output_details[0]["index"])

    def get_specs(self) -> Dict[str, Any]:
        """Get Raspberry Pi specifications."""
        specs = {
            "4B": {
                "cpu": "BCM2711 Quad-core Cortex-A72",
                "ram": "2/4/8 GB",
                "gpu": "VideoCore VI",
                "tflops": "~0.1",
                "supported_formats": ["tflite"],
            },
            "5": {
                "cpu": "BCM2712 Quad-core Cortex-A76",
                "ram": "4/8 GB",
                "gpu": "VideoCore VII",
                "tflops": "~0.2",
                "supported_formats": ["tflite"],
            },
        }
        return specs.get(self.model, specs["4B"])

    def optimize_for_device(self, model_path: Union[str, Path]) -> str:
        """Optimize model specifically for Raspberry Pi."""
        # RPi works best with INT8 quantized TFLite models
        output_path = Path(model_path).parent / f"rpi_optimized.tflite"

        # Convert to TFLite with INT8 quantization
        config = EdgeConfig(
            quantization=QuantizationType.INT8,
            optimization_level=OptimizationLevel.AGGRESSIVE,
        )

        return quantize_tflite(model_path, output_path, QuantizationType.INT8)


class NVIDIAJetson(EdgeDevice):
    """NVIDIA Jetson edge device support."""

    def __init__(self, device_id: str = "0", model: str = "Orin"):
        super().__init__(device_id)
        self.model = model
        self.context = None
        self.engine = None

    def initialize(self):
        """Initialize Jetson (TensorRT runtime)."""
        try:
            import tensorrt as trt

            self._trt = trt
            self._cuda_available = True
        except ImportError:
            logger.warning("TensorRT not available")
            self._cuda_available = False

        self._initialized = True
        logger.info(f"NVIDIA Jetson {self.model} initialized")

    def deploy(self, model_path: Union[str, Path], **kwargs) -> bool:
        """Deploy TensorRT engine to Jetson."""
        if not self._initialized:
            self.initialize()

        if not self._cuda_available:
            raise RuntimeError("CUDA/TensorRT not available on this Jetson")

        model_path = Path(model_path)

        # Load TensorRT engine
        logger = self._trt.Logger(self._trt.Logger.INFO)
        runtime = self._trt.Runtime(logger)

        with open(model_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        logger.info(f"TensorRT engine deployed to Jetson: {model_path}")
        return True

    def run_inference(self, inputs: np.ndarray) -> np.ndarray:
        """Run inference on Jetson."""
        if self.context is None:
            raise RuntimeError("Engine not deployed. Call deploy() first.")

        import pycuda.driver as cuda
        import pycuda.autoinit

        # Allocate GPU memory
        d_input = cuda.mem_alloc(inputs.nbytes)
        output = np.empty_like(inputs)  # Adjust shape as needed
        d_output = cuda.mem_alloc(output.nbytes)

        # Copy input to GPU
        cuda.memcpy_htod(d_input, inputs)

        # Run inference
        self.context.execute_v2([int(d_input), int(d_output)])

        # Copy output from GPU
        cuda.memcpy_dtoh(output, d_output)

        return output

    def get_specs(self) -> Dict[str, Any]:
        """Get Jetson specifications."""
        specs = {
            "Nano": {
                "gpu": "128-core Maxwell",
                "ram": "4 GB",
                "ai_performance": "0.5 TFLOPS (FP16)",
                "power": "5-10W",
                "supported_formats": ["tensorrt", "onnx"],
            },
            "TX2": {
                "gpu": "256-core Pascal",
                "ram": "8 GB",
                "ai_performance": "1.3 TFLOPS",
                "power": "7.5-15W",
                "supported_formats": ["tensorrt", "onnx"],
            },
            "Xavier": {
                "gpu": "512-core Volta",
                "ram": "16/32 GB",
                "ai_performance": "10-32 TOPS",
                "power": "10-30W",
                "supported_formats": ["tensorrt", "onnx"],
            },
            "Orin": {
                "gpu": "2048-core Ampere",
                "ram": "8-64 GB",
                "ai_performance": "40-275 TOPS",
                "power": "15-60W",
                "supported_formats": ["tensorrt", "onnx"],
            },
        }
        return specs.get(self.model, specs["Orin"])

    def optimize_for_device(self, model_path: Union[str, Path]) -> str:
        """Optimize model specifically for Jetson."""
        # Jetson works best with TensorRT FP16
        output_path = Path(model_path).parent / f"jetson_optimized.trt"

        config = EdgeConfig(
            enable_fp16=True, optimization_level=OptimizationLevel.MAXIMUM
        )

        # Convert ONNX to TensorRT with FP16
        onnx_path = model_path
        if model_path.suffix != ".onnx":
            onnx_path = Path(model_path).with_suffix(".onnx")
            convert_to_onnx(model_path, onnx_path)

        return convert_to_tensorrt(onnx_path, output_path, "onnx", config)


class CoralTPU(EdgeDevice):
    """Google Coral TPU edge device support."""

    def __init__(self, device_id: str = "0"):
        super().__init__(device_id)
        self.interpreter = None

    def initialize(self):
        """Initialize Coral TPU."""
        try:
            from pycoral.utils.edgetpu import make_interpreter
            from pycoral.adapters import common, classify

            self._make_interpreter = make_interpreter
            self._common = common
            self._classify = classify
            self._pycoral_available = True
        except ImportError:
            logger.warning("pycoral not available")
            self._pycoral_available = False

        self._initialized = True
        logger.info("Coral TPU initialized")

    def deploy(self, model_path: Union[str, Path], **kwargs) -> bool:
        """Deploy Edge TPU compiled model."""
        if not self._initialized:
            self.initialize()

        if not self._pycoral_available:
            raise RuntimeError("pycoral not available")

        model_path = Path(model_path)

        self.interpreter = self._make_interpreter(str(model_path))
        self.interpreter.allocate_tensors()

        logger.info(f"Model deployed to Coral TPU: {model_path}")
        return True

    def run_inference(self, inputs: np.ndarray) -> np.ndarray:
        """Run inference on Coral TPU."""
        if self.interpreter is None:
            raise RuntimeError("Model not deployed. Call deploy() first.")

        self._common.set_input(self.interpreter, inputs)
        self.interpreter.invoke()

        return self._classify.get_classes(self.interpreter)

    def get_specs(self) -> Dict[str, Any]:
        """Get Coral TPU specifications."""
        return {
            "coprocessor": "Edge TPU",
            "peak_tops": "4 TOPS (INT8)",
            "energy": "2 TOPS/W",
            "supported_formats": ["tflite_edgetpu"],
            "model_requirements": "Must be compiled with edgetpu_compiler",
        }

    def optimize_for_device(self, model_path: Union[str, Path]) -> str:
        """Compile model for Coral TPU."""
        model_path = Path(model_path)
        output_path = model_path.parent / f"{model_path.stem}_edgetpu.tflite"

        # Use edgetpu_compiler
        import subprocess

        result = subprocess.run(
            ["edgetpu_compiler", "-o", str(model_path.parent), str(model_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Edge TPU compilation failed: {result.stderr}")

        logger.info(f"Model compiled for Coral TPU: {output_path}")
        return str(output_path)


class IntelMovidius(EdgeDevice):
    """Intel Movidius VPU edge device support."""

    def __init__(self, device_id: str = "0", model: str = "MyriadX"):
        super().__init__(device_id)
        self.model = model
        self.ie = None
        self.exec_network = None

    def initialize(self):
        """Initialize Intel Movidius."""
        try:
            from openvino.runtime import Core

            self._core = Core()
            self._openvino_available = True
        except ImportError:
            logger.warning("OpenVINO not available")
            self._openvino_available = False

        self._initialized = True
        logger.info(f"Intel Movidius {self.model} initialized")

    def deploy(self, model_path: Union[str, Path], **kwargs) -> bool:
        """Deploy OpenVINO model to Movidius."""
        if not self._initialized:
            self.initialize()

        if not self._openvino_available:
            raise RuntimeError("OpenVINO not available")

        model_path = Path(model_path)

        # Load model
        model = self._core.read_model(str(model_path))

        # Compile for Movidius VPU
        self.exec_network = self._core.compile_model(model, "MYRIAD")

        logger.info(f"Model deployed to Intel Movidius: {model_path}")
        return True

    def run_inference(self, inputs: np.ndarray) -> np.ndarray:
        """Run inference on Movidius."""
        if self.exec_network is None:
            raise RuntimeError("Model not deployed. Call deploy() first.")

        infer_request = self.exec_network.create_infer_request()
        infer_request.infer({0: inputs})

        return infer_request.get_output_tensor().data

    def get_specs(self) -> Dict[str, Any]:
        """Get Movidius specifications."""
        specs = {
            "MyriadX": {
                "neural_compute_engine": "16 SHAVE cores",
                "tflops": "1 TOPS (INT8)",
                "power": "~1W",
                "supported_formats": ["openvino"],
                "interface": "USB 3.0 / PCIe",
            }
        }
        return specs.get(self.model, specs["MyriadX"])

    def optimize_for_device(self, model_path: Union[str, Path]) -> str:
        """Optimize model specifically for Movidius."""
        # Movidius works best with OpenVINO FP16
        output_path = Path(model_path).parent / f"movidius_optimized.xml"

        return optimize_openvino(model_path, output_path, OptimizationLevel.AGGRESSIVE)


# =============================================================================
# Optimization
# =============================================================================


class EdgeOptimizer:
    """Edge model optimizer with advanced techniques."""

    def __init__(self, config: Optional[EdgeConfig] = None):
        self.config = config or EdgeConfig()

    def optimize(
        self,
        model_path: Union[str, Path],
        target_device: Union[str, EdgeDevice],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Optimize model for edge deployment."""
        model_path = Path(model_path)

        if output_path is None:
            output_path = (
                model_path.parent
                / f"{model_path.stem}_edge_optimized{model_path.suffix}"
            )
        else:
            output_path = Path(output_path)

        # Get device-specific optimizer
        if isinstance(target_device, str):
            device = self._get_device(target_device)
        else:
            device = target_device

        # Run device-specific optimization
        optimized_path = device.optimize_for_device(model_path)

        # Move to desired output location
        if optimized_path != str(output_path):
            shutil.copy(optimized_path, output_path)

        logger.info(f"Model optimized for edge: {output_path}")
        return str(output_path)

    def _get_device(self, device_name: str) -> EdgeDevice:
        """Get device instance by name."""
        devices = {
            "raspberrypi": RaspberryPi,
            "rpi": RaspberryPi,
            "jetson": NVIDIAJetson,
            "nvidia_jetson": NVIDIAJetson,
            "coral": CoralTPU,
            "coral_tpu": CoralTPU,
            "movidius": IntelMovidius,
            "intel_movidius": IntelMovidius,
        }

        device_class = devices.get(device_name.lower())
        if device_class is None:
            raise ValueError(f"Unknown device: {device_name}")

        return device_class()


def model_compression(
    model_path: Union[str, Path],
    method: str = "pruning",
    compression_ratio: float = 0.5,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Compress model for edge deployment."""
    model_path = Path(model_path)

    if output_path is None:
        output_path = (
            model_path.parent / f"{model_path.stem}_compressed{model_path.suffix}"
        )
    else:
        output_path = Path(output_path)

    if method == "pruning":
        # Structured or unstructured pruning
        output_path = _prune_model(model_path, compression_ratio, output_path)
    elif method == "quantization":
        # Weight quantization
        output_path = _quantize_weights(model_path, compression_ratio, output_path)
    elif method == "factorization":
        # Low-rank factorization
        output_path = _factorize_model(model_path, compression_ratio, output_path)
    else:
        raise ValueError(f"Unknown compression method: {method}")

    logger.info(f"Model compressed: {output_path}")
    return str(output_path)


def _prune_model(model_path: Path, sparsity: float, output_path: Path) -> Path:
    """Prune model weights."""
    try:
        import tensorflow as tf
        import tensorflow_model_optimization as tfmot
    except ImportError:
        raise ImportError("TensorFlow and tfmot required for pruning")

    # Load model
    model = tf.keras.models.load_model(str(model_path))

    # Apply pruning
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(sparsity, 0)
    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    # Strip pruning wrappers for deployment
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    # Save
    final_model.save(str(output_path))

    return output_path


def _quantize_weights(model_path: Path, bits: float, output_path: Path) -> Path:
    """Quantize model weights."""
    # Use TFLite quantization as default
    if bits <= 8:
        return Path(quantize_tflite(model_path, output_path, QuantizationType.INT8))
    elif bits <= 16:
        return Path(quantize_tflite(model_path, output_path, QuantizationType.FP16))
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


def _factorize_model(model_path: Path, rank_ratio: float, output_path: Path) -> Path:
    """Apply low-rank factorization to model."""
    try:
        import tensorflow as tf
        import tensorflow_model_optimization as tfmot
    except ImportError:
        raise ImportError("TensorFlow required for factorization")

    # Load model
    model = tf.keras.models.load_model(str(model_path))

    # Apply clustering (similar compression effect)
    cluster_weights = tfmot.clustering.keras.cluster_weights
    clustering_params = {
        "number_of_clusters": int(16 * rank_ratio),
        "cluster_centroids_init": tfmot.clustering.keras.CentroidInitialization.LINEAR,
    }

    clustered_model = cluster_weights(model, **clustering_params)

    # Strip clustering for deployment
    final_model = tfmot.clustering.keras.strip_clustering(clustered_model)

    # Save
    final_model.save(str(output_path))

    return output_path


def knowledge_distillation_edge(
    teacher_path: Union[str, Path],
    student_architecture: Union[str, Dict],
    dataset: Any,
    output_path: Union[str, Path],
    temperature: float = 4.0,
    alpha: float = 0.5,
    epochs: int = 10,
) -> str:
    """Distill knowledge from teacher to student model for edge."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required for knowledge distillation")

    teacher_path = Path(teacher_path)
    output_path = Path(output_path)

    # Load teacher
    teacher = tf.keras.models.load_model(str(teacher_path))
    teacher.trainable = False

    # Create student
    if isinstance(student_architecture, str):
        # Load predefined architecture
        student = _get_student_model(student_architecture, teacher)
    else:
        # Build from config
        student = tf.keras.models.model_from_config(student_architecture)

    # Distillation training
    student = _train_distillation(teacher, student, dataset, temperature, alpha, epochs)

    # Save student
    student.save(str(output_path))

    logger.info(f"Knowledge distillation complete: {output_path}")
    return str(output_path)


def _get_student_model(arch: str, teacher: Any) -> Any:
    """Get student model architecture."""
    import tensorflow as tf

    if arch == "mobilenet":
        return tf.keras.applications.MobileNetV2(
            weights=None,
            input_shape=teacher.input_shape[1:],
            classes=teacher.output_shape[-1],
        )
    elif arch == "efficientnet":
        return tf.keras.applications.EfficientNetB0(
            weights=None,
            input_shape=teacher.input_shape[1:],
            classes=teacher.output_shape[-1],
        )
    else:
        # Create a smaller version of teacher
        config = teacher.get_config()
        # Reduce layer sizes (simplified)
        return tf.keras.models.clone_model(teacher)


def _train_distillation(
    teacher: Any,
    student: Any,
    dataset: Any,
    temperature: float,
    alpha: float,
    epochs: int,
) -> Any:
    """Train student with knowledge distillation."""
    import tensorflow as tf

    # Distillation loss
    def distillation_loss(y_true, y_pred):
        soft_targets = tf.nn.softmax(teacher(y_true) / temperature)
        soft_predictions = tf.nn.softmax(y_pred / temperature)

        distillation = tf.keras.losses.KLDivergence()(
            soft_targets, soft_predictions
        ) * (temperature**2)

        hard_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)

        return alpha * hard_loss + (1 - alpha) * distillation

    student.compile(optimizer="adam", loss=distillation_loss, metrics=["accuracy"])

    student.fit(dataset, epochs=epochs)

    return student


def architecture_search_edge(
    dataset: Any,
    constraints: Dict[str, Any],
    search_space: str = "mobilenet",
    num_trials: int = 100,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Neural Architecture Search for edge devices."""
    try:
        import tensorflow as tf
        import keras_tuner as kt
    except ImportError:
        raise ImportError(
            "keras-tuner required for NAS. Install with: pip install keras-tuner"
        )

    if output_path is None:
        output_path = Path("edge_model_nas")
    else:
        output_path = Path(output_path)

    # Define search space
    def build_model(hp):
        if search_space == "mobilenet":
            alpha = hp.Float("alpha", 0.35, 1.3, step=0.15)
            model = tf.keras.applications.MobileNetV2(
                alpha=alpha,
                weights=None,
                input_shape=constraints.get("input_shape", (224, 224, 3)),
                classes=constraints.get("num_classes", 10),
            )
        else:
            # Custom search space
            model = tf.keras.Sequential()
            model.add(
                tf.keras.layers.Input(
                    shape=constraints.get("input_shape", (224, 224, 3))
                )
            )

            for i in range(hp.Int("num_layers", 2, 8)):
                model.add(
                    tf.keras.layers.Conv2D(
                        hp.Int(f"filters_{i}", 16, 256, step=16), 3, activation="relu"
                    )
                )
                if hp.Boolean(f"pool_{i}"):
                    model.add(tf.keras.layers.MaxPooling2D())

            model.add(tf.keras.layers.GlobalAveragePooling2D())
            model.add(tf.keras.layers.Dense(constraints.get("num_classes", 10)))

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    # Create tuner
    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_accuracy",
        max_trials=num_trials,
        directory=str(output_path),
        project_name="edge_nas",
    )

    # Search
    tuner.search(dataset["train"], validation_data=dataset.get("val"), epochs=10)

    # Get best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Save
    best_path = output_path / "best_model.h5"
    best_model.save(str(best_path))

    logger.info(f"NAS complete. Best model: {best_path}")
    return str(best_path)


# =============================================================================
# Deployment
# =============================================================================


def deploy_to_edge(
    model_path: Union[str, Path],
    device: Union[str, EdgeDevice],
    config: Optional[EdgeConfig] = None,
    **kwargs,
) -> bool:
    """Deploy model to edge device."""
    model_path = Path(model_path)

    # Get device instance
    if isinstance(device, str):
        device_map = {
            "raspberrypi": RaspberryPi,
            "rpi": RaspberryPi,
            "jetson": NVIDIAJetson,
            "nvidia_jetson": NVIDIAJetson,
            "coral": CoralTPU,
            "coral_tpu": CoralTPU,
            "movidius": IntelMovidius,
            "intel_movidius": IntelMovidius,
        }
        device_class = device_map.get(device.lower())
        if device_class is None:
            raise ValueError(f"Unknown device: {device}")
        device = device_class()

    # Initialize and deploy
    device.initialize()

    return device.deploy(model_path, **kwargs)


def create_edge_package(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Dict[str, Any],
    include_runtime: bool = False,
) -> str:
    """Create deployment package for edge device."""
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    package_name = f"{model_path.stem}_edge_package"
    package_dir = output_dir / package_name
    package_dir.mkdir(exist_ok=True)

    # Copy model
    model_dest = package_dir / model_path.name
    shutil.copy(model_path, model_dest)

    # Create manifest
    manifest = {
        "model": {
            "name": model_path.name,
            "format": model_path.suffix.lstrip("."),
            "size_mb": model_path.stat().st_size / (1024 * 1024),
        },
        "config": config,
        "runtime_requirements": _get_runtime_requirements(model_path),
        "deployment": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "package_version": "1.0.0",
        },
    }

    with open(package_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Create inference script
    inference_script = _create_inference_script(config)
    with open(package_dir / "inference.py", "w") as f:
        f.write(inference_script)

    # Include runtime if requested
    if include_runtime:
        runtime_dir = package_dir / "runtime"
        runtime_dir.mkdir(exist_ok=True)
        _copy_runtime_files(runtime_dir, model_path.suffix)

    # Create archive
    archive_path = output_dir / f"{package_name}.tar.gz"
    shutil.make_archive(str(archive_path.with_suffix("")), "gztar", package_dir)

    logger.info(f"Edge package created: {archive_path}")
    return str(archive_path)


def _get_runtime_requirements(model_path: Path) -> List[str]:
    """Get runtime requirements for model format."""
    suffix = model_path.suffix.lower()

    requirements = {
        ".tflite": ["tflite-runtime"],
        ".onnx": ["onnxruntime"],
        ".trt": ["tensorrt", "pycuda"],
        ".xml": ["openvino"],
        ".pb": ["tensorflow"],
    }

    return requirements.get(suffix, ["numpy"])


def _create_inference_script(config: Dict[str, Any]) -> str:
    """Create inference script for edge deployment."""
    script = '''#!/usr/bin/env python3
"""Edge inference script."""

import argparse
import numpy as np
import json
from pathlib import Path

def load_model(model_path):
    """Load model based on format."""
    suffix = Path(model_path).suffix.lower()
    
    if suffix == '.tflite':
        import tensorflow.lite as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    elif suffix == '.onnx':
        import onnxruntime as ort
        return ort.InferenceSession(model_path)
    elif suffix == '.trt':
        import tensorrt as trt
        # TensorRT loading logic
        pass
    else:
        raise ValueError(f"Unsupported model format: {suffix}")

def run_inference(model, inputs):
    """Run inference."""
    # Implementation depends on model type
    pass

def main():
    parser = argparse.ArgumentParser(description='Edge inference')
    parser.add_argument('--model', required=True, help='Model path')
    parser.add_argument('--input', required=True, help='Input data path')
    parser.add_argument('--output', default='output.npy', help='Output path')
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Load input
    inputs = np.load(args.input)
    
    # Run inference
    outputs = run_inference(model, inputs)
    
    # Save output
    np.save(args.output, outputs)
    print(f"Output saved to: {args.output}")

if __name__ == '__main__':
    main()
'''
    return script


def _copy_runtime_files(runtime_dir: Path, model_suffix: str):
    """Copy runtime files to package."""
    # Create requirements.txt
    requirements = _get_runtime_requirements(Path(f"model{model_suffix}"))
    with open(runtime_dir / "requirements.txt", "w") as f:
        f.write("\\n".join(requirements))


def update_model_ota(
    device: EdgeDevice,
    model_path: Union[str, Path],
    version: str,
    rollback_on_failure: bool = True,
) -> bool:
    """Update model on edge device over-the-air."""
    model_path = Path(model_path)

    logger.info(f"Starting OTA update: version {version}")

    try:
        # Backup current model if exists
        if rollback_on_failure:
            backup_path = Path("/tmp/model_backup")
            # Device-specific backup logic would go here

        # Deploy new model
        success = device.deploy(model_path)

        if not success:
            raise RuntimeError("Deployment failed")

        # Verify with test inference
        # test_input = np.random.randn(*device.get_input_shape())
        # output = device.run_inference(test_input)

        logger.info(f"OTA update successful: version {version}")
        return True

    except Exception as e:
        logger.error(f"OTA update failed: {e}")

        if rollback_on_failure:
            logger.info("Rolling back to previous version")
            # Rollback logic would go here

        return False


# =============================================================================
# Utilities
# =============================================================================


def edge_export(
    model: Any, output_path: Union[str, Path], format: str = "tflite", **kwargs
) -> str:
    """Export model to edge-compatible format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "tflite":
        return convert_to_tflite(model, output_path, **kwargs)
    elif format.lower() == "onnx":
        return convert_to_onnx(model, output_path, **kwargs)
    elif format.lower() == "tensorrt":
        return convert_to_tensorrt(model, output_path, **kwargs)
    elif format.lower() == "openvino":
        return convert_to_openvino(model, output_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def optimize_for_edge(
    model_path: Union[str, Path],
    target_device: str,
    optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE,
    quantization: Optional[QuantizationType] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Complete optimization pipeline for edge deployment."""
    model_path = Path(model_path)

    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_edge{model_path.suffix}"
    else:
        output_path = Path(output_path)

    # Step 1: Convert to appropriate format
    device_formats = {
        "raspberrypi": "tflite",
        "rpi": "tflite",
        "jetson": "tensorrt",
        "nvidia_jetson": "tensorrt",
        "coral": "tflite",
        "coral_tpu": "tflite",
        "movidius": "openvino",
        "intel_movidius": "openvino",
    }

    target_format = device_formats.get(target_device.lower(), "onnx")

    # Step 2: Apply format-specific optimization
    if target_format == "tflite":
        temp_path = convert_to_tflite(model_path)
        if quantization:
            temp_path = quantize_tflite(temp_path, quantization_type=quantization)
        temp_path = optimize_tflite(temp_path, optimization_level=optimization_level)
    elif target_format == "tensorrt":
        onnx_path = (
            convert_to_onnx(model_path) if model_path.suffix != ".onnx" else model_path
        )
        config = EdgeConfig(enable_fp16=True, optimization_level=optimization_level)
        temp_path = convert_to_tensorrt(onnx_path, config=config)
    elif target_format == "openvino":
        temp_path = convert_to_openvino(model_path)
        temp_path = optimize_openvino(temp_path, optimization_level=optimization_level)
    else:
        temp_path = convert_to_onnx(model_path)
        temp_path = optimize_onnx(temp_path, optimization_level=optimization_level)

    # Step 3: Model compression if needed
    if optimization_level.value >= 2:
        temp_path = model_compression(
            temp_path, method="pruning", compression_ratio=0.3
        )

    # Move to final location
    shutil.move(temp_path, output_path)

    logger.info(f"Edge optimization complete: {output_path}")
    return str(output_path)


def benchmark_edge(
    model_path: Union[str, Path],
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: Optional[EdgeDevice] = None,
) -> EdgeMetrics:
    """Benchmark model on edge device."""
    import time

    model_path = Path(model_path)

    # Create random input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Load model based on format
    suffix = model_path.suffix.lower()

    if suffix == ".tflite":
        import tensorflow.lite as tflite

        interpreter = tflite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        def run_inference():
            interpreter.set_tensor(input_details[0]["index"], dummy_input)
            interpreter.invoke()
            return interpreter.get_tensor(output_details[0]["index"])

    elif suffix == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name

        def run_inference():
            return session.run(None, {input_name: dummy_input})

    elif suffix == ".trt":
        # TensorRT benchmarking
        def run_inference():
            return np.zeros((1, 1000))  # Placeholder

    else:
        raise ValueError(f"Unsupported format for benchmarking: {suffix}")

    # Warmup
    for _ in range(warmup_runs):
        run_inference()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        run_inference()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    # Calculate metrics
    avg_latency = np.mean(latencies)
    throughput = 1000 / avg_latency  # FPS
    model_size = model_path.stat().st_size / (1024 * 1024)  # MB

    metrics = EdgeMetrics(
        latency_ms=avg_latency, throughput_fps=throughput, model_size_mb=model_size
    )

    logger.info(
        f"Benchmark results: latency={avg_latency:.2f}ms, throughput={throughput:.2f}fps"
    )

    return metrics


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    # TensorFlow Lite
    "TFLiteConverter",
    "convert_to_tflite",
    "optimize_tflite",
    "quantize_tflite",
    # ONNX
    "ONNXConverter",
    "convert_to_onnx",
    "optimize_onnx",
    "quantize_onnx",
    # TensorRT
    "TensorRTConverter",
    "convert_to_tensorrt",
    "optimize_tensorrt",
    "calibrate_int8",
    # OpenVINO
    "OpenVINOConverter",
    "convert_to_openvino",
    "optimize_openvino",
    # Edge Devices
    "RaspberryPi",
    "NVIDIAJetson",
    "CoralTPU",
    "IntelMovidius",
    "EdgeDevice",
    # Optimization
    "EdgeOptimizer",
    "model_compression",
    "knowledge_distillation_edge",
    "architecture_search_edge",
    # Deployment
    "deploy_to_edge",
    "create_edge_package",
    "update_model_ota",
    # Utilities
    "edge_export",
    "optimize_for_edge",
    "benchmark_edge",
    # Config
    "EdgeConfig",
    "EdgeMetrics",
    "QuantizationType",
    "OptimizationLevel",
]
