"""
Model Quantization Utilities for fishstick

Provides comprehensive quantization utilities including dynamic quantization,
static quantization, post-training quantization (PTQ), and quantization-aware
training (QAT) setup.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Union, Callable, Tuple, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import copy

import torch
from torch import nn
from torch import Tensor
from torch.quantization import (
    QConfig,
    QConfigDynamic,
    default_dynamic_qconfig,
    float16_dynamic_qconfig,
    default_qconfig,
    default_per_channel_qconfig,
)
from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx


class QuantizationType(Enum):
    """Types of quantization."""

    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"
    POST_TRAINING = "post_training"


class QuantizationDtype(Enum):
    """Quantization data types."""

    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    UINT8 = "uint8"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    dtype: QuantizationDtype = QuantizationDtype.INT8
    qconfig: Optional[QConfig] = None
    dynamic_qconfig: Optional[QConfigDynamic] = None
    observer_enabled: bool = True
    prepare_mappings: Optional[Dict[Type, Type]] = None


@dataclass
class CalibrationConfig:
    """Configuration for calibration."""

    num_samples: int = 100
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0


@dataclass
class QuantizationResult:
    """Results from quantization."""

    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    quantized_model: nn.Module
    config: QuantizationConfig


class DynamicQuantizer:
    """
    Dynamic quantization for deployment.

    Applies dynamic quantization to model layers, typically for
    linear and recurrent layers.

    Example:
        >>> quantizer = DynamicQuantizer()
        >>> qmodel = quantizer.quantize(model)
    """

    def __init__(
        self,
        dtype: QuantizationDtype = QuantizationDtype.INT8,
    ):
        self.dtype = dtype

    def quantize(
        self,
        model: nn.Module,
        layer_types: Optional[List[Type[nn.Module]]] = None,
    ) -> nn.Module:
        """
        Apply dynamic quantization to model.

        Args:
            model: PyTorch model
            layer_types: Types of layers to quantize

        Returns:
            Quantized model
        """
        if layer_types is None:
            layer_types = [nn.Linear, nn.LSTM, nn.GRU, nn.RNN]

        if self.dtype == QuantizationDtype.FP16:
            qconfig = float16_dynamic_qconfig
        else:
            qconfig = default_dynamic_qconfig

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {lt: qconfig for lt in layer_types},
            dtype=torch.qint8
            if self.dtype == QuantizationDtype.INT8
            else torch.float16,
        )

        return quantized_model

    def quantize_with_config(
        self,
        model: nn.Module,
        qconfig: QConfigDynamic,
    ) -> nn.Module:
        """
        Quantize with custom config.

        Args:
            model: PyTorch model
            qconfig: Custom quantization config

        Returns:
            Quantized model
        """
        layer_types = [nn.Linear, nn.LSTM, nn.GRU, nn.RNN]

        return torch.quantization.quantize_dynamic(
            model,
            {lt: qconfig for lt in layer_types},
        )


class StaticQuantizer:
    """
    Static quantization for deployment.

    Requires calibration with representative dataset.

    Example:
        >>> quantizer = StaticQuantizer()
        >>> qmodel = quantizer.quantize(model, calibration_data)
    """

    def __init__(
        self,
        qconfig: Optional[QConfig] = None,
    ):
        self.qconfig = qconfig or default_qconfig

    def prepare(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Prepare model for static quantization.

        Args:
            model: PyTorch model

        Returns:
            Prepared model
        """
        model.eval()
        model.qconfig = self.qconfig
        torch.quantization.prepare(model, inplace=True)
        return model

    def calibrate(
        self,
        model: nn.Module,
        calibration_data: Union[List[Tensor], Callable],
        num_samples: int = 100,
    ) -> None:
        """
        Run calibration.

        Args:
            model: Prepared model
            calibration_data: Data or data loader for calibration
            num_samples: Number of samples to use
        """
        model.eval()

        samples_processed = 0

        if callable(calibration_data):
            with torch.no_grad():
                for batch in calibration_data():
                    if samples_processed >= num_samples:
                        break

                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch

                    model(inputs)
                    samples_processed += inputs.size(0)
        else:
            data_iter = iter(calibration_data)

            with torch.no_grad():
                while samples_processed < num_samples:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(calibration_data)
                        batch = next(data_iter)

                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch

                    model(inputs)
                    samples_processed += inputs.size(0)

    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[Union[List[Tensor], Callable]] = None,
    ) -> nn.Module:
        """
        Quantize model with optional calibration.

        Args:
            model: PyTorch model
            calibration_data: Optional calibration data

        Returns:
            Quantized model
        """
        prepared = self.prepare(model)

        if calibration_data is not None:
            self.calibrate(prepared, calibration_data)

        quantized = torch.quantization.convert(prepared, inplace=False)

        return quantized


class QATQuantizer:
    """
    Quantization-Aware Training (QAT) setup.

    Provides utilities for setting up QAT which typically yields
    better accuracy than post-training quantization.

    Example:
        >>> quantizer = QATQuantizer()
        >>> qmodel = quantizer.prepare(model)
        >>> # Train model...
        >>> qmodel = quantizer.convert(qmodel)
    """

    def __init__(
        self,
        qconfig: Optional[QConfig] = None,
    ):
        self.qconfig = qconfig or default_qconfig

    def prepare(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Prepare model for QAT.

        Args:
            model: PyTorch model

        Returns:
            Prepared model for QAT
        """
        model.train()
        model.qconfig = self.qconfig
        torch.quantization.prepare_qat(model, inplace=True)
        return model

    def convert(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Convert QAT model to quantized model.

        Args:
            model: QAT-prepared model

        Returns:
            Quantized model
        """
        model.eval()
        return torch.quantization.convert(model, inplace=False)

    def prepare_and_convert_fx(
        self,
        model: nn.Module,
        example_inputs: Tuple[Tensor, ...],
    ) -> nn.Module:
        """
        Prepare and convert using FX graph mode.

        Args:
            model: PyTorch model
            example_inputs: Example inputs for tracing

        Returns:
            Quantized model
        """
        model.eval()

        qconfig_dict = {"": default_qconfig}

        prepared = prepare_qat_fx(model, qconfig_dict, example_inputs)

        converted = convert_fx(prepared)

        return converted


class PTQQuantizer:
    """
    Post-Training Quantization (PTQ) utilities.

    Applies quantization after training without requiring gradient computation.

    Example:
        >>> quantizer = PTQQuantizer()
        >>> qmodel = quantizer.quantize(model, calibration_data)
    """

    def __init__(
        self,
        per_channel: bool = False,
        calibration_config: Optional[CalibrationConfig] = None,
    ):
        self.per_channel = per_channel
        self.calibration_config = calibration_config or CalibrationConfig()

    def quantize(
        self,
        model: nn.Module,
        calibration_data: Union[List[Tensor], Callable, torch.utils.data.DataLoader],
    ) -> nn.Module:
        """
        Apply post-training quantization.

        Args:
            model: PyTorch model
            calibration_data: Data for calibration

        Returns:
            Quantized model
        """
        qconfig = default_per_channel_qconfig if self.per_channel else default_qconfig

        quantizer = StaticQuantizer(qconfig=qconfig)
        return quantizer.quantize(model, calibration_data)

    def quantize_with_validation(
        self,
        model: nn.Module,
        calibration_data: Union[List[Tensor], Callable],
        validation_data: Union[List[Tensor], Callable],
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ) -> Tuple[nn.Module, float]:
        """
        Quantize with accuracy validation.

        Args:
            model: PyTorch model
            calibration_data: Data for calibration
            validation_data: Data for validation
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Tuple of (quantized_model, max_difference)
        """
        model.eval()

        qconfig = default_per_channel_qconfig if self.per_channel else default_qconfig

        quantizer = StaticQuantizer(qconfig=qconfig)
        quantized = quantizer.quantize(model, calibration_data)

        max_diff = self._validate_accuracy(
            model, quantized, validation_data, rtol, atol
        )

        return quantized, max_diff

    def _validate_accuracy(
        self,
        original: nn.Module,
        quantized: nn.Module,
        validation_data: Union[List[Tensor], Callable],
        rtol: float,
        atol: float,
    ) -> float:
        """Validate output accuracy."""
        max_diff = 0.0

        original.eval()
        quantized.eval()

        if callable(validation_data):
            data_iter = validation_data()
        else:
            data_iter = iter(validation_data)

        with torch.no_grad():
            for batch in data_iter:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                original_out = original(inputs)
                quantized_out = quantized(inputs)

                if isinstance(original_out, Tensor):
                    original_out = [original_out]
                if isinstance(quantized_out, Tensor):
                    quantized_out = [quantized_out]

                for orig, quant in zip(original_out, quantized_out):
                    diff = torch.max(torch.abs(orig - quant)).item()
                    max_diff = max(max_diff, diff)

        return max_diff


class ModelQuantizer:
    """
    Unified quantizer with all quantization methods.

    Provides a unified interface for different quantization types.

    Example:
        >>> quantizer = ModelQuantizer(QuantizationType.DYNAMIC)
        >>> result = quantizer.quantize(model)
    """

    def __init__(
        self,
        quantization_type: QuantizationType = QuantizationType.DYNAMIC,
        dtype: QuantizationDtype = QuantizationDtype.INT8,
        config: Optional[QuantizationConfig] = None,
    ):
        self.quantization_type = quantization_type
        self.dtype = dtype
        self.config = config

        if quantization_type == QuantizationType.DYNAMIC:
            self._quantizer = DynamicQuantizer(dtype=dtype)
        elif quantization_type == QuantizationType.STATIC:
            self._quantizer = StaticQuantizer()
        elif quantization_type == QuantizationType.QAT:
            self._quantizer = QATQuantizer()
        else:
            self._quantizer = PTQQuantizer(
                per_channel=(dtype == QuantizationDtype.INT8)
            )

    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[Union[List[Tensor], Callable]] = None,
    ) -> QuantizationResult:
        """
        Quantize the model.

        Args:
            model: PyTorch model
            calibration_data: Optional calibration data

        Returns:
            QuantizationResult with model and stats
        """
        original_size = self._get_model_size(model)

        if self.quantization_type == QuantizationType.DYNAMIC:
            quantized = self._quantizer.quantize(model)
        elif self.quantization_type == QuantizationType.STATIC:
            quantized = self._quantizer.quantize(model, calibration_data)
        else:
            raise ValueError("QAT requires prepare and convert steps")

        quantized_size = self._get_model_size(quantized)

        return QuantizationResult(
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=original_size / quantized_size,
            quantized_model=quantized,
            config=self.config
            or QuantizationConfig(
                quantization_type=self.quantization_type,
                dtype=self.dtype,
            ),
        )

    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / (1024 * 1024)


def quantize_dynamic(
    model: nn.Module,
    layer_types: Optional[List[Type[nn.Module]]] = None,
) -> nn.Module:
    """
    Convenience function for dynamic quantization.

    Args:
        model: PyTorch model
        layer_types: Layers to quantize

    Returns:
        Quantized model
    """
    quantizer = DynamicQuantizer()
    return quantizer.quantize(model, layer_types)


def quantize_static(
    model: nn.Module,
    calibration_data: Union[List[Tensor], Callable],
) -> nn.Module:
    """
    Convenience function for static quantization.

    Args:
        model: PyTorch model
        calibration_data: Calibration data

    Returns:
        Quantized model
    """
    quantizer = StaticQuantizer()
    return quantizer.quantize(model, calibration_data)


def quantize_model(
    model: nn.Module,
    quantization_type: QuantizationType = QuantizationType.DYNAMIC,
    calibration_data: Optional[Union[List[Tensor], Callable]] = None,
) -> nn.Module:
    """
    General quantization convenience function.

    Args:
        model: PyTorch model
        quantization_type: Type of quantization
        calibration_data: Optional calibration data

    Returns:
        Quantized model
    """
    quantizer = ModelQuantizer(quantization_type)

    if quantization_type == QuantizationType.QAT:
        return quantizer._quantizer.prepare(model)

    result = quantizer.quantize(model, calibration_data)
    return result.quantized_model


def get_quantization_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Get quantization statistics for a model.

    Args:
        model: PyTorch model

    Returns:
        Statistics dictionary
    """
    stats = {
        "has_qconfig": model.qconfig is not None,
        "qconfig": str(model.qconfig) if model.qconfig else None,
    }

    quantized_layers = []
    for name, module in model.named_modules():
        if hasattr(module, "weight_quantizer"):
            quantized_layers.append(name)

    stats["quantized_layers"] = quantized_layers
    stats["num_quantized_layers"] = len(quantized_layers)

    return stats
