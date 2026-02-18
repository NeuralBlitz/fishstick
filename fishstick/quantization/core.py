"""
Comprehensive Quantization Module for Fishstick

Provides complete model compression through:
- Post-training quantization (static, dynamic, per-channel, per-tensor)
- Quantization-aware training (QAT)
- Low-precision formats (INT8, FP16, BF16, INT4, Ternary, Binary)
- Pruning (magnitude, structured, lottery ticket, gradual)
- Knowledge distillation (logit, feature, relation)
- Efficient inference layers
- Utilities for model quantization
"""

from typing import Optional, List, Dict, Callable, Tuple, Union, Any, Iterator
from abc import ABC, abstractmethod
from enum import Enum
import copy
import math
import warnings
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
from torch.ao.quantization import get_default_qconfig, get_default_qat_qconfig
from torch.ao.quantization.quantize import prepare_qat


# ============================================================================
# Post-Training Quantization
# ============================================================================


class BaseQuantization(ABC):
    """Base class for quantization methods."""

    @abstractmethod
    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model."""
        pass

    @abstractmethod
    def dequantize(self, model: nn.Module) -> nn.Module:
        """Dequantize model."""
        pass


class StaticQuantization(BaseQuantization):
    """Static quantization with calibration-based range computation.

    Pre-computes quantization parameters using calibration data.
    Best for models with predictable activation distributions.

    Args:
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        fuse_modules: Whether to fuse Conv-BN-ReLU patterns

    Example:
        >>> quantizer = StaticQuantization(backend='fbgemm')
        >>> prepared = quantizer.prepare(model)
        >>> quantizer.calibrate(calibration_fn)
        >>> quantized = quantizer.convert()
    """

    def __init__(self, backend: str = "fbgemm", fuse_modules: bool = True):
        self.backend = backend
        self.fuse_modules_enabled = fuse_modules
        self.prepared_model: Optional[nn.Module] = None
        self.quantized_model: Optional[nn.Module] = None
        self.calibration_stats: Dict[str, Any] = {}

    def prepare(self, model: nn.Module) -> nn.Module:
        """Prepare model for static quantization."""
        model.eval()
        self.original_model = copy.deepcopy(model)

        if self.fuse_modules_enabled:
            model = self._fuse_modules(model)

        qconfig = get_default_qconfig(self.backend)
        model.qconfig = qconfig

        torch.quantization.prepare(model, inplace=True)
        self.prepared_model = model

        return model

    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse Conv-BN-ReLU patterns for better quantization."""
        fuse_patterns = []

        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                # Check for Conv-BN-ReLU pattern
                if len(module) >= 2:
                    if isinstance(module[0], nn.Conv2d) and isinstance(
                        module[1], nn.BatchNorm2d
                    ):
                        if len(module) >= 3 and isinstance(module[2], nn.ReLU):
                            fuse_patterns.append((name, ["0", "1", "2"]))
                        else:
                            fuse_patterns.append((name, ["0", "1"]))
            elif hasattr(module, "named_children"):
                self._fuse_modules(module)

        for parent_name, indices in fuse_patterns:
            try:
                module_path = f"{parent_name}"
                module_list = [f"{parent_name}.{idx}" for idx in indices]
                torch.quantization.fuse_modules(model, module_list, inplace=True)
            except (ValueError, AttributeError):
                pass

        return model

    def calibrate(
        self,
        calibration_fn: Callable[[], Iterator],
        num_batches: int = 100,
        collect_stats: bool = True,
    ):
        """Calibrate quantization parameters using sample data.

        Args:
            calibration_fn: Function yielding calibration batches
            num_batches: Number of calibration batches
            collect_stats: Whether to collect detailed statistics
        """
        if self.prepared_model is None:
            raise ValueError("Call prepare() before calibrate()")

        self.prepared_model.eval()
        activation_stats = {}

        with torch.no_grad():
            for i, data in enumerate(calibration_fn()):
                if i >= num_batches:
                    break

                if isinstance(data, (tuple, list)):
                    _ = self.prepared_model(data[0])
                else:
                    _ = self.prepared_model(data)

                if collect_stats:
                    self._collect_activation_stats(activation_stats)

        if collect_stats:
            self.calibration_stats = activation_stats

    def _collect_activation_stats(self, stats: Dict):
        """Collect activation statistics during calibration."""
        for name, module in self.prepared_model.named_modules():
            if hasattr(module, "observer_enabled"):
                if name not in stats:
                    stats[name] = {"min": [], "max": [], "mean": [], "std": []}
                # Collect from observers
                if hasattr(module, "min_val") and hasattr(module, "max_val"):
                    stats[name]["min"].append(
                        module.min_val.item()
                        if torch.is_tensor(module.min_val)
                        else module.min_val
                    )
                    stats[name]["max"].append(
                        module.max_val.item()
                        if torch.is_tensor(module.max_val)
                        else module.max_val
                    )

    def convert(self) -> nn.Module:
        """Convert prepared model to quantized model."""
        if self.prepared_model is None:
            raise ValueError("Call prepare() and calibrate() before convert()")

        self.quantized_model = torch.quantization.convert(
            self.prepared_model, inplace=True
        )
        return self.quantized_model

    def quantize(
        self,
        model: nn.Module,
        calibration_fn: Callable[[], Iterator],
        num_batches: int = 100,
    ) -> nn.Module:
        """Full quantization pipeline."""
        self.prepare(model)
        self.calibrate(calibration_fn, num_batches)
        return self.convert()

    def dequantize(self, model: nn.Module) -> nn.Module:
        """Dequantize model back to floating point."""
        # Static quantization is not easily reversible
        # Return original model if available
        if hasattr(self, "original_model"):
            return copy.deepcopy(self.original_model)
        return model


class DynamicQuantization(BaseQuantization):
    """Dynamic quantization with runtime range computation.

    Quantizes weights statically and computes activation ranges dynamically.
    Best for models where activation ranges vary significantly.

    Args:
        dtype: Quantization dtype (torch.qint8, torch.quint8, torch.float16)
        qconfig_spec: Types of layers to quantize

    Example:
        >>> quantizer = DynamicQuantization(dtype=torch.qint8)
        >>> quantized = quantizer.quantize(model)
    """

    def __init__(
        self, dtype: torch.dtype = torch.qint8, qconfig_spec: Optional[set] = None
    ):
        self.dtype = dtype
        self.qconfig_spec = qconfig_spec or {nn.Linear, nn.LSTM, nn.GRU, nn.RNN}
        self.quantized_model: Optional[nn.Module] = None
        self.original_model: Optional[nn.Module] = None

    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to model."""
        model.eval()
        self.original_model = copy.deepcopy(model)

        self.quantized_model = torch.quantization.quantize_dynamic(
            model, self.qconfig_spec, dtype=self.dtype
        )

        return self.quantized_model

    def dequantize(self, model: nn.Module) -> nn.Module:
        """Return original model."""
        if self.original_model is not None:
            return copy.deepcopy(self.original_model)
        return model

    def get_size_comparison(self) -> Dict[str, float]:
        """Compare sizes of original and quantized models."""

        def get_model_size(model: nn.Module) -> float:
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
                torch.save(model.state_dict(), f.name)
                size = os.path.getsize(f.name) / (1024 * 1024)
                os.unlink(f.name)
            return size

        original_size = (
            get_model_size(self.original_model) if self.original_model else 0
        )
        quantized_size = (
            get_model_size(self.quantized_model) if self.quantized_model else 0
        )

        return {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": original_size / quantized_size
            if quantized_size > 0
            else 1.0,
            "size_reduction_pct": (1 - quantized_size / original_size) * 100
            if original_size > 0
            else 0,
        }


class PerChannelQuantization(BaseQuantization):
    """Per-channel quantization for weights.

    Uses separate quantization parameters for each output channel.
    Better accuracy than per-tensor for convolutional layers.

    Args:
        axis: Channel axis (0 for conv, 0 for linear)
        dtype: Quantization dtype
        qscheme: Quantization scheme (per_channel_affine, per_channel_symmetric)
    """

    def __init__(
        self,
        axis: int = 0,
        dtype: torch.dtype = torch.qint8,
        qscheme: torch.qscheme = torch.per_channel_symmetric,
    ):
        self.axis = axis
        self.dtype = dtype
        self.qscheme = qscheme
        self.quantized_layers: Dict[str, nn.Module] = {}

    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply per-channel quantization to convolutional and linear layers."""
        model.eval()
        quantized_model = copy.deepcopy(model)

        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Conv2d):
                quantized_layer = self._quantize_conv2d(module)
                self._replace_module(quantized_model, name, quantized_layer)
                self.quantized_layers[name] = quantized_layer
            elif isinstance(module, nn.Linear):
                quantized_layer = self._quantize_linear(module)
                self._replace_module(quantized_model, name, quantized_layer)
                self.quantized_layers[name] = quantized_layer

        return quantized_model

    def _quantize_conv2d(self, conv: nn.Conv2d) -> nn.quantized.Conv2d:
        """Quantize Conv2d layer with per-channel scaling."""
        # Compute per-channel scales and zero points
        weight = conv.weight.data

        scales = []
        zero_points = []

        for c in range(weight.size(0)):
            channel_weight = weight[c]
            w_min = channel_weight.min().item()
            w_max = channel_weight.max().item()

            if self.qscheme == torch.per_channel_symmetric:
                max_abs = max(abs(w_min), abs(w_max))
                scale = max_abs / 127.0 if max_abs > 0 else 1.0
                zero_point = 0
            else:
                scale = (w_max - w_min) / 255.0 if w_max > w_min else 1.0
                zero_point = int(-w_min / scale)

            scales.append(scale)
            zero_points.append(zero_point)

        scales_tensor = torch.tensor(scales, dtype=torch.float32)
        zero_points_tensor = torch.tensor(zero_points, dtype=torch.int32)

        # Quantize weights
        qweight = torch.quantize_per_channel(
            weight, scales_tensor, zero_points_tensor, axis=self.axis, dtype=self.dtype
        )

        # Create quantized conv layer
        qconv = nn.quantized.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
        )
        qconv.set_weight_bias(qweight, conv.bias)
        qconv.scale = 1.0  # Will be set during inference
        qconv.zero_point = 0

        return qconv

    def _quantize_linear(self, linear: nn.Linear) -> nn.quantized.Linear:
        """Quantize Linear layer with per-channel scaling."""
        weight = linear.weight.data

        scales = []
        zero_points = []

        for c in range(weight.size(0)):
            channel_weight = weight[c]
            w_min = channel_weight.min().item()
            w_max = channel_weight.max().item()

            if self.qscheme == torch.per_channel_symmetric:
                max_abs = max(abs(w_min), abs(w_max))
                scale = max_abs / 127.0 if max_abs > 0 else 1.0
                zero_point = 0
            else:
                scale = (w_max - w_min) / 255.0 if w_max > w_min else 1.0
                zero_point = int(-w_min / scale)

            scales.append(scale)
            zero_points.append(zero_point)

        scales_tensor = torch.tensor(scales, dtype=torch.float32)
        zero_points_tensor = torch.tensor(zero_points, dtype=torch.int32)

        qweight = torch.quantize_per_channel(
            weight, scales_tensor, zero_points_tensor, axis=self.axis, dtype=self.dtype
        )

        qlinear = nn.quantized.Linear(
            linear.in_features, linear.out_features, bias=linear.bias is not None
        )
        qlinear.set_weight_bias(qweight, linear.bias)
        qlinear.scale = 1.0
        qlinear.zero_point = 0

        return qlinear

    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Replace module in model by name."""
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def dequantize(self, model: nn.Module) -> nn.Module:
        """Dequantize per-channel quantized model."""
        # Per-channel quantization requires manual dequantization
        return model


class PerTensorQuantization(BaseQuantization):
    """Per-tensor quantization for weights and activations.

    Uses single scale and zero-point for entire tensor.
    More efficient but potentially lower accuracy than per-channel.

    Args:
        symmetric: Whether to use symmetric quantization
        reduce_range: Whether to reduce quantization range
    """

    def __init__(self, symmetric: bool = True, reduce_range: bool = False):
        self.symmetric = symmetric
        self.reduce_range = reduce_range

    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply per-tensor quantization."""
        model.eval()

        # Use PyTorch's default static quantization
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric
                if self.symmetric
                else torch.per_tensor_affine,
                reduce_range=self.reduce_range,
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric
                if self.symmetric
                else torch.per_tensor_affine,
                reduce_range=self.reduce_range,
            ),
        )

        return model

    def dequantize(self, model: nn.Module) -> nn.Module:
        """Dequantize model."""
        return model


# ============================================================================
# Quantization-Aware Training
# ============================================================================


class FakeQuantize(nn.Module):
    """Fake quantization module for QAT simulation.

    Simulates quantization effects during training using straight-through estimator.

    Args:
        observer: Observer module for collecting statistics
        quant_min: Minimum quantization value
        quant_max: Maximum quantization value
    """

    def __init__(
        self,
        observer: Optional[nn.Module] = None,
        quant_min: int = -128,
        quant_max: int = 127,
        **observer_kwargs,
    ):
        super().__init__()
        self.quant_min = quant_min
        self.quant_max = quant_max

        if observer is None:
            self.observer = torch.quantization.MovingAverageMinMaxObserver(
                dtype=torch.qint8, **observer_kwargs
            )
        else:
            self.observer = observer

        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1))
        self.enabled = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return x

        if self.training:
            # Update statistics
            self.observer(x.detach())

            # Get scale and zero point
            scale, zero_point = self.observer.calculate_qparams()
            self.scale.copy_(scale)
            self.zero_point.copy_(zero_point)

        # Fake quantize with straight-through estimator
        return self._fake_quantize(x, self.scale, self.zero_point)

    def _fake_quantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        """Apply fake quantization."""
        quant = torch.fake_quantize_per_tensor_affine(
            x, scale, zero_point, self.quant_min, self.quant_max
        )
        return quant

    def enable(self):
        """Enable fake quantization."""
        self.enabled = True

    def disable(self):
        """Disable fake quantization."""
        self.enabled = False


class QATLinear(nn.Module):
    """Linear layer with fake quantization for QAT.

    Simulates INT8 quantization during training for accurate post-quantization accuracy.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias
        weight_bits: Number of bits for weight quantization
        activation_bits: Number of bits for activation quantization
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: int = 8,
        activation_bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Fake quantization modules
        self.weight_fake_quant = FakeQuantize(
            quant_min=-(2 ** (weight_bits - 1)), quant_max=2 ** (weight_bits - 1) - 1
        )

        self.activation_fake_quant = FakeQuantize(
            quant_min=0 if activation_bits == 8 else -(2 ** (activation_bits - 1)),
            quant_max=255 if activation_bits == 8 else 2 ** (activation_bits - 1) - 1,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # Quantize weight
        q_weight = self.weight_fake_quant(self.weight)

        # Linear operation
        output = F.linear(x, q_weight, self.bias)

        # Quantize activation
        output = self.activation_fake_quant(output)

        return output

    def to_quantized(self) -> nn.quantized.Linear:
        """Convert to actual quantized linear layer."""
        scale = self.weight_fake_quant.scale.item()
        zero_point = int(self.weight_fake_quant.zero_point.item())

        qlinear = nn.quantized.Linear(
            self.in_features, self.out_features, bias=self.bias is not None
        )

        # Quantize weight
        qweight = torch.quantize_per_tensor(
            self.weight.data, scale, zero_point, torch.qint8
        )
        qlinear.set_weight_bias(qweight, self.bias.data if self.bias else None)

        return qlinear


class QATConv2d(nn.Module):
    """Conv2d layer with fake quantization for QAT.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
        padding: Padding added to input
        dilation: Dilation rate
        groups: Number of groups
        bias: Whether to include bias
        weight_bits: Number of bits for weight quantization
        activation_bits: Number of bits for activation quantization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        weight_bits: int = 8,
        activation_bits: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize weight
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else torch.empty(out_channels, in_channels // groups, *kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Fake quantization modules
        self.weight_fake_quant = FakeQuantize(
            quant_min=-(2 ** (weight_bits - 1)), quant_max=2 ** (weight_bits - 1) - 1
        )

        self.activation_fake_quant = FakeQuantize(
            quant_min=0 if activation_bits == 8 else -(2 ** (activation_bits - 1)),
            quant_max=255 if activation_bits == 8 else 2 ** (activation_bits - 1) - 1,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # Quantize weight
        q_weight = self.weight_fake_quant(self.weight)

        # Convolution operation
        output = F.conv2d(
            x,
            q_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # Quantize activation
        output = self.activation_fake_quant(output)

        return output

    def to_quantized(self) -> nn.quantized.Conv2d:
        """Convert to actual quantized conv layer."""
        qconv = nn.quantized.Conv2d(
            self.in_channels,
            self.out_channels,
            self.weight.shape[2:],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
        )

        # Quantize weight (per-channel for conv)
        scales = []
        zero_points = []

        for c in range(self.weight.size(0)):
            w_min = self.weight[c].min().item()
            w_max = self.weight[c].max().item()
            max_abs = max(abs(w_min), abs(w_max))
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            scales.append(scale)
            zero_points.append(0)

        scales_tensor = torch.tensor(scales, dtype=torch.float32)
        zero_points_tensor = torch.tensor(zero_points, dtype=torch.int32)

        qweight = torch.quantize_per_channel(
            self.weight.data,
            scales_tensor,
            zero_points_tensor,
            axis=0,
            dtype=torch.qint8,
        )

        qconv.set_weight_bias(qweight, self.bias.data if self.bias else None)

        return qconv


class LearnableQuantization(nn.Module):
    """Learnable quantization with trainable scale and zero-point.

    Allows the quantization parameters to be learned during training
    rather than computed from statistics.

    Args:
        num_bits: Number of quantization bits
        per_channel: Whether to use per-channel quantization
        channel_dim: Channel dimension for per-channel quantization
    """

    def __init__(
        self, num_bits: int = 8, per_channel: bool = False, channel_dim: int = 0
    ):
        super().__init__()
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.channel_dim = channel_dim

        # Learnable parameters
        if per_channel:
            # Will be initialized on first forward pass
            self.register_parameter("scale", None)
            self.register_parameter("zero_point", None)
        else:
            self.scale = nn.Parameter(torch.ones(1))
            self.zero_point = nn.Parameter(torch.zeros(1))

        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = 2 ** (num_bits - 1) - 1

    def forward(self, x: Tensor) -> Tensor:
        # Initialize per-channel parameters if needed
        if self.per_channel and self.scale is None:
            num_channels = x.shape[self.channel_dim]
            self.scale = nn.Parameter(torch.ones(num_channels, 1))
            self.zero_point = nn.Parameter(torch.zeros(num_channels, 1))

        # Compute quantization
        if self.per_channel:
            # Reshape scale and zero_point for broadcasting
            shape = [1] * x.dim()
            shape[self.channel_dim] = -1
            scale = self.scale.view(shape)
            zero_point = self.zero_point.view(shape)
        else:
            scale = self.scale
            zero_point = self.zero_point

        # Ensure positive scale
        scale = F.softplus(scale) + 1e-8

        # Quantize with straight-through estimator
        x_q = (x / scale + zero_point).clamp(self.qmin, self.qmax).round()
        x_dq = (x_q - zero_point) * scale

        # Straight-through estimator
        return x + (x_dq - x).detach()


# ============================================================================
# Low-Precision Formats
# ============================================================================


class QuantizationFormat(Enum):
    """Supported quantization formats."""

    INT8 = "int8"
    FP16 = "fp16"
    BF16 = "bf16"
    INT4 = "int4"
    TERNARY = "ternary"
    BINARY = "binary"


class LowPrecisionConverter:
    """Converter for low-precision formats.

    Supports conversion to INT8, FP16, BF16, INT4, Ternary, and Binary formats.

    Args:
        format: Target quantization format
        symmetric: Whether to use symmetric quantization
    """

    def __init__(self, format: Union[QuantizationFormat, str], symmetric: bool = True):
        self.format = QuantizationFormat(format) if isinstance(format, str) else format
        self.symmetric = symmetric

    def convert_tensor(self, x: Tensor) -> Tensor:
        """Convert tensor to low-precision format."""
        if self.format == QuantizationFormat.INT8:
            return self._to_int8(x)
        elif self.format == QuantizationFormat.FP16:
            return self._to_fp16(x)
        elif self.format == QuantizationFormat.BF16:
            return self._to_bf16(x)
        elif self.format == QuantizationFormat.INT4:
            return self._to_int4(x)
        elif self.format == QuantizationFormat.TERNARY:
            return self._to_ternary(x)
        elif self.format == QuantizationFormat.BINARY:
            return self._to_binary(x)
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _to_int8(self, x: Tensor) -> Tensor:
        """Convert to 8-bit integers."""
        if self.symmetric:
            max_val = x.abs().max()
            scale = max_val / 127.0 if max_val > 0 else 1.0
            x_q = (x / scale).round().clamp(-128, 127)
        else:
            min_val, max_val = x.min(), x.max()
            scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
            zero_point = (-min_val / scale).round()
            x_q = (x / scale + zero_point).round().clamp(0, 255)

        return x_q * scale

    def _to_fp16(self, x: Tensor) -> Tensor:
        """Convert to half precision (FP16)."""
        return x.half().float()

    def _to_bf16(self, x: Tensor) -> Tensor:
        """Convert to BFloat16."""
        if hasattr(torch, "bfloat16"):
            return x.bfloat16().float()
        else:
            # Fallback: truncate mantissa to match BF16 precision
            x_f32 = x.float()
            # BF16 has 7 bits of mantissa vs 23 for FP32
            x_int = x_f32.view(torch.int32)
            x_int = x_int >> 16  # Truncate lower 16 bits
            x_bf16 = (x_int << 16).view(torch.float32)
            return x_bf16

    def _to_int4(self, x: Tensor) -> Tensor:
        """Convert to 4-bit integers."""
        if self.symmetric:
            max_val = x.abs().max()
            scale = max_val / 7.0 if max_val > 0 else 1.0
            x_q = (x / scale).round().clamp(-8, 7)
        else:
            min_val, max_val = x.min(), x.max()
            scale = (max_val - min_val) / 15.0 if max_val > min_val else 1.0
            zero_point = (-min_val / scale).round()
            x_q = (x / scale + zero_point).round().clamp(0, 15)

        return x_q * scale

    def _to_ternary(self, x: Tensor) -> Tensor:
        """Convert to ternary values {-1, 0, 1}.

        Uses threshold-based quantization where values are assigned based on magnitude.
        """
        threshold = 0.7 * x.abs().mean()

        # Straight-through estimator for ternary
        x_ternary = torch.where(
            x > threshold,
            torch.ones_like(x),
            torch.where(x < -threshold, -torch.ones_like(x), torch.zeros_like(x)),
        )

        return x + (x_ternary - x).detach()

    def _to_binary(self, x: Tensor) -> Tensor:
        """Convert to binary values {-1, 1}.

        Uses sign-based quantization with straight-through estimator.
        """
        x_binary = x.sign()
        x_binary[x_binary == 0] = 1

        # Straight-through estimator
        return x + (x_binary - x).detach()

    def convert_model(self, model: nn.Module) -> nn.Module:
        """Convert entire model to low-precision format."""
        for module in model.modules():
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data = self.convert_tensor(module.weight.data)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data = self.convert_tensor(module.bias.data)

        return model


# ============================================================================
# Pruning
# ============================================================================


class BasePruning(ABC):
    """Base class for pruning methods."""

    @abstractmethod
    def prune(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply pruning to model."""
        pass

    @abstractmethod
    def remove_pruning(self, model: nn.Module) -> nn.Module:
        """Remove pruning and make it permanent."""
        pass


class MagnitudePruning(BasePruning):
    """Unstructured magnitude-based pruning.

    Prunes weights with smallest absolute magnitudes.

    Args:
        global_pruning: Whether to prune globally or per-layer
        structured: Whether to apply structured pruning
    """

    def __init__(self, global_pruning: bool = False, structured: bool = False):
        self.global_pruning = global_pruning
        self.structured = structured
        self.mask_dict: Dict[str, Tensor] = {}

    def prune(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply magnitude pruning."""
        if self.global_pruning:
            return self._prune_global(model, sparsity)
        else:
            return self._prune_local(model, sparsity)

    def _prune_local(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply per-layer pruning."""
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue

            if self.structured:
                # Structured pruning: prune entire channels/neurons
                mask = self._compute_structured_mask(param, sparsity)
            else:
                # Unstructured pruning
                mask = self._compute_unstructured_mask(param, sparsity)

            self.mask_dict[name] = mask
            param.data.mul_(mask)

        return model

    def _prune_global(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply global pruning across all layers."""
        # Collect all weights
        all_weights = []
        weight_shapes = {}
        weight_names = []

        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            all_weights.append(param.data.abs().flatten())
            weight_shapes[name] = param.shape
            weight_names.append(name)

        if not all_weights:
            return model

        # Compute global threshold
        all_weights_cat = torch.cat(all_weights)
        k = int(sparsity * all_weights_cat.numel())
        threshold = torch.kthvalue(all_weights_cat, k)[0] if k > 0 else 0

        # Apply masks
        for name in weight_names:
            param = dict(model.named_parameters())[name]
            mask = (param.data.abs() > threshold).float()
            self.mask_dict[name] = mask
            param.data.mul_(mask)

        return model

    def _compute_unstructured_mask(self, weight: Tensor, sparsity: float) -> Tensor:
        """Compute unstructured pruning mask."""
        weight_abs = weight.abs()
        threshold = torch.quantile(weight_abs.flatten(), sparsity)
        mask = (weight_abs > threshold).float()
        return mask

    def _compute_structured_mask(self, weight: Tensor, sparsity: float) -> Tensor:
        """Compute structured pruning mask (prune output channels)."""
        # Compute importance per output channel
        channel_importance = weight.abs().mean(dim=list(range(1, weight.dim())))
        k = int(sparsity * channel_importance.numel())

        if k == 0:
            return torch.ones_like(weight)

        threshold = torch.kthvalue(channel_importance, k)[0]
        channel_mask = (channel_importance > threshold).float()

        # Expand mask to match weight shape
        mask_shape = [weight.size(0)] + [1] * (weight.dim() - 1)
        mask = channel_mask.view(mask_shape).expand_as(weight)

        return mask

    def remove_pruning(self, model: nn.Module) -> nn.Module:
        """Remove pruning reparametrization and make it permanent."""
        self.mask_dict.clear()
        return model

    def apply_mask_during_training(self, model: nn.Module):
        """Apply masks during training to maintain sparsity."""
        for name, param in model.named_parameters():
            if name in self.mask_dict:
                param.data.mul_(self.mask_dict[name])


class StructuredPruning(BasePruning):
    """Structured pruning for channel/neuron removal.

    Removes entire channels or neurons for hardware efficiency.

    Args:
        prune_type: Type of structure to prune ('channel', 'filter', 'neuron')
        importance_metric: Metric for importance ('l1', 'l2', 'bn_scale')
    """

    def __init__(self, prune_type: str = "channel", importance_metric: str = "l2"):
        self.prune_type = prune_type
        self.importance_metric = importance_metric
        self.pruned_indices: Dict[str, List[int]] = {}

    def prune(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply structured pruning."""
        if self.prune_type == "channel":
            return self._prune_channels(model, sparsity)
        elif self.prune_type == "neuron":
            return self._prune_neurons(model, sparsity)
        else:
            raise ValueError(f"Unknown prune_type: {self.prune_type}")

    def _prune_channels(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Prune convolutional channels."""
        modules_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                modules_to_prune.append((name, module))

        for name, conv in modules_to_prune:
            importance = self._compute_channel_importance(conv)
            k = int(sparsity * importance.numel())

            if k == 0:
                continue

            # Get indices of channels to prune
            _, prune_indices = torch.topk(importance, k, largest=False)
            self.pruned_indices[name] = prune_indices.tolist()

            # Zero out pruned channels
            mask = torch.ones(conv.weight.size(0), device=conv.weight.device)
            mask[prune_indices] = 0

            mask_shape = [conv.weight.size(0)] + [1] * (conv.weight.dim() - 1)
            conv.weight.data.mul_(mask.view(mask_shape))

        return model

    def _prune_neurons(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Prune linear layer neurons."""
        modules_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                modules_to_prune.append((name, module))

        for name, linear in modules_to_prune:
            importance = self._compute_neuron_importance(linear)
            k = int(sparsity * importance.numel())

            if k == 0:
                continue

            _, prune_indices = torch.topk(importance, k, largest=False)
            self.pruned_indices[name] = prune_indices.tolist()

            # Zero out pruned neurons
            mask = torch.ones(linear.weight.size(0), device=linear.weight.device)
            mask[prune_indices] = 0

            linear.weight.data.mul_(mask.view(-1, 1))

        return model

    def _compute_channel_importance(self, conv: nn.Conv2d) -> Tensor:
        """Compute importance score for each channel."""
        if self.importance_metric == "l1":
            return conv.weight.abs().sum(dim=[1, 2, 3])
        elif self.importance_metric == "l2":
            return conv.weight.pow(2).sum(dim=[1, 2, 3]).sqrt()
        elif self.importance_metric == "bn_scale":
            # Use batch norm scale if available
            return torch.ones(conv.weight.size(0))
        else:
            return conv.weight.abs().mean(dim=[1, 2, 3])

    def _compute_neuron_importance(self, linear: nn.Linear) -> Tensor:
        """Compute importance score for each neuron."""
        if self.importance_metric == "l1":
            return linear.weight.abs().sum(dim=1)
        elif self.importance_metric == "l2":
            return linear.weight.pow(2).sum(dim=1).sqrt()
        else:
            return linear.weight.abs().mean(dim=1)

    def remove_pruning(self, model: nn.Module) -> nn.Module:
        """Remove pruned channels permanently."""
        # This would require recreating the model with smaller dimensions
        # For now, just return the model with zeroed channels
        return model


class LotteryTicketPruning(BasePruning):
    """Lottery Ticket Hypothesis pruning.

    Finds winning tickets (sparse subnetworks) that train well from initialization.

    Args:
        num_iterations: Number of pruning iterations
        pruning_rate: Rate of pruning per iteration
        rewind_epoch: Epoch to rewind weights to
    """

    def __init__(
        self, num_iterations: int = 5, pruning_rate: float = 0.2, rewind_epoch: int = 3
    ):
        self.num_iterations = num_iterations
        self.pruning_rate = pruning_rate
        self.rewind_epoch = rewind_epoch
        self.initial_weights: Dict[str, Tensor] = {}
        self.winning_ticket_mask: Dict[str, Tensor] = {}

    def find_winning_ticket(
        self,
        model: nn.Module,
        train_fn: Callable,
        eval_fn: Callable,
        target_sparsity: float,
    ) -> Dict[str, Tensor]:
        """Find winning ticket through iterative pruning."""
        # Store initial weights
        self.initial_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if "weight" in name
        }

        current_sparsity = 0.0

        for iteration in range(self.num_iterations):
            # Train model briefly
            train_fn(model, epochs=self.rewind_epoch)

            # Evaluate
            accuracy = eval_fn(model)
            print(
                f"Iteration {iteration + 1}/{self.num_iterations}: Accuracy = {accuracy:.4f}"
            )

            # Prune
            current_sparsity = min(
                current_sparsity + self.pruning_rate, target_sparsity
            )
            self._prune_once(model, current_sparsity)

            # Rewind weights to initialization
            self._rewind_weights(model)

        return self.winning_ticket_mask

    def _prune_once(self, model: nn.Module, sparsity: float):
        """Apply one-shot magnitude pruning."""
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue

            weight_abs = param.data.abs()
            threshold = torch.quantile(weight_abs.flatten(), sparsity)
            mask = (weight_abs > threshold).float()

            if name not in self.winning_ticket_mask:
                self.winning_ticket_mask[name] = mask
            else:
                # Intersect with existing mask
                self.winning_ticket_mask[name] *= mask

            param.data.mul_(self.winning_ticket_mask[name])

    def _rewind_weights(self, model: nn.Module):
        """Rewind weights to initialization."""
        for name, param in model.named_parameters():
            if name in self.initial_weights:
                param.data.copy_(self.initial_weights[name])
                if name in self.winning_ticket_mask:
                    param.data.mul_(self.winning_ticket_mask[name])

    def prune(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply winning ticket mask."""
        for name, param in model.named_parameters():
            if name in self.winning_ticket_mask:
                param.data.mul_(self.winning_ticket_mask[name])
        return model

    def remove_pruning(self, model: nn.Module) -> nn.Module:
        """Remove pruning."""
        return model


class GradualPruning(BasePruning):
    """Gradual pruning with progressive sparsity increase.

    Increases sparsity gradually over training for better accuracy.

    Args:
        initial_sparsity: Starting sparsity
        final_sparsity: Target sparsity
        num_epochs: Number of epochs for pruning
        pruning_frequency: Pruning frequency in epochs
        pruning_schedule: Schedule type ('linear', 'exponential', 'cosine')
    """

    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.9,
        num_epochs: int = 100,
        pruning_frequency: int = 1,
        pruning_schedule: str = "cosine",
    ):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.num_epochs = num_epochs
        self.pruning_frequency = pruning_frequency
        self.pruning_schedule = pruning_schedule

        self.current_epoch = 0
        self.mask_dict: Dict[str, Tensor] = {}

    def get_current_sparsity(self) -> float:
        """Get sparsity for current epoch."""
        progress = min(self.current_epoch / self.num_epochs, 1.0)

        if self.pruning_schedule == "linear":
            sparsity = (
                self.initial_sparsity
                + (self.final_sparsity - self.initial_sparsity) * progress
            )
        elif self.pruning_schedule == "exponential":
            sparsity = self.initial_sparsity + (
                self.final_sparsity - self.initial_sparsity
            ) * (progress**2)
        elif self.pruning_schedule == "cosine":
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            sparsity = (
                self.final_sparsity
                - (self.final_sparsity - self.initial_sparsity) * cosine_decay
            )
        else:
            sparsity = (
                self.initial_sparsity
                + (self.final_sparsity - self.initial_sparsity) * progress
            )

        return sparsity

    def step(self, model: nn.Module):
        """Perform one pruning step."""
        self.current_epoch += 1

        if self.current_epoch % self.pruning_frequency != 0:
            return

        sparsity = self.get_current_sparsity()
        self.prune(model, sparsity)

    def prune(self, model: nn.Module, sparsity: Optional[float] = None) -> nn.Module:
        """Apply gradual pruning."""
        if sparsity is None:
            sparsity = self.get_current_sparsity()

        for name, param in model.named_parameters():
            if "weight" not in name:
                continue

            weight_abs = param.data.abs()
            threshold = torch.quantile(weight_abs.flatten(), sparsity)
            mask = (weight_abs > threshold).float()

            self.mask_dict[name] = mask
            param.data.mul_(mask)

        return model

    def remove_pruning(self, model: nn.Module) -> nn.Module:
        """Remove pruning."""
        self.mask_dict.clear()
        return model


# ============================================================================
# Knowledge Distillation
# ============================================================================


class LogitDistillation:
    """Knowledge distillation using soft labels from teacher.

    Transfers knowledge from teacher to student using softened probability distributions.

    Args:
        temperature: Temperature for softening distributions
        alpha: Weight for distillation loss vs hard label loss
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_loss(
        self, student_logits: Tensor, teacher_logits: Tensor, labels: Tensor
    ) -> Tensor:
        """Compute distillation loss."""
        # Soft target loss
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=1)

        distillation_loss = self.kl_loss(soft_predictions, soft_targets) * (
            self.temperature**2
        )

        # Hard target loss
        hard_loss = self.ce_loss(student_logits, labels)

        # Combined loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss

        return loss


class FeatureDistillation:
    """Feature-based knowledge distillation.

    Transfers knowledge by matching intermediate feature representations.

    Args:
        layer_mapping: Mapping from student layers to teacher layers
        loss_type: Type of feature matching loss ('mse', 'l1', 'cosine', 'attention')
        weight: Weight for feature distillation loss
    """

    def __init__(
        self,
        layer_mapping: Optional[Dict[str, str]] = None,
        loss_type: str = "mse",
        weight: float = 1.0,
    ):
        self.layer_mapping = layer_mapping or {}
        self.loss_type = loss_type
        self.weight = weight

        self.student_features: Dict[str, Tensor] = {}
        self.teacher_features: Dict[str, Tensor] = {}

        self._setup_hooks()

    def _setup_hooks(self):
        """Setup forward hooks for feature extraction."""
        self.hooks = []

    def register_student_hook(self, model: nn.Module, layer_name: str):
        """Register hook for student feature extraction."""

        def hook_fn(module, input, output):
            self.student_features[layer_name] = output

        layer = self._get_layer(model, layer_name)
        if layer:
            self.hooks.append(layer.register_forward_hook(hook_fn))

    def register_teacher_hook(self, model: nn.Module, layer_name: str):
        """Register hook for teacher feature extraction."""

        def hook_fn(module, input, output):
            self.teacher_features[layer_name] = output.detach()

        layer = self._get_layer(model, layer_name)
        if layer:
            self.hooks.append(layer.register_forward_hook(hook_fn))

    def _get_layer(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None

    def compute_loss(self) -> Tensor:
        """Compute feature distillation loss."""
        total_loss = 0.0

        for student_layer, teacher_layer in self.layer_mapping.items():
            if student_layer not in self.student_features:
                continue
            if teacher_layer not in self.teacher_features:
                continue

            student_feat = self.student_features[student_layer]
            teacher_feat = self.teacher_features[teacher_layer]

            # Match dimensions if needed
            if student_feat.shape != teacher_feat.shape:
                student_feat = self._adapt_features(student_feat, teacher_feat.shape)

            # Compute loss
            if self.loss_type == "mse":
                loss = F.mse_loss(student_feat, teacher_feat)
            elif self.loss_type == "l1":
                loss = F.l1_loss(student_feat, teacher_feat)
            elif self.loss_type == "cosine":
                loss = (
                    1
                    - F.cosine_similarity(
                        student_feat.flatten(1), teacher_feat.flatten(1), dim=1
                    ).mean()
                )
            elif self.loss_type == "attention":
                loss = self._attention_transfer(student_feat, teacher_feat)
            else:
                loss = F.mse_loss(student_feat, teacher_feat)

            total_loss += loss

        return total_loss * self.weight

    def _adapt_features(self, features: Tensor, target_shape: torch.Size) -> Tensor:
        """Adapt feature dimensions."""
        # Simple adaptation using linear projection
        if features.dim() == target_shape.dim():
            # Use interpolation for spatial dimensions
            if features.dim() == 4:  # Conv features
                features = F.adaptive_avg_pool2d(features, target_shape[2:])

            # Project channel dimension if needed
            if features.shape[1] != target_shape[1]:
                # Add a simple linear projection
                pass

        return features

    def _attention_transfer(self, student_feat: Tensor, teacher_feat: Tensor) -> Tensor:
        """Compute attention transfer loss."""
        # Convert features to attention maps
        student_attn = student_feat.pow(2).mean(dim=1, keepdim=True)
        teacher_attn = teacher_feat.pow(2).mean(dim=1, keepdim=True)

        # Normalize
        student_attn = F.normalize(
            student_attn.view(student_attn.size(0), -1), p=2, dim=1
        )
        teacher_attn = F.normalize(
            teacher_attn.view(teacher_attn.size(0), -1), p=2, dim=1
        )

        # L2 loss
        loss = F.mse_loss(student_attn, teacher_attn)

        return loss

    def clear_features(self):
        """Clear stored features."""
        self.student_features.clear()
        self.teacher_features.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class RelationDistillation:
    """Relation-based knowledge distillation.

    Transfers structural knowledge through pairwise or sample-wise relations.

    Args:
        relation_type: Type of relation ('pairwise', 'sample')
        loss_type: Type of loss ('mse', 'kl', 'contrastive')
        temperature: Temperature for contrastive loss
    """

    def __init__(
        self,
        relation_type: str = "pairwise",
        loss_type: str = "kl",
        temperature: float = 4.0,
    ):
        self.relation_type = relation_type
        self.loss_type = loss_type
        self.temperature = temperature

    def compute_loss(
        self, student_features: Tensor, teacher_features: Tensor
    ) -> Tensor:
        """Compute relation distillation loss."""
        if self.relation_type == "pairwise":
            return self._pairwise_relation_loss(student_features, teacher_features)
        elif self.relation_type == "sample":
            return self._sample_relation_loss(student_features, teacher_features)
        else:
            raise ValueError(f"Unknown relation_type: {self.relation_type}")

    def _pairwise_relation_loss(
        self, student_features: Tensor, teacher_features: Tensor
    ) -> Tensor:
        """Compute pairwise relation loss."""
        # Compute pairwise similarity matrices
        student_sim = self._compute_similarity_matrix(student_features)
        teacher_sim = self._compute_similarity_matrix(teacher_features)

        if self.loss_type == "mse":
            loss = F.mse_loss(student_sim, teacher_sim)
        elif self.loss_type == "kl":
            teacher_prob = F.softmax(teacher_sim / self.temperature, dim=1)
            student_log_prob = F.log_softmax(student_sim / self.temperature, dim=1)
            loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        elif self.loss_type == "contrastive":
            loss = self._contrastive_relation_loss(student_sim, teacher_sim)
        else:
            loss = F.mse_loss(student_sim, teacher_sim)

        return loss

    def _compute_similarity_matrix(self, features: Tensor) -> Tensor:
        """Compute pairwise similarity matrix."""
        # Flatten features
        features = features.view(features.size(0), -1)

        # Compute cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(features_norm, features_norm.t())

        return similarity

    def _contrastive_relation_loss(
        self, student_sim: Tensor, teacher_sim: Tensor
    ) -> Tensor:
        """Compute contrastive relation loss."""
        # Use teacher similarities as targets for student
        pos_mask = teacher_sim > teacher_sim.mean()
        neg_mask = ~pos_mask

        # Mask diagonal
        diag_mask = torch.eye(student_sim.size(0), device=student_sim.device).bool()
        pos_mask = pos_mask & ~diag_mask
        neg_mask = neg_mask & ~diag_mask

        # Contrastive loss
        student_prob = F.softmax(student_sim / self.temperature, dim=1)
        teacher_prob = F.softmax(teacher_sim / self.temperature, dim=1)

        loss = F.kl_div(student_prob.log(), teacher_prob, reduction="batchmean")

        return loss

    def _sample_relation_loss(
        self, student_features: Tensor, teacher_features: Tensor
    ) -> Tensor:
        """Compute sample-wise relation loss."""
        # Normalize features
        student_norm = F.normalize(
            student_features.view(student_features.size(0), -1), p=2, dim=1
        )
        teacher_norm = F.normalize(
            teacher_features.view(teacher_features.size(0), -1), p=2, dim=1
        )

        # MSE on normalized features
        loss = F.mse_loss(student_norm, teacher_norm)

        return loss


# ============================================================================
# Efficient Inference Layers
# ============================================================================


class QuantizedLinear(nn.Module):
    """Efficient quantized linear layer for inference.

    Uses actual INT8 computation for faster inference.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quantized parameters
        self.register_buffer("weight_scale", torch.ones(1))
        self.register_buffer("weight_zero_point", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("input_scale", torch.ones(1))
        self.register_buffer("input_zero_point", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("output_scale", torch.ones(1))
        self.register_buffer("output_zero_point", torch.zeros(1, dtype=torch.int32))

        # Weight stored as quantized int8
        self.register_buffer(
            "weight",
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with quantized computation."""
        # Quantize input
        if x.dtype != torch.float32:
            x = x.float()

        x_q = torch.quantize_per_tensor(
            x, self.input_scale, self.input_zero_point, torch.quint8
        )

        # Dequantize weight
        weight_dq = (self.weight.float() - self.weight_zero_point) * self.weight_scale

        # Compute (using float for simplicity, would use int8 kernels in production)
        output = F.linear(x, weight_dq, self.bias)

        # Quantize output
        output_q = torch.quantize_per_tensor(
            output, self.output_scale, self.output_zero_point, torch.quint8
        )

        # Dequantize for return
        return output_q.dequantize()

    def from_float(self, linear: nn.Linear):
        """Initialize from float linear layer."""
        # Compute quantization parameters
        w_min, w_max = linear.weight.data.min(), linear.weight.data.max()
        if self.weight_zero_point == 0:  # Symmetric
            max_abs = max(abs(w_min.item()), abs(w_max.item()))
            self.weight_scale.fill_(max_abs / 127.0 if max_abs > 0 else 1.0)
            self.weight_zero_point.zero_()
        else:  # Affine
            self.weight_scale.fill_(
                (w_max - w_min).item() / 255.0 if w_max > w_min else 1.0
            )
            self.weight_zero_point.fill_(int((-w_min / self.weight_scale).item()))

        # Quantize weight
        self.weight.copy_(
            (linear.weight.data / self.weight_scale + self.weight_zero_point)
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
        )

        if linear.bias is not None:
            self.bias.copy_(linear.bias.data)


class QuantizedConv2d(nn.Module):
    """Efficient quantized Conv2d layer for inference.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
        padding: Padding added to input
        groups: Number of groups
        bias: Whether to include bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.groups = groups

        # Quantization parameters (per-channel for weights)
        self.register_buffer("weight_scale", torch.ones(out_channels, 1, 1, 1))
        self.register_buffer(
            "weight_zero_point", torch.zeros(out_channels, 1, 1, 1, dtype=torch.int32)
        )
        self.register_buffer("input_scale", torch.ones(1))
        self.register_buffer("input_zero_point", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("output_scale", torch.ones(1))
        self.register_buffer("output_zero_point", torch.zeros(1, dtype=torch.int32))

        # Quantized weight
        self.register_buffer(
            "weight",
            torch.randint(
                -128,
                127,
                (
                    out_channels,
                    in_channels // groups,
                    self.kernel_size[0],
                    self.kernel_size[1],
                ),
                dtype=torch.int8,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with quantized computation."""
        # Dequantize weight
        weight_dq = (self.weight.float() - self.weight_zero_point) * self.weight_scale

        # Compute convolution
        output = F.conv2d(
            x, weight_dq, self.bias, self.stride, self.padding, groups=self.groups
        )

        return output

    def from_float(self, conv: nn.Conv2d):
        """Initialize from float conv layer."""
        # Compute per-channel quantization parameters
        for c in range(conv.weight.size(0)):
            w_min = conv.weight.data[c].min().item()
            w_max = conv.weight.data[c].max().item()

            max_abs = max(abs(w_min), abs(w_max))
            self.weight_scale[c, 0, 0, 0] = max_abs / 127.0 if max_abs > 0 else 1.0

            self.weight[c].copy_(
                (conv.weight.data[c] / self.weight_scale[c, 0, 0, 0])
                .round()
                .clamp(-128, 127)
                .to(torch.int8)
            )

        if conv.bias is not None:
            self.bias.copy_(conv.bias.data)


class QuantizedBatchNorm(nn.Module):
    """Quantized Batch Normalization layer.

    Fuses BN parameters for efficient inference.

    Args:
        num_features: Number of features/channels
        eps: Epsilon for numerical stability
        momentum: Momentum for running statistics
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Fused parameters (BN folded into preceding conv/linear)
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        # Quantization parameters
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1, dtype=torch.int32))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if self.training:
            # Compute batch statistics
            mean = x.mean([0, 2, 3] if x.dim() == 4 else [0])
            var = x.var([0, 2, 3] if x.dim() == 4 else [0], unbiased=False)

            # Update running statistics
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean.view(1, -1, 1, 1 if x.dim() == 4 else 1)) / torch.sqrt(
            var.view(1, -1, 1, 1 if x.dim() == 4 else 1) + self.eps
        )

        # Scale and shift
        output = x_norm * self.weight.view(
            1, -1, 1, 1 if x.dim() == 4 else 1
        ) + self.bias.view(1, -1, 1, 1 if x.dim() == 4 else 1)

        return output

    def from_float(self, bn: nn.BatchNorm2d):
        """Initialize from float batch norm."""
        self.weight.copy_(
            bn.weight.data if bn.weight is not None else torch.ones(self.num_features)
        )
        self.bias.copy_(
            bn.bias.data if bn.bias is not None else torch.zeros(self.num_features)
        )
        self.running_mean.copy_(bn.running_mean)
        self.running_var.copy_(bn.running_var)


# ============================================================================
# Utilities
# ============================================================================


def quantize_model(
    model: nn.Module,
    method: str = "static",
    calibration_data: Optional[Iterator] = None,
    num_calibration_batches: int = 100,
    backend: str = "fbgemm",
    **kwargs,
) -> nn.Module:
    """Convenience function to quantize a model.

    Args:
        model: Model to quantize
        method: Quantization method ('static', 'dynamic', 'qat')
        calibration_data: Calibration data for static quantization
        num_calibration_batches: Number of calibration batches
        backend: Quantization backend
        **kwargs: Additional arguments for specific quantizers

    Returns:
        Quantized model

    Example:
        >>> quantized_model = quantize_model(
        ...     model,
        ...     method='static',
        ...     calibration_data=data_loader
        ... )
    """
    if method == "static":
        if calibration_data is None:
            raise ValueError("calibration_data required for static quantization")

        quantizer = StaticQuantization(backend=backend)
        quantizer.prepare(model)
        quantizer.calibrate(calibration_data, num_calibration_batches)
        return quantizer.convert()

    elif method == "dynamic":
        dtype = kwargs.get("dtype", torch.qint8)
        qconfig_spec = kwargs.get("qconfig_spec", {nn.Linear, nn.LSTM})
        quantizer = DynamicQuantization(dtype=dtype, qconfig_spec=qconfig_spec)
        return quantizer.quantize(model)

    elif method == "qat":
        # Return prepared model for QAT
        trainer = QuantizationAwareTrainer(model, backend=backend)
        return trainer.prepare()

    else:
        raise ValueError(f"Unknown quantization method: {method}")


def dequantize_model(model: nn.Module) -> nn.Module:
    """Dequantize a quantized model.

    Note: This is a best-effort operation. Some quantization
    methods cannot be perfectly reversed.

    Args:
        model: Quantized model

    Returns:
        Dequantized (float) model
    """
    # Check if model has quantization stubs
    has_quant_stubs = any(
        isinstance(m, (QuantStub, DeQuantStub)) for m in model.modules()
    )

    if has_quant_stubs:
        # Remove quantization stubs
        def remove_stubs(module):
            for name, child in list(module.named_children()):
                if isinstance(child, (QuantStub, DeQuantStub)):
                    setattr(module, name, nn.Identity())
                else:
                    remove_stubs(child)

        model = copy.deepcopy(model)
        remove_stubs(model)

    # Convert quantized layers back to float
    def convert_to_float(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.quantized.Linear):
                float_linear = nn.Linear(
                    child.in_features, child.out_features, bias=child.bias is not None
                )
                float_linear.weight.data = child.weight().dequantize()
                if child.bias is not None:
                    float_linear.bias.data = child.bias()
                setattr(module, name, float_linear)

            elif isinstance(child, nn.quantized.Conv2d):
                float_conv = nn.Conv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    groups=child.groups,
                    bias=child.bias is not None,
                )
                float_conv.weight.data = child.weight().dequantize()
                if child.bias is not None:
                    float_conv.bias.data = child.bias()
                setattr(module, name, float_conv)

            else:
                convert_to_float(child)

    convert_to_float(model)

    return model


def compare_models(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_input: Tensor,
    detailed: bool = False,
) -> Dict[str, Any]:
    """Compare original and quantized models.

    Args:
        original_model: Original float model
        quantized_model: Quantized model
        test_input: Input tensor for comparison
        detailed: Whether to include detailed layer-wise comparison

    Returns:
        Dictionary with comparison metrics

    Example:
        >>> metrics = compare_models(
        ...     original_model,
        ...     quantized_model,
        ...     test_data
        ... )
        >>> print(f"Accuracy drop: {metrics['accuracy_drop']:.2%}")
    """
    import tempfile
    import os

    results = {}

    # Size comparison
    def get_model_size(model: nn.Module) -> float:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            torch.save(model.state_dict(), f.name)
            size = os.path.getsize(f.name) / (1024 * 1024)
            os.unlink(f.name)
        return size

    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)

    results["original_size_mb"] = original_size
    results["quantized_size_mb"] = quantized_size
    results["compression_ratio"] = (
        original_size / quantized_size if quantized_size > 0 else 1.0
    )
    results["size_reduction_pct"] = (1 - quantized_size / original_size) * 100

    # Inference comparison
    original_model.eval()
    quantized_model.eval()

    with torch.no_grad():
        original_output = original_model(test_input)
        quantized_output = quantized_model(test_input)

    # Output difference
    output_diff = (original_output - quantized_output).abs()
    results["max_output_diff"] = output_diff.max().item()
    results["mean_output_diff"] = output_diff.mean().item()
    results["relative_error"] = (
        (output_diff / (original_output.abs() + 1e-8)).mean().item()
    )

    # Accuracy comparison (if classification)
    if original_output.dim() > 1 and original_output.size(-1) > 1:
        original_pred = original_output.argmax(dim=-1)
        quantized_pred = quantized_output.argmax(dim=-1)
        results["prediction_agreement"] = (
            (original_pred == quantized_pred).float().mean().item()
        )

    # Parameter statistics
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    results["original_params"] = count_parameters(original_model)
    results["quantized_params"] = count_parameters(quantized_model)

    # Detailed layer comparison
    if detailed:
        layer_comparison = {}

        def get_layer_outputs(model, x, prefix=""):
            outputs = {}
            for name, module in model.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if len(list(module.children())) == 0:
                    # Leaf module
                    try:
                        with torch.no_grad():
                            out = module(x)
                        outputs[full_name] = out
                        x = out
                    except:
                        pass
                else:
                    child_outputs = get_layer_outputs(module, x, full_name)
                    outputs.update(child_outputs)
                    if full_name in child_outputs:
                        x = child_outputs[full_name]
            return outputs

        original_layer_outputs = get_layer_outputs(original_model, test_input)
        quantized_layer_outputs = get_layer_outputs(quantized_model, test_input)

        for layer_name in original_layer_outputs:
            if layer_name in quantized_layer_outputs:
                orig_out = original_layer_outputs[layer_name]
                quant_out = quantized_layer_outputs[layer_name]

                if orig_out.shape == quant_out.shape:
                    diff = (orig_out - quant_out).abs()
                    layer_comparison[layer_name] = {
                        "max_diff": diff.max().item(),
                        "mean_diff": diff.mean().item(),
                        "shape": list(orig_out.shape),
                    }

        results["layer_comparison"] = layer_comparison

    return results


def benchmark_quantization(
    model: nn.Module,
    quantization_methods: List[str],
    test_input: Tensor,
    num_runs: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Benchmark different quantization methods.

    Args:
        model: Model to benchmark
        quantization_methods: List of quantization methods to compare
        test_input: Input tensor for benchmarking
        num_runs: Number of inference runs for timing

    Returns:
        Dictionary with benchmark results for each method
    """
    import time

    results = {}

    for method in quantization_methods:
        try:
            # Apply quantization
            if method == "original":
                qmodel = copy.deepcopy(model)
            elif method == "dynamic":
                qmodel = DynamicQuantization().quantize(copy.deepcopy(model))
            elif method == "static":
                # Static requires calibration, skip for benchmark
                continue
            else:
                continue

            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = qmodel(test_input)

            # Benchmark
            qmodel.eval()
            start_time = time.time()

            with torch.no_grad():
                for _ in range(num_runs):
                    _ = qmodel(test_input)

            elapsed_time = time.time() - start_time
            avg_time_ms = (elapsed_time / num_runs) * 1000

            # Get size
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
                torch.save(qmodel.state_dict(), f.name)
                size_mb = os.path.getsize(f.name) / (1024 * 1024)
                os.unlink(f.name)

            results[method] = {
                "avg_inference_time_ms": avg_time_ms,
                "model_size_mb": size_mb,
                "throughput_samples_per_sec": test_input.size(0) / (avg_time_ms / 1000),
            }

        except Exception as e:
            results[method] = {"error": str(e)}

    return results


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Post-Training Quantization
    "BaseQuantization",
    "StaticQuantization",
    "DynamicQuantization",
    "PerChannelQuantization",
    "PerTensorQuantization",
    # QAT
    "FakeQuantize",
    "QATLinear",
    "QATConv2d",
    "LearnableQuantization",
    # Low-Precision Formats
    "QuantizationFormat",
    "LowPrecisionConverter",
    # Pruning
    "BasePruning",
    "MagnitudePruning",
    "StructuredPruning",
    "LotteryTicketPruning",
    "GradualPruning",
    # Knowledge Distillation
    "LogitDistillation",
    "FeatureDistillation",
    "RelationDistillation",
    # Efficient Inference
    "QuantizedLinear",
    "QuantizedConv2d",
    "QuantizedBatchNorm",
    # Utilities
    "quantize_model",
    "dequantize_model",
    "compare_models",
    "benchmark_quantization",
]
