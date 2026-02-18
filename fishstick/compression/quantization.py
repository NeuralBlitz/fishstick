"""
Advanced Quantization Methods for Model Compression

Supports dynamic quantization, static quantization, quantization-aware training (QAT),
fake quantize modules, and INT8/FP16 precision.
"""

from typing import Optional, List, Dict, Callable, Tuple, Union, Any
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
from torch.ao.quantization import get_default_qconfig, get_default_qat_qconfig
from torch.ao.quantization.quantize import prepare_qat


class FakeQuantizeModule(nn.Module):
    """Fake quantization module for QAT simulation.

    Simulates quantization effects during training without actually quantizing.

    Args:
        qi: Quantization range for inputs
        qw: Quantization range for weights
        qa: Quantization range for activations
        num_bits: Number of quantization bits
    """

    def __init__(
        self,
        qi: float = 0.0,
        qo: float = 6.0,
        num_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel

        self.qmin = -(2 ** (num_bits - 1)) if symmetric else 0
        self.qmax = 2 ** (num_bits - 1) - 1 if symmetric else 2**num_bits - 1

        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1))

        self.min_val = qi
        self.max_val = qo
        self.enabled = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled or not self.training:
            return x

        with torch.no_grad():
            self._update_stats(x)

        return self._fake_quantize(x)

    def _update_stats(self, x: Tensor):
        """Update quantization parameters from tensor statistics."""
        if self.per_channel and x.dim() > 1:
            min_val = x.min(dim=tuple(range(1, x.dim()))).values
            max_val = x.max(dim=tuple(range(1, x.dim()))).values
        else:
            min_val = torch.tensor(x.min().item())
            max_val = torch.tensor(x.max().item())

        min_val = min_val.clamp(self.min_val, self.max_val)
        max_val = max_val.clamp(self.min_val, self.max_val)

        self.scale = (max_val - min_val) / (self.qmax - self.qmin)
        self.scale = self.scale.clamp(min=1e-8)

        if self.symmetric:
            self.zero_point = torch.zeros_like(self.scale)
        else:
            self.zero_point = self.qmin - min_val / self.scale
            self.zero_point = (
                self.zero_point.round().clamp(self.qmin, self.qmax).to(torch.int32)
            )

    def _fake_quantize(self, x: Tensor) -> Tensor:
        """Apply fake quantization to tensor."""
        q = (x / self.scale + self.zero_point).round()
        q = q.clamp(self.qmin, self.qmax)
        return (q - self.zero_point) * self.scale

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


class QuantizedLinear(nn.Module):
    """Quantized linear layer with fake quantization support for QAT.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias
        num_bits: Quantization bits
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.weight_fake_quant = FakeQuantizeModule(num_bits=num_bits, symmetric=True)
        self.activation_fake_quant = FakeQuantizeModule(
            num_bits=num_bits, symmetric=False
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        quantized_weight = self.weight_fake_quant(self.weight)
        output = F.linear(x, quantized_weight, self.bias)
        return self.activation_fake_quant(output)

    def to_quantized(self) -> nn.quantized.Linear:
        """Convert to actual quantized linear layer."""
        return nn.quantized.Linear(
            self.weight_fake_quant.scale.item(),
            self.weight_fake_quant.zero_point.item(),
            self.weight,
            self.bias,
        )


class QuantizedConv2d(nn.Module):
    """Quantized Conv2d layer with fake quantization support for QAT.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
        padding: Padding added to input
        num_bits: Quantization bits
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        num_bits: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.weight_fake_quant = FakeQuantizeModule(
            num_bits=num_bits, symmetric=True, per_channel=True
        )
        self.activation_fake_quant = FakeQuantizeModule(
            num_bits=num_bits, symmetric=False
        )

    def forward(self, x: Tensor) -> Tensor:
        quantized_weight = self.weight_fake_quant(self.conv.weight)
        output = F.conv2d(
            x,
            quantized_weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        return self.activation_fake_quant(output)


import math


class DynamicQuantizer:
    """Dynamic quantization for post-training quantization.

    Quantizes weights statically and activations dynamically at runtime.

    Args:
        model: Model to quantize
        dtype: Quantization dtype (torch.qint8, torch.quint8, torch.float16)
        qconfig_spec: Types of layers to quantize
    """

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
        qconfig_spec: Optional[set] = None,
    ):
        self.model = model
        self.dtype = dtype
        self.qconfig_spec = qconfig_spec or {nn.Linear, nn.LSTM, nn.GRU, nn.RNN}
        self.quantized_model: Optional[nn.Module] = None

    def quantize(self) -> nn.Module:
        """Apply dynamic quantization to the model."""
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            self.qconfig_spec,
            dtype=self.dtype,
        )
        return self.quantized_model

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

        original_size = get_model_size(self.model)
        quantized_size = (
            get_model_size(self.quantized_model) if self.quantized_model else 0
        )

        return {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": original_size / quantized_size
            if quantized_size > 0
            else 0,
        }


class StaticQuantizer:
    """Static quantization with calibration.

    Requires calibration data to determine optimal quantization ranges.

    Args:
        model: Model to quantize
        qconfig: Quantization configuration
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
    """

    def __init__(
        self,
        model: nn.Module,
        backend: str = "fbgemm",
    ):
        self.model = model
        self.backend = backend
        self.prepared_model: Optional[nn.Module] = None
        self.quantized_model: Optional[nn.Module] = None

    def prepare(self, example_inputs: Optional[Tensor] = None) -> nn.Module:
        """Prepare model for static quantization."""
        self.model.eval()

        self._fuse_modules()

        qconfig = get_default_qconfig(self.backend)
        self.model.qconfig = qconfig

        torch.quantization.prepare(self.model, inplace=True)
        self.prepared_model = self.model

        return self.prepared_model

    def _fuse_modules(self):
        """Fuse modules for better quantization (Conv-BN-ReLU patterns)."""
        fuse_list = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                fuse_list.append(name)

        if len(fuse_list) >= 2:
            try:
                torch.quantization.fuse_modules(self.model, fuse_list[:2], inplace=True)
            except (ValueError, AttributeError):
                pass

    def calibrate(self, calibration_fn: Callable, num_batches: int = 100):
        """Calibrate quantization parameters using sample data."""
        if self.prepared_model is None:
            raise ValueError("Call prepare() before calibrate()")

        self.prepared_model.eval()

        with torch.no_grad():
            for i, data in enumerate(calibration_fn()):
                if i >= num_batches:
                    break
                if isinstance(data, (tuple, list)):
                    self.prepared_model(data[0])
                else:
                    self.prepared_model(data)

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
        calibration_fn: Callable,
        num_batches: int = 100,
        example_inputs: Optional[Tensor] = None,
    ) -> nn.Module:
        """Full quantization pipeline."""
        self.prepare(example_inputs)
        self.calibrate(calibration_fn, num_batches)
        return self.convert()


class QuantizationAwareTrainer:
    """Quantization-Aware Training (QAT) for model quantization.

    Simulates quantization during training for better accuracy after quantization.

    Args:
        model: Model to train with QAT
        backend: Quantization backend
        qconfig: Quantization configuration
    """

    def __init__(
        self,
        model: nn.Module,
        backend: str = "fbgemm",
    ):
        self.model = model
        self.backend = backend
        self.prepared_model: Optional[nn.Module] = None

    def prepare(self) -> nn.Module:
        """Prepare model for QAT."""
        self.model.train()

        self._fuse_modules()

        qconfig = get_default_qat_qconfig(self.backend)
        self.model.qconfig = qconfig

        self.prepared_model = torch.quantization.prepare_qat(self.model, inplace=True)

        return self.prepared_model

    def _fuse_modules(self):
        """Fuse modules for QAT."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Sequential):
                if len(module) >= 2:
                    if isinstance(module[0], nn.Conv2d) and isinstance(
                        module[1], nn.BatchNorm2d
                    ):
                        torch.quantization.fuse_modules(
                            self.model, [f"{name}.0", f"{name}.1"], inplace=True
                        )

    def train_epoch(
        self,
        train_loader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """Train for one epoch with QAT."""
        if self.prepared_model is None:
            raise ValueError("Call prepare() before training")

        self.prepared_model.train()
        self.prepared_model.to(device)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = self.prepared_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            total_correct += (pred == target).sum().item()
            total_samples += target.size(0)

        return {
            "loss": total_loss / len(train_loader),
            "accuracy": total_correct / total_samples,
        }

    def convert(self) -> nn.Module:
        """Convert trained model to quantized model."""
        if self.prepared_model is None:
            raise ValueError("Call prepare() and train before convert()")

        self.prepared_model.eval()
        quantized_model = torch.quantization.convert(self.prepared_model, inplace=True)
        return quantized_model


class MixedPrecisionQuantizer:
    """Mixed precision quantization with different bit-widths per layer.

    Args:
        model: Model to quantize
        bits_config: Dictionary mapping layer names to bit-widths
    """

    def __init__(
        self,
        model: nn.Module,
        bits_config: Optional[Dict[str, int]] = None,
        default_bits: int = 8,
    ):
        self.model = model
        self.bits_config = bits_config or {}
        self.default_bits = default_bits
        self.fake_quant_modules: Dict[str, FakeQuantizeModule] = {}

    def auto_configure_bits(self, sensitivity_scores: Dict[str, float]):
        """Automatically configure bit-widths based on sensitivity scores."""
        if not sensitivity_scores:
            return

        scores = torch.tensor(list(sensitivity_scores.values()))
        min_score, max_score = scores.min(), scores.max()

        for name, score in sensitivity_scores.items():
            normalized = (score - min_score) / (max_score - min_score + 1e-8)
            if normalized > 0.7:
                self.bits_config[name] = 16
            elif normalized > 0.3:
                self.bits_config[name] = 8
            else:
                self.bits_config[name] = 4

    def prepare(self) -> nn.Module:
        """Prepare model with mixed precision fake quantization."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                bits = self.bits_config.get(name, self.default_bits)
                fq = FakeQuantizeModule(num_bits=bits)
                self.fake_quant_modules[name] = fq

        return self.model

    def quantize_weights(self):
        """Apply fake quantization to weights."""
        for name, param in self.model.named_parameters():
            if "weight" in name:
                layer_name = name.rsplit(".weight", 1)[0]
                if layer_name in self.fake_quant_modules:
                    fq = self.fake_quant_modules[layer_name]
                    with torch.no_grad():
                        fq._update_stats(param)
                        param.copy_(fq._fake_quantize(param))


class FP16Quantizer:
    """Half-precision (FP16) quantization for GPU acceleration.

    Args:
        model: Model to quantize
        keep_fp32_copy: Keep FP32 copy of weights for mixed precision
    """

    def __init__(self, model: nn.Module, keep_fp32_copy: bool = True):
        self.model = model
        self.keep_fp32_copy = keep_fp32_copy
        self.fp32_state_dict: Optional[Dict[str, Tensor]] = None

    def quantize(self) -> nn.Module:
        """Convert model to FP16."""
        if self.keep_fp32_copy:
            self.fp32_state_dict = {
                name: param.clone() for name, param in self.model.named_parameters()
            }

        self.model = self.model.half()
        return self.model

    def restore_fp32(self) -> nn.Module:
        """Restore model to FP32."""
        if self.fp32_state_dict is None:
            raise ValueError("No FP32 state dict saved")

        self.model = self.model.float()
        self.model.load_state_dict(self.fp32_state_dict)
        return self.model


class INT8Quantizer:
    """INT8 quantization for efficient inference.

    Args:
        model: Model to quantize
        per_channel: Use per-channel quantization
        symmetric: Use symmetric quantization range
    """

    def __init__(
        self,
        model: nn.Module,
        per_channel: bool = True,
        symmetric: bool = True,
    ):
        self.model = model
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.quantized_model: Optional[nn.Module] = None
        self.scale: Dict[str, Tensor] = {}
        self.zero_point: Dict[str, Tensor] = {}

    def compute_quantization_params(self, tensor: Tensor, name: str):
        """Compute scale and zero_point for quantization."""
        if self.per_channel and tensor.dim() > 1:
            min_val = tensor.min(dim=tuple(range(1, tensor.dim()))).values
            max_val = tensor.max(dim=tuple(range(1, tensor.dim()))).values
        else:
            min_val = torch.tensor(tensor.min().item())
            max_val = torch.tensor(tensor.max().item())

        qmin, qmax = -128, 127 if self.symmetric else 0, 255

        scale = (max_val - min_val) / (qmax - qmin)
        scale = scale.clamp(min=1e-8)

        if self.symmetric:
            zero_point = torch.zeros_like(scale)
        else:
            zero_point = qmin - min_val / scale
            zero_point = zero_point.round().clamp(qmin, qmax).to(torch.int32)

        self.scale[name] = scale
        self.zero_point[name] = zero_point

    def quantize_tensor(self, tensor: Tensor, name: str) -> Tensor:
        """Quantize a tensor to INT8."""
        self.compute_quantization_params(tensor, name)

        scale = self.scale[name]
        zero_point = self.zero_point[name]

        q_tensor = (tensor / scale + zero_point).round()
        q_tensor = q_tensor.clamp(-128, 127).to(torch.int8)

        return q_tensor

    def dequantize_tensor(self, q_tensor: Tensor, name: str) -> Tensor:
        """Dequantize INT8 tensor back to float."""
        scale = self.scale[name]
        zero_point = self.zero_point[name]

        return (q_tensor.float() - zero_point.float()) * scale


__all__ = [
    "FakeQuantizeModule",
    "QuantizedLinear",
    "QuantizedConv2d",
    "DynamicQuantizer",
    "StaticQuantizer",
    "QuantizationAwareTrainer",
    "MixedPrecisionQuantizer",
    "FP16Quantizer",
    "INT8Quantizer",
]
