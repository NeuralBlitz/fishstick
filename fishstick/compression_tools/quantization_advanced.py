"""
Advanced Quantization Methods for Model Compression

Post-training quantization (PTQ), dynamic quantization, static quantization,
calibration-based quantization, and mixed-precision quantization support.

References:
- https://pytorch.org/docs/stable/quantization.html
- https://arxiv.org/abs/1806.08342 (PyTorch Quantization)
- https://arxiv.org/abs/1903.08066 (HAWQ: Hessian AWare Quantization)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Callable, Tuple, Union, Any, Literal
from enum import Enum
import copy
import warnings

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
from torch.ao.quantization import (
    get_default_qconfig,
    get_default_qat_qconfig,
    observer,
    fuse_modules,
)
from torch.ao.quantization.quantize import prepare, prepare_qat, convert
from torch.ao.quantization.qconfig import QConfig, QConfigAny


class QuantizationMode(Enum):
    """Supported quantization modes."""

    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"
    PTQ = "ptq"


class CalibrationMethod(Enum):
    """Methods for calibration during PTQ."""

    MINMAX = "minmax"
    PERCENTILE = "percentile"
    HISTOGRAM = "histogram"
    ENTROPY = "entropy"


class PTQQuantizer:
    """Post-Training Quantization (PTQ) with calibration.

    Performs quantization after training using a calibration dataset
    to determine optimal quantization parameters.

    Args:
        model: The trained model to quantize
        mode: Quantization mode (dynamic, static, or qat)
        num_bits: Number of bits for quantization (default: 8)
        calibration_method: Method for calibration
        per_channel: Whether to use per-channel quantization for weights
        symmetric: Whether to use symmetric quantization

    Example:
        >>> quantizer = PTQQuantizer(model, mode='static', num_bits=8)
        >>> quantized_model = quantizer.calibrate(calibration_dataloader)
        >>> result = quantized_model(input)
    """

    def __init__(
        self,
        model: nn.Module,
        mode: Union[str, QuantizationMode] = "static",
        num_bits: int = 8,
        calibration_method: Union[str, CalibrationMethod] = "minmax",
        per_channel: bool = False,
        symmetric: bool = True,
    ):
        self.original_model = model
        self.model = copy.deepcopy(model)
        self.mode = QuantizationMode(mode) if isinstance(mode, str) else mode
        self.num_bits = num_bits
        self.calibration_method = (
            CalibrationMethod(calibration_method)
            if isinstance(calibration_method, str)
            else calibration_method
        )
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.quantized_model: Optional[nn.Module] = None
        self.scale: Dict[str, Tensor] = {}
        self.zero_point: Dict[str, Tensor] = {}

    def _get_qconfig(self) -> QConfig:
        """Get quantization configuration."""
        if self.mode == QuantizationMode.QAT:
            qconfig = get_default_qat_qconfig("fbgemm")
        else:
            qconfig = get_default_qconfig("fbgemm")

        return qconfig

    def _prepare_model(self) -> nn.Module:
        """Prepare model for quantization."""
        self.model.eval()

        if self.mode == QuantizationMode.STATIC:
            self.model.qconfig = self._get_qconfig()
            prepare(self.model, inplace=True)
        elif self.mode == QuantizationMode.QAT:
            self.model.qconfig = self._get_qconfig()
            prepare_qat(self.model, inplace=True)

        return self.model

    def calibrate(
        self,
        calibration_data: List[Tensor],
        num_batches: int = 10,
        batch_size: int = 32,
    ) -> nn.Module:
        """Run calibration on representative dataset.

        Args:
            calibration_data: List of calibration tensors
            num_batches: Number of batches to use for calibration
            batch_size: Batch size for calibration

        Returns:
            Quantized model after calibration
        """
        self._prepare_model()
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(calibration_data[: num_batches * batch_size]):
                if isinstance(data, tuple):
                    self.model(*data)
                else:
                    self.model(data.unsqueeze(0) if data.dim() == 1 else data)

        return self._convert()

    def _convert(self) -> nn.Module:
        """Convert model to quantized version."""
        if self.mode in [QuantizationMode.STATIC, QuantizationMode.QAT]:
            self.quantized_model = convert(self.model, inplace=False)
        else:
            self.quantized_model = self.model

        return self.quantized_model

    def quantize(self) -> nn.Module:
        """Apply quantization to the model."""
        if self.quantized_model is None:
            raise RuntimeError("Must call calibrate() before quantize()")
        return self.quantized_model


class DynamicQuantizationEngine:
    """Dynamic quantization engine with flexible configuration.

    Performs dynamic quantization where weights are quantized statically
    but activations are quantized dynamically at runtime.

    Args:
        model: Model to quantize
        num_bits: Weight quantization bits
        layer_types: Types of layers to quantize
        symmetric: Use symmetric quantization

    Example:
        >>> engine = DynamicQuantizationEngine(model, num_bits=8)
        >>> q_model = engine.quantize()
    """

    def __init__(
        self,
        model: nn.Module,
        num_bits: int = 8,
        layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d),
        symmetric: bool = True,
    ):
        self.model = model
        self.num_bits = num_bits
        self.layer_types = layer_types
        self.symmetric = symmetric

    def quantize(self) -> nn.Module:
        """Apply dynamic quantization to model."""
        quantized_model = copy.deepcopy(self.model)

        for module in quantized_model.modules():
            if isinstance(module, self.layer_types):
                if isinstance(module, nn.Linear):
                    quantized_module = nn.quantized.dynamic.Linear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        dtype=torch.qint8,
                    )
                elif isinstance(module, nn.Conv2d):
                    quantized_module = nn.quantized.dynamic.Conv2d(
                        module.in_channels,
                        module.out_channels,
                        module.kernel_size,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                        module.bias is not None,
                        module.padding_mode,
                        dtype=torch.qint8,
                    )

                    quantized_module.weight = nn.Parameter(
                        torch.quantize_per_tensor(
                            module.weight.data,
                            scale=1.0,
                            zero_point=0,
                            dtype=torch.qint8,
                        )
                    )
                    if module.bias is not None:
                        quantized_module.bias = module.bias

                module = quantized_module

        quantized_model.eval()
        return torch.quantization.quantize_dynamic(
            quantized_model,
            {nn.Linear},
            dtype=torch.qint8 if self.num_bits == 8 else torch.float16,
        )

    def get_compression_ratio(self) -> float:
        """Estimate compression ratio from quantization."""
        original_size = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        quantized_size = sum(p.numel() * 1 for p in self.model.parameters())
        return original_size / max(quantized_size, 1)


class StaticQuantizationEngine:
    """Static quantization engine with calibration.

    Performs static quantization where both weights and activations
    are quantized using calibration data.

    Args:
        model: Model to quantize
        num_bits: Number of quantization bits
        calibration_method: Method for calibration
        observers: Custom observers for different layer types
    """

    def __init__(
        self,
        model: nn.Module,
        num_bits: int = 8,
        calibration_method: CalibrationMethod = CalibrationMethod.MINMAX,
        observers: Optional[Dict[type, type]] = None,
    ):
        self.model = model
        self.num_bits = num_bits
        self.calibration_method = calibration_method
        self.observers = observers or self._default_observers()
        self.quantized_model: Optional[nn.Module] = None

    def _default_observers(self) -> Dict[type, type]:
        """Get default observers for layer types."""
        return {
            nn.Linear: observer.MinMaxObserver,
            nn.Conv2d: observer.MinMaxObserver,
            nn.Conv3d: observer.MinMaxObserver,
        }

    def _prepare_model(self) -> nn.Module:
        """Prepare model with observers."""
        model = copy.deepcopy(self.model)
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)
        return model

    def calibrate(
        self,
        calibration_data: List[Tensor],
        num_samples: int = 100,
    ) -> nn.Module:
        """Run calibration on dataset."""
        model = self._prepare_model()
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(calibration_data[:num_samples]):
                if isinstance(data, tuple):
                    model(*data)
                else:
                    model(data)

        self.quantized_model = torch.quantization.convert(model, inplace=False)
        return self.quantized_model

    def quantize(self, calibration_data: Optional[List[Tensor]] = None) -> nn.Module:
        """Apply static quantization to model."""
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
        return self.calibrate(calibration_data)


class MixedPrecisionQuantizer:
    """Mixed-precision quantization manager.

    Applies different precision to different layers based on their
    sensitivity to quantization.

    Args:
        model: Model to quantize
        layer_precisions: Dict mapping layer names to bit precision
        default_precision: Default precision for unlisted layers
    """

    def __init__(
        self,
        model: nn.Module,
        layer_precisions: Optional[Dict[str, int]] = None,
        default_precision: int = 8,
    ):
        self.model = model
        self.layer_precisions = layer_precisions or {}
        self.default_precision = default_precision
        self.sensitivity_scores: Dict[str, float] = {}

    def compute_sensitivity(
        self,
        dataloader: List[Tuple[Tensor, Tensor]],
        num_samples: int = 50,
    ) -> Dict[str, float]:
        """Compute quantization sensitivity for each layer.

        Uses Hessian-based sensitivity estimation to determine
        which layers are most sensitive to quantization.

        Args:
            dataloader: Data for sensitivity computation
            num_samples: Number of samples to use

        Returns:
            Dict mapping layer names to sensitivity scores
        """
        self.model.eval()
        sensitivities = {}

        for name, module in self.model.named_modules():
            if not (isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)):
                continue

            original_weight = module.weight.data.clone()
            sensitivity = 0.0

            with torch.no_grad():
                for i, (data, target) in enumerate(dataloader[:num_samples]):
                    output = self.model(data)
                    if isinstance(target, Tensor):
                        loss = F.cross_entropy(output, target)
                    else:
                        loss = F.cross_entropy(output, torch.tensor(target))

                    original_loss = loss.item()

                    module.weight.data = original_weight * 0.5
                    perturbed_loss = self.model(data).float()
                    if isinstance(target, Tensor):
                        perturbed_loss = F.cross_entropy(perturbed_loss, target)
                    else:
                        perturbed_loss = F.cross_entropy(
                            perturbed_loss, torch.tensor(target)
                        )

                    sensitivity += abs(perturbed_loss.item() - original_loss)
                    module.weight.data = original_weight.clone()

            sensitivities[name] = sensitivity / max(num_samples, 1)
            self.sensitivity_scores = sensitivities

        return sensitivities

    def quantize(
        self,
        sensitivity_based: bool = True,
    ) -> nn.Module:
        """Apply mixed-precision quantization.

        Args:
            sensitivity_based: Use sensitivity scores to determine precision

        Returns:
            Quantized model with mixed precision
        """
        model = copy.deepcopy(self.model)

        if sensitivity_based and not self.sensitivity_scores:
            warnings.warn("No sensitivity scores computed. Using default precision.")

        for name, module in model.named_modules():
            if not (isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)):
                continue

            if sensitivity_based and name in self.sensitivity_scores:
                sensitivity = self.sensitivity_scores[name]
                if sensitivity > 0.5:
                    precision = max(4, self.default_precision - 4)
                elif sensitivity > 0.1:
                    precision = self.default_precision
                else:
                    precision = min(16, self.default_precision + 4)
            else:
                precision = self.layer_precisions.get(name, self.default_precision)

            module.quantization_precision = precision

        return model

    def get_layer_precision(self, layer_name: str) -> int:
        """Get quantization precision for a specific layer."""
        return self.layer_precisions.get(layer_name, self.default_precision)


class QuantizationAwareTrainer:
    """Quantization-aware training wrapper.

    Simulates quantization effects during training to improve
    quantized model accuracy.

    Args:
        model: Model to wrap
        num_bits: Number of quantization bits
        forward_hook: Optional hook for custom forward behavior

    Example:
        >>> trainer = QuantizationAwareTrainer(model, num_bits=8)
        >>> for epoch in range(epochs):
        ...     trainer.train_epoch(train_loader)
        ...     trainer.validate(val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        num_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
    ):
        self.model = model
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.fake_quant_modules: Dict[str, nn.Module] = {}

        self._setup_fake_quantization()

    def _setup_fake_quantization(self):
        """Set up fake quantization modules for each layer."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                fake_quant = FakeQuantize(
                    num_bits=self.num_bits,
                    symmetric=self.symmetric,
                    per_channel=self.per_channel,
                )
                self.fake_quant_modules[name] = fake_quant

    def _fake_quantize_weights(self):
        """Apply fake quantization to weights during forward pass."""
        for name, module in self.model.named_modules():
            if name in self.fake_quant_modules:
                original_weight = module.weight.data
                quantized_weight = self.fake_quant_modules[name](original_weight)
                module.weight.data = quantized_weight

    def train_epoch(
        self,
        dataloader: List[Tuple[Tensor, Tensor]],
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: str = "cpu",
    ) -> float:
        """Train for one epoch with fake quantization.

        Args:
            dataloader: Training data
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            self._fake_quantize_weights()
            output = self.model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def validate(
        self,
        dataloader: List[Tuple[Tensor, Tensor]],
        criterion: Callable,
        device: str = "cpu",
    ) -> float:
        """Validate model.

        Args:
            dataloader: Validation data
            criterion: Loss function
            device: Device to validate on

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def convert_to_quantized(self) -> nn.Module:
        """Convert to actual quantized model after training."""
        quantized_model = copy.deepcopy(self.model)

        for name, module in quantized_model.named_modules():
            if name in self.fake_quant_modules:
                fake_quant = self.fake_quant_modules[name]
                module.weight.data = fake_quant(module.weight.data)

        return torch.quantization.quantize_dynamic(
            quantized_model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8,
        )


class FakeQuantize(nn.Module):
    """Fake quantization module for simulation.

    Simulates quantization effects during training.

    Args:
        num_bits: Number of quantization bits
        symmetric: Use symmetric quantization
        per_channel: Use per-channel quantization
    """

    def __init__(
        self,
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

    def forward(self, x: Tensor) -> Tensor:
        """Apply fake quantization."""
        if not self.training:
            return x

        with torch.no_grad():
            self._update_scale_zero_point(x)

        return self._quantize_dequantize(x)

    def _update_scale_zero_point(self, x: Tensor):
        """Update scale and zero point from tensor statistics."""
        if self.per_channel and x.dim() > 1:
            dims = list(range(1, x.dim()))
            min_val = x.amin(dim=dims, keepdim=True)
            max_val = x.amax(dim=dims, keepdim=True)
        else:
            min_val = x.min()
            max_val = x.max()

        if self.symmetric:
            abs_max = torch.maximum(torch.abs(min_val), torch.abs(max_val))
            self.scale = abs_max / (self.qmax - self.qmin)
            self.zero_point = torch.zeros_like(self.scale)
        else:
            self.scale = (max_val - min_val) / (self.qmax - self.qmin)
            self.scale = self.scale.clamp(min=1e-8)
            self.zero_point = self.qmin - min_val / self.scale

    def _quantize_dequantize(self, x: Tensor) -> Tensor:
        """Quantize and dequantize tensor."""
        q = (x / self.scale + self.zero_point).round()
        q = q.clamp(self.qmin, self.qmax)
        return (q - self.zero_point) * self.scale


class QuantizationObserver:
    """Custom observer for tracking tensor statistics.

    Used during calibration to determine optimal quantization
    parameters.

    Args:
        method: Method for computing quantization parameters
        percentile: Percentile for percentile method
    """

    def __init__(
        self,
        method: CalibrationMethod = CalibrationMethod.MINMAX,
        percentile: float = 99.9,
    ):
        self.method = method
        self.percentile = percentile
        self.min_val: Optional[Tensor] = None
        self.max_val: Optional[Tensor] = None
        self.histogram: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        """Update statistics with new tensor."""
        if self.method == CalibrationMethod.MINMAX:
            if self.min_val is None:
                self.min_val = x.min()
                self.max_val = x.max()
            else:
                self.min_val = min(self.min_val, x.min())
                self.max_val = max(self.max_val, x.max())
        elif self.method == CalibrationMethod.PERCENTILE:
            flat_x = x.flatten()
            self.min_val = flat_x.quantile(1 - self.percentile / 100)
            self.max_val = flat_x.quantile(self.percentile / 100)
        elif self.method == CalibrationMethod.HISTOGRAM:
            if self.histogram is None:
                self.histogram = torch.histc(x, bins=256)
            else:
                self.histogram += torch.histc(x, bins=256)

        return x

    def get_scale_zero_point(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Get scale and zero point from collected statistics."""
        if self.method in [
            CalibrationMethod.MINMAX,
            CalibrationMethod.PERCENTILE,
        ]:
            if self.min_val is None or self.max_val is None:
                raise RuntimeError("No statistics collected")

            qmin = -(2 ** (num_bits - 1)) if symmetric else 0
            qmax = 2 ** (num_bits - 1) - 1 if symmetric else 2**num_bits - 1

            if symmetric:
                abs_max = max(abs(self.min_val), abs(self.max_val))
                scale = abs_max / (qmax - qmin)
                zero_point = torch.zeros_like(scale)
            else:
                scale = (self.max_val - self.min_val) / (qmax - qmin)
                zero_point = qmin - self.min_val / scale

            return scale, zero_point

        raise NotImplementedError(f"Method {self.method} not supported")
