"""
Model Quantization Module

Implements various quantization techniques:
- Post-Training Quantization (PTQ)
- Quantization-Aware Training (QAT)
- INT8 quantization
- FP16 (half-precision) quantization
- Mixed precision quantization
- Dynamic and Static quantization
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.quantization import (
    quantize_dynamic,
    quantize_static,
    QConfig,
    QConfigDynamic,
    default_dynamic_qconfig,
    default_per_channel_qconfig,
    default_qconfig,
    QTensor,
)
from torch.quantization.observer import (
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    HistogramObserver,
    PerChannelMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PlaceholderObserver,
)


class BaseQuantizer:
    """Base class for all quantization methods."""

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
        observer: str = "minmax",
    ):
        self.model = model
        self.dtype = dtype
        self.observer = observer
        self.quantized_model: Optional[nn.Module] = None

    def prepare(self):
        """Prepare model for quantization."""
        raise NotImplementedError

    def convert(self):
        """Convert model to quantized version."""
        raise NotImplementedError

    def quantize(self) -> nn.Module:
        """Full quantization pipeline."""
        self.prepare()
        return self.convert()


class PTQQuantizer(BaseQuantizer):
    """Post-Training Quantization - quantizes model after training."""

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
        observer: str = "minmax",
        sample_inputs: Optional[Tuple[Tensor, ...]] = None,
        qconfig: Optional[QConfig] = None,
    ):
        super().__init__(model, dtype, observer)
        self.sample_inputs = sample_inputs
        self.qconfig = qconfig or self._get_default_qconfig()

    def _get_default_qconfig(self) -> QConfig:
        """Get default quantization configuration."""
        if self.observer == "minmax":
            obs = MinMaxObserver.with_args(dtype=self.dtype)
        elif self.observer == "moving_avg":
            obs = MovingAverageMinMaxObserver.with_args(dtype=self.dtype)
        elif self.observer == "histogram":
            obs = HistogramObserver.with_args(dtype=self.dtype)
        else:
            obs = MinMaxObserver.with_args(dtype=self.dtype)

        return QConfig(activation=obs, weight=obs)

    def prepare(self):
        """Prepare model with observers."""
        self.model.qconfig = self.qconfig
        torch.quantization.prepare(self.model, inplace=True)

        if self.sample_inputs is not None:
            with torch.no_grad():
                self.model(*self.sample_inputs)

    def convert(self) -> nn.Module:
        """Convert model to quantized version."""
        self.quantized_model = torch.quantization.convert(self.model, inplace=False)
        return self.quantized_model


class QATQuantizer(BaseQuantizer):
    """Quantization-Aware Training - simulates quantization during training."""

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
        observer: str = "moving_avg",
        qconfig: Optional[QConfig] = None,
    ):
        super().__init__(model, dtype, observer)
        self.qconfig = qconfig or self._get_default_qconfig()

    def _get_default_qconfig(self) -> QConfig:
        """Get default QAT quantization configuration."""
        if self.observer == "moving_avg":
            act_obs = MovingAverageMinMaxObserver.with_args(dtype=self.dtype)
            weight_obs = MovingAveragePerChannelMinMaxObserver.with_args(
                dtype=self.dtype
            )
        else:
            act_obs = HistogramObserver.with_args(dtype=self.dtype)
            weight_obs = PerChannelMinMaxObserver.with_args(dtype=self.dtype)

        return QConfig(activation=act_obs, weight=weight_obs)

    def prepare(self):
        """Prepare model for QAT."""
        self.model.qconfig = torch.quantization.get_default_qat_qconfig(
            "fbgemm",  # x86 backend
        )
        torch.quantization.prepare_qat(self.model, inplace=True)

    def convert(self) -> nn.Module:
        """Convert QAT model to quantized version."""
        self.model.eval()
        self.quantized_model = torch.quantization.convert(self.model, inplace=False)
        return self.quantized_model

    def train_with_quantization(
        self, num_epochs: int, train_loader, optimizer, criterion
    ):
        """Train model with quantization simulation."""
        self.model.train()

        for epoch in range(num_epochs):
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        return self.model


class INT8Quantizer(BaseQuantizer):
    """INT8 quantization wrapper."""

    def __init__(
        self,
        model: nn.Module,
        observer: str = "minmax",
        sample_inputs: Optional[Tuple[Tensor, ...]] = None,
        per_channel: bool = True,
    ):
        super().__init__(model, torch.qint8, observer)
        self.sample_inputs = sample_inputs
        self.per_channel = per_channel

    def _get_qconfig(self) -> QConfig:
        """Get INT8 quantization config."""
        if self.observer == "histogram":
            act_obs = HistogramObserver.with_args(dtype=torch.quint8)
        else:
            act_obs = MinMaxObserver.with_args(dtype=torch.quint8)

        if self.per_channel:
            weight_obs = PerChannelMinMaxObserver.with_args(dtype=torch.qint8)
        else:
            weight_obs = MinMaxObserver.with_args(dtype=torch.qint8)

        return QConfig(activation=act_obs, weight=weight_obs)

    def prepare(self):
        """Prepare model for INT8 quantization."""
        qconfig = self._get_qconfig()
        self.model.qconfig = qconfig
        torch.quantization.prepare(self.model, inplace=True)

        if self.sample_inputs is not None:
            with torch.no_grad():
                self.model(*self.sample_inputs)

    def convert(self) -> nn.Module:
        """Convert to INT8 quantized model."""
        self.quantized_model = torch.quantization.convert(self.model, inplace=False)
        return self.quantized_model


class FP16Quantizer(BaseQuantizer):
    """FP16 (half-precision) quantization using torch.cuda.amp."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
    ):
        super().__init__(model, torch.float16, observer="none")
        self.device = device
        self.use_cuda = torch.cuda.is_available() and device == "cuda"

    def prepare(self):
        """Prepare model for FP16 conversion."""
        if self.use_cuda:
            self.model = self.model.to(self.device)
        self.model.half()

    def convert(self) -> nn.Module:
        """Convert to FP16 model."""
        return self.model

    def to_fp16(self) -> nn.Module:
        """Convert model to FP16."""
        return self.model.half()

    @staticmethod
    def to_fp32(model: nn.Module) -> nn.Module:
        """Convert model back to FP32."""
        return model.float()


class MixedPrecisionQuantizer(BaseQuantizer):
    """Mixed precision quantization - different precisions for different layers."""

    def __init__(
        self,
        model: nn.Module,
        precision_map: Optional[Dict[str, torch.dtype]] = None,
    ):
        super().__init__(model, torch.float16, observer="none")
        self.precision_map = precision_map or {}

    def set_layer_precision(self, layer_name: str, dtype: torch.dtype):
        """Set precision for a specific layer."""
        self.precision_map[layer_name] = dtype

    def _get_module_precision(self, module_name: str) -> torch.dtype:
        """Get precision for a module."""
        return self.precision_map.get(module_name, torch.float32)

    def prepare(self):
        """Prepare model with mixed precision."""
        for name, module in self.model.named_modules():
            precision = self._get_module_precision(name)
            if precision == torch.float16:
                module.half()
            elif precision == torch.bfloat16:
                module = module.to(dtype=torch.bfloat16)

    def convert(self) -> nn.Module:
        """Convert model with mixed precision."""
        return self.model


class DynamicQuantizer(BaseQuantizer):
    """Dynamic quantization - weights quantized, activations in float."""

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
        qconfig: Optional[QConfigDynamic] = None,
    ):
        super().__init__(model, dtype, observer="dynamic")
        self.qconfig = qconfig or default_dynamic_qconfig

    def prepare(self):
        """Dynamic quantization doesn't require preparation."""
        pass

    def quantize(self) -> nn.Module:
        """Apply dynamic quantization."""
        self.quantized_model = quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.GRU, nn.RNN},
            dtype=self.dtype,
            qconfig=self.qconfig,
        )
        return self.quantized_model


class StaticQuantizer(BaseQuantizer):
    """Static quantization - requires calibration with sample data."""

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
        observer: str = "minmax",
        sample_inputs: Optional[Tuple[Tensor, ...]] = None,
    ):
        super().__init__(model, dtype, observer)
        self.sample_inputs = sample_inputs

    def prepare(self):
        """Prepare model with observers."""
        if self.observer == "histogram":
            qconfig = QConfig(
                activation=HistogramObserver.with_args(dtype=torch.quint8),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8),
            )
        else:
            qconfig = default_per_channel_qconfig

        self.model.qconfig = qconfig
        torch.quantization.prepare(self.model, inplace=True)

    def calibrate(self, data_loader):
        """Run calibration with sample data."""
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                self.model(inputs)

                if self.sample_inputs is None:
                    break

    def convert(self) -> nn.Module:
        """Convert to statically quantized model."""
        self.quantized_model = torch.quantization.convert(self.model, inplace=False)
        return self.quantized_model


class FakeQuantizer(BaseQuantizer):
    """Fake quantization for training simulation."""

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
        observer: str = "moving_avg",
        qconfig: Optional[QConfig] = None,
    ):
        super().__init__(model, dtype, observer)
        self.qconfig = qconfig or self._get_default_qconfig()

    def _get_default_qconfig(self) -> QConfig:
        """Get fake quantization config."""
        if self.observer == "moving_avg":
            act_obs = MovingAverageMinMaxObserver.with_args(dtype=self.dtype)
        else:
            act_obs = MinMaxObserver.with_args(dtype=self.dtype)

        weight_obs = PerChannelMinMaxObserver.with_args(dtype=self.dtype)

        return QConfig(activation=act_obs, weight=weight_obs)

    def prepare(self):
        """Prepare model with fake quantization."""
        self.model.qconfig = self.qconfig
        torch.quantization.prepare(self.model, inplace=True)

    def convert(self) -> nn.Module:
        """Convert model (fake quant becomes actual quant)."""
        self.quantized_model = torch.quantization.convert(self.model, inplace=False)
        return self.quantized_model

    def enable_fake_quant(self):
        """Enable fake quantization."""
        for module in self.model.modules():
            if hasattr(module, "qconfig") and module.qconfig is not None:
                module.qconfig.activation = module.qconfig.activation()
                module.qconfig.weight = module.qconfig.weight()

    def disable_fake_quant(self):
        """Disable fake quantization (use observers only)."""
        for module in self.model.modules():
            if hasattr(module, "qconfig") and module.qconfig is not None:
                module.qconfig.activation = PlaceholderObserver.with_args(
                    dtype=torch.float32
                )
                module.qconfig.weight = PlaceholderObserver.with_args(
                    dtype=torch.float32
                )
