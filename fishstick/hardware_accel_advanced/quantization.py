from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import (
    QuantStub,
    DeQuantStub,
    QConfig,
    default_dynamic_qconfig,
    default_observer,
    default_weight_observer,
)


class QuantizationType(Enum):
    INT8 = "int8"
    FP16 = "fp16"
    BFLOAT16 = "bfloat16"
    MIXED = "mixed"


@dataclass
class QuantizationConfig:
    quant_type: QuantizationType = QuantizationType.INT8
    per_channel: bool = False
    symmetric: bool = True
    observer_type: str = "minmax"
    dtype: torch.dtype = torch.qint8
    reduce_range: bool = False


class BaseQuantizer(nn.Module):
    def __init__(self, config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config or QuantizationConfig()
        self.quant_stub = QuantStub()
        self.dequant_stub = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class INT8Quantizer(BaseQuantizer):
    def __init__(
        self,
        config: Optional[QuantizationConfig] = None,
        qconfig: Optional[QConfig] = None,
    ):
        super().__init__(config or QuantizationConfig(quant_type=QuantizationType.INT8))
        self.qconfig = qconfig or QConfig(
            activation=default_observer,
            weight=default_weight_observer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_stub(x)
        return self.dequant_stub(x)

    def prepare(self, model: nn.Module) -> nn.Module:
        model.qconfig = self.qconfig
        torch.quantization.prepare(model, inplace=True)
        return model

    def convert(self, model: nn.Module) -> nn.Module:
        torch.quantization.convert(model, inplace=True)
        return model


class FP16Quantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.half()

    def to_fp16(self, model: nn.Module) -> nn.Module:
        return model.half()

    def convert_module(self, module: nn.Module) -> nn.Module:
        for name, child in module.named_children():
            self.convert_module(child)
            if isinstance(child, nn.Parameter):
                continue
            try:
                child = child.half()
                setattr(module, name, child)
            except Exception:
                pass
        return module


class BFloat16Quantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.bfloat16()

    def to_bf16(self, model: nn.Module) -> nn.Module:
        return model.to(dtype=torch.bfloat16)

    def convert_module(self, module: nn.Module) -> nn.Module:
        for name, child in module.named_children():
            self.convert_module(child)
            if isinstance(child, nn.Parameter):
                continue
            try:
                child = child.to(dtype=torch.bfloat16)
                setattr(module, name, child)
            except Exception:
                pass
        return module


class DynamicQuantizer:
    def __init__(
        self,
        dtype: torch.dtype = torch.qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
    ):
        self.dtype = dtype
        self.qscheme = qscheme

    def quantize(self, model: nn.Module) -> nn.Module:
        torch.quantization.quantize_dynamic(
            model,
            {nn.Linear: qconfig},
            dtype=self.dtype,
            inplace=True,
        )
        return model


class StaticQuantizer:
    def __init__(
        self,
        qconfig: Optional[QConfig] = None,
    ):
        self.qconfig = qconfig or default_dynamic_qconfig

    def prepare(self, model: nn.Module) -> nn.Module:
        model.qconfig = self.qconfig
        torch.quantization.prepare(model, inplace=True)
        return model

    def convert(self, model: nn.Module) -> nn.Module:
        torch.quantization.convert(model, inplace=True)
        return model


class MixedPrecisionManager:
    def __init__(
        self,
        default_dtype: torch.dtype = torch.float32,
        layer_config: Optional[Dict[str, torch.dtype]] = None,
    ):
        self.default_dtype = default_dtype
        self.layer_config = layer_config or {}
        self.original_dtypes: Dict[str, torch.dtype] = {}

    def apply(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if name in self.layer_config:
                dtype = self.layer_config[name]
                self.original_dtypes[name] = next(module.parameters()).dtype
                module = module.to(dtype=dtype)
        return model

    def restore(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if name in self.original_dtypes:
                dtype = self.original_dtypes[name]
                module = module.to(dtype=dtype)
        return model

    def get_layer_precision(self, layer_name: str) -> torch.dtype:
        return self.layer_config.get(layer_name, self.default_dtype)


class QuantizationWrapper(nn.Module):
    def __init__(self, module: nn.Module, quantizer: BaseQuantizer):
        super().__init__()
        self.module = module
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantizer.quant_stub(x)
        x = self.module(x)
        x = self.quantizer.dequant_stub(x)
        return x


def apply_quantization(
    model: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    if config.quant_type == QuantizationType.INT8:
        quantizer = INT8Quantizer(config)
        return quantizer.prepare(model)
    elif config.quant_type == QuantizationType.FP16:
        quantizer = FP16Quantizer()
        return quantizer.to_fp16(model)
    elif config.quant_type == QuantizationType.BFLOAT16:
        quantizer = BFloat16Quantizer()
        return quantizer.to_bf16(model)
    elif config.quant_type == QuantizationType.MIXED:
        return model
    return model


def calibrate_model(
    model: nn.Module,
    dataloader: Any,
    num_batches: int = 10,
    device: str = "cpu",
) -> None:
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) for b in batch]
            else:
                batch = batch.to(device)
            model(batch)


def get_quantized_state_dict(model: nn.Module) -> Dict[str, Any]:
    return model.state_dict()


def fuse_model(model: nn.Module) -> nn.Module:
    fuse_modules = [
        (nn.Conv2d, nn.BatchNorm2d, "conv_bn"),
        (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, "conv_bn_relu"),
        (nn.Linear, nn.ReLU, "linear_relu"),
    ]
    for modules in fuse_modules:
        try:
            torch.quantization.fuse_modules(
                model, [m[0].__name__ for m in modules], inplace=True
            )
        except Exception:
            pass
    return model
