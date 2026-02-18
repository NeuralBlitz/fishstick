from typing import Optional, Dict, Any, List, Tuple, Literal, Callable
from enum import Enum
from contextlib import contextmanager

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class QuantType(Enum):
    INT8 = "int8"
    FP16 = "fp16"
    BFLOAT16 = "bfloat16"
    FLOAT8 = "float8"


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    TF32 = "tf32"


class QuantizationConfig:
    def __init__(
        self,
        quant_type: QuantType = QuantType.INT8,
        per_channel: bool = True,
        per_tensor: bool = False,
        calibration_method: Literal["minmax", "histogram", "percentile"] = "minmax",
        calibration_samples: int = 512,
        fold_bn: bool = True,
        fuse_modules: bool = True,
    ):
        self.quant_type = quant_type
        self.per_channel = per_channel
        self.per_tensor = per_tensor
        self.calibration_method = calibration_method
        self.calibration_samples = calibration_samples
        self.fold_bn = fold_bn
        self.fuse_modules = fuse_modules

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quant_type": self.quant_type.value,
            "per_channel": self.per_channel,
            "per_tensor": self.per_tensor,
            "calibration_method": self.calibration_method,
            "calibration_samples": self.calibration_samples,
            "fold_bn": self.fold_bn,
            "fuse_modules": self.fuse_modules,
        }


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: QuantType = QuantType.INT8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_scale", torch.ones(out_features))
        self.register_buffer("weight_zero_point", torch.zeros(out_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0, mode="fan_in")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, input_scale: Optional[Tensor] = None) -> Tensor:
        if input_scale is not None:
            x = x * input_scale

        if self.quant_type == QuantType.INT8:
            weight_int = torch.round(self.weight * self.weight_scale).to(torch.int8)
            x_int = x.to(torch.int32)
            output = torch.nn.functional.linear(x_int, weight_int, self.bias)
            output = output / (
                self.weight_scale * (input_scale if input_scale is not None else 1.0)
            )
        else:
            output = torch.nn.functional.linear(x, self.weight, self.bias)

        return output

    def set_quantized_weights(
        self, weight: Tensor, scale: Tensor, zero_point: Optional[Tensor] = None
    ):
        self.weight.data = weight
        self.weight_scale.data = scale
        if zero_point is not None:
            self.weight_zero_point.data = zero_point


class DynamicQuantizer:
    def __init__(self, quant_type: QuantType = QuantType.INT8):
        self.quant_type = quant_type

    def quantize_tensor(self, tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.quant_type == QuantType.INT8:
            scale = tensor.abs().max() / 127.0
            if scale > 0:
                quantized = torch.round(tensor / scale).to(torch.int8)
            else:
                quantized = torch.zeros_like(tensor, dtype=torch.int8)
            zero_point = torch.zeros((), dtype=torch.int8)
            return quantized, scale, zero_point

        elif self.quant_type == QuantType.FP16:
            return tensor.half(), torch.ones_like(tensor), torch.zeros_like(tensor)

        elif self.quant_type == QuantType.BFLOAT16:
            return tensor.bfloat16(), torch.ones_like(tensor), torch.zeros_like(tensor)

        return tensor, torch.ones_like(tensor), torch.zeros_like(tensor)

    def dequantize_tensor(
        self, quantized: Tensor, scale: Tensor, zero_point: Tensor
    ) -> Tensor:
        if self.quant_type == QuantType.INT8:
            return (quantized.float() * scale) + zero_point.float()
        return quantized.float()


class StaticQuantizer:
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.calibration_data: List[Tensor] = []
        self._observer_ready = False

    def prepare(self, model: nn.Module) -> nn.Module:
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Linear):
                new_module = QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    self.config.quant_type,
                )
                new_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_module.bias.data = module.bias.data.clone()
                module.replace_module(new_module)
        return model

    def collect_stats(self, model: nn.Module, dataloader: Any, num_batches: int = 10):
        model.eval()
        self.calibration_data = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(next(model.parameters()).device)
                _ = model(x)
                self.calibration_data.append(x)

    def calculate_scales(self, model: nn.Module) -> Dict[str, Tuple[Tensor, Tensor]]:
        scales = {}

        if self.config.calibration_method == "minmax":
            for i, data in enumerate(self.calibration_data):
                if i == 0:
                    min_vals = data.min()
                    max_vals = data.max()
                else:
                    min_vals = torch.min(min_vals, data.min())
                    max_vals = torch.max(max_vals, data.max())

            for name, module in model.named_modules():
                if isinstance(module, QuantizedLinear):
                    w_min = module.weight.min()
                    w_max = module.weight.max()
                    w_scale = (w_max - w_min) / 255.0
                    scales[f"{name}.weight"] = (w_scale, torch.zeros_like(w_scale))

        return scales


class MixedPrecisionManager:
    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        loss_scale: Optional[float] = None,
        dynamic: bool = True,
    ):
        self.dtype = dtype
        self.loss_scale = loss_scale
        self.dynamic = dynamic
        self.scaler = None

        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=loss_scale or 2**16,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000 if dynamic else float("inf"),
            )

    @contextmanager
    def autocast(self, device_type: str = "cuda"):
        if torch.cuda.is_available() and device_type == "cuda":
            with torch.cuda.amp.autocast(dtype=self.dtype):
                yield
        else:
            yield

    def scale_loss(self, loss: Tensor) -> Tensor:
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: Any, closure: Optional[Callable] = None):
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        elif closure is not None:
            closure()

    def update(self):
        if self.scaler is not None:
            self.scaler.update()


class BFloat16Manager:
    @staticmethod
    def convert_module(module: nn.Module) -> nn.Module:
        for name, param in module.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
        for name, buffer in module.named_buffers():
            if buffer.dtype == torch.float32:
                buffer.data = buffer.data.to(torch.bfloat16)
        return module

    @staticmethod
    def convert_back(module: nn.Module) -> nn.Module:
        for name, param in module.named_parameters():
            if param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float32)
        for name, buffer in module.named_buffers():
            if buffer.dtype == torch.bfloat16:
                buffer.data = buffer.data.to(torch.float32)
        return module

    @contextmanager
    def bfloat16_context(self):
        old_dtypes = {}
        for name, param in self._module.named_parameters():
            old_dtypes[name] = param.dtype
            param.data = param.data.to(torch.bfloat16)

        try:
            yield
        finally:
            for name, dtype in old_dtypes.items():
                if name in self._module.state_dict():
                    self._module.state_dict()[name].data = self._module.state_dict()[
                        name
                    ].data.to(dtype)

    def __init__(self, module: nn.Module):
        self._module = module


def quantize_model(
    model: nn.Module,
    quant_type: QuantType = QuantType.INT8,
    calibration_config: Optional[QuantizationConfig] = None,
) -> nn.Module:
    if quant_type == QuantType.INT8:
        if calibration_config is None:
            calibration_config = QuantizationConfig(quant_type=quant_type)
        quantizer = StaticQuantizer(calibration_config)
        return quantizer.prepare(model)

    elif quant_type == QuantType.FP16:
        return model.half()

    elif quant_type == QuantType.BFLOAT16:
        for module in model.modules():
            for param in module.parameters():
                if param.dtype == torch.float32:
                    param.data = param.data.to(torch.bfloat16)
        return model

    return model


def dynamic_quantize(
    model: nn.Module,
    quant_type: QuantType = QuantType.INT8,
) -> nn.Module:
    if quant_type == QuantType.INT8:
        return torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
    return model


def convert_to_fp16(model: nn.Module, inplace: bool = True) -> nn.Module:
    if not inplace:
        model = model.clone()
    return model.half()


def convert_to_bf16(model: nn.Module, inplace: bool = True) -> nn.Module:
    if not inplace:
        model = model.clone()
    for module in model.modules():
        for param in module.parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
    return model


def convert_to_fp32(model: nn.Module, inplace: bool = True) -> nn.Module:
    if not inplace:
        model = model.clone()
    return model.float()


def get_quantization_stats(model: nn.Module) -> Dict[str, Any]:
    stats = {
        "total_params": 0,
        "fp32_params": 0,
        "fp16_params": 0,
        "bf16_params": 0,
        "int8_params": 0,
    }

    for param in model.parameters():
        stats["total_params"] += param.numel()
        dtype = param.dtype
        if dtype == torch.float32:
            stats["fp32_params"] += param.numel()
        elif dtype == torch.float16:
            stats["fp16_params"] += param.numel()
        elif dtype == torch.bfloat16:
            stats["bf16_params"] += param.numel()
        elif dtype in (torch.int8, torch.qint8):
            stats["int8_params"] += param.numel()

    return stats
