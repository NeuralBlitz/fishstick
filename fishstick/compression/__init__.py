"""
Model Compression Module for fishstick

Advanced model compression techniques including:
- Pruning: Magnitude, structured, lottery ticket, sensitivity-based
- Quantization: Dynamic, static, QAT, INT8/FP16, mixed precision
- Knowledge Distillation: Logit, feature, attention, progressive
- Sparsity: Unstructured, block, N:M (2:4), vector, semi-structured
- Utilities: Model size, MACs counting, compression ratios
"""

from typing import Optional, Dict, Any, Union, Callable
import torch
from torch import nn
import numpy as np
import copy

from .pruning import (
    MagnitudePruner,
    StructuredPruner,
    LotteryTicketPruner,
    GradualMagnitudeScheduler,
    SensitivityPruner,
    DependencyAwarePruner,
)

from .quantization import (
    FakeQuantizeModule,
    QuantizedLinear,
    QuantizedConv2d,
    DynamicQuantizer,
    StaticQuantizer,
    QuantizationAwareTrainer,
    MixedPrecisionQuantizer,
    FP16Quantizer,
    INT8Quantizer,
)

from .distillation import (
    LogitDistillation,
    FeatureDistillation,
    AttentionTransfer,
    RelationDistillation,
    ProgressiveDistillation,
    MultiStageDistillation,
    ContrastiveDistillation,
    ComprehensiveDistillation,
    DistillationTrainer,
)

from .sparsity import (
    UnstructuredSparsity,
    BlockSparsity,
    NMSparsity,
    TwoFourSparsity,
    VectorSparsity,
    SemiStructuredSparsity,
    SparsityScheduler,
)

from .compression_utils import (
    count_parameters,
    count_nonzero_parameters,
    get_model_sparsity,
    get_layer_sparsity,
    estimate_model_size,
    get_actual_model_size,
    count_model_macs,
    MACsCounter,
    calculate_compression_ratio,
    compare_models,
    get_model_summary,
    print_model_summary,
)


class Pruner:
    """
    High-level pruning interface combining multiple pruning methods.

    Supports magnitude-based, structured, and unstructured pruning.
    """

    def __init__(self, model: nn.Module, pruning_type: str = "magnitude"):
        self.model = model
        self.pruning_type = pruning_type
        self.pruned_params = {}

    def prune_magnitude(self, amount: float = 0.3) -> nn.Module:
        """Prune weights with smallest magnitudes."""
        pruned_model = copy.deepcopy(self.model)

        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                weight = module.weight.data.abs()
                threshold = torch.quantile(weight.flatten(), amount)
                mask = weight > threshold
                module.weight.data *= mask.float()

                self.pruned_params[name] = {
                    "original_shape": module.weight.shape,
                    "num_params": module.weight.numel(),
                    "num_pruned": (~mask).sum().item(),
                }

        return pruned_model

    def prune_structured(self, amount: float = 0.3, dim: int = 0) -> nn.Module:
        """Structured pruning (remove entire neurons/channels)."""
        pruned_model = copy.deepcopy(self.model)

        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                norms = weight.abs().sum(dim=1 - dim)

                k = int(amount * len(norms))
                threshold = torch.kthvalue(norms, k)[0] if k > 0 else norms.min() - 1

                mask = norms > threshold

                if dim == 0:
                    module.weight.data[mask == False] = 0
                    if module.bias is not None:
                        module.bias.data[mask == False] = 0
                else:
                    module.weight.data[:, mask == False] = 0

        return pruned_model

    def get_sparsity(self, model: nn.Module) -> float:
        """Calculate sparsity of model."""
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0


class Quantizer:
    """
    High-level quantization interface.

    Supports INT8 and FP16 quantization.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def quantize_dynamic(self, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """Dynamic quantization (activations quantized at runtime)."""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=dtype
        )
        return quantized_model

    def quantize_static(self, calibration_data: torch.Tensor) -> nn.Module:
        """Static quantization (requires calibration)."""
        model = copy.deepcopy(self.model)
        model.eval()

        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)

        with torch.no_grad():
            model(calibration_data)

        torch.quantization.convert(model, inplace=True)

        return model

    def to_fp16(self) -> nn.Module:
        """Convert model to half precision (FP16)."""
        return self.model.half()

    def get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            torch.save(model.state_dict(), f.name)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        return size


class KnowledgeDistiller:
    """
    High-level knowledge distillation interface.

    Transfer knowledge from teacher to student model.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

        self.teacher.eval()

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss."""
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=1)

        distillation_loss = nn.functional.kl_div(
            soft_prob, soft_targets, reduction="batchmean"
        ) * (self.temperature**2)

        hard_loss = nn.functional.cross_entropy(student_logits, labels)

        loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss

        return loss

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        """Single training step with distillation."""
        self.student.train()
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = self.teacher(x)

        student_logits = self.student(x)

        loss = self.distillation_loss(student_logits, teacher_logits, y)

        loss.backward()
        optimizer.step()

        return loss.item()


class ONNXExporter:
    """Export models to ONNX format for cross-platform deployment."""

    def __init__(self, model: nn.Module):
        self.model = model

    def export(
        self,
        output_path: str,
        input_shape: tuple = (1, 3, 224, 224),
        opset_version: int = 11,
    ) -> str:
        """Export model to ONNX."""
        try:
            import onnx
        except ImportError:
            raise ImportError("onnx not installed. Run: pip install onnx")

        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        return output_path


class TorchScriptCompiler:
    """Compile models to TorchScript for production."""

    def __init__(self, model: nn.Module):
        self.model = model

    def compile_script(self, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """Compile model using tracing."""
        self.model.eval()
        scripted_model = torch.jit.trace(self.model, example_input)
        return scripted_model

    def compile_annotate(self) -> torch.jit.ScriptModule:
        """Compile model using annotations."""
        return torch.jit.script(self.model)

    def save(self, scripted_model: torch.jit.ScriptModule, path: str) -> None:
        """Save compiled model."""
        scripted_model.save(path)

    def load(self, path: str) -> torch.jit.ScriptModule:
        """Load compiled model."""
        return torch.jit.load(path)


class ModelCompressor:
    """
    High-level model compression pipeline.

    Applies multiple optimization techniques in sequence.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.optimization_history = []

    def compress(
        self,
        prune_amount: float = 0.0,
        quantize: bool = False,
        sparsity_type: Optional[str] = None,
        sparsity_amount: float = 0.5,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Apply compression pipeline."""
        compressed_model = copy.deepcopy(self.model)

        if prune_amount > 0:
            pruner = MagnitudePruner(compressed_model, sparsity=prune_amount)
            pruner.prune()
            sparsity = get_model_sparsity(compressed_model)
            self.optimization_history.append(f"Pruned: {sparsity:.2%} sparsity")

        if sparsity_type is not None:
            if sparsity_type == "unstructured":
                sparsity = UnstructuredSparsity(sparsity_amount)
            elif sparsity_type == "block":
                sparsity = BlockSparsity(sparsity_amount)
            elif sparsity_type == "2:4":
                sparsity = TwoFourSparsity()
            else:
                sparsity = UnstructuredSparsity(sparsity_amount)

            sparsity.apply_to_model(compressed_model)
            self.optimization_history.append(f"Sparsity: {sparsity_type}")

        if quantize:
            quantizer = Quantizer(compressed_model)
            if calibration_data is not None:
                compressed_model = quantizer.quantize_static(calibration_data)
            else:
                compressed_model = quantizer.quantize_dynamic()
            self.optimization_history.append("Quantized: INT8")

        return compressed_model

    def benchmark(
        self, model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark model inference speed."""
        model.eval()

        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)

        import time

        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)

        elapsed = time.time() - start_time

        return {
            "total_time": elapsed,
            "avg_time": elapsed / num_runs,
            "throughput": num_runs / elapsed,
        }

    def get_compression_report(self) -> str:
        """Get compression history report."""
        return "\n".join(self.optimization_history)


def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Prune model weights."""
    pruner = Pruner(model)
    return pruner.prune_magnitude(amount)


def quantize_model(
    model: nn.Module, calibration_data: Optional[torch.Tensor] = None
) -> nn.Module:
    """Quantize model."""
    quantizer = Quantizer(model)
    if calibration_data is not None:
        return quantizer.quantize_static(calibration_data)
    return quantizer.quantize_dynamic()


def export_onnx(
    model: nn.Module, output_path: str, input_shape: tuple = (1, 3, 224, 224)
) -> str:
    """Export model to ONNX."""
    exporter = ONNXExporter(model)
    return exporter.export(output_path, input_shape)


def compile_torchscript(
    model: nn.Module, example_input: torch.Tensor
) -> torch.jit.ScriptModule:
    """Compile model to TorchScript."""
    compiler = TorchScriptCompiler(model)
    return compiler.compile_script(example_input)


def compress_model(
    model: nn.Module,
    prune_amount: float = 0.0,
    quantize: bool = False,
) -> nn.Module:
    """Apply full compression pipeline."""
    compressor = ModelCompressor(model)
    return compressor.compress(prune_amount=prune_amount, quantize=quantize)


__all__ = [
    "MagnitudePruner",
    "StructuredPruner",
    "LotteryTicketPruner",
    "GradualMagnitudeScheduler",
    "SensitivityPruner",
    "DependencyAwarePruner",
    "FakeQuantizeModule",
    "QuantizedLinear",
    "QuantizedConv2d",
    "DynamicQuantizer",
    "StaticQuantizer",
    "QuantizationAwareTrainer",
    "MixedPrecisionQuantizer",
    "FP16Quantizer",
    "INT8Quantizer",
    "LogitDistillation",
    "FeatureDistillation",
    "AttentionTransfer",
    "RelationDistillation",
    "ProgressiveDistillation",
    "MultiStageDistillation",
    "ContrastiveDistillation",
    "ComprehensiveDistillation",
    "DistillationTrainer",
    "UnstructuredSparsity",
    "BlockSparsity",
    "NMSparsity",
    "TwoFourSparsity",
    "VectorSparsity",
    "SemiStructuredSparsity",
    "SparsityScheduler",
    "count_parameters",
    "count_nonzero_parameters",
    "get_model_sparsity",
    "get_layer_sparsity",
    "estimate_model_size",
    "get_actual_model_size",
    "count_model_macs",
    "MACsCounter",
    "calculate_compression_ratio",
    "compare_models",
    "get_model_summary",
    "print_model_summary",
    "Pruner",
    "Quantizer",
    "KnowledgeDistiller",
    "ONNXExporter",
    "TorchScriptCompiler",
    "ModelCompressor",
    "prune_model",
    "quantize_model",
    "export_onnx",
    "compile_torchscript",
    "compress_model",
]
