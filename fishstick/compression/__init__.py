"""
Model Compression and Optimization Module for fishstick

Provides tools for:
- Pruning: Remove redundant weights
- Quantization: Reduce precision for faster inference
- Knowledge Distillation: Transfer to smaller models
- ONNX Export: Cross-platform deployment
- TorchScript: Production optimization
"""

from typing import Optional, Dict, Any, Union, Callable
import torch
from torch import nn
import numpy as np
import copy


class Pruner:
    """
    Model pruning for removing redundant parameters.

    Supports magnitude-based, structured, and unstructured pruning.
    """

    def __init__(self, model: nn.Module, pruning_type: str = "magnitude"):
        """
        Args:
            model: Model to prune
            pruning_type: "magnitude", "structured", or "random"
        """
        self.model = model
        self.pruning_type = pruning_type
        self.pruned_params = {}

    def prune_magnitude(self, amount: float = 0.3) -> nn.Module:
        """
        Prune weights with smallest magnitudes.

        Args:
            amount: Fraction of weights to prune (0-1)

        Returns:
            Pruned model
        """
        pruned_model = copy.deepcopy(self.model)

        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                # Get weight magnitude
                weight = module.weight.data.abs()

                # Compute threshold
                threshold = torch.quantile(weight.flatten(), amount)

                # Create mask
                mask = weight > threshold

                # Apply mask
                module.weight.data *= mask.float()

                # Store pruning info
                self.pruned_params[name] = {
                    "original_shape": module.weight.shape,
                    "num_params": module.weight.numel(),
                    "num_pruned": (~mask).sum().item(),
                }

        return pruned_model

    def prune_structured(self, amount: float = 0.3, dim: int = 0) -> nn.Module:
        """
        Structured pruning (remove entire neurons/channels).

        Args:
            amount: Fraction to prune
            dim: Dimension to prune (0 for output, 1 for input)

        Returns:
            Pruned model
        """
        pruned_model = copy.deepcopy(self.model)

        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                # Compute L1 norm of each neuron/channel
                weight = module.weight.data
                norms = weight.abs().sum(dim=1 - dim)

                # Compute threshold
                k = int(amount * len(norms))
                threshold = torch.kthvalue(norms, k)[0] if k > 0 else norms.min() - 1

                # Create mask
                mask = norms > threshold

                # Apply mask
                if dim == 0:  # Prune output neurons
                    module.weight.data[mask == False] = 0
                    if module.bias is not None:
                        module.bias.data[mask == False] = 0
                else:  # Prune input connections
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
    Model quantization for efficient inference.

    Supports INT8 and FP16 quantization.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def quantize_dynamic(self, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        Dynamic quantization (activations quantized at runtime).

        Args:
            dtype: Quantization type (torch.qint8 or torch.float16)

        Returns:
            Quantized model
        """
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=dtype
        )
        return quantized_model

    def quantize_static(self, calibration_data: torch.Tensor) -> nn.Module:
        """
        Static quantization (requires calibration).

        Args:
            calibration_data: Sample data for calibration

        Returns:
            Quantized model
        """
        model = copy.deepcopy(self.model)
        model.eval()

        # Configure quantization
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

        # Prepare for quantization
        torch.quantization.prepare(model, inplace=True)

        # Calibrate
        with torch.no_grad():
            model(calibration_data)

        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)

        return model

    def to_fp16(self) -> nn.Module:
        """Convert model to half precision (FP16)."""
        return self.model.half()

    def get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes."""
        torch.save(model.state_dict(), "/tmp/temp_model.pt")
        size = Path("/tmp/temp_model.pt").stat().st_size
        Path("/tmp/temp_model.pt").unlink()
        return size


class KnowledgeDistiller:
    """
    Knowledge distillation for training smaller models.

    Transfer knowledge from teacher to student model.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        """
        Args:
            teacher: Teacher model (large, trained)
            student: Student model (small, to be trained)
            temperature: Softmax temperature for distillation
            alpha: Weight for distillation loss (vs hard targets)
        """
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
        """
        Compute distillation loss.

        Loss = α * soft_cross_entropy + (1-α) * hard_cross_entropy
        """
        # Soft targets from teacher
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=1)

        # Distillation loss (KL divergence)
        distillation_loss = nn.functional.kl_div(
            soft_prob, soft_targets, reduction="batchmean"
        ) * (self.temperature**2)

        # Hard targets loss
        hard_loss = nn.functional.cross_entropy(student_logits, labels)

        # Combined loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss

        return loss

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        """Single training step with distillation."""
        self.student.train()
        optimizer.zero_grad()

        # Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        # Get student predictions
        student_logits = self.student(x)

        # Compute loss
        loss = self.distillation_loss(student_logits, teacher_logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()


class ONNXExporter:
    """
    Export models to ONNX format for cross-platform deployment.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def export(
        self,
        output_path: str,
        input_shape: tuple = (1, 3, 224, 224),
        opset_version: int = 11,
    ) -> str:
        """
        Export model to ONNX.

        Args:
            output_path: Path to save ONNX model
            input_shape: Input tensor shape
            opset_version: ONNX opset version

        Returns:
            Path to exported model
        """
        try:
            import onnx
        except ImportError:
            raise ImportError("onnx not installed. Run: pip install onnx")

        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Export
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

        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        return output_path


class TorchScriptCompiler:
    """
    Compile models to TorchScript for production.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compile_script(self, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """
        Compile model using tracing.

        Args:
            example_input: Example input for tracing

        Returns:
            Compiled TorchScript model
        """
        self.model.eval()
        scripted_model = torch.jit.trace(self.model, example_input)
        return scripted_model

    def compile_annotate(self) -> torch.jit.ScriptModule:
        """
        Compile model using annotations (if model supports it).

        Returns:
            Compiled TorchScript model
        """
        return torch.jit.script(self.model)

    def save(self, scripted_model: torch.jit.ScriptModule, path: str) -> None:
        """Save compiled model."""
        scripted_model.save(path)

    def load(self, path: str) -> torch.jit.ScriptModule:
        """Load compiled model."""
        return torch.jit.load(path)


class ModelOptimizer:
    """
    High-level model optimization pipeline.

    Applies multiple optimization techniques in sequence.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.optimization_history = []

    def optimize(
        self,
        prune_amount: float = 0.0,
        quantize: bool = False,
        compile_torchscript: bool = False,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """
        Apply optimization pipeline.

        Args:
            prune_amount: Amount to prune (0-1)
            quantize: Whether to quantize
            compile_torchscript: Whether to compile to TorchScript
            calibration_data: Data for calibration (if quantizing)

        Returns:
            Optimized model
        """
        optimized_model = self.model

        # Pruning
        if prune_amount > 0:
            pruner = Pruner(optimized_model)
            optimized_model = pruner.prune_magnitude(prune_amount)
            sparsity = pruner.get_sparsity(optimized_model)
            self.optimization_history.append(f"Pruned: {sparsity:.2%} sparsity")

        # Quantization
        if quantize:
            quantizer = Quantizer(optimized_model)
            if calibration_data is not None:
                optimized_model = quantizer.quantize_static(calibration_data)
            else:
                optimized_model = quantizer.quantize_dynamic()
            self.optimization_history.append("Quantized: INT8")

        # TorchScript compilation
        if compile_torchscript:
            compiler = TorchScriptCompiler(optimized_model)
            dummy_input = torch.randn(1, 3, 224, 224)
            optimized_model = compiler.compile_script(dummy_input)
            self.optimization_history.append("Compiled: TorchScript")

        return optimized_model

    def benchmark(
        self, model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark model inference speed.

        Args:
            model: Model to benchmark
            input_tensor: Input for inference
            num_runs: Number of inference runs

        Returns:
            Benchmark results
        """
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)

        # Benchmark
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


# Convenience functions
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


def optimize_model(
    model: nn.Module, prune_amount: float = 0.0, quantize: bool = False
) -> nn.Module:
    """Apply full optimization pipeline."""
    optimizer = ModelOptimizer(model)
    return optimizer.optimize(prune_amount=prune_amount, quantize=quantize)
