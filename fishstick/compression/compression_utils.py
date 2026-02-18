"""
Compression Utilities

Model size estimation, MACs counting, and compression ratio calculation.
"""

from typing import Optional, List, Dict, Tuple, Union, Callable
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from collections import OrderedDict
import re


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    """Count the number of non-zero parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of non-zero parameters
    """
    total = 0
    for param in model.parameters():
        total += (param != 0).sum().item()
    return total


def get_model_sparsity(model: nn.Module) -> float:
    """Calculate the sparsity ratio of a model.

    Args:
        model: PyTorch model

    Returns:
        Sparsity ratio (0-1)
    """
    total = count_parameters(model, trainable_only=False)
    nonzero = count_nonzero_parameters(model)
    return 1.0 - nonzero / total if total > 0 else 0.0


def get_layer_sparsity(model: nn.Module) -> Dict[str, float]:
    """Get sparsity for each layer.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping layer names to sparsity ratios
    """
    sparsity = {}
    for name, param in model.named_parameters():
        if param.numel() > 0:
            nonzero = (param != 0).sum().item()
            sparsity[name] = 1.0 - nonzero / param.numel()
    return sparsity


def estimate_model_size(
    model: nn.Module,
    bits: int = 32,
    include_buffers: bool = True,
) -> Dict[str, float]:
    """Estimate model size in different units.

    Args:
        model: PyTorch model
        bits: Bits per parameter (32 for FP32, 16 for FP16, 8 for INT8)
        include_buffers: Include buffer tensors in estimation

    Returns:
        Dictionary with size estimates in bytes, KB, MB, GB
    """
    param_size = sum(p.numel() * bits // 8 for p in model.parameters())

    buffer_size = 0
    if include_buffers:
        buffer_size = sum(b.numel() * bits // 8 for b in model.buffers())

    total_bytes = param_size + buffer_size

    return {
        "bytes": total_bytes,
        "kb": total_bytes / 1024,
        "mb": total_bytes / (1024 * 1024),
        "gb": total_bytes / (1024 * 1024 * 1024),
        "param_bytes": param_size,
        "buffer_bytes": buffer_size,
    }


def get_actual_model_size(model: nn.Module) -> float:
    """Get actual model size by saving to temporary file.

    Args:
        model: PyTorch model

    Returns:
        Size in megabytes
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        temp_path = f.name

    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.unlink(temp_path)

    return size_mb


def count_conv2d_macs(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    input_size: Tuple[int, int],
    groups: int = 1,
    bias: bool = True,
) -> int:
    """Calculate MACs for a Conv2d layer.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size (height, width)
        input_size: Input spatial size (height, width)
        groups: Number of groups
        bias: Whether layer has bias

    Returns:
        Number of MACs
    """
    kh, kw = kernel_size
    ih, iw = input_size
    oh = ih - kh + 1
    ow = iw - kw + 1

    conv_macs = (kh * kw * in_channels // groups) * out_channels * oh * ow

    bias_macs = out_channels * oh * ow if bias else 0

    return conv_macs + bias_macs


def count_linear_macs(in_features: int, out_features: int, bias: bool = True) -> int:
    """Calculate MACs for a Linear layer.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether layer has bias

    Returns:
        Number of MACs
    """
    matmul_macs = in_features * out_features
    bias_macs = out_features if bias else 0
    return matmul_macs + bias_macs


def count_bn_macs(num_features: int, spatial_size: Tuple[int, int]) -> int:
    """Calculate MACs for BatchNorm layer.

    Args:
        num_features: Number of features/channels
        spatial_size: Spatial size (height, width)

    Returns:
        Number of MACs
    """
    h, w = spatial_size
    return 4 * num_features * h * w


def count_attention_macs(
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    include_output_proj: bool = True,
) -> int:
    """Calculate MACs for multi-head attention.

    Args:
        seq_len: Sequence length
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        include_output_proj: Include output projection

    Returns:
        Number of MACs
    """
    head_dim = embed_dim // num_heads

    qkv_macs = 3 * seq_len * embed_dim * embed_dim

    attn_scores_macs = num_heads * seq_len * seq_len * head_dim

    attn_output_macs = num_heads * seq_len * seq_len * head_dim

    output_proj_macs = seq_len * embed_dim * embed_dim if include_output_proj else 0

    return qkv_macs + attn_scores_macs + attn_output_macs + output_proj_macs


class MACsCounter:
    """Counter for Multiply-Accumulate operations (MACs/FLOPs).

    Args:
        model: PyTorch model
        input_size: Input tensor size (excluding batch dimension)
    """

    def __init__(self, model: nn.Module, input_size: Tuple[int, ...]):
        self.model = model
        self.input_size = input_size
        self.hooks = []
        self.layer_macs = OrderedDict()
        self._feature_sizes = {}

    def count(self) -> Dict[str, int]:
        """Count MACs for the model.

        Returns:
            Dictionary with layer-wise and total MACs
        """
        self._register_hooks()

        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, *self.input_size).to(device)

        self.model.eval()
        with torch.no_grad():
            _ = self.model(dummy_input)

        self._remove_hooks()

        total_macs = sum(self.layer_macs.values())

        return {
            "total_macs": total_macs,
            "total_flops": total_macs * 2,
            "layer_macs": dict(self.layer_macs),
            "total_gmacs": total_macs / 1e9,
            "total_mflops": total_macs * 2 / 1e6,
        }

    def _register_hooks(self):
        """Register forward hooks to count MACs."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(self._conv2d_hook(name))
                self.hooks.append(hook)
            elif isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._linear_hook(name))
                self.hooks.append(hook)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                hook = module.register_forward_hook(self._bn_hook(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _conv2d_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            batch_size = input[0].size(0)
            in_channels = input[0].size(1)
            out_channels = output.size(1)
            kernel_size = module.kernel_size

            ih, iw = input[0].size(2), input[0].size(3)
            oh, ow = output.size(2), output.size(3)

            kernel_ops = kernel_size[0] * kernel_size[1] * in_channels // module.groups
            output_elements = batch_size * out_channels * oh * ow

            macs = kernel_ops * output_elements

            if module.bias is not None:
                macs += out_channels * oh * ow * batch_size

            self.layer_macs[name] = macs

        return hook

    def _linear_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            batch_size = input[0].size(0)
            in_features = input[0].size(-1)
            out_features = output.size(-1)

            macs = batch_size * in_features * out_features

            if module.bias is not None:
                macs += batch_size * out_features

            self.layer_macs[name] = macs

        return hook

    def _bn_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            batch_size = input[0].size(0)
            num_features = input[0].size(1)

            spatial_size = input[0].numel() // (batch_size * num_features)

            macs = 4 * batch_size * num_features * spatial_size

            self.layer_macs[name] = macs

        return hook


def count_model_macs(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, int]:
    """Convenience function to count MACs.

    Args:
        model: PyTorch model
        input_size: Input tensor size (excluding batch dimension)

    Returns:
        Dictionary with MACs information
    """
    counter = MACsCounter(model, input_size)
    return counter.count()


def calculate_compression_ratio(
    original_size: float,
    compressed_size: float,
) -> Dict[str, float]:
    """Calculate compression metrics.

    Args:
        original_size: Original model size (in bytes or parameter count)
        compressed_size: Compressed model size (same unit as original)

    Returns:
        Dictionary with compression metrics
    """
    ratio = original_size / compressed_size if compressed_size > 0 else float("inf")
    reduction = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0

    return {
        "compression_ratio": ratio,
        "compression_ratio_x": f"{ratio:.2f}x",
        "size_reduction_percent": reduction,
        "space_savings": original_size - compressed_size,
    }


def compare_models(
    original: nn.Module,
    compressed: nn.Module,
    input_size: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Union[int, float, str]]:
    """Compare original and compressed models.

    Args:
        original: Original model
        compressed: Compressed model
        input_size: Optional input size for MACs calculation

    Returns:
        Dictionary with comparison metrics
    """
    orig_params = count_parameters(original, trainable_only=False)
    comp_params = count_parameters(compressed, trainable_only=False)

    orig_size = estimate_model_size(original)
    comp_size = estimate_model_size(compressed)

    orig_sparsity = get_model_sparsity(original)
    comp_sparsity = get_model_sparsity(compressed)

    comparison = {
        "original_params": orig_params,
        "compressed_params": comp_params,
        "param_reduction": orig_params - comp_params,
        "param_reduction_percent": (1 - comp_params / orig_params) * 100
        if orig_params > 0
        else 0,
        "original_size_mb": orig_size["mb"],
        "compressed_size_mb": comp_size["mb"],
        "size_reduction_mb": orig_size["mb"] - comp_size["mb"],
        "original_sparsity": orig_sparsity,
        "compressed_sparsity": comp_sparsity,
    }

    comparison.update(
        calculate_compression_ratio(orig_size["bytes"], comp_size["bytes"])
    )

    if input_size is not None:
        try:
            orig_macs = count_model_macs(original, input_size)
            comp_macs = count_model_macs(compressed, input_size)
            comparison["original_macs"] = orig_macs["total_macs"]
            comparison["compressed_macs"] = comp_macs["total_macs"]
            comparison["macs_reduction_percent"] = (
                (1 - comp_macs["total_macs"] / orig_macs["total_macs"]) * 100
                if orig_macs["total_macs"] > 0
                else 0
            )
        except Exception:
            pass

    return comparison


def get_model_summary(
    model: nn.Module,
    input_size: Optional[Tuple[int, ...]] = None,
) -> str:
    """Generate a summary of the model.

    Args:
        model: PyTorch model
        input_size: Optional input size for MACs calculation

    Returns:
        Summary string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append("=" * 80)

    total_params = 0
    trainable_params = 0

    lines.append(f"\n{'Layer':<40} {'Output Shape':<25} {'Params':<15}")
    lines.append("-" * 80)

    def get_output_shape(output):
        if isinstance(output, Tensor):
            return tuple(output.shape)
        return str(type(output))

    def register_hook(module):
        def hook(module, input, output):
            nonlocal total_params, trainable_params

            class_name = module.__class__.__name__
            module_name = ""
            for name, m in model.named_modules():
                if m is module:
                    module_name = name
                    break

            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

            total_params += params
            trainable_params += trainable

            output_shape = get_output_shape(output)

            if params > 0:
                lines.append(
                    f"{module_name[:40]:<40} {str(output_shape)[:25]:<25} {params:>15,}"
                )

        return hook

    hooks = []
    for module in model.modules():
        if len(list(module.children())) == 0:
            h = module.register_forward_hook(register_hook(module))
            hooks.append(h)

    if input_size is not None:
        device = (
            next(model.parameters()).device
            if len(list(model.parameters())) > 0
            else "cpu"
        )
        dummy_input = torch.randn(1, *input_size).to(device)
        with torch.no_grad():
            _ = model(dummy_input)

    for h in hooks:
        h.remove()

    lines.append("-" * 80)
    lines.append(f"Total Parameters: {total_params:,}")
    lines.append(f"Trainable Parameters: {trainable_params:,}")
    lines.append(f"Non-trainable Parameters: {total_params - trainable_params:,}")

    size_info = estimate_model_size(model)
    lines.append(f"Model Size: {size_info['mb']:.2f} MB")

    sparsity = get_model_sparsity(model)
    lines.append(f"Sparsity: {sparsity:.2%}")

    if input_size is not None:
        try:
            macs_info = count_model_macs(model, input_size)
            lines.append(
                f"MACs: {macs_info['total_macs']:,} ({macs_info['total_gmacs']:.2f} GMACs)"
            )
            lines.append(
                f"FLOPs: {macs_info['total_flops']:,} ({macs_info['total_mflops']:.2f} MFLOPs)"
            )
        except Exception:
            pass

    lines.append("=" * 80)

    return "\n".join(lines)


def print_model_summary(model: nn.Module, input_size: Optional[Tuple[int, ...]] = None):
    """Print model summary.

    Args:
        model: PyTorch model
        input_size: Optional input size for MACs calculation
    """
    print(get_model_summary(model, input_size))


__all__ = [
    "count_parameters",
    "count_nonzero_parameters",
    "get_model_sparsity",
    "get_layer_sparsity",
    "estimate_model_size",
    "get_actual_model_size",
    "count_conv2d_macs",
    "count_linear_macs",
    "count_bn_macs",
    "count_attention_macs",
    "MACsCounter",
    "count_model_macs",
    "calculate_compression_ratio",
    "compare_models",
    "get_model_summary",
    "print_model_summary",
]
