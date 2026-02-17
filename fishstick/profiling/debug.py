"""
Model Profiling and Debugging Tools

Profile model performance, detect issues, and debug training problems.
"""

from typing import Optional, Dict, List, Tuple, Callable
import torch
import torch.nn as nn
import time
import sys
import warnings
from collections import defaultdict
import numpy as np
from contextlib import contextmanager


class ModelProfiler:
    """
    Profile model performance - memory usage, inference time, FLOPs.

    Example:
        >>> profiler = ModelProfiler(model)
        >>> profiler.profile(input_shape=(1, 3, 224, 224))
        >>> profiler.print_summary()
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.layer_times = {}
        self.layer_memory = {}
        self.hooks = []

    def profile(
        self, input_shape: Tuple[int, ...], num_runs: int = 100, warmup: int = 10
    ) -> Dict[str, float]:
        """
        Profile model performance.

        Args:
            input_shape: Shape of input tensor
            num_runs: Number of profiling runs
            warmup: Number of warmup runs

        Returns:
            Dictionary of profiling results
        """
        dummy_input = torch.randn(input_shape).to(self.device)

        # Register hooks for layer-wise profiling
        self._register_hooks()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # Profile
        self.layer_times = defaultdict(list)
        self.layer_memory = {}

        start_mem = torch.cuda.memory_allocated() if self.device == "cuda" else 0

        with torch.no_grad():
            for _ in range(num_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()

                start_time = time.time()
                _ = self.model(dummy_input)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                end_time = time.time()
                self.layer_times["total"].append((end_time - start_time) * 1000)

        end_mem = torch.cuda.memory_allocated() if self.device == "cuda" else 0

        # Remove hooks
        self._remove_hooks()

        # Compile results
        results = {
            "avg_inference_time_ms": np.mean(self.layer_times["total"]),
            "std_inference_time_ms": np.std(self.layer_times["total"]),
            "min_inference_time_ms": np.min(self.layer_times["total"]),
            "max_inference_time_ms": np.max(self.layer_times["total"]),
            "memory_allocated_mb": (end_mem - start_mem) / 1024**2
            if self.device == "cuda"
            else 0,
        }

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        results["total_params"] = total_params
        results["trainable_params"] = trainable_params
        results["model_size_mb"] = total_params * 4 / 1024**2  # Assuming float32

        return results

    def _register_hooks(self):
        """Register forward hooks to profile each layer."""

        def get_hook(name):
            def hook(module, input, output):
                if self.device == "cuda":
                    torch.cuda.synchronize()

            return hook

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                hook = module.register_forward_hook(get_hook(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def print_summary(self):
        """Print profiling summary."""
        print("\n" + "=" * 60)
        print("Model Profiling Summary")
        print("=" * 60)

        # Print layer-wise info if available
        if self.layer_times:
            for name, times in self.layer_times.items():
                if name != "total" and times:
                    avg_time = np.mean(times)
                    print(f"{name:40s}: {avg_time:8.3f} ms")

        print("\nTotal:")
        if "total" in self.layer_times:
            total_times = self.layer_times["total"]
            print(
                f"  Average inference time: {np.mean(total_times):.3f} ± {np.std(total_times):.3f} ms"
            )
            print(
                f"  Min/Max: {np.min(total_times):.3f} / {np.max(total_times):.3f} ms"
            )

        print(f"\nMemory:")
        print(f"  Device: {self.device}")
        if self.device == "cuda":
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    def count_flops(self, input_shape: Tuple[int, ...]) -> int:
        """
        Estimate FLOPs (simplified).

        Args:
            input_shape: Input tensor shape

        Returns:
            Estimated FLOPs
        """
        total_flops = 0

        def conv_flops(module, input, output):
            batch_size = output.shape[0]
            output_height, output_width = output.shape[2], output.shape[3]
            kernel_height, kernel_width = module.kernel_size
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups

            filters_per_channel = out_channels // groups
            conv_per_position_flops = (
                kernel_height * kernel_width * in_channels // groups
            )

            active_elements_count = batch_size * output_height * output_width
            overall_conv_flops = (
                conv_per_position_flops * active_elements_count * filters_per_channel
            )

            bias_flops = 0
            if module.bias is not None:
                bias_flops = out_channels * active_elements_count

            return overall_conv_flops + bias_flops

        def linear_flops(module, input, output):
            # input shape: (batch_size, in_features)
            # output shape: (batch_size, out_features)
            batch_size = input[0].shape[0]
            in_features = module.in_features
            out_features = module.out_features

            return batch_size * in_features * out_features

        # Register hooks to count FLOPs
        flops = []

        def make_hook(flop_fn):
            def hook(module, input, output):
                flops.append(flop_fn(module, input, output))

            return hook

        hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(make_hook(conv_flops)))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(linear_flops)))

        # Forward pass
        dummy_input = torch.randn(input_shape).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return sum(flops)


class GradientChecker:
    """
    Check for gradient issues like vanishing/exploding gradients.

    Example:
        >>> checker = GradientChecker(model)
        >>> stats = checker.check_gradients(dataloader)
        >>> checker.report_issues()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_stats = defaultdict(list)
        self.issues = []

    def check_gradients(
        self,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        num_batches: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Check gradients over multiple batches.

        Args:
            data_loader: Data loader
            loss_fn: Loss function
            num_batches: Number of batches to check

        Returns:
            Dictionary of gradient statistics per layer
        """
        self.model.train()
        self.gradient_stats = defaultdict(list)

        for i, (data, target) in enumerate(data_loader):
            if i >= num_batches:
                break

            self.model.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            loss.backward()

            # Collect gradient statistics
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.data
                    self.gradient_stats[name].append(
                        {
                            "mean": grad.abs().mean().item(),
                            "std": grad.std().item(),
                            "max": grad.abs().max().item(),
                            "min": grad.abs().min().item(),
                            "norm": grad.norm().item(),
                        }
                    )

        # Compute statistics
        summary = {}
        for name, stats_list in self.gradient_stats.items():
            summary[name] = {
                "mean": np.mean([s["mean"] for s in stats_list]),
                "std": np.mean([s["std"] for s in stats_list]),
                "max": np.max([s["max"] for s in stats_list]),
                "min": np.min([s["min"] for s in stats_list]),
            }

        return summary

    def report_issues(
        self, vanishing_threshold: float = 1e-7, exploding_threshold: float = 1e3
    ):
        """
        Report potential gradient issues.

        Args:
            vanishing_threshold: Threshold for vanishing gradients
            exploding_threshold: Threshold for exploding gradients
        """
        self.issues = []

        for name, stats_list in self.gradient_stats.items():
            max_grad = max(s["max"] for s in stats_list)
            mean_grad = np.mean([s["mean"] for s in stats_list])

            if max_grad > exploding_threshold:
                self.issues.append(
                    {
                        "type": "exploding",
                        "layer": name,
                        "value": max_grad,
                        "severity": "high"
                        if max_grad > exploding_threshold * 10
                        else "medium",
                    }
                )

            if mean_grad < vanishing_threshold:
                self.issues.append(
                    {
                        "type": "vanishing",
                        "layer": name,
                        "value": mean_grad,
                        "severity": "high"
                        if mean_grad < vanishing_threshold / 10
                        else "medium",
                    }
                )

        if not self.issues:
            print("No gradient issues detected!")
        else:
            print(f"\nFound {len(self.issues)} potential gradient issues:")
            for issue in self.issues:
                print(
                    f"  [{issue['severity'].upper()}] {issue['type']} gradients in {issue['layer']}: {issue['value']:.2e}"
                )

        return self.issues

    def plot_gradient_flow(self, save_path: Optional[str] = None):
        """Plot gradient flow through layers."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return

        layers = []
        avg_grads = []
        max_grads = []

        for name, stats_list in self.gradient_stats.items():
            if "weight" in name and "bn" not in name and "bias" not in name:
                layers.append(name.replace(".weight", ""))
                avg_grads.append(np.mean([s["mean"] for s in stats_list]))
                max_grads.append(np.max([s["max"] for s in stats_list]))

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(layers)), avg_grads, alpha=0.3, color="b", label="Average")
        plt.bar(range(len(layers)), max_grads, alpha=0.3, color="r", label="Max")
        plt.hlines(0, 0, len(layers) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(layers), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(layers))
        plt.xlabel("Layers")
        plt.ylabel("Gradient Magnitude")
        plt.title("Gradient Flow")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


class DeadNeuronDetector:
    """
    Detect dead neurons in the model.

    Example:
        >>> detector = DeadNeuronDetector(model)
        >>> detector.analyze(dataloader)
        >>> detector.report()
    """

    def __init__(self, model: nn.Module, threshold: float = 1e-6):
        self.model = model
        self.threshold = threshold
        self.activations = defaultdict(list)
        self.hooks = []

    def analyze(self, data_loader: torch.utils.data.DataLoader, num_batches: int = 10):
        """Analyze activations to detect dead neurons."""
        self._register_hooks()
        self.model.eval()

        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= num_batches:
                    break

                _ = self.model(data)

        self._remove_hooks()

    def _register_hooks(self):
        """Register hooks to capture activations."""

        def get_hook(name):
            def hook(module, input, output):
                # Track mean activation per neuron
                if isinstance(output, torch.Tensor):
                    self.activations[name].append(
                        output.abs().mean(dim=0).cpu().numpy()
                    )

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(get_hook(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def report(self) -> Dict[str, int]:
        """Report dead neurons per layer."""
        dead_neurons = {}

        for name, activations_list in self.activations.items():
            if not activations_list:
                continue

            # Average across batches
            avg_activation = np.mean(activations_list, axis=0)

            # Count dead neurons
            dead = np.sum(avg_activation < self.threshold)
            total = avg_activation.size

            if dead > 0:
                dead_neurons[name] = dead
                print(
                    f"{name}: {dead}/{total} dead neurons ({100 * dead / total:.1f}%)"
                )

        if not dead_neurons:
            print("No dead neurons detected!")

        return dead_neurons


class WeightAnalyzer:
    """
    Analyze weight distributions and initialization quality.

    Example:
        >>> analyzer = WeightAnalyzer(model)
        >>> analyzer.analyze()
        >>> analyzer.print_report()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.stats = {}

    def analyze(self):
        """Analyze all weights in the model."""
        self.stats = {}

        for name, param in self.model.named_parameters():
            if "weight" in name:
                w = param.data.cpu().numpy()

                self.stats[name] = {
                    "mean": np.mean(w),
                    "std": np.std(w),
                    "min": np.min(w),
                    "max": np.max(w),
                    "sparsity": np.sum(np.abs(w) < 1e-6) / w.size,
                    "norm": np.linalg.norm(w),
                }

    def print_report(self):
        """Print weight analysis report."""
        print("\n" + "=" * 80)
        print("Weight Analysis Report")
        print("=" * 80)

        for name, stats in self.stats.items():
            print(f"\n{name}:")
            print(f"  Mean: {stats['mean']:+.4f}, Std: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:+.4f}, {stats['max']:+.4f}]")
            print(f"  Sparsity: {stats['sparsity'] * 100:.2f}%")
            print(f"  Norm: {stats['norm']:.4f}")

        # Check for potential issues
        issues = []
        for name, stats in self.stats.items():
            if stats["std"] < 1e-4:
                issues.append(f"{name}: very small std ({stats['std']:.2e})")
            if stats["sparsity"] > 0.5:
                issues.append(f"{name}: high sparsity ({stats['sparsity'] * 100:.1f}%)")

        if issues:
            print("\nPotential Issues:")
            for issue in issues:
                print(f"  - {issue}")


@contextmanager
def measure_time(name: str = "Operation"):
    """Context manager to measure execution time."""
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {(end - start) * 1000:.2f} ms")


class TrainingDebugger:
    """
    Debug training issues like NaN losses, unstable training, etc.

    Example:
        >>> debugger = TrainingDebugger(model)
        >>> debugger.enable_nan_detection()
        >>> # Training loop with automatic debugging
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.nan_detected = False
        self.inf_detected = False
        self.hooks = []

    def enable_nan_detection(self):
        """Enable automatic NaN detection in forward pass."""

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    if torch.isnan(output).any():
                        print(f"NaN detected in {name}")
                        self.nan_detected = True
                    if torch.isinf(output).any():
                        print(f"Inf detected in {name}")
                        self.inf_detected = True

            return hook

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)

    def check_loss(self, loss: torch.Tensor, step: int):
        """Check loss value for issues."""
        if torch.isnan(loss):
            print(f"Step {step}: Loss is NaN!")
            return False
        elif torch.isinf(loss):
            print(f"Step {step}: Loss is Inf!")
            return False
        elif loss.item() > 1e6:
            print(f"Step {step}: Loss is very large ({loss.item():.2e})")

        return True

    def get_diagnosis(self) -> List[str]:
        """Get diagnosis of potential issues."""
        diagnoses = []

        if self.nan_detected:
            diagnoses.append("NaN values detected. Possible causes:")
            diagnoses.append("  - Learning rate too high")
            diagnoses.append("  - Gradient explosion (try gradient clipping)")
            diagnoses.append("  - Numerical instability (check log(0) operations)")

        if self.inf_detected:
            diagnoses.append("Inf values detected. Possible causes:")
            diagnoses.append("  - Division by zero")
            diagnoses.append("  - Exponential overflow")

        return diagnoses


def profile_memory_usage(
    model: nn.Module, input_shape: Tuple[int, ...], device: str = "cuda"
):
    """
    Profile detailed memory usage.

    Args:
        model: Model to profile
        input_shape: Input tensor shape
        device: Device to use

    Returns:
        Dictionary of memory statistics
    """
    if device != "cuda":
        print("Memory profiling only available on CUDA")
        return {}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = model.to(device)
    model.eval()

    # Memory before forward
    mem_before = torch.cuda.memory_allocated()

    # Forward pass
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        _ = model(dummy_input)

    mem_after_forward = torch.cuda.memory_allocated()

    # Memory for gradients (simulate backward)
    model.train()
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

    mem_after_backward = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()

    return {
        "model_params_mb": sum(p.numel() for p in model.parameters()) * 4 / 1024**2,
        "input_mb": np.prod(input_shape) * 4 / 1024**2,
        "forward_only_mb": (mem_after_forward - mem_before) / 1024**2,
        "with_gradients_mb": (mem_after_backward - mem_before) / 1024**2,
        "peak_memory_mb": peak_mem / 1024**2,
    }


__all__ = [
    "ModelProfiler",
    "GradientChecker",
    "DeadNeuronDetector",
    "WeightAnalyzer",
    "TrainingDebugger",
    "measure_time",
    "profile_memory_usage",
]
