"""
Model Compression Tools

Utilities for model compression including pruning, quantization, and knowledge distillation.
"""

from typing import Optional, List, Dict, Callable, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization
import copy
from collections import OrderedDict


class MagnitudePruner:
    """
    Magnitude-based unstructured pruning.

    Prunes weights with smallest absolute magnitude.

    Args:
        model: Model to prune
        amount: Fraction of weights to prune (0-1)
        module_types: Types of modules to prune (default: Linear, Conv2d)

    Example:
        >>> pruner = MagnitudePruner(model, amount=0.3)
        >>> pruner.prune()  # Prune 30% of weights
        >>> pruner.remove()  # Make pruning permanent
    """

    def __init__(
        self,
        model: nn.Module,
        amount: float = 0.3,
        module_types: Optional[tuple] = None,
    ):
        self.model = model
        self.amount = amount
        self.module_types = module_types or (nn.Linear, nn.Conv2d)
        self.pruned_modules = []

    def prune(self):
        """Apply magnitude pruning to the model."""
        for name, module in self.model.named_modules():
            if isinstance(module, self.module_types):
                prune.l1_unstructured(module, name="weight", amount=self.amount)
                self.pruned_modules.append((name, module))

        total_params = sum(p.numel() for p in self.model.parameters())
        nonzero_params = sum((p != 0).sum().item() for p in self.model.parameters())
        sparsity = 1 - (nonzero_params / total_params)

        print(f"Pruned model: {sparsity * 100:.2f}% sparsity")
        return sparsity

    def remove(self):
        """Make pruning permanent by removing pruning reparameterization."""
        for name, module in self.pruned_modules:
            prune.remove(module, "weight")

    def get_sparsity(self) -> Dict[str, float]:
        """Get sparsity statistics for each layer."""
        stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, self.module_types):
                weight = module.weight
                sparsity = (weight == 0).sum().item() / weight.numel()
                stats[name] = sparsity
        return stats


class StructuredPruner:
    """
    Structured pruning (remove entire channels/filters).

    Args:
        model: Model to prune
        amount: Fraction of channels to prune (0-1)
        norm: Norm to use for ranking (1 or 2)

    Example:
        >>> pruner = StructuredPruner(model, amount=0.2)
        >>> pruner.prune_conv2d()  # Prune Conv2d channels
    """

    def __init__(self, model: nn.Module, amount: float = 0.2, norm: int = 1):
        self.model = model
        self.amount = amount
        self.norm = norm

    def prune_conv2d(self):
        """Prune Conv2d filters based on L-norm."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Compute norm for each filter
                weight = module.weight.data
                norms = (
                    weight.abs().pow(self.norm).sum(dim=[1, 2, 3]).pow(1 / self.norm)
                )

                # Determine how many filters to prune
                n_prune = int(self.amount * len(norms))

                if n_prune > 0:
                    # Get indices of filters with smallest norms
                    prune_indices = torch.argsort(norms)[:n_prune]

                    # Zero out pruned filters
                    with torch.no_grad():
                        weight[prune_indices] = 0


class LotteryTicketPruner:
    """
    Lottery Ticket Hypothesis pruning.

    Iterative pruning with weight rewinding.

    Reference: Frankle & Carbin, "The Lottery Ticket Hypothesis", 2019

    Args:
        model: Model to prune
        initial_amount: Initial pruning amount
        pruning_rate: Rate to increase pruning per iteration
        num_iterations: Number of pruning iterations

    Example:
        >>> ltp = LotteryTicketPruner(model, initial_amount=0.1, num_iterations=5)
        >>> masks = ltp.prune()  # Get winning ticket masks
    """

    def __init__(
        self,
        model: nn.Module,
        initial_amount: float = 0.1,
        pruning_rate: float = 0.2,
        num_iterations: int = 5,
    ):
        self.model = model
        self.initial_amount = initial_amount
        self.pruning_rate = pruning_rate
        self.num_iterations = num_iterations

        # Save initial weights
        self.initial_weights = {
            name: param.clone() for name, param in model.named_parameters()
        }

    def prune(self) -> Dict[str, torch.Tensor]:
        """
        Perform iterative magnitude pruning with rewinding.

        Returns:
            Dictionary of pruning masks
        """
        masks = {}
        current_amount = self.initial_amount

        for iteration in range(self.num_iterations):
            print(
                f"Iteration {iteration + 1}/{self.num_iterations} - Pruning {current_amount * 100:.1f}%"
            )

            # Magnitude pruning
            pruner = MagnitudePruner(self.model, amount=current_amount)
            pruner.prune()

            # Save masks
            for name, module in self.model.named_modules():
                if hasattr(module, "weight_mask"):
                    masks[name] = module.weight_mask.clone()

            # Rewind to initial weights
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.initial_weights:
                        param.copy_(self.initial_weights[name])

            # Apply masks
            for name, module in self.model.named_modules():
                if name in masks and hasattr(module, "weight"):
                    module.weight.data *= masks[name]

            current_amount += self.pruning_rate
            current_amount = min(current_amount, 0.9)  # Max 90% sparsity

        return masks


class DynamicQuantizer:
    """
    Dynamic quantization for inference optimization.

    Quantizes weights to int8 for faster inference on CPU.

    Args:
        model: Model to quantize
        dtype: Quantization dtype (qint8 or quint8)

    Example:
        >>> quantizer = DynamicQuantizer(model)
        >>> quantized_model = quantizer.quantize()
        >>> # 4x smaller, 2-4x faster on CPU
    """

    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.qint8):
        self.model = model
        self.dtype = dtype

    def quantize(self) -> nn.Module:
        """Apply dynamic quantization to the model."""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=self.dtype
        )
        return quantized_model

    def compare_sizes(self, quantized_model: nn.Module):
        """Compare model sizes before and after quantization."""

        def get_size(model):
            torch.save(model.state_dict(), "temp.pt")
            size = torch.os.path.getsize("temp.pt") / 1e6
            torch.os.remove("temp.pt")
            return size

        original_size = get_size(self.model)
        quantized_size = get_size(quantized_model)

        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")


class StaticQuantizer:
    """
    Static quantization with calibration.

    Requires calibration data to determine quantization ranges.

    Args:
        model: Model to quantize
        qconfig: Quantization configuration

    Example:
        >>> quantizer = StaticQuantizer(model)
        >>> quantizer.prepare()
        >>> quantizer.calibrate(calibration_loader)
        >>> quantized_model = quantizer.convert()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.prepared_model = None

    def prepare(self):
        """Prepare model for static quantization."""
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(self.model, inplace=True)
        self.prepared_model = self.model

    def calibrate(self, data_loader, num_batches: int = 100):
        """Calibrate quantization ranges using sample data."""
        if self.prepared_model is None:
            raise ValueError("Call prepare() before calibrate()")

        self.prepared_model.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= num_batches:
                    break
                self.prepared_model(data)

    def convert(self) -> nn.Module:
        """Convert to quantized model."""
        if self.prepared_model is None:
            raise ValueError("Call prepare() and calibrate() before convert()")

        quantized_model = torch.quantization.convert(self.prepared_model, inplace=True)
        return quantized_model


class KnowledgeDistiller:
    """
    Knowledge distillation for model compression.

    Transfer knowledge from a large teacher model to a smaller student model.

    Reference: Hinton et al., "Distilling the Knowledge in a Neural Network", 2015

    Args:
        teacher: Teacher model (large, trained)
        student: Student model (small, untrained)
        temperature: Temperature for softening distributions
        alpha: Weight for distillation loss vs hard target loss

    Example:
        >>> distiller = KnowledgeDistiller(teacher, student, temperature=4.0)
        >>> student = distiller.train(student_loader, teacher_loader, epochs=10)
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
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        hard_loss_fn: nn.Module,
    ) -> torch.Tensor:
        """
        Compute combined distillation and hard target loss.

        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions (soft targets)
            targets: Hard targets
            hard_loss_fn: Loss function for hard targets

        Returns:
            Combined loss
        """
        # Soft targets with temperature
        soft_loss = nn.KLDivLoss(reduction="batchmean")(
            torch.log_softmax(student_logits / self.temperature, dim=1),
            torch.softmax(teacher_logits / self.temperature, dim=1),
        ) * (self.temperature**2)

        # Hard targets
        hard_loss = hard_loss_fn(student_logits, targets)

        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        device: str = "cuda",
        validate_fn: Optional[Callable] = None,
    ) -> nn.Module:
        """
        Train student model using knowledge distillation.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer for student
            epochs: Number of epochs
            device: Device to use
            validate_fn: Optional validation function

        Returns:
            Trained student model
        """
        self.student.to(device)
        self.teacher.to(device)

        hard_loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.student.train()
            total_loss = 0

            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)

                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits = self.teacher(data)

                # Get student predictions
                student_logits = self.student(data)

                # Compute distillation loss
                loss = self.distillation_loss(
                    student_logits, teacher_logits, targets, hard_loss_fn
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

            if validate_fn is not None:
                acc = validate_fn(self.student)
                print(f"  Validation Accuracy: {acc:.4f}")

        return self.student


class WeightClustering:
    """
    Weight clustering for model compression.

    Groups similar weights together to reduce unique values.

    Args:
        model: Model to cluster
        num_clusters: Number of clusters per layer
        module_types: Types of modules to cluster

    Example:
        >>> clusterer = WeightClustering(model, num_clusters=256)
        >>> clusterer.cluster()  # Reduces precision of weights
    """

    def __init__(
        self,
        model: nn.Module,
        num_clusters: int = 256,
        module_types: Optional[tuple] = None,
    ):
        self.model = model
        self.num_clusters = num_clusters
        self.module_types = module_types or (nn.Linear, nn.Conv2d)

    def cluster(self):
        """Apply weight clustering to the model."""
        for name, module in self.model.named_modules():
            if isinstance(module, self.module_types):
                self._cluster_module(module)

    def _cluster_module(self, module: nn.Module):
        """Cluster weights in a single module."""
        weight = module.weight.data
        original_shape = weight.shape

        # Flatten weight tensor
        weight_flat = weight.view(-1)

        # Use k-means clustering
        from sklearn.cluster import KMeans

        # Sample if too large
        if len(weight_flat) > 100000:
            indices = torch.randperm(len(weight_flat))[:100000]
            samples = weight_flat[indices].cpu().numpy().reshape(-1, 1)
        else:
            samples = weight_flat.cpu().numpy().reshape(-1, 1)

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init=10)
        kmeans.fit(samples)

        # Assign all weights to nearest centroid
        all_weights = weight_flat.cpu().numpy().reshape(-1, 1)
        clusters = kmeans.predict(all_weights)
        clustered_weights = kmeans.cluster_centers_[clusters].flatten()

        # Update module weights
        module.weight.data = (
            torch.from_numpy(clustered_weights.reshape(original_shape))
            .to(weight.device)
            .to(weight.dtype)
        )


class ModelCompressor:
    """
    High-level model compression pipeline.

    Combines multiple compression techniques.

    Args:
        model: Model to compress

    Example:
        >>> compressor = ModelCompressor(model)
        >>> compressed = compressor.compress(
        ...     pruning_amount=0.5,
        ...     quantization='dynamic',
        ...     clustering_clusters=256
        ... )
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.original_model = copy.deepcopy(model)

    def compress(
        self,
        pruning_amount: Optional[float] = None,
        quantization: Optional[str] = None,
        clustering_clusters: Optional[int] = None,
    ) -> nn.Module:
        """
        Compress model using specified techniques.

        Args:
            pruning_amount: Fraction of weights to prune (0-1)
            quantization: 'dynamic' or 'static'
            clustering_clusters: Number of weight clusters

        Returns:
            Compressed model
        """
        compressed_model = copy.deepcopy(self.original_model)

        # Pruning
        if pruning_amount is not None:
            print(f"Applying magnitude pruning: {pruning_amount * 100:.1f}%")
            pruner = MagnitudePruner(compressed_model, amount=pruning_amount)
            pruner.prune()
            pruner.remove()

        # Clustering (before quantization)
        if clustering_clusters is not None:
            print(f"Applying weight clustering: {clustering_clusters} clusters")
            clusterer = WeightClustering(
                compressed_model, num_clusters=clustering_clusters
            )
            clusterer.cluster()

        # Quantization
        if quantization == "dynamic":
            print("Applying dynamic quantization")
            quantizer = DynamicQuantizer(compressed_model)
            compressed_model = quantizer.quantize()
        elif quantization == "static":
            print("Static quantization requires calibration data - skipping")

        # Report compression
        self._report_compression(compressed_model)

        return compressed_model

    def _report_compression(self, compressed_model: nn.Module):
        """Report compression statistics."""
        orig_params = sum(p.numel() for p in self.original_model.parameters())
        comp_params = sum(p.numel() for p in compressed_model.parameters())

        print(f"\nCompression Report:")
        print(f"  Original parameters: {orig_params:,}")
        print(f"  Compressed parameters: {comp_params:,}")
        print(f"  Parameter reduction: {(1 - comp_params / orig_params) * 100:.1f}%")


def measure_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
    num_runs: int = 100,
    warmup: int = 10,
) -> float:
    """
    Measure average inference time.

    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor
        device: Device to use
        num_runs: Number of inference runs
        warmup: Number of warmup runs

    Returns:
        Average inference time in milliseconds
    """
    model.to(device)
    model.eval()

    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                start.record()
                _ = model(dummy_input)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                import time

                start_time = time.time()
                _ = model(dummy_input)
                times.append((time.time() - start_time) * 1000)

    avg_time = sum(times) / len(times)
    return avg_time


__all__ = [
    "MagnitudePruner",
    "StructuredPruner",
    "LotteryTicketPruner",
    "DynamicQuantizer",
    "StaticQuantizer",
    "KnowledgeDistiller",
    "WeightClustering",
    "ModelCompressor",
    "measure_inference_time",
]
