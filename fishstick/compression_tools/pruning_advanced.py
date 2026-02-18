"""
Advanced Pruning Methods for Model Compression

Magnitude pruning, lottery ticket hypothesis, movement pruning,
structured pruning, and adaptive pruning schedules.

References:
- https://arxiv.org/abs/1803.03635 (Lottery Ticket Hypothesis)
- https://arxiv.org.org/abs/1906.02692 (Movement Pruning)
- https://arxiv.org/abs/1910.12108 (Rethinking Pruning)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Callable, Tuple, Union, Any
from enum import Enum
import copy
import math
import warnings

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Optimizer


class PruningSchedule(Enum):
    """Pruning schedule types."""

    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CUBIC = "cubic"
    COSINE = "cosine"
    POLY = "poly"


class PruningType(Enum):
    """Types of pruning."""

    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"
    CHANNEL = "channel"
    FILTER = "filter"
    ATOMIC = "atomic"


class MagnitudePrunerAdvanced:
    """Advanced magnitude-based pruner with gradual pruning.

    Prunes weights based on their absolute values with support for
    various pruning schedules and gradual sparsity increase.

    Args:
        model: Model to prune
        initial_sparsity: Initial sparsity level (0.0 to 1.0)
        final_sparsity: Final sparsity level to achieve
        schedule: Pruning schedule type
        epoch_steps: Number of steps per epoch
        total_epochs: Total number of pruning epochs
        warmup_epochs: Number of epochs before pruning starts

    Example:
        >>> pruner = MagnitudePrunerAdvanced(model, initial_sparsity=0.0, final_sparsity=0.8)
        >>> for epoch in range(epochs):
        ...     pruner.step(epoch, sparsity_target)
    """

    def __init__(
        self,
        model: nn.Module,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        schedule: Union[str, PruningSchedule] = PruningSchedule.COSINE,
        epoch_steps: int = 100,
        total_epochs: int = 50,
        warmup_epochs: int = 5,
    ):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.schedule = (
            PruningSchedule(schedule) if isinstance(schedule, str) else schedule
        )
        self.epoch_steps = epoch_steps
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.current_sparsity = initial_sparsity

        self.masks: Dict[str, Tensor] = {}
        self.original_weights: Dict[str, Tensor] = {}
        self._store_original_weights()

    def _store_original_weights(self):
        """Store original weights for potential restoration."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.original_weights[name] = module.weight.data.clone()
                self.masks[name] = torch.ones_like(module.weight.data)

    def _get_sparsity_for_step(self, current_step: int) -> float:
        """Calculate target sparsity for current step."""
        total_steps = self.epoch_steps * (self.total_epochs - self.warmup_epochs)
        current_step = min(current_step, total_steps)

        if current_step <= 0:
            return self.initial_sparsity

        progress = current_step / max(total_steps, 1)

        if self.schedule == PruningSchedule.CONSTANT:
            return (
                self.initial_sparsity
                + (self.final_sparsity - self.initial_sparsity) * progress
            )
        elif self.schedule == PruningSchedule.LINEAR:
            return (
                self.initial_sparsity
                + (self.final_sparsity - self.initial_sparsity) * progress
            )
        elif self.schedule == PruningSchedule.EXPONENTIAL:
            return self.initial_sparsity + (
                self.final_sparsity - self.initial_sparsity
            ) * (progress**2)
        elif self.schedule == PruningSchedule.CUBIC:
            return self.initial_sparsity + (
                self.final_sparsity - self.initial_sparsity
            ) * (progress**3)
        elif self.schedule == PruningSchedule.COSINE:
            return (
                self.initial_sparsity
                + (self.final_sparsity - self.initial_sparsity)
                * (1 - math.cos(math.pi * progress))
                / 2
            )
        elif self.schedule == PruningSchedule.POLY:
            return self.initial_sparsity + (
                self.final_sparsity - self.initial_sparsity
            ) * (progress**1.5)
        else:
            return self.initial_sparsity

    def step(self, current_step: int):
        """Update pruning masks for current step."""
        target_sparsity = self._get_sparsity_for_step(current_step)
        self.current_sparsity = target_sparsity
        self._compute_masks(target_sparsity)

    def _compute_masks(self, sparsity: float):
        """Compute binary masks based on weight magnitudes."""
        for name, module in self.model.named_modules():
            if name not in self.masks:
                continue

            weight = module.weight.data
            threshold = torch.quantile(weight.abs().flatten(), sparsity)
            mask = (weight.abs() > threshold).float()
            self.masks[name] = mask
            module.weight.data = weight * mask

    def apply_masks(self):
        """Apply stored masks to model weights."""
        for name, module in self.model.named_modules():
            if name in self.masks:
                module.weight.data = self.original_weights[name] * self.masks[name]

    def restore_weights(self):
        """Restore original weights without masks."""
        for name, module in self.model.named_modules():
            if name in self.original_weights:
                module.weight.data = self.original_weights[name].clone()

    def get_sparsity(self) -> float:
        """Get current model sparsity."""
        total_params = 0
        pruned_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                total_params += weight.numel()
                pruned_params += (weight.abs() < 1e-8).sum().item()

        return pruned_params / max(total_params, 1)


class LotteryTicketFinder:
    """Lottery Ticket Hypothesis subnet finder.

    Finds trainable subnetworks that can reach similar accuracy
    to the full model when trained from scratch.

    Implements iterative magnitude pruning to find winning tickets.

    Args:
        model: Model to find subnet in
        pruning_rate: Fraction of weights to prune per iteration
        num_iterations: Number of pruning iterations
        reinit: Whether to reinitialize weights after pruning

    Example:
        >>> finder = LotteryTicketFinder(model, pruning_rate=0.2, num_iterations=6)
        >>> ticket = finder.find_ticket(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_rate: float = 0.2,
        num_iterations: int = 6,
        reinit: bool = True,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.pruning_rate = pruning_rate
        self.num_iterations = num_iterations
        self.reinit = reinit
        self.seed = seed

        self.best_ticket: Optional[Dict[str, Tensor]] = None
        self.best_accuracy: float = 0.0

    def find_ticket(
        self,
        train_loader: List[Tuple[Tensor, Tensor]],
        val_loader: List[Tuple[Tensor, Tensor]],
        num_epochs: int = 10,
        optimizer: Optional[Optimizer] = None,
        criterion: Callable = F.cross_entropy,
        device: str = "cpu",
    ) -> nn.Module:
        """Find winning lottery ticket.

        Args:
            train_loader: Training data
            val_loader: Validation data
            num_epochs: Epochs to train per iteration
            optimizer: Optimizer factory
            criterion: Loss function
            device: Device to train on

        Returns:
            Model with found winning ticket
        """
        if optimizer is None:
            optimizer = torch.optim.SGD

        original_weights = self._save_weights()

        for iteration in range(self.num_iterations):
            model = self._create_model_from_weights(original_weights)

            if self.reinit:
                model = self._rewind_weights(model, iteration == 0)

            optimizer_instance = optimizer(model.parameters(), lr=0.01)
            self._train_model(
                model, train_loader, optimizer_instance, criterion, num_epochs, device
            )

            accuracy = self._evaluate(model, val_loader, criterion, device)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_ticket = self._save_weights()

            original_weights = self._prune_weights(original_weights)
            print(
                f"Iteration {iteration + 1}/{self.num_iterations}: "
                f"Sparsity: {self._calculate_sparsity(original_weights):.2%}, "
                f"Accuracy: {accuracy:.2f}%"
            )

        return self._create_model_from_weights(self.best_ticket)

    def _save_weights(self) -> Dict[str, Tensor]:
        """Save current model weights."""
        return {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }

    def _create_model_from_weights(self, weights: Dict[str, Tensor]) -> nn.Module:
        """Create model from saved weights."""
        model = copy.deepcopy(self.model)
        for name, param in model.named_parameters():
            if name in weights:
                param.data = weights[name].clone()
        return model

    def _rewind_weights(self, model: nn.Module, first_iteration: bool) -> nn.Module:
        """Rewind weights to initial state."""
        if first_iteration:
            return model

        original = copy.deepcopy(self.model)
        for (name, param), (orig_name, orig_param) in zip(
            model.named_parameters(), original.named_parameters()
        ):
            if name == orig_name:
                param.data = orig_param.data.clone()

        return model

    def _prune_weights(self, weights: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Prune weights by magnitude."""
        pruned = {}
        for name, weight in weights.items():
            weight_flat = weight.abs().flatten()
            threshold = torch.quantile(weight_flat, self.pruning_rate)
            mask = weight.abs() > threshold
            pruned[name] = weight * mask.float()
        return pruned

    def _train_model(
        self,
        model: nn.Module,
        train_loader: List[Tuple[Tensor, Tensor]],
        optimizer: Optimizer,
        criterion: Callable,
        num_epochs: int,
        device: str,
    ):
        """Train model for one iteration."""
        model.train()
        for epoch in range(num_epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    def _evaluate(
        self,
        model: nn.Module,
        val_loader: List[Tuple[Tensor, Tensor]],
        criterion: Callable,
        device: str,
    ) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        return 100.0 * correct / max(total, 1)

    def _calculate_sparsity(self, weights: Dict[str, Tensor]) -> float:
        """Calculate sparsity of weights."""
        total = sum(w.numel() for w in weights.values())
        zero = sum((w.abs() < 1e-8).sum().item() for w in weights.values())
        return zero / max(total, 1)


class MovementPruner:
    """Movement pruning implementation.

    Prunes weights based on their movement (accumulated importance
    scores) during training, as described in the Movement Pruning paper.

    Args:
        model: Model to prune
        sparsity: Target sparsity level
        temperature: Temperature for soft thresholding

    Example:
        >>> pruner = MovementPruner(model, sparsity=0.5)
        >>> for batch in train_loader:
        ...     pruner.update_scores(model)
        ...     pruner.prune()
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        temperature: float = 2.0,
    ):
        self.model = model
        self.sparsity = sparsity
        self.temperature = temperature
        self.scores: Dict[str, Tensor] = {}
        self.masks: Dict[str, Tensor] = {}
        self.optimizer: Optional[Optimizer] = None

        self._initialize_scores()
        self._initialize_masks()

    def _initialize_scores(self):
        """Initialize importance scores for all prunable layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.scores[name] = torch.zeros_like(module.weight.data)

    def _initialize_masks(self):
        """Initialize pruning masks."""
        for name in self.scores:
            self.masks[name] = torch.ones_like(self.scores[name])

    def update_scores(self):
        """Update importance scores based on weight movements."""
        for name, module in self.model.named_modules():
            if name not in self.scores:
                continue

            weight = module.weight.data
            grad = module.weight.grad

            if grad is not None:
                movement = weight * grad
                self.scores[name] = self.scores[name] + movement.abs()

    def prune(self):
        """Apply pruning based on current scores."""
        all_scores = torch.cat([s.flatten() for s in self.scores.values()])
        threshold = torch.quantile(all_scores, self.sparsity)

        for name in self.scores:
            mask = self.scores[name] > threshold
            self.masks[name] = mask.float()

    def apply_masks(self):
        """Apply masks to model weights."""
        for name, module in self.model.named_modules():
            if name in self.masks:
                module.weight.data *= self.masks[name]

    def get_sparsity(self) -> float:
        """Calculate current sparsity."""
        total = sum(m.numel() for m in self.masks.values())
        pruned = sum((m == 0).sum().item() for m in self.masks.values())
        return pruned / max(total, 1)

    def set_optimizer(self, optimizer: Optimizer):
        """Set optimizer for gradient tracking."""
        self.optimizer = optimizer

    def step(self):
        """Perform one pruning step."""
        if self.optimizer is not None:
            self.update_scores()
        self.prune()
        self.apply_masks()


class StructuredPrunerAdvanced:
    """Advanced structured pruning for channels and filters.

    Prunes entire channels or filters while maintaining valid
    model architecture for efficient inference.

    Args:
        model: Model to prune
        pruning_type: Type of structured pruning (channel or filter)
        sparsity: Target sparsity level

    Example:
        >>> pruner = StructuredPrunerAdvanced(model, pruning_type='channel', sparsity=0.5)
        >>> pruner.compute_importance()
        >>> pruner.prune()
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_type: Union[str, PruningType] = PruningType.CHANNEL,
        sparsity: float = 0.3,
    ):
        self.model = model
        self.pruning_type = (
            PruningType(pruning_type) if isinstance(pruning_type, str) else pruning_type
        )
        self.target_sparsity = sparsity
        self.channel_masks: Dict[str, Tensor] = {}
        self.importance_scores: Dict[str, Tensor] = {}

        self._analyze_architecture()

    def _analyze_architecture(self):
        """Analyze model architecture to identify prunable structures."""
        self.layer_info = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.layer_info.append(
                    {
                        "name": name,
                        "type": "conv2d",
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "module": module,
                    }
                )
            elif isinstance(module, nn.Linear):
                self.layer_info.append(
                    {
                        "name": name,
                        "type": "linear",
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "module": module,
                    }
                )

    def compute_importance(
        self,
        dataloader: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_bn: bool = True,
    ):
        """Compute importance scores for channels/filters.

        Args:
            dataloader: Optional data for activation-based importance
            use_bn: Use BatchNorm scaling for importance
        """
        if use_bn:
            self._compute_bn_importance()
        else:
            self._compute_weight_importance()

    def _compute_weight_importance(self):
        """Compute importance based on weight statistics."""
        for layer in self.layer_info:
            module = layer["module"]
            weight = module.weight.data

            if layer["type"] == "conv2d":
                if self.pruning_type == PruningType.CHANNEL:
                    importance = weight.abs().mean(dim=(1, 2, 3))
                else:
                    importance = weight.abs().mean(dim=(0, 2, 3))
            else:
                importance = weight.abs().mean(dim=1)

            self.importance_scores[layer["name"]] = importance

    def _compute_bn_importance(self):
        """Compute importance using BatchNorm gamma scaling."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.importance_scores[name] = module.weight.data.abs()

    def prune(self):
        """Apply structured pruning based on importance scores."""
        for layer in self.layer_info:
            name = layer["name"]

            if name not in self.importance_scores:
                continue

            importance = self.importance_scores[name]
            threshold = torch.quantile(importance, self.target_sparsity)
            mask = importance > threshold

            if layer["type"] == "conv2d":
                if self.pruning_type == PruningType.CHANNEL:
                    self._prune_conv_input_channels(layer, mask)
                else:
                    self._prune_conv_output_channels(layer, mask)
            else:
                self._prune_linear(layer, mask)

    def _prune_conv_input_channels(self, layer: Dict[str, Any], mask: Tensor):
        """Prune input channels of conv layer."""
        module = layer["module"]
        module.weight.data = module.weight.data[:, mask, :, :].clone()

        if module.bias is not None:
            module.bias.data = module.bias.data[mask].clone()

    def _prune_conv_output_channels(self, layer: Dict[str, Any], mask: Tensor):
        """Prune output channels of conv layer."""
        module = layer["module"]
        module.weight.data = module.weight.data[mask, :, :, :].clone()

        if module.bias is not None:
            module.bias.data = module.bias.data[mask].clone()

    def _prune_linear(self, layer: Dict[str, Any], mask: Tensor):
        """Prune linear layer."""
        module = layer["module"]
        module.weight.data = module.weight.data[:, mask].clone()
        module.in_features = mask.sum().item()

    def get_compression_ratio(self) -> float:
        """Calculate expected compression ratio."""
        original_params = sum(p.numel() for p in self.model.parameters())

        pruned_params = 0
        for layer in self.layer_info:
            module = layer["module"]
            if layer["type"] == "conv2d":
                if self.pruning_type == PruningType.CHANNEL:
                    current_channels = module.weight.shape[1]
                else:
                    current_channels = module.weight.shape[0]
                pruned_params += (
                    current_channels * module.weight.shape[2] * module.weight.shape[3]
                )
            else:
                pruned_params += module.weight.numel()

        return original_params / max(pruned_params, 1)


class PruningScheduler:
    """Adaptive pruning scheduler with multiple schedule types.

    Manages pruning rate over training with various scheduling strategies
    including polynomial, exponential, and cosine decay.

    Args:
        initial_sparsity: Starting sparsity
        final_sparsity: Target final sparsity
        schedule: Schedule type
        total_epochs: Total training epochs

    Example:
        >>> scheduler = PruningScheduler(0.0, 0.8, 'cosine', total_epochs=100)
        >>> for epoch in range(epochs):
        ...     sparsity = scheduler(epoch)
    """

    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        schedule: Union[str, PruningSchedule] = PruningSchedule.COSINE,
        total_epochs: int = 100,
        warmup_epochs: int = 5,
    ):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.schedule = (
            PruningSchedule(schedule) if isinstance(schedule, str) else schedule
        )
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def __call__(self, epoch: int) -> float:
        """Get sparsity for given epoch."""
        if epoch < self.warmup_epochs:
            return self.initial_sparsity

        progress = (epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        progress = min(progress, 1.0)

        return self._compute_sparsity(progress)

    def _compute_sparsity(self, progress: float) -> float:
        """Compute sparsity based on schedule type."""
        delta = self.final_sparsity - self.initial_sparsity

        if self.schedule == PruningSchedule.LINEAR:
            return self.initial_sparsity + delta * progress
        elif self.schedule == PruningSchedule.EXPONENTIAL:
            return self.initial_sparsity + delta * (2 ** (10 * progress - 10))
        elif self.schedule == PruningSchedule.COSINE:
            return (
                self.initial_sparsity + delta * (1 - math.cos(math.pi * progress)) / 2
            )
        elif self.schedule == PruningSchedule.POLY:
            return self.initial_sparsity + delta * (progress**2)
        elif self.schedule == PruningSchedule.CUBIC:
            return self.initial_sparsity + delta * (progress**3)
        else:
            return self.initial_sparsity + delta * progress

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "PruningScheduler":
        """Create scheduler from config dict."""
        return PruningScheduler(
            initial_sparsity=config.get("initial_sparsity", 0.0),
            final_sparsity=config.get("final_sparsity", 0.5),
            schedule=config.get("schedule", "cosine"),
            total_epochs=config.get("total_epochs", 100),
            warmup_epochs=config.get("warmup_epochs", 5),
        )


class SensitivityPruner:
    """Sensitivity-aware pruner that adapts pruning per layer.

    Automatically determines optimal sparsity for each layer
    based on its sensitivity to pruning.

    Args:
        model: Model to prune
        sensitivity_metric: Metric to use for sensitivity

    Example:
        >>> pruner = SensitivityPruner(model)
        >>> pruner.analyze_sensitivity(val_loader)
        >>> pruner.prune(target_sparsity=0.5)
    """

    def __init__(
        self,
        model: nn.Module,
        sensitivity_metric: str = "loss_increase",
    ):
        self.model = model
        self.sensitivity_metric = sensitivity_metric
        self.layer_sensitivities: Dict[str, float] = {}

    def analyze_sensitivity(
        self,
        dataloader: List[Tuple[Tensor, Tensor]],
        sparsity_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    ) -> Dict[str, Dict[float, float]]:
        """Analyze pruning sensitivity for each layer.

        Args:
            dataloader: Data for sensitivity evaluation
            sparsity_levels: Sparsity levels to test

        Returns:
            Dict of layer names to sensitivity values at each sparsity
        """
        baseline_loss = self._evaluate_loss(dataloader)

        sensitivities = {}

        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            layer_sensitivities = {}
            original_weight = module.weight.data.clone()

            for sparsity in sparsity_levels:
                threshold = torch.quantile(original_weight.abs().flatten(), sparsity)
                mask = original_weight.abs() > threshold
                module.weight.data = original_weight * mask.float()

                pruned_loss = self._evaluate_loss(dataloader)
                layer_sensitivities[sparsity] = abs(pruned_loss - baseline_loss)

                module.weight.data = original_weight.clone()

            sensitivities[name] = layer_sensitivities

        self.layer_sensitivities = {
            name: min(scores.values()) for name, scores in sensitivities.items()
        }

        return sensitivities

    def _evaluate_loss(self, dataloader: List[Tuple[Tensor, Tensor]]) -> float:
        """Evaluate model loss on dataloader."""
        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for data, target in dataloader[:10]:
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                total_loss += loss.item()
                count += 1

        return total_loss / max(count, 1)

    def prune(self, target_sparsity: float) -> nn.Module:
        """Prune model with sensitivity-aware per-layer sparsity.

        Args:
            target_sparsity: Global target sparsity

        Returns:
            Pruned model
        """
        if not self.layer_sensitivities:
            warnings.warn("No sensitivity analysis done. Using uniform sparsity.")
            return self._uniform_prune(target_sparsity)

        total_sensitivity = sum(self.layer_sensitivities.values())

        for name, module in self.model.named_modules():
            if name not in self.layer_sensitivities:
                continue

            sensitivity = self.layer_sensitivities[name]
            normalized_sensitivity = sensitivity / max(total_sensitivity, 1e-8)

            layer_sparsity = target_sparsity * (1 - normalized_sensitivity * 0.5)

            weight = module.weight.data
            threshold = torch.quantile(weight.abs().flatten(), layer_sparsity)
            mask = weight.abs() > threshold
            module.weight.data = weight * mask.float()

        return self.model

    def _uniform_prune(self, sparsity: float) -> nn.Module:
        """Apply uniform pruning to all layers."""
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                threshold = torch.quantile(weight.abs().flatten(), sparsity)
                mask = weight.abs() > threshold
                module.weight.data = weight * mask.float()

        return self.model
