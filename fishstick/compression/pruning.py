"""
Advanced Pruning Methods for Model Compression

Includes magnitude pruning, structured pruning, lottery ticket hypothesis,
and gradual magnitude scheduling.
"""

from typing import Optional, List, Dict, Callable, Tuple, Union
import math
import torch
from torch import nn, Tensor
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from collections import OrderedDict
import copy


class MagnitudePruner:
    """Magnitude-based unstructured pruning with support for gradual scheduling.

    Prunes weights with smallest absolute magnitude.

    Args:
        model: Model to prune
        sparsity: Target sparsity ratio (0-1)
        module_types: Types of modules to prune
        prune_biases: Whether to prune bias parameters
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        module_types: Optional[Tuple] = None,
        prune_biases: bool = False,
    ):
        self.model = model
        self.sparsity = sparsity
        self.module_types = module_types or (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)
        self.prune_biases = prune_biases
        self.masks: Dict[str, Tensor] = {}
        self.pruned_modules: List[Tuple[str, nn.Module]] = []

    def compute_threshold(self, weight: Tensor, amount: float) -> float:
        """Compute magnitude threshold for pruning."""
        if amount <= 0:
            return 0.0
        if amount >= 1.0:
            return float("inf")
        return float(torch.quantile(weight.abs().flatten().float(), amount))

    def prune(self) -> float:
        """Apply magnitude pruning to the model."""
        total_params = 0
        pruned_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, self.module_types):
                if hasattr(module, "weight") and module.weight is not None:
                    weight = module.weight.data
                    threshold = self.compute_threshold(weight, self.sparsity)
                    mask = weight.abs() > threshold
                    self.masks[f"{name}.weight"] = mask
                    weight.data *= mask.float()

                    total_params += weight.numel()
                    pruned_params += (~mask).sum().item()

                if (
                    self.prune_biases
                    and hasattr(module, "bias")
                    and module.bias is not None
                ):
                    bias = module.bias.data
                    threshold = self.compute_threshold(bias, self.sparsity)
                    mask = bias.abs() > threshold
                    self.masks[f"{name}.bias"] = mask
                    bias.data *= mask.float()

                    total_params += bias.numel()
                    pruned_params += (~mask).sum().item()

                self.pruned_modules.append((name, module))

        return pruned_params / total_params if total_params > 0 else 0.0

    def apply_masks(self):
        """Apply stored masks to parameters."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name].float()

    def get_sparsity(self) -> Dict[str, float]:
        """Get sparsity statistics for each layer."""
        stats = {}
        for name, param in self.model.named_parameters():
            if "weight" in name or "bias" in name:
                sparsity = (param == 0).sum().item() / param.numel()
                stats[name] = sparsity
        return stats

    def remove(self):
        """Make pruning permanent by removing pruning reparameterization."""
        for name, module in self.pruned_modules:
            if hasattr(module, "weight_mask"):
                prune.remove(module, "weight")
            if hasattr(module, "bias_mask"):
                prune.remove(module, "bias")


class StructuredPruner:
    """Structured pruning that removes entire neurons, filters, or channels.

    Args:
        model: Model to prune
        sparsity: Target sparsity ratio (0-1)
        prune_type: Type of structured pruning ('filter', 'channel', 'column')
        criterion: Criterion for ranking ('l1', 'l2', 'geometric_median')
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.3,
        prune_type: str = "filter",
        criterion: str = "l1",
    ):
        self.model = model
        self.sparsity = sparsity
        self.prune_type = prune_type
        self.criterion = criterion
        self.pruned_indices: Dict[str, Tensor] = {}

    def _compute_importance(self, weight: Tensor, dim: int) -> Tensor:
        """Compute importance scores for structures."""
        if self.criterion == "l1":
            return weight.abs().sum(
                dim=tuple(i for i in range(weight.dim()) if i != dim)
            )
        elif self.criterion == "l2":
            return (
                weight.pow(2)
                .sum(dim=tuple(i for i in range(weight.dim()) if i != dim))
                .sqrt()
            )
        elif self.criterion == "geometric_median":
            mean = weight.mean(
                dim=tuple(i for i in range(weight.dim()) if i != dim), keepdim=True
            )
            return (
                (weight - mean)
                .abs()
                .sum(dim=tuple(i for i in range(weight.dim()) if i != dim))
            )
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def prune_conv2d(self) -> Dict[str, int]:
        """Prune Conv2d filters based on importance."""
        pruned_counts = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                out_channels = weight.shape[0]

                importance = self._compute_importance(weight, dim=0)
                n_prune = int(self.sparsity * out_channels)

                if n_prune > 0 and n_prune < out_channels:
                    _, indices = torch.topk(importance, n_prune, largest=False)
                    weight.data[indices] = 0

                    if module.bias is not None:
                        module.bias.data[indices] = 0

                    self.pruned_indices[name] = indices
                    pruned_counts[name] = n_prune

        return pruned_counts

    def prune_linear(self) -> Dict[str, int]:
        """Prune Linear layer neurons."""
        pruned_counts = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                out_features = weight.shape[0]

                importance = self._compute_importance(weight, dim=0)
                n_prune = int(self.sparsity * out_features)

                if n_prune > 0 and n_prune < out_features:
                    _, indices = torch.topk(importance, n_prune, largest=False)
                    weight.data[indices] = 0

                    if module.bias is not None:
                        module.bias.data[indices] = 0

                    self.pruned_indices[name] = indices
                    pruned_counts[name] = n_prune

        return pruned_counts

    def prune(self) -> Dict[str, int]:
        """Apply structured pruning to the model."""
        counts = {}
        counts.update(self.prune_conv2d())
        counts.update(self.prune_linear())
        return counts


class LotteryTicketPruner:
    """Lottery Ticket Hypothesis pruning with iterative magnitude pruning.

    Finds sparse subnetworks that can train from scratch to match original performance.

    Reference: Frankle & Carbin, "The Lottery Ticket Hypothesis", ICLR 2019

    Args:
        model: Model to prune
        sparsity: Final target sparsity
        prune_epochs: Number of pruning iterations
        rewind_epoch: Epoch to rewind weights to (0 for initialization)
        prune_type: 'global' or 'local' pruning
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.9,
        prune_epochs: int = 10,
        rewind_epoch: int = 0,
        prune_type: str = "global",
    ):
        self.model = model
        self.sparsity = sparsity
        self.prune_epochs = prune_epochs
        self.rewind_epoch = rewind_epoch
        self.prune_type = prune_type
        self.masks: Dict[str, Tensor] = {}
        self.initial_weights: Optional[Dict[str, Tensor]] = None
        self.rewind_weights: Optional[Dict[str, Tensor]] = None

    def save_initial_weights(self):
        """Save initial weights for rewinding."""
        self.initial_weights = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

    def save_rewind_weights(self):
        """Save weights at rewind point."""
        self.rewind_weights = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

    def get_current_sparsity(self, epoch: int) -> float:
        """Compute sparsity schedule using cubic scheduling."""
        if self.prune_type == "global":
            return self.sparsity
        final_sparsity = self.sparsity
        initial_sparsity = 0.0
        start_epoch = 0
        end_epoch = self.prune_epochs

        if epoch < start_epoch:
            return initial_sparsity
        if epoch >= end_epoch:
            return final_sparsity

        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        return (
            final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3
        )

    def prune(self, epoch: int = 0) -> Dict[str, Tensor]:
        """Perform one iteration of magnitude pruning."""
        current_sparsity = self.get_current_sparsity(epoch)

        if self.prune_type == "global":
            self._global_prune(current_sparsity)
        else:
            self._local_prune(current_sparsity)

        return self.masks

    def _global_prune(self, sparsity: float):
        """Global magnitude pruning across all layers."""
        all_weights = []
        param_names = []

        for name, param in self.model.named_parameters():
            if "weight" in name and param.requires_grad:
                all_weights.append(param.abs().flatten())
                param_names.append(name)

        if not all_weights:
            return

        all_weights = torch.cat(all_weights)
        threshold = torch.quantile(all_weights.float(), sparsity)

        for name, param in self.model.named_parameters():
            if "weight" in name and param.requires_grad:
                mask = param.abs() > threshold
                self.masks[name] = mask
                param.data *= mask.float()

    def _local_prune(self, sparsity: float):
        """Local magnitude pruning per layer."""
        for name, param in self.model.named_parameters():
            if "weight" in name and param.requires_grad:
                threshold = torch.quantile(param.abs().flatten().float(), sparsity)
                mask = param.abs() > threshold
                self.masks[name] = mask
                param.data *= mask.float()

    def rewind(self, use_initial: bool = False):
        """Rewind model weights to saved state with current mask."""
        weights = self.initial_weights if use_initial else self.rewind_weights

        if weights is None:
            raise ValueError(
                "No saved weights to rewind to. Call save_initial_weights() or save_rewind_weights() first."
            )

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])
                    if name in self.masks:
                        param *= self.masks[name].float()

    def apply_masks(self):
        """Apply current masks to model parameters."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name].float()

    def get_winning_ticket(self) -> nn.Module:
        """Return a copy of the model with winning ticket mask applied."""
        ticket = copy.deepcopy(self.model)
        for name, param in ticket.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name].float()
        return ticket


class GradualMagnitudeScheduler:
    """Gradual Magnitude Scheduling (GMP) for pruning during training.

    Gradually increases sparsity from initial to final target over training.

    Args:
        model: Model to prune
        initial_sparsity: Starting sparsity
        final_sparsity: Target sparsity
        start_epoch: Epoch to start pruning
        end_epoch: Epoch to end pruning
        prune_frequency: How often to prune (in epochs)
        schedule_type: 'cubic', 'linear', or 'exponential'
    """

    def __init__(
        self,
        model: nn.Module,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.9,
        start_epoch: int = 0,
        end_epoch: int = 100,
        prune_frequency: int = 1,
        schedule_type: str = "cubic",
    ):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.prune_frequency = prune_frequency
        self.schedule_type = schedule_type
        self.masks: Dict[str, Tensor] = {}
        self.current_sparsity = initial_sparsity

    def get_sparsity(self, epoch: int) -> float:
        """Compute target sparsity for current epoch."""
        if epoch < self.start_epoch:
            return self.initial_sparsity
        if epoch >= self.end_epoch:
            return self.final_sparsity

        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

        if self.schedule_type == "linear":
            return (
                self.initial_sparsity
                + (self.final_sparsity - self.initial_sparsity) * progress
            )
        elif self.schedule_type == "cubic":
            return (
                self.final_sparsity
                + (self.initial_sparsity - self.final_sparsity) * (1 - progress) ** 3
            )
        elif self.schedule_type == "exponential":
            return self.final_sparsity + (
                self.initial_sparsity - self.final_sparsity
            ) * math.exp(-3 * progress)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def step(self, epoch: int) -> float:
        """Perform pruning step if scheduled."""
        if epoch < self.start_epoch:
            return self.current_sparsity

        if epoch % self.prune_frequency != 0 and epoch != self.end_epoch:
            return self.current_sparsity

        self.current_sparsity = self.get_sparsity(epoch)
        self._update_masks()

        return self.current_sparsity

    def _update_masks(self):
        """Update pruning masks based on current sparsity."""
        for name, param in self.model.named_parameters():
            if "weight" in name and param.requires_grad:
                threshold = torch.quantile(
                    param.abs().flatten().float(), self.current_sparsity
                )
                mask = param.abs() > threshold
                self.masks[name] = mask
                param.data *= mask.float()

    def apply_masks(self):
        """Apply stored masks to parameters."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name].float()


class SensitivityPruner:
    """Sensitivity-based pruning using gradient information.

    Prunes weights based on their contribution to the loss.

    Args:
        model: Model to prune
        sparsity: Target sparsity
        sensitivity_type: 'gradient', 'taylor', or 'hessian_diag'
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        sensitivity_type: str = "gradient",
    ):
        self.model = model
        self.sparsity = sparsity
        self.sensitivity_type = sensitivity_type
        self.masks: Dict[str, Tensor] = {}
        self.sensitivity_scores: Dict[str, Tensor] = {}

    def compute_sensitivity(self, data: Tensor, target: Tensor, loss_fn: Callable):
        """Compute sensitivity scores for each weight."""
        self.model.zero_grad()
        output = self.model(data)
        loss = loss_fn(output, target)
        loss.backward()

        for name, param in self.model.named_parameters():
            if "weight" in name and param.grad is not None:
                if self.sensitivity_type == "gradient":
                    sensitivity = param.grad.abs()
                elif self.sensitivity_type == "taylor":
                    sensitivity = (param.data * param.grad).abs()
                elif self.sensitivity_type == "hessian_diag":
                    sensitivity = self._estimate_hessian_diag(name, param)
                else:
                    raise ValueError(
                        f"Unknown sensitivity type: {self.sensitivity_type}"
                    )

                self.sensitivity_scores[name] = sensitivity

    def _estimate_hessian_diag(self, name: str, param: Tensor) -> Tensor:
        """Estimate diagonal of Hessian using Hutchinson's method."""
        hessian_diag = torch.zeros_like(param)
        n_samples = 10

        for _ in range(n_samples):
            v = torch.randn_like(param)
            grad_v = (param.grad * v).sum()
            grad_v.backward(retain_graph=True)
            hessian_diag += param.grad * v
            self.model.zero_grad()

        return hessian_diag / n_samples

    def prune(self):
        """Prune based on computed sensitivity scores."""
        if not self.sensitivity_scores:
            raise ValueError("Call compute_sensitivity() before prune()")

        for name, param in self.model.named_parameters():
            if name in self.sensitivity_scores:
                threshold = torch.quantile(
                    self.sensitivity_scores[name].flatten().float(), self.sparsity
                )
                mask = self.sensitivity_scores[name] > threshold
                self.masks[name] = mask
                param.data *= mask.float()


class DependencyAwarePruner:
    """Dependency-aware structured pruning for CNNs.

    Ensures pruning respects layer dependencies (e.g., consecutive conv layers).

    Args:
        model: Model to prune
        sparsity: Target sparsity per layer
        dependency_groups: Groups of layers with shared dependencies
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.3,
        dependency_groups: Optional[List[List[str]]] = None,
    ):
        self.model = model
        self.sparsity = sparsity
        self.dependency_groups = dependency_groups or []
        self.masks: Dict[str, Tensor] = {}

    def auto_detect_dependencies(self):
        """Automatically detect layer dependencies in sequential models."""
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)

        self.dependency_groups = []
        i = 0
        while i < len(conv_layers):
            group = [conv_layers[i]]
            j = i + 1
            while j < len(conv_layers):
                curr_name = conv_layers[j]
                prev_name = conv_layers[j - 1]

                curr_module = dict(self.model.named_modules())[curr_name]
                prev_module = dict(self.model.named_modules())[prev_name]

                if curr_module.in_channels == prev_module.out_channels:
                    group.append(curr_name)
                    j += 1
                else:
                    break

            if len(group) > 1:
                self.dependency_groups.append(group)
            i = j

    def prune_group(self, group: List[str]):
        """Prune a dependency group consistently."""
        first_module = dict(self.model.named_modules())[group[0]]

        importance_scores = []
        for name in group:
            module = dict(self.model.named_modules())[name]
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                importance = weight.abs().sum(dim=(1, 2, 3))
                importance_scores.append(importance)

        if importance_scores:
            avg_importance = torch.stack(importance_scores).mean(dim=0)
            n_prune = int(self.sparsity * len(avg_importance))

            if n_prune > 0 and n_prune < len(avg_importance):
                _, indices_to_prune = torch.topk(avg_importance, n_prune, largest=False)

                for name in group:
                    module = dict(self.model.named_modules())[name]
                    if isinstance(module, nn.Conv2d):
                        module.weight.data[indices_to_prune] = 0
                        if module.bias is not None:
                            module.bias.data[indices_to_prune] = 0

    def prune(self):
        """Apply dependency-aware pruning."""
        if not self.dependency_groups:
            self.auto_detect_dependencies()

        for group in self.dependency_groups:
            self.prune_group(group)


__all__ = [
    "MagnitudePruner",
    "StructuredPruner",
    "LotteryTicketPruner",
    "GradualMagnitudeScheduler",
    "SensitivityPruner",
    "DependencyAwarePruner",
]
