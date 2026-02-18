"""
Model Pruning Module

Implements various pruning techniques:
- Magnitude-based pruning
- Gradient-based pruning
- Movement pruning
- Lottery ticket hypothesis
- Structured pruning
"""

from typing import Optional, Dict, Any, List, Callable
import torch
import torch.nn as nn
from torch import Tensor


class BasePruner:
    """Base class for all pruning methods."""

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        schedule: str = "exponential",
        scope: str = "global",
    ):
        self.model = model
        self.sparsity = sparsity
        self.schedule = schedule
        self.scope = scope
        self.masks: Dict[str, Tensor] = {}
        self.original_weights: Dict[str, Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to track activations."""
        pass

    def _create_mask(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Create a binary mask for the given tensor."""
        return torch.ones_like(tensor, dtype=torch.bool)

    def apply_mask(self):
        """Apply current masks to model weights."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name].float()

    def prune(self, sparsity: Optional[float] = None):
        """Apply pruning to model weights."""
        raise NotImplementedError

    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if name in self.original_weights:
                param.data = self.original_weights[name].clone()

    def get_sparsity(self) -> float:
        """Calculate current model sparsity."""
        total_params = 0
        zero_params = 0
        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        return zero_params / total_params if total_params > 0 else 0.0


class MagnitudePruner(BasePruner):
    """Magnitude-based pruning - removes weights with smallest absolute values."""

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        schedule: str = "exponential",
        scope: str = "global",
        threshold_fn: Optional[Callable] = None,
    ):
        super().__init__(model, sparsity, schedule, scope)
        self.threshold_fn = threshold_fn or self._default_threshold
        self._store_original_weights()

    def _store_original_weights(self):
        """Store original weights before pruning."""
        for name, param in self.model.named_parameters():
            self.original_weights[name] = param.data.clone()

    def _default_threshold(self, tensor: Tensor) -> float:
        """Calculate threshold using global magnitude."""
        flat = torch.abs(tensor).flatten()
        if len(flat) == 0:
            return 0.0
        return torch.kthvalue(flat, int(self.sparsity * len(flat)))[0].item()

    def prune(self, sparsity: Optional[float] = None):
        """Apply magnitude-based pruning."""
        sparsity = sparsity or self.sparsity
        self.masks = {}

        if self.scope == "global":
            all_weights = torch.cat(
                [p.data.abs().flatten() for p in self.model.parameters()]
            )
            threshold = torch.kthvalue(all_weights, int(sparsity * len(all_weights)))[
                0
            ].item()

            for name, param in self.model.named_parameters():
                mask = param.data.abs() > threshold
                self.masks[name] = mask
                param.data *= mask.float()
        else:
            for name, param in self.model.named_parameters():
                threshold = self.threshold_fn(param.data)
                mask = param.data.abs() > threshold
                self.masks[name] = mask
                param.data *= mask.float()


class GradientPruner(BasePruner):
    """Gradient-based pruning - removes weights with smallest gradient magnitudes."""

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        schedule: str = "exponential",
        scope: str = "global",
        gradient_accumulation: int = 1,
    ):
        super().__init__(model, sparsity, schedule, scope)
        self.gradient_accumulation = gradient_accumulation
        self.gradient_sum: Dict[str, Tensor] = {}
        self.step_count = 0
        self._store_original_weights()
        self._register_hooks()

    def _store_original_weights(self):
        for name, param in self.model.named_parameters():
            self.original_weights[name] = param.data.clone()
            self.gradient_sum[name] = torch.zeros_like(param.data)

    def _register_hooks(self):
        """Register backward hooks to track gradients."""

        def grad_hook(grad: Tensor, name: str) -> Tensor:
            if name not in self.gradient_sum:
                self.gradient_sum[name] = torch.zeros_like(grad)
            self.gradient_sum[name] += grad.abs()
            return grad

        for name, param in self.model.named_parameters():
            param.register_hook(lambda g, n=name: grad_hook(g, n))

    def prune(self, sparsity: Optional[float] = None):
        """Apply gradient-based pruning."""
        sparsity = sparsity or self.sparsity
        self.step_count += 1

        if self.scope == "global":
            all_grads = torch.cat([g.flatten() for g in self.gradient_sum.values()])
            threshold = torch.kthvalue(all_grads, int(sparsity * len(all_grads)))[
                0
            ].item()

            for name, param in self.model.named_parameters():
                mask = self.gradient_sum[name] > threshold
                self.masks[name] = mask
                param.data *= mask.float()
                self.gradient_sum[name] = torch.zeros_like(param.data)
        else:
            for name, param in self.model.named_parameters():
                threshold = torch.kthvalue(
                    self.gradient_sum[name].flatten(),
                    int(sparsity * self.gradient_sum[name].numel()),
                )[0].item()
                mask = self.gradient_sum[name] > threshold
                self.masks[name] = mask
                param.data *= mask.float()
                self.gradient_sum[name] = torch.zeros_like(param.data)


class MovementPruner(BasePruner):
    """Movement pruning - removes weights that move towards zero during training."""

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        schedule: str = "exponential",
        scope: str = "global",
        warmup_steps: int = 1000,
    ):
        super().__init__(model, sparsity, schedule, scope)
        self.warmup_steps = warmup_steps
        self.movement_scores: Dict[str, Tensor] = {}
        self.step_count = 0
        self._initialize_movement_scores()

    def _initialize_movement_scores(self):
        """Initialize movement scores for each parameter."""
        for name, param in self.model.named_parameters():
            self.movement_scores[name] = torch.zeros_like(param.data)
            self.original_weights[name] = param.data.clone()

    def update_scores(self):
        """Update movement scores based on weight changes."""
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            return

        for name, param in self.model.named_parameters():
            movement = (param.data - self.original_weights[name]).abs()
            self.movement_scores[name] += movement
            self.original_weights[name] = param.data.clone()

    def prune(self, sparsity: Optional[float] = None):
        """Apply movement-based pruning."""
        if self.step_count < self.warmup_steps:
            return

        sparsity = sparsity or self.sparsity
        self.masks = {}

        if self.scope == "global":
            all_scores = torch.cat([s.flatten() for s in self.movement_scores.values()])
            threshold = torch.kthvalue(all_scores, int(sparsity * len(all_scores)))[
                0
            ].item()

            for name, param in self.model.named_parameters():
                mask = self.movement_scores[name] > threshold
                self.masks[name] = mask
                param.data *= mask.float()
        else:
            for name, param in self.model.named_parameters():
                threshold = torch.kthvalue(
                    self.movement_scores[name].flatten(),
                    int(sparsity * self.movement_scores[name].numel()),
                )[0].item()
                mask = self.movement_scores[name] > threshold
                self.masks[name] = mask
                param.data *= mask.float()


class LotteryTicketPruner(BasePruner):
    """Lottery Ticket Hypothesis - finds winning tickets via iterative pruning."""

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        schedule: str = "exponential",
        scope: str = "global",
        pruning_epochs: int = 10,
        reinit: bool = False,
    ):
        super().__init__(model, sparsity, schedule, scope)
        self.pruning_epochs = pruning_epochs
        self.reinit = reinit
        self.initial_weights: Dict[str, Tensor] = {}
        self.best_weights: Dict[str, Tensor] = {}
        self.best_accuracy = 0.0
        self.current_epoch = 0
        self._store_initial_weights()

    def _store_initial_weights(self):
        """Store initial weights for lottery ticket reinitialization."""
        for name, param in self.model.named_parameters():
            self.initial_weights[name] = param.data.clone()

    def update_masks(self):
        """Update pruning masks based on current weights."""
        self.masks = {}

        for name, param in self.model.named_parameters():
            flat = param.data.abs().flatten()
            threshold = torch.kthvalue(flat, int(self.sparsity * len(flat)))[0].item()
            self.masks[name] = param.data.abs() > threshold

    def apply_reinit(self):
        """Reinitialize pruned weights to initial values (Lottery Ticket)."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                param.data[~mask] = self.initial_weights[name][~mask]

    def prune_step(self):
        """Perform one step of iterative pruning."""
        current_sparsity = (
            self.sparsity * (self.current_epoch + 1) / self.pruning_epochs
        )
        current_sparsity = min(current_sparsity, self.sparsity)

        self.masks = {}
        for name, param in self.model.named_parameters():
            if param.numel() == 0:
                continue
            flat = param.data.abs().flatten()
            threshold = torch.kthvalue(flat, int(current_sparsity * len(flat)))[
                0
            ].item()
            mask = param.data.abs() > threshold
            self.masks[name] = mask
            param.data *= mask.float()

    def step(self, accuracy: float):
        """Track best weights and update epoch."""
        self.current_epoch += 1
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            for name, param in self.model.named_parameters():
                self.best_weights[name] = param.data.clone()

    def get_winning_ticket(self) -> nn.Module:
        """Return the winning ticket model."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name].float()
        return self.model


class StructuredPruner(BasePruner):
    """Structured pruning - removes entire channels/neurons/filters."""

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        schedule: str = "exponential",
        scope: str = "global",
        prunable_types: Optional[List[type]] = None,
    ):
        super().__init__(model, sparsity, schedule, scope)
        self.prunable_types = prunable_types or [nn.Linear, nn.Conv2d]
        self.channel_scores: Dict[str, Tensor] = {}
        self._initialize_channel_scores()

    def _initialize_channel_scores(self):
        """Initialize importance scores for each channel."""
        for name, module in self.model.named_modules():
            if type(module) in self.prunable_types:
                if isinstance(module, nn.Conv2d):
                    self.channel_scores[name] = torch.zeros(module.out_channels)
                elif isinstance(module, nn.Linear):
                    self.channel_scores[name] = torch.zeros(module.out_features)

    def _calculate_channel_importance(self, module: nn.Module) -> Tensor:
        """Calculate importance scores for each channel."""
        if isinstance(module, nn.Conv2d):
            weight = module.weight
            return weight.abs().sum(dim=(1, 2, 3))
        elif isinstance(module, nn.Linear):
            weight = module.weight
            return weight.abs().sum(dim=1)
        return torch.zeros(1)

    def update_scores(self):
        """Update channel importance scores."""
        for name, module in self.model.named_modules():
            if name in self.channel_scores:
                scores = self._calculate_channel_importance(module)
                self.channel_scores[name] += scores

    def prune(self, sparsity: Optional[float] = None):
        """Apply structured pruning to channels."""
        sparsity = sparsity or self.sparsity

        for name, module in self.model.named_modules():
            if name not in self.channel_scores:
                continue

            scores = self.channel_scores[name]
            if len(scores) == 0:
                continue

            threshold = torch.kthvalue(scores, int(sparsity * len(scores)))[0].item()
            mask = scores > threshold

            if isinstance(module, nn.Conv2d):
                module.weight.data = module.weight.data[mask]
                if module.bias is not None:
                    module.bias.data = module.bias.data[mask]
                module.out_channels = mask.sum().item()
            elif isinstance(module, nn.Linear):
                module.weight.data = module.weight.data[mask]
                if module.bias is not None:
                    module.bias.data = module.bias.data[mask]
                module.out_features = mask.sum().item()


class RandomPruner(BasePruner):
    """Random pruning - randomly removes weights for baseline comparison."""

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        schedule: str = "exponential",
        scope: str = "global",
        seed: Optional[int] = None,
    ):
        super().__init__(model, sparsity, schedule, scope)
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def prune(self, sparsity: Optional[float] = None):
        """Apply random pruning."""
        sparsity = sparsity or self.sparsity
        self.masks = {}

        for name, param in self.model.named_parameters():
            mask = torch.rand(param.shape) > sparsity
            self.masks[name] = mask
            param.data *= mask.float()
