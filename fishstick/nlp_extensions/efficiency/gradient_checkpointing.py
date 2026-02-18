"""
Gradient Checkpointing Utilities

Memory-efficient training through gradient checkpointing.
Implements various checkpointing strategies to reduce memory
usage at the cost of extra computation.
"""

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint_sequential
from typing import Callable, List, Optional, Sequence
import functools


class GradientCheckpointing:
    """Gradient checkpointing manager for efficient memory usage.

    Reduces memory usage by recomputing intermediate activations
    during backward pass instead of storing them forward.
    """

    @staticmethod
    def checkpoint_sequential(
        modules: Sequence[nn.Module],
        segments: int,
        input: Tensor,
        use_reentrant: bool = True,
    ) -> Tensor:
        """Checkpoint a sequential module.

        Args:
            modules: Sequence of modules to checkpoint
            segments: Number of segments to split into
            input: Input tensor
            use_reentrant: Whether to use reentrant checkpointing

        Returns:
            Output tensor
        """
        return checkpoint_sequential(modules, segments, input, use_reentrant)

    @staticmethod
    def checkpoint(
        function: Callable,
        use_reentrant: bool = True,
        *args,
        **kwargs,
    ) -> Tensor:
        """Checkpoint a function call.

        Args:
            function: Function to checkpoint
            use_reentrant: Whether to use reentrant checkpointing
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Output tensor
        """
        return torch.utils.checkpoint.checkpoint(
            function,
            *args,
            use_reentrant=use_reentrant,
            **kwargs,
        )


class CheckpointWrapper(nn.Module):
    """Wrapper that applies gradient checkpointing to a module.

    Can be applied to any nn.Module to enable memory-efficient
    forward passes.
    """

    def __init__(
        self,
        module: nn.Module,
        use_reentrant: bool = True,
    ):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        """Forward pass with gradient checkpointing."""
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self.module.forward,
                *args,
                use_reentrant=self.use_reentrant,
                **kwargs,
            )
        return self.module(*args, **kwargs)


class ModularGradientCheckpointing:
    """Modular gradient checkpointing with custom strategies.

    Supports different checkpointing strategies for different
    parts of the model.
    """

    def __init__(self, checkpoint_ratio: float = 0.5):
        self.checkpoint_ratio = checkpoint_ratio

    def apply_to_model(
        self,
        model: nn.Module,
        strategy: str = "uniform",
    ) -> nn.Module:
        """Apply gradient checkpointing to a model.

        Args:
            model: Model to apply checkpointing to
            strategy: Checkpointing strategy ('uniform', 'attention', 'mlp')

        Returns:
            Model with gradient checkpointing applied
        """
        if strategy == "uniform":
            return self._apply_uniform(model)
        elif strategy == "attention":
            return self._apply_attention_only(model)
        elif strategy == "mlp":
            return self._apply_mlp_only(model)
        return model

    def _apply_uniform(self, model: nn.Module) -> nn.Module:
        """Apply uniform checkpointing to all layers."""
        layers = []

        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                wrapped = self._apply_uniform(module)
                layers.append(wrapped)
            else:
                checkpointed = CheckpointWrapper(module)
                layers.append(checkpointed)

        return nn.Sequential(*layers)

    def _apply_attention_only(self, model: nn.Module) -> nn.Module:
        """Apply checkpointing only to attention layers."""
        for name, module in model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                wrapped = CheckpointWrapper(module)
                setattr(parent, child_name, wrapped)

        return model

    def _apply_mlp_only(self, model: nn.Module) -> nn.Module:
        """Apply checkpointing only to MLP layers."""
        for name, module in model.named_modules():
            if "mlp" in name.lower() or "feedforward" in name.lower():
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                wrapped = CheckpointWrapper(module)
                setattr(parent, child_name, wrapped)

        return model


def checkpoint_wrapper(module: nn.Module) -> nn.Module:
    """Convenience function to wrap a module with gradient checkpointing.

    Args:
        module: Module to wrap

    Returns:
        Wrapped module with gradient checkpointing
    """
    return CheckpointWrapper(module)


def create_gradient_checkpointing_function(
    module: nn.Module,
    num_checkpoint_segments: int,
) -> Callable:
    """Create a checkpointed forward function for a module.

    Args:
        module: Module to checkpoint
        num_checkpoint_segments: Number of segments to split into

    Returns:
        Checkpointed forward function
    """
    layers = list(module.children())

    @functools.wraps(module.forward)
    def checkpointed_forward(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint_sequential(
            layers,
            num_checkpoint_segments,
            *args,
            **kwargs,
        )

    return checkpointed_forward


class MixedPrecisionCheckpointing:
    """Mixed precision gradient checkpointing.

    Combines gradient checkpointing with mixed precision training
    for maximum memory efficiency.
    """

    def __init__(self, checkpoint_ratio: float = 0.5):
        self.checkpoint_ratio = checkpoint_ratio

    def apply(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None,
    ) -> nn.Module:
        """Apply mixed precision checkpointing to model.

        Args:
            model: Model to apply checkpointing to
            layer_indices: Specific layer indices to checkpoint

        Returns:
            Model with mixed precision checkpointing
        """
        layers = []

        for idx, module in enumerate(model.children()):
            if layer_indices is None or idx in layer_indices:
                checkpointed = CheckpointWrapper(module)
                layers.append(checkpointed)
            else:
                layers.append(module)

        return nn.Sequential(*layers)


class SelectiveCheckpointing:
    """Selective gradient checkpointing based on layer size.

    Only checkpoints layers above a certain size threshold
    to optimize the trade-off between memory and compute.
    """

    def __init__(self, size_threshold: int = 100000):
        self.size_threshold = size_threshold

    def apply(self, model: nn.Module) -> nn.Module:
        """Apply selective checkpointing based on parameter count.

        Args:
            model: Model to apply checkpointing to

        Returns:
            Model with selective checkpointing
        """
        layers = []

        for module in model.children():
            num_params = sum(p.numel() for p in module.parameters())

            if num_params > self.size_threshold:
                checkpointed = CheckpointWrapper(module)
                layers.append(checkpointed)
            else:
                layers.append(module)

        return nn.Sequential(*layers)


class RecomputationScheduler:
    """Dynamic recomputation scheduler for adaptive checkpointing.

    Adjusts checkpointing strategy based on available memory
    during training.
    """

    def __init__(
        self,
        initial_segments: int = 1,
        max_segments: int = 8,
        memory_growth_factor: float = 1.2,
    ):
        self.initial_segments = initial_segments
        self.max_segments = max_segments
        self.memory_growth_factor = memory_growth_factor
        self.current_segments = initial_segments

    def increase_segments(self) -> None:
        """Increase checkpointing granularity."""
        if self.current_segments < self.max_segments:
            self.current_segments = min(
                self.current_segments + 1,
                self.max_segments,
            )

    def decrease_segments(self) -> None:
        """Decrease checkpointing granularity."""
        if self.current_segments > 1:
            self.current_segments -= 1

    def get_segments(self) -> int:
        """Get current number of segments."""
        return self.current_segments
