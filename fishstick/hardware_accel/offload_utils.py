"""
CPU/GPU Offload Utilities for fishstick.

Provides utilities for offloading model parameters and activations
between GPU and CPU memory to handle large models.

Based on:
- PyTorch offloading patterns
- DeepSpeed ZeRO optimizations
- "Reducing Transformer Memory Footprint" techniques
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Callable, Union
from contextlib import contextmanager
import threading
import weakref

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass
class OffloadConfig:
    """Configuration for CPU offloading."""

    offload_params: bool = True
    offload_buffers: bool = False
    offload_optimizer: bool = True
    pin_memory: bool = True
    prefetch_batches: int = 1
    async_offload: bool = True


class CPUOffload:
    """
    CPU offloading manager for large models.

    Automatically manages moving parameters and optimizer states
    between GPU and CPU to reduce GPU memory usage.

    Attributes:
        config: Offload configuration
    """

    def __init__(self, config: Optional[OffloadConfig] = None):
        self.config = config or OffloadConfig()
        self._offloaded_params: Dict[int, Tensor] = {}
        self._original_devices: Dict[int, torch.device] = {}
        self._lock = threading.Lock()

    def offload_param(
        self, param: Tensor, device: Optional[torch.device] = None
    ) -> Tensor:
        """
        Offload a parameter to CPU.

        Args:
            param: Parameter to offload
            device: Target device (CPU if None)

        Returns:
            Offloaded parameter
        """
        param_id = id(param)

        with self._lock:
            if param_id in self._offloaded_params:
                return self._offloaded_params[param_id]

            # Store original device
            self._original_devices[param_id] = param.device

            # Move to CPU
            cpu_device = device or torch.device("cpu")

            if self.config.pin_memory and cpu_device.type == "cpu":
                offloaded = param.cpu().pin_memory()
            else:
                offloaded = param.cpu()

            self._offloaded_params[param_id] = offloaded

            return offloaded

    def restore_param(
        self, param: Tensor, device: Optional[torch.device] = None
    ) -> Tensor:
        """
        Restore an offloaded parameter to GPU.

        Args:
            param: Parameter to restore
            device: Target device (original if None)

        Returns:
            Restored parameter
        """
        param_id = id(param)

        with self._lock:
            if param_id in self._original_devices:
                target_device = device or self._original_devices[param_id]

                if param_id in self._offloaded_params:
                    offloaded = self._offloaded_params[param_id]
                    param.data.copy_(offloaded.data)

                    # Clean up
                    del self._offloaded_params[param_id]

                param.data = param.data.to(target_device)
                del self._original_devices[param_id]

            return param

    def offload_model(self, model: nn.Module) -> None:
        """
        Offload entire model to CPU.

        Args:
            model: Model to offload
        """
        for param in model.parameters():
            if param.requires_grad:
                self.offload_param(param)

    def restore_model(
        self, model: nn.Module, device: Optional[torch.device] = None
    ) -> None:
        """
        Restore entire model to GPU.

        Args:
            model: Model to restore
            device: Target device
        """
        target_device = device or torch.device("cuda:0")

        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(target_device)

    def swap_param(
        self,
        param: Tensor,
        target_device: torch.device,
    ) -> Tensor:
        """
        Swap parameter to target device.

        Args:
            param: Parameter to swap
            target_device: Target device

        Returns:
            Swapped parameter
        """
        param_id = id(param)

        if target_device.type == "cpu":
            return self.offload_param(param)
        else:
            return self.restore_param(param, target_device)


class OffloadableModule(nn.Module):
    """
    Module that supports CPU offloading.

    Wraps a module with automatic offloading capabilities.

    Attributes:
        module: Module to wrap
        offload_manager: CPU offload manager
    """

    def __init__(
        self,
        module: nn.Module,
        offload_manager: Optional[CPUOffload] = None,
        offload_on_forward: bool = True,
        offload_on_backward: bool = False,
    ):
        super().__init__()
        self.module = module
        self.offload_manager = offload_manager or CPUOffload()
        self.offload_on_forward = offload_on_forward
        self.offload_on_backward = offload_on_backward

        self._offloaded_state: Dict[str, Any] = {}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward with optional offloading."""
        # Ensure params are on GPU for forward
        if self.offload_on_forward and self.training:
            self._restore_params()

        output = self.module(*args, **kwargs)

        # Offload back to CPU after forward
        if self.offload_on_forward and self.training:
            self._offload_params()

        return output

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        """Backward with optional offloading."""
        # Ensure params are on GPU for backward
        if self.offload_on_backward:
            self._restore_params()

        output = (
            self.module.backward(*args, **kwargs)
            if hasattr(self.module, "backward")
            else None
        )

        # Offload after backward
        if self.offload_on_backward:
            self._offload_params()

        return output

    def _offload_params(self) -> None:
        """Offload all parameters to CPU."""
        for name, param in self.module.named_parameters():
            self._offloaded_state[name] = self.offload_manager.offload_param(param)
            # Detach to break computation graph
            self._offloaded_state[name] = self._offloaded_state[name].detach()

    def _restore_params(self) -> None:
        """Restore all parameters to GPU."""
        for name, param in self.module.named_parameters():
            if name in self._offloaded_state:
                cpu_param = self._offloaded_state[name]
                param.data = cpu_param.to(param.device)
                del self._offloaded_state[name]


class ParameterOffloader:
    """
    Utility for managing parameter offloading with context awareness.

    Provides context managers for offloading during specific
    parts of the training loop.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cuda:0")
        self._offloaded: Dict[str, Tensor] = {}

    @contextmanager
    def offload_context(self, module_names: Optional[List[str]] = None):
        """
        Context manager for temporary offloading.

        Args:
            module_names: Specific modules to offload (None = all)
        """
        # Store original devices
        original_devices = {}

        try:
            # Offload specified modules
            for name, param in self.model.named_parameters():
                if module_names is None or any(m in name for m in module_names):
                    original_devices[name] = param.device
                    self._offloaded[name] = param.data.clone().to("cpu")

            yield

        finally:
            # Restore
            for name, param in self.model.named_parameters():
                if name in self._offloaded:
                    param.data = self._offloaded[name].to(self.device)
                    del self._offloaded[name]


def offload_to_cpu(
    tensor: Tensor,
    pin_memory: bool = True,
) -> Tensor:
    """
    Offload tensor to CPU.

    Args:
        tensor: Tensor to offload
        pin_memory: Whether to pin memory

    Returns:
        CPU tensor
    """
    if pin_memory and torch.cuda.is_available():
        return tensor.cpu().pin_memory()
    return tensor.cpu()


def offload_to_gpu(
    tensor: Tensor,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Offload tensor to GPU.

    Args:
        tensor: Tensor to offload
        device: Target GPU device

    Returns:
        GPU tensor
    """
    device = device or torch.device("cuda:0")
    return tensor.to(device)


class ActivationOffloader:
    """
    Offloader for model activations.

    Stores activations to CPU during forward pass and
    restores during backward to save GPU memory.
    """

    def __init__(self):
        self._stored_activations: Dict[str, Tensor] = {}
        self._hooks: List[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        """Register hooks to capture activations."""

        def forward_hook(name: str, module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, Tensor):
                self._stored_activations[name] = offload_to_cpu(output.detach())

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                hook = module.register_forward_hook(
                    lambda n, m, i, o, name=name: forward_hook(name, m, i, o)
                )
                self._hooks.append(hook)

    def clear_hooks(self) -> None:
        """Clear registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_activations(self) -> Dict[str, Tensor]:
        """Get stored activations."""
        return self._stored_activations

    def restore_activations(self) -> Dict[str, Tensor]:
        """Restore activations to GPU."""
        restored = {}
        for name, act in self._stored_activations.items():
            restored[name] = offload_to_gpu(act)
        return restored

    def clear(self) -> None:
        """Clear stored activations."""
        self._stored_activations.clear()


class OffloadOptimizerWrapper:
    """
    Wrapper for optimizer with CPU offloading.

    Keeps optimizer state on CPU and only moves to GPU
    during update steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        offload: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.optimizer = optimizer
        self.offload = offload
        self.device = device or torch.device("cuda:0")
        self.cpu_state: List[Dict[str, Any]] = []

        if offload:
            self._init_cpu_state()

    def _init_cpu_state(self) -> None:
        """Initialize CPU state for optimizer."""
        for group in self.optimizer.param_groups:
            group_state = {}
            for p in group["params"]:
                state = self.optimizer.state[p]
                cpu_state = {}
                for k, v in state.items():
                    if isinstance(v, Tensor):
                        cpu_state[k] = v.cpu()
                    else:
                        cpu_state[k] = v
                group_state[id(p)] = cpu_state
            self.cpu_state.append(group_state)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform optimizer step."""
        if not self.offload:
            return self.optimizer.step(closure)

        # Move states to GPU temporarily
        self._restore_to_gpu()

        # Perform step
        result = self.optimizer.step(closure)

        # Move states back to CPU
        self._save_to_cpu()

        return result

    def _restore_to_gpu(self) -> None:
        """Restore optimizer state to GPU."""
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for p in group["params"]:
                if id(p) in self.cpu_state[group_idx]:
                    state = self.optimizer.state[p]
                    cpu_state = self.cpu_state[group_idx][id(p)]
                    for k, v in cpu_state.items():
                        if isinstance(v, Tensor):
                            state[k] = v.to(self.device)
                        else:
                            state[k] = v

    def _save_to_cpu(self) -> None:
        """Save optimizer state to CPU."""
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for p in group["params"]:
                state = self.optimizer.state[p]
                cpu_state = {}
                for k, v in state.items():
                    if isinstance(v, Tensor):
                        cpu_state[k] = v.cpu()
                    else:
                        cpu_state[k] = v
                self.cpu_state[group_idx][id(p)] = cpu_state

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        self.optimizer.load_state_dict(state_dict)
        if self.offload:
            self._init_cpu_state()


__all__ = [
    "OffloadConfig",
    "CPUOffload",
    "OffloadableModule",
    "ParameterOffloader",
    "offload_to_cpu",
    "offload_to_gpu",
    "ActivationOffloader",
    "OffloadOptimizerWrapper",
]
