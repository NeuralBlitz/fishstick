"""
TPU Compatibility Layer for fishstick.

Provides seamless integration with Google Tensor Processing Units (TPUs)
via PyTorch/XLA and JAX backends.

Based on:
- PyTorch/XLA documentation
- Google Cloud TPU documentation
- JAX/Flax TPU integration
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any, List, Callable, Union
from contextlib import contextmanager
import threading
import os

import torch
from torch import Tensor, nn


# Try to import torch_xla for TPU support
_TPU_AVAILABLE = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    _TPU_AVAILABLE = True
except ImportError:
    xm = None


class TPUStrategy(Enum):
    """TPU distribution strategies."""

    SINGLE = auto()  # Single TPU core
    DATA_PARALLEL = auto()  # Data parallel across cores
    PIPELINE = auto()  # Pipeline parallelism
    TENSOR_PARALLEL = auto()  # Tensor parallelism
    SHARDING = auto()  # JAX-style mesh sharding


@dataclass
class TPUConfig:
    """Configuration for TPU training."""

    strategy: TPUStrategy = TPUStrategy.DATA_PARALLEL
    num_cores: int = 1
    mesh_shape: Optional[Tuple[int, int]] = None
    backend: str = "pjrt"  # pjrt or xla
    deterministic: bool = False
    allow_tf32: bool = True


class TPUModel(nn.Module):
    """
    Wrapper for TPU-compatible models.

    Handles TPU-specific operations like model sharding,
    replicated parameters, and gradient synchronization.

    Attributes:
        module: The underlying model
        config: TPU configuration
    """

    def __init__(
        self,
        module: nn.Module,
        config: Optional[TPUConfig] = None,
    ):
        super().__init__()
        self.module = module
        self.config = config or TPUConfig()

        self._is_tpu = is_tpu()
        self._replicated_params: Dict[str, Tensor] = {}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass with TPU optimization."""
        return self.module(*args, **kwargs)

    def topu(self) -> "TPUModel_t":
        """Move model to TPU."""
        if not self._is_tpu:
            raise RuntimeError("TPU not available")

        # Use xm.move_to_device for TPU
        if xm is not None:
            self.module = xm.send_cpu_data_to_device(self.module, xm.xla_device())

        return self

    def xla_mesh_reduce(
        self,
        tensor: Tensor,
        reduce_op: str = "sum",
    ) -> Tensor:
        """
        Perform XLA mesh reduction across TPU cores.

        Args:
            tensor: Tensor to reduce
            reduce_op: Reduction operation ("sum", "mean", "min", "max")

        Returns:
            Reduced tensor
        """
        if not self._is_tpu or xm is None:
            return tensor

        reduce_fn = {
            "sum": xm.mesh_reduce,
            "mean": xm.mesh_reduce,
        }.get(reduce_op, xm.mesh_reduce)

        return reduce_fn(lambda x: x, tensor)

    def mark_step(self) -> None:
        """
        Mark a step boundary for TPU computation.

        Triggers computation compilation and device transfer.
        """
        if xm is not None:
            xm.mark_step()

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        """
        Perform TPU-optimized optimizer step.

        Args:
            optimizer: Optimizer
            closure: Loss closure

        Returns:
            Loss value
        """
        if xm is not None:
            return xm.optimizer_step(optimizer, closure)

        return optimizer.step(closure) if closure else None

    def broadcast(self, tensor: Tensor, src: int = 0) -> Tensor:
        """
        Broadcast tensor across TPU cores.

        Args:
            tensor: Tensor to broadcast
            src: Source core

        Returns:
            Broadcasted tensor
        """
        if xm is not None:
            return xm.all_reduce("min", [tensor])[0]
        return tensor

    def get_local_world_size(self) -> int:
        """Get number of TPU cores in current process."""
        if xm is not None:
            return xm.xrt_world_size()
        return 1

    def get_local_rank(self) -> int:
        """Get rank of current TPU core."""
        if xm is not None:
            return xm.get_local_ordinal()
        return 0

    def get_global_rank(self) -> int:
        """Get global rank across all TPU cores."""
        if xm is not None:
            return xm.get_ordinal()
        return 0


def is_tpu() -> bool:
    """
    Check if TPU is available.

    Returns:
        True if TPU is available
    """
    return _TPU_AVAILABLE


def tpu_mesh(
    mesh_shape: Optional[Tuple[int, int]] = None,
    axis_names: Tuple[str, str] = ("batch", "model"),
) -> Dict[str, Any]:
    """
    Create a TPU mesh for distributed computation.

    Args:
        mesh_shape: Shape of the mesh (batch, model)
        axis_names: Names for mesh axes

    Returns:
        Mesh configuration
    """
    if not is_tpu():
        return {"available": False}

    if mesh_shape is None:
        # Auto-detect TPU cores
        devices = get_tpu_devices()
        mesh_shape = (len(devices), 1)

    return {
        "available": True,
        "mesh_shape": mesh_shape,
        "axis_names": axis_names,
    }


def get_tpu_devices() -> List[str]:
    """
    Get list of available TPU devices.

    Returns:
        List of TPU device strings
    """
    if not is_tpu() or xm is None:
        return []

    return [str(d) for d in xm.get_xla_devices()]


def configure_tpu(
    config: Optional[TPUConfig] = None,
    env_vars: Optional[Dict[str, str]] = None,
) -> TPUConfig:
    """
    Configure TPU environment and settings.

    Args:
        config: TPU configuration
        env_vars: Additional environment variables

    Returns:
        Applied TPU configuration
    """
    config = config or TPUConfig()

    # Set environment variables
    env_vars = env_vars or {}

    if config.deterministic:
        env_vars["XLA_DETERMINISTIC"] = "1"
    if not config.allow_tf32:
        env_vars["XLA_ALLOW_TF32"] = "0"

    env_vars["PJRT_DEVICE"] = config.backend

    for key, value in env_vars.items():
        os.environ[key] = value

    return config


class TPUDistributedSampler:
    """
    Distributed sampler for TPU data loading.

    Ensures each TPU core gets a unique batch of data.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
            indices = torch.randperm(len(dataset), generator=generator).tolist()
        else:
            indices = list(range(len(dataset)))

        self.indices = indices

    def __iter__(self) -> Any:
        # Split indices across replicas
        start = self.rank
        step = self.num_replicas

        return iter(self.indices[start::step])

    def __len__(self) -> int:
        return (len(self.dataset) + self.num_replicas - 1) // self.num_replicas


class TPUStrategyExecutor:
    """
    Execute training with specific TPU strategy.

    Supports data parallelism, pipelining, and tensor parallelism
    across TPU cores.
    """

    def __init__(
        self,
        strategy: TPUStrategy,
        config: Optional[TPUConfig] = None,
    ):
        self.strategy = strategy
        self.config = config or TPUConfig()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the strategy."""
        if self.strategy == TPUStrategy.DATA_PARALLEL:
            self._init_data_parallel()
        elif self.strategy == TPUStrategy.SHARDING:
            self._init_sharding()

        self._initialized = True

    def _init_data_parallel(self) -> None:
        """Initialize data parallel strategy."""
        if not is_tpu():
            return

        # Initialize XLA collective
        if xm is not None:
            xm.rendezvous("data_parallel_init")

    def _init_sharding(self) -> None:
        """Initialize sharding strategy."""
        if not is_tpu() or xm is None:
            return

        # Configure mesh
        if self.config.mesh_shape:
            xm.set_mesh(*self.config.mesh_shape)

    def all_reduce(
        self,
        tensor: Tensor,
        op: str = "sum",
    ) -> Tensor:
        """
        Perform all-reduce across TPU cores.

        Args:
            tensor: Input tensor
            op: Reduction operation

        Returns:
            Reduced tensor
        """
        if not is_tpu() or xm is None:
            return tensor

        return xm.all_reduce(op, [tensor])[0]

    def all_gather(self, tensor: Tensor) -> Tensor:
        """
        Gather tensors from all TPU cores.

        Args:
            tensor: Input tensor

        Returns:
            Gathered tensor
        """
        if not is_tpu() or xm is None:
            return tensor

        return xm.all_gather([tensor])[0]

    def barrier(self) -> None:
        """Synchronize all TPU cores."""
        if is_tpu() and xm is not None:
            xm.rendezvous("barrier")


def create_tpu_model(
    model: nn.Module,
    strategy: TPUStrategy = TPUStrategy.DATA_PARALLEL,
    config: Optional[TPUConfig] = None,
) -> TPUModel:
    """
    Create a TPU-optimized model.

    Args:
        model: Base model
        strategy: Distribution strategy
        config: TPU configuration

    Returns:
        TPUModel wrapper
    """
    tpu_model = TPUModel(model, config)

    if is_tpu():
        tpu_model = tpu_model.to_tpu()

    return tpu_model


__all__ = [
    "TPUConfig",
    "TPUStrategy",
    "TPUModel",
    "is_tpu",
    "tpu_mesh",
    "get_tpu_devices",
    "configure_tpu",
    "TPUDistributedSampler",
    "TPUStrategyExecutor",
    "create_tpu_model",
]
