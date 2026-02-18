"""
Device Management for fishstick.

Provides unified device management across CPU, GPU, and TPU backends.

Based on:
- PyTorch device management
- JAX device abstractions
- NVIDIA CUDA runtime API
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union, List, Dict, Any, Callable
from contextlib import contextmanager
import threading
import os

import torch
from torch import Tensor, nn


class DeviceType(Enum):
    """Supported device types for computation."""

    CPU = auto()
    CUDA = auto()
    MPS = auto()  # Apple Silicon
    TPU = auto()
    IPU = auto()  # Graphcore
    XLA = auto()
    AUTO = auto()  # Auto-select best available


@dataclass
class DeviceProperties:
    """Properties of a compute device."""

    name: str
    device_type: DeviceType
    memory_total: int  # bytes
    memory_available: int  # bytes
    compute_capability: Optional[tuple[int, int]] = None
    multiprocessor_count: Optional[int] = None
    max_threads_per_block: Optional[int] = None
    num_threads: Optional[int] = None
    cores: Optional[int] = None
    tpu_cores: Optional[int] = None

    @property
    def memory_total_gb(self) -> float:
        """Total memory in GB."""
        return self.memory_total / (1024**3)

    @property
    def memory_available_gb(self) -> float:
        """Available memory in GB."""
        return self.memory_available / (1024**3)


class DeviceManager:
    """
    Unified device management for CPU, GPU, and TPU.

    Provides a single interface to manage device allocation,
    memory tracking, and device selection across backends.

    Attributes:
        current_device: Currently active device
        device_type: Type of device being used
    """

    _instance: Optional[DeviceManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> DeviceManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._device_type = DeviceType.AUTO
        self._current_device_id = 0
        self._device_cache: Dict[int, DeviceProperties] = {}
        self._tpu_initialized = False
        self._tpu_strategy: Optional[str] = None

    @property
    def device_type(self) -> DeviceType:
        """Get current device type."""
        if self._device_type == DeviceType.AUTO:
            return self._auto_detect_device_type()
        return self._device_type

    @device_type.setter
    def device_type(self, value: DeviceType) -> None:
        """Set device type."""
        self._device_type = value

    def _auto_detect_device_type(self) -> DeviceType:
        """Auto-detect the best available device type."""
        if is_tpu_available():
            return DeviceType.TPU
        elif torch.cuda.is_available():
            return DeviceType.CUDA
        elif torch.backends.mps.is_available():
            return DeviceType.MPS
        else:
            return DeviceType.CPU

    def get_device(self, device_id: int = 0) -> torch.device:
        """
        Get torch.device for the specified device.

        Args:
            device_id: Device index (for multi-GPU)

        Returns:
            torch.device object
        """
        if self.device_type == DeviceType.CUDA:
            return torch.device(f"cuda:{device_id}")
        elif self.device_type == DeviceType.MPS:
            return torch.device("mps")
        elif self.device_type == DeviceType.TPU:
            return torch.device("xla:0")
        else:
            return torch.device("cpu")

    def get_device_properties(
        self, device: Optional[torch.device] = None
    ) -> DeviceProperties:
        """
        Get properties for a specific device.

        Args:
            device: Target device (uses current if None)

        Returns:
            DeviceProperties with device information
        """
        if device is None:
            device = self.get_device()

        device_id = device.index or 0

        if device_id in self._device_cache:
            return self._device_cache[device_id]

        if self.device_type == DeviceType.CUDA:
            props = torch.cuda.get_device_properties(device_id)
            props_dict = {
                "name": props.name,
                "device_type": DeviceType.CUDA,
                "memory_total": props.total_memory,
                "memory_available": torch.cuda.mem_get_info(device_id)[0],
                "compute_capability": (props.major, props.multi_processor_count),
                "multiprocessor_count": props.multi_processor_count,
                "max_threads_per_block": props.max_threads_per_block,
            }
        elif self.device_type == DeviceType.MPS:
            props_dict = {
                "name": "Apple Silicon MPS",
                "device_type": DeviceType.MPS,
                "memory_total": 0,  # Not exposed by MPS
                "memory_available": 0,
            }
        elif self.device_type == DeviceType.TPU:
            props_dict = self._get_tpu_properties()
        else:
            import psutil

            vm = psutil.virtual_memory()
            props_dict = {
                "name": "CPU",
                "device_type": DeviceType.CPU,
                "memory_total": vm.total,
                "memory_available": vm.available,
                "num_threads": os.cpu_count(),
                "cores": os.cpu_count(),
            }

        props = DeviceProperties(**props_dict)
        self._device_cache[device_id] = props
        return props

    def _get_tpu_properties(self) -> Dict[str, Any]:
        """Get TPU properties if available."""
        try:
            import jax

            tpu_devices = jax.devices("tpu")
            return {
                "name": f"TPU v{len(tpu_devices)}",
                "device_type": DeviceType.TPU,
                "memory_total": 0,  # Not typically exposed
                "memory_available": 0,
                "tpu_cores": len(tpu_devices),
            }
        except ImportError:
            return {
                "name": "TPU",
                "device_type": DeviceType.TPU,
                "memory_total": 0,
                "memory_available": 0,
            }

    def set_device(self, device: Union[int, str, torch.device]) -> None:
        """
        Set the current device.

        Args:
            device: Device index, device string, or torch.device
        """
        if isinstance(device, int):
            self._current_device_id = device
            if self.device_type == DeviceType.CUDA and torch.cuda.is_available():
                torch.cuda.set_device(device)
        elif isinstance(device, str):
            if device.startswith("cuda"):
                self._device_type = DeviceType.CUDA
                self._current_device_id = (
                    int(device.split(":")[1]) if ":" in device else 0
                )
            elif device == "mps":
                self._device_type = DeviceType.MPS
            elif device == "xla" or device.startswith("tpu"):
                self._device_type = DeviceType.TPU
            else:
                self._device_type = DeviceType.CPU

    @contextmanager
    def device_context(self, device: Union[int, str, torch.device]) -> None:
        """
        Context manager for temporary device change.

        Args:
            device: Device to use within context
        """
        old_device = self.get_device()
        old_device_type = self._device_type

        try:
            self.set_device(device)
            yield
        finally:
            self.set_device(old_device)
            self._device_type = old_device_type

    def synchronize(self) -> None:
        """Synchronize all pending operations."""
        if self.device_type == DeviceType.CUDA:
            torch.cuda.synchronize()
        elif self.device_type == DeviceType.MPS:
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        """Empty CUDA cache to free memory."""
        if self.device_type == DeviceType.CUDA:
            torch.cuda.empty_cache()

    def memory_summary(self) -> str:
        """Get memory summary for current device."""
        if self.device_type == DeviceType.CUDA:
            return torch.cuda.memory_summary()
        return "No CUDA memory summary available"

    def reset_peak_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.device_type == DeviceType.CUDA:
            torch.cuda.reset_peak_memory_stats()


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_tpu_available() -> bool:
    """Check if TPU is available."""
    try:
        import torch_xla

        return True
    except ImportError:
        try:
            import jax

            return len(jax.devices("tpu")) > 0
        except ImportError:
            return False


def get_optimal_device(
    prefer_gpu: bool = True,
    memory_threshold_gb: float = 2.0,
) -> torch.device:
    """
    Get the optimal device for computation.

    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        memory_threshold_gb: Minimum required free memory for GPU

    Returns:
        Optimal torch.device
    """
    manager = DeviceManager()

    if prefer_gpu:
        if torch.cuda.is_available():
            props = manager.get_device_properties()
            if props.memory_available_gb >= memory_threshold_gb:
                return manager.get_device()
        elif torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


def get_device_properties(device: Optional[torch.device] = None) -> DeviceProperties:
    """
    Get properties for a device.

    Args:
        device: Target device (uses CUDA:0 if None and CUDA available)

    Returns:
        DeviceProperties
    """
    manager = DeviceManager()
    if device is None and torch.cuda.is_available():
        device = torch.device("cuda:0")
    return manager.get_device_properties(device)


_global_device_manager = DeviceManager()


__all__ = [
    "DeviceManager",
    "DeviceType",
    "DeviceProperties",
    "is_cuda_available",
    "is_tpu_available",
    "get_optimal_device",
    "get_device_properties",
]
