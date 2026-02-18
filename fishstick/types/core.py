"""Fishstick Core Types Module.

Comprehensive type definitions for the Fishstick deep learning framework.
Provides type hints, protocols, and utilities for tensor operations,
model definitions, data handling, training, and inference.
"""

from __future__ import annotations

import inspect
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

if TYPE_CHECKING:
    import numpy as np

    try:
        import torch

        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        torch = Any  # type: ignore
else:
    TORCH_AVAILABLE = False
    torch = Any


# =============================================================================
# Generic Types
# =============================================================================

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
K = TypeVar("K")
V = TypeVar("V")

Number = Union[int, float, complex]
Numeric = Union[int, float, np.ndarray, Any]  # Any for tensor types
Scalar = Union[int, float, bool, str, bytes]
PathLike = Union[str, "os.PathLike[str]"]
JSONValue = Union[
    None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]
]


# =============================================================================
# Shape and Data Type Types
# =============================================================================


class DType(Enum):
    """Enumeration of supported data types."""

    # Floating point
    FLOAT16 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    BFLOAT16 = auto()

    # Integer
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()

    # Unsigned integer
    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    UINT64 = auto()

    # Boolean
    BOOL = auto()

    # Complex
    COMPLEX64 = auto()
    COMPLEX128 = auto()

    # Quantized
    QINT8 = auto()
    QUINT8 = auto()
    QINT32 = auto()


class Shape(NamedTuple):
    """Represents a tensor shape."""

    dims: Tuple[int, ...]

    def __init__(self, *dims: int) -> None:
        object.__setattr__(self, "dims", dims)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.dims)

    @property
    def size(self) -> int:
        """Total number of elements."""
        result = 1
        for d in self.dims:
            result *= d
        return result

    @property
    def batch_dim(self) -> Optional[int]:
        """Get batch dimension (typically first dimension)."""
        return self.dims[0] if self.dims else None

    def __iter__(self) -> Iterator[int]:
        return iter(self.dims)

    def __len__(self) -> int:
        return len(self.dims)

    def __getitem__(self, idx: Union[int, slice]) -> Union[int, Tuple[int, ...]]:
        return self.dims[idx]  # type: ignore

    def __repr__(self) -> str:
        return f"Shape{self.dims}"

    def __str__(self) -> str:
        return f"({', '.join(map(str, self.dims))})"

    @classmethod
    def from_list(cls, dims: List[int]) -> "Shape":
        """Create Shape from list of dimensions."""
        return cls(*dims)

    @classmethod
    def from_tuple(cls, dims: Tuple[int, ...]) -> "Shape":
        """Create Shape from tuple of dimensions."""
        return cls(*dims)

    def compatible_with(self, other: "Shape") -> bool:
        """Check if shapes are compatible (same rank, matching non-batch dims)."""
        if self.ndim != other.ndim:
            return False
        return all(s == o or s == -1 or o == -1 for s, o in zip(self.dims, other.dims))

    def broadcast_with(self, other: "Shape") -> Optional["Shape"]:
        """Compute broadcasted shape."""
        if self.ndim < other.ndim:
            return other.broadcast_with(self)

        result = list(self.dims)
        offset = self.ndim - other.ndim

        for i, dim in enumerate(other.dims):
            idx = offset + i
            if result[idx] == 1:
                result[idx] = dim
            elif dim != 1 and result[idx] != dim:
                return None

        return Shape(*result)


# =============================================================================
# Tensor Types
# =============================================================================


@dataclass
class TensorInfo:
    """Metadata for tensor storage."""

    shape: Shape
    dtype: DType
    device: str = "cpu"
    requires_grad: bool = False

    @property
    def numel(self) -> int:
        """Total number of elements."""
        return self.shape.size


class Tensor:
    """Generic tensor type wrapper.

    Provides a unified interface for different tensor backends
    (PyTorch, NumPy, etc.).
    """

    def __init__(
        self,
        data: Any,
        shape: Optional[Shape] = None,
        dtype: Optional[DType] = None,
        device: Optional[str] = None,
    ) -> None:
        self._data = data
        self._info = TensorInfo(
            shape=shape or self._infer_shape(data),
            dtype=dtype or self._infer_dtype(data),
            device=device or "cpu",
        )

    def _infer_shape(self, data: Any) -> Shape:
        """Infer shape from data."""
        if hasattr(data, "shape"):
            return Shape(*data.shape)
        return Shape()

    def _infer_dtype(self, data: Any) -> DType:
        """Infer dtype from data."""
        if hasattr(data, "dtype"):
            dtype_str = str(data.dtype)
            dtype_map = {
                "float32": DType.FLOAT32,
                "float64": DType.FLOAT64,
                "float16": DType.FLOAT16,
                "int32": DType.INT32,
                "int64": DType.INT64,
                "bool": DType.BOOL,
            }
            return dtype_map.get(dtype_str, DType.FLOAT32)
        return DType.FLOAT32

    @property
    def data(self) -> Any:
        """Raw tensor data."""
        return self._data

    @property
    def shape(self) -> Shape:
        """Tensor shape."""
        return self._info.shape

    @property
    def dtype(self) -> DType:
        """Tensor data type."""
        return self._info.dtype

    @property
    def device(self) -> str:
        """Device where tensor is stored."""
        return self._info.device

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._info.shape.ndim

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._info.numel

    def to(self, device: str) -> "Tensor":
        """Move tensor to device."""
        # Placeholder implementation
        return Tensor(self._data, self._info.shape, self._info.dtype, device)

    def cpu(self) -> "Tensor":
        """Move tensor to CPU."""
        return self.to("cpu")

    def cuda(self, device: Optional[int] = None) -> "Tensor":
        """Move tensor to CUDA."""
        device_str = f"cuda:{device}" if device is not None else "cuda"
        return self.to(device_str)

    def numpy(self) -> "np.ndarray":
        """Convert to NumPy array."""
        if TORCH_AVAILABLE and isinstance(self._data, torch.Tensor):
            return self._data.cpu().numpy()
        if isinstance(self._data, np.ndarray):
            return self._data
        return np.array(self._data)

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype.name}, device={self.device})"
        )

    def __getitem__(self, key: Any) -> "Tensor":
        """Get item/slice."""
        return Tensor(self._data[key])

    def __add__(self, other: Union["Tensor", Number]) -> "Tensor":
        """Add operation."""
        if isinstance(other, Tensor):
            return Tensor(self._data + other._data)
        return Tensor(self._data + other)

    def __mul__(self, other: Union["Tensor", Number]) -> "Tensor":
        """Multiply operation."""
        if isinstance(other, Tensor):
            return Tensor(self._data * other._data)
        return Tensor(self._data * other)


class SparseTensor:
    """Sparse tensor representation.

    Stores only non-zero values with their indices.
    """

    def __init__(
        self,
        indices: Tensor,
        values: Tensor,
        shape: Shape,
        format: str = "coo",  # coo, csr, csc
    ) -> None:
        self.indices = indices
        self.values = values
        self._shape = shape
        self.format = format

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return self.values.size

    @property
    def density(self) -> float:
        """Ratio of non-zero to total elements."""
        return self.nnz / self._shape.size

    def to_dense(self) -> Tensor:
        """Convert to dense tensor."""
        # Placeholder implementation
        return Tensor(
            np.zeros(self._shape.dims),
            shape=self._shape,
            dtype=self.values.dtype,
        )

    @classmethod
    def from_dense(cls, tensor: Tensor, threshold: float = 0.0) -> "SparseTensor":
        """Create sparse tensor from dense."""
        # Placeholder implementation
        arr = tensor.numpy()
        mask = np.abs(arr) > threshold
        indices = np.argwhere(mask)
        values = arr[mask]

        return cls(
            indices=Tensor(indices),
            values=Tensor(values),
            shape=tensor.shape,
        )


class NamedTensor:
    """Tensor with named dimensions.

    Enables semantic tensor operations using dimension names.
    """

    def __init__(
        self,
        data: Tensor,
        names: Tuple[str, ...],
    ) -> None:
        if len(names) != data.ndim:
            raise ValueError(
                f"Number of names ({len(names)}) must match tensor dimensions ({data.ndim})"
            )
        self._data = data
        self._names = names

    @property
    def data(self) -> Tensor:
        return self._data

    @property
    def names(self) -> Tuple[str, ...]:
        return self._names

    @property
    def shape(self) -> Dict[str, int]:
        """Get shape as dimension name mapping."""
        return dict(zip(self._names, self._data.shape.dims))

    def rename(self, **kwargs: str) -> "NamedTensor":
        """Rename dimensions."""
        new_names = list(self._names)
        for old, new in kwargs.items():
            if old in self._names:
                idx = self._names.index(old)
                new_names[idx] = new
        return NamedTensor(self._data, tuple(new_names))

    def align_to(self, *names: str) -> "NamedTensor":
        """Permute dimensions to match order."""
        # Placeholder implementation
        return NamedTensor(self._data, names)

    def __repr__(self) -> str:
        dims = ", ".join(f"{n}={s}" for n, s in self.shape.items())
        return f"NamedTensor({dims})"


class QuantizedTensor:
    """Quantized tensor for efficient inference.

    Stores low-bit representation with scale and zero point.
    """

    def __init__(
        self,
        data: Tensor,
        scale: float,
        zero_point: int,
        num_bits: int = 8,
        symmetric: bool = True,
    ) -> None:
        self._data = data
        self.scale = scale
        self.zero_point = zero_point
        self.num_bits = num_bits
        self.symmetric = symmetric

    @property
    def data(self) -> Tensor:
        return self._data

    @property
    def shape(self) -> Shape:
        return self._data.shape

    @property
    def dtype(self) -> DType:
        """Returns quantized dtype based on num_bits."""
        if self.num_bits == 8:
            return DType.QINT8 if self.symmetric else DType.QUINT8
        elif self.num_bits == 32:
            return DType.QINT32
        return DType.QINT8

    def dequantize(self) -> Tensor:
        """Convert back to floating point."""
        # Placeholder implementation
        data = self._data.numpy().astype(np.float32)
        dequantized = (data - self.zero_point) * self.scale
        return Tensor(dequantized, shape=self._data.shape)

    @classmethod
    def quantize(
        cls,
        tensor: Tensor,
        num_bits: int = 8,
        symmetric: bool = True,
    ) -> "QuantizedTensor":
        """Quantize a floating point tensor."""
        arr = tensor.numpy()
        min_val = arr.min()
        max_val = arr.max()

        qmax = 2 ** (num_bits - 1) - 1 if symmetric else 2**num_bits - 1
        qmin = -(2 ** (num_bits - 1)) if symmetric else 0

        if symmetric:
            scale = max(abs(min_val), abs(max_val)) / qmax
            zero_point = 0
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = int(round(qmin - min_val / scale))

        quantized = np.clip(np.round(arr / scale) + zero_point, qmin, qmax)

        return cls(
            data=Tensor(quantized.astype(np.int32)),
            scale=scale,
            zero_point=zero_point,
            num_bits=num_bits,
            symmetric=symmetric,
        )


# =============================================================================
# Model Types
# =============================================================================


@dataclass
class ModelConfig:
    """Base configuration for models."""

    name: str = "model"
    version: str = "1.0.0"
    description: str = ""
    input_shape: Optional[Shape] = None
    output_shape: Optional[Shape] = None
    dtype: DType = DType.FLOAT32
    device: str = "cpu"
    training: bool = True


class Module(ABC):
    """Base class for all neural network modules.

    Follows PyTorch-like API for module composition.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = name or self.__class__.__name__
        self._parameters: Dict[str, Tensor] = {}
        self._buffers: Dict[str, Tensor] = {}
        self._modules: Dict[str, "Module"] = {}
        self._training = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def training(self) -> bool:
        return self._training

    def parameters(self) -> Iterator[Tensor]:
        """Iterate over all parameters."""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        """Iterate over named parameters."""
        for name, param in self._parameters.items():
            yield f"{self._name}.{name}", param
        for name, module in self._modules.items():
            for n, p in module.named_parameters():
                yield f"{self._name}.{n}", p

    def buffers(self) -> Iterator[Tensor]:
        """Iterate over all buffers."""
        for buffer in self._buffers.values():
            yield buffer
        for module in self._modules.values():
            yield from module.buffers()

    def modules(self) -> Iterator["Module"]:
        """Iterate over all sub-modules."""
        for module in self._modules.values():
            yield module
            yield from module.modules()

    def add_module(self, name: str, module: "Module") -> None:
        """Add a sub-module."""
        self._modules[name] = module

    def register_parameter(self, name: str, param: Tensor) -> None:
        """Register a parameter."""
        self._parameters[name] = param

    def register_buffer(self, name: str, buffer: Tensor) -> None:
        """Register a buffer."""
        self._buffers[name] = buffer

    def train(self, mode: bool = True) -> "Module":
        """Set training mode."""
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """Set evaluation mode."""
        return self.train(False)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call forward pass."""
        return self.forward(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept module assignment."""
        if isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name})"


class Layer(Module):
    """Single layer in a neural network.

    Layers are the building blocks of modules.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through layer."""
        raise NotImplementedError


class Model(Module):
    """Complete neural network model.

    Composed of layers and modules with training/inference capabilities.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self._layers: List[Layer] = []
        self._compiled = False

    def add_layer(self, layer: Layer) -> "Model":
        """Add a layer to the model."""
        self._layers.append(layer)
        self.add_module(f"layer_{len(self._layers)}", layer)
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers."""
        for layer in self._layers:
            x = layer(x)
        return x

    def compile(self, **kwargs: Any) -> "Model":
        """Compile model for training."""
        self._compiled = True
        return self

    def summary(self) -> str:
        """Get model summary."""
        lines = [f"Model: {self.config.name}", "=" * 50]
        total_params = 0
        for name, param in self.named_parameters():
            size = param.size
            total_params += size
            lines.append(f"{name}: {param.shape} ({size:,} parameters)")
        lines.append("=" * 50)
        lines.append(f"Total parameters: {total_params:,}")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save model to disk."""
        # Placeholder implementation
        pass

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load model from disk."""
        # Placeholder implementation
        return cls()


class Loss:
    """Base class for loss functions."""

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute loss."""
        loss = self.compute(predictions, targets)
        return self._reduce(loss)

    @abstractmethod
    def compute(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute unreduced loss."""
        raise NotImplementedError

    def _reduce(self, loss: Tensor) -> Tensor:
        """Apply reduction."""
        if self.reduction == "mean":
            return Tensor(loss.numpy().mean())
        elif self.reduction == "sum":
            return Tensor(loss.numpy().sum())
        return loss


class Optimizer:
    """Base class for optimization algorithms."""

    def __init__(
        self,
        params: Iterator[Tensor],
        lr: float = 0.001,
        **kwargs: Any,
    ) -> None:
        self.param_groups: List[Dict[str, Any]] = [
            {"params": list(params), "lr": lr, **kwargs}
        ]
        self.state: Dict[int, Dict[str, Any]] = {}

    def step(self) -> None:
        """Perform single optimization step."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Zero out parameter gradients."""
        # Placeholder - real implementation would zero gradients
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return {
            "param_groups": self.param_groups,
            "state": self.state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state."""
        self.param_groups = state_dict["param_groups"]
        self.state = state_dict["state"]


class Scheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.last_epoch = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: Optional[int] = None) -> None:
        """Step the scheduler."""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self.get_lr(i)

    def get_lr(self, group_idx: int) -> float:
        """Get learning rate for parameter group."""
        return self.base_lrs[group_idx]


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class Sample(Generic[T]):
    """Single data sample.

    Contains input features and optional target/label.
    """

    input: T
    target: Optional[T] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        has_target = self.target is not None
        return f"Sample(input={type(self.input).__name__}, target={has_target})"


@dataclass
class Batch(Generic[T]):
    """Batch of samples.

    Batched data for efficient processing.
    """

    inputs: T
    targets: Optional[T] = None
    indices: Optional[List[int]] = None
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        """Batch size."""
        if hasattr(self.inputs, "__len__"):
            return len(self.inputs)
        if hasattr(self.inputs, "shape"):
            return self.inputs.shape[0]  # type: ignore
        return 0

    @property
    def size(self) -> int:
        """Batch size alias."""
        return len(self)

    def __repr__(self) -> str:
        return f"Batch(size={len(self)})"


class Transform(ABC, Generic[T]):
    """Base class for data transformations.

    Transforms can be composed and applied to samples.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def __call__(self, sample: T) -> T:
        """Apply transformation to sample."""
        raise NotImplementedError

    def __add__(self, other: "Transform[T]") -> "Compose[T]":
        """Compose transforms."""
        return Compose([self, other])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class Compose(Transform[T]):
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Transform[T]]) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, sample: T) -> T:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self) -> str:
        transforms_str = ", ".join(t.name for t in self.transforms)
        return f"Compose([{transforms_str}])"


class Dataset(ABC, Generic[T]):
    """Abstract dataset interface.

    Provides access to data samples.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self._transforms: List[Transform[T]] = []

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample[T]:
        """Get sample by index."""
        raise NotImplementedError

    def add_transform(self, transform: Transform[T]) -> "Dataset[T]":
        """Add a transform to be applied to samples."""
        self._transforms.append(transform)
        return self

    def _apply_transforms(self, sample: Sample[T]) -> Sample[T]:
        """Apply all registered transforms."""
        for transform in self._transforms:
            sample = Sample(
                input=transform(sample.input),
                target=transform(sample.target) if sample.target is not None else None,
                metadata=sample.metadata,
            )
        return sample

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Tuple["Dataset[T]", "Dataset[T]", "Dataset[T]"]:
        """Split dataset into train/val/test sets."""
        # Placeholder implementation
        return self, self, self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, size={len(self)})"


class DataLoader(ABC, Generic[T]):
    """Abstract data loader.

    Efficiently loads and batches data.
    """

    def __init__(
        self,
        dataset: Dataset[T],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        collate_fn: Optional[Callable[[List[Sample[T]]], Batch[T]]] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.collate_fn = collate_fn or self._default_collate

    def _default_collate(self, samples: List[Sample[T]]) -> Batch[T]:
        """Default collation function."""
        # Placeholder - real implementation would batch samples properly
        return Batch(
            inputs=[s.input for s in samples],
            targets=[s.target for s in samples if s.target is not None] or None,
        )

    def __len__(self) -> int:
        """Number of batches."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    @abstractmethod
    def __iter__(self) -> Iterator[Batch[T]]:
        """Iterate over batches."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset={self.dataset.name}, "
            f"batch_size={self.batch_size}, "
            f"batches={len(self)})"
        )


# =============================================================================
# Training Types
# =============================================================================


@dataclass
class MetricValue:
    """Metric computation result."""

    name: str
    value: float
    step: int = 0
    epoch: int = 0
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class Metric(ABC):
    """Base class for metrics.

    Metrics track model performance during training.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._values: List[MetricValue] = []
        self._accumulator: List[float] = []

    @abstractmethod
    def compute(self, predictions: Tensor, targets: Tensor) -> float:
        """Compute metric value."""
        raise NotImplementedError

    def update(self, predictions: Tensor, targets: Tensor) -> float:
        """Update metric with new batch."""
        value = self.compute(predictions, targets)
        self._accumulator.append(value)
        return value

    def aggregate(self) -> float:
        """Aggregate accumulated values."""
        if not self._accumulator:
            return 0.0
        return sum(self._accumulator) / len(self._accumulator)

    def reset(self) -> None:
        """Reset metric state."""
        self._accumulator = []

    def log(
        self,
        value: Optional[float] = None,
        step: int = 0,
        epoch: int = 0,
    ) -> MetricValue:
        """Log metric value."""
        v = value if value is not None else self.aggregate()
        metric_value = MetricValue(
            name=self.name,
            value=v,
            step=step,
            epoch=epoch,
        )
        self._values.append(metric_value)
        return metric_value

    def history(self) -> List[MetricValue]:
        """Get logged metric history."""
        return self._values.copy()

    def __call__(self, predictions: Tensor, targets: Tensor) -> float:
        """Compute metric directly."""
        return self.compute(predictions, targets)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class Callback(ABC):
    """Base class for training callbacks.

    Callbacks hook into training lifecycle events.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at start of training."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at end of training."""
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at start of each epoch."""
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int) -> None:
        """Called at end of each epoch."""
        pass

    def on_batch_begin(self, trainer: "Trainer", batch: Batch[Any]) -> None:
        """Called at start of each batch."""
        pass

    def on_batch_end(self, trainer: "Trainer", batch: Batch[Any], loss: Tensor) -> None:
        """Called at end of each batch."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


@dataclass
class History:
    """Training history tracker."""

    metrics: Dict[str, List[MetricValue]] = field(default_factory=dict)
    losses: List[Tuple[int, int, float]] = field(
        default_factory=list
    )  # (epoch, step, loss)

    def log_metric(self, value: MetricValue) -> None:
        """Log a metric value."""
        if value.name not in self.metrics:
            self.metrics[value.name] = []
        self.metrics[value.name].append(value)

    def log_loss(self, epoch: int, step: int, loss: float) -> None:
        """Log a loss value."""
        self.losses.append((epoch, step, loss))

    def get_metric(self, name: str) -> List[MetricValue]:
        """Get history for specific metric."""
        return self.metrics.get(name, [])

    def get_epoch_losses(self, epoch: int) -> List[float]:
        """Get all losses for an epoch."""
        return [loss for e, _, loss in self.losses if e == epoch]

    def best_metric(self, name: str, mode: str = "max") -> Optional[MetricValue]:
        """Get best metric value."""
        values = self.get_metric(name)
        if not values:
            return None
        if mode == "max":
            return max(values, key=lambda v: v.value)
        return min(values, key=lambda v: v.value)


@dataclass
class Checkpoint:
    """Model checkpoint."""

    epoch: int
    step: int
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    def save(self, path: str) -> None:
        """Save checkpoint to disk."""
        import json
        import pickle

        data = {
            "epoch": self.epoch,
            "step": self.step,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }

        # Save metadata as JSON
        with open(f"{path}.json", "w") as f:
            json.dump(data, f, indent=2)

        # Save states as pickle
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(
                {
                    "model_state": self.model_state,
                    "optimizer_state": self.optimizer_state,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        """Load checkpoint from disk."""
        import json
        import pickle

        with open(f"{path}.json", "r") as f:
            metadata = json.load(f)

        with open(f"{path}.pkl", "rb") as f:
            states = pickle.load(f)

        return cls(
            epoch=metadata["epoch"],
            step=metadata["step"],
            model_state=states["model_state"],
            optimizer_state=states["optimizer_state"],
            metrics=metadata["metrics"],
            timestamp=metadata.get("timestamp", 0),
        )


class Trainer:
    """Model trainer.

    Orchestrates training loop with callbacks and metrics.
    """

    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        loss_fn: Loss,
        metrics: Optional[List[Metric]] = None,
        callbacks: Optional[List[Callback]] = None,
        max_epochs: int = 10,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.callbacks = callbacks or []
        self.max_epochs = max_epochs
        self.device = device
        self.history = History()
        self.current_epoch = 0
        self.current_step = 0
        self._should_stop = False

    def add_callback(self, callback: Callback) -> "Trainer":
        """Add a callback."""
        self.callbacks.append(callback)
        return self

    def add_metric(self, metric: Metric) -> "Trainer":
        """Add a metric."""
        self.metrics.append(metric)
        return self

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: Optional[DataLoader[Any]] = None,
    ) -> History:
        """Train the model."""
        self._call_callbacks("on_train_begin")

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self._train_epoch(train_loader)

            if val_loader is not None:
                self._validate_epoch(val_loader)

            if self._should_stop:
                break

        self._call_callbacks("on_train_end")
        return self.history

    def _train_epoch(self, loader: DataLoader[Any]) -> None:
        """Train for one epoch."""
        self._call_callbacks("on_epoch_begin", self.current_epoch)
        self.model.train()

        for batch in loader:
            self._train_step(batch)
            self.current_step += 1

        self._call_callbacks("on_epoch_end", self.current_epoch)

    def _train_step(self, batch: Batch[Any]) -> None:
        """Single training step."""
        self._call_callbacks("on_batch_begin", batch)

        # Forward pass
        predictions = self.model(batch.inputs)
        loss = self.loss_fn(predictions, batch.targets)

        # Backward pass
        self.optimizer.zero_grad()
        # loss.backward()  # Would call backward on real tensor
        self.optimizer.step()

        # Update metrics
        loss_value = loss.numpy() if hasattr(loss, "numpy") else float(loss)
        self.history.log_loss(self.current_epoch, self.current_step, loss_value)

        for metric in self.metrics:
            if batch.targets is not None:
                metric.update(predictions, batch.targets)

        self._call_callbacks("on_batch_end", batch, loss)

    def _validate_epoch(self, loader: DataLoader[Any]) -> None:
        """Validation epoch."""
        self.model.eval()

        for metric in self.metrics:
            metric.reset()

        # Validation loop placeholder
        for batch in loader:
            predictions = self.model(batch.inputs)
            if batch.targets is not None:
                for metric in self.metrics:
                    metric.update(predictions, batch.targets)

        # Log validation metrics
        for metric in self.metrics:
            metric_value = metric.log(
                epoch=self.current_epoch,
                step=self.current_step,
            )
            self.history.log_metric(metric_value)

    def _call_callbacks(self, event: str, *args: Any) -> None:
        """Call callbacks for an event."""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method is not None:
                method(self, *args)

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = Checkpoint(
            epoch=self.current_epoch,
            step=self.current_step,
            model_state={},  # Would extract from model
            optimizer_state=self.optimizer.state_dict(),
            metrics={m.name: m.aggregate() for m in self.metrics},
        )
        checkpoint.save(path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = Checkpoint.load(path)
        self.current_epoch = checkpoint.epoch
        self.current_step = checkpoint.step
        self.optimizer.load_state_dict(checkpoint.optimizer_state)


# =============================================================================
# Inference Types
# =============================================================================


@dataclass
class Input:
    """Inference input."""

    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    preprocess_fn: Optional[Callable[[Any], Any]] = None

    def preprocess(self) -> Any:
        """Apply preprocessing."""
        if self.preprocess_fn is not None:
            return self.preprocess_fn(self.data)
        return self.data


@dataclass
class Output:
    """Inference output."""

    data: Any
    probabilities: Optional[Tensor] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    postprocess_fn: Optional[Callable[[Any], Any]] = None

    def postprocess(self) -> Any:
        """Apply postprocessing."""
        if self.postprocess_fn is not None:
            return self.postprocess_fn(self.data)
        return self.data

    def top_k(self, k: int = 5) -> List[Tuple[int, float]]:
        """Get top-k predictions."""
        if self.probabilities is None:
            return []
        probs = self.probabilities.numpy()
        indices = probs.argsort()[-k:][::-1]
        return [(int(i), float(probs[i])) for i in indices]


class Predictor:
    """Model predictor for inference.

    Handles preprocessing, inference, and postprocessing.
    """

    def __init__(
        self,
        model: Model,
        device: str = "cpu",
        batch_size: int = 1,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.model.to(device)
        self.model.eval()

    def predict(self, input_data: Input) -> Output:
        """Run prediction on single input."""
        processed = input_data.preprocess()

        # Convert to tensor
        if not isinstance(processed, Tensor):
            processed = Tensor(processed)

        # Run inference
        with self._inference_context():
            predictions = self.model(processed)

        # Create output
        return Output(
            data=predictions,
            metadata={"device": self.device},
        )

    def predict_batch(self, inputs: List[Input]) -> List[Output]:
        """Run prediction on batch of inputs."""
        outputs = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            for inp in batch:
                outputs.append(self.predict(inp))
        return outputs

    def _inference_context(self):
        """Context manager for inference."""

        class InferenceContext:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return InferenceContext()

    def __call__(self, input_data: Input) -> Output:
        """Predict shorthand."""
        return self.predict(input_data)


class Pipeline:
    """Inference pipeline.

    Chains multiple processing stages.
    """

    def __init__(self, stages: List[Tuple[str, Callable[[Any], Any]]]) -> None:
        self.stages = stages
        self._outputs: Dict[str, Any] = {}

    def run(self, input_data: Any) -> Output:
        """Run pipeline on input."""
        data = input_data

        for name, stage in self.stages:
            data = stage(data)
            self._outputs[name] = data

        return Output(data=data, metadata=self._outputs)

    def __call__(self, input_data: Any) -> Output:
        """Run shorthand."""
        return self.run(input_data)

    def __repr__(self) -> str:
        stage_names = " -> ".join(name for name, _ in self.stages)
        return f"Pipeline({stage_names})"


# =============================================================================
# Protocol Types
# =============================================================================


@runtime_checkable
class Trainable(Protocol):
    """Protocol for trainable objects.

    Objects that can be trained must implement this protocol.
    """

    def train(self, mode: bool = True) -> Any:
        """Set training mode."""
        ...

    def parameters(self) -> Iterator[Tensor]:
        """Get trainable parameters."""
        ...


@runtime_checkable
class Predictable(Protocol):
    """Protocol for predictable objects.

    Objects that can make predictions must implement this protocol.
    """

    def predict(self, input_data: Any) -> Any:
        """Make prediction."""
        ...

    def eval(self) -> Any:
        """Set evaluation mode."""
        ...


@runtime_checkable
class Saveable(Protocol):
    """Protocol for saveable objects.

    Objects that can be saved to disk.
    """

    def save(self, path: str) -> None:
        """Save object state."""
        ...

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary."""
        ...


@runtime_checkable
class Loadable(Protocol):
    """Protocol for loadable objects.

    Objects that can be loaded from disk.
    """

    @classmethod
    def load(cls, path: str) -> Any:
        """Load object from path."""
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        ...


# =============================================================================
# Utilities
# =============================================================================


class TypeCheckError(TypeError):
    """Error raised when type check fails."""

    pass


def type_check(
    value: Any,
    expected_type: Union[Type[T], Tuple[Type[T], ...]],
    raise_error: bool = True,
) -> bool:
    """Check if value matches expected type.

    Args:
        value: Value to check
        expected_type: Expected type or tuple of types
        raise_error: Whether to raise on mismatch

    Returns:
        True if type matches, False otherwise (if raise_error=False)

    Raises:
        TypeCheckError: If type doesn't match and raise_error=True
    """
    # Handle Union types
    origin = getattr(expected_type, "__origin__", None)
    if origin is Union:
        args = getattr(expected_type, "__args__", ())
        if any(type_check(value, arg, False) for arg in args):
            return True

    # Handle Optional
    elif origin is type(Optional):
        if value is None:
            return True
        args = getattr(expected_type, "__args__", (type(None), T))
        for arg in args:
            if arg is not type(None) and type_check(value, arg, False):
                return True

    # Handle generic types
    elif origin is not None:
        if not isinstance(value, origin):
            if raise_error:
                raise TypeCheckError(f"Expected {origin}, got {type(value)}")
            return False

        # Check generic args if available
        args = getattr(expected_type, "__args__", None)
        if args and hasattr(value, "__args__"):
            value_args = getattr(value, "__args__", ())
            if value_args and not all(
                type_check(v, a, False) for v, a in zip(value_args, args)
            ):
                if raise_error:
                    raise TypeCheckError(
                        f"Generic type mismatch: {value_args} vs {args}"
                    )
                return False
        return True

    # Simple type check
    elif isinstance(expected_type, type):
        if not isinstance(value, expected_type):
            if raise_error:
                raise TypeCheckError(
                    f"Expected {expected_type.__name__}, got {type(value).__name__}"
                )
            return False
        return True

    # Tuple of types
    elif isinstance(expected_type, tuple):
        if not any(type_check(value, t, False) for t in expected_type):
            if raise_error:
                type_names = ", ".join(t.__name__ for t in expected_type)
                raise TypeCheckError(
                    f"Expected one of ({type_names}), got {type(value).__name__}"
                )
            return False
        return True

    return True


def cast_type(value: Any, target_type: Type[T]) -> T:
    """Safely cast value to target type.

    Args:
        value: Value to cast
        target_type: Target type to cast to

    Returns:
        Cast value

    Raises:
        TypeCheckError: If cast is not possible
    """
    # Direct cast for compatible types
    if isinstance(value, target_type):
        return value

    # Handle numeric conversions
    if target_type in (int, float, complex):
        try:
            return target_type(value)  # type: ignore
        except (ValueError, TypeError) as e:
            raise TypeCheckError(f"Cannot cast {value} to {target_type}: {e}")

    # Handle Tensor conversions
    if target_type is Tensor:
        if isinstance(value, (list, tuple, np.ndarray)):
            return Tensor(value)
        if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            return Tensor(value)
        raise TypeCheckError(f"Cannot cast {type(value)} to Tensor")

    # Handle list/tuple conversions
    if target_type in (list, tuple):
        try:
            return target_type(value)  # type: ignore
        except TypeError as e:
            raise TypeCheckError(f"Cannot cast to {target_type}: {e}")

    # Generic cast using constructor
    try:
        return target_type(value)  # type: ignore
    except Exception as e:
        raise TypeCheckError(f"Cannot cast {type(value)} to {target_type}: {e}")


def is_instance(value: Any, classinfo: Union[Type[T], Tuple[Type[T], ...]]) -> bool:
    """Enhanced isinstance check with protocol support.

    Args:
        value: Value to check
        classinfo: Type or tuple of types to check against

    Returns:
        True if value is instance of classinfo
    """
    # Handle protocols
    if isinstance(classinfo, type) and hasattr(classinfo, "__protocol_attrs__"):
        # Check protocol conformance
        for attr in getattr(classinfo, "__protocol_attrs__", set()):
            if not hasattr(value, attr):
                return False
        return True

    # Standard isinstance
    return isinstance(value, classinfo)


# Type aliases for cleaner imports
Check = type_check
Cast = cast_type
Instance = is_instance


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Generic Types
    "T",
    "T_co",
    "T_contra",
    "K",
    "V",
    "Number",
    "Numeric",
    "Scalar",
    "PathLike",
    "JSONValue",
    # Shape and DType
    "DType",
    "Shape",
    # Tensor Types
    "Tensor",
    "TensorInfo",
    "SparseTensor",
    "NamedTensor",
    "QuantizedTensor",
    # Model Types
    "Module",
    "Layer",
    "Model",
    "ModelConfig",
    "Loss",
    "Optimizer",
    "Scheduler",
    # Data Types
    "Sample",
    "Batch",
    "Dataset",
    "DataLoader",
    "Transform",
    "Compose",
    # Training Types
    "Trainer",
    "Callback",
    "Metric",
    "MetricValue",
    "History",
    "Checkpoint",
    # Inference Types
    "Input",
    "Output",
    "Predictor",
    "Pipeline",
    # Protocol Types
    "Trainable",
    "Predictable",
    "Saveable",
    "Loadable",
    # Utilities
    "type_check",
    "Check",
    "cast_type",
    "Cast",
    "is_instance",
    "Instance",
    "TypeCheckError",
]


if __name__ == "__main__":
    # Example usage
    print("Fishstick Types Module")
    print("=" * 50)

    # Tensor example
    tensor = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    print(f"Created: {tensor}")
    print(f"Shape: {tensor.shape}")
    print(f"DType: {tensor.dtype}")

    # Shape operations
    shape1 = Shape(2, 3, 4)
    shape2 = Shape(2, 3, 4)
    print(f"\nShapes compatible: {shape1.compatible_with(shape2)}")

    # Type checking
    print(f"\nType check Tensor: {type_check(tensor, Tensor)}")
    print(f"Type check int: {type_check(42, int)}")

    # Casting
    casted = cast_type([1, 2, 3], Tensor)
    print(f"\nCasted list to Tensor: {casted}")

    print("\nAll type definitions loaded successfully!")
