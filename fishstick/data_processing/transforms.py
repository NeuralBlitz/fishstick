"""
Data Transformation Pipelines Module for fishstick

Provides composable and configurable transformation pipelines for data
processing with validation, conditional application, and lazy evaluation.

Features:
- Composable transformation chains
- Conditional transformations
- Batch-level transforms
- Lazy evaluation support
- Transform validation
"""

from __future__ import annotations

from typing import (
    Optional,
    Callable,
    List,
    Union,
    Dict,
    Any,
    Tuple,
    Iterator,
    Sequence,
    TypeVar,
    Generic,
    Protocol,
    runtime_checkable,
)
from dataclasses import dataclass, field
from enum import Enum
import threading
import numpy as np
import torch
from torch import Tensor


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_transform = TypeVar("T_transform")


class TransformType(Protocol[T]):
    """Protocol for transformation functions."""

    def __call__(self, data: T) -> T: ...


@runtime_checkable
class TensorTransform(Protocol):
    """Protocol for tensor transformations."""

    def __call__(self, x: Tensor) -> Tensor: ...


@dataclass
class TransformStep:
    """Single step in a transformation pipeline."""

    transform: Callable
    name: str = ""
    enabled: bool = True
    probability: float = 1.0

    def __post_init__(self):
        if not self.name:
            self.name = getattr(self.transform, "__name__", "anonymous")


class TransformPipeline:
    """
    Composable transformation chain.

    Applies multiple transformations sequentially with support for
    conditional application and validation.
    """

    def __init__(
        self,
        transforms: Optional[List[Union[Callable, TransformStep]]] = None,
        name: str = "pipeline",
        validate_output: bool = False,
    ):
        """
        Args:
            transforms: List of transforms or TransformSteps
            name: Pipeline name
            validate_output: Whether to validate outputs
        """
        self.name = name
        self.validate_output = validate_output
        self._transforms: List[TransformStep] = []

        if transforms:
            for t in transforms:
                if isinstance(t, TransformStep):
                    self._transforms.append(t)
                else:
                    self._transforms.append(TransformStep(transform=t))

    def add(
        self, transform: Callable, name: str = "", probability: float = 1.0
    ) -> "TransformPipeline":
        """Add a transform to the pipeline."""
        step = TransformStep(transform=transform, name=name, probability=probability)
        self._transforms.append(step)
        return self

    def remove(self, name: str) -> "TransformPipeline":
        """Remove a transform by name."""
        self._transforms = [t for t in self._transforms if t.name != name]
        return self

    def enable(self, name: str) -> "TransformPipeline":
        """Enable a transform by name."""
        for t in self._transforms:
            if t.name == name:
                t.enabled = True
        return self

    def disable(self, name: str) -> "TransformPipeline":
        """Disable a transform by name."""
        for t in self._transforms:
            if t.name == name:
                t.enabled = False
        return self

    def __call__(self, data: Any) -> Any:
        """Apply all enabled transforms sequentially."""
        result = data

        for step in self._transforms:
            if not step.enabled:
                continue

            if step.probability < 1.0 and np.random.random() > step.probability:
                continue

            try:
                result = step.transform(result)
            except Exception as e:
                raise RuntimeError(f"Transform '{step.name}' failed: {e}")

        return result

    def __len__(self) -> int:
        return len(self._transforms)

    def __getitem__(self, idx: int) -> TransformStep:
        return self._transforms[idx]


class ConditionalTransform:
    """
    Conditional transformation wrapper.

    Applies transform only when condition is met.
    """

    def __init__(
        self,
        transform: Callable[[T], T],
        condition: Callable[[Any], bool],
        else_transform: Optional[Callable[[T], T]] = None,
    ):
        """
        Args:
            transform: Transform to apply if condition is true
            condition: Function that returns True/False
            else_transform: Transform to apply if condition is false
        """
        self.transform = transform
        self.condition = condition
        self.else_transform = else_transform

    def __call__(self, data: Any) -> Any:
        if self.condition(data):
            return self.transform(data)
        elif self.else_transform:
            return self.else_transform(data)
        return data


class BatchTransform:
    """
    Transform applied at batch level.

    Applies transformations to entire batches rather than individual samples.
    """

    def __init__(self, transform: Callable[[Tensor], Tensor]):
        """
        Args:
            transform: Transform that operates on batch tensors
        """
        self.transform = transform

    def __call__(self, batch: Union[Tensor, List[Any]]) -> Union[Tensor, List[Any]]:
        if isinstance(batch, Tensor):
            return self.transform(batch)
        elif isinstance(batch, torch.Tensor):
            return self.transform(batch)
        return batch


class LazyTransform(Callable[[], T]):
    """
    Lazy evaluation wrapper for transforms.

    Defers transform execution until data is actually accessed.
    """

    def __init__(
        self,
        transform_factory: Callable[[], Callable[[Any], Any]],
        *args,
        **kwargs,
    ):
        """
        Args:
            transform_factory: Factory function that creates the transform
            *args: Arguments for factory
            **kwargs: Keyword arguments for factory
        """
        self.transform_factory = transform_factory
        self.args = args
        self.kwargs = kwargs
        self._transform: Optional[Callable] = None
        self._lock = threading.Lock()

    def _get_transform(self) -> Callable:
        if self._transform is None:
            with self._lock:
                if self._transform is None:
                    self._transform = self.transform_factory(*self.args, **self.kwargs)
        return self._transform

    def __call__(self, data: Any) -> Any:
        return self._get_transform()(data)


class TransformValidator:
    """
    Validates transform outputs.

    Ensures transforms produce valid outputs within expected ranges.
    """

    def __init__(
        self,
        checks: Optional[Dict[str, Callable[[Any], bool]]] = None,
        strict: bool = False,
    ):
        """
        Args:
            checks: Dict of check name to validation function
            strict: Whether to raise exception on validation failure
        """
        self.checks = checks or {}
        self.strict = strict
        self._validation_errors: List[str] = []

    def add_check(
        self, name: str, check_fn: Callable[[Any], bool]
    ) -> "TransformValidator":
        """Add a validation check."""
        self.checks[name] = check_fn
        return self

    def validate(self, data: Any, transform_name: str = "unknown") -> bool:
        """Validate data against all checks."""
        self._validation_errors = []

        for name, check in self.checks.items():
            try:
                if not check(data):
                    error = f"Validation failed for '{name}' in '{transform_name}'"
                    self._validation_errors.append(error)
                    if self.strict:
                        raise ValueError(error)
            except Exception as e:
                error = f"Check '{name}' raised exception: {e}"
                self._validation_errors.append(error)
                if self.strict:
                    raise

        return len(self._validation_errors) == 0

    @property
    def errors(self) -> List[str]:
        """Get validation errors."""
        return self._validation_errors.copy()


class ChainedTransform:
    """
    Chains multiple transforms with early exit capability.

    Allows transforms to signal that further processing should be skipped.
    """

    def __init__(self, transforms: List[Callable]):
        """
        Args:
            transforms: List of transforms to chain
        """
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        result = data

        for transform in self.transforms:
            if result is None:
                break
            result = transform(result)

        return result


class AdaptiveTransform:
    """
    Transform that adapts based on input statistics.

    Automatically adjusts parameters based on data characteristics.
    """

    def __init__(
        self,
        base_transform: Callable[[Tensor, ...], Tensor],
        stat_window: int = 100,
    ):
        """
        Args:
            base_transform: The actual transform function
            stat_window: Window size for computing statistics
        """
        self.base_transform = base_transform
        self.stat_window = stat_window
        self._stats: List[Dict[str, float]] = []
        self._lock = threading.Lock()

    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        stats = self._compute_stats(x)

        with self._lock:
            self._stats.append(stats)
            if len(self._stats) > self.stat_window:
                self._stats.pop(0)

        avg_stats = self._get_average_stats()
        return self.base_transform(x, **kwargs, **avg_stats)

    def _compute_stats(self, x: Tensor) -> Dict[str, float]:
        return {
            "mean": x.mean().item(),
            "std": x.std().item(),
            "min": x.min().item(),
            "max": x.max().item(),
        }

    def _get_average_stats(self) -> Dict[str, float]:
        if not self._stats:
            return {}
        keys = self._stats[0].keys()
        return {k: sum(s[k] for s in self._stats) / len(self._stats) for k in keys}


class Compose:
    """
    Compose multiple transforms into a single callable.

    Simplified interface for pipeline creation.
    """

    def __init__(self, transforms: List[Union[Callable, Tuple[Callable, float]]]):
        """
        Args:
            transforms: List of transforms or (transform, probability) tuples
        """
        self.transforms = []
        for t in transforms:
            if isinstance(t, tuple):
                transform, prob = t
                self.transforms.append(
                    TransformStep(transform=transform, probability=prob)
                )
            else:
                self.transforms.append(TransformStep(transform=t))

    def __call__(self, data: Any) -> Any:
        result = data
        for step in self.transforms:
            if step.probability < 1.0 and np.random.random() > step.probability:
                continue
            result = step.transform(result)
        return result

    def __len__(self) -> int:
        return len(self.transforms)


class OneOf:
    """
    Apply exactly one transform from a list with given probabilities.

    Randomly selects and applies a single transform.
    """

    def __init__(
        self,
        transforms: List[Callable],
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            transforms: List of transforms to choose from
            weights: Probabilities for each transform
        """
        self.transforms = transforms
        self.weights = weights or [1.0] * len(transforms)
        self.weights = [w / sum(self.weights) for w in self.weights]

    def __call__(self, data: Any) -> Any:
        idx = np.random.choice(len(self.transforms), p=self.weights)
        return self.transforms[idx](data)


class RandomApply:
    """
    Apply a transform with given probability.
    """

    def __init__(self, transform: Callable, probability: float = 0.5):
        """
        Args:
            transform: Transform to potentially apply
            probability: Probability of applying transform
        """
        self.transform = transform
        self.probability = probability

    def __call__(self, data: Any) -> Any:
        if np.random.random() < self.probability:
            return self.transform(data)
        return data


class LambdaTransform:
    """
    Transform from a lambda/function.
    """

    def __init__(self, fn: Callable[[Any], Any]):
        """
        Args:
            fn: Transformation function
        """
        self.fn = fn

    def __call__(self, data: Any) -> Any:
        return self.fn(data)


class IdentityTransform:
    """
    Identity transform that returns input unchanged.
    """

    def __call__(self, data: T) -> T:
        return data


class NormalizeTransform:
    """Normalization transform for tensors."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        """
        Args:
            mean: Mean values per channel
            std: Standard deviation per channel
        """
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class DenormalizeTransform:
    """Denormalization transform for tensors."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        """
        Args:
            mean: Mean values per channel
            std: Standard deviation per channel
        """
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean


class ResizeTransform:
    """Resize transform for images."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        """
        Args:
            size: Target size (int for square, tuple for hxw)
        """
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            return torch.nn.functional.interpolate(
                x.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False
            ).squeeze(0)
        elif x.dim() == 4:
            return torch.nn.functional.interpolate(
                x, size=self.size, mode="bilinear", align_corners=False
            )
        return x


class FlattenTransform:
    """Flatten transform for tensors."""

    def __init__(self, start_dim: int = 1):
        """
        Args:
            start_dim: Dimension to start flattening from
        """
        self.start_dim = start_dim

    def __call__(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim)


class ToTensorTransform:
    """Convert numpy arrays to tensors."""

    def __call__(self, x: np.ndarray) -> Tensor:
        if isinstance(x, Tensor):
            return x
        return torch.from_numpy(x)


class ToNumpyTransform:
    """Convert tensors to numpy arrays."""

    def __call__(self, x: Union[Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        return x.numpy()


def create_transform_pipeline(
    config: Dict[str, Any],
) -> TransformPipeline:
    """
    Create a TransformPipeline from configuration.

    Args:
        config: Dict with transform configs

    Returns:
        Configured TransformPipeline
    """
    pipeline = TransformPipeline(name=config.get("name", "pipeline"))

    transforms = config.get("transforms", [])
    for t in transforms:
        if isinstance(t, dict):
            transform_fn = t.get("function")
            prob = t.get("probability", 1.0)
            name = t.get("name", "")
            if transform_fn:
                pipeline.add(transform_fn, name=name, probability=prob)

    return pipeline
