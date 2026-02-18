"""Composable augmentation pipelines."""

import torch
import numpy as np
from typing import List, Optional, Callable, Union
from abc import ABC, abstractmethod


class Transform(ABC):
    """Base class for transforms."""

    @abstractmethod
    def __call__(self, data):
        pass


class Compose(Transform):
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self):
        return f"Compose({self.transforms})"


class OneOf(Transform):
    """Select and apply exactly one transform from a list."""

    def __init__(
        self, transforms: List[Callable], probabilities: Optional[List[float]] = None
    ):
        self.transforms = transforms
        if probabilities is None:
            probabilities = [1.0 / len(transforms)] * len(transforms)
        self.probabilities = probabilities

    def __call__(self, data):
        idx = np.random.choice(len(self.transforms), p=self.probabilities)
        return self.transforms[idx](data)


class Sometimes(Transform):
    """Apply transform with given probability."""

    def __init__(self, transform: Callable, probability: float = 0.5):
        self.transform = transform
        self.probability = probability

    def __call__(self, data):
        if torch.rand(1).item() < self.probability:
            return self.transform(data)
        return data


class Repeat(Transform):
    """Apply transform multiple times."""

    def __init__(self, transform: Callable, times: int = 2):
        self.transform = transform
        self.times = times

    def __call__(self, data):
        for _ in range(self.times):
            data = self.transform(data)
        return data


class Replay(Transform):
    """Cache and reuse augmented output."""

    def __init__(self, transform: Callable, replay_probability: float = 0.5):
        self.transform = transform
        self.replay_probability = replay_probability
        self.cached_data = None

    def __call__(self, data):
        if (
            torch.rand(1).item() < self.replay_probability
            and self.cached_data is not None
        ):
            return self.cached_data

        self.cached_data = self.transform(data)
        return self.cached_data


class Lambda(Transform):
    """Apply custom lambda function."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, data):
        return self.fn(data)


class RandomChoice(Transform):
    """Randomly choose from multiple transforms for each call."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data):
        idx = torch.randint(0, len(self.transforms), (1,)).item()
        return self.transforms[idx](data)


class Sequential(Transform):
    """Sequential application with intermediate storage."""

    def __init__(self, transforms: List[Callable], store_intermediate: bool = False):
        self.transforms = transforms
        self.store_intermediate = store_intermediate
        self.intermediate_results = []

    def __call__(self, data):
        self.intermediate_results = [data]
        for transform in self.transforms:
            data = transform(data)
            if self.store_intermediate:
                self.intermediate_results.append(data)
        return data


class ApplyWithCondition(Transform):
    """Apply transform only when condition is met."""

    def __init__(self, transform: Callable, condition_fn: Callable[[any], bool]):
        self.transform = transform
        self.condition_fn = condition_fn

    def __call__(self, data):
        if self.condition_fn(data):
            return self.transform(data)
        return data


class BatchAware:
    """Mixin for transforms that need batch information."""

    def __init__(self):
        self.batch_size = None

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size


def apply_compose(transforms: List[Callable], data, random_order: bool = False):
    """Apply list of transforms to data."""
    if random_order:
        transforms = transforms.copy()
        np.random.shuffle(transforms)

    for transform in transforms:
        data = transform(data)
    return data


def create_pipeline(config: List[dict]) -> Compose:
    """Create pipeline from configuration."""
    transforms = []
    for item in config:
        transform_type = item.pop("type")
        if transform_type == "compose":
            transforms.append(Compose(**item))
        elif transform_type == "oneof":
            transforms.append(OneOf(**item))
        elif transform_type == "sometimes":
            transforms.append(Sometimes(**item))
        elif transform_type == "lambda":
            transforms.append(Lambda(**item))
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    return Compose(transforms)
