"""
Sampling Methods for Differential Privacy.

This module provides various sampling strategies used in DP training,
including Poisson sampling, batch sampling, and stratified sampling.

Example:
    >>> from fishstick.privacy.sampling import PoissonSampler, BatchSampler
    >>>
    >>> sampler = PoissonSampler(sample_rate=0.01)
    >>> batch = sampler.sample(dataset)
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

Tensor = torch.Tensor

T = TypeVar("T")


@dataclass
class SamplingConfig:
    """Configuration for sampling.

    Attributes:
        sample_rate: Fraction of data to sample.
        batch_size: Size of each batch.
        shuffle: Whether to shuffle data.
        drop_last: Whether to drop incomplete last batch.
        replacement: Whether to sample with replacement.
    """

    sample_rate: float = 0.01
    batch_size: int = 256
    shuffle: bool = True
    drop_last: bool = False
    replacement: bool = False


class PoissonSampler(Sampler):
    """Poisson subsampler for differential privacy.

    Each sample is included independently with probability q.
    This provides optimal privacy amplification.

    Reference:
        Balle et al., "Privacy Amplification by Subsampling", ICML 2019.

    Args:
        data_size: Size of dataset.
        sample_rate: Probability of including each sample.
        generator: Random generator for reproducibility.

    Example:
        >>> sampler = PoissonSampler(data_size=10000, sample_rate=0.01)
        >>> for indices in sampler:
        ...     batch = dataset[indices]
    """

    def __init__(
        self,
        data_size: int,
        sample_rate: float,
        generator: Optional[torch.Generator] = None,
    ):
        self.data_size = data_size
        self.sample_rate = sample_rate
        self.generator = generator

    def __iter__(self) -> Iterator[List[int]]:
        """Generate sampled indices.

        Yields:
            List of sampled indices.
        """
        for _ in range(len(self)):
            indices = self._sample_indices()
            if len(indices) > 0:
                yield indices

    def __len__(self) -> int:
        """Return number of batches."""
        return max(1, int(self.data_size * self.sample_rate))

    def _sample_indices(self) -> List[int]:
        """Sample indices with Poisson sampling.

        Returns:
            List of sampled indices.
        """
        indices = []

        rng = random.Random()
        if self.generator is not None:
            rng.seed(self.generator.initial_seed())

        for i in range(self.data_size):
            if rng.random() < self.sample_rate:
                indices.append(i)

        return indices


class BatchSampler(Sampler):
    """Generic batch sampler with sampling control.

    Args:
        data_size: Size of dataset.
        batch_size: Size of each batch.
        sample_rate: Sampling rate (if using subsampling).
        shuffle: Whether to shuffle indices.
        drop_last: Whether to drop incomplete last batch.
        sampler: Optional base sampler to use.

    Example:
        >>> sampler = BatchSampler(data_size=10000, batch_size=256, sample_rate=0.1)
        >>> for batch in sampler:
        ...     print(batch)  # List of indices
    """

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        sample_rate: Optional[float] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        sampler: Optional[Sampler] = None,
    ):
        self.data_size = data_size
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_sampler = sampler

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of indices.

        Yields:
            List of indices forming a batch.
        """
        if self.base_sampler is not None:
            indices = list(self.base_sampler)
        else:
            indices = list(range(self.data_size))

        if self.shuffle:
            random.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(idx)

            if len(batch) == self.batch_size:
                if self.sample_rate is not None:
                    if random.random() < self.sample_rate:
                        yield batch
                    batch = []
                else:
                    yield batch
                    batch = []

        if len(batch) > 0 and not self.drop_last:
            if self.sample_rate is None or random.random() < self.sample_rate:
                yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return self.data_size // self.batch_size
        else:
            return (self.data_size + self.batch_size - 1) // self.batch_size


class StratifiedSampler(Sampler):
    """Stratified sampling maintaining class balance.

    Ensures each batch has proportional representation of classes.

    Args:
        labels: Class labels for stratification.
        batch_size: Size of each batch.
        num_batches: Number of batches to generate.
        shuffle: Whether to shuffle within strata.

    Example:
        >>> labels = torch.randint(0, 10, (10000,))
        >>> sampler = StratifiedSampler(labels, batch_size=256)
        >>> for batch in sampler:
        ...     print(batch)  # Balanced batch
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        num_batches: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.labels = list(labels)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.shuffle = shuffle

        self._stratify()

    def _stratify(self) -> None:
        """Create stratified indices."""
        self.class_indices: Dict[int, List[int]] = {}

        for idx, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        """Generate stratified batches.

        Yields:
            List of indices forming a balanced batch.
        """
        num_classes = len(self.class_indices)
        samples_per_class = self.batch_size // num_classes

        num_batches = self.num_batches
        if num_batches is None:
            min_class_size = min(
                len(indices) for indices in self.class_indices.values()
            )
            num_batches = min_class_size // samples_per_class

        for _ in range(num_batches):
            batch = []

            for label, indices in self.class_indices.items():
                selected = random.sample(indices, min(samples_per_class, len(indices)))
                batch.extend(selected)

            if self.shuffle:
                random.shuffle(batch)

            yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        if self.num_batches is not None:
            return self.num_batches

        num_classes = len(self.class_indices)
        samples_per_class = self.batch_size // num_classes
        min_class_size = min(len(indices) for indices in self.class_indices.values())

        return min_class_size // samples_per_class


class WeightedSampler(Sampler):
    """Weighted sampling with custom weights.

    Samples according to given weights, useful for imbalanced datasets
    or privacy-weighted sampling.

    Args:
        weights: Sampling weights for each data point.
        batch_size: Size of each batch.
        num_samples: Total number of samples to draw.
        replacement: Whether to sample with replacement.

    Example:
        >>> weights = torch.rand(10000)
        >>> sampler = WeightedSampler(weights, batch_size=256)
    """

    def __init__(
        self,
        weights: Sequence[float],
        batch_size: int,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        self.weights = torch.tensor(weights, dtype=torch.float64)
        self.batch_size = batch_size
        self.num_samples = num_samples or len(weights)
        self.replacement = replacement

    def __iter__(self) -> Iterator[List[int]]:
        """Generate weighted batches.

        Yields:
            List of sampled indices.
        """
        num_batches = self.num_samples // self.batch_size

        for _ in range(num_batches):
            indices = torch.multinomial(
                self.weights,
                self.batch_size,
                replacement=self.replacement,
            ).tolist()
            yield indices

    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_samples // self.batch_size


class PrivacyAwareSampler(Sampler):
    """Sampler that accounts for privacy budget.

    Adjusts sampling rate based on remaining privacy budget
    to ensure target epsilon is not exceeded.

    Args:
        data_size: Size of dataset.
        batch_size: Size of each batch.
        target_epsilon: Target privacy budget.
        current_epsilon: Current epsilon spent.
        noise_multiplier: Noise multiplier for DP.

    Example:
        >>> sampler = PrivacyAwareSampler(10000, 256, target_epsilon=8.0)
        >>> for batch in sampler:
        ...     # Training step
        ...     sampler.update_epsilon(new_epsilon)
    """

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        target_epsilon: float,
        current_epsilon: float = 0.0,
        noise_multiplier: float = 1.0,
    ):
        self.data_size = data_size
        self.batch_size = batch_size
        self.target_epsilon = target_epsilon
        self.current_epsilon = current_epsilon
        self.noise_multiplier = noise_multiplier

        self._remaining_eps = target_epsilon - current_epsilon
        self._sample_rate = self._compute_sample_rate()

    def _compute_sample_rate(self) -> float:
        """Compute optimal sample rate.

        Returns:
            Sample rate to stay within budget.
        """
        if self._remaining_eps <= 0:
            return 0.0

        sigma = self.noise_multiplier
        base_eps = sigma * math.sqrt(2 * math.log(1.25 / 1e-5))

        sample_rate = self._remaining_eps / (base_eps * 100)

        return min(1.0, max(0.001, sample_rate))

    def update_epsilon(self, new_epsilon: float) -> None:
        """Update current epsilon and recompute sample rate.

        Args:
            new_epsilon: New current epsilon.
        """
        self.current_epsilon = new_epsilon
        self._remaining_eps = self.target_epsilon - new_epsilon
        self._sample_rate = self._compute_sample_rate()

    @property
    def sample_rate(self) -> float:
        """Get current sample rate."""
        return self._sample_rate

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with adaptive sampling.

        Yields:
            List of sampled indices.
        """
        all_indices = list(range(self.data_size))

        if self.shuffle:
            random.shuffle(all_indices)

        batch = []
        for idx in all_indices:
            if random.random() < self.sample_rate:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if len(batch) > 0:
            yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        expected_samples = int(self.data_size * self.sample_rate)
        return (expected_samples + self.batch_size - 1) // self.batch_size


def create_sampler(
    sampler_type: str,
    data_size: int,
    batch_size: int,
    **kwargs,
) -> Sampler:
    """Factory function to create samplers.

    Args:
        sampler_type: Type of sampler.
        data_size: Size of dataset.
        batch_size: Batch size.
        **kwargs: Additional arguments.

    Returns:
        Configured sampler.

    Example:
        >>> sampler = create_sampler('poisson', 10000, 256, sample_rate=0.01)
    """
    sampler_type = sampler_type.lower()

    if sampler_type == "poisson":
        return PoissonSampler(data_size, kwargs.get("sample_rate", 0.01))
    elif sampler_type == "batch":
        return BatchSampler(
            data_size,
            batch_size,
            sample_rate=kwargs.get("sample_rate"),
            shuffle=kwargs.get("shuffle", True),
            drop_last=kwargs.get("drop_last", False),
        )
    elif sampler_type == "stratified":
        return StratifiedSampler(
            kwargs["labels"],
            batch_size,
            shuffle=kwargs.get("shuffle", True),
        )
    elif sampler_type == "weighted":
        return WeightedSampler(
            kwargs["weights"],
            batch_size,
            replacement=kwargs.get("replacement", True),
        )
    elif sampler_type == "privacy":
        return PrivacyAwareSampler(
            data_size,
            batch_size,
            kwargs.get("target_epsilon", 8.0),
            current_epsilon=kwargs.get("current_epsilon", 0.0),
            noise_multiplier=kwargs.get("noise_multiplier", 1.0),
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def compute_effective_sample_rate(
    base_rate: float,
    num_epochs: int,
    amplification_factor: float = 1.0,
) -> float:
    """Compute effective sample rate over training.

    Args:
        base_rate: Base sampling rate per epoch.
        num_epochs: Number of training epochs.
        amplification_factor: Privacy amplification factor.

    Returns:
        Effective sample rate.
    """
    effective_rate = base_rate * num_epochs / amplification_factor

    return min(1.0, effective_rate)
