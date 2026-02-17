"""
Privacy Amplification Techniques.

This module provides methods for amplifying privacy guarantees through
subsampling, shuffling, and other techniques.

Example:
    >>> from fishstick.privacy import amplify_privacy, Subsampler
    >>>
    >>> # Amplify privacy through subsampling
    >>> amplifier = Subsampler(sample_rate=0.01)
    >>> new_epsilon = amplifier.amplify(epsilon, sample_rate)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor

Tensor = torch.Tensor


@dataclass
class AmplificationResult:
    """Result of privacy amplification.

    Attributes:
        original_epsilon: Epsilon before amplification.
        amplified_epsilon: Epsilon after amplification.
        amplification_factor: Ratio of amplified to original.
        method: Method used for amplification.
    """

    original_epsilon: float
    amplified_epsilon: float
    amplification_factor: float
    method: str


class PrivacyAmplifier(ABC):
    """Abstract base class for privacy amplification methods.

    Example:
        >>> class MyAmplifier(PrivacyAmplifier):
        ...     def amplify(self, epsilon: float, q: float) -> float: pass
    """

    @abstractmethod
    def amplify(
        self,
        epsilon: float,
        sample_rate: float,
    ) -> float:
        """Amplify privacy guarantee.

        Args:
            epsilon: Original epsilon.
            sample_rate: Sampling rate.

        Returns:
            Amplified epsilon.
        """
        pass


class SubsampleAmplifier(PrivacyAmplifier):
    """Privacy amplification by subsampling.

    When only a random fraction of data is used in each training step,
    privacy guarantees are amplified.

    Reference:
        Balle et al., "Privacy Amplification by Subsampling", ICML 2019.

    Args:
        method: Amplification method ('simple', 'strong', 'optimal').

    Example:
        >>> amplifier = SubsampleAmplifier(method='strong')
        >>> new_eps = amplifier.amplify(epsilon=1.0, sample_rate=0.01)
    """

    def __init__(
        self,
        method: str = "strong",
    ):
        self.method = method

    def amplify(
        self,
        epsilon: float,
        sample_rate: float,
    ) -> float:
        """Amplify privacy via subsampling.

        Args:
            epsilon: Original epsilon.
            sample_rate: Sampling rate (fraction of data used).

        Returns:
            Amplified epsilon.
        """
        if self.method == "simple":
            return self._simple_amplify(epsilon, sample_rate)
        elif self.method == "strong":
            return self._strong_amplify(epsilon, sample_rate)
        elif self.method == "optimal":
            return self._optimal_amplify(epsilon, sample_rate)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _simple_amplify(self, epsilon: float, q: float) -> float:
        """Simple amplification (linear).

        Args:
            epsilon: Original epsilon.
            q: Sample rate.

        Returns:
            Amplified epsilon.
        """
        return epsilon * q

    def _strong_amplify(self, epsilon: float, q: float) -> float:
        """Strong amplification (logarithmic composition).

        Args:
            epsilon: Original epsilon.
            q: Sample rate.

        Returns:
            Amplified epsilon.
        """
        if q >= 1.0 or q <= 0:
            return epsilon

        term = (1 - q) * math.exp(epsilon) + q
        term = math.log(term) / (1 - q)

        return epsilon * term

    def _optimal_amplify(self, epsilon: float, q: float) -> float:
        """Optimal amplification for Poisson sampling.

        Args:
            epsilon: Original epsilon.
            q: Sample rate.

        Returns:
            Amplified epsilon.
        """
        if q >= 1.0 or q <= 0:
            return epsilon

        delta = 1e-6

        def compose(eps1, eps2):
            return eps1 + eps2 + eps1 * eps2

        def RDP_to_DP(eps_alpha, alpha, delta):
            return eps_alpha + math.log(1 / delta) / (alpha - 1)

        min_eps = float("inf")
        for alpha in range(2, 100):
            rho = epsilon * (alpha - 1) / alpha
            rho_new = (1 - q) * (math.exp(rho / (alpha - 1)) - 1)

            if rho_new < 0:
                continue

            eps_alpha = alpha * math.log(1 + rho_new / (1 - q))
            eps = RDP_to_DP(eps_alpha, alpha, delta)
            min_eps = min(min_eps, eps)

        return min_eps if min_eps < float("inf") else epsilon


class ShuffleAmplifier(PrivacyAmplifier):
    """Privacy amplification by shuffling.

    When data is randomly shuffled before processing, additional
    privacy is obtained through the shuffle privacy model.

    Reference:
        Cheu et al., "Distributed Differential Privacy", ICALP 2019.

    Args:
        num_users: Number of users contributing data.
        shuffle_fraction: Fraction of data shuffled.

    Example:
        >>> amplifier = ShuffleAmplifier(num_users=1000, shuffle_fraction=0.1)
        >>> new_eps = amplifier.amplify(epsilon=1.0, sample_rate=0.01)
    """

    def __init__(
        self,
        num_users: int = 1,
        shuffle_fraction: float = 1.0,
    ):
        self.num_users = num_users
        self.shuffle_fraction = shuffle_fraction

    def amplify(
        self,
        epsilon: float,
        sample_rate: float,
    ) -> float:
        """Amplify privacy via shuffling.

        Args:
            epsilon: Original epsilon.
            sample_rate: Sampling rate.

        Returns:
            Amplified epsilon.
        """
        if self.shuffle_fraction <= 0:
            return epsilon

        effective_users = max(1, self.num_users * self.shuffle_fraction)

        shuffle_epsilon = math.log(1 + effective_users * (math.exp(epsilon) - 1))

        return min(epsilon, shuffle_epsilon)


class LocalAmplifier(PrivacyAmplifier):
    """Privacy amplification in local DP setting.

    Provides amplification guarantees for local differential privacy
    where each user adds their own noise.

    Args:
        local_epsilon: Local privacy parameter.
        num_users: Number of users.
        aggregation_method: How users are aggregated ('sum', 'mean').

    Example:
        >>> amplifier = LocalAmplifier(local_epsilon=2.0, num_users=100)
        >>> new_eps = amplifier.amplify(epsilon=0.0, sample_rate=0.1)
    """

    def __init__(
        self,
        local_epsilon: float = 1.0,
        num_users: int = 1,
        aggregation_method: str = "mean",
    ):
        self.local_epsilon = local_epsilon
        self.num_users = num_users
        self.aggregation_method = aggregation_method

    def amplify(
        self,
        epsilon: float,
        sample_rate: float,
    ) -> float:
        """Amplify privacy in local DP setting.

        Args:
            epsilon: Original epsilon (typically 0 for local).
            sample_rate: Fraction of users selected.

        Returns:
            Amplified epsilon.
        """
        if self.aggregation_method == "sum":
            scale = self.num_users
        else:
            scale = 1.0

        amplified_epsilon = self.local_epsilon / scale

        if sample_rate < 1.0:
            amplified_epsilon *= sample_rate

        return amplified_epsilon


class ComposeAmplifier(PrivacyAmplifier):
    """Combine multiple amplification techniques.

    Args:
        amplifiers: List of amplifiers to compose.
        composition_method: How to combine ('max', 'sum').

    Example:
        >>> amplifier = ComposeAmplifier([SubsampleAmplifier(), ShuffleAmplifier()])
        >>> new_eps = amplifier.amplify(epsilon=1.0, sample_rate=0.01)
    """

    def __init__(
        self,
        amplifiers: List[PrivacyAmplifier],
        composition_method: str = "max",
    ):
        self.amplifiers = amplifiers
        self.composition_method = composition_method

    def amplify(
        self,
        epsilon: float,
        sample_rate: float,
    ) -> float:
        """Apply composed amplification.

        Args:
            epsilon: Original epsilon.
            sample_rate: Sampling rate.

        Returns:
            Amplified epsilon.
        """
        if self.composition_method == "max":
            result = epsilon
            for amp in self.amplifiers:
                result = min(result, amp.amplify(epsilon, sample_rate))
            return result
        elif self.composition_method == "sum":
            return sum(amp.amplify(epsilon, sample_rate) for amp in self.amplifiers)
        else:
            raise ValueError(f"Unknown composition method: {self.composition_method}")


def amplify_privacy(
    epsilon: float,
    sample_rate: float,
    method: str = "subsample",
    **kwargs,
) -> Tuple[float, AmplificationResult]:
    """Convenience function for privacy amplification.

    Args:
        epsilon: Original epsilon.
        sample_rate: Sampling rate.
        method: Amplification method.
        **kwargs: Additional method-specific arguments.

    Returns:
        Tuple of (amplified epsilon, result object).

    Example:
        >>> new_eps, result = amplify_privacy(1.0, 0.01, method='subsample')
    """
    if method == "subsample":
        amplifier = SubsampleAmplifier(**kwargs)
    elif method == "shuffle":
        amplifier = ShuffleAmplifier(**kwargs)
    elif method == "local":
        amplifier = LocalAmplifier(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    amplified_epsilon = amplifier.amplify(epsilon, sample_rate)
    factor = amplified_epsilon / epsilon if epsilon > 0 else float("inf")

    result = AmplificationResult(
        original_epsilon=epsilon,
        amplified_epsilon=amplified_epsilon,
        amplification_factor=factor,
        method=method,
    )

    return amplified_epsilon, result


def compute_subsampled_epsilon(
    epsilon: float,
    q: float,
    num_steps: int,
    delta: float = 1e-5,
) -> float:
    """Compute epsilon with subsampling composition.

    Args:
        epsilon: Base epsilon per step.
        q: Sample rate.
        num_steps: Number of steps.
        delta: Target delta.

    Returns:
        Total epsilon after composition.
    """

    def compose(eps1, eps2):
        return eps1 + eps2 + eps1 * eps2

    total_eps = 0.0
    for _ in range(num_steps):
        subsampled_eps = SubsampleAmplifier("strong").amplify(epsilon, q)
        total_eps = compose(total_eps, subsampled_eps)

    return total_eps


def get_amplification_factor(
    epsilon: float,
    sample_rate: float,
    method: str = "subsample",
) -> float:
    """Get privacy amplification factor.

    Args:
        epsilon: Original epsilon.
        sample_rate: Sampling rate.
        method: Amplification method.

    Returns:
        Amplification factor (how much epsilon is reduced).
    """
    amplified_eps, _ = amplify_privacy(epsilon, sample_rate, method)
    return epsilon / amplified_eps if amplified_eps > 0 else float("inf")
