"""
Noise Addition Mechanisms for Differential Privacy.

This module provides various noise addition mechanisms for implementing
differential privacy in machine learning, including Gaussian, Laplace,
and exponential mechanisms.

Example:
    >>> from fishstick.privacy import GaussianMechanism, LaplaceMechanism
    >>>
    >>> # Add Gaussian noise to gradients
    >>> noise = GaussianMechanism(sigma=1.0, sensitivity=0.01)
    >>> noisy_grad = noise.add_noise(grad)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

Tensor = torch.Tensor
Module = nn.Module


class NoiseMechanism(ABC):
    """Base class for differential privacy noise mechanisms.

    Attributes:
        epsilon: Privacy budget parameter (higher = less privacy).
        delta: Probability of privacy violation (typically 1e-5 or 1e-6).

    Example:
        >>> mechanism = GaussianMechanism(sigma=1.0)
        >>> noisy_value = mechanism.add_noise(value)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
    ):
        self.epsilon = epsilon
        self.delta = delta

    @abstractmethod
    def add_noise(self, value: Tensor) -> Tensor:
        """Add calibrated noise to a value.

        Args:
            value: Input tensor to add noise to.

        Returns:
            Noisy tensor.
        """
        pass

    @abstractmethod
    def get_privacy_spent(
        self,
        num_steps: int,
        sample_rate: float = 1.0,
    ) -> Tuple[float, float]:
        """Calculate privacy budget spent.

        Args:
            num_steps: Number of optimization steps.
            sample_rate: Sampling rate (fraction of dataset).

        Returns:
            Tuple of (epsilon, delta) spent.
        """
        pass


@dataclass
class NoiseConfig:
    """Configuration for noise addition.

    Attributes:
        mechanism: Type of noise mechanism ('gaussian', 'laplace', 'exponential').
        epsilon: Privacy budget parameter.
        delta: Probability of privacy violation.
        sensitivity: Maximum change in output from single sample.
        sigma: Standard deviation for Gaussian noise.
        scale: Scale parameter for Laplace noise.
    """

    mechanism: str = "gaussian"
    epsilon: float = 1.0
    delta: float = 1e-5
    sensitivity: float = 1.0
    sigma: Optional[float] = None
    scale: Optional[float] = None


class GaussianMechanism(NoiseMechanism):
    """Gaussian mechanism for differential privacy.

    Adds Gaussian noise calibrated to provide (epsilon, delta)-DP.
    More commonly used in practice due to better empirical properties.

    Reference:
        Dwork and Roth, "The Algorithmic Foundations of Differential Privacy".

    Args:
        epsilon: Privacy budget parameter.
        delta: Probability of privacy violation.
        sigma: Standard deviation of Gaussian noise. If None, computed from epsilon.

    Example:
        >>> mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5)
        >>> noisy_grad = mechanism.add_noise(gradient)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sigma: Optional[float] = None,
    ):
        super().__init__(epsilon, delta)

        if sigma is not None:
            self.sigma = sigma
        else:
            self.sigma = self._compute_sigma(epsilon, delta)

    @staticmethod
    def _compute_sigma(epsilon: float, delta: float) -> float:
        """Compute sigma for (epsilon, delta)-DP using Gaussian mechanism.

        Uses the approximate formula: sigma = sqrt(2 * log(1.25/delta)) / epsilon

        Args:
            epsilon: Privacy budget.
            delta: Failure probability.

        Returns:
            Standard deviation of Gaussian noise.
        """
        import math

        return math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def add_noise(self, value: Tensor) -> Tensor:
        """Add Gaussian noise to the input value.

        Args:
            value: Input tensor.

        Returns:
            Noisy tensor with Gaussian noise added.
        """
        noise = torch.randn_like(value) * self.sigma
        return value + noise

    def add_noise_scaled(
        self,
        value: Tensor,
        sensitivity: float,
    ) -> Tensor:
        """Add scaled Gaussian noise based on sensitivity.

        Args:
            value: Input tensor.
            sensitivity: Maximum L2 sensitivity of the function.

        Returns:
            Noisy tensor.
        """
        noise = torch.randn_like(value) * self.sigma * sensitivity
        return value + noise

    def get_privacy_spent(
        self,
        num_steps: int,
        sample_rate: float = 1.0,
    ) -> Tuple[float, float]:
        """Calculate privacy budget spent after multiple steps.

        Args:
            num_steps: Number of optimization steps.
            sample_rate: Sampling rate.

        Returns:
            Tuple of (epsilon, delta) spent.
        """
        import math

        sigma = self.sigma
        delta = self.delta

        epsilon = (
            sample_rate
            * num_steps
            * (
                self.epsilon
                + math.sqrt(2 * math.log(1.25 / delta) * num_steps) * sample_rate
            )
        )

        return epsilon, delta

    def __repr__(self) -> str:
        return f"GaussianMechanism(epsilon={self.epsilon}, delta={self.delta}, sigma={self.sigma:.4f})"


class LaplaceMechanism(NoiseMechanism):
    """Laplace mechanism for differential privacy.

    Adds Laplace noise which provides pure epsilon-DP.
    More interpretable privacy guarantees but often requires more noise.

    Reference:
        Dwork and Roth, "The Algorithmic Foundations of Differential Privacy".

    Args:
        epsilon: Privacy budget parameter (pure DP).
        sensitivity: Maximum L1 sensitivity of the function.
        scale: Scale parameter for Laplace distribution. If None, computed from epsilon.

    Example:
        >>> mechanism = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        >>> noisy_output = mechanism.add_noise(output)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
        scale: Optional[float] = None,
    ):
        super().__init__(epsilon, delta=0.0)
        self.sensitivity = sensitivity

        if scale is not None:
            self.scale = scale
        else:
            self.scale = sensitivity / epsilon

    def add_noise(self, value: Tensor) -> Tensor:
        """Add Laplace noise to the input value.

        Args:
            value: Input tensor.

        Returns:
            Noisy tensor with Laplace noise added.
        """
        noise = torch.distributions.Laplace(0, self.scale).sample(value.shape)
        return value + noise

    def add_noise_scaled(
        self,
        value: Tensor,
        sensitivity: float,
    ) -> Tensor:
        """Add scaled Laplace noise based on sensitivity.

        Args:
            value: Input tensor.
            sensitivity: L1 sensitivity.

        Returns:
            Noisy tensor.
        """
        scale = sensitivity / self.epsilon
        noise = torch.distributions.Laplace(0, scale).sample(value.shape)
        return value + noise

    def get_privacy_spent(
        self,
        num_steps: int,
        sample_rate: float = 1.0,
    ) -> Tuple[float, float]:
        """Calculate privacy budget spent (pure DP).

        Args:
            num_steps: Number of optimization steps.
            sample_rate: Sampling rate.

        Returns:
            Tuple of (epsilon, delta=0) spent.
        """
        epsilon_total = sample_rate * num_steps * self.epsilon
        return epsilon_total, 0.0

    def __repr__(self) -> str:
        return f"LaplaceMechanism(epsilon={self.epsilon}, sensitivity={self.sensitivity}, scale={self.scale:.4f})"


class ExponentialMechanism(NoiseMechanism):
    """Exponential mechanism for differential privacy.

    Used for selecting the best candidate from a set based on a score function.
    Provides (epsilon, 0)-DP (pure DP).

    Reference:
        McSherry and Talwar, "Mechanism Design via Differential Privacy".

    Args:
        epsilon: Privacy budget parameter.
        sensitivity: Sensitivity of the score function.

    Example:
        >>> mechanism = ExponentialMechanism(epsilon=1.0, sensitivity=1.0)
        >>> best_candidate = mechanism.select(scores, candidates)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
    ):
        super().__init__(epsilon, delta=0.0)
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon

    def select(
        self,
        scores: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        """Select a candidate exponentially proportional to score.

        Args:
            scores: Score for each candidate (higher is better).
            candidates: Candidate values to choose from.

        Returns:
            Selected candidate.
        """
        scores = scores / self.scale
        scores = scores - scores.max()

        exp_scores = torch.exp(scores)
        probs = exp_scores / exp_scores.sum()

        idx = torch.multinomial(probs, num_samples=1)
        return candidates[idx]

    def add_noise(self, value: Tensor) -> Tensor:
        """Not applicable for exponential mechanism."""
        raise NotImplementedError("Use select() for exponential mechanism")

    def get_privacy_spent(
        self,
        num_steps: int,
        sample_rate: float = 1.0,
    ) -> Tuple[float, float]:
        """Calculate privacy budget spent.

        Args:
            num_steps: Number of selections.
            sample_rate: Sampling rate.

        Returns:
            Tuple of (epsilon, delta=0) spent.
        """
        epsilon_total = sample_rate * num_steps * self.epsilon
        return epsilon_total, 0.0

    def __repr__(self) -> str:
        return f"ExponentialMechanism(epsilon={self.epsilon}, sensitivity={self.sensitivity})"


class GaussianMixtureMechanism(NoiseMechanism):
    """Gaussian Mixture mechanism for improved utility.

    Combines Gaussian noise with a small amount of Laplace noise
    to achieve better privacy/utility trade-offs.

    Reference:
        Geng et al., "The Staircase Mechanism: Adding Noise"

    Args:
        epsilon: Privacy budget parameter.
        delta: Probability of privacy violation.
        mixture_ratio: Ratio of Laplace to Gaussian noise.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        mixture_ratio: float = 0.1,
    ):
        super().__init__(epsilon, delta)
        self.mixture_ratio = mixture_ratio
        self.gaussian = GaussianMechanism(epsilon, delta)
        self.laplace = LaplaceMechanism(epsilon, sensitivity=1.0)

    def add_noise(self, value: Tensor) -> Tensor:
        """Add mixture of Gaussian and Laplace noise.

        Args:
            value: Input tensor.

        Returns:
            Noisy tensor.
        """
        noise_gaussian = torch.randn_like(value) * self.gaussian.sigma
        noise_laplace = torch.distributions.Laplace(0, self.laplace.scale).sample(
            value.shape
        )

        noise = (
            1 - self.mixture_ratio
        ) * noise_gaussian + self.mixture_ratio * noise_laplace
        return value + noise

    def get_privacy_spent(
        self,
        num_steps: int,
        sample_rate: float = 1.0,
    ) -> Tuple[float, float]:
        """Calculate privacy budget spent.

        Args:
            num_steps: Number of optimization steps.
            sample_rate: Sampling rate.

        Returns:
            Tuple of (epsilon, delta) spent.
        """
        return self.gaussian.get_privacy_spent(num_steps, sample_rate)

    def __repr__(self) -> str:
        return f"GaussianMixtureMechanism(epsilon={self.epsilon}, delta={self.delta})"


def create_noise_mechanism(
    mechanism: str,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    **kwargs,
) -> NoiseMechanism:
    """Factory function to create noise mechanisms.

    Args:
        mechanism: Type of mechanism ('gaussian', 'laplace', 'exponential', 'mixture').
        epsilon: Privacy budget parameter.
        delta: Probability of privacy violation.
        **kwargs: Additional arguments for specific mechanisms.

    Returns:
        Configured noise mechanism.

    Example:
        >>> gaussian = create_noise_mechanism('gaussian', epsilon=1.0)
        >>> laplace = create_noise_mechanism('laplace', epsilon=2.0)
    """
    mechanism = mechanism.lower()

    if mechanism == "gaussian":
        return GaussianMechanism(epsilon=epsilon, delta=delta, **kwargs)
    elif mechanism == "laplace":
        return LaplaceMechanism(epsilon=epsilon, **kwargs)
    elif mechanism == "exponential":
        return ExponentialMechanism(epsilon=epsilon, **kwargs)
    elif mechanism == "mixture":
        return GaussianMixtureMechanism(epsilon=epsilon, delta=delta, **kwargs)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
