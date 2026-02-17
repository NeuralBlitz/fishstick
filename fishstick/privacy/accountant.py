"""
Privacy Budget Accountant for Differential Privacy.

This module implements privacy budget accounting using Rényi Differential Privacy (RDP)
and provides utilities for tracking cumulative privacy loss across multiple steps.

Example:
    >>> from fishstick.privacy import RDPAccountant, PrivacyAccountant
    >>>
    >>> accountant = RDPAccountant(epsilon=8.0, delta=1e-5)
    >>> for step in range(100):
    ...     accountant.step(sample_rate=0.01)
    >>> eps, delta = accountant.get_privacy_spent()
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


@dataclass
class PrivacyAccount:
    """Record of privacy budget spent.

    Attributes:
        epsilon: Total epsilon spent.
        delta: Total delta spent.
        num_steps: Number of privacy-preserving steps.
        alpha: Rényi alpha parameter (for RDP accounting).
        rho: Privacy loss random variable parameter.
    """

    epsilon: float = 0.0
    delta: float = 0.0
    num_steps: int = 0
    alpha: float = float("inf")
    rho: float = 0.0


class PrivacyAccountant(ABC):
    """Abstract base class for privacy accountants.

    Example:
        >>> class MyAccountant(PrivacyAccountant):
        ...     def step(self, sample_rate: float): pass
        ...     def get_privacy_spent(self) -> Tuple[float, float]: pass
    """

    @abstractmethod
    def step(self, sample_rate: float = 1.0) -> None:
        """Record a single optimization step.

        Args:
            sample_rate: Sampling rate (fraction of dataset).
        """
        pass

    @abstractmethod
    def get_privacy_spent(
        self,
        target_delta: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Get total privacy budget spent.

        Args:
            target_delta: Target delta for computing epsilon.

        Returns:
            Tuple of (epsilon, delta) spent.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the accountant state."""
        pass


class RDPAccountant(PrivacyAccountant):
    """Rényi Differential Privacy (RDP) accountant.

    Uses RDP composition theorem for tighter privacy guarantees.
    Converts RDP to (epsilon, delta)-DP using the conversion formula.

    Reference:
        Mironov, "Rényi Differential Privacy".

    Args:
        epsilon: Target epsilon for early stopping.
        delta: Target delta for (epsilon, delta)-DP.
        alpha: Initial Rényi alpha parameter.

    Example:
        >>> accountant = RDPAccountant(epsilon=8.0, delta=1e-5)
        >>> for _ in range(100):
        ...     accountant.step(sample_rate=0.01)
        >>> eps, delta = accountant.get_privacy_spent()
    """

    def __init__(
        self,
        epsilon: float = float("inf"),
        delta: float = 1e-5,
        alpha: float = float("inf"),
    ):
        self.target_epsilon = epsilon
        self.target_delta = delta
        self.alpha = alpha

        self._num_steps = 0
        self._sample_rate = 0.0
        self._rho = 0.0

        self._alphas: List[float] = []
        self._rhos: List[float] = []

    def step(
        self,
        sample_rate: float = 1.0,
        noise_multiplier: float = 1.0,
        alphas: Optional[List[float]] = None,
    ) -> None:
        """Record a single DP-SGD step.

        Args:
            sample_rate: Sampling rate (fraction of dataset).
            noise_multiplier: Ratio of noise standard deviation to clipping norm.
            alphas: List of alpha values for RDP accounting.
        """
        if alphas is None:
            alphas = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

        self._sample_rate = sample_rate
        self._num_steps += 1

        for alpha in alphas:
            if alpha == 1:
                continue

            if alpha == float("inf"):
                rho = self._compute_rho_inf(sample_rate, noise_multiplier)
            else:
                rho = self._compute_rho(alpha, sample_rate, noise_multiplier)

            if len(self._alphas) < len(alphas):
                self._rhos.append(rho)
            else:
                self._rhos[len(self._alphas) % len(alphas)] += rho

            if len(self._alphas) < len(alphas):
                self._alphas.append(alpha)

        self._rho = sum(self._rhos)

    def _compute_rho(self, alpha: float, q: float, sigma: float) -> float:
        """Compute rho parameter for RDP at given alpha.

        Args:
            alpha: Rényi parameter.
            q: Sampling rate.
            sigma: Noise multiplier.

        Returns:
            Rho value.
        """
        if alpha == float("inf"):
            return self._compute_rho_inf(q, sigma)

        if alpha == 1:
            return float("inf")

        q_sigma = q / sigma

        if q_sigma == 0:
            return 0.0

        term1 = (alpha - 1) * math.log(1 - q_sigma**2)
        term2 = alpha * math.log(1 + q_sigma**2 / (alpha - 1))

        rho = (term1 + term2) / (2 * sigma**2)
        return max(0, rho)

    def _compute_rho_inf(self, q: float, sigma: float) -> float:
        """Compute rho for infinite alpha.

        Args:
            q: Sampling rate.
            sigma: Noise multiplier.

        Returns:
            Rho value.
        """
        q_sigma = q / sigma

        if q_sigma == 0:
            return 0.0

        return (q_sigma**2) / (2 * sigma**2)

    def get_privacy_spent(
        self,
        target_delta: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Get total privacy budget spent.

        Args:
            target_delta: Target delta (uses default if None).

        Returns:
            Tuple of (epsilon, delta) spent.
        """
        delta = target_delta if target_delta is not None else self.target_delta

        if self._rho == 0:
            return 0.0, 0.0

        epsilon = self._rho + math.log(1 / delta) / (self.alpha - 1)
        epsilon = max(0, epsilon)

        return epsilon, delta

    def get_privacy_spent_per_alpha(
        self,
    ) -> Dict[float, Tuple[float, float]]:
        """Get privacy spent for each alpha value.

        Returns:
            Dictionary mapping alpha to (epsilon, delta).
        """
        result = {}
        delta = self.target_delta

        for i, alpha in enumerate(self._alphas):
            rho = self._rhos[i] if i < len(self._rhos) else 0

            if alpha == float("inf"):
                eps = rho + math.log(1 / delta)
            else:
                eps = rho + math.log(1 / delta) / (alpha - 1)

            result[alpha] = (max(0, eps), delta)

        return result

    def reset(self) -> None:
        """Reset the accountant state."""
        self._num_steps = 0
        self._sample_rate = 0.0
        self._rho = 0.0
        self._alphas = []
        self._rhos = []

    @property
    def num_steps(self) -> int:
        """Get number of steps recorded."""
        return self._num_steps

    def __repr__(self) -> str:
        eps, delta = self.get_privacy_spent()
        return (
            f"RDPAccountant(steps={self._num_steps}, epsilon={eps:.4f}, delta={delta})"
        )


class BasicAccountant(PrivacyAccountant):
    """Basic privacy accountant using simple composition.

    Uses basic composition theorem (epsilon accumulates linearly).
    Provides worst-case guarantees but often looser than RDP.

    Args:
        epsilon: Target epsilon.
        delta: Target delta.

    Example:
        >>> accountant = BasicAccountant(epsilon=8.0, delta=1e-5)
        >>> accountant.step(sample_rate=0.01)
        >>> eps, delta = accountant.get_privacy_spent()
    """

    def __init__(
        self,
        epsilon: float = float("inf"),
        delta: float = 1e-5,
    ):
        self.target_epsilon = epsilon
        self.target_delta = delta
        self._epsilon_spent = 0.0
        self._num_steps = 0
        self._sample_rate = 0.0

    def step(self, sample_rate: float = 1.0) -> None:
        """Record a single step.

        Args:
            sample_rate: Sampling rate.
        """
        self._sample_rate = sample_rate
        self._num_steps += 1
        self._epsilon_spent += sample_rate * self.target_epsilon

    def get_privacy_spent(
        self,
        target_delta: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Get privacy spent.

        Args:
            target_delta: Target delta.

        Returns:
            Tuple of (epsilon, delta) spent.
        """
        delta = target_delta if target_delta is not None else self.target_delta
        return self._epsilon_spent, delta

    def reset(self) -> None:
        """Reset the accountant."""
        self._epsilon_spent = 0.0
        self._num_steps = 0
        self._sample_rate = 0.0


class GaussianAccountant(PrivacyAccountant):
    """Privacy accountant for Gaussian mechanism.

    Uses advanced composition for Gaussian noise.

    Args:
        epsilon: Target epsilon.
        delta: Target delta.
        sigma: Noise standard deviation.

    Example:
        >>> accountant = GaussianAccountant(epsilon=8.0, delta=1e-5, sigma=1.0)
        >>> accountant.step(sample_rate=0.01)
    """

    def __init__(
        self,
        epsilon: float = float("inf"),
        delta: float = 1e-5,
        sigma: float = 1.0,
    ):
        self.target_epsilon = epsilon
        self.target_delta = delta
        self.sigma = sigma
        self._num_steps = 0
        self._sample_rate = 0.0

    def step(self, sample_rate: float = 1.0) -> None:
        """Record a single step.

        Args:
            sample_rate: Sampling rate.
        """
        self._sample_rate = sample_rate
        self._num_steps += 1

    def get_privacy_spent(
        self,
        target_delta: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Get privacy spent using Gaussian composition.

        Args:
            target_delta: Target delta.

        Returns:
            Tuple of (epsilon, delta) spent.
        """
        import math

        delta = target_delta if target_delta is not None else self.target_delta
        q = self._sample_rate
        sigma = self.sigma

        if self._num_steps == 0:
            return 0.0, 0.0

        epsilon = math.sqrt(
            2 * self._num_steps * math.log(1.25 / delta) * q
        ) + self._num_steps * q * (math.exp(self.target_epsilon / sigma) - 1)

        return epsilon, delta

    def reset(self) -> None:
        """Reset the accountant."""
        self._num_steps = 0
        self._sample_rate = 0.0


class PrivacyBudgetTracker:
    """High-level privacy budget tracking with milestones.

    Provides utilities for tracking privacy budget and checking
    if target privacy has been achieved.

    Args:
        target_epsilon: Target epsilon.
        target_delta: Target delta.
        accountant: Privacy accountant to use.

    Example:
        >>> tracker = PrivacyBudgetTracker(8.0, 1e-5)
        >>> for step in range(100):
        ...     tracker.step(sample_rate=0.01)
        ...     if tracker.is_target_reached():
        ...         print("Privacy target achieved!")
    """

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float = 1e-5,
        accountant: Optional[PrivacyAccountant] = None,
    ):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        if accountant is None:
            self.accountant = RDPAccountant(target_epsilon, target_delta)
        else:
            self.accountant = accountant

        self._history: List[PrivacyAccount] = []

    def step(
        self,
        sample_rate: float = 1.0,
        noise_multiplier: float = 1.0,
    ) -> None:
        """Record a step and update history.

        Args:
            sample_rate: Sampling rate.
            noise_multiplier: Noise multiplier.
        """
        self.accountant.step(sample_rate, noise_multiplier)

        eps, delta = self.accountant.get_privacy_spent()
        account = PrivacyAccount(
            epsilon=eps,
            delta=delta,
            num_steps=self.accountant.num_steps,
        )
        self._history.append(account)

    def is_target_reached(self) -> bool:
        """Check if target privacy has been reached.

        Returns:
            True if target epsilon is within budget.
        """
        eps, _ = self.accountant.get_privacy_spent()
        return eps <= self.target_epsilon

    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget.

        Returns:
            Tuple of (remaining epsilon, remaining delta).
        """
        eps, delta = self.accountant.get_privacy_spent()
        remaining_eps = max(0, self.target_epsilon - eps)
        return remaining_eps, delta

    def get_history(self) -> List[PrivacyAccount]:
        """Get privacy spending history.

        Returns:
            List of PrivacyAccount records.
        """
        return self._history.copy()

    def summary(self) -> str:
        """Get summary of privacy budget.

        Returns:
            Formatted summary string.
        """
        eps, delta = self.accountant.get_privacy_spent()
        return (
            f"Privacy Budget Summary:\n"
            f"  Steps: {self.accountant.num_steps}\n"
            f"  Epsilon spent: {eps:.4f} / {self.target_epsilon}\n"
            f"  Delta: {delta:.2e}\n"
            f"  Target reached: {self.is_target_reached()}"
        )


def create_accountant(
    accountant_type: str = "rdp",
    epsilon: float = float("inf"),
    delta: float = 1e-5,
    **kwargs,
) -> PrivacyAccountant:
    """Factory function to create privacy accountants.

    Args:
        accountant_type: Type of accountant ('rdp', 'basic', 'gaussian').
        epsilon: Target epsilon.
        delta: Target delta.
        **kwargs: Additional arguments.

    Returns:
        Configured privacy accountant.

    Example:
        >>> accountant = create_accountant('rdp', epsilon=8.0, delta=1e-5)
    """
    accountant_type = accountant_type.lower()

    if accountant_type == "rdp":
        return RDPAccountant(epsilon, delta, **kwargs)
    elif accountant_type == "basic":
        return BasicAccountant(epsilon, delta)
    elif accountant_type == "gaussian":
        return GaussianAccountant(epsilon, delta, **kwargs)
    else:
        raise ValueError(f"Unknown accountant type: {accountant_type}")
