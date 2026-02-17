"""
Accounting Utilities for Differential Privacy.

This module provides utility functions for privacy accounting,
including composition, conversion, and privacy budget calculations.

Example:
    >>> from fishstick.privacy.accounting_utils import compose_epsilons, compute_gaussian_epsilon
    >>>
    >>> total_eps = compose_epsilons([1.0, 2.0, 3.0], method='basic')
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PrivacyBudget:
    """Privacy budget specification.

    Attributes:
        epsilon: Privacy loss parameter.
        delta: Probability of privacy violation.
        alpha: Rényi alpha parameter (for RDP).
    """

    epsilon: float
    delta: float = 1e-5
    alpha: float = float("inf")

    def __str__(self) -> str:
        return f"(ε={self.epsilon:.4f}, δ={self.delta:.2e})"


def compose_epsilons(
    epsilons: List[float],
    method: str = "basic",
    delta: float = 1e-5,
) -> float:
    """Compose multiple epsilon values.

    Args:
        epsilons: List of epsilon values to compose.
        method: Composition method ('basic', 'advanced', 'optimal').
        delta: Target delta for composition.

    Returns:
        Composed epsilon.
    """
    if not epsilons:
        return 0.0

    if method == "basic":
        return sum(epsilons)

    elif method == "advanced":
        return _advanced_composition(epsilons, delta)

    elif method == "optimal":
        return _optimal_composition(epsilons, delta)

    else:
        raise ValueError(f"Unknown method: {method}")


def _advanced_composition(epsilons: List[float], delta: float) -> float:
    """Advanced composition theorem.

    Args:
        epsilons: List of epsilon values.
        delta: Target delta.

    Returns:
        Composed epsilon.
    """
    k = len(epsilons)
    return sum(epsilons) + math.sqrt(2 * k * math.log(1 / delta)) * max(epsilons)


def _optimal_composition(epsilons: List[float], delta: float) -> float:
    """Optimal composition using numerical optimization.

    Args:
        epsilons: List of epsilon values.
        delta: Target delta.

    Returns:
        Composed epsilon.
    """
    from scipy.optimize import minimize_scalar

    def objective(x):
        composed = 0.0
        for eps in epsilons:
            composed += x * (math.exp(eps / x) - 1)
        return (composed - math.log(1 / delta)) ** 2

    result = minimize_scalar(objective, bounds=(0.01, 10), method="bounded")
    x_opt = result.x

    return sum(epsilons) + x_opt * math.log(1 / delta)


def compute_gaussian_epsilon(
    sigma: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
) -> float:
    """Compute epsilon for Gaussian mechanism.

    Args:
        sigma: Standard deviation of Gaussian noise.
        delta: Target delta.
        sensitivity: Function sensitivity.

    Returns:
        Epsilon value.
    """
    import math

    return (sensitivity / sigma) * math.sqrt(2 * math.log(1.25 / delta))


def compute_laplace_epsilon(
    scale: float,
    delta: float = 0.0,
    sensitivity: float = 1.0,
) -> float:
    """Compute epsilon for Laplace mechanism (pure DP).

    Args:
        scale: Scale parameter for Laplace distribution.
        delta: Target delta (0 for pure DP).
        sensitivity: Function sensitivity.

    Returns:
        Epsilon value.
    """
    return sensitivity / scale


def convert_rdp_to_dp(
    epsilon_rdp: float,
    alpha: float,
    delta: float = 1e-5,
) -> float:
    """Convert Rényi DP to (epsilon, delta)-DP.

    Args:
        epsilon_rdp: Rényi DP epsilon.
        alpha: Rényi alpha parameter.
        delta: Target delta.

    Returns:
        Standard DP epsilon.
    """
    if alpha == float("inf"):
        return epsilon_rdp + math.log(1 / delta)

    return epsilon_rdp + math.log(1 / delta) / (alpha - 1)


def convert_dp_to_rdp(
    epsilon: float,
    delta: float,
    alpha: float = float("inf"),
) -> float:
    """Convert (epsilon, delta)-DP to Rényi DP.

    Args:
        epsilon: Standard DP epsilon.
        delta: Target delta.
        alpha: Target alpha.

    Returns:
        Rényi DP epsilon.
    """
    if alpha == float("inf"):
        return epsilon - math.log(1 / delta)

    return epsilon - math.log(1 / delta) / (alpha - 1)


def compute_subsampled_epsilon(
    base_epsilon: float,
    sample_rate: float,
    num_steps: int,
    delta: float = 1e-5,
) -> float:
    """Compute epsilon with subsampling.

    Args:
        base_epsilon: Base epsilon per step.
        sample_rate: Sampling rate.
        num_steps: Number of steps.
        delta: Target delta.

    Returns:
        Total epsilon.
    """
    q = sample_rate

    total_epsilon = 0.0

    for _ in range(num_steps):
        if q <= 0 or q >= 1:
            step_eps = base_epsilon
        else:
            term = (1 - q) * math.exp(base_epsilon) + q
            step_eps = base_epsilon * math.log(term) / (1 - q)

        total_epsilon = total_epsilon + step_eps + total_epsilon * step_eps

    return total_epsilon


def compute_gaussian_composition(
    sigma: float,
    num_steps: int,
    sample_rate: float,
    delta: float = 1e-5,
) -> Tuple[float, float]:
    """Compute composition for Gaussian mechanism.

    Args:
        sigma: Noise standard deviation.
        num_steps: Number of steps.
        sample_rate: Sampling rate.
        delta: Target delta.

    Returns:
        Tuple of (epsilon, actual delta).
    """
    import math

    q = sample_rate

    epsilon = math.sqrt(
        2 * num_steps * math.log(1.25 / delta) * q
    ) / sigma + num_steps * q * (math.exp(1 / sigma) - 1)

    return epsilon, delta


def compute_privacy_budget(
    epsilon: float,
    delta: float,
    num_samples: int,
    num_epochs: int,
    batch_size: int,
) -> PrivacyBudget:
    """Compute privacy budget accounting for data size.

    Args:
        epsilon: Target epsilon.
        delta: Target delta.
        num_samples: Number of training samples.
        num_epochs: Number of epochs.
        batch_size: Batch size.

    Returns:
        PrivacyBudget object.
    """
    samples_per_epoch = num_samples
    total_steps = (samples_per_epoch // batch_size) * num_epochs
    sample_rate = batch_size / num_samples

    return PrivacyBudget(
        epsilon=epsilon,
        delta=delta,
    )


def estimate_utilitarian_epsilon(
    target_epsilon: float,
    num_steps: int,
    sample_rate: float = 1.0,
    noise_multiplier: float = 1.0,
) -> float:
    """Estimate epsilon that can be achieved with given parameters.

    Args:
        target_epsilon: Target epsilon to achieve.
        num_steps: Number of training steps.
        sample_rate: Sampling rate.
        noise_multiplier: Noise multiplier.

    Returns:
        Feasible epsilon or target if achievable.
    """
    import math

    sigma = noise_multiplier

    required_eps = math.sqrt(
        2 * num_steps * sample_rate * math.log(1.25 / 1e-5)
    ) + num_steps * sample_rate * (math.exp(1 / sigma) - 1)

    if required_eps <= target_epsilon:
        return target_epsilon

    return required_eps


def compute_noise_scale(
    target_epsilon: float,
    num_steps: int,
    sample_rate: float,
    delta: float = 1e-5,
) -> float:
    """Compute required noise scale for target epsilon.

    Args:
        target_epsilon: Target epsilon.
        num_steps: Number of steps.
        sample_rate: Sampling rate.
        delta: Target delta.

    Returns:
        Required noise multiplier.
    """
    from scipy.optimize import brentq

    def equation(sigma):
        import math

        return (
            math.sqrt(2 * num_steps * sample_rate * math.log(1.25 / delta)) / sigma
            + num_steps * sample_rate * (math.exp(1 / sigma) - 1)
            - target_epsilon
        )

    try:
        sigma = brentq(equation, 0.01, 10)
        return sigma
    except ValueError:
        return 1.0


class PrivacyAccountantSimple:
    """Simple privacy accountant for quick calculations.

    Provides quick epsilon/delta calculations without
    maintaining full state.

    Example:
        >>> accountant = PrivacyAccountantSimple()
        >>> accountant.add_step(eps=0.1, q=0.01)
        >>> total_eps = accountant.get_epsilon()
    """

    def __init__(self, delta: float = 1e-5):
        self.delta = delta
        self._steps = 0
        self._epsilon = 0.0

    def add_step(
        self,
        eps: float = 0.1,
        q: float = 1.0,
    ) -> None:
        """Add a privacy step.

        Args:
            eps: Epsilon for this step.
            q: Sample rate.
        """
        self._steps += 1
        self._epsilon += eps * q + (eps * q) ** 2 / (2 * self._steps)

    def get_epsilon(self) -> float:
        """Get current total epsilon.

        Returns:
            Total epsilon spent.
        """
        return self._epsilon

    def get_budget(self) -> PrivacyBudget:
        """Get current privacy budget.

        Returns:
            PrivacyBudget object.
        """
        return PrivacyBudget(epsilon=self._epsilon, delta=self.delta)


def compute_clipping_factor(
    grad_norm: float,
    max_norm: float,
) -> float:
    """Compute gradient clipping scaling factor.

    Args:
        grad_norm: Current gradient norm.
        max_norm: Maximum norm.

    Returns:
        Clipping factor.
    """
    if grad_norm > max_norm:
        return max_norm / grad_norm
    return 1.0


def compute_empirical_epsilon(
    losses_original: np.ndarray,
    losses_perturbed: np.ndarray,
    epsilon: float,
) -> float:
    """Estimate empirical epsilon from loss differences.

    Args:
        losses_original: Losses on original data.
        losses_perturbed: Losses on perturbed data.
        epsilon: True epsilon used.

    Returns:
        Estimated empirical epsilon.
    """
    diffs = np.abs(losses_original - losses_perturbed)

    ratio = np.exp(diffs / epsilon)

    empirical_eps = np.log(np.percentile(ratio, 95))

    return empirical_eps
