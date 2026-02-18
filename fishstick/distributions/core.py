"""
Core probability distribution utilities.

Provides base classes for distributions, parameter handling,
and fundamental transformations.
"""

from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import torch
from torch import Tensor
import numpy as np


@dataclass
class DistributionParams:
    """Base container for distribution parameters."""

    pass


@dataclass
class NormalParams(DistributionParams):
    """Parameters for Normal distribution."""

    loc: Tensor
    scale: Tensor


class BaseDistribution:
    """
    Base class for all probability distributions.

    Provides interface for:
    - Sampling
    - Log probability computation
    - Parameter transformation
    - Statistics (mean, variance, entropy)
    """

    def __init__(self, params: DistributionParams):
        self.params = params

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        """Generate samples from the distribution."""
        raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of samples."""
        raise NotImplementedError

    def entropy(self) -> Tensor:
        """Compute differential entropy."""
        raise NotImplementedError

    def mean(self) -> Tensor:
        """Compute the mean."""
        raise NotImplementedError

    def variance(self) -> Tensor:
        """Compute the variance."""
        raise NotImplementedError

    def kl_divergence(self, other: "BaseDistribution") -> Tensor:
        """Compute KL divergence to another distribution."""
        raise NotImplementedError


class TransformedDistribution(BaseDistribution):
    """
    Distribution obtained by applying a transformation to a base distribution.

    If X ~ p(X) and Y = f(X), then:
        p(Y) = p_X(f^{-1}(Y)) * |det J(f^{-1})|
        log p(Y) = log p_X(f^{-1}(Y)) + log|det J(f^{-1})|
    """

    def __init__(
        self,
        base_distribution: BaseDistribution,
        transform: Callable[[Tensor], Tensor],
        inverse_transform: Callable[[Tensor], Tensor],
        log_det_jacobian: Callable[[Tensor], Tensor],
    ):
        super().__init__(base_distribution.params)
        self.base_distribution = base_distribution
        self.transform = transform
        self.inverse_transform = inverse_transform
        self.log_det_jacobian = log_det_jacobian

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        x = self.base_distribution.sample(sample_shape)
        return self.transform(x)

    def log_prob(self, y: Tensor) -> Tensor:
        x = self.inverse_transform(y)
        log_prob = self.base_distribution.log_prob(x)
        return log_prob + self.log_det_jacobian(x)

    def entropy(self) -> Tensor:
        raise NotImplementedError("Entropy not available for transformed distributions")


class MixtureDistribution(BaseDistribution):
    """
    Mixture of distributions.

    p(x) = Σ_k π_k p_k(x), where Σ_k π_k = 1, π_k ≥ 0
    """

    def __init__(
        self,
        components: Tuple[BaseDistribution, ...],
        weights: Optional[Tensor] = None,
    ):
        n_components = len(components)
        if weights is None:
            weights = torch.ones(n_components) / n_components
        self.components = components
        self.weights = weights / weights.sum()
        super().__init__(components[0].params)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        component_idx = torch.multinomial(
            self.weights.expand(*sample_shape, -1), 1
        ).squeeze(-1)

        samples = []
        for i, comp in enumerate(self.components):
            mask = component_idx == i
            if mask.any():
                comp_samples = comp.sample(sample_shape)
                samples.append(
                    torch.where(
                        mask.unsqueeze(-1), comp_samples, torch.zeros_like(comp_samples)
                    )
                )

        return torch.stack(samples).sum(dim=0)

    def log_prob(self, x: Tensor) -> Tensor:
        log_probs = torch.stack([comp.log_prob(x) for comp in self.components], dim=-1)
        log_weights = torch.log(self.weights + 1e-8)
        return torch.logsumexp(log_probs + log_weights, dim=-1)

    def entropy(self) -> Tensor:
        component_entropies = torch.stack([c.entropy() for c in self.components])
        return (self.weights * component_entropies).sum() - torch.sum(
            self.weights * torch.log(self.weights + 1e-8)
        )


class CompositeDistribution(BaseDistribution):
    """
    Distribution representing product of independent distributions.

    p(x₁, x₂, ...) = p₁(x₁) * p₂(x₂) * ...
    """

    def __init__(self, distributions: Tuple[BaseDistribution, ...]):
        self.distributions = distributions
        super().__init__(distributions[0].params)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return torch.cat([d.sample(sample_shape) for d in self.distributions], dim=-1)

    def log_prob(self, x: Tensor) -> Tensor:
        split_sizes = [d.params.loc.shape[-1] for d in self.distributions]
        xs = torch.split(x, split_sizes, dim=-1)
        return sum(d.log_prob(x_i) for d, x_i in zip(self.distributions, xs))

    def marginal(self, idx: int) -> BaseDistribution:
        """Get marginal distribution for component idx."""
        return self.distributions[idx]

    def conditional(self, idx: int, value: Tensor) -> "CompositeDistribution":
        """Get conditional distribution given value for component idx."""
        raise NotImplementedError


def safe_scale(scale: Tensor, min_scale: float = 1e-6) -> Tensor:
    """Ensure scale parameter is positive and finite."""
    return torch.clamp(scale, min=min_scale)


def safe_prob(prob: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Ensure probability is in valid range [0, 1]."""
    return torch.clamp(prob, epsilon, 1 - epsilon)


def standardize(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Standardize x using mean and standard deviation."""
    return (x - mean) / std


def destandardize(z: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Inverse standardization."""
    return z * std + mean
