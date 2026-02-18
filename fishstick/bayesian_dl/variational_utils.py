"""
Variational Inference Utilities for Bayesian Deep Learning.

Provides utility functions for variational inference including
priors, posteriors, and loss computation.
"""

from typing import Optional, Tuple, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Normal, Laplace, Distribution, Dirichlet, Beta, Gamma


class Prior:
    """Base class for priors in Bayesian neural networks.

    Args:
        sigma: Standard deviation of prior
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability under prior."""
        raise NotImplementedError

    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        """Sample from prior."""
        raise NotImplementedError


class NormalPrior(Prior):
    """Gaussian prior for weights.

    Args:
        mu: Mean of prior
        sigma: Standard deviation of prior
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        super().__init__(sigma)
        self.mu = mu

    def log_prob(self, x: Tensor) -> Tensor:
        dist = Normal(self.mu, self.sigma)
        return dist.log_prob(x).sum()

    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        return torch.randn(shape) * self.sigma + self.mu


class LaplacePrior(Prior):
    """Laplace prior for sparse Bayesian learning.

    Args:
        mu: Mean of prior
        sigma: Scale parameter
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        super().__init__(sigma)
        self.mu = mu

    def log_prob(self, x: Tensor) -> Tensor:
        dist = Laplace(self.mu, self.sigma)
        return dist.log_prob(x).sum()

    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        return torch.distributions.Laplace(self.mu, self.sigma).sample(shape)


class HorseshoePrior(Prior):
    """Horseshoe prior for sparse Bayesian learning.

    Implements the horseshoe prior which encourages sparsity
    while maintaining regularization properties.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__(scale)
        self.scale = scale

    def log_prob(self, x: Tensor) -> Tensor:
        w = torch.abs(x) / self.scale
        log_prob = (
            torch.log1p(w**2)
            + torch.log(torch.tensor(1e-10))
            - 2 * (1 + w**2).sqrt().log()
        )
        return log_prob.sum()

    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        lam = torch.distributions.HalfCauchy(self.scale).sample(shape)
        return torch.randn(shape) * lam


class SpikeAndSlabPrior(Prior):
    """Spike and slab prior for variable selection.

    Args:
        sigma_active: Standard deviation for active variables
        sigma_inactive: Standard deviation for inactive variables
        pi: Prior probability of being active
    """

    def __init__(
        self,
        sigma_active: float = 1.0,
        sigma_inactive: float = 0.01,
        pi: float = 0.5,
    ):
        super().__init__(sigma_active)
        self.sigma_active = sigma_active
        self.sigma_inactive = sigma_inactive
        self.pi = pi

    def log_prob(self, x: Tensor) -> Tensor:
        log_prob_active = Normal(0, self.sigma_active).log_prob(x).sum()
        log_prob_inactive = Normal(0, self.sigma_inactive).log_prob(x).sum()

        return self.pi * log_prob_active + (1 - self.pi) * log_prob_inactive

    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        active = torch.rand(shape) < self.pi
        sample = torch.where(
            active,
            torch.randn(shape) * self.sigma_active,
            torch.randn(shape) * self.sigma_inactive,
        )
        return sample


class VariationalPosterior(Distribution):
    """Base class for variational posterior distributions.

    Args:
        param: Distribution parameters
    """

    def __init__(self, param: Tensor):
        self.param = param

    def sample(self, shape: Optional[Tuple[int, ...]] = None) -> Tensor:
        raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class MeanFieldVariationalPosterior(VariationalPosterior):
    """Mean-field variational posterior.

    Assumes independence between all parameters.

    Args:
        mu: Mean parameters
        log_sigma: Log of standard deviation parameters
    """

    def __init__(self, mu: Tensor, log_sigma: Tensor):
        super().__init__(mu)
        self.mu = mu
        self.log_sigma = log_sigma

    def sample(self, shape: Optional[Tuple[int, ...]] = None) -> Tensor:
        if shape is None:
            shape = self.mu.shape

        sigma = torch.exp(self.log_sigma)
        epsilon = torch.randn_like(sigma)

        return self.mu + sigma * epsilon

    def log_prob(self, x: Tensor) -> Tensor:
        sigma = torch.exp(self.log_sigma)
        dist = Normal(self.mu, sigma)
        return dist.log_prob(x).sum()


class KLDivergence:
    """Computes KL divergence between variational posterior and prior."""

    @staticmethod
    def diagonal_gaussian(
        mu: Tensor,
        log_sigma: Tensor,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ) -> Tensor:
        """KL divergence between diagonal Gaussian and prior Gaussian.

        Args:
            mu: Posterior mean
            log_sigma: Log of posterior standard deviation
            prior_mu: Prior mean
            prior_sigma: Prior standard deviation

        Returns:
            KL divergence
        """
        sigma_sq = torch.exp(2 * log_sigma)
        prior_sigma_sq = prior_sigma**2

        kl = (
            torch.log(prior_sigma)
            - log_sigma
            + (sigma_sq + (mu - prior_mu) ** 2) / (2 * prior_sigma_sq)
            - 0.5
        )

        return kl.sum()

    @staticmethod
    def monte_carlo(
        posterior_sample: Tensor,
        log_q: Tensor,
        log_prior: Tensor,
    ) -> Tensor:
        """Monte Carlo estimate of KL divergence.

        Args:
            posterior_sample: Sample from posterior
            log_q: Log probability under posterior
            log_prior: Log probability under prior

        Returns:
            Monte Carlo estimate of KL
        """
        return (log_q - log_prior).mean()


class RenyiDivergence:
    """Renyi alpha-divergence for variational inference.

    Provides a family of divergences including KL (alpha=1)
    and Hellinger (alpha=0.5).
    """

    @staticmethod
    def alpha_divergence(
        p: Tensor,
        q: Tensor,
        alpha: float = 1.0,
        eps: float = 1e-8,
    ) -> Tensor:
        """Compute Renyi alpha-divergence.

        Args:
            p: Target distribution
            q: Approximation distribution
            alpha: Alpha parameter
            eps: Small constant for numerical stability

        Returns:
            Alpha-divergence
        """
        if abs(alpha - 1.0) < 1e-5:
            return (p * torch.log((p + eps) / (q + eps))).sum()

        p_alpha = p**alpha
        q_alpha = q ** (alpha - 1)

        return (1 / (alpha - 1)) * torch.log((p_alpha * q_alpha).sum() + eps)


class VariationalLoss:
    """Variational loss functions for training BNNs."""

    @staticmethod
    def elbo(
        log_likelihood: Tensor,
        kl_div: Tensor,
        n_data: int,
        beta: float = 1.0,
    ) -> Tensor:
        """Evidence Lower Bound (ELBO) loss.

        Args:
            log_likelihood: Log-likelihood of data
            kl_div: KL divergence from posterior to prior
            n_data: Number of data points
            beta: KL weighting factor

        Returns:
            Negative ELBO
        """
        return -log_likelihood / n_data + beta * kl_div

    @staticmethod
    def iwae(
        log_likelihood: Tensor,
        kl_div: Tensor,
        n_data: int,
        beta: float = 1.0,
        iwae_samples: int = 1,
    ) -> Tensor:
        """Importance Weighted Autoencoder (IWAE) loss.

        Args:
            log_likelihood: Log-likelihood of data
            kl_div: KL divergence from posterior to prior
            n_data: Number of data points
            beta: KL weighting factor
            iwae_samples: Number of IWAE samples

        Returns:
            IWAE loss
        """
        log_weight = log_likelihood - beta * kl_div / n_data

        log_sum_exp = torch.logsumexp(log_weight, dim=0)
        nll = -log_sum_exp + torch.log(torch.tensor(iwae_samples))

        return nll


class PackagedKL:
    """KL divergence that can be packaged as a loss component.

    Useful for integrating with custom training loops.
    """

    def __init__(self, reduction: str = "sum"):
        self.reduction = reduction

    def __call__(
        self,
        posterior_mu: Tensor,
        posterior_log_sigma: Tensor,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ) -> Tensor:
        """Compute KL divergence.

        Args:
            posterior_mu: Posterior mean
            posterior_log_sigma: Log of posterior standard deviation
            prior_mu: Prior mean
            prior_sigma: Prior standard deviation

        Returns:
            KL divergence
        """
        kl = KLDivergence.diagonal_gaussian(
            posterior_mu,
            posterior_log_sigma,
            prior_mu,
            prior_sigma,
        )

        if self.reduction == "mean":
            return kl / posterior_mu.numel()
        elif self.reduction == "sum":
            return kl
        else:
            return kl


class CholeskyVariationalPosterior(VariationalPosterior):
    """Variational posterior with Cholesky covariance structure.

    More expressive than diagonal but still tractable.

    Args:
        mu: Mean parameters
        L: Lower triangular Cholesky factor
    """

    def __init__(self, mu: Tensor, L: Tensor):
        super().__init__(mu)
        self.mu = mu
        self.L = L

    def sample(self, shape: Optional[Tuple[int, ...]] = None) -> Tensor:
        if shape is None:
            shape = self.mu.shape

        epsilon = torch.randn_like(self.mu)

        return self.mu + (self.L @ epsilon.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, x: Tensor) -> Tensor:
        diff = x - self.mu
        log_det = 2 * torch.diagonal(self.L).log().sum()

        mahalanobis = (
            torch.linalg.solve_triangular(
                self.L, diff.unsqueeze(-1), upper=False
            ).squeeze(-1)
            ** 2
        ).sum()

        return -0.5 * (
            diff.shape[0] * torch.log(torch.tensor(2 * 3.14159)) + log_det + mahalanobis
        )


class FlowVariationalPosterior(VariationalPosterior):
    """Variational posterior using normalizing flow.

    Uses a transformation to increase expressiveness of the posterior.

    Args:
        base_dist: Base distribution
        flow: Normalizing flow transformation
    """

    def __init__(self, base_dist: Distribution, flow: nn.Module):
        super().__init__(Tensor())
        self.base_dist = base_dist
        self.flow = flow

    def sample(self, shape: Optional[Tuple[int, ...]] = None) -> Tensor:
        if shape is None:
            shape = (1,)

        z = self.base_dist.sample(shape)
        return self.flow(z)

    def log_prob(self, x: Tensor) -> Tensor:
        z, log_det = self.flow(x, reverse=True)

        log_prob = self.base_dist.log_prob(z)

        return log_prob + log_det


class ScaleMixture:
    """Scale mixture prior combining multiple distributions.

    Args:
        sigma1: First scale parameter
        sigma2: Second scale parameter
        pi: Mixture weight
    """

    def __init__(self, sigma1: float = 1.0, sigma2: float = 0.01, pi: float = 0.5):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.pi = pi

    def log_prob(self, x: Tensor) -> Tensor:
        dist1 = Normal(0, self.sigma1).log_prob(x)
        dist2 = Normal(0, self.sigma2).log_prob(x)

        return torch.logaddexp(
            torch.log(torch.tensor(self.pi)) + dist1,
            torch.log(torch.tensor(1 - self.pi)) + dist2,
        ).sum()


def make_prior(name: str, **kwargs) -> Prior:
    """Factory function to create prior distributions.

    Args:
        name: Name of prior ('normal', 'laplace', 'horseshoe', 'spike_slab')
        **kwargs: Prior-specific arguments

    Returns:
        Prior distribution
    """
    priors = {
        "normal": NormalPrior,
        "laplace": LaplacePrior,
        "horseshoe": HorseshoePrior,
        "spike_slab": SpikeAndSlabPrior,
    }

    if name not in priors:
        raise ValueError(f"Unknown prior: {name}")

    return priors[name](**kwargs)


def make_posterior(name: str, mu: Tensor, sigma: Tensor) -> VariationalPosterior:
    """Factory function to create variational posteriors.

    Args:
        name: Name of posterior ('mean_field', 'cholesky')
        mu: Mean parameters
        sigma: Standard deviation parameters

    Returns:
        Variational posterior
    """
    if name == "mean_field":
        return MeanFieldVariationalPosterior(mu, torch.log(sigma))
    elif name == "cholesky":
        d = mu.shape[0]
        L = torch.eye(d).unsqueeze(0) * sigma
        return CholeskyVariationalPosterior(mu, L)
    else:
        raise ValueError(f"Unknown posterior: {name}")
