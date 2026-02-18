"""
Parameterized probability distributions.

Implements common probability distributions with learnable parameters,
including univariate and multivariate distributions.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor
import numpy as np

from .core import (
    BaseDistribution,
    DistributionParams,
    NormalParams,
    safe_scale,
)


class NormalDistribution(BaseDistribution):
    """
    Normal (Gaussian) distribution: N(μ, σ²).

    PDF: p(x) = (2πσ²)^{-1/2} exp(-(x-μ)²/(2σ²))

    Supports both univariate and multivariate forms.
    """

    def __init__(
        self,
        loc: Tensor,
        scale: Optional[Tensor] = None,
        covariance: Optional[Tensor] = None,
    ):
        if scale is None and covariance is None:
            scale = torch.ones_like(loc)
        elif scale is not None:
            scale = safe_scale(scale)
        elif covariance is not None:
            scale = torch.linalg.cholesky(covariance)

        params = NormalParams(loc=loc, scale=scale)
        super().__init__(params)
        self.loc = loc
        self.scale = scale
        self.covariance = covariance

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return (
            torch.randn(*sample_shape, *self.loc.shape, device=self.loc.device)
            * self.scale
            + self.loc
        )

    def log_prob(self, x: Tensor) -> Tensor:
        if self.covariance is not None:
            return self._log_prob_multivariate(x)
        return self._log_prob_univariate(x)

    def _log_prob_univariate(self, x: Tensor) -> Tensor:
        return (
            -0.5 * torch.log(2 * np.pi * self.scale**2)
            - 0.5 * ((x - self.loc) / self.scale) ** 2
        )

    def _log_prob_multivariate(self, x: Tensor) -> Tensor:
        diff = x - self.loc
        log_det = 2 * torch.sum(torch.log(torch.diag(self.scale)))
        mahal = torch.sum(
            diff @ torch.linalg.inv(self.covariance.unsqueeze(0)) * diff, dim=-1
        )
        return -0.5 * (self.loc.shape[-1] * np.log(2 * np.pi) + log_det + mahal)

    def entropy(self) -> Tensor:
        if self.covariance is not None:
            d = self.loc.shape[-1]
            return 0.5 * (d * (1 + np.log(2 * np.pi)) + torch.logdet(self.covariance))
        return 0.5 * (1 + np.log(2 * np.pi)) + torch.log(self.scale)

    def mean(self) -> Tensor:
        return self.loc

    def variance(self) -> Tensor:
        if self.covariance is not None:
            return torch.diagonal(self.covariance, dim1=-2, dim2=-1)
        return self.scale**2

    def kl_divergence(self, other: "NormalDistribution") -> Tensor:
        """KL(N1 || N2) = 0.5 * (tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - k + ln(|Σ₂|/|Σ₁|))"""
        diff = other.loc - self.loc
        if self.covariance is not None and other.covariance is not None:
            cov_inv = torch.linalg.inv(other.covariance.unsqueeze(0))
            trace_term = torch.einsum(
                "bij,bji->b", cov_inv, self.covariance.unsqueeze(0)
            )
            mahal = torch.einsum("bi,bij,bj->b", diff, cov_inv, diff)
            log_det_ratio = torch.logdet(other.covariance) - torch.logdet(
                self.covariance
            )
            return 0.5 * (trace_term + mahal - self.loc.shape[-1] + log_det_ratio)

        var_ratio = (other.scale**2) / (self.scale**2)
        return 0.5 * (var_ratio + (diff / other.scale) ** 2 - 1 + torch.log(var_ratio))


class ExponentialDistribution(BaseDistribution):
    """
    Exponential distribution: Exp(λ).

    PDF: p(x) = λ exp(-λx) for x ≥ 0

    Mean: 1/λ, Variance: 1/λ²
    """

    def __init__(self, rate: Tensor):
        self.rate = safe_scale(rate, min_scale=1e-8)
        params = DistributionParams()
        super().__init__(params)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return torch.exponential(1.0 / self.rate).expand(*sample_shape, -1)

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.log(self.rate) - self.rate * x

    def entropy(self) -> Tensor:
        return 1 - torch.log(self.rate)

    def mean(self) -> Tensor:
        return 1.0 / self.rate

    def variance(self) -> Tensor:
        return 1.0 / (self.rate**2)


class GammaDistribution(BaseDistribution):
    """
    Gamma distribution: Γ(k, θ) or equivalently Γ(α, β).

    PDF: p(x) = x^{α-1} exp(-x/θ) / (Γ(α)θ^{α})

    Parameters:
        shape (α): shape parameter
        rate (β): rate parameter (1/θ)
    """

    def __init__(
        self,
        shape: Tensor,
        rate: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ):
        if scale is not None:
            rate = 1.0 / scale
        self.shape = safe_scale(shape, min_scale=1e-8)
        self.rate = safe_scale(rate, min_scale=1e-8)
        params = DistributionParams()
        super().__init__(params)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return torch.distributions.Gamma(
            torch.clamp(self.shape, min=1e-6), torch.clamp(self.rate, min=1e-6)
        ).sample(sample_shape)

    def log_prob(self, x: Tensor) -> Tensor:
        return (
            (self.shape - 1) * torch.log(x)
            - x * self.rate
            - torch.lgamma(self.shape)
            + self.shape * torch.log(self.rate)
        )

    def mean(self) -> Tensor:
        return self.shape / self.rate

    def variance(self) -> Tensor:
        return self.shape / (self.rate**2)

    def entropy(self) -> Tensor:
        return (
            self.shape
            - torch.log(self.rate)
            + torch.lgamma(self.shape)
            + (1 - self.shape) * torch.digamma(self.shape)
        )


class BetaDistribution(BaseDistribution):
    """
    Beta distribution: Beta(α, β).

    PDF: p(x) = x^{α-1} (1-x)^{β-1} / B(α, β) for x ∈ [0, 1]

    Used for modeling proportions and probabilities.
    """

    def __init__(self, alpha: Tensor, beta: Tensor):
        self.alpha = safe_scale(alpha, min_scale=1e-8)
        self.beta = safe_scale(beta, min_scale=1e-8)
        params = DistributionParams()
        super().__init__(params)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return torch.distributions.Beta(
            torch.clamp(self.alpha, min=1e-6), torch.clamp(self.beta, min=1e-6)
        ).sample(sample_shape)

    def log_prob(self, x: Tensor) -> Tensor:
        return (
            (self.alpha - 1) * torch.log(x)
            + (self.beta - 1) * torch.log(1 - x)
            - torch.betaln(self.alpha, self.beta)
        )

    def mean(self) -> Tensor:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> Tensor:
        ab_sum = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab_sum**2 * (ab_sum + 1))


class LaplaceDistribution(BaseDistribution):
    """
    Laplace distribution (double exponential): Laplace(μ, b).

    PDF: p(x) = (1/(2b)) exp(-|x-μ|/b)

    Heavier tails than Normal distribution.
    """

    def __init__(self, loc: Tensor, scale: Tensor):
        self.loc = loc
        self.scale = safe_scale(scale, min_scale=1e-8)
        params = DistributionParams()
        super().__init__(params)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        u = torch.rand(*sample_shape, *self.loc.shape, device=self.loc.device) - 0.5
        return self.loc - self.scale * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))

    def log_prob(self, x: Tensor) -> Tensor:
        return -torch.log(2 * self.scale) - torch.abs(x - self.loc) / self.scale

    def mean(self) -> Tensor:
        return self.loc

    def variance(self) -> Tensor:
        return 2 * self.scale**2

    def entropy(self) -> Tensor:
        return 1 + torch.log(2 * self.scale)


class LogNormalDistribution(BaseDistribution):
    """
    Log-normal distribution: log(X) ~ N(μ, σ²).

    PDF: p(x) = (1/(xσ√(2π))) exp(-(ln x - μ)²/(2σ²)) for x > 0

    Commonly used for modeling positive random variables with right skew.
    """

    def __init__(self, loc: Tensor, scale: Tensor):
        self.loc = loc
        self.scale = safe_scale(scale, min_scale=1e-8)
        params = DistributionParams()
        super().__init__(params)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return torch.exp(
            torch.randn(*sample_shape, *self.loc.shape, device=self.loc.device)
            * self.scale
            + self.loc
        )

    def log_prob(self, x: Tensor) -> Tensor:
        return (
            -torch.log(x)
            - 0.5 * torch.log(2 * np.pi * self.scale**2)
            - ((torch.log(x) - self.loc) / self.scale) ** 2 / 2
        )

    def mean(self) -> Tensor:
        return torch.exp(self.loc + 0.5 * self.scale**2)

    def variance(self) -> Tensor:
        return (torch.exp(self.scale**2) - 1) * torch.exp(2 * self.loc + self.scale**2)


class CategoricalDistribution(BaseDistribution):
    """
    Categorical distribution over K categories.

    PMF: P(X=k) = π_k, where Σ π_k = 1

    Equivalent to multinomial with n=1.
    """

    def __init__(self, probs: Tensor):
        self.probs = torch.softmax(probs, dim=-1)
        params = DistributionParams()
        super().__init__(params)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return torch.multinomial(self.probs.expand(*sample_shape, -1), 1).squeeze(-1)

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.gather(torch.log(self.probs + 1e-8), -1, x.unsqueeze(-1)).squeeze(
            -1
        )

    def entropy(self) -> Tensor:
        return -torch.sum(self.probs * torch.log(self.probs + 1e-8), dim=-1)

    def mean(self) -> Tensor:
        return torch.argmax(self.probs, dim=-1)


class MultivariateNormalDistribution(BaseDistribution):
    """
    Multivariate Normal distribution: N(μ, Σ).

    PDF: p(x) = |2πΣ|^{-1/2} exp(-0.5 (x-μ)ᵀΣ^{-1}(x-μ))

    Supports full covariance, diagonal covariance, and low-rank approximations.
    """

    def __init__(
        self,
        loc: Tensor,
        covariance: Optional[Tensor] = None,
        scale_tril: Optional[Tensor] = None,
        precision: Optional[Tensor] = None,
    ):
        self.loc = loc
        self.covariance = covariance
        self.scale_tril = scale_tril
        self.precision = precision

        if scale_tril is not None:
            self._scale_tril = scale_tril
        elif covariance is not None:
            self._scale_tril = torch.linalg.cholesky(covariance)
        else:
            self._scale_tril = torch.eye(loc.shape[-1], device=loc.device)

        params = DistributionParams()
        super().__init__(params)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        eps = torch.randn(*sample_shape, *self.loc.shape, device=self.loc.device)
        return self.loc.unsqueeze(0) + torch.einsum(
            "...ij,...j->...i", self._scale_tril, eps
        )

    def log_prob(self, x: Tensor) -> Tensor:
        diff = x - self.loc
        log_det = 2 * torch.sum(
            torch.log(torch.diagonal(self._scale_tril, dim1=-2, dim2=-1)), dim=-1
        )
        mahalanobis = torch.sum(
            torch.linalg.solve_triangular(
                self._scale_tril, diff.unsqueeze(-1), upper=False
            )
            ** 2,
            dim=(-2, -1),
        )
        d = self.loc.shape[-1]
        return -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis)

    def mean(self) -> Tensor:
        return self.loc

    def variance(self) -> Tensor:
        return torch.sum(self._scale_tril**2, dim=-2)

    def entropy(self) -> Tensor:
        d = self.loc.shape[-1]
        log_det = 2 * torch.sum(
            torch.log(torch.diagonal(self._scale_tril, dim1=-2, dim2=-1)), dim=-1
        )
        return 0.5 * (d * (1 + np.log(2 * np.pi)) + log_det)


class DirichletDistribution(BaseDistribution):
    """
    Dirichlet distribution: Dirichlet(α).

    PDF: p(p) = (1/B(α)) ∏ p_i^{α_i-1}, where p ∈ simplex

    Conjugate prior for Categorical and Multinomial distributions.
    """

    def __init__(self, concentration: Tensor):
        self.concentration = safe_scale(concentration, min_scale=1e-8)
        params = DistributionParams()
        super().__init__(params)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return torch.distributions.Dirichlet(
            torch.clamp(self.concentration, min=1e-6)
        ).sample(sample_shape)

    def log_prob(self, x: Tensor) -> Tensor:
        return (
            torch.lgamma(self.concentration.sum())
            - torch.lgamma(self.concentration).sum()
            + ((self.concentration - 1) * torch.log(x)).sum(dim=-1)
        )

    def mean(self) -> Tensor:
        return self.concentration / self.concentration.sum()

    def variance(self) -> Tensor:
        alpha_0 = self.concentration.sum()
        alpha_i = self.concentration
        return alpha_i * (alpha_0 - alpha_i) / (alpha_0**2 * (alpha_0 + 1))


class WishartDistribution(BaseDistribution):
    """
    Wishart distribution: Wishart(ν, S).

    PDF: p(X) = |X|^{(ν-p-1)/2} exp(-tr(S^{-1}X)/2) / (2^{νp/2} |S|^{ν/2} Γ_p(ν/2))

    Distribution over symmetric positive definite matrices.
    Conjugate prior for precision matrix of Gaussian.
    """

    def __init__(self, df: Tensor, scale: Tensor):
        self.df = df
        self.scale = scale
        self.p = scale.shape[-1]
        params = DistributionParams()
        super().__init__(params)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return torch.distributions.Wishart(
            self.df, torch.linalg.cholesky(self.scale)
        ).sample(sample_shape)

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.distributions.Wishart(
            self.df, torch.linalg.cholesky(self.scale)
        ).log_prob(x)


class VonMisesFisherDistribution(BaseDistribution):
    """
    Von Mises-Fisher distribution on the unit sphere S^{p-1}.

    PDF: p(x) = C_p(κ) exp(κ μᵀx), where |x| = 1

    Parameters:
        mean: μ (unit vector) - mean direction
        concentration: κ ≥ 0 - concentration parameter
    """

    def __init__(self, mean: Tensor, concentration: Tensor):
        self.mean = mean / torch.norm(mean, dim=-1, keepdim=True)
        self.concentration = safe_scale(concentration, min_scale=1e-8)
        self.p = mean.shape[-1]
        self._normalizer = self._compute_normalizer()
        params = DistributionParams()
        super().__init__(params)

    def _compute_normalizer(self) -> Tensor:
        """Compute normalization constant C_p(κ)."""
        if self.p == 3:
            return 4 * np.pi * torch.sinh(self.concentration) / self.concentration
        else:
            bessel_i0 = torch.i0(self.concentration)
            bessel_i1 = torch.i1(self.concentration)
            return (self.concentration / (2 * np.pi * bessel_i0)) * (
                (self.p // 2 - 1) * torch.log(self.concentration)
                - torch.log(bessel_i1)
                - np.log(2)
            )

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        """Sample using rejection sampling or approximation."""
        m = (self.p - 1) / 2
        x = torch.randn(*sample_shape, self.p, device=self.mean.device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x

    def log_prob(self, x: Tensor) -> Tensor:
        return self.concentration * torch.sum(self.mean * x, dim=-1) + torch.log(
            self._normalizer
        )

    def mean(self) -> Tensor:
        return self.mean
