"""
Distribution fitting algorithms.

Implements methods for fitting probability distributions to data,
including MLE, method of moments, and Bayesian fitting.
"""

from typing import Optional, Tuple, List, Callable, Dict, Any
from dataclasses import dataclass
import torch
from torch import Tensor
import numpy as np
from scipy import optimize
from scipy import stats as scipy_stats


@dataclass
class FitResult:
    """Container for distribution fitting results."""

    params: Dict[str, Tensor]
    log_likelihood: float
    AIC: float
    BIC: float
    n_samples: int
    distribution_name: str


class DistributionFitter:
    """
    Base class for distribution fitting.

    Provides common interface for fitting distributions
    via MLE, method of moments, or Bayesian inference.
    """

    def __init__(self, distribution_name: str):
        self.distribution_name = distribution_name
        self.fit_result: Optional[FitResult] = None

    def fit_mle(
        self,
        data: Tensor,
        **kwargs,
    ) -> FitResult:
        """Maximum Likelihood Estimation."""
        raise NotImplementedError

    def fit_moments(
        self,
        data: Tensor,
        **kwargs,
    ) -> FitResult:
        """Method of Moments estimation."""
        raise NotImplementedError

    def fit_bayesian(
        self,
        data: Tensor,
        prior_params: Dict[str, Tensor],
        n_samples: int = 1000,
    ) -> FitResult:
        """Bayesian posterior estimation."""
        raise NotImplementedError


class NormalFitter(DistributionFitter):
    """Fitter for Normal distribution."""

    def __init__(self):
        super().__init__("Normal")

    def fit_mle(self, data: Tensor) -> FitResult:
        """Fit Normal via MLE."""
        mu_hat = data.mean()
        sigma_hat = data.std(unbiased=True)

        n = len(data)
        log_likelihood = (
            -n / 2 * np.log(2 * np.pi)
            - n * np.log(sigma_hat)
            - 0.5 * ((data - mu_hat) ** 2).sum() / (sigma_hat**2)
        )

        k = 2
        AIC = 2 * k - 2 * log_likelihood
        BIC = k * np.log(n) - 2 * log_likelihood

        return FitResult(
            params={"loc": mu_hat, "scale": sigma_hat},
            log_likelihood=log_likelihood,
            AIC=AIC,
            BIC=BIC,
            n_samples=n,
            distribution_name=self.distribution_name,
        )

    def fit_moments(self, data: Tensor) -> FitResult:
        """Fit Normal via method of moments (same as MLE for Normal)."""
        return self.fit_mle(data)


class ExponentialFitter(DistributionFitter):
    """Fitter for Exponential distribution."""

    def __init__(self):
        super().__init__("Exponential")

    def fit_mle(self, data: Tensor) -> FitResult:
        """Fit Exponential via MLE: λ_hat = 1/mean(x)."""
        rate_hat = 1.0 / data.mean()

        n = len(data)
        log_likelihood = n * np.log(rate_hat) - rate_hat * data.sum()

        k = 1
        AIC = 2 * k - 2 * log_likelihood
        BIC = k * np.log(n) - 2 * log_likelihood

        return FitResult(
            params={"rate": rate_hat},
            log_likelihood=log_likelihood,
            AIC=AIC,
            BIC=BIC,
            n_samples=n,
            distribution_name=self.distribution_name,
        )


class GammaFitter(DistributionFitter):
    """Fitter for Gamma distribution."""

    def __init__(self):
        super().__init__("Gamma")

    def fit_mle(self, data: Tensor) -> FitResult:
        """Fit Gamma via numerical MLE."""
        data_np = data.cpu().numpy()

        def neg_log_likelihood(params):
            shape, rate = params
            if shape <= 0 or rate <= 0:
                return 1e10
            return -scipy_stats.gamma.logpdf(data_np, a=shape, scale=1 / rate).sum()

        result = optimize.minimize(
            neg_log_likelihood,
            x0=[data.mean() ** 2 / data.var(), data.mean() / data.var()],
            method="L-BFGS-B",
            bounds=[(1e-6, 100), (1e-6, 100)],
        )

        shape_hat, rate_hat = result.x

        n = len(data)
        k = 2
        log_likelihood = -result.fun
        AIC = 2 * k - 2 * log_likelihood
        BIC = k * np.log(n) - 2 * log_likelihood

        return FitResult(
            params={"shape": torch.tensor(shape_hat), "rate": torch.tensor(rate_hat)},
            log_likelihood=log_likelihood,
            AIC=AIC,
            BIC=BIC,
            n_samples=n,
            distribution_name=self.distribution_name,
        )

    def fit_moments(self, data: Tensor) -> FitResult:
        """Fit Gamma via method of moments."""
        mean = data.mean()
        var = data.var(unbiased=True)

        shape_hat = mean**2 / var
        rate_hat = mean / var

        n = len(data)
        log_likelihood = scipy_stats.gamma.logpdf(
            data.cpu().numpy(), a=shape_hat, scale=1 / rate_hat
        ).sum()

        k = 2
        AIC = 2 * k - 2 * log_likelihood
        BIC = k * np.log(n) - 2 * log_likelihood

        return FitResult(
            params={"shape": torch.tensor(shape_hat), "rate": torch.tensor(rate_hat)},
            log_likelihood=log_likelihood,
            AIC=AIC,
            BIC=BIC,
            n_samples=n,
            distribution_name=self.distribution_name,
        )


class BetaFitter(DistributionFitter):
    """Fitter for Beta distribution."""

    def __init__(self):
        super().__init__("Beta")

    def fit_mle(self, data: Tensor) -> FitResult:
        """Fit Beta via numerical MLE."""
        data_np = np.clip(data.cpu().numpy(), 1e-6, 1 - 1e-6)

        def neg_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return 1e10
            return -scipy_stats.beta.logpdf(data_np, alpha, beta).sum()

        mean_data = data_np.mean()
        var_data = data_np.var()

        common = mean_data * (1 - mean_data) / var_data - 1
        alpha_hat = mean_data * common
        beta_hat = (1 - mean_data) * common

        result = optimize.minimize(
            neg_log_likelihood,
            x0=[alpha_hat, beta_hat],
            method="L-BFGS-B",
            bounds=[(1e-6, 100), (1e-6, 100)],
        )

        alpha_hat, beta_hat = result.x

        n = len(data)
        log_likelihood = -result.fun
        k = 2
        AIC = 2 * k - 2 * log_likelihood
        BIC = k * np.log(n) - 2 * log_likelihood

        return FitResult(
            params={"alpha": torch.tensor(alpha_hat), "beta": torch.tensor(beta_hat)},
            log_likelihood=log_likelihood,
            AIC=AIC,
            BIC=BIC,
            n_samples=n,
            distribution_name=self.distribution_name,
        )


class PoissonFitter(DistributionFitter):
    """Fitter for Poisson distribution."""

    def __init__(self):
        super().__init__("Poisson")

    def fit_mle(self, data: Tensor) -> FitResult:
        """Fit Poisson via MLE: λ_hat = mean(x)."""
        lambda_hat = data.mean()

        n = len(data)
        log_likelihood = scipy_stats.poisson.logpmf(
            data.cpu().numpy().astype(int), lambda_hat
        ).sum()

        k = 1
        AIC = 2 * k - 2 * log_likelihood
        BIC = k * np.log(n) - 2 * log_likelihood

        return FitResult(
            params={"lambda": lambda_hat},
            log_likelihood=log_likelihood,
            AIC=AIC,
            BIC=BIC,
            n_samples=n,
            distribution_name=self.distribution_name,
        )


class WeibullFitter(DistributionFitter):
    """Fitter for Weibull distribution."""

    def __init__(self):
        super().__init__("Weibull")

    def fit_mle(self, data: Tensor) -> FitResult:
        """Fit Weibull via numerical MLE."""
        data_np = data.cpu().numpy()

        def neg_log_likelihood(params):
            shape, scale = params
            if shape <= 0 or scale <= 0:
                return 1e10
            return -scipy_stats.weibull_min.logpdf(data_np, c=shape, scale=scale).sum()

        result = optimize.minimize(
            neg_log_likelihood,
            x0=[1.0, data.mean()],
            method="L-BFGS-B",
            bounds=[(1e-6, 100), (1e-6, 100)],
        )

        shape_hat, scale_hat = result.x

        n = len(data)
        log_likelihood = -result.fun
        k = 2
        AIC = 2 * k - 2 * log_likelihood
        BIC = k * np.log(n) - 2 * log_likelihood

        return FitResult(
            params={"shape": torch.tensor(shape_hat), "scale": torch.tensor(scale_hat)},
            log_likelihood=log_likelihood,
            AIC=AIC,
            BIC=BIC,
            n_samples=n,
            distribution_name=self.distribution_name,
        )


class LogNormalFitter(DistributionFitter):
    """Fitter for Log-Normal distribution."""

    def __init__(self):
        super().__init__("LogNormal")

    def fit_mle(self, data: Tensor) -> FitResult:
        """Fit Log-Normal via MLE."""
        log_data = torch.log(data + 1e-10)
        mu_hat = log_data.mean()
        sigma_hat = log_data.std(unbiased=True)

        n = len(data)
        log_likelihood = (
            -n / 2 * np.log(2 * np.pi)
            - n * np.log(sigma_hat)
            - ((log_data - mu_hat) ** 2).sum() / (2 * sigma_hat**2)
            - torch.log(data + 1e-10).sum()
        )

        k = 2
        AIC = 2 * k - 2 * log_likelihood
        BIC = k * np.log(n) - 2 * log_likelihood

        return FitResult(
            params={"loc": mu_hat, "scale": sigma_hat},
            log_likelihood=log_likelihood,
            AIC=AIC,
            BIC=BIC,
            n_samples=n,
            distribution_name=self.distribution_name,
        )


class MultivariateNormalFitter(DistributionFitter):
    """Fitter for Multivariate Normal distribution."""

    def __init__(self):
        super().__init__("MultivariateNormal")

    def fit_mle(self, data: Tensor) -> FitResult:
        """Fit Multivariate Normal via MLE."""
        mean_hat = data.mean(dim=0)

        centered = data - mean_hat.unsqueeze(0)
        cov_hat = (centered.T @ centered) / len(data)

        n = len(data)
        d = data.shape[-1]

        log_likelihood = -n / 2 * (
            d * np.log(2 * np.pi) + torch.logdet(cov_hat)
        ) - 0.5 * torch.sum(centered @ torch.linalg.inv(cov_hat) * centered)

        k = d + d * (d + 1) // 2
        AIC = 2 * k - 2 * log_likelihood.item()
        BIC = k * np.log(n) - 2 * log_likelihood.item()

        return FitResult(
            params={"loc": mean_hat, "covariance": cov_hat},
            log_likelihood=log_likelihood.item(),
            AIC=AIC,
            BIC=BIC,
            n_samples=n,
            distribution_name=self.distribution_name,
        )


class GaussianMixtureFitter:
    """
    Fitter for Gaussian Mixture Model.

    Uses EM algorithm for fitting.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.weights: Optional[Tensor] = None
        self.means: Optional[Tensor] = None
        self.covariances: Optional[Tensor] = None

    def fit(
        self,
        data: Tensor,
        n_iter: int = 100,
        tol: float = 1e-4,
    ) -> FitResult:
        """
        Fit GMM using EM algorithm.

        Args:
            data: [n_samples, n_features] data tensor
            n_iter: Maximum number of EM iterations
            tol: Convergence tolerance
        """
        n_samples, n_features = data.shape

        self.weights = torch.ones(self.n_components) / self.n_components
        self.means = data[torch.randperm(n_samples)[: self.n_components]]
        self.covariances = torch.stack(
            [torch.eye(n_features) for _ in range(self.n_components)]
        )

        log_likelihood_prev = float("-inf")

        for _ in range(n_iter):
            responsibilities = self._e_step(data)
            self._m_step(data, responsibilities)

            log_likelihood = self._compute_log_likelihood(data)

            if abs(log_likelihood - log_likelihood_prev) < tol:
                break

            log_likelihood_prev = log_likelihood

        n = n_samples
        k = (
            self.n_components * (n_features + n_features * (n_features + 1) // 2 + 1)
            - 1
        )
        AIC = 2 * k - 2 * log_likelihood
        BIC = k * np.log(n) - 2 * log_likelihood

        return FitResult(
            params={
                "weights": self.weights,
                "means": self.means,
                "covariances": self.covariances,
            },
            log_likelihood=log_likelihood,
            AIC=AIC,
            BIC=BIC,
            n_samples=n,
            distribution_name=f"GaussianMixture({self.n_components})",
        )

    def _e_step(self, data: Tensor) -> Tensor:
        """E-step: compute responsibilities."""
        log_probs = torch.zeros(len(data), self.n_components)

        for k in range(self.n_components):
            diff = data - self.means[k]
            log_probs[:, k] = (
                torch.log(self.weights[k] + 1e-10)
                - 0.5 * (n_features := data.shape[-1]) * np.log(2 * np.pi)
                - 0.5 * torch.logdet(self.covariances[k])
                - 0.5
                * torch.sum(diff @ torch.linalg.inv(self.covariances[k]) * diff, dim=-1)
            )

        responsibilities = torch.softmax(log_probs, dim=1)
        return responsibilities

    def _m_step(self, data: Tensor, responsibilities: Tensor) -> None:
        """M-step: update parameters."""
        Nk = responsibilities.sum(dim=0)

        self.weights = Nk / len(data)

        self.means = (responsibilities.unsqueeze(-1) * data.unsqueeze(1)).sum(
            dim=0
        ) / Nk.unsqueeze(-1)

        self.covariances = torch.zeros_like(
            self.means.unsqueeze(0).expand(self.n_components, -1, -1)
        )
        for k in range(self.n_components):
            diff = data - self.means[k]
            weighted_diff = responsibilities[:, k].unsqueeze(-1) * diff
            self.covariances[k] = (weighted_diff.T @ diff) / Nk[k]
            self.covariances[k] += 1e-6 * torch.eye(data.shape[-1])

    def _compute_log_likelihood(self, data: Tensor) -> float:
        """Compute total log-likelihood."""
        log_probs = torch.zeros(len(data), self.n_components)

        for k in range(self.n_components):
            diff = data - self.means[k]
            log_probs[:, k] = (
                torch.log(self.weights[k] + 1e-10)
                - 0.5 * (n_features := data.shape[-1]) * np.log(2 * np.pi)
                - 0.5 * torch.logdet(self.covariances[k])
                - 0.5
                * torch.sum(diff @ torch.linalg.inv(self.covariances[k]) * diff, dim=-1)
            )

        return torch.logsumexp(log_probs, dim=1).sum().item()


class KernelDensityEstimator:
    """
    Kernel Density Estimation (non-parametric).

    Provides non-parametric density estimation using kernels.
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.data: Optional[Tensor] = None

    def fit(self, data: Tensor) -> "KernelDensityEstimator":
        """Fit KDE to data."""
        self.data = data.clone()
        return self

    def score_samples(self, x: Tensor) -> Tensor:
        """Compute log-density at points x."""
        if self.data is None:
            raise ValueError("KDE not fitted. Call fit() first.")

        n = len(self.data)
        d = x.shape[-1]

        if self.kernel == "gaussian":
            diff = x.unsqueeze(1) - self.data.unsqueeze(0)
            log_kernel = -0.5 * (diff / self.bandwidth) ** 2 - 0.5 * np.log(2 * np.pi)
            log_density = (
                torch.logsumexp(log_kernel.sum(dim=-1), dim=1)
                - np.log(n)
                - d * np.log(self.bandwidth)
            )

        return log_density

    def sample(self, n_samples: int) -> Tensor:
        """Sample from fitted KDE."""
        if self.data is None:
            raise ValueError("KDE not fitted. Call fit() first.")

        indices = torch.randint(0, len(self.data), (n_samples,))
        samples = self.data[indices]

        noise = torch.randn_like(samples) * self.bandwidth
        return samples + noise


def fit_distribution(
    data: Tensor,
    distribution: str,
    method: str = "mle",
) -> FitResult:
    """
    Fit a distribution to data.

    Args:
        data: Sample data
        distribution: Name of distribution ('normal', 'gamma', 'beta', etc.)
        method: Fitting method ('mle', 'moments', 'bayesian')

    Returns:
        FitResult with fitted parameters
    """
    fitters = {
        "normal": NormalFitter,
        "exponential": ExponentialFitter,
        "gamma": GammaFitter,
        "beta": BetaFitter,
        "poisson": PoissonFitter,
        "weibull": WeibullFitter,
        "lognormal": LogNormalFitter,
        "multivariate_normal": MultivariateNormalFitter,
    }

    if distribution.lower() not in fitters:
        raise ValueError(f"Unknown distribution: {distribution}")

    fitter_class = fitters[distribution.lower()]
    fitter = fitter_class()

    if method == "mle":
        return fitter.fit_mle(data)
    elif method == "moments":
        return fitter.fit_moments(data)
    else:
        raise ValueError(f"Unknown method: {method}")


def compare_distributions(
    data: Tensor,
    distributions: List[str],
) -> Dict[str, FitResult]:
    """
    Compare multiple distributions and select best via AIC/BIC.

    Args:
        data: Sample data
        distributions: List of distribution names to compare

    Returns:
        Dictionary mapping distribution names to FitResults
    """
    results = {}

    for dist in distributions:
        try:
            results[dist] = fit_distribution(data, dist)
        except Exception as e:
            print(f"Failed to fit {dist}: {e}")

    return results


def select_best_distribution(
    results: Dict[str, FitResult],
    criterion: str = "AIC",
) -> Tuple[str, FitResult]:
    """
    Select best distribution based on information criterion.

    Args:
        results: Dictionary of FitResults
        criterion: 'AIC' or 'BIC'

    Returns:
        Tuple of (best_distribution_name, FitResult)
    """
    if criterion == "AIC":
        best = min(results.items(), key=lambda x: x[1].AIC)
    elif criterion == "BIC":
        best = min(results.items(), key=lambda x: x[1].BIC)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return best[0], best[1]
