"""
Bayesian statistics primitives.

Provides tools for Bayesian inference including conjugate priors,
Markov Chain Monte Carlo, Variational Inference, and Bayesian estimators.
"""

from typing import Optional, Tuple, List, Callable, Dict, Any
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import numpy as np


@dataclass
class Prior:
    """Base class for prior distributions."""

    pass


@dataclass
class Posterior:
    """Base class for posterior distributions."""

    pass


class NormalNormalConjugate:
    """
    Normal-Normal conjugate model for Bayesian linear regression.

    Likelihood: y | X, w ~ N(Xw, σ²I)
    Prior: w ~ N(μ₀, Σ₀)
    Posterior: w | y, X ~ N(μₙ, Σₙ)

    where:
        Σₙ = (Σ₀⁻¹ + XᵀX/σ²)⁻¹
        μₙ = Σₙ(Σ₀⁻¹μ₀ + Xᵀy/σ²)
    """

    def __init__(
        self,
        prior_mean: Tensor,
        prior_cov: Tensor,
        noise_std: float = 1.0,
    ):
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.noise_std = noise_std
        self.prior_precision = torch.linalg.inv(prior_cov)

    def posterior(
        self,
        X: Tensor,
        y: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute posterior parameters.

        Args:
            X: Design matrix [n_samples, n_features]
            y: Target values [n_samples]

        Returns:
            Tuple of (posterior_mean, posterior_cov)
        """
        XtX = X.T @ X
        Xty = X.T @ y

        noise_precision = 1.0 / (self.noise_std**2)

        posterior_precision = self.prior_precision + noise_precision * XtX
        posterior_cov = torch.linalg.inv(posterior_precision)

        posterior_mean = posterior_cov @ (
            self.prior_precision @ self.prior_mean + noise_precision * Xty
        )

        return posterior_mean, posterior_cov

    def predictive(
        self,
        X: Tensor,
        posterior_mean: Tensor,
        posterior_cov: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute predictive distribution parameters.

        Returns:
            Tuple of (predictive_mean, predictive_variance)
        """
        predictive_mean = X @ posterior_mean
        predictive_var = self.noise_std**2 + torch.sum(X @ posterior_cov * X, dim=-1)

        return predictive_mean, predictive_var


class BetaBernoulliConjugate:
    """
    Beta-Bernoulli conjugate model for Bayesian binary classification.

    Likelihood: x | θ ~ Bern(θ)
    Prior: θ ~ Beta(α, β)
    Posterior: θ | x ~ Beta(α + Σxᵢ, β + n - Σxᵢ)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def posterior(self, x: Tensor) -> Tuple[float, float]:
        """
        Compute posterior parameters.

        Args:
            x: Binary observations [n_samples]

        Returns:
            Tuple of (posterior_alpha, posterior_beta)
        """
        n_successes = x.sum().item()
        n_failures = len(x) - n_successes

        return self.alpha + n_successes, self.beta + n_failures

    def predictive_mean(self, posterior_alpha: float, posterior_beta: float) -> float:
        """Mean of posterior predictive."""
        return posterior_alpha / (posterior_alpha + posterior_beta)

    def credible_interval(
        self,
        posterior_alpha: float,
        posterior_beta: float,
        level: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute credible interval for θ."""
        from scipy import stats

        return stats.beta.ppf(
            (1 - level) / 2, posterior_alpha, posterior_beta
        ), stats.beta.ppf(1 - (1 - level) / 2, posterior_alpha, posterior_beta)


class GammaPoissonConjugate:
    """
    Gamma-Poisson conjugate model for count data.

    Likelihood: x | λ ~ Pois(λ)
    Prior: λ ~ Γ(α, β)
    Posterior: λ | x ~ Γ(α + Σxᵢ, β + n)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def posterior(self, x: Tensor) -> Tuple[float, float]:
        """
        Compute posterior parameters.

        Args:
            x: Count observations [n_samples]

        Returns:
            Tuple of (posterior_alpha, posterior_beta)
        """
        total_count = x.sum().item()
        n = len(x)

        return self.alpha + total_count, self.beta + n

    def predictive_mean(self, posterior_alpha: float, posterior_beta: float) -> float:
        """Mean of posterior predictive (negative binomial)."""
        return posterior_alpha / posterior_beta


class MetropolisHastings:
    """
    Metropolis-Hastings MCMC sampler.

    Generic MCMC algorithm for sampling from arbitrary posterior.
    """

    def __init__(
        self,
        log_posterior_fn: Callable[[Tensor], Tensor],
        proposal_fn: Optional[Callable[[Tensor], Tensor]] = None,
        initial_sample: Optional[Tensor] = None,
    ):
        self.log_posterior = log_posterior_fn
        self.proposal_fn = proposal_fn or self._default_proposal
        self.current_sample = initial_sample
        self.samples: List[Tensor] = []
        self.acceptance_rate = 0.0

    def _default_proposal(self, x: Tensor) -> Tensor:
        """Default Gaussian random walk proposal."""
        return x + torch.randn_like(x) * 0.1

    def sample(
        self,
        n_samples: int,
        burn_in: int = 100,
        thinning: int = 1,
    ) -> Tensor:
        """
        Run MCMC sampling.

        Args:
            n_samples: Number of samples to generate
            burn_in: Number of initial samples to discard
            thinning: Keep every thinning-th sample

        Returns:
            Tensor of shape [n_samples, *dim]
        """
        if self.current_sample is None:
            raise ValueError("Initial sample not set")

        self.samples = []
        n_accept = 0

        current = self.current_sample
        current_log_prob = self.log_posterior(current)

        for i in range(burn_in + n_samples * thinning):
            proposal = self.proposal_fn(current)
            proposal_log_prob = self.log_posterior(proposal)

            log_accept_ratio = proposal_log_prob - current_log_prob

            if torch.log(torch.rand(())) < log_accept_ratio:
                current = proposal
                current_log_prob = proposal_log_prob
                if i >= burn_in:
                    n_accept += 1

            if i >= burn_in and (i - burn_in) % thinning == 0:
                self.samples.append(current.clone())

        self.current_sample = current
        self.acceptance_rate = n_accept / n_samples

        return torch.stack(self.samples)

    def set_initial_sample(self, sample: Tensor) -> None:
        """Set initial sample."""
        self.current_sample = sample


class HamiltonianMonteCarlo:
    """
    Hamiltonian Monte Carlo (HMC) sampler.

    Uses gradient information for efficient exploration of posterior.
    """

    def __init__(
        self,
        log_posterior_fn: Callable[[Tensor], Tensor],
        step_size: float = 0.01,
        n_leapfrog: int = 10,
        initial_sample: Optional[Tensor] = None,
    ):
        self.log_posterior = log_posterior_fn
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.current_sample = initial_sample
        self.samples: List[Tensor] = []

    def _grad_log_posterior(self, x: Tensor) -> Tensor:
        """Compute gradient of log posterior."""
        x = x.requires_grad_(True)
        log_prob = self.log_posterior(x)
        return torch.autograd.grad(log_prob, x, retain_graph=False)[0]

    def _leapfrog(self, x: Tensor, p: Tensor) -> Tuple[Tensor, Tensor]:
        """Leapfrog integrator for Hamiltonian dynamics."""
        for _ in range(self.n_leapfrog):
            p = p + 0.5 * self.step_size * self._grad_log_posterior(x)
            x = x + self.step_size * p
            p = p + 0.5 * self.step_size * self._grad_log_posterior(x)

        return x, -p

    def sample(
        self,
        n_samples: int,
        burn_in: int = 100,
        thinning: int = 1,
    ) -> Tensor:
        """
        Run HMC sampling.

        Args:
            n_samples: Number of samples to generate
            burn_in: Number of initial samples to discard
            thinning: Keep every thinning-th sample

        Returns:
            Tensor of shape [n_samples, *dim]
        """
        if self.current_sample is None:
            raise ValueError("Initial sample not set")

        self.samples = []
        current = self.current_sample

        for i in range(burn_in + n_samples * thinning):
            p = torch.randn_like(current)

            proposed_x, proposed_p = self._leapfrog(current, p)

            current_log_prob = self.log_posterior(current)
            proposed_log_prob = self.log_posterior(proposed_x)

            h_current = -current_log_prob + 0.5 * torch.sum(p**2)
            h_proposed = -proposed_log_prob + 0.5 * torch.sum(proposed_p**2)

            if torch.log(torch.rand(())) < -(h_proposed - h_current):
                current = proposed_x

            if i >= burn_in and (i - burn_in) % thinning == 0:
                self.samples.append(current.clone())

        self.current_sample = current

        return torch.stack(self.samples)

    def set_initial_sample(self, sample: Tensor) -> None:
        """Set initial sample."""
        self.current_sample = sample


class VariationalInference:
    """
    Variational Inference for Bayesian models.

    Approximates posterior using optimization.
    """

    def __init__(
        self,
        log_likelihood_fn: Callable[[Tensor, Tensor], Tensor],
        log_prior_fn: Callable[[Tensor], Tensor],
        variational_family: str = "mean_field",
    ):
        self.log_likelihood = log_likelihood_fn
        self.log_prior = log_prior_fn
        self.variational_family = variational_family

    def elbo(
        self,
        params: Tensor,
        samples: Tensor,
    ) -> Tensor:
        """
        Evidence Lower Bound (ELBO).

        ELBO = E[q(θ)] [log p(x|θ)] + E[q(θ)] [log p(θ)] - E[q(θ)] [log q(θ)]
        """
        log_lik = self.log_likelihood(params, samples).mean()
        log_prior = self.log_prior(params).mean()

        entropy = 0.5 * (1 + np.log(2 * np.pi)) + torch.log(params.std())

        return log_lik + log_prior - entropy

    def fit(
        self,
        data: Tensor,
        initial_params: Tensor,
        n_iter: int = 1000,
        lr: float = 0.01,
    ) -> Tensor:
        """
        Fit variational distribution to posterior.

        Args:
            data: Observed data
            initial_params: Initial variational parameters
            n_iter: Number of optimization iterations
            lr: Learning rate

        Returns:
            Fitted variational parameters
        """
        params = initial_params.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=lr)

        for _ in range(n_iter):
            optimizer.zero_grad()

            samples = params.unsqueeze(0) + torch.randn(
                10, *params.shape
            ) * params.unsqueeze(0)

            loss = -self.elbo(params, data)
            loss.backward()
            optimizer.step()

        return params.detach()


class BayesianModelAverage:
    """
    Bayesian Model Averaging for combining predictions from multiple models.

    Weights models by their posterior model probability.
    """

    def __init__(self):
        self.models: List[Callable] = []
        self.log_marginal_likelihoods: List[float] = []
        self.posterior_weights: Optional[Tensor] = None

    def add_model(
        self,
        model: Callable[[Tensor], Tensor],
        log_marginal_likelihood: float,
    ) -> None:
        """Add a model with its log marginal likelihood."""
        self.models.append(model)
        self.log_marginal_likelihoods.append(log_marginal_likelihood)

    def fit(self) -> None:
        """Compute posterior model probabilities."""
        log_ml = torch.tensor(self.log_marginal_likelihoods)

        log_posterior = log_ml - torch.logsumexp(log_ml, dim=0)
        self.posterior_weights = torch.exp(log_posterior)

    def predict(self, x: Tensor) -> Tensor:
        """
        Compute weighted prediction.

        Returns weighted average of model predictions.
        """
        if self.posterior_weights is None:
            self.fit()

        predictions = torch.stack([model(x) for model in self.models])

        weighted_pred = torch.einsum("n,nmk->mk", self.posterior_weights, predictions)

        return weighted_pred


class EmpiricalBayes:
    """
    Empirical Bayes for hyperparameter estimation.

    Estimates prior hyperparameters from data using marginal likelihood.
    """

    def __init__(self, prior_type: str = "normal"):
        self.prior_type = prior_type

    def fit(
        self,
        data: Tensor,
        initial_hyperparams: Dict[str, float],
        n_iter: int = 100,
    ) -> Dict[str, float]:
        """
        Estimate hyperparameters via Empirical Bayes.

        Uses type II maximum likelihood.
        """
        hyperparams = initial_hyperparams.copy()

        for _ in range(n_iter):
            if self.prior_type == "normal":
                mu_0 = hyperparams["mu_0"]
                sigma_0 = hyperparams["sigma_0"]

                n = len(data)
                x_bar = data.mean()
                s2 = data.var()

                sigma_n = 1 / (1 / sigma_0**2 + n / 1.0)
                mu_n = sigma_n * (mu_0 / sigma_0**2 + n * x_bar / 1.0)

                hyperparams["mu_0"] = mu_n.item()
                hyperparams["sigma_0"] = np.sqrt(sigma_n).item()

        return hyperparams


class BayesianLinearRegression:
    """
    Complete Bayesian Linear Regression model.

    Combines prior, likelihood, and posterior with predictive distribution.
    """

    def __init__(
        self,
        prior_mean: Tensor,
        prior_cov: Tensor,
        noise_precision: float = 1.0,
    ):
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.noise_precision = noise_precision
        self.posterior_mean: Optional[Tensor] = None
        self.posterior_cov: Optional[Tensor] = None
        self.fitted = False

    def fit(self, X: Tensor, y: Tensor) -> "BayesianLinearRegression":
        """Fit model to data."""
        XtX = X.T @ X
        Xty = X.T @ y

        prior_precision = torch.linalg.inv(self.prior_cov)

        posterior_precision = prior_precision + self.noise_precision * XtX
        self.posterior_cov = torch.linalg.inv(posterior_precision)
        self.posterior_mean = self.posterior_cov @ (
            prior_precision @ self.prior_mean + self.noise_precision * Xty
        )

        self.fitted = True
        return self

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict with uncertainty.

        Returns:
            Tuple of (mean, variance)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        mean = X @ self.posterior_mean

        epistemic_var = torch.sum(X @ self.posterior_cov * X, dim=-1)
        aleatoric_var = 1.0 / self.noise_precision

        var = epistemic_var + aleatoric_var

        return mean, var

    def sample_posterior(self, n_samples: int) -> Tensor:
        """Sample from posterior distribution."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return torch.multivariate_normal(
            self.posterior_mean, self.posterior_cov, (n_samples,)
        )


def compute_log_marginal_likelihood(
    log_likelihood: float,
    n_params: int,
    n_samples: int,
) -> float:
    """
    Compute approximate log marginal likelihood using BIC approximation.

    log p(x) ≈ log p(x|θ*) + (d/2) log n - log p(θ*|x)
    """
    return log_likelihood - 0.5 * n_params * np.log(n_samples)


def bayesian_information_criterion(
    log_likelihood: float,
    n_params: int,
    n_samples: int,
) -> float:
    """BIC = k log n - 2 log L"""
    return n_params * np.log(n_samples) - 2 * log_likelihood


def akaike_information_criterion(
    log_likelihood: float,
    n_params: int,
) -> float:
    """AIC = 2k - 2 log L"""
    return 2 * n_params - 2 * log_likelihood


class ConjugateGradientDescent:
    """
    Conjugate Gradient optimization for quadratic problems.

    Used in Gaussian Process regression and ridge regression.
    """

    def __init__(self):
        self.solution: Optional[Tensor] = None

    def solve(
        self,
        A: Tensor,
        b: Tensor,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> Tensor:
        """
        Solve Ax = b using Conjugate Gradient.

        Args:
            A: Positive definite matrix [n, n]
            b: Right-hand side [n]
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            Solution vector x
        """
        n = A.shape[0]

        x = torch.zeros(n)
        r = b - A @ x
        p = r.clone()
        rsold = r @ r

        for _ in range(max_iter):
            Ap = A @ p
            alpha = rsold / (p @ Ap + 1e-10)

            x = x + alpha * p
            r = r - alpha * Ap

            rsnew = r @ r

            if torch.sqrt(rsnew) < tol:
                break

            p = r + (rsnew / rsold) * p
            rsold = rsnew

        self.solution = x
        return x
