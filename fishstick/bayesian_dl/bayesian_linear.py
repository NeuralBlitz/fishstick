"""
Bayesian Linear Regression Module.

Implements Bayesian linear regression with various prior options,
including conjugate priors and scalable variational approximations.
"""

from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Normal, Gamma, Beta, MVN


class BayesianLinearRegression(nn.Module):
    """Bayesian Linear Regression with conjugate or variational inference.

    Implements Bayesian linear regression using either:
    - Conjugate gradient descent for exact posterior
    - Variational inference for scalable approximation

    Args:
        in_features: Number of input features
        out_features: Number of output dimensions
        alpha_prior: Prior precision (inverse variance) for weights
        beta_prior: Prior precision for noise
        use_vi: Whether to use variational inference (default: True)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        use_vi: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.use_vi = use_vi

        if use_vi:
            self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
            self.weight_log_sigma = nn.Parameter(torch.zeros(out_features, in_features))
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_log_sigma = nn.Parameter(torch.zeros(out_features))
            self.log_noise_precision = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("weight_mean", torch.zeros(out_features, in_features))
            self.register_buffer(
                "weight_precision", torch.eye(in_features) * alpha_prior
            )
            self.register_buffer("bias_mean", torch.zeros(out_features))
            self.register_buffer("noise_precision", torch.tensor(beta_prior))

    def forward(self, x: Tensor) -> Tensor:
        if self.use_vi:
            if self.training:
                weight = (
                    self.weight_mu
                    + torch.randn_like(self.weight_sigma) * self.weight_sigma
                )
                bias = (
                    self.bias_mu + torch.randn_like(self.bias_sigma) * self.bias_sigma
                )
            else:
                weight = self.weight_mu
                bias = self.bias_mu
        else:
            weight = self.weight_mean
            bias = self.bias_mean

        return F.linear(x, weight, bias)

    def predict(
        self,
        x: Tensor,
        return_uncertainty: bool = True,
        n_samples: int = 100,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Make predictions with uncertainty estimates.

        Args:
            x: Input features
            return_uncertainty: Whether to return uncertainty estimates
            n_samples: Number of samples for MC estimation

        Returns:
            predictions: Mean predictions
            uncertainty: Standard deviation (if return_uncertainty=True)
        """
        if not return_uncertainty:
            return self.forward(x), None

        samples = []
        for _ in range(n_samples):
            if self.use_vi:
                weight = self.weight_mu + torch.randn_like(
                    torch.exp(self.weight_log_sigma)
                ) * torch.exp(self.weight_log_sigma)
                bias = self.bias_mu + torch.randn_like(
                    torch.exp(self.bias_log_sigma)
                ) * torch.exp(self.bias_log_sigma)
            else:
                weight = (
                    torch.distributions.MVN(
                        self.weight_mean.view(-1),
                        torch.linalg.inv(self.weight_precision),
                    )
                    .sample((1,))
                    .view(self.weight_mean.size())
                )
                bias = self.bias_mean

            samples.append(F.linear(x, weight, bias))

        samples = torch.stack(samples)
        predictions = samples.mean(dim=0)
        uncertainty = samples.std(dim=0)

        return predictions, uncertainty

    def kl_divergence(self) -> Tensor:
        """Compute KL divergence for variational approximation."""
        if not self.use_vi:
            return torch.tensor(0.0)

        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        prior_precision = self.alpha_prior

        weight_kl = (
            -0.5 * self.weight_log_sigma.sum()
            + 0.5 * prior_precision * (weight_sigma**2 + self.weight_mu**2).sum()
            - 0.5 * self.weight_mu.numel() * torch.log(torch.tensor(prior_precision))
        )

        bias_kl = (
            -0.5 * self.bias_log_sigma.sum()
            + 0.5 * prior_precision * (bias_sigma**2 + self.bias_mu**2).sum()
            - 0.5 * self.bias_mu.numel() * torch.log(torch.tensor(prior_precision))
        )

        return weight_kl + bias_kl


class SparseBayesianLinearRegression(nn.Module):
    """Sparse Bayesian Linear Regression with ARD prior.

    Implements automatic relevance determination (ARD) which learns
    a separate precision parameter for each input feature, enabling
    feature selection.

    Args:
        in_features: Number of input features
        out_features: Number of output dimensions
    """

    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_alpha = nn.Parameter(torch.zeros(in_features))
        self.log_noise_precision = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            alpha = torch.exp(self.weight_log_alpha)
            weight = self.weight_mu + torch.randn_like(self.weight_mu) / torch.sqrt(
                alpha
            )
        else:
            weight = self.weight_mu

        return F.linear(x, weight)

    def get_active_features(self, threshold: float = 1.0) -> Tensor:
        """Get indices of active features based on learned precision.

        Args:
            threshold: Threshold for determining active features

        Returns:
            Indices of features with precision below threshold
        """
        alpha = torch.exp(self.weight_log_alpha)
        return (alpha < threshold).nonzero(as_tuple=True)[0]

    def kl_divergence(self) -> Tensor:
        alpha = torch.exp(self.weight_log_alpha)

        kl = (
            0.5 * (self.weight_mu**2 * alpha).sum()
            - 0.5 * self.weight_log_alpha.sum()
            + 0.5 * alpha.sum() * torch.log(torch.tensor(2.0 * 3.14159))
        )

        return kl


class RobustBayesianLinearRegression(nn.Module):
    """Robust Bayesian Linear Regression with heavy-tailed likelihood.

    Uses a Student-t distribution for the likelihood instead of Gaussian,
    providing robustness to outliers.

    Args:
        in_features: Number of input features
        out_features: Number of output dimensions
        df: Degrees of freedom for Student-t (lower = heavier tails)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        df: float = 4.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.df = df

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_sigma = nn.Parameter(torch.zeros(out_features))
        self.log_nu = nn.Parameter(torch.tensor(torch.log(torch.tensor(df))))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            weight = self.weight_mu + torch.randn_like(
                torch.exp(self.weight_log_sigma)
            ) * torch.exp(self.weight_log_sigma)
            bias = self.bias_mu + torch.randn_like(
                torch.exp(self.bias_log_sigma)
            ) * torch.exp(self.bias_log_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def robust_loss(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute robust loss using Student-t log-likelihood.

        Args:
            pred: Predictions
            target: Target values

        Returns:
            Negative log-likelihood
        """
        nu = torch.exp(self.log_nu)
        residual = target - pred

        nll = (
            torch.lgamma((nu + 1) / 2)
            - torch.lgamma(nu / 2)
            - 0.5 * torch.log(nu * 3.14159)
            - (nu + 1) / 2 * torch.log(1 + residual**2 / nu)
        )

        return -nll.mean()


class MultiTaskBayesianLinearRegression(nn.Module):
    """Multi-task Bayesian Linear Regression.

    Implements shared feature learning across multiple tasks with
    task-specific precision matrices.

    Args:
        in_features: Number of input features
        out_features: Number of output dimensions (tasks)
        task_covariance: Whether to learn task correlations
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        task_covariance: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.task_covariance = task_covariance

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.zeros(out_features, in_features))

        if task_covariance:
            self.register_parameter(
                "task_precision", nn.Parameter(torch.eye(out_features))
            )
        else:
            self.task_precision = None

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            weight = self.weight_mu + torch.randn_like(
                torch.exp(self.weight_log_sigma)
            ) * torch.exp(self.weight_log_sigma)
        else:
            weight = self.weight_mu

        return F.linear(x, weight)

    def predict(
        self,
        x: Tensor,
        n_samples: int = 100,
    ) -> Tuple[Tensor, Tensor]:
        """Predict with uncertainty estimates accounting for task correlations.

        Args:
            x: Input features
            n_samples: Number of MC samples

        Returns:
            mean: Mean predictions
            covariance: Task covariance matrices
        """
        samples = []
        for _ in range(n_samples):
            weight = self.weight_mu + torch.randn_like(
                torch.exp(self.weight_log_sigma)
            ) * torch.exp(self.weight_log_sigma)
            samples.append(F.linear(x, weight))

        samples = torch.stack(samples)
        mean = samples.mean(dim=0)

        if self.task_covariance:
            cov = samples.var(dim=0) + F.linear(
                torch.eye(self.out_features).to(x.device),
                torch.linalg.inv(self.task_precision),
            ).unsqueeze(0).expand(x.size(0), -1, -1)
        else:
            cov = samples.var(dim=0)

        return mean, cov


class EmpiricalBayesLinearRegression(nn.Module):
    """Empirical Bayes Linear Regression.

    Estimates hyperparameters (prior precision, noise precision) from data
    using type II maximum likelihood (evidence approximation).

    Args:
        in_features: Number of input features
        out_features: Number of output dimensions
    """

    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))

    def fit_em(
        self,
        x: Tensor,
        y: Tensor,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None:
        """Fit using Empirical Bayes via EM algorithm.

        Args:
            x: Training features
            y: Training targets
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
        """
        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)

        for _ in range(max_iter):
            alpha_old = alpha.item()
            beta_old = beta.item()

            x_t = x.t()
            precision = alpha * x_t @ x + beta * torch.eye(self.in_features)
            cov = torch.linalg.inv(precision)
            mean = beta * cov @ x_t @ y

            alpha_new = self.in_features / (
                (self.weight - mean).pow(2).sum() + (cov * precision).sum()
            )
            beta_new = len(y) / ((y - x @ mean - self.bias).pow(2).sum())

            alpha = alpha_new.detach().requires_grad_(True)
            beta = beta_new.detach().requires_grad_(True)

            if abs(alpha - alpha_old) < tol and abs(beta - beta_old) < tol:
                break

        self.log_alpha.data = torch.log(alpha.clamp(min=1e-6))
        self.log_beta.data = torch.log(beta.clamp(min=1e-6))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)

        precision = alpha * x.t() @ x + beta * torch.eye(self.in_features)
        cov = torch.linalg.inv(precision)
        mean = x @ self.weight + self.bias
        var = 1 / beta + (x * (x @ cov)).sum(dim=1, keepdim=True)

        return mean, torch.sqrt(var)
