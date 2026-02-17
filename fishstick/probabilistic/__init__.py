"""
Probabilistic Neural Networks and Bayesian Deep Learning.

Implements:
- Bayesian layers with variational inference
- Monte Carlo Dropout for uncertainty
- Deep Ensembles
- Evidential Deep Learning
- Conformal prediction
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Distribution
import numpy as np


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with learned mean and variance.

    Uses reparameterization trick for stochastic weights:
        w = μ + σ * ε, where ε ~ N(0, I)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        prior_mean: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.prior_mean = prior_mean

        # Variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        # Initialize
        nn.init.normal_(self.weight_mu, mean=prior_mean, std=0.1)
        nn.init.constant_(self.weight_rho, -3.0)  # σ = log(1 + exp(ρ))
        nn.init.normal_(self.bias_mu, mean=prior_mean, std=0.1)
        nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x: Tensor, sample: bool = True) -> Tensor:
        """
        Forward pass with stochastic weights.

        Args:
            x: Input [batch, in_features]
            sample: Whether to sample weights (False for MAP estimate)

        Returns:
            Output [batch, out_features]
        """
        if sample and self.training:
            # Sample weights
            weight_sigma = F.softplus(self.weight_rho)
            bias_sigma = F.softplus(self.bias_rho)

            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)

            weight = self.weight_mu + weight_sigma * weight_eps
            bias = self.bias_mu + bias_sigma * bias_eps
        else:
            # Use mean (MAP estimate)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> Tensor:
        """Compute KL divergence between variational posterior and prior."""
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        # KL for weights
        kl_weight = torch.sum(
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma**2 + (self.weight_mu - self.prior_mean) ** 2)
            / (2 * self.prior_sigma**2)
            - 0.5
        )

        # KL for biases
        kl_bias = torch.sum(
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma**2 + (self.bias_mu - self.prior_mean) ** 2)
            / (2 * self.prior_sigma**2)
            - 0.5
        )

        return kl_weight + kl_bias


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.

    Dropout applied at test time to estimate epistemic uncertainty.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        """Apply dropout always (both train and test)."""
        return F.dropout(x, p=self.p, training=True, inplace=self.inplace)


class VariationalLayer(nn.Module):
    """
    Variational layer that outputs a distribution.

    Instead of point estimates, outputs mean and variance.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, out_features)
        self.log_var_layer = nn.Linear(hidden_dim, out_features)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Returns:
            mean: Predictive mean [batch, out_features]
            log_var: Log variance [batch, out_features]
        """
        h = self.shared(x)
        mean = self.mean_layer(h)
        log_var = self.log_var_layer(h)
        return mean, log_var

    def sample(self, x: Tensor, n_samples: int = 100) -> Tensor:
        """Sample from the predictive distribution."""
        mean, log_var = self.forward(x)
        std = torch.exp(0.5 * log_var)

        samples = []
        for _ in range(n_samples):
            eps = torch.randn_like(mean)
            samples.append(mean + std * eps)

        return torch.stack(samples)


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty quantification.

    Multiple models trained with different initializations.
    """

    def __init__(
        self,
        model_class: Callable,
        n_models: int = 5,
        **model_kwargs,
    ):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList(
            [model_class(**model_kwargs) for _ in range(n_models)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through all models.

        Returns:
            mean: Ensemble mean prediction
            uncertainty: Predictive uncertainty (std)
        """
        predictions = torch.stack([model(x) for model in self.models])
        mean = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        return mean, uncertainty

    def predict_with_uncertainty(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predict with full uncertainty decomposition.

        Returns:
            mean: Predictive mean
            aleatoric: Data uncertainty
            epistemic: Model uncertainty
        """
        predictions = torch.stack([model(x) for model in self.models])
        mean = predictions.mean(dim=0)

        # Epistemic uncertainty (variance between models)
        epistemic = predictions.var(dim=0)

        # Aleatoric uncertainty (average uncertainty)
        # Assuming models output (mean, log_var)
        # This is a simplified version
        aleatoric = predictions.var(dim=0) * 0.1  # Placeholder

        return mean, aleatoric, epistemic


class EvidentialLayer(nn.Module):
    """
    Evidential Deep Learning for uncertainty quantification.

    Instead of predicting mean and variance, predicts parameters
    of a higher-order distribution (Normal-Inverse-Gamma).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # NIG parameters
        self.gamma = nn.Linear(hidden_dim, out_features)  # Mean
        self.nu = nn.Linear(hidden_dim, out_features)  # Degrees of freedom
        self.alpha = nn.Linear(hidden_dim, out_features)  # Shape
        self.beta = nn.Linear(hidden_dim, out_features)  # Scale

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass.

        Returns:
            Dict with NIG parameters
        """
        h = self.network(x)

        gamma = self.gamma(h)
        nu = F.softplus(self.nu(h)) + 1.0
        alpha = F.softplus(self.alpha(h)) + 1.0
        beta = F.softplus(self.beta(h)) + 1e-8

        return {
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
        }

    def nig_nll(self, y: Tensor, params: Dict[str, Tensor]) -> Tensor:
        """
        Negative log-likelihood for Normal-Inverse-Gamma.

        Loss from: Evidential Deep Learning (Amini et al., 2020)
        """
        gamma, nu, alpha, beta = (
            params["gamma"],
            params["nu"],
            params["alpha"],
            params["beta"],
        )

        omega = 2 * beta * (1 + nu)
        nll = (
            0.5 * torch.log(np.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        return nll.mean()

    def nig_reg(self, y: Tensor, params: Dict[str, Tensor]) -> Tensor:
        """Regularization term for evidential learning."""
        gamma, nu, alpha, beta = (
            params["gamma"],
            params["nu"],
            params["alpha"],
            params["beta"],
        )

        error = torch.abs(y - gamma)
        reg = error * (2 * nu + alpha)
        return reg.mean()


class ConformalPredictor:
    """
    Conformal prediction for calibrated uncertainty.

    Provides prediction sets with guaranteed coverage.
    """

    def __init__(self, model: nn.Module, alpha: float = 0.05):
        """
        Args:
            model: Trained model
            alpha: Miscoverage rate (1-alpha = coverage)
        """
        self.model = model
        self.alpha = alpha
        self.q_hat = None
        self.calibration_scores = []

    def calibrate(self, x_cal: Tensor, y_cal: Tensor) -> None:
        """
        Calibrate using calibration set.

        Args:
            x_cal: Calibration inputs
            y_cal: Calibration targets
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "forward"):
                pred = self.model(x_cal)
            else:
                pred = self.model(x_cal)[0]  # Assuming returns (mean, var)

            # Compute non-conformity scores
            scores = torch.abs(pred - y_cal)

            # Compute quantile
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self.q_hat = torch.quantile(scores, q_level)

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict with prediction intervals.

        Returns:
            mean: Point prediction
            interval: (lower, upper) prediction interval
        """
        if self.q_hat is None:
            raise ValueError("Must calibrate before prediction")

        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "forward"):
                pred = self.model(x)
            else:
                pred = self.model(x)[0]

        lower = pred - self.q_hat
        upper = pred + self.q_hat

        return pred, (lower, upper)


class BayesianNeuralNetwork(nn.Module):
    """
    Complete Bayesian neural network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_sigma))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(BayesianLinear(prev_dim, output_dim, prior_sigma))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor, sample: bool = True) -> Tensor:
        """Forward pass."""
        # Need to handle sampling in BayesianLinear layers
        output = x
        for layer in self.network:
            if isinstance(layer, BayesianLinear):
                output = layer(output, sample=sample)
            else:
                output = layer(output)
        return output

    def kl_divergence(self) -> Tensor:
        """Total KL divergence."""
        kl = 0.0
        for layer in self.network:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_divergence()
        return kl

    def elbo_loss(
        self,
        x: Tensor,
        y: Tensor,
        n_samples: int = 1,
        beta: float = 1.0,
    ) -> Tensor:
        """
        Evidence Lower Bound (ELBO) loss.

        L = E_q[log p(y|x,w)] - β * KL(q(w)||p(w))
        """
        log_likelihood = 0.0
        for _ in range(n_samples):
            pred = self.forward(x, sample=True)
            log_likelihood += -F.mse_loss(pred, y, reduction="sum")
        log_likelihood /= n_samples

        kl = self.kl_divergence()

        return -(log_likelihood - beta * kl) / x.size(0)

    def predict_with_uncertainty(
        self, x: Tensor, n_samples: int = 100
    ) -> Tuple[Tensor, Tensor]:
        """
        Monte Carlo prediction with uncertainty.

        Returns:
            mean: Predictive mean
            uncertainty: Predictive uncertainty (std)
        """
        self.eval()
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, sample=True)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean, uncertainty


class StochasticVariationalGP(nn.Module):
    """
    Stochastic Variational Gaussian Process.

    Scalable GP using inducing points.
    """

    def __init__(
        self,
        input_dim: int,
        num_inducing: int = 100,
        kernel: str = "rbf",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_inducing = num_inducing

        # Inducing points
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, input_dim))

        # Variational parameters
        self.q_mu = nn.Parameter(torch.zeros(num_inducing))
        self.q_log_var = nn.Parameter(torch.zeros(num_inducing))

        # Kernel parameters
        self.log_lengthscale = nn.Parameter(torch.zeros(1))
        self.log_outputscale = nn.Parameter(torch.zeros(1))

    def kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        """RBF kernel."""
        lengthscale = torch.exp(self.log_lengthscale)
        outputscale = torch.exp(self.log_outputscale)

        dist = torch.cdist(x1, x2) / lengthscale
        return outputscale * torch.exp(-0.5 * dist**2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Returns:
            mean: Predictive mean
            var: Predictive variance
        """
        # Compute kernels
        Kxx = self.kernel(x, x)
        Kxz = self.kernel(x, self.inducing_points)
        Kzz = self.kernel(self.inducing_points, self.inducing_points)

        # Add jitter for numerical stability
        Kzz = Kzz + 1e-4 * torch.eye(self.num_inducing, device=x.device)

        # Compute predictive distribution
        Kzz_inv = torch.linalg.inv(Kzz)

        q_var = torch.exp(self.q_log_var)
        predictive_mean = Kxz @ Kzz_inv @ self.q_mu

        # Simplified predictive variance
        predictive_var = Kxx.diagonal() - (Kxz @ Kzz_inv * Kxz).sum(dim=1)
        predictive_var = predictive_var + q_var.mean()

        return predictive_mean, predictive_var


class ProbabilisticLosses:
    """Collection of probabilistic loss functions."""

    @staticmethod
    def negative_log_likelihood(
        pred_mean: Tensor,
        pred_log_var: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Negative log-likelihood for Gaussian.

        -log p(y|μ,σ²) = 0.5 * (log(2πσ²) + (y-μ)²/σ²)
        """
        var = torch.exp(pred_log_var)
        nll = 0.5 * (torch.log(2 * np.pi * var) + (target - pred_mean) ** 2 / var)
        return nll.mean()

    @staticmethod
    def beta_nll(
        pred_mean: Tensor,
        pred_var: Tensor,
        target: Tensor,
        beta: float = 0.5,
    ) -> Tensor:
        """
        Beta-NLL for heteroscedastic regression.

        From: On the Pitfalls of Heteroscedastic Uncertainty Estimation
        """
        var = torch.clamp(pred_var, min=1e-6)
        inv_var = var ** (-beta)

        loss = 0.5 * (inv_var * (target - pred_mean) ** 2 + torch.log(var))
        return loss.mean()

    @staticmethod
    def quantile_loss(
        pred: Tensor,
        target: Tensor,
        quantile: float = 0.5,
    ) -> Tensor:
        """Pinball loss for quantile regression."""
        error = target - pred
        loss = torch.where(error >= 0, quantile * error, (quantile - 1) * error)
        return loss.mean()
