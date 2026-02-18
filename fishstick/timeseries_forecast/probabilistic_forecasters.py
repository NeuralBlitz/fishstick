"""
Probabilistic Time Series Forecasting Models.

Implements probabilistic and uncertainty-aware forecasting:
- Deep Variational Prediction (DVP)
- Quantile Regression Forecaster
- Gaussian Process-based forecasting
- Ensemble probabilistic methods

Example:
    >>> from fishstick.timeseries_forecast import (
    ...     QuantileForecaster,
    ...     DeepVariationalForecaster,
    ...     GaussianProcessForecaster,
    ...     EnsembleProbabilisticForecaster,
    ... )
"""

from typing import Optional, Tuple, List, Dict, Any, Callable, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass


class QuantileLoss(nn.Module):
    """Quantile loss for multi-quantile forecasting.

    Args:
        quantiles: List of quantiles to predict
    """

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        quantile: float,
    ) -> Tensor:
        """Compute quantile loss.

        Args:
            pred: Predictions [B, L, 1] or [B, 1]
            target: Target values [B, L, 1] or [B, 1]
            quantile: Quantile value

        Returns:
            Quantile loss
        """
        error = target - pred
        loss = torch.max((quantile - 1) * error, quantile * error)
        return loss.mean()

    def total_loss(
        self,
        preds: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Compute total loss across all quantiles.

        Args:
            preds: Predictions [B, L, N_quantiles]
            target: Target values [B, L, 1]

        Returns:
            Total loss
        """
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            loss += self.forward(preds[:, :, i], target, q)
        return loss / len(self.quantiles)


class QuantileForecaster(nn.Module):
    """Quantile regression forecaster for prediction intervals.

    Predicts multiple quantiles simultaneously to construct
    prediction intervals without distributional assumptions.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        pred_len: Prediction horizon
        quantiles: List of quantiles to predict

    Example:
        >>> model = QuantileForecaster(
        ...     input_dim=7,
        ...     hidden_dim=128,
        ...     num_layers=2,
        ...     pred_len=24,
        ...     quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        ... )
        >>> x = torch.randn(32, 96, 7)
        >>> output = model(x)
        >>> # output shape: [32, 24, 5] (5 quantiles)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 24,
        quantiles: Optional[List[float]] = None,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        self.decoder = nn.ModuleList(
            [nn.Linear(hidden_dim, pred_len) for _ in range(self.num_quantiles)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input [B, L, D]

        Returns:
            Quantile predictions [B, L_pred, N_quantiles]
        """
        B = x.shape[0]

        enc_out, (h, c) = self.encoder(x)

        outputs = []
        for i, decoder in enumerate(self.decoder):
            q = decoder(enc_out[:, -1, :])
            q = q.unsqueeze(-1).expand(-1, -1, self.pred_len)
            outputs.append(q)

        output = torch.cat(outputs, dim=-1)
        return output

    def predict_interval(
        self,
        x: Tensor,
        lower_q: float = 0.1,
        upper_q: float = 0.9,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict with confidence interval.

        Args:
            x: Input [B, L, D]
            lower_q: Lower quantile
            upper_q: Upper quantile

        Returns:
            median: Median prediction
            lower: Lower bound
            upper: Upper bound
        """
        preds = self.forward(x)

        q_idx = self.quantiles.index(0.5)
        median = preds[:, :, q_idx]

        lower_idx = self.quantiles.index(lower_q)
        upper_idx = self.quantiles.index(upper_q)

        lower = preds[:, :, lower_idx]
        upper = preds[:, :, upper_idx]

        return median, lower, upper


class DeepVariationalForecaster(nn.Module):
    """Deep Variational Prediction for probabilistic forecasting.

    Uses variational inference to model predictive uncertainty.
    Outputs mean and variance of the forecast distribution.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        pred_len: Prediction horizon
        latent_dim: Latent dimension for variational layer

    Example:
        >>> model = DeepVariationalForecaster(
        ...     input_dim=7,
        ...     hidden_dim=128,
        ...     num_layers=2,
        ...     pred_len=24,
        ...     latent_dim=32,
        ... )
        >>> x = torch.randn(32, 96, 7)
        >>> mean, log_var = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 24,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.latent_dim = latent_dim

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        self.fc_mean = nn.Linear(hidden_dim, pred_len)
        self.fc_log_var_out = nn.Linear(hidden_dim, pred_len)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick.

        Args:
            mu: Mean
            log_var: Log variance

        Returns:
            Sampled latent
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input [B, L, D]

        Returns:
            mean: Predicted mean [B, pred_len]
            log_var: Log variance [B, pred_len]
        """
        B = x.shape[0]

        enc_out, (h, c) = self.encoder(x)

        mu = self.fc_mu(enc_out[:, -1, :])
        log_var = self.fc_log_var(enc_out[:, -1, :])

        z = self.reparameterize(mu, log_var)
        z = z.unsqueeze(1).expand(-1, self.pred_len, -1)

        dec_out, _ = self.decoder(z)

        mean = self.fc_mean(dec_out).squeeze(-1)
        log_var_out = self.fc_log_var_out(dec_out).squeeze(-1)

        return mean, log_var_out

    def sample(
        self,
        x: Tensor,
        n_samples: int = 100,
    ) -> Tensor:
        """Sample from predictive distribution.

        Args:
            x: Input [B, L, D]
            n_samples: Number of samples

        Returns:
            Samples [B, n_samples, pred_len]
        """
        mean, log_var = self.forward(x)
        std = torch.exp(0.5 * log_var)

        samples = []
        for _ in range(n_samples):
            eps = torch.randn_like(mean)
            samples.append(mean + std * eps)

        return torch.stack(samples, dim=1)


class GaussianProcessForecaster(nn.Module):
    """Scalable Gaussian Process Forecaster.

    Uses inducing point approximation for scalable GP regression.

    Args:
        input_dim: Number of input features
        num_inducing: Number of inducing points
        pred_len: Prediction horizon

    Example:
        >>> model = GaussianProcessForecaster(
        ...     input_dim=7,
        ...     num_inducing=50,
        ...     pred_len=24,
        ... )
        >>> x = torch.randn(32, 96, 7)
        >>> mean, variance = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        num_inducing: int = 50,
        pred_len: int = 24,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_inducing = num_inducing
        self.pred_len = pred_len

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.inducing_points = nn.Parameter(torch.randn(num_inducing, 64))

        self.q_mu = nn.Parameter(torch.zeros(num_inducing, pred_len))
        self.q_log_var = nn.Parameter(torch.zeros(num_inducing, pred_len))

        self.mean_function = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len),
        )

        self.log_lengthscale = nn.Parameter(torch.zeros(1))
        self.log_outputscale = nn.Parameter(torch.zeros(1))

    def rbf_kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        """RBF kernel."""
        lengthscale = torch.exp(self.log_lengthscale)
        outputscale = torch.exp(self.log_outputscale)

        dist = torch.cdist(x1, x2) / lengthscale
        return outputscale * torch.exp(-0.5 * dist**2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input [B, L, D]

        Returns:
            mean: Predicted mean [B, pred_len]
            variance: Predicted variance [B, pred_len]
        """
        B = x.shape[0]

        enc_out, _ = self.encoder(x)
        x_enc = enc_out[:, -1, :]

        Kzz = self.rbf_kernel(self.inducing_points, self.inducing_points)
        Kzz = Kzz + 1e-4 * torch.eye(self.num_inducing, device=x.device)

        Kxz = self.rbf_kernel(x_enc, self.inducing_points)

        Kzz_inv = torch.linalg.inv(Kzz)
        q_var = torch.exp(self.q_log_var)

        predictive_mean = Kxz @ Kzz_inv @ self.q_mu

        Kxx_diag = torch.ones(B, device=x.device)
        predictive_var = Kxx_diag.unsqueeze(-1) + 1e-4
        predictive_var = predictive_var + q_var.mean()

        return predictive_mean, predictive_var

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: int = 100,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Monte Carlo prediction with uncertainty.

        Args:
            x: Input [B, L, D]
            n_samples: Number of samples

        Returns:
            mean: Predictive mean
            lower: Lower bound (2.5%)
            upper: Upper bound (97.5%)
        """
        mean, variance = self.forward(x)
        std = torch.sqrt(variance + 1e-6)

        z_975 = 1.96
        z_025 = -1.96

        return mean, mean + z_025 * std, mean + z_975 * std


class EnsembleProbabilisticForecaster(nn.Module):
    """Ensemble probabilistic forecaster.

    Combines multiple forecasters with uncertainty estimation.

    Args:
        forecasters: List of forecaster modules
        weights: Optional weights for ensemble

    Example:
        >>> forecasters = [
        ...     QuantileForecaster(input_dim=7, pred_len=24),
        ...     DeepVariationalForecaster(input_dim=7, pred_len=24),
        ... ]
        >>> ensemble = EnsembleProbabilisticForecaster(forecasters)
    """

    def __init__(
        self,
        forecasters: List[nn.Module],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.forecasters = nn.ModuleList(forecasters)

        if weights is None:
            weights = [1.0 / len(forecasters)] * len(forecasters)
        self.weights = nn.Parameter(torch.tensor(weights))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input [B, L, D]

        Returns:
            mean: Ensemble mean prediction
            uncertainty: Predictive uncertainty (std)
        """
        predictions = []
        for forecaster in self.forecasters:
            if hasattr(forecaster, "forward"):
                pred = forecaster(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                predictions.append(pred)

        predictions = torch.stack(predictions)

        weights = F.softmax(self.weights, dim=0)
        weights = weights.view(-1, 1, 1, 1)

        mean = (predictions * weights).sum(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean, uncertainty

    def predict_with_intervals(
        self,
        x: Tensor,
        confidence: float = 0.95,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict with confidence intervals.

        Args:
            x: Input [B, L, D]
            confidence: Confidence level

        Returns:
            mean: Point prediction
            lower: Lower bound
            upper: Upper bound
        """
        predictions = []
        for forecaster in self.forecasters:
            if hasattr(forecaster, "forward"):
                pred = forecaster(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)

        z = 1.96 if confidence == 0.95 else 2.576
        std = predictions.std(dim=0)

        return mean, mean - z * std, mean + z * std


class MixtureDensityForecaster(nn.Module):
    """Mixture Density Network for time series forecasting.

    Models the output distribution as a mixture of Gaussians.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        pred_len: Prediction horizon
        n_components: Number of mixture components

    Example:
        >>> model = MixtureDensityForecaster(
        ...     input_dim=7,
        ...     hidden_dim=128,
        ...     num_layers=2,
        ...     pred_len=24,
        ...     n_components=5,
        ... )
        >>> x = torch.randn(32, 96, 7)
        >>> pi, mu, sigma = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 24,
        n_components: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.n_components = n_components

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components * pred_len)
        self.fc_sigma = nn.Linear(hidden_dim, n_components * pred_len)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input [B, L, D]

        Returns:
            pi: Mixture weights [B, n_components]
            mu: Means [B, n_components, pred_len]
            sigma: Std devs [B, n_components, pred_len]
        """
        enc_out, _ = self.encoder(x)
        h = enc_out[:, -1, :]

        pi = F.softmax(self.fc_pi(h), dim=-1)

        mu = self.fc_mu(h)
        mu = mu.view(-1, self.n_components, self.pred_len)

        sigma = F.softplus(self.fc_sigma(h))
        sigma = sigma.view(-1, self.n_components, self.pred_len)

        return pi, mu, sigma

    def sample(self, x: Tensor) -> Tensor:
        """Sample from the mixture distribution.

        Args:
            x: Input [B, L, D]

        Returns:
            Samples [B, pred_len]
        """
        pi, mu, sigma = self.forward(x)

        B = x.shape[0]
        component_idx = torch.multinomial(pi, 1).squeeze(-1)

        samples = []
        for b in range(B):
            idx = component_idx[b]
            eps = torch.randn(self.pred_len, device=x.device)
            sample = mu[b, idx] + sigma[b, idx] * eps
            samples.append(sample)

        return torch.stack(samples)


class ProbabilisticLosses:
    """Collection of probabilistic loss functions."""

    @staticmethod
    def gaussian_nll(
        mean: Tensor,
        log_var: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Gaussian negative log-likelihood.

        Args:
            mean: Predicted mean
            log_var: Predicted log variance
            target: Target values

        Returns:
            NLL loss
        """
        var = torch.exp(log_var)
        nll = 0.5 * (torch.log(2 * np.pi * var) + (target - mean) ** 2 / var)
        return nll.mean()

    @staticmethod
    def mixture_nll(
        pi: Tensor,
        mu: Tensor,
        sigma: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Mixture of Gaussians negative log-likelihood.

        Args:
            pi: Mixture weights
            mu: Means
            sigma: Standard deviations
            target: Target values

        Returns:
            NLL loss
        """
        B, K, L = mu.shape
        target_expanded = target.unsqueeze(1).expand(-1, K, -1)

        normal = torch.distributions.Normal(mu, sigma)
        log_probs = normal.log_prob(target_expanded)

        log_probs = log_probs.sum(dim=-1)

        log_pi = torch.log(pi + 1e-8)
        log_sum_exp = torch.logsumexp(log_pi + log_probs, dim=-1)

        return -log_sum_exp.mean()

    @staticmethod
    def dkl_gaussian(
        mu1: Tensor,
        log_var1: Tensor,
        mu2: Tensor,
        log_var2: Tensor,
    ) -> Tensor:
        """KL divergence between two Gaussians.

        Args:
            mu1: First mean
            log_var1: First log variance
            mu2: Second mean
            log_var2: Second log variance

        Returns:
            KL divergence
        """
        var1 = torch.exp(log_var1)
        var2 = torch.exp(log_var2)

        kl = 0.5 * ((var1 + (mu1 - mu2) ** 2) / var2 - 1 + torch.log(var2 / var1))
        return kl.mean()


@dataclass
class ForecastWithUncertainty:
    """Container for probabilistic forecast results."""

    mean: Tensor
    lower: Tensor
    upper: Tensor
    variance: Optional[Tensor] = None
    quantiles: Optional[Dict[float, Tensor]] = None


def create_probabilistic_forecaster(
    model_type: str = "quantile",
    input_dim: int = 7,
    pred_len: int = 24,
    **kwargs,
) -> nn.Module:
    """Factory function to create probabilistic forecasters.

    Args:
        model_type: Type of model ('quantile', 'variational', 'gp', 'mixture')
        input_dim: Number of input features
        pred_len: Prediction horizon
        **kwargs: Additional arguments

    Returns:
        Initialized forecaster

    Example:
        >>> model = create_probabilistic_forecaster('variational', input_dim=7)
    """
    if model_type == "quantile":
        return QuantileForecaster(
            input_dim=input_dim,
            pred_len=pred_len,
            **kwargs,
        )
    elif model_type == "variational":
        return DeepVariationalForecaster(
            input_dim=input_dim,
            pred_len=pred_len,
            **kwargs,
        )
    elif model_type == "gp":
        return GaussianProcessForecaster(
            input_dim=input_dim,
            pred_len=pred_len,
            **kwargs,
        )
    elif model_type == "mixture":
        return MixtureDensityForecaster(
            input_dim=input_dim,
            pred_len=pred_len,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
