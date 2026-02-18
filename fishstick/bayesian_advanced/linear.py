import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.noise_std = noise_std

        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * -2.0)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features) - 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_std = F.softplus(self.weight_rho)
        bias_std = F.softplus(self.bias_rho)

        weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        weight_std = F.softplus(self.weight_rho)
        bias_std = F.softplus(self.bias_rho)

        weight_kl = (
            torch.log(self.prior_std / weight_std)
            + (weight_std**2 + (self.weight_mu - self.prior_mean) ** 2)
            / (2 * self.prior_std**2)
            - 0.5
        ).sum()

        bias_kl = (
            torch.log(self.prior_std / bias_std)
            + (bias_std**2 + (self.bias_mu - self.prior_mean) ** 2)
            / (2 * self.prior_std**2)
            - 0.5
        ).sum()

        return weight_kl + bias_kl


class VariationalLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_precision: float = 1.0,
        num_samples: int = 1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_precision = prior_precision
        self.num_samples = num_samples

        self.weight_mean = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.bias_log_var = nn.Parameter(torch.zeros(out_features))

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.weight_mean)
        nn.init.constant_(self.weight_log_var, -5.0)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            weight_std = torch.exp(0.5 * self.weight_log_var)
            bias_std = torch.exp(0.5 * self.bias_log_var)

            weight = self.weight_mean + weight_std * torch.randn_like(self.weight_mean)
            bias = self.bias_mean + bias_std * torch.randn_like(self.bias_mean)
        else:
            weight = self.weight_mean
            bias = self.bias_mean

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        weight_var = torch.exp(self.weight_log_var)
        bias_var = torch.exp(self.bias_log_var)

        weight_kl = (
            0.5
            * (
                self.prior_precision * (weight_var + self.weight_mean**2)
                - self.weight_log_var
                - 1
                + torch.log(self.prior_precision)
            ).sum()
        )

        bias_kl = (
            0.5
            * (
                self.prior_precision * (bias_var + self.bias_mean**2)
                - self.bias_log_var
                - 1
                + torch.log(self.prior_precision)
            ).sum()
        )

        return weight_kl + bias_kl


class ELBOLoss(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        prior_precision: float = 1.0,
        num_data: int = 1,
    ):
        super().__init__()
        self.model = model
        self.prior_precision = prior_precision
        self.num_data = num_data

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        nll = F.mse_loss(pred, target, reduction="sum")

        kl = sum(
            layer.kl_divergence()
            for layer in self.model.modules()
            if isinstance(layer, VariationalLinear)
        )

        elbo = -nll - kl * self.num_data / len(target)
        return -elbo, {"elbo": elbo.item(), "nll": nll.item(), "kl": kl.item()}


class EvidenceLowerBound(nn.Module):
    def __init__(
        self,
        likelihood_std: float = 0.1,
        kl_weight: float = 1.0,
    ):
        super().__init__()
        self.likelihood_std = likelihood_std
        self.kl_weight = kl_weight

    def compute_log_likelihood(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self.likelihood_std
        nll = (
            0.5 * torch.log(2 * torch.pi * sigma**2)
            + 0.5 * (pred - target) ** 2 / sigma**2
        ).sum()
        return -nll

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        pred = model(x)
        log_likelihood = self.compute_log_likelihood(pred, target)

        kl = sum(
            layer.kl_divergence()
            for layer in model.modules()
            if hasattr(layer, "kl_divergence")
        )

        elbo = log_likelihood - self.kl_weight * kl

        return -elbo, {
            "elbo": elbo.item(),
            "log_likelihood": log_likelihood.item(),
            "kl": kl.item() if isinstance(kl, torch.Tensor) else kl,
        }


class BayesianRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        prior_std: float = 1.0,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        layers.append(
            BayesianLinear(
                input_dim,
                hidden_dim,
                prior_mean=0,
                prior_std=prior_std,
                noise_std=noise_std,
            )
        )
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(
                BayesianLinear(
                    hidden_dim,
                    hidden_dim,
                    prior_mean=0,
                    prior_std=prior_std,
                    noise_std=noise_std,
                )
            )
            layers.append(nn.ReLU())

        layers.append(
            BayesianLinear(
                hidden_dim, 1, prior_mean=0, prior_std=prior_std, noise_std=noise_std
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    def sample_predict(
        self, x: torch.Tensor, num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        return mean, std
