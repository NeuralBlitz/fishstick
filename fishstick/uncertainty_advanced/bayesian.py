import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict, Any, Callable, Tuple
from abc import ABC, abstractmethod
import math


class BayesianModule(nn.Module, ABC):
    @abstractmethod
    def sample(self) -> None:
        pass

    @abstractmethod
    def kl_divergence(self) -> torch.Tensor:
        pass


class MCDropout(nn.Module):
    def __init__(self, model: nn.Module, dropout_rate: float = 0.5):
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self._apply_dropout()

    def _apply_dropout(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate
            elif isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                module_dropout = nn.Dropout(p=self.dropout_rate)
                module.register_forward_hook(lambda m, inp, out: module_dropout(out))

    def enable_dropout(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def disable_dropout(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def forward(self, x: Tensor, n_samples: int = 10) -> Tuple[Tensor, Tensor]:
        self.enable_dropout()
        self.model.eval()

        outputs = []
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(x)
                outputs.append(output)

        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)

        return mean, variance

    def predict_with_uncertainty(
        self, x: Tensor, n_samples: int = 10
    ) -> Dict[str, Tensor]:
        mean, variance = self.forward(x, n_samples)
        std = torch.sqrt(variance)

        return {
            "mean": mean,
            "variance": variance,
            "std": std,
            "predictions": mean.argmax(dim=-1),
        }


class VariationalInference(nn.Module):
    def __init__(
        self,
        model: BayesianModule,
        kl_weight: float = 1.0,
        n_samples: int = 1,
    ):
        super().__init__()
        self.model = model
        self.kl_weight = kl_weight
        self.n_samples = n_samples

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if target is not None:
            return self.forward_with_loss(x, target)
        else:
            return self.predict(x)

    def predict(self, x: Tensor) -> Dict[str, Tensor]:
        self.model.sample()

        outputs = []
        for _ in range(self.n_samples):
            output = self.model(x)
            outputs.append(output)

        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)

        return {
            "mean": mean,
            "variance": variance,
            "predictions": mean.argmax(dim=-1),
        }

    def forward_with_loss(self, x: Tensor, target: Tensor) -> Dict[str, Tensor]:
        self.model.sample()

        log_likelihoods = []
        for _ in range(self.n_samples):
            output = self.model(x)
            log_likelihood = F.cross_entropy(output, target, reduction="none")
            log_likelihoods.append(log_likelihood)

        log_likelihood = torch.stack(log_likelihoods).mean()
        kl = self.model.kl_divergence()

        loss = -log_likelihood + self.kl_weight * kl

        return {
            "loss": loss,
            "log_likelihood": log_likelihood,
            "kl_divergence": kl,
            "predictions": self.model(x).argmax(dim=-1),
        }


class SWAG(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        n_models: int = 10,
        rank: int = 20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.base_model = base_model
        self.n_models = n_models
        self.rank = rank
        self.device = device

        self._collect_parameters()
        self._init_covariance_matrices()

    def _collect_parameters(self) -> None:
        self.param_names = []
        self.param_shapes = []

        for name, param in self.base_model.named_parameters():
            self.param_names.append(name)
            self.param_shapes.append(param.shape)

    def _init_covariance_matrices(self) -> None:
        self.mean_parameters: Dict[str, Tensor] = {}
        self.cov_factor: Dict[str, Tensor] = {}
        self.deviations: List[Dict[str, Tensor]] = []

        for name, param in self.base_model.named_parameters():
            self.mean_parameters[name] = param.data.clone()
            flat_param = param.numel()
            self.cov_factor[name] = torch.zeros(
                flat_param, self.rank, device=self.device
            )

    def collect_deviation(self) -> None:
        deviation = {}
        for name, param in self.base_model.named_parameters():
            dev = param.data - self.mean_parameters[name]
            deviation[name] = dev.flatten()

        if len(self.deviations) < self.rank:
            for name in self.param_names:
                self.cov_factor[name][:, len(self.deviations)] = deviation[name]

        self.deviations.append(deviation)

    def sample(self) -> None:
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                mean = self.mean_parameters[name]
                cov_factor = self.cov_factor[name]

                if len(self.deviations) > 0:
                    diagonal = torch.ones(cov_factor.shape[0], device=self.device) * 0.5
                    cov_matrix = cov_factor @ cov_factor.T + torch.diag(diagonal)
                    noise = torch.randn(cov_factor.shape[1], device=self.device)
                    sample = mean + (cov_factor @ noise).reshape(param.shape)
                else:
                    sample = mean

                param.data = sample

    def fit(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1) -> None:
        self.base_model.train()
        optimizer = torch.optim.SGD(self.base_model.parameters(), lr=0.01)

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.base_model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    self.collect_deviation()

    def predict_with_uncertainty(
        self, x: Tensor, n_samples: Optional[int] = None
    ) -> Dict[str, Tensor]:
        n_samples = n_samples or self.n_models
        self.base_model.eval()

        outputs = []
        with torch.no_grad():
            for _ in range(n_samples):
                self.sample()
                output = self.base_model(x)
                outputs.append(output)

        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)

        return {
            "mean": mean,
            "variance": variance,
            "std": torch.sqrt(variance),
            "predictions": mean.argmax(dim=-1),
        }


class FlipoutLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_samples: int = 1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_samples = n_samples

        self.weight_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_std = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_std = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias_mean", None)
            self.register_parameter("bias_log_std", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_mean)
        nn.init.constant_(self.weight_log_std, -5.0)
        if self.bias_mean is not None:
            nn.init.zeros_(self.bias_mean)
            nn.init.constant_(self.bias_log_std, -5.0)

    def sample_weights(self) -> Tuple[Tensor, Optional[Tensor]]:
        weight_std = torch.exp(self.weight_log_std)
        weight = self.weight_mean + torch.randn_like(self.weight_mean) * weight_std

        bias = None
        if self.bias_mean is not None:
            bias_std = torch.exp(self.bias_log_std)
            bias = self.bias_mean + torch.randn_like(self.bias_mean) * bias_std

        return weight, bias

    def kl_divergence(self) -> Tensor:
        weight_kl = (
            torch.exp(2 * self.weight_log_std) / 2
            + self.weight_mean**2 / 2
            - self.weight_log_std
            - 0.5
        ).sum()

        if self.bias_mean is not None:
            bias_kl = (
                torch.exp(2 * self.bias_log_std) / 2
                + self.bias_mean**2 / 2
                - self.bias_log_std
                - 0.5
            ).sum()
            return weight_kl + bias_kl

        return weight_kl

    def forward(self, x: Tensor) -> Tensor:
        weight, bias = self.sample_weights()
        return F.linear(x, weight, bias)


class FlipoutConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_mean = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.weight_log_std = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )

        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_channels))
            self.bias_log_std = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias_mean", None)
            self.register_parameter("bias_log_std", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_mean)
        nn.init.constant_(self.weight_log_std, -5.0)
        if self.bias_mean is not None:
            nn.init.zeros_(self.bias_mean)
            nn.init.constant_(self.bias_log_std, -5.0)

    def sample_weights(self) -> Tuple[Tensor, Optional[Tensor]]:
        weight_std = torch.exp(self.weight_log_std)
        weight = self.weight_mean + torch.randn_like(self.weight_mean) * weight_std

        bias = None
        if self.bias_mean is not None:
            bias_std = torch.exp(self.bias_log_std)
            bias = self.bias_mean + torch.randn_like(self.bias_mean) * bias_std

        return weight, bias

    def kl_divergence(self) -> Tensor:
        weight_kl = (
            torch.exp(2 * self.weight_log_std) / 2
            + self.weight_mean**2 / 2
            - self.weight_log_std
            - 0.5
        ).sum()

        if self.bias_mean is not None:
            bias_kl = (
                torch.exp(2 * self.bias_log_std) / 2
                + self.bias_mean**2 / 2
                - self.bias_log_std
                - 0.5
            ).sum()
            return weight_kl + bias_kl

        return weight_kl

    def forward(self, x: Tensor) -> Tensor:
        weight, bias = self.sample_weights()
        return F.conv2d(x, weight, bias, self.stride, self.padding)


class SWAGWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        subspace_dim: int = 20,
        n_devices: int = 1,
    ):
        super().__init__()
        self.model = model
        self.subspace_dim = subspace_dim
        self.n_devices = n_devices

        self._init_swag()

    def _init_swag(self) -> None:
        self.swa_mean: List[Tensor] = []
        self.swa_var: List[Tensor] = []
        self.swa_params: List[Tensor] = []

    def update_swa(self) -> None:
        for param in self.model.parameters():
            self.swa_params.append(param.data.clone())

    def sample_network(self) -> None:
        pass


class BayesianLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.weight_ih_mean = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_ih_log_std = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh_mean = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.weight_hh_log_std = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size)
        )

        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_log_std = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter("bias_mean", None)
            self.register_parameter("bias_log_std", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_ih_mean)
        nn.init.xavier_uniform_(self.weight_hh_mean)
        nn.init.constant_(self.weight_ih_log_std, -5.0)
        nn.init.constant_(self.weight_hh_log_std, -5.0)
        if self.bias_mean is not None:
            nn.init.zeros_(self.bias_mean)
            nn.init.constant_(self.bias_log_std, -5.0)

    def sample_weights(self) -> Tuple[List[Tensor], Optional[Tensor]]:
        weight_ih_std = torch.exp(self.weight_ih_log_std)
        weight_ih = (
            self.weight_ih_mean + torch.randn_like(self.weight_ih_mean) * weight_ih_std
        )

        weight_hh_std = torch.exp(self.weight_hh_log_std)
        weight_hh = (
            self.weight_hh_mean + torch.randn_like(self.weight_hh_mean) * weight_hh_std
        )

        weights = [weight_ih, weight_hh]

        bias = None
        if self.bias_mean is not None:
            bias_std = torch.exp(self.bias_log_std)
            bias = self.bias_mean + torch.randn_like(self.bias_mean) * bias_std

        return weights, bias

    def kl_divergence(self) -> Tensor:
        kl_ih = (
            torch.exp(2 * self.weight_ih_log_std) / 2
            + self.weight_ih_mean**2 / 2
            - self.weight_ih_log_std
            - 0.5
        ).sum()

        kl_hh = (
            torch.exp(2 * self.weight_hh_log_std) / 2
            + self.weight_hh_mean**2 / 2
            - self.weight_hh_log_std
            - 0.5
        ).sum()

        total_kl = kl_ih + kl_hh

        if self.bias_mean is not None:
            kl_bias = (
                torch.exp(2 * self.bias_log_std) / 2
                + self.bias_mean**2 / 2
                - self.bias_log_std
                - 0.5
            ).sum()
            total_kl = total_kl + kl_bias

        return total_kl


def enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def disable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.eval()


def mc_dropout_prediction(
    model: nn.Module,
    x: Tensor,
    n_samples: int = 10,
    dropout_layers: Optional[List[nn.Module]] = None,
) -> Dict[str, Tensor]:
    if dropout_layers is None:
        dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]

    for layer in dropout_layers:
        layer.train()

    model.eval()
    outputs = []

    with torch.no_grad():
        for _ in range(n_samples):
            output = model(x)
            outputs.append(output)

    outputs = torch.stack(outputs)
    mean = outputs.mean(dim=0)
    variance = outputs.var(dim=0)
    std = torch.sqrt(variance)

    probs = F.softmax(mean, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "predictions": mean.argmax(dim=-1),
        "probs": probs,
        "entropy": entropy,
    }
