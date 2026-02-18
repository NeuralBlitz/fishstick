"""
Bayesian Neural Network Base Classes.

Provides base classes for building Bayesian neural networks
with various inference methods and uncertainty estimation.
"""

from typing import Optional, Tuple, Dict, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Normal, Laplace


class BayesianModule(nn.Module):
    """Base class for Bayesian neural network modules.

    Provides common interface for computing KL divergence
    and sampling from posterior.
    """

    def __init__(self):
        super().__init__()

    def kl_divergence(self) -> Tensor:
        """Compute KL divergence between posterior and prior.

        Should be implemented by subclasses.
        """
        raise NotImplementedError

    def sample(self, n_samples: int = 1) -> Tensor:
        """Sample from the posterior distribution.

        Should be implemented by subclasses.
        """
        raise NotImplementedError


class BayesianConv2d(BayesianModule):
    """Bayesian Convolutional 2D layer with variational inference.

    Implements a convolutional layer with distributions over weights
    using the reparameterization trick.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel
        stride: Stride of convolution
        padding: Padding added to input
        prior_sigma: Standard deviation of prior
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.weight_log_sigma = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )

        self.bias_mu = nn.Parameter(torch.empty(out_channels))
        self.bias_log_sigma = nn.Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, a=0, mode="fan_in")
        nn.init.uniform_(self.weight_log_sigma, -5, -3)
        nn.init.zeros_(self.bias_mu)
        nn.init.uniform_(self.bias_log_sigma, -5, -3)

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

        return F.conv2d(x, weight, bias, self.stride, self.padding)

    def kl_divergence(self) -> Tensor:
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        kl = (
            torch.log(self.prior_sigma)
            - self.weight_log_sigma
            + 0.5 * (weight_sigma**2 + self.weight_mu**2) / (self.prior_sigma**2)
            - 0.5
        ).sum() + (
            torch.log(self.prior_sigma)
            - self.bias_log_sigma
            + 0.5 * (bias_sigma**2 + self.bias_mu**2) / (self.prior_sigma**2)
            - 0.5
        ).sum()

        return kl


class BayesianLinear(BayesianModule):
    """Bayesian Linear layer with variational inference.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        prior_sigma: Standard deviation of prior
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_log_sigma = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, a=0, mode="fan_in")
        nn.init.uniform_(self.weight_log_sigma, -5, -3)
        nn.init.zeros_(self.bias_mu)
        nn.init.uniform_(self.bias_log_sigma, -5, -3)

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

    def kl_divergence(self) -> Tensor:
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        kl = (
            torch.log(self.prior_sigma)
            - self.weight_log_sigma
            + 0.5 * (weight_sigma**2 + self.weight_mu**2) / (self.prior_sigma**2)
            - 0.5
        ).sum() + (
            torch.log(self.prior_sigma)
            - self.bias_log_sigma
            + 0.5 * (bias_sigma**2 + self.bias_mu**2) / (self.prior_sigma**2)
            - 0.5
        ).sum()

        return kl


class BayesianSequential(nn.Module):
    """Sequential container for Bayesian layers.

    Args:
        *args: Bayesian modules
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x

    def kl_divergence(self) -> Tensor:
        kl = 0.0
        for module in self._modules.values():
            if hasattr(module, "kl_divergence"):
                kl = kl + module.kl_divergence()
        return kl


class BayesianNeuralNetwork(nn.Module):
    """Complete Bayesian Neural Network.

    A fully Bayesian neural network with all variational layers
    and automatic KL computation.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        prior_sigma: Prior standard deviation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        prior_sigma: float = 1.0,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_sigma))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(BayesianLinear(prev_dim, output_dim, prior_sigma))

        self.network = nn.Sequential(*layers)
        self.prior_sigma = prior_sigma

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def kl_divergence(self) -> Tensor:
        kl = 0.0
        for module in self.network.modules():
            if hasattr(module, "kl_divergence"):
                kl = kl + module.kl_divergence()
        return kl

    def predict(
        self,
        x: Tensor,
        n_samples: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        """Make predictions with uncertainty.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            mean: Mean prediction
            variance: Prediction variance
        """
        self.train()

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(self.network(x))

        samples = torch.stack(samples)
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)

        return mean, variance


class LaplaceApproximation(nn.Module):
    """Laplace Approximation for Bayesian neural networks.

    Uses Laplace approximation to approximate the posterior
    as a Gaussian centered at the MAP estimate.

    Args:
        base_model: Neural network to approximate
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.precision: Optional[Tensor] = None
        self.mean: Dict[str, Tensor] = {}

    def fit(self, data_loader, alpha: float = 1.0):
        """Fit Laplace approximation to data.

        Args:
            data_loader: Training data
            alpha: Regularization strength
        """
        self.base_model.eval()

        params = dict(self.base_model.named_parameters())
        self.mean = {k: v.clone() for k, v in params.items()}

        hessian = self._compute_hessian(data_loader, alpha)

        try:
            self.precision = hessian + alpha * torch.eye(
                sum(p.numel() for p in params.values())
            )
        except RuntimeError:
            self.precision = hessian + alpha * torch.eye(hessian.size(0))

    def _compute_hessian(
        self,
        data_loader,
        alpha: float,
    ) -> Tensor:
        """Compute Hessian of the loss."""
        params = dict(self.base_model.named_parameters())

        loss = 0.0
        for x, y in data_loader:
            output = self.base_model(x)
            loss = loss + F.cross_entropy(output, y)

        grads = torch.autograd.grad(loss, params.values(), create_graph=True)
        grads = torch.cat([g.flatten() for g in grads])

        hessian = torch.zeros(grads.numel(), grads.numel())

        for i, g in enumerate(grads):
            hessian[:, i] = torch.autograd.grad(g, params.values(), retain_graph=True)[
                0
            ].flatten()

        return hessian

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict with uncertainty from Laplace approximation."""
        self.base_model.eval()

        with torch.no_grad():
            mean_pred = self.base_model(x)

        if self.precision is not None:
            variance = torch.diag(
                torch.linalg.inv(self.precision)
            ).sum() * torch.ones_like(mean_pred)
        else:
            variance = torch.ones_like(mean_pred)

        return mean_pred, variance


class SWAG(nn.Module):
    """Stochastic Weight Averaging Gaussian (SWAG).

    Collects models during training and approximates posterior
    using a Gaussian distribution over collected weights.

    Args:
        base_model: Model to approximate
        num_samples: Number of weight samples to store
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_samples: int = 20,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_samples = num_samples

        self.weight_samples: List[Dict[str, Tensor]] = []
        self.mean: Optional[Dict[str, Tensor]] = None

    def collect_weights(self):
        """Collect current model weights."""
        state = {k: v.clone() for k, v in self.base_model.state_dict().items()}

        if len(self.weight_samples) >= self.num_samples:
            self.weight_samples.pop(0)

        self.weight_samples.append(state)

        if self.mean is None:
            self.mean = {k: v.clone() for k, v in state.items()}
        else:
            for k in self.mean:
                self.mean[k] = self.mean[k] + (state[k] - self.mean[k]) / len(
                    self.weight_samples
                )

    def predict(self, x: Tensor, n_samples: int = 10) -> Tuple[Tensor, Tensor]:
        """Predict with SWAG uncertainty."""
        if not self.weight_samples:
            raise RuntimeError("No weight samples collected")

        samples = []
        indices = np.random.choice(
            len(self.weight_samples),
            min(n_samples, len(self.weight_samples)),
            replace=False,
        )

        for idx in indices:
            state = self.weight_samples[idx]
            self.base_model.load_state_dict(state)
            self.base_model.eval()

            with torch.no_grad():
                samples.append(self.base_model(x))

        self.base_model.load_state_dict(self.weight_samples[-1])

        samples = torch.stack(samples)
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)

        return mean, variance


class RadfordNeal(nn.Module):
    """Radford Neal's Hamiltonian Monte Carlo for BNNs.

    Uses HMC to sample from the posterior distribution.

    Args:
        base_model: Model to sample from
        n_samples: Number of samples to collect
        burn_in: Number of burn-in steps
        leapfrog_steps: Number of leapfrog steps
        step_size: Leapfrog step size
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_samples: int = 100,
        burn_in: int = 50,
        leapfrog_steps: int = 10,
        step_size: float = 0.01,
    ):
        super().__init__()
        self.base_model = base_model
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.leapfrog_steps = leapfrog_steps
        self.step_size = step_size

    def sample(self, data_loader) -> List[Dict[str, Tensor]]:
        """Sample from posterior using HMC."""
        samples = []

        params = {k: v.clone() for k, v in self.base_model.named_parameters()}
        momentum = {k: torch.randn_like(v) for k, v in params.items()}

        current_energy = self._energy(params, data_loader)

        for _ in range(self.burn_in + self.n_samples):
            params, current_energy = self._hmc_step(params, momentum, data_loader)

            if _ >= self.burn_in:
                samples.append({k: v.clone() for k, v in params.items()})

        return samples

    def _energy(self, params: Dict[str, Tensor], data_loader) -> float:
        """Compute Hamiltonian energy."""
        self.base_model.load_state_dict(params)

        log_likelihood = 0.0
        for x, y in data_loader:
            output = self.base_model(x)
            log_likelihood = log_likelihood + F.cross_entropy(output, y)

        log_prior = sum(p.pow(2).sum() for p in params.values()) / 2

        return -log_likelihood + log_prior

    def _hmc_step(
        self,
        params: Dict[str, Tensor],
        momentum: Dict[str, Tensor],
        data_loader,
    ) -> Tuple[Dict[str, Tensor], float]:
        """Perform one HMC step."""
        current_params = {k: v.clone() for k, v in params.items()}
        current_momentum = {k: v.clone() for k, v in momentum.items()}

        for p in current_momentum.values():
            p.normal_()

        current_energy = self._energy(current_params, data_loader)

        for _ in range(self.leapfrog_steps):
            grads = torch.autograd.grad(
                current_energy,
                current_params.values(),
                create_graph=True,
                retain_graph=True,
            )

            for k, g in zip(current_params.keys(), grads):
                current_momentum[k] = current_momentum[k] - 0.5 * self.step_size * g

            for k in current_params:
                current_params[k] = (
                    current_params[k] + self.step_size * current_momentum[k]
                )

            energy = self._energy(current_params, data_loader)
            grads = torch.autograd.grad(
                energy, current_params.values(), retain_graph=True
            )

            for k, g in zip(current_params.keys(), grads):
                current_momentum[k] = current_momentum[k] - 0.5 * self.step_size * g

        proposed_energy = self._energy(current_params, data_loader)

        delta_energy = proposed_energy - current_energy
        delta_momentum = sum(m.pow(2).sum() for m in current_momentum.values()) - sum(
            m.pow(2).sum() for m in momentum.values()
        )

        if torch.rand(1).log() < -0.5 * (delta_energy + delta_momentum):
            return current_params, proposed_energy

        return params, current_energy


from collections import OrderedDict
from typing import List
import numpy as np
