"""Neural coding implementations: rate, temporal, population, and mixed coding."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RateEncoder(nn.Module):
    """Rate-based neural encoder.

    Converts input signals to firing rates using a non-linear transfer function.

    Args:
        input_dim: Input dimension
        output_dim: Number of neurons
        nonlin: Activation function ('relu', 'sigmoid', 'tanh', 'exp')
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nonlin: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = nn.Linear(input_dim, output_dim)

        self.nonlin = nonlin

    def forward(self, x: Tensor) -> Tensor:
        """Encode input as firing rates.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Firing rates (batch, output_dim)
        """
        rates = self.encoder(x)

        if self.nonlin == "relu":
            rates = F.relu(rates)
        elif self.nonlin == "sigmoid":
            rates = torch.sigmoid(rates)
        elif self.nonlin == "tanh":
            rates = torch.tanh(rates)
        elif self.nonlin == "exp":
            rates = torch.exp(torch.clamp(rates, max=10))

        return rates


class PoissonEncoder(nn.Module):
    """Poisson spike train encoder.

    Generates spike trains with firing rates following a Poisson process.

    Args:
        input_dim: Input dimension
        output_dim: Number of neurons
        dt: Time step (ms)
        tau_ref: Refractory period (ms)
        max_rate: Maximum firing rate (Hz)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dt: float = 1.0,
        tau_ref: float = 5.0,
        max_rate: float = 1000.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dt = dt
        self.tau_ref = tau_ref
        self.max_rate = max_rate

        self.encoder = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        x: Tensor,
        n_steps: int = 100,
    ) -> Tensor:
        """Generate Poisson spike trains.

        Args:
            x: Input tensor (batch, input_dim)
            n_steps: Number of simulation steps

        Returns:
            Spike trains (batch, n_steps, output_dim)
        """
        rates = self.encoder(x)
        rates = torch.clamp(rates, min=0, max=self.max_rate)

        spike_probs = rates * (self.dt / 1000.0)

        rand_vals = torch.rand(x.shape[0], n_steps, self.output_dim, device=x.device)

        spikes = (rand_vals < spike_probs.unsqueeze(1)).float()

        return spikes


class TemporalEncoder(nn.Module):
    """Temporal coding encoder using latency codes.

    Encodes information in the timing of spikes relative to a reference signal.

    Args:
        input_dim: Input dimension
        output_dim: Number of neurons
        n_bins: Number of time bins
        time_constant: Neural time constant
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_bins: int = 50,
        time_constant: float = 10.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_bins = n_bins
        self.time_constant = time_constant

        self.encoder = nn.Linear(input_dim, output_dim)

        times = torch.linspace(0, n_bins - 1, n_bins)
        self.register_buffer("time_points", times)

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Encode as temporal spike latencies.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Tuple of (latencies, spikes)
            - latencies: Spike times (batch, output_dim)
            - spikes: One-hot spike trains (batch, n_bins, output_dim)
        """
        rates = self.encoder(x)

        spike_latencies = self.n_bins * torch.exp(-rates / self.time_constant)

        spike_latencies = torch.clamp(spike_latencies, 1, self.n_bins - 1).long()

        batch_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        spikes = torch.zeros(x.shape[0], self.n_bins, self.output_dim, device=x.device)
        spikes[
            batch_idx, spike_latencies, torch.arange(self.output_dim, device=x.device)
        ] = 1.0

        return spike_latencies, spikes


class PopulationEncoder(nn.Module):
    """Population coding encoder.

    Uses a population of neurons with overlapping tuning curves to encode values.

    Args:
        input_dim: Input dimension
        n_neurons: Number of neurons per dimension
        sigma: Width of Gaussian tuning curves
    """

    def __init__(
        self,
        input_dim: int,
        n_neurons: int = 20,
        sigma: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.sigma = sigma

        neuron_centers = torch.linspace(0, 1, n_neurons)
        self.register_buffer("centers", neuron_centers)

    def forward(self, x: Tensor) -> Tensor:
        """Encode using population response.

        Args:
            x: Input values (batch, input_dim) in [0, 1]

        Returns:
            Population response (batch, n_neurons)
        """
        x_expanded = x.unsqueeze(-1)
        centers_expanded = self.centers.view(1, 1, -1)

        distances = (x_expanded - centers_expanded) ** 2

        response = torch.exp(-distances / (2 * self.sigma**2))

        if self.input_dim > 1:
            response = response.prod(dim=1)

        return response


class MixedCodeEncoder(nn.Module):
    """Mixed rate and temporal coding.

    Combines rate and temporal coding for robust information transmission.

    Args:
        input_dim: Input dimension
        output_dim: Number of neurons
        n_bins: Number of time bins
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_bins: int = 20,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_bins = n_bins

        self.rate_encoder = nn.Linear(input_dim, output_dim)

        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Mixed encoding.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Tuple of (rate_code, temporal_code)
        """
        rate_code = torch.sigmoid(self.rate_encoder(x))

        temporal_weights = self.temporal_encoder(x)
        temporal_code = torch.softmax(temporal_weights, dim=-1)

        return rate_code, temporal_code


class LatentPopulationEncoder(nn.Module):
    """Latent population coding with learnable tuning curves.

    Learns optimal tuning curves for encoding from data.

    Args:
        input_dim: Input dimension
        n_neurons: Number of neurons
        latent_dim: Latent encoding dimension
    """

    def __init__(
        self,
        input_dim: int,
        n_neurons: int,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.ReLU(),
        )

        self.tuning_curves = nn.Parameter(torch.randn(n_neurons, latent_dim) * 0.5)

        self.baseline = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, x: Tensor) -> Tensor:
        """Encode with learnable tuning.

        Args:
            x: Input tensor

        Returns:
            Population response
        """
        z = self.encoder(x)

        response = torch.matmul(z, self.tuning_curves.t()) + self.baseline

        response = F.softplus(response)

        return response


class DeltaEncoder(nn.Module):
    """Delta modulation encoder.

    Encodes changes (deltas) rather than absolute values, efficient for
    slowly varying signals.

    Args:
        input_dim: Input dimension
        output_dim: Number of neurons
        threshold: Minimum change to encode
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        threshold: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.threshold = threshold

        self.prev_input = None

    def forward(self, x: Tensor) -> Tensor:
        """Encode as changes from previous input.

        Args:
            x: Input tensor

        Returns:
            Delta-encoded values
        """
        if self.prev_input is None or not self.training:
            delta = x
        else:
            delta = x - self.prev_input

            delta = torch.where(
                delta.abs() > self.threshold, delta, torch.zeros_like(delta)
            )

        self.prev_input = x.detach()

        return delta


class GridCellEncoder(nn.Module):
    """Grid cell-inspired encoder.

    Encodes position using multiple spatial scales similar to grid cells
    in the entorhinal cortex.

    Args:
        input_dim: Input dimension (2 for x, y position)
        n_modules: Number of grid cell modules
        n_neurons_per_module: Neurons per module
    """

    def __init__(
        self,
        input_dim: int = 2,
        n_modules: int = 4,
        n_neurons_per_module: int = 6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_modules = n_modules
        self.n_neurons_per_module = n_neurons_per_module

        self.scales = nn.Parameter(torch.tensor([1.0, 2.0, 4.0, 8.0])[:n_modules])

        self.phases = nn.Parameter(
            torch.rand(n_modules, n_neurons_per_module, input_dim) * 2 * 3.14159
        )

    def forward(self, position: Tensor) -> Tensor:
        """Encode position as grid cell responses.

        Args:
            position: Position tensor (batch, 2) in [0, 1] range

        Returns:
            Grid cell responses (batch, n_modules * n_neurons_per_module)
        """
        batch_size = position.shape[0]

        responses = []

        for m in range(self.n_modules):
            scale = self.scales[m]
            phase = self.phases[m]

            pos_scaled = position.unsqueeze(1) * scale * 2 * 3.14159

            grid_pattern = torch.cos(pos_scaled.unsqueeze(2) - phase.unsqueeze(0))

            response = grid_pattern.prod(dim=-1)

            responses.append(response.reshape(batch_size, -1))

        return torch.cat(responses, dim=-1)
