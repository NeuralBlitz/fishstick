"""
Channel Capacity Estimators Module

Provides tools for estimating channel capacity and information rates:
- Channel capacity bounds
- Rate-distortion theory
- Information rate estimation
- GSN (Gradient-based Similarity Network) capacity
- Neural channel estimation
"""

from typing import Optional, Tuple, Dict, Callable, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np


@dataclass
class ChannelMetrics:
    """Container for channel capacity metrics."""

    capacity: float
    mutual_information: float
    entropy_x: float
    entropy_y: float


class ChannelCapacityEstimator:
    """
    Estimates channel capacity using various methods.

    For a channel p(y|x), estimates max I(X;Y) over input distributions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_samples: int = 1000,
    ):
        """
        Initialize capacity estimator.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_samples: Number of samples for estimation
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_samples = num_samples

    def estimate_gaussian_channel(
        self,
        channel_matrix: Tensor,
        noise_std: float = 0.1,
    ) -> float:
        """
        Estimate capacity of Gaussian channel.

        C = sum(log2(1 + SNR_i)) bits

        Args:
            channel_matrix: Channel transition matrix
            noise_std: Noise standard deviation

        Returns:
            Channel capacity in bits
        """
        channel_matrix = channel_matrix.to(
            noise_std.device if hasattr(noise_std, "device") else "cpu"
        )

        svd_vals = torch.linalg.svdvals(channel_matrix)
        snr = svd_vals**2 / (noise_std**2)

        capacity = torch.sum(torch.log2(1 + snr)).item()

        return capacity

    def estimate_discrete_channel(
        self,
        transition_matrix: Tensor,
        num_iterations: int = 100,
    ) -> float:
        """
        Estimate capacity of discrete channel using iterative method.

        Args:
            transition_matrix: p(y|x) transition matrix
            num_iterations: Number of iterations

        Returns:
            Channel capacity in bits
        """
        num_inputs = transition_matrix.shape[0]

        p_x = torch.ones(num_inputs) / num_inputs

        for _ in range(num_iterations):
            p_y = p_x @ transition_matrix

            mutual_info = 0.0
            for x in range(num_inputs):
                for y in range(transition_matrix.shape[1]):
                    if p_x[x] > 0 and transition_matrix[x, y] > 0 and p_y[y] > 0:
                        p_x_y = transition_matrix[x, y] * p_x[x] / p_y[y]
                        mutual_info += (
                            p_x[x] * transition_matrix[x, y] * torch.log2(p_x_y + 1e-10)
                        )

            gradient = mutual_info / p_x
            p_x = p_x * torch.exp(gradient)
            p_x = p_x / p_x.sum()

        return max(0, mutual_info.item())

    def estimate_neural_channel(
        self,
        channel_network: nn.Module,
        input_distribution: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Estimate capacity of neural channel.

        Args:
            channel_network: Neural network modeling the channel
            input_distribution: Input distribution (uniform if None)

        Returns:
            Estimated capacity
        """
        if input_distribution is None:
            input_distribution = torch.ones(self.input_dim) / self.input_dim

        inputs = torch.multinomial(
            input_distribution, self.num_samples, replacement=True
        )
        inputs_one_hot = F.one_hot(inputs, self.input_dim).float()

        with torch.no_grad():
            outputs = channel_network(inputs_one_hot)

        from .mutual_info import knn_mi_estimator

        mi = knn_mi_estimator(inputs_one_hot.float(), outputs)

        return mi


class InformationRateEstimator:
    """
    Estimates information rate for sequential data.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
    ):
        """
        Initialize rate estimator.

        Args:
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension
        """
        self.embedding_dim = embedding_dim

        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.rate_estimator = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: Tensor) -> Tensor:
        """
        Estimate information rate of sequence.

        Args:
            sequence: Input sequence (batch, seq_len, dim)

        Returns:
            Estimated rate
        """
        rnn_out, _ = self.rnn(sequence)

        rates = self.rate_estimator(rnn_out)

        return rates.mean()


class RateDistortionCurve:
    """
    Computes rate-distortion curve for a source.
    """

    def __init__(
        self,
        source_distribution: Tensor,
        distortion_measure: str = "hamming",
    ):
        """
        Initialize RD curve computer.

        Args:
            source_distribution: P(X) source distribution
            distortion_measure: Distortion measure
        """
        self.source_distribution = source_distribution
        self.distortion_measure = distortion_measure

    def compute_curve(
        self,
        num_points: int = 20,
        max_distortion: float = 1.0,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute rate-distortion curve.

        Args:
            num_points: Number of points on curve
            max_distortion: Maximum distortion level

        Returns:
            Tuple of (rates, distortions)
        """
        rates = []
        distortions = []

        for i in range(num_points):
            target_distortion = max_distortion * (i + 1) / num_points

            rate = self._compute_rate_for_distortion(target_distortion)
            distortions.append(target_distortion)
            rates.append(rate)

        return rates, distortions

    def _compute_rate_for_distortion(self, distortion: float) -> float:
        """Compute rate for given distortion level."""
        source_entropy = -(
            self.source_distribution * torch.log2(self.source_distribution + 1e-10)
        ).sum()

        rate = max(0, source_entropy.item() - distortion * 10)

        return rate


class BlahutArimoto:
    """
    Blahut-Arigo algorithm for channel capacity computation.
    """

    def __init__(
        self,
        channel_matrix: Tensor,
        num_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        """
        Initialize Blahut-Arimoto algorithm.

        Args:
            channel_matrix: Transition probabilities p(y|x)
            num_iterations: Maximum iterations
            tolerance: Convergence tolerance
        """
        self.channel_matrix = channel_matrix
        self.num_iterations = num_iterations
        self.tolerance = tolerance

    def compute_capacity(self) -> Tuple[float, Tensor]:
        """
        Compute channel capacity.

        Returns:
            Tuple of (capacity, optimal_input_distribution)
        """
        num_inputs = self.channel_matrix.shape[0]
        num_outputs = self.channel_matrix.shape[1]

        p_x = torch.ones(num_inputs) / num_inputs

        for iteration in range(self.num_iterations):
            p_y = p_x @ self.channel_matrix

            q = self.channel_matrix / (p_y.unsqueeze(0) + 1e-10)
            q = q * p_x.unsqueeze(1)
            q = q / (q.sum(dim=1, keepdim=True) + 1e-10)

            p_new = torch.exp(
                torch.sum(self.channel_matrix * torch.log(q + 1e-10), dim=1)
            )
            p_new = p_new / p_new.sum()

            diff = torch.max(torch.abs(p_new - p_x))
            p_x = p_new

            if diff < self.tolerance:
                break

        p_y = p_x @ self.channel_matrix

        capacity = 0.0
        for x in range(num_inputs):
            for y in range(num_outputs):
                if p_x[x] > 0 and self.channel_matrix[x, y] > 0 and p_y[y] > 0:
                    p_x_y = self.channel_matrix[x, y] * p_x[x] / p_y[y]
                    capacity += (
                        p_x[x] * self.channel_matrix[x, y] * torch.log2(p_x_y + 1e-10)
                    )

        return max(0, capacity.item()), p_x


class NeuralChannel(nn.Module):
    """
    Learnable neural channel model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        noise_std: float = 0.1,
    ):
        """
        Initialize neural channel.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden dimension
            noise_std: Channel noise standard deviation
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_std = noise_std

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.channel = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Pass through noisy channel.

        Args:
            x: Input tensor

        Returns:
            Noisy output
        """
        h = self.encoder(x)
        h = self.channel(h)

        noise = torch.randn_like(h) * self.noise_std
        y = h + noise

        return y


class ChannelCapacityLoss(nn.Module):
    """
    Loss for training neural channel to maximize capacity.
    """

    def __init__(
        self,
        target_rate: float = 1.0,
        beta: float = 0.1,
    ):
        """
        Initialize capacity loss.

        Args:
            target_rate: Target information rate
            beta: Regularization strength
        """
        super().__init__()
        self.target_rate = target_rate
        self.beta = beta

    def forward(
        self,
        inputs: Tensor,
        outputs: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute capacity loss.

        Args:
            inputs: Channel inputs
            outputs: Channel outputs

        Returns:
            Loss dictionary
        """
        from .mutual_info import knn_mi_estimator

        mi = knn_mi_estimator(inputs, outputs)

        rate_loss = torch.abs(mi - self.target_rate)

        input_entropy = -(
            inputs.mean(dim=0) * torch.log(inputs.mean(dim=0) + 1e-10)
        ).sum()
        output_entropy = -(
            outputs.mean(dim=0) * torch.log(outputs.mean(dim=0) + 1e-10)
        ).sum()

        capacity_loss = -mi + self.beta * (input_entropy + output_entropy)

        return {
            "total_loss": rate_loss + capacity_loss,
            "mi": mi,
            "rate_loss": rate_loss,
            "capacity_loss": capacity_loss,
        }


class InformationBottleneckChannel(nn.Module):
    """
    Channel with information bottleneck regularization.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        beta: float = 1.0,
    ):
        """
        Initialize IB channel.

        Args:
            input_dim: Input dimension
            bottleneck_dim: Bottleneck dimension
            output_dim: Output dimension
            beta: IB parameter
        """
        super().__init__()

        self.bottleneck_dim = bottleneck_dim
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through IB channel.

        Args:
            x: Input tensor

        Returns:
            Tuple of (output, metrics)
        """
        params = self.encoder(x)
        mean, logvar = params.chunk(2, dim=-1)

        std = torch.exp(0.5 * logvar)
        z = mean + torch.randn_like(std) * std

        y = self.decoder(z)

        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean()

        metrics = {
            "kl_divergence": kl,
            "bottleneck_mean": z.abs().mean(),
            "output": y,
        }

        return y, metrics


class GSNCapacity(nn.Module):
    """
    Gradient-based Similarity Network capacity estimator.

    Measures information capacity via gradient-based similarity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
    ):
        """
        Initialize GSN capacity estimator.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def compute_capacity(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Compute gradient-based similarity capacity.

        Args:
            x: First input
            y: Second input

        Returns:
            Estimated capacity
        """
        x.requires_grad_(True)

        f_x = self.network(x)
        grad = torch.autograd.grad(f_x.sum(), x, create_graph=True, retain_graph=True)[
            0
        ]

        similarity = (grad * grad).sum(dim=-1).mean()

        return similarity


class TransferEntropyEstimator:
    """
    Estimates transfer entropy (directed information) between time series.

    TE(X->Y) = I(X_t; Y_{t+k} | Y_t)
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        k: int = 1,
    ):
        """
        Initialize transfer entropy estimator.

        Args:
            embedding_dim: Embedding dimension for state space reconstruction
            k: Time lag
        """
        self.embedding_dim = embedding_dim
        self.k = k

    def estimate(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Estimate transfer entropy from X to Y.

        Args:
            x: Source time series
            y: Target time series

        Returns:
            Transfer entropy estimate
        """
        from .mutual_info import conditional_mutual_information

        x = x.detach()
        y = y.detach()

        x_history = []
        y_history = []

        for i in range(self.embedding_dim, len(x) - self.k):
            x_history.append(x[i - self.embedding_dim : i])
            y_history.append(y[i - self.embedding_dim : i])

        x_history = torch.stack(x_history)
        y_history = torch.stack(y_history)

        x_future = x[self.embedding_dim + self.k :]
        y_current = y[self.embedding_dim : -self.k]

        te = conditional_mutual_information(
            x_future.unsqueeze(1),
            y_current.unsqueeze(1),
            torch.cat([x_history, y_history], dim=-1),
            method="knn",
        )

        return te


def compute_channel_snr(
    signal_power: float,
    noise_power: float,
) -> float:
    """
    Compute signal-to-noise ratio.

    Args:
        signal_power: Signal power
        noise_power: Noise power

    Returns:
        SNR in dB
    """
    snr_linear = signal_power / (noise_power + 1e-10)
    return 10 * np.log10(snr_linear)


def compute_shannon_limit(
    bandwidth: float,
    snr: float,
) -> float:
    """
    Compute Shannon channel capacity limit.

    C = B * log2(1 + SNR)

    Args:
        bandwidth: Channel bandwidth (Hz)
        snr: Signal-to-noise ratio (linear)

    Returns:
        Capacity in bits/second
    """
    return bandwidth * np.log2(1 + snr)


def mutual_information_rate(
    x: Tensor,
    y: Tensor,
    window_size: int = 100,
) -> Tensor:
    """
    Compute mutual information rate for sequences.

    Args:
        x: First sequence
        y: Second sequence
        window_size: Window for rate computation

    Returns:
        MI rate
    """
    from .mutual_info import knn_mi_estimator

    if len(x) < window_size:
        window_size = len(x)

    rates = []
    for i in range(0, len(x) - window_size, window_size // 2):
        x_window = x[i : i + window_size]
        y_window = y[i : i + window_size]

        mi = knn_mi_estimator(x_window, y_window)
        rates.append(mi.item())

    return torch.tensor(np.mean(rates))
