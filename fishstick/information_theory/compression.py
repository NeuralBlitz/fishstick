"""
Compression-Aware Training Module

Provides tools for compression-aware neural network training:
- Rate-distortion optimization
- Entropy coding estimation
- Bit-allocation optimization
- Quantization-aware training
- Neural compression losses
"""

from typing import Optional, Tuple, List, Dict, Callable
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CompressionMetrics:
    """Container for compression metrics."""

    rate: float
    distortion: float
    rd_loss: float
    entropy: float


class RateDistortionLoss(nn.Module):
    """
    Rate-Distortion loss for learned compression.

    L = D + lambda * R

    where D is distortion and R is rate (bits).
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        distortion: str = "mse",
    ):
        """
        Initialize RD loss.

        Args:
            lambda_: Rate-distortion trade-off parameter
            distortion: Distortion metric ("mse", "ms_ssim", "l1")
        """
        super().__init__()
        self.lambda_ = lambda_
        self.distortion = distortion

    def forward(
        self,
        x: Tensor,
        reconstructed: Tensor,
        likelihoods: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute rate-distortion loss.

        Args:
            x: Original input
            reconstructed: Reconstructed output
            likelihoods: Latent likelihoods for rate estimation

        Returns:
            Dictionary with loss components
        """
        if self.distortion == "mse":
            distortion = F.mse_loss(reconstructed, x)
        elif self.distortion == "l1":
            distortion = F.l1_loss(reconstructed, x)
        else:
            raise ValueError(f"Unknown distortion: {self.distortion}")

        rate = -torch.log(likelihoods + 1e-10).mean() / torch.log(torch.tensor(2.0))

        total_loss = distortion + self.lambda_ * rate

        return {
            "total_loss": total_loss,
            "distortion": distortion,
            "rate": rate,
        }


class EntropyBottleneck(nn.Module):
    """
    Entropy bottleneck for learned compression.

    Models latent distribution with a hyperprior.
    """

    def __init__(
        self,
        latent_dim: int,
        num_channels: int,
    ):
        """
        Initialize entropy bottleneck.

        Args:
            latent_dim: Latent dimension
            num_channels: Number of channels
        """
        super().__init__()

        self.entropy_model = nn.Sequential(
            nn.Conv2d(num_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_channels * 2, 3, padding=1),
        )

        self.quantile_net = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_channels * 2, 3, padding=1),
        )

    def forward(
        self,
        z: Tensor,
        training: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with entropy coding.

        Args:
            z: Latent representation
            training: Whether in training mode

        Returns:
            Tuple of (quantized latent, likelihood)
        """
        params = self.entropy_model(z)
        mean, logscale = params.chunk(2, dim=1)

        logscale = torch.clamp(logscale, min=-10, max=10)

        if training:
            z_noisy = z + torch.randn_like(z) * torch.exp(-logscale)
        else:
            z_noisy = z

        z_quantized = torch.round(z_noisy)

        half = 0.5
        lower = z_quantized - half
        upper = z_quantized + half

        CDF = self._compute_cdf(z_noisy, mean, torch.exp(logscale))

        likelihood = (
            upper * CDF(upper)
            - lower * CDF(lower)
            - (CDF(upper) - CDF(lower)) * z_noisy
        )
        likelihood = torch.clamp(likelihood, min=1e-9)

        return z_quantized, likelihood

    def _compute_cdf(
        self,
        x: Tensor,
        mean: Tensor,
        scale: Tensor,
    ) -> Callable[[Tensor], Tensor]:
        """Compute CDF of the entropy model."""

        def cdf(z):
            return 0.5 * torch.erfc(-(z - mean) / (scale * 1.41421356))

        return cdf


class NeuralCompressionEncoder(nn.Module):
    """
    Neural compression encoder network.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 192,
    ):
        """
        Initialize encoder.

        Args:
            in_channels: Input channels
            latent_channels: Latent space channels
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_channels, 5, stride=2, padding=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode to latent space."""
        return self.encoder(x)


class NeuralCompressionDecoder(nn.Module):
    """
    Neural compression decoder network.
    """

    def __init__(
        self,
        latent_channels: int = 192,
        out_channels: int = 3,
    ):
        """
        Initialize decoder.

        Args:
            latent_channels: Latent space channels
            out_channels: Output channels
        """
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, 256, 5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, out_channels, 5, stride=2, padding=2, output_padding=1
            ),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode from latent space."""
        return self.decoder(z)


class LearnedCompressionModel(nn.Module):
    """
    Full learned compression model with encoder, decoder, and entropy model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 192,
    ):
        """
        Initialize compression model.

        Args:
            in_channels: Input channels
            latent_channels: Latent channels
        """
        super().__init__()

        self.encoder = NeuralCompressionEncoder(in_channels, latent_channels)
        self.entropy_bottleneck = EntropyBottleneck(latent_channels, latent_channels)
        self.decoder = NeuralCompressionDecoder(latent_channels, in_channels)

    def forward(
        self,
        x: Tensor,
        training: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass.

        Args:
            x: Input image
            training: Training mode flag

        Returns:
            Tuple of (reconstruction, metrics)
        """
        z = self.encoder(x)
        z_quantized, likelihood = self.entropy_bottleneck(z, training=training)
        reconstruction = self.decoder(z_quantized)

        metrics = {
            "rate": -torch.log(likelihood + 1e-10).mean()
            / torch.log(torch.tensor(2.0)),
            "distortion": F.mse_loss(reconstruction, x),
        }

        return reconstruction, metrics


class QuantizationAwareTraining:
    """
    Quantization-aware training wrapper.
    """

    def __init__(
        self,
        model: nn.Module,
        num_bits: int = 8,
        ema_decay: float = 0.999,
    ):
        """
        Initialize QAT wrapper.

        Args:
            model: Model to quantize
            num_bits: Number of quantization bits
            ema_decay: EMA decay for learned quantization levels
        """
        self.model = model
        self.num_bits = num_bits
        self.ema_decay = ema_decay

        self._init_quantization_params()

    def _init_quantization_params(self):
        """Initialize learnable quantization parameters."""
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))

    def quantize(self, x: Tensor) -> Tensor:
        """Simulate quantization during forward."""
        scale = torch.sigmoid(self.scale) * 10
        x_scaled = x / scale
        x_quantized = torch.round(x_scaled + self.zero_point)
        x_dequantized = (x_quantized - self.zero_point) * scale
        return x_dequantized

    def forward(self, x: Tensor) -> Tensor:
        """Forward with quantization."""
        x_quantized = self.quantize(x)
        return self.model(x_quantized)


class BitAllocationOptimizer:
    """
    Optimizes bit allocation across network layers.
    """

    def __init__(
        self,
        num_layers: int,
        total_bits: int,
    ):
        """
        Initialize bit allocator.

        Args:
            num_layers: Number of layers to allocate bits
            total_bits: Total bit budget
        """
        self.num_layers = num_layers
        self.total_bits = total_bits

        self.allocations = nn.Parameter(torch.ones(num_layers) / num_layers)

    def get_bit_allocation(self) -> List[int]:
        """Get integer bit allocations."""
        weights = F.softmax(self.allocations, dim=0)
        bits = (weights * self.total_bits).floor().long()

        remaining = self.total_bits - bits.sum().item()
        if remaining > 0:
            bits[weights.argmax()] += remaining

        return bits.tolist()

    def forward(
        self,
        layer_outputs: List[Tensor],
    ) -> Tensor:
        """
        Compute bit allocation loss.

        Args:
            layer_outputs: List of layer outputs

        Returns:
            Allocation loss
        """
        weights = F.softmax(self.allocations, dim=0)

        entropies = []
        for out in layer_outputs:
            out_norm = (out - out.mean()) / (out.std() + 1e-10)
            prob = torch.histc(out_norm, bins=256) / out.numel()
            prob = prob[prob > 0]
            entropy = -(prob * torch.log2(prob + 1e-10)).sum()
            entropies.append(entropy)

        entropies = torch.stack(entropies)

        target = entropies * weights
        return target.sum()


class AdaptiveEntropyCoding:
    """
    Adaptive entropy coding for variable-rate compression.
    """

    def __init__(
        self,
        num_symbols: int = 256,
        context_dim: int = 64,
    ):
        """
        Initialize adaptive entropy coder.

        Args:
            num_symbols: Number of possible symbols
            context_dim: Context embedding dimension
        """
        self.num_symbols = num_symbols

        self.context_net = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_symbols),
        )

        self.symbol_probs = nn.Parameter(torch.ones(num_symbols) / num_symbols)

    def forward(
        self,
        z: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute adaptive coding likelihood.

        Args:
            z: Latent tensor
            context: Context for adaptive coding

        Returns:
            Coding likelihood
        """
        z_flat = z.flatten(start_dim=1)

        if context is not None:
            probs = self.context_net(context)
        else:
            probs = self.symbol_probs.unsqueeze(0).expand(z_flat.shape[0], -1)

        probs = F.softmax(probs, dim=-1)

        indices = z_flat.long().clamp(0, self.num_symbols - 1)
        likelihood = probs.gather(1, indices)

        return likelihood


class CompressionRegularizer(nn.Module):
    """
    Regularizer for compression-aware training.

    Encourages sparse, compressible representations.
    """

    def __init__(
        self,
        sparsity_weight: float = 0.01,
        correlation_weight: float = 0.01,
    ):
        """
        Initialize compression regularizer.

        Args:
            sparsity_weight: Weight for sparsity penalty
            correlation_weight: Weight for correlation penalty
        """
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.correlation_weight = correlation_weight

    def forward(self, z: Tensor) -> Dict[str, Tensor]:
        """
        Compute compression regularizer.

        Args:
            z: Latent representation

        Returns:
            Dictionary with regularizer components
        """
        sparsity = torch.abs(z).mean()

        z_centered = z - z.mean(dim=0, keepdim=True)
        corr = torch.mm(z_centered.T, z_centered) / (z.shape[0] - 1)

        off_diag = corr * (1 - torch.eye(corr.shape[0], device=corr.device))
        correlation = off_diag.pow(2).sum() / corr.shape[0]

        return {
            "sparsity": sparsity,
            "correlation": correlation,
            "total": self.sparsity_weight * sparsity
            + self.correlation_weight * correlation,
        }


class RateDistortionScheduler:
    """
    Scheduler for rate-distortion lambda.
    """

    def __init__(
        self,
        initial_lambda: float = 1.0,
        final_lambda: float = 0.01,
        schedule: str = "exponential",
    ):
        """
        Initialize RD scheduler.

        Args:
            initial_lambda: Starting lambda
            final_lambda: Final lambda
            schedule: Schedule type
        """
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.schedule = schedule
        self.current_step = 0

    def step(self):
        """Step the scheduler."""
        self.current_step += 1

    def get_lambda(self) -> float:
        """Get current lambda value."""
        progress = self.current_step

        if self.schedule == "exponential":
            factor = progress / (progress + 1)
            return (
                self.initial_lambda
                * (self.final_lambda / self.initial_lambda) ** factor
            )
        elif self.schedule == "linear":
            return self.initial_lambda - (
                self.initial_lambda - self.final_lambda
            ) * min(1.0, progress / 1000)
        else:
            return self.initial_lambda


class SSIMLoss(nn.Module):
    """
    SSIM loss for image compression.
    """

    def __init__(
        self,
        window_size: int = 11,
        channel: int = 3,
    ):
        """
        Initialize SSIM loss.

        Args:
            window_size: Size of SSIM window
            channel: Number of channels
        """
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)

    def _create_window(self, window_size: int, channel: int) -> Tensor:
        """Create Gaussian window."""

        def gaussian(window_size, sigma):
            gauss = torch.tensor(
                [
                    torch.exp(
                        torch.tensor(
                            -((x - window_size // 2) ** 2) / float(2 * sigma**2)
                        )
                    )
                    for x in range(window_size)
                ]
            )
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute 1 - SSIM."""
        window = self.window.to(x.device)

        C1 = 0.01**2
        C2 = 0.03**2

        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=self.channel)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=self.channel)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.conv2d(x * x, window, padding=self.window_size // 2, groups=self.channel)
            - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(y * y, window, padding=self.window_size // 2, groups=self.channel)
            - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(x * y, window, padding=self.window_size // 2, groups=self.channel)
            - mu_xy
        )

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )

        return 1 - ssim_map.mean()
