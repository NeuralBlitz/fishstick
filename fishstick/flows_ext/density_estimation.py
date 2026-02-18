"""
Flow-based Density Estimation.

Implements various normalizing flow architectures for density estimation:
- RealNVP-style flows
- Glow-style flows
- Flow-based VAE integration
- Invertible networks

References:
- Dinh et al. (2017) "RealNVP: Real-valued Non-Volume Preserving Flows"
- Kingma & Dhariwal (2018) "Glow: Generative Flow with Invertible 1x1 Convolutions"
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class RealNVP(nn.Module):
    """
    RealNVP: Real-valued Non-Volume Preserving flows.

    Implements affine coupling layers with multi-scale architecture.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_layers: Number of coupling layers
        num_scales: Number of multi-scale levels
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_scales: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_scales = num_scales

        self.layers = nn.ModuleList()
        self.scalers = nn.ModuleList()

        for scale in range(num_scales):
            scale_layers = nn.ModuleList()
            scale_dim = input_dim // (2**scale)

            for i in range(num_layers // num_scales):
                mask = self._create_mask(scale_dim, i)

                scale_layers.append(
                    AffineCouplingLayer(
                        input_dim=scale_dim,
                        hidden_dim=hidden_dim,
                        mask=mask,
                    )
                )

            self.layers.append(scale_layers)

            if scale < num_scales - 1:
                self.scalers.append(SqueezeLayer(scale_dim * (2**scale)))

    def _create_mask(self, dim: int, layer_idx: int) -> Tensor:
        """Create checkerboard or channel mask."""
        mask = torch.zeros(dim)
        half = dim // 2

        if layer_idx % 2 == 0:
            mask[:half] = 1.0
        else:
            mask[half:] = 1.0

        return mask

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply RealNVP flow."""
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        log_det = torch.zeros(x.shape[0], device=x.device)

        for scale in range(self.num_scales):
            for layer in self.layers[scale]:
                x, ld = layer(x)
                log_det = log_det + ld

            if scale < self.num_scales - 1:
                x, ld = self.scalers[scale](x)
                log_det = log_det + ld

        return x, log_det

    def _inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        log_det = torch.zeros(z.shape[0], device=z.device)

        for scale in range(self.num_scales - 1, -1, -1):
            if scale > 0:
                z, ld = self.scalers[scale - 1](z, inverse=True)
                log_det = log_det + ld

            for i in range(len(self.layers[scale]) - 1, -1, -1):
                z, ld = self.layers[scale][i](z, inverse=True)
                log_det = log_det + ld

        return z, log_det

    def sample(self, num_samples: int, device: str = "cpu") -> Tensor:
        """Generate samples."""
        z = torch.randn(num_samples, self.input_dim, device=device)
        x, _ = self.forward(z, inverse=True)
        return x

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability."""
        z, log_det = self.forward(x, inverse=False)
        log_prob = -0.5 * (z**2).sum(dim=-1)
        log_prob = log_prob - 0.5 * self.input_dim * math.log(2 * math.pi)
        return log_prob + log_det


class SqueezeLayer(nn.Module):
    """
    Squeeze layer for multi-scale architecture.

    Args:
        channels: Number of channels
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply squeeze/unsqueeze."""
        if inverse:
            return self._unsqueeze(x)
        return self._squeeze(x)

    def _squeeze(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Squeeze spatial dimensions into channels."""
        B, C = x.shape

        x = x.view(B, C // 4, 2, 2)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, -1)

        return x, torch.zeros(B, device=x.device)

    def _unsqueeze(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Unsqueeze channels into spatial dimensions."""
        B = y.shape[0]

        y = y.view(B, 2, 2, -1)
        y = y.transpose(1, 2).contiguous()
        y = y.view(B, -1)

        return y, torch.zeros(B, device=y.device)


class Glow(nn.Module):
    """
    Glow: Generative Flow with Invertible 1x1 Convolutions.

    Args:
        num_channels: Number of channels
        hidden_channels: Hidden channels
        num_flows: Number of flow steps per level
        num_levels: Number of multi-scale levels
    """

    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 512,
        num_flows: int = 32,
        num_levels: int = 3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_channels = hidden_channels
        self.num_flows = num_flows
        self.num_levels = num_levels

        self.flows = nn.ModuleList()
        self.level_scales = nn.ModuleList()

        current_channels = num_channels

        for level in range(num_levels):
            level_flows = nn.ModuleList()

            for i in range(num_flows):
                level_flows.append(ActNorm(num_channels=current_channels))
                level_flows.append(InvertibleConv1x1(num_channels=current_channels))
                level_flows.append(
                    FlowStep(
                        num_channels=current_channels,
                        hidden_channels=hidden_channels,
                    )
                )

            self.flows.append(level_flows)

            if level < num_levels - 1:
                self.level_scales.append(
                    InvertibleDownsample(num_channels=current_channels)
                )
                current_channels *= 4

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply Glow flow."""
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        log_det = torch.zeros(x.shape[0], device=x.device)

        for level in range(self.num_levels):
            for flow in self.flows[level]:
                x, ld = flow(x)
                log_det = log_det + ld

            if level < self.num_levels - 1:
                x, ld = self.level_scales[level](x)
                log_det = log_det + ld

        return x, log_det

    def _inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        log_det = torch.zeros(z.shape[0], device=z.device)

        for level in range(self.num_levels - 1, -1, -1):
            if level > 0:
                z, ld = self.level_scales[level - 1](z, inverse=True)
                log_det = log_det + ld

            for i in range(len(self.flows[level]) - 1, -1, -1):
                z, ld = self.flows[level][i](z, inverse=True)
                log_det = log_det + ld

        return z, log_det

    def sample(self, num_samples: int, device: str = "cpu") -> Tensor:
        """Generate samples."""
        z = torch.randn(
            num_samples,
            self.num_channels * (4 ** (self.num_levels - 1)),
            device=device,
        )
        x, _ = self.forward(z, inverse=True)
        return x

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability."""
        z, log_det = self.forward(x, inverse=False)
        log_prob = -0.5 * (z**2).sum(dim=-1)
        log_prob = log_prob - 0.5 * z.shape[-1] * math.log(2 * math.pi)
        return log_prob + log_det


class InvertibleConv1x1(nn.Module):
    """
    Invertible 1x1 Convolution.

    Args:
        num_channels: Number of channels
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

        self.weight = nn.Parameter(torch.Tensor(num_channels, num_channels))
        self.register_buffer("LU", None)

        self._init_weight()

    def _init_weight(self) -> None:
        """Initialize with random orthogonal matrix."""
        nn.init.orthogonal_(self.weight)

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply invertible convolution."""
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        output = F.conv2d(x, self.weight.unsqueeze(2).unsqueeze(3))

        log_det = torch.slogdet(self.weight)[1]
        log_det = log_det * x.shape[2] * x.shape[3]

        return output, log_det

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        weight_inv = torch.inverse(self.weight)
        output = F.conv2d(y, weight_inv.unsqueeze(2).unsqueeze(3))

        log_det = -torch.slogdet(self.weight)[1]
        log_det = log_det * y.shape[2] * y.shape[3]

        return output, log_det


class InvertibleDownsample(nn.Module):
    """
    Invertible downsampling (split by 2).

    Args:
        num_channels: Number of channels
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply invertible downsample."""
        if inverse:
            return self._upsample(x)
        return self._downsample(x)

    def _downsample(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Downsample by factor of 2."""
        B, C, H, W = x.shape

        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * 4, H // 2, W // 2)

        return x, torch.zeros(B, device=x.device)

    def _upsample(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Upsample by factor of 2."""
        B, C, H, W = y.shape

        y = y.view(B, C // 4, 2, 2, H, W)
        y = y.permute(0, 1, 4, 2, 5, 3).contiguous()
        y = y.view(B, C // 4, H * 2, W * 2)

        return y, torch.zeros(B, device=y.device)


class FlowStep(nn.Module):
    """
    Single flow step: activation normalization + 1x1 conv + coupling.

    Args:
        num_channels: Number of channels
        hidden_channels: Hidden channels for coupling
    """

    def __init__(
        self,
        num_channels: int,
        hidden_channels: int = 512,
    ):
        super().__init__()

        self.coupling = ChannelCoupling(
            num_channels=num_channels,
            hidden_channels=hidden_channels,
        )

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply flow step."""
        return self.coupling(x, inverse=inverse)


class ChannelCoupling(nn.Module):
    """
    Channel-wise coupling layer.

    Args:
        num_channels: Number of channels
        hidden_channels: Hidden channels
    """

    def __init__(
        self,
        num_channels: int,
        hidden_channels: int = 512,
    ):
        super().__init__()
        self.num_channels = num_channels

        half = num_channels // 2

        self.net = nn.Sequential(
            nn.Conv2d(half, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, num_channels, 3, padding=1),
        )

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply channel coupling."""
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        B, C, H, W = x.shape
        half = C // 2

        x_a = x[:, :half]
        x_b = x[:, half:]

        params = self.net(x_a)
        log_scale, shift = params[:, half:], params[:, :half]

        log_scale = torch.tanh(log_scale)

        y_b = x_b * torch.exp(log_scale) + shift

        y = torch.cat([x_a, y_b], dim=1)

        log_det = log_scale.view(B, -1).sum(dim=-1)

        return y, log_det

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        B, C, H, W = y.shape
        half = C // 2

        y_a = y[:, :half]
        y_b = y[:, half:]

        params = self.net(y_a)
        log_scale, shift = params[:, half:], params[:, :half]

        log_scale = torch.tanh(log_scale)

        x_b = (y_b - shift) * torch.exp(-log_scale)

        x = torch.cat([y_a, x_b], dim=1)

        log_det = -log_scale.view(B, -1).sum(dim=-1)

        return x, log_det


class ActNorm(nn.Module):
    """
    Activation Normalization layer.

    Args:
        num_channels: Number of channels
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        self.initialized = False

    def initialize(self, x: Tensor) -> None:
        """Initialize from first batch."""
        with torch.no_grad():
            bias = -x.mean(dim=(0, 2, 3), keepdim=True)
            var = ((x + bias) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            log_scale = -0.5 * torch.log(var + 1e-6)

            self.bias.data = bias
            self.log_scale.data = log_scale

        self.initialized = True

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply activation norm."""
        if not self.initialized:
            self.initialize(x)

        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        output = (x + self.bias) * torch.exp(self.log_scale)

        log_det = self.log_scale.sum() * x.shape[2] * x.shape[3]
        log_det = log_det.expand(x.shape[0])

        return output, log_det

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        x = y * torch.exp(-self.log_scale) - self.bias

        log_det = -self.log_scale.sum() * y.shape[2] * y.shape[3]
        log_det = log_det.expand(y.shape[0])

        return x, log_det


class FlowBasedVAE(nn.Module):
    """
    Flow-based Variational Autoencoder.

    Combines VAE encoder/decoder with normalizing flow in latent space.

    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        hidden_dim: Hidden dimension
        num_flows: Number of flow layers
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        num_flows: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim * 2),
        )

        self.flow = RealNVP(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_flows,
            num_scales=1,
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input to latent parameters."""
        params = self.encoder(x)
        mean, log_var = params.chunk(2, dim=-1)

        log_var = torch.tanh(log_var)

        return mean, log_var

    def reparameterize(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode latent to output."""
        params = self.decoder(z)
        mean, log_var = params.chunk(2, dim=-1)

        log_var = torch.tanh(log_var)

        return mean, log_var

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full VAE forward pass."""
        mean, log_var = self.encode(x)

        z = self.reparameterize(mean, log_var)

        z_flow, log_det = self.flow(z)

        recon_mean, recon_log_var = self.decode(z_flow)

        return recon_mean, recon_log_var, z_flow, log_det

    def sample(self, num_samples: int, device: str = "cpu") -> Tensor:
        """Generate samples."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        z, _ = self.flow(z, inverse=True)

        recon_mean, _ = self.decode(z)

        return recon_mean

    def elbo(
        self,
        x: Tensor,
        beta: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute ELBO.

        Returns:
            Tuple of (total_elbo, recon_loss, kl_loss)
        """
        recon_mean, recon_log_var, z, log_det = self.forward(x)

        recon_loss = -0.5 * (
            (x - recon_mean) ** 2 * torch.exp(-recon_log_var) + recon_log_var
        )
        recon_loss = recon_loss.sum(dim=-1).mean()

        mean, log_var = self.encode(x)

        z_prior = torch.randn_like(z)
        z_prior, _ = self.flow(z_prior, inverse=True)

        kl_loss = -0.5 * (1 + log_var - mean**2 - torch.exp(log_var))
        kl_loss = kl_loss.sum(dim=-1).mean()

        kl_loss = kl_loss - log_det.mean()

        elbo = recon_loss - beta * kl_loss

        return elbo, recon_loss, kl_loss


class InvertibleMLP(nn.Module):
    """
    Invertible Multi-Layer Perceptron.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden dimensions
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim

        dims = [input_dim] + (hidden_dims or [64, 64]) + [input_dim]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                layers.append(self._get_activation(activation))

        self.net = nn.Sequential(*layers)

        self.log_scale = nn.Parameter(torch.zeros(input_dim))

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply invertible MLP."""
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        output = self.net(x)

        output = output * torch.exp(self.log_scale)

        log_det = self.log_scale.sum().expand(x.shape[0])

        return output, log_det

    def _inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass using fixed-point iteration."""
        x = y.clone()

        for _ in range(20):
            output = self.net(x)
            residual = (output - y) / (torch.exp(self.log_scale) + 1e-5)
            x = x - 0.1 * residual

        log_det = -self.log_scale.sum().expand(y.shape[0])

        return x, log_det


class DensityEstimator(nn.Module):
    """
    Generic density estimator using normalizing flows.

    Args:
        input_dim: Input dimension
        flow_type: Type of flow ('realnvp', 'maf', 'glow')
        hidden_dim: Hidden dimension
        num_layers: Number of flow layers
    """

    def __init__(
        self,
        input_dim: int,
        flow_type: str = "realnvp",
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.flow_type = flow_type

        if flow_type == "realnvp":
            self.flow = RealNVP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        elif flow_type == "maf":
            from .masked_autoregressive_flow import MAF

            self.flow = MAF(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        else:
            self.flow = RealNVP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply flow."""
        return self.flow(x, inverse=inverse)

    def sample(self, num_samples: int, device: str = "cpu") -> Tensor:
        """Generate samples."""
        return self.flow.sample(num_samples, device)

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability."""
        return self.flow.log_prob(x)
