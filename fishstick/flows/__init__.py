"""
Normalizing Flows for Density Estimation and Generative Modeling.

Implements:
- RealNVP (Dinh et al., 2017)
- Glow (Kingma & Dhariwal, 2018)
- Continuous normalizing flows (FFJORD)
- Autoregressive flows (MAF)
"""

from typing import Optional, Tuple, List, Callable, Dict
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


class CouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP.

    Transforms: x = [x1, x2]
        y1 = x1
        y2 = x2 * exp(s(x1)) + t(x1)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        mask_type: str = "alternating",
    ):
        super().__init__()
        self.dim = dim
        self.mask_type = mask_type

        # Create mask
        if mask_type == "alternating":
            self.register_buffer("mask", torch.arange(dim) % 2)
        elif mask_type == "half":
            mask = torch.zeros(dim)
            mask[dim // 2 :] = 1
            self.register_buffer("mask", mask)

        # Scale and translation networks
        self.scale_net = self._build_mlp(dim, dim, hidden_dim, n_hidden)
        self.trans_net = self._build_mlp(dim, dim, hidden_dim, n_hidden)

    def _build_mlp(
        self, in_dim: int, out_dim: int, hidden_dim: int, n_hidden: int
    ) -> nn.Module:
        """Build MLP for scale/translation."""
        layers = []
        for i in range(n_hidden + 1):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
            elif i == n_hidden:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass (inverse for sampling).

        Returns:
            y: Transformed variable
            logdet: Log determinant of Jacobian
        """
        mask = self.mask.view(1, -1).expand(x.size(0), -1)

        x1 = x * (1 - mask)

        s = self.scale_net(x1) * mask
        t = self.trans_net(x1) * mask

        y = x1 + (x * mask - t) * torch.exp(-s)
        logdet = -s.sum(dim=-1)

        return y, logdet

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inverse transformation (forward for density estimation).

        Returns:
            x: Original variable
            logdet: Log determinant of Jacobian
        """
        mask = self.mask.view(1, -1).expand(y.size(0), -1)

        y1 = y * (1 - mask)

        s = self.scale_net(y1) * mask
        t = self.trans_net(y1) * mask

        x = y1 + (y * mask * torch.exp(s) + t)
        logdet = s.sum(dim=-1)

        return x, logdet


class RealNVP(nn.Module):
    """
    Real-valued Non-Volume Preserving flow.

    Stack of coupling layers with alternating masks.
    """

    def __init__(
        self,
        dim: int,
        n_coupling: int = 8,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        base_dist: str = "gaussian",
    ):
        super().__init__()
        self.dim = dim
        self.base_dist = base_dist

        # Coupling layers with alternating masks
        self.coupling_layers = nn.ModuleList(
            [
                CouplingLayer(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    n_hidden=n_hidden,
                    mask_type="alternating" if i % 2 == 0 else "half",
                )
                for i in range(n_coupling)
            ]
        )

        # Permutation between layers
        self.permutations = nn.ModuleList(
            [InvertibleLinear(dim) for _ in range(n_coupling - 1)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward (sampling direction: z → x).

        Returns:
            z: Latent variable
            logdet: Total log determinant
        """
        logdet_total = 0

        for i, layer in enumerate(self.coupling_layers):
            x, logdet = layer(x)
            logdet_total += logdet

            if i < len(self.permutations):
                x, logdet_perm = self.permutations[i].inverse(x)
                logdet_total += logdet_perm

        return x, logdet_total

    def inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inverse (density estimation: x → z).

        Returns:
            x: Data space variable
            logdet: Total log determinant
        """
        logdet_total = 0

        for i in range(len(self.coupling_layers) - 1, -1, -1):
            if i < len(self.permutations):
                z, logdet_perm = self.permutations[i].forward(z)
                logdet_total += logdet_perm

            z, logdet = self.coupling_layers[i].inverse(z)
            logdet_total += logdet

        return z, logdet_total

    def sample(self, n_samples: int, device: str = "cpu") -> Tensor:
        """Sample from the flow."""
        if self.base_dist == "gaussian":
            z = torch.randn(n_samples, self.dim, device=device)
        else:
            z = torch.randn(n_samples, self.dim, device=device)

        x, _ = self.inverse(z)
        return x

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability."""
        z, logdet = self.forward(x)

        if self.base_dist == "gaussian":
            log_pz = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=-1)
        else:
            log_pz = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=-1)

        return log_pz + logdet


class InvertibleLinear(nn.Module):
    """Invertible 1x1 convolution (generalization of permutation)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Initialize with random orthogonal matrix
        w_init = np.random.randn(dim, dim)
        w_init = np.linalg.qr(w_init)[0]  # Orthogonal
        self.weight = nn.Parameter(torch.from_numpy(w_init).float())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward multiplication."""
        y = F.linear(x, self.weight)
        logdet = torch.slogdet(self.weight)[1]
        logdet = logdet.expand(x.size(0))
        return y, logdet

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse multiplication."""
        w_inv = torch.inverse(self.weight)
        x = F.linear(y, w_inv)
        logdet = -torch.slogdet(self.weight)[1]
        logdet = logdet.expand(y.size(0))
        return x, logdet


class ActNorm(nn.Module):
    """
    Activation normalization (Glow).

    Normalizes activations per-channel: y = s * (x - b)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        if not self.initialized and self.training:
            self._initialize(x)

        y = (x - self.bias) * torch.exp(-self.log_scale)
        logdet = -self.log_scale.sum().expand(x.size(0))
        return y, logdet

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass."""
        x = y * torch.exp(self.log_scale) + self.bias
        logdet = self.log_scale.sum().expand(y.size(0))
        return x, logdet

    def _initialize(self, x: Tensor) -> None:
        """Data-dependent initialization."""
        with torch.no_grad():
            mean = x.mean(dim=0)
            std = x.std(dim=0) + 1e-6

            self.bias.data = mean
            self.log_scale.data = torch.log(std)

        self.initialized = True


class GlowStep(nn.Module):
    """Single step of Glow (ActNorm → 1x1 Conv → Affine Coupling)."""

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actnorm = ActNorm(dim)
        self.conv = InvertibleLinear(dim)
        self.coupling = CouplingLayer(dim, hidden_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward (z → x direction)."""
        logdet = 0

        x, ld = self.actnorm.forward(x)
        logdet += ld

        x, ld = self.conv.forward(x)
        logdet += ld

        x, ld = self.coupling.forward(x)
        logdet += ld

        return x, logdet

    def inverse(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse (x → z direction)."""
        logdet = 0

        x, ld = self.coupling.inverse(x)
        logdet += ld

        x, ld = self.conv.inverse(x)
        logdet += ld

        x, ld = self.actnorm.inverse(x)
        logdet += ld

        return x, logdet


class Glow(nn.Module):
    """Glow: Generative Flow with Invertible 1×1 Convolutions."""

    def __init__(
        self,
        dim: int,
        n_levels: int = 3,
        n_steps_per_level: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.n_levels = n_levels

        # Simple implementation without splitting
        self.flows = nn.ModuleList(
            [GlowStep(dim, hidden_dim) for _ in range(n_levels * n_steps_per_level)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward transformation."""
        logdet = 0
        for flow in self.flows:
            x, ld = flow.forward(x)
            logdet += ld
        return x, logdet

    def inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse transformation."""
        logdet = 0
        for flow in reversed(self.flows):
            z, ld = flow.inverse(z)
            logdet += ld
        return z, logdet

    def sample(self, n_samples: int, device: str = "cpu") -> Tensor:
        """Sample from the model."""
        z = torch.randn(n_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability."""
        z, logdet = self.forward(x)
        log_pz = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + logdet


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation.

    Enables autoregressive property via masking.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        num_masks: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

        # Build network as separate layers for masking
        self.layers = nn.ModuleList()
        in_size = input_dim
        for i in range(n_hidden):
            self.layers.append(nn.Linear(in_size, hidden_dim))
            in_size = hidden_dim
        self.layers.append(
            nn.Linear(hidden_dim, input_dim * 2)
        )  # μ and logσ for each dim

        # Activation
        self.activation = nn.ReLU()

        # Create masks
        self.num_masks = num_masks
        self.current_mask = 0
        self._create_masks()

    def _create_masks(self) -> None:
        """Create connectivity masks for autoregressive property."""
        masks = []

        for _ in range(self.num_masks):
            # Assign each unit a number from 1 to input_dim
            m = {}
            m[0] = torch.arange(1, self.input_dim + 1)

            for i in range(self.n_hidden):
                m[i + 1] = torch.randint(1, self.input_dim, (self.hidden_dim,))

            # Create masks based on connectivity rules
            # Note: layer.weight shape is [out, in], so mask should be [out, in]
            mask_list = []
            for i in range(self.n_hidden):
                # mask[i] has shape [in_dim], mask[i+1] has shape [out_dim]
                # We need mask with shape [out_dim, in_dim]
                mask = (m[i].unsqueeze(0) <= m[i + 1].unsqueeze(1)).float()
                mask_list.append(mask)

            # Output mask - output is 2*input_dim (mean and log_scale)
            # Layer weight is [2*input_dim, hidden_dim], so mask should be [2*input_dim, hidden_dim]
            mask_out = (m[self.n_hidden].unsqueeze(0) < m[0].unsqueeze(1)).float()
            mask_out = mask_out.repeat(2, 1)  # Repeat for both mean and log_scale
            mask_list.append(mask_out)

            masks.append(mask_list)

        self.masks = masks

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with current mask."""
        h = x
        mask_idx = self.current_mask

        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                # Hidden layers with mask and activation
                h = F.linear(h, layer.weight * self.masks[mask_idx][i], layer.bias)
                h = self.activation(h)
            else:
                # Output layer with mask, no activation
                h = F.linear(h, layer.weight * self.masks[mask_idx][i], layer.bias)

        mu, log_scale = h.chunk(2, dim=-1)
        return mu, log_scale

    def sample(self, n_samples: int, device: str = "cpu") -> Tensor:
        """Sample autoregressively."""
        x = torch.zeros(n_samples, self.input_dim, device=device)

        for i in range(self.input_dim):
            mu, log_scale = self.forward(x)
            scale = torch.exp(log_scale)
            x[:, i] = mu[:, i] + scale[:, i] * torch.randn(n_samples, device=device)

        return x


class MAF(nn.Module):
    """
    Masked Autoregressive Flow.

    Uses MADE for autoregressive transformations.
    """

    def __init__(
        self,
        dim: int,
        n_mades: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.dim = dim

        self.mades = nn.ModuleList(
            [MADE(dim, hidden_dim, num_masks=n_mades) for _ in range(n_mades)]
        )

        # Permutations between MADEs
        self.perms = [torch.randperm(dim) for _ in range(n_mades - 1)]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward (density estimation)."""
        logdet = 0

        for i, made in enumerate(self.mades):
            mu, log_scale = made(x)
            x = (x - mu) * torch.exp(-log_scale)
            logdet -= log_scale.sum(dim=-1)

            if i < len(self.perms):
                x = x[:, self.perms[i]]

        return x, logdet

    def inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse (sampling)."""
        logdet = 0

        for i in range(len(self.mades) - 1, -1, -1):
            if i < len(self.perms):
                inv_perm = torch.argsort(self.perms[i])
                z = z[:, inv_perm]

            made = self.mades[i]
            mu, log_scale = made(z)
            z = z * torch.exp(log_scale) + mu
            logdet += log_scale.sum(dim=-1)

        return z, logdet

    def sample(self, n_samples: int, device: str = "cpu") -> Tensor:
        """Sample from the model."""
        z = torch.randn(n_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability."""
        z, logdet = self.forward(x)
        log_pz = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + logdet


class ConditionalNormalizingFlow(nn.Module):
    """Conditional normalizing flow p(x|y)."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 256,
        n_coupling: int = 6,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # Condition embedding
        self.condition_encoder = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coupling layers (now conditional)
        self.coupling_layers = nn.ModuleList(
            [
                ConditionalCouplingLayer(x_dim, hidden_dim, hidden_dim)
                for _ in range(n_coupling)
            ]
        )

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass conditioned on y."""
        condition = self.condition_encoder(y)

        logdet = 0
        for layer in self.coupling_layers:
            x, ld = layer.forward(x, condition)
            logdet += ld

        return x, logdet

    def inverse(self, z: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass conditioned on y."""
        condition = self.condition_encoder(y)

        logdet = 0
        for layer in reversed(self.coupling_layers):
            z, ld = layer.inverse(z, condition)
            logdet += ld

        return z, logdet

    def sample(self, y: Tensor) -> Tensor:
        """Sample x given condition y."""
        z = torch.randn(y.size(0), self.x_dim, device=y.device)
        x, _ = self.inverse(z, y)
        return x


class ConditionalCouplingLayer(nn.Module):
    """Conditional version of coupling layer."""

    def __init__(self, dim: int, condition_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim

        # Scale and translation networks (now take condition)
        self.scale_net = nn.Sequential(
            nn.Linear(dim // 2 + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - dim // 2),
        )

        self.trans_net = nn.Sequential(
            nn.Linear(dim // 2 + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - dim // 2),
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward (sampling direction)."""
        x1, x2 = x.chunk(2, dim=-1)

        # Concatenate with condition
        x1_cond = torch.cat([x1, condition], dim=-1)

        s = self.scale_net(x1_cond)
        t = self.trans_net(x1_cond)

        y1 = x1
        y2 = (x2 - t) * torch.exp(-s)

        y = torch.cat([y1, y2], dim=-1)
        logdet = -s.sum(dim=-1)

        return y, logdet

    def inverse(self, y: Tensor, condition: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse (density estimation)."""
        y1, y2 = y.chunk(2, dim=-1)

        y1_cond = torch.cat([y1, condition], dim=-1)

        s = self.scale_net(y1_cond)
        t = self.trans_net(y1_cond)

        x1 = y1
        x2 = y2 * torch.exp(s) + t

        x = torch.cat([x1, x2], dim=-1)
        logdet = s.sum(dim=-1)

        return x, logdet
