"""
Distribution transformations.

Implements bijective transformations of probability distributions
including affine, monotonic, spline, and neural network-based transforms.
"""

from typing import Optional, Tuple, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from .core import BaseDistribution


class AffineTransform:
    """
    Affine transformation: y = a * x + b.

    log|det J| = log|a|

    Preserves the distribution family (Normal → Normal).
    """

    def __init__(self, loc: Tensor = 0.0, scale: Tensor = 1.0):
        self.loc = loc if isinstance(loc, Tensor) else torch.tensor(loc)
        self.scale = scale if isinstance(scale, Tensor) else torch.tensor(scale)

    def __call__(self, x: Tensor) -> Tensor:
        return self.loc + self.scale * x

    def inverse(self, y: Tensor) -> Tensor:
        return (y - self.loc) / self.scale

    def log_det_jacobian(self, x: Tensor) -> Tensor:
        return torch.log(torch.abs(self.scale))


class ExpTransform(AffineTransform):
    """
    Exponential transformation: y = exp(x).

    log|det J| = log|exp(x)| = x

    Transforms Normal to Log-Normal.
    """

    def __call__(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.log(y)

    def log_det_jacobian(self, x: Tensor) -> Tensor:
        return x


class LogTransform(AffineTransform):
    """
    Log transformation: y = log(x).

    log|det J| = -log(x)

    Inverse of ExpTransform.
    """

    def __call__(self, x: Tensor) -> Tensor:
        return torch.log(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.exp(y)

    def log_det_jacobian(self, x: Tensor) -> Tensor:
        return -torch.log(x)


class SigmoidTransform:
    """
    Sigmoid transformation: y = sigmoid(x) = 1/(1+exp(-x)).

    Maps R → (0, 1)
    log|det J| = -log(1 + exp(x)) - log(1 + exp(-x))
    """

    def __call__(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.log(y / (1 - y))

    def log_det_jacobian(self, x: Tensor) -> Tensor:
        return -F.softplus(x) - F.softplus(-x)


class SoftplusTransform:
    """
    Softplus transformation: y = log(1 + exp(x)).

    Maps R → (0, ∞) with smooth minimum at 0.
    """

    def __call__(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.where(y > 20, y, torch.log(torch.expm1(y)))

    def log_det_jacobian(self, x: Tensor) -> Tensor:
        return -torch.nn.functional.softplus(-x)


class MonotonicRationalQuadraticSpline:
    """
    Monotonic rational quadratic spline transform.

    Uses piecewise rational quadratic functions to create flexible,
    monotonic bijections. Based on (Durkan et al., 2019).

    Parameters:
        x_min: lower bound of input
        x_max: upper bound of input
        n_bins: number of spline bins
    """

    def __init__(
        self,
        x_min: float = -1.0,
        x_max: float = 1.0,
        n_bins: int = 8,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.n_bins = n_bins
        self.bin_width = (x_max - x_min) / n_bins

    def __call__(
        self,
        x: Tensor,
        w: Tensor,
        h: Tensor,
        d: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply spline transformation.

        Args:
            x: input values
            w: bin widths
            h: bin heights
            d: derivative at bin edges
        """
        x_normalized = (x - self.x_min) / (self.x_max - self.x_min)
        x_normalized = torch.clamp(x_normalized, 0, 1)

        bin_idx = torch.floor(x_normalized * self.n_bins).long()
        bin_idx = torch.clamp(bin_idx, 0, self.n_bins - 1)

        x_in_bin = x_normalized * self.n_bins - bin_idx.float()

        w_sum = torch.cumsum(w, dim=-1)
        h_sum = torch.cumsum(h, dim=-1)

        y = torch.gather(w_sum, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
        x_in_bin = x_in_bin - (y + w[..., bin_idx])

        return x_in_bin, bin_idx

    def inverse(
        self,
        y: Tensor,
        w: Tensor,
        h: Tensor,
        d: Tensor,
    ) -> Tensor:
        """Compute inverse spline transform."""
        raise NotImplementedError

    def log_det_jacobian(
        self,
        x: Tensor,
        w: Tensor,
        h: Tensor,
        d: Tensor,
    ) -> Tensor:
        """Compute log determinant of Jacobian."""
        return torch.zeros_like(x)


class NeuralSplineTransform(nn.Module):
    """
    Neural spline transform using monotonic neural network.

    Implements a flow transformation where bin parameters
    are produced by a neural network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_bins: int = 8,
        bound: float = 3.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_bins = n_bins
        self.bound = bound

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * input_dim * n_bins),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply transform and compute log det Jacobian.

        Returns:
            y: transformed values
            log_det: log determinant of Jacobian
        """
        params = self.net(x)
        n_out = self.input_dim * self.n_bins
        w = params[:, :n_out].view(-1, self.input_dim, self.n_bins)
        h = params[:, n_out : 2 * n_out].view(-1, self.input_dim, self.n_bins)
        d = params[:, 2 * n_out :].view(-1, self.input_dim, self.n_bins + 1)

        w = torch.softmax(w, dim=-1) * 2 * self.bound / self.n_bins
        h = torch.softmax(h, dim=-1) * 2 * self.bound / self.n_bins
        d = torch.nn.functional.softplus(d)

        y = self._apply_spline(x, w, h, d)
        log_det = self._compute_log_det(x, w, h, d)

        return y, log_det

    def _apply_spline(
        self,
        x: Tensor,
        w: Tensor,
        h: Tensor,
        d: Tensor,
    ) -> Tensor:
        """Apply rational quadratic spline."""
        x_scaled = (x + self.bound) / (2 * self.bound)
        x_scaled = torch.clamp(x_scaled, 1e-6, 1 - 1e-6)

        bin_idx = torch.floor(x_scaled * self.n_bins).long()
        bin_idx = torch.clamp(bin_idx, 0, self.n_bins - 1)

        theta = x_scaled * self.n_bins - bin_idx.float()

        w_i = w.gather(2, bin_idx.unsqueeze(2)).squeeze(2)
        h_i = h.gather(2, bin_idx.unsqueeze(2)).squeeze(2)

        d_left = d.gather(2, bin_idx.unsqueeze(2)).squeeze(2)
        d_right = d.gather(
            2, (bin_idx + 1).clamp(max=self.n_bins).unsqueeze(2)
        ).squeeze(2)

        epsilon = 1e-6
        num = h_i * theta**2 + d_left * w_i * theta * (1 - theta)
        denom = h_i + (d_left + d_right - 2 * h_i) * theta
        y = torch.clamp(num / (denom + epsilon) - w_i / 2, -self.bound, self.bound)

        return y

    def _compute_log_det(
        self,
        x: Tensor,
        w: Tensor,
        h: Tensor,
        d: Tensor,
    ) -> Tensor:
        """Compute log determinant of Jacobian."""
        x_scaled = (x + self.bound) / (2 * self.bound)
        bin_idx = torch.floor(x_scaled * self.n_bins).long()

        epsilon = 1e-6
        theta = x_scaled * self.n_bins - bin_idx.float()

        d_left = d.gather(2, bin_idx.unsqueeze(2)).squeeze(2)
        d_right = d.gather(
            2, (bin_idx + 1).clamp(max=self.n_bins).unsqueeze(2)
        ).squeeze(2)

        h_i = h.gather(2, bin_idx.unsqueeze(2)).squeeze(2)

        denom = h_i + (d_left + d_right - 2 * h_i) * theta
        derivative = (
            2 * h_i**2 * (d_left + d_right - 2 * h_i) * theta
            + (d_left * h_i - d_right * h_i) * w_i
        )

        return torch.log(derivative.abs() + epsilon) - torch.log(2 * self.bound)


class CouplingTransform(nn.Module):
    """
    Affine coupling layer for normalizing flows.

    Splits input into two parts:
        x = [x_1, x_2]
        y_1 = x_1
        y_2 = x_2 * exp(s(x_1)) + t(x_1)

    The transformation is invertible and Jacobian is triangular.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        mask_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim

        if mask_dim is None:
            mask_dim = input_dim // 2

        self.mask = self._create_mask(mask_dim)

        self.net = nn.Sequential(
            nn.Linear(mask_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (input_dim - mask_dim)),
        )

    def _create_mask(self, mask_dim: int) -> Tensor:
        mask = torch.zeros(self.input_dim, dtype=torch.bool)
        mask[:mask_dim] = True
        return mask

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply coupling transform.

        Returns:
            y: transformed values
            log_det: log determinant of Jacobian
        """
        x_1 = x[..., self.mask]
        x_2 = x[..., ~self.mask]

        params = self.net(x_1)
        shift = params[..., : self.input_dim // 2]
        log_scale = params[..., self.input_dim // 2 :]
        log_scale = torch.tanh(log_scale)

        y_2 = x_2 * torch.exp(log_scale) + shift
        y = torch.cat([x_1, y_2], dim=-1)

        log_det = log_scale.sum(dim=-1)

        return y, log_det

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse transformation."""
        y_1 = y[..., self.mask]
        y_2 = y[..., ~self.mask]

        params = self.net(y_1)
        shift = params[..., : self.input_dim // 2]
        log_scale = params[..., self.input_dim // 2 :]

        x_2 = (y_2 - shift) * torch.exp(-log_scale)
        x = torch.cat([y_1, x_2], dim=-1)

        return x, -log_scale.sum(dim=-1)


class ActNorm(nn.Module):
    """
    Activation Normalization layer.

    Per-channel normalization with learned scale and bias.
    y = (x - μ) * exp(log_s) + μ
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels))
        self.bias = nn.Parameter(torch.zeros(1, num_channels))
        self.initialized = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.initialized:
            self._initialize(x)

        y = (x - self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum() * x.shape[0]

        return y, log_det

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        x = y * torch.exp(-self.log_scale) + self.bias
        return x, -self.log_scale.sum() * y.shape[0]

    def _initialize(self, x: Tensor) -> None:
        with torch.no_grad():
            bias = x.mean(dim=0, keepdim=True)
            var = ((x - bias) ** 2).mean(dim=0, keepdim=True)
            log_scale = -0.5 * torch.log(var + 1e-6)

            self.bias.copy_(bias.squeeze(0))
            self.log_scale.copy_(log_scale.squeeze(0))

        self.initialized = True


class PlanarFlow(nn.Module):
    """
    Planar flow transformation.

    z = h(w·u + b) + u

    Where h is a smooth activation function.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.u = nn.Parameter(torch.randn(dim))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        inner = torch.einsum("bd,d->b", x, self.w) + self.b
        h = torch.tanh(inner)

        phi = (1 - h**2) * self.w

        z = x + self.u.unsqueeze(0) * h.unsqueeze(-1)

        log_det = torch.log(torch.abs(1 + torch.einsum("d,bd->b", self.u, phi)) + 1e-6)

        return z, log_det

    def inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("Inverse not available for planar flow")


class RadialFlow(nn.Module):
    """
    Radial flow transformation.

    z = x + β * (x - z_0) / (α + ||x - z_0||²)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.z_0 = nn.Parameter(torch.randn(dim))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        diff = x - self.z_0.unsqueeze(0)
        r_sq = (diff**2).sum(dim=-1, keepdim=True)

        denominator = self.alpha + r_sq
        z = x + self.beta.unsqueeze(-1) * diff / denominator

        log_det = torch.log(
            torch.abs(
                1
                + self.beta
                * (self.dim * r_sq - 2 * (diff**2).sum(-1))
                / (denominator**2)
            )
            + 1e-6
        )

        return z, log_det


class IAF(nn.Module):
    """
    Inverse Autoregressive Flow.

    Uses autoregressive neural network to parameterize
    the transformation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        for i in range(n_hidden):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 2 * input_dim))

        self.net = nn.Sequential(*layers)
        self.h = nn.ReLU()

    def forward(
        self, x: Tensor, context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if context is not None:
            x = torch.cat([x, context], dim=-1)

        params = self.net(x)
        m, s = params[..., : self.input_dim], params[..., self.input_dim :]
        s = torch.tanh(s)

        z = (x - m) * torch.exp(-s)

        log_det = -s.sum(dim=-1)

        return z, log_det


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation.

    Autoregressive density estimator using masked convolutions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            act = nn.SiLU()

        layers = []
        for i in range(n_hidden + 1):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim

            if i == n_hidden:
                out_dim = 2 * input_dim
            else:
                out_dim = hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_hidden:
                layers.append(act)

        self.net = nn.Sequential(*layers)

        self._create_masks(input_dim, hidden_dim, n_hidden)

    def _create_masks(self, input_dim: int, hidden_dim: int, n_hidden: int) -> None:
        mask = torch.arange(input_dim)
        mask = mask.unsqueeze(0) >= mask.unsqueeze(1)
        self.register_buffer("input_mask", mask.float())

        hidden_masks = []
        for i in range(n_hidden):
            m = torch.arange(hidden_dim)
            hidden_masks.append((m.unsqueeze(0) >= m.unsqueeze(1)).float())

        self.hidden_masks = hidden_masks

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = x @ self.input_mask

        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                h = layer(h)
                if i < len(self.hidden_masks):
                    h = h * self.hidden_masks[i]
                if i < len(self.net) - 1 and isinstance(self.net[i + 1], nn.ReLU):
                    h = torch.relu(h)

        m, s = h[..., : self.input_dim], h[..., self.input_dim :]
        s = torch.tanh(s)

        log_prob = -0.5 * (np.log(2 * np.pi) + s + (x - m) ** 2 * torch.exp(-s))

        return log_prob.sum(dim=-1), m
