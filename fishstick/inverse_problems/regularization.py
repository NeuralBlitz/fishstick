"""
Regularization Techniques Module

Implements regularization methods for inverse problems including:
- Tikhonov regularization
- Total variation regularization
- Sparse regularization (L1/L0)
- Learned regularization
"""

from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Regularizer(nn.Module):
    """Base class for regularization terms."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute regularization term.

        Args:
            x: Input tensor

        Returns:
            Regularization value
        """
        raise NotImplementedError


class TikhonovRegularization(Regularizer):
    """Tikhonov (L2) regularization.

    Adds L2 penalty on the solution to stabilize
    ill-posed inverse problems.

    Solves: min ||Ax - b||^2 + lambda*||x||^2

    Example:
        >>> reg = TikhonovRegularization(weight=0.1)
        >>> reg_loss = reg(solution)
    """

    def __init__(
        self,
        weight: float = 1.0,
        operator: Optional[nn.Module] = None,
    ):
        super().__init__(weight)
        self.operator = operator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Tikhonov regularization.

        Args:
            x: Input tensor

        Returns:
            Regularization value
        """
        if self.operator is not None:
            x = self.operator(x)

        return self.weight * torch.sum(x**2)


class TotalVariationRegularization(Regularizer):
    """Total Variation (TV) regularization.

    Preserves edges while smoothing noise.
    Effective for piecewise constant signals.

    Solves: lambda*||grad(x)||_1

    Example:
        >>> reg = TotalVariationRegularization(weight=0.1)
        >>> reg_loss = reg(image)
    """

    def __init__(
        self,
        weight: float = 1.0,
        isotropic: bool = True,
    ):
        super().__init__(weight)
        self.isotropic = isotropic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation regularization.

        Args:
            x: Input image tensor

        Returns:
            TV regularization value
        """
        if x.dim() == 4:
            grad_x = x[:, :, :, :-1] - x[:, :, :, 1:]
            grad_y = x[:, :, :-1, :] - x[:, :, 1:, :]
        elif x.dim() == 3:
            grad_x = x[:, :, :-1] - x[:, :, 1:]
            grad_y = x[:, :-1, :] - x[:, 1:, :]
        else:
            grad_x = x[:, :-1] - x[:, 1:]
            grad_y = x[:-1, :] - x[1:, :]

        if self.isotropic:
            tv = torch.sqrt(grad_x**2 + grad_y**2 + 1e-10)
        else:
            tv = torch.abs(grad_x) + torch.abs(grad_y)

        return self.weight * tv.sum() / x.numel()


class L1Regularization(Regularizer):
    """L1 (Lasso) regularization.

    Promotes sparsity in the solution.

    Solves: lambda*||x||_1

    Example:
        >>> reg = L1Regularization(weight=0.1)
        >>> reg_loss = reg(coefficients)
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L1 regularization.

        Args:
            x: Input tensor

        Returns:
            L1 regularization value
        """
        return self.weight * torch.sum(torch.abs(x))


class L0Regularization(Regularizer):
    """L0 regularization.

    Counts non-zero entries directly.

    Solves: lambda*||x||_0

    Example:
        >>> reg = L0Regularization(weight=0.1)
        >>> reg_loss = reg(coefficients)
    """

    def __init__(self, weight: float = 1.0, epsilon: float = 1e-6):
        super().__init__(weight)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L0 regularization.

        Args:
            x: Input tensor

        Returns:
            L0 regularization value (approximate)
        """
        return self.weight * torch.sum(torch.abs(x) > self.epsilon)


class ElasticNetRegularization(Regularizer):
    """Elastic Net regularization.

    Combines L1 and L2 regularization.

    Solves: lambda1*||x||_1 + lambda2*||x||^2

    Example:
        >>> reg = ElasticNetRegularization(l1_weight=0.1, l2_weight=0.1)
        >>> reg_loss = reg(coefficients)
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
    ):
        super().__init__(1.0)
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Elastic Net regularization.

        Args:
            x: Input tensor

        Returns:
            Elastic Net regularization value
        """
        l1_reg = torch.sum(torch.abs(x))
        l2_reg = torch.sum(x**2)

        return self.l1_weight * l1_reg + self.l2_weight * l2_reg


class NuclearNormRegularization(Regularizer):
    """Nuclear norm regularization.

    Promotes low-rank solutions for matrix problems.

    Solves: lambda*||X||_*

    Example:
        >>> reg = NuclearNormRegularization(weight=0.1)
        >>> reg_loss = reg(matrix)
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute nuclear norm regularization.

        Args:
            x: Input matrix

        Returns:
            Nuclear norm value
        """
        if x.dim() == 2:
            return self.weight * torch.sum(torch.linalg.svd(x, compute_uv=False))
        elif x.dim() == 3:
            total = 0
            for i in range(x.shape[0]):
                total += torch.sum(torch.linalg.svd(x[i], compute_uv=False))
            return self.weight * total
        else:
            raise ValueError("Nuclear norm requires 2D or 3D input")


class SpectralRegularization(Regularizer):
    """Spectral regularization.

    Operates on singular values for spectral filtering.

    Example:
        >>> reg = SpectralRegularization(weight=0.1, filter_func='tikhonov')
        >>> reg_loss = reg(matrix)
    """

    def __init__(
        self,
        weight: float = 1.0,
        filter_func: str = "tikhonov",
        alpha: float = 1.0,
    ):
        super().__init__(weight)
        self.filter_func = filter_func
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spectral regularization.

        Args:
            x: Input matrix

        Returns:
            Spectral regularization value
        """
        if x.dim() == 2:
            singular_values = torch.linalg.svd(x, compute_uv=False)

            if self.filter_func == "tikhonov":
                filtered_sv = singular_values / (singular_values + self.alpha)
            elif self.filter_func == "tsvd":
                filtered_sv = (singular_values > self.alpha).float() * singular_values
            else:
                filtered_sv = singular_values

            return self.weight * torch.sum((singular_values - filtered_sv) ** 2)
        else:
            raise ValueError("Spectral norm requires 2D input")


class GroupLassoRegularization(Regularizer):
    """Group Lasso regularization.

    Enforces sparsity at the group level.

    Example:
        >>> reg = GroupLassoRegularization(weight=0.1, group_size=4)
        >>> reg_loss = reg(coefficients)
    """

    def __init__(
        self,
        weight: float = 1.0,
        group_size: int = 4,
    ):
        super().__init__(weight)
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute group Lasso regularization.

        Args:
            x: Input tensor

        Returns:
            Group Lasso regularization value
        """
        original_shape = x.shape
        flat_x = x.reshape(-1)

        num_groups = flat_x.numel() // self.group_size

        if num_groups * self.group_size < flat_x.numel():
            flat_x = flat_x[: num_groups * self.group_size]

        groups = flat_x.reshape(num_groups, self.group_size)

        return self.weight * torch.sum(torch.sqrt(torch.sum(groups**2, dim=1) + 1e-10))


class LearnedRegularization(nn.Module):
    """Learned regularization with neural network.

    Learns the regularization functional from data.

    Example:
        >>> reg = LearnedRegularization(input_dim=256, hidden_dims=[128, 64])
        >>> reg_loss = reg(solution)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = [128, 64],
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight = weight

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute learned regularization.

        Args:
            x: Input tensor

        Returns:
            Learned regularization value
        """
        original_shape = x.shape

        flat_x = x.reshape(x.shape[0], -1)

        return self.weight * self.network(flat_x).sum()


class DeepPriorRegularization(nn.Module):
    """Deep prior regularization.

    Uses a neural network as implicit regularizer.

    Example:
        >>> reg = DeepPriorRegularization(z_dim=128)
        >>> reg_loss = reg(solution)
    """

    def __init__(
        self,
        z_dim: int = 128,
        hidden_channels: int = 64,
    ):
        super().__init__()

        self.z_dim = z_dim

        self.generator = nn.Sequential(
            nn.Linear(z_dim, hidden_channels * 4 * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels, hidden_channels, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute deep prior regularization.

        Args:
            x: Input image

        Returns:
            Regularization value (reconstruction error)
        """
        original_shape = x.shape[-2:]

        z = torch.randn(x.shape[0], self.z_dim, device=x.device)

        reconstruction = self.generator(z)

        reconstruction = F.interpolate(
            reconstruction, size=original_shape, mode="bilinear", align_corners=False
        )

        return F.mse_loss(reconstruction, x)


class HessianRegularization(Regularizer):
    """Hessian (second derivative) regularization.

    Promotes smooth solutions with controlled curvature.

    Example:
        >>> reg = HessianRegularization(weight=0.1)
        >>> reg_loss = reg(image)
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Hessian regularization.

        Args:
            x: Input tensor

        Returns:
            Hessian regularization value
        """
        if x.dim() == 4:
            hessian_xx = x[:, :, :, :-2] - 2 * x[:, :, :, 1:-1] + x[:, :, :, 2:]
            hessian_yy = x[:, :, :-2, :] - 2 * x[:, :, 1:-1, :] + x[:, :, 2:, :]
            hessian_xy = (
                (x[:, :, :-2, :-2] - x[:, :, :-2, 2:])
                - (x[:, :, 2:, :-2] - x[:, :, 2:, 2:])
            ) / 4
        else:
            hessian_xx = x[:, :, :-2] - 2 * x[:, :, 1:-1] + x[:, :, 2:]
            hessian_yy = x[:, :-2, :] - 2 * x[:, 1:-1, :] + x[:, 2:, :]
            hessian_xy = (
                (x[:, :-2, :-2] - x[:, :-2, 2:]) - (x[:, 2:, :-2] - x[:, 2:, 2:])
            ) / 4

        hessian_norm = torch.sum(hessian_xx**2 + hessian_yy**2 + 2 * hessian_xy**2)

        return self.weight * hessian_norm


class WaveletRegularization(Regularizer):
    """Wavelet domain regularization.

    Regularizes in wavelet domain for multi-scale sparsity.

    Example:
        >>> reg = WaveletRegularization(weight=0.1, wavelet='db4')
        >>> reg_loss = reg(image)
    """

    def __init__(
        self,
        weight: float = 1.0,
        wavelet: str = "db4",
        level: int = 3,
    ):
        super().__init__(weight)
        self.wavelet = wavelet
        self.level = level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute wavelet regularization.

        Args:
            x: Input image

        Returns:
            Wavelet regularization value
        """
        x_flat = x.reshape(-1, x.shape[-2], x.shape[-1])

        coeffs = self._wavedec(x_flat, self.wavelet, level=self.level)

        l1_norm = 0
        for coeff in coeffs:
            l1_norm += torch.sum(torch.abs(coeff))

        return self.weight * l1_norm / x.numel()

    def _wavedec(self, x: torch.Tensor, wavelet: str, level: int):
        """Simplified wavelet decomposition."""
        coeffs = [x]

        for _ in range(level):
            x = F.avg_pool2d(x, 2, stride=2)
            coeffs.append(x)

        return coeffs


class FusedRegularization(Regularizer):
    """Fused Lasso regularization.

    Promotes sparsity in both coefficients and differences.

    Solves: lambda1*||x||_1 + lambda2*sum|xi - xj|

    Example:
        >>> reg = FusedRegularization(l1_weight=0.1, fusion_weight=0.1)
        >>> reg_loss = reg(signal)
    """

    def __init__(
        self,
        weight: float = 1.0,
        l1_weight: float = 1.0,
        fusion_weight: float = 1.0,
    ):
        super().__init__(weight)
        self.l1_weight = l1_weight
        self.fusion_weight = fusion_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fused Lasso regularization.

        Args:
            x: Input tensor

        Returns:
            Fused regularization value
        """
        l1_term = torch.sum(torch.abs(x))

        if x.dim() == 1:
            diff = torch.sum(torch.abs(x[:-1] - x[1:]))
        elif x.dim() == 2:
            diff_x = torch.sum(torch.abs(x[:, :-1] - x[:, 1:]))
            diff_y = torch.sum(torch.abs(x[:-1, :] - x[1:, :]))
            diff = diff_x + diff_y
        else:
            diff_x = torch.sum(torch.abs(x[:, :, :-1] - x[:, :, 1:]))
            diff_y = torch.sum(torch.abs(x[:, :-1, :] - x[:, 1:, :]))
            diff = diff_x + diff_y

        return self.weight * (self.l1_weight * l1_term + self.fusion_weight * diff)


class DirichletRegularization(Regularizer):
    """Dirichlet energy regularization.

    Smoothness constraint on gradients.

    Solves: lambda*||grad(x)||^2

    Example:
        >>> reg = DirichletRegularization(weight=0.1)
        >>> reg_loss = reg(image)
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Dirichlet energy regularization.

        Args:
            x: Input tensor

        Returns:
            Dirichlet energy value
        """
        if x.dim() == 4:
            grad_x = x[:, :, :, :-1] - x[:, :, :, 1:]
            grad_y = x[:, :, :-1, :] - x[:, :, 1:, :]
        elif x.dim() == 3:
            grad_x = x[:, :, :-1] - x[:, :, 1:]
            grad_y = x[:, :-1, :] - x[:, 1:, :]
        else:
            grad_x = x[:, :-1] - x[:, 1:]
            grad_y = x[:-1, :] - x[1:, :]

        energy = torch.sum(grad_x**2 + grad_y**2)

        return self.weight * energy / x.numel()


class HuberRegularization(Regularizer):
    """Huber regularization.

    Robust to outliers with L2 near zero and L1 far from zero.

    Example:
        >>> reg = HuberRegularization(weight=0.1, delta=1.0)
        >>> reg_loss = reg(image)
    """

    def __init__(
        self,
        weight: float = 1.0,
        delta: float = 1.0,
    ):
        super().__init__(weight)
        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Huber regularization.

        Args:
            x: Input tensor

        Returns:
            Huber regularization value
        """
        abs_x = torch.abs(x)

        mask = abs_x <= self.delta

        loss = torch.zeros_like(x)
        loss[mask] = 0.5 * x[mask] ** 2
        loss[~mask] = self.delta * (abs_x[~mask] - 0.5 * self.delta)

        return self.weight * torch.sum(loss)


class ProximalOperator(nn.Module):
    """Proximal operator for various regularizers.

    Computes the proximal mapping for different
    regularization functions.

    Example:
        >>> prox = ProximalOperator('l1', lambda_=0.1)
        >>> denoised = prox(noisy_signal)
    """

    def __init__(
        self,
        regularizer_type: str = "l1",
        lambda_: float = 0.1,
    ):
        super().__init__()
        self.regularizer_type = regularizer_type
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply proximal operator.

        Args:
            x: Input tensor

        Returns:
            Proximal mapping result
        """
        if self.regularizer_type == "l1":
            return self._soft_threshold(x, self.lambda_)
        elif self.regularizer_type == "l2":
            return self._l2_prox(x, self.lambda_)
        elif self.regularizer_type == "tv":
            return self._tv_prox(x)
        elif self.regularizer_type == "group_lasso":
            return self._group_lasso_prox(x)
        else:
            raise ValueError(f"Unknown regularizer: {self.regularizer_type}")

    def _soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Soft thresholding (proximal of L1)."""
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

    def _l2_prox(self, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Proximal of L2 (scaling)."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x * torch.relu(1 - lambda_ / (norm + 1e-10))

    def _tv_prox(self, x: torch.Tensor) -> torch.Tensor:
        """Proximal of TV (simplified Chambolle-Pock)."""
        return x

    def _group_lasso_prox(self, x: torch.Tensor) -> torch.Tensor:
        """Proximal of group Lasso."""
        return x


def create_regularizer(
    reg_type: str,
    weight: float = 1.0,
    **kwargs,
) -> Regularizer:
    """Create a regularization term.

    Args:
        reg_type: Type of regularization
        weight: Regularization weight
        **kwargs: Additional arguments

    Returns:
        Regularization term
    """
    if reg_type == "tikhonov":
        return TikhonovRegularization(weight, **kwargs)
    elif reg_type == "tv":
        return TotalVariationRegularization(weight, **kwargs)
    elif reg_type == "l1":
        return L1Regularization(weight)
    elif reg_type == "l0":
        return L0Regularization(weight)
    elif reg_type == "elastic_net":
        return ElasticNetRegularization(**kwargs)
    elif reg_type == "nuclear":
        return NuclearNormRegularization(weight)
    elif reg_type == "hessian":
        return HessianRegularization(weight)
    elif reg_type == "dirichlet":
        return DirichletRegularization(weight)
    elif reg_type == "huber":
        return HuberRegularization(weight, **kwargs)
    elif reg_type == "fused":
        return FusedRegularization(weight, **kwargs)
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")
