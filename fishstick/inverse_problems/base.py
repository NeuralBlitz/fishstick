"""
Inverse Problems Base Module

Core utilities and base classes for inverse problem formulations.
"""

from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np


class LinearOperator(nn.Module):
    """Base class for linear operators in inverse problems.

    Implements the forward and adjoint operations for linear
    operators that appear in inverse problems (e.g., blur kernels,
    sampling matrices, Radon transforms).
    """

    def __init__(self, matrix: Optional[torch.Tensor] = None):
        super().__init__()
        self.matrix = matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward operator.

        Args:
            x: Input tensor

        Returns:
            Applied operator result
        """
        raise NotImplementedError

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint (transpose) operator.

        Args:
            y: Input tensor

        Returns:
            Adjoint operation result
        """
        raise NotImplementedError


class SensingMatrix(LinearOperator):
    """Compressed sensing measurement matrix.

    Implements various types of sensing matrices including
    random Gaussian, random Bernoulli, and Fourier-based.
    """

    def __init__(
        self,
        num_measurements: int,
        signal_dim: int,
        matrix_type: str = "gaussian",
    ):
        super().__init__()
        self.num_measurements = num_measurements
        self.signal_dim = signal_dim
        self.matrix_type = matrix_type

        if matrix_type == "gaussian":
            matrix = torch.randn(num_measurements, signal_dim) / np.sqrt(
                num_measurements
            )
        elif matrix_type == "bernoulli":
            matrix = torch.randn(num_measurements, signal_dim)
            matrix = (matrix > 0).float() * 2 - 1
            matrix = matrix / np.sqrt(num_measurements)
        elif matrix_type == "rademacher":
            matrix = torch.randint(0, 2, (num_measurements, signal_dim)).float() * 2 - 1
            matrix = matrix / np.sqrt(num_measurements)
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

        self.matrix = nn.Parameter(matrix, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sensing matrix.

        Args:
            x: Input signal (batch, signal_dim) or (signal_dim,)

        Returns:
            Measurements (batch, num_measurements) or (num_measurements,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        result = x @ self.matrix.T

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint (transpose) of sensing matrix.

        Args:
            y: Measurements (batch, num_measurements) or (num_measurements,)

        Returns:
            Reconstructed signal (batch, signal_dim) or (signal_dim,)
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        result = y @ self.matrix

        if squeeze_output:
            result = result.squeeze(0)

        return result


class BlurKernel(nn.Module):
    """Convolution kernel for image blur operators.

    Implements common blur kernels including Gaussian,
    motion blur, and disk kernels.
    """

    def __init__(
        self,
        kernel_size: int = 11,
        kernel_type: str = "gaussian",
        sigma: float = 3.0,
        angle: float = 0.0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.angle = angle

        kernel = self._create_kernel()
        self.register_buffer("kernel", kernel)

    def _create_kernel(self) -> torch.Tensor:
        """Create the blur kernel."""
        if self.kernel_type == "gaussian":
            return self._gaussian_kernel()
        elif self.kernel_type == "motion":
            return self._motion_kernel()
        elif self.kernel_type == "disk":
            return self._disk_kernel()
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def _gaussian_kernel(self) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        ax = torch.arange(-self.kernel_size // 2 + 1.0, self.kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        return kernel / kernel.sum()

    def _motion_kernel(self) -> torch.Tensor:
        """Create motion blur kernel."""
        kernel = torch.zeros(self.kernel_size, self.kernel_size)
        cx, cy = self.kernel_size // 2, self.kernel_size // 2
        angle_rad = self.angle * np.pi / 180

        length = self.kernel_size // 2
        for i in range(-length, length + 1):
            x = int(cx + i * np.cos(angle_rad))
            y = int(cy + i * np.sin(angle_rad))
            if 0 <= x < self.kernel_size and 0 <= y < self.kernel_size:
                kernel[y, x] = 1.0

        return kernel / kernel.sum()

    def _disk_kernel(self) -> torch.Tensor:
        """Create disk (pillbox) kernel."""
        ax = torch.arange(-self.kernel_size // 2 + 1.0, self.kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        r = torch.sqrt(xx**2 + yy**2)
        kernel = (r <= self.sigma).float()
        return kernel / kernel.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply blur kernel via convolution.

        Args:
            x: Input image (batch, channels, height, width)

        Returns:
            Blurred image
        """
        return torch.nn.functional.conv2d(
            x,
            self.kernel.unsqueeze(0).unsqueeze(0),
            padding=self.kernel_size // 2,
            groups=x.shape[1],
        )

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint of blur kernel (same as forward for symmetric kernels).

        Args:
            y: Blurred image

        Returns:
            Adjoint convolution result
        """
        return self.forward(y)


class InverseProblemLoss(nn.Module):
    """Loss function for inverse problems.

    Combines data fidelity term with regularization.
    """

    def __init__(
        self,
        forward_operator: nn.Module,
        data_fidelity: str = "l2",
        regularization_weight: float = 0.0,
        regularizer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.forward_operator = forward_operator
        self.data_fidelity = data_fidelity
        self.regularization_weight = regularization_weight
        self.regularizer = regularizer

    def forward(
        self,
        x_reconstructed: torch.Tensor,
        y_measured: torch.Tensor,
    ) -> torch.Tensor:
        """Compute inverse problem loss.

        Args:
            x_reconstructed: Reconstructed signal
            y_measured: Measured/observed signal

        Returns:
            Total loss value
        """
        y_pred = self.forward_operator(x_reconstructed)

        if self.data_fidelity == "l2":
            data_loss = torch.mean((y_pred - y_measured) ** 2)
        elif self.data_fidelity == "l1":
            data_loss = torch.mean(torch.abs(y_pred - y_measured))
        elif self.data_fidelity == "huber":
            data_loss = torch.nn.functional.huber_loss(y_pred, y_measured)
        else:
            raise ValueError(f"Unknown data fidelity: {self.data_fidelity}")

        loss = data_loss

        if self.regularizer is not None and self.regularization_weight > 0:
            reg_loss = self.regularizer(x_reconstructed)
            loss = loss + self.regularization_weight * reg_loss

        return loss


def psnr(
    reconstructed: torch.Tensor,
    ground_truth: torch.Tensor,
    max_val: float = 1.0,
) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        reconstructed: Reconstructed image
        ground_truth: Ground truth image
        max_val: Maximum possible pixel value

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((reconstructed - ground_truth) ** 2)
    psnr_val = 10 * torch.log10(max_val**2 / mse)
    return psnr_val


def ssim(
    reconstructed: torch.Tensor,
    ground_truth: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute Structural Similarity Index.

    Args:
        reconstructed: Reconstructed image
        ground_truth: Ground truth image
        window_size: Size of the sliding window

    Returns:
        SSIM value
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = torch.nn.functional.avg_pool2d(
        ground_truth, window_size, stride=1, padding=window_size // 2
    )
    mu2 = torch.nn.functional.avg_pool2d(
        reconstructed, window_size, stride=1, padding=window_size // 2
    )

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        torch.nn.functional.avg_pool2d(
            ground_truth**2, window_size, stride=1, padding=window_size // 2
        )
        - mu1_sq
    )
    sigma2_sq = (
        torch.nn.functional.avg_pool2d(
            reconstructed**2, window_size, stride=1, padding=window_size // 2
        )
        - mu2_sq
    )
    sigma12 = (
        torch.nn.functional.avg_pool2d(
            ground_truth * reconstructed,
            window_size,
            stride=1,
            padding=window_size // 2,
        )
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return torch.mean(ssim_map)
