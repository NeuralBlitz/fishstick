"""
Scene Understanding Utilities

Shared utilities for the scene understanding module.
"""

from typing import Tuple, List, Optional, Union, Dict, Any
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_psnr(
    pred: Tensor,
    target: Tensor,
    max_val: float = 1.0,
) -> Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted values
        target: Target values
        max_val: Maximum possible value

    Returns:
        PSNR value
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 10 * torch.log10(max_val**2 / mse)


def compute_ssim(
    img1: Tensor,
    img2: Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> Tensor:
    """
    Compute Structural Similarity Index.

    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        window_size: Size of the Gaussian window
        size_average: Whether to average the SSIM across batch

    Returns:
        SSIM value
    """
    C1 = 0.01**2
    C2 = 0.03**2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor(
        [
            np.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
            for x in range(window_size)
        ]
    )
    window = gauss / gauss.sum()
    window = window.unsqueeze(1)
    window = window.mm(window.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.to(img1.device)

    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.shape[1])
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.shape[1])
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1])
        - mu1_mu2
    )

    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


def normalize_tensor(
    x: Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Tensor:
    """
    Normalize tensor to [0, 1] range.

    Args:
        x: Input tensor
        min_val: Minimum value (if None, use tensor min)
        max_val: Maximum value (if None, use tensor max)

    Returns:
        Normalized tensor
    """
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    return (x - min_val) / (max_val - min_val + 1e-8)


def safe_divide(
    numerator: Tensor,
    denominator: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """
    Safely divide tensors, avoiding division by zero.

    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        epsilon: Small value to add to denominator

    Returns:
        Result of division
    """
    return numerator / (denominator + epsilon)


def gradient_x(img: Tensor) -> Tensor:
    """
    Compute image gradient in x direction.

    Args:
        img: Input image [B, C, H, W]

    Returns:
        Gradient in x direction
    """
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def gradient_y(img: Tensor) -> Tensor:
    """
    Compute image gradient in y direction.

    Args:
        img: Input image [B, C, H, W]

    Returns:
        Gradient in y direction
    """
    return img[:, :, :-1, :] - img[:, :, 1:, :]


class IntermediateLayerGetter:
    """
    Extract intermediate layer outputs from a model.
    """

    def __init__(
        self,
        model: nn.Module,
        return_layers: List[str],
    ):
        """
        Args:
            model: Model to extract features from
            return_layers: List of layer names to return
        """
        self.model = model
        self.return_layers = return_layers

        self._features = {}
        self._hooks = []

        for name, module in model.named_modules():
            if name in return_layers:
                self._hooks.append(module.register_forward_hook(self._get_hook(name)))

    def _get_hook(self, name: str):
        def hook(module, input, output):
            self._features[name] = output

        return hook

    def __call__(self, x: Tensor) -> Dict[str, Tensor]:
        self._features.clear()
        self.model(x)
        return self._features

    def __del__(self):
        for hook in self._hooks:
            hook.remove()


def meshgrid(
    height: int,
    width: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    """
    Create a mesh grid of (x, y) coordinates.

    Args:
        height: Grid height
        width: Grid width
        device: Device to create tensor on

    Returns:
        Tuple of (x_grid, y_grid)
    """
    x_grid = torch.linspace(0, width - 1, width, device=device)
    y_grid = torch.linspace(0, height - 1, height, device=device)
    y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing="ij")
    return x_grid, y_grid


def get_gaussian_kernel(
    kernel_size: int = 3,
    sigma: float = 1.0,
    channels: int = 1,
) -> Tensor:
    """
    Create a Gaussian kernel.

    Args:
        kernel_size: Size of the kernel
        sigma: Standard deviation
        channels: Number of channels

    Returns:
        Gaussian kernel [channels, 1, kernel_size, kernel_size]
    """
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    kernel = kernel.repeat(channels, 1, 1, 1)

    return kernel


def apply_bilateral_filter(
    img: Tensor,
    kernel_size: int = 5,
    sigma_color: float = 0.1,
    sigma_spatial: float = 1.0,
) -> Tensor:
    """
    Apply bilateral filter to an image.

    Args:
        img: Input image [B, C, H, W]
        kernel_size: Size of the filter kernel
        sigma_color: Color similarity sigma
        sigma_spatial: Spatial proximity sigma

    Returns:
        Filtered image
    """
    B, C, H, W = img.shape
    pad = kernel_size // 2

    # Compute spatial kernel
    x = torch.arange(-pad, pad + 1, dtype=torch.float32, device=img.device)
    spatial_kernel = torch.exp(-(x**2) / (2 * sigma_spatial**2))
    spatial_kernel = spatial_kernel.unsqueeze(0) * spatial_kernel.unsqueeze(1)
    spatial_kernel = spatial_kernel / spatial_kernel.sum()

    # Pad image
    padded = F.pad(img, [pad, pad, pad, pad], mode="replicate")

    # Apply filter
    output = torch.zeros_like(img)

    for i in range(H):
        for j in range(W):
            patch = padded[:, :, i : i + kernel_size, j : j + kernel_size]

            # Color similarity
            diff = patch - img[:, :, i : i + 1, j : j + 1]
            color_kernel = torch.exp(-(diff**2) / (2 * sigma_color**2))

            # Combined kernel
            kernel = spatial_kernel.unsqueeze(0) * color_kernel
            kernel = kernel / (kernel.sum(dim=(2, 3), keepdim=True) + 1e-8)

            output[:, :, i : i + 1, j : j + 1] = (patch * kernel).sum(
                dim=(2, 3), keepdim=True
            )

    return output
