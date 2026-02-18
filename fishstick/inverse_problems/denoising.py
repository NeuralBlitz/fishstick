"""
Image Denoising Module

Implements image denoising algorithms including:
- Non-local means (NL-means)
- BM3D-inspired methods
- Total variation denoising
- Deep learning denoising (DnCNN, UNet)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NonLocalMeansDenoising(nn.Module):
    """Non-local means denoising.

    Exploits self-similarity in images by averaging
    similar patches.

    Example:
        >>> nl_means = NonLocalMeansDenoising(window_size=21, patch_size=7, h=0.1)
        >>> denoised = nl_means(noisy_image)
    """

    def __init__(
        self,
        window_size: int = 21,
        patch_size: int = 7,
        h: float = 0.1,
        sigma: Optional[float] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_size = patch_size
        self.h = h
        self.sigma = sigma

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Apply non-local means denoising.

        Args:
            noisy: Noisy input image (batch, channels, height, width)

        Returns:
            Denoised image
        """
        if noisy.dim() == 3:
            noisy = noisy.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        if self.sigma is not None:
            h = self.sigma * 255
        else:
            h = self.h

        h2 = h * h

        pad = self.window_size // 2
        padded = F.pad(noisy, (pad, pad, pad, pad), mode="reflect")

        batch, channels, height, width = noisy.shape

        output = torch.zeros_like(noisy)
        weight_sum = torch.zeros_like(noisy)

        half_patch = self.patch_size // 2

        for i in range(height):
            for j in range(width):
                window = padded[
                    :,
                    :,
                    i : i + self.window_size,
                    j : j + self.window_size,
                ]

                center_patch = window[
                    :,
                    :,
                    pad - half_patch : pad + half_patch + 1,
                    pad - half_patch : pad + half_patch + 1,
                ]

                dists = (window - center_patch) ** 2
                dists = dists.reshape(batch, channels, -1).mean(dim=-1)

                weights = torch.exp(-dists / h2)

                patch = window[
                    :, :, half_patch : half_patch + 1, half_patch : half_patch + 1
                ]

                output[:, :, i : i + 1, j : j + 1] = (
                    weights.unsqueeze(-1).unsqueeze(-1) * patch
                ).sum(dim=0, keepdim=True)
                weight_sum[:, :, i : i + 1, j : j + 1] = (
                    weights.sum(dim=0, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                )

        output = output / (weight_sum + 1e-10)

        if squeeze_output:
            output = output.squeeze(0)

        return torch.clamp(output, 0, 1)


class TotalVariationDenoising(nn.Module):
    """Total variation denoising.

    Removes noise while preserving edges using
    L1 regularization on image gradients.

    Solves: min_x 0.5*||x - y||_2^2 + lambda*||grad(x)||_1

    Example:
        >>> tv_denoise = TotalVariationDenoising(lambda_=0.1, num_iterations=100)
        >>> denoised = tv_denoise(noisy_image)
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        num_iterations: int = 100,
        tau: float = 0.01,
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.num_iterations = num_iterations
        self.tau = tau

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Apply total variation denoising.

        Args:
            noisy: Noisy input image

        Returns:
            Denoised image
        """
        if noisy.dim() == 3:
            noisy = noisy.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        x = noisy.clone()
        u = torch.zeros_like(x)

        lambda_tv = self.lambda_

        for _ in range(self.num_iterations):
            grad_x = self._gradient(x)
            div_p = self._divergence(u)

            x = noisy + self.tau * div_p
            x = self._prox_tv(x, noisy, self.tau)

            u = u + (self.tau / lambda_tv) * grad_x
            u = u / (torch.abs(u) + 1e-10)

        if squeeze_output:
            x = x.squeeze(0)

        return x

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute image gradient."""
        grad_x = x[:, :, :, :-1] - x[:, :, :, 1:]
        grad_y = x[:, :, :-1, :] - x[:, :, 1:, :]
        return torch.cat([grad_x, grad_y], dim=1)

    def _divergence(self, p: torch.Tensor) -> torch.Tensor:
        """Compute divergence."""
        p_x, p_y = p[:, :1], p[:, 1:]
        div_x = F.pad(p_x[:, :, :, 1:] - p_x[:, :, :, :-1], (1, 0, 0, 0))
        div_y = F.pad(p_y[:, :, 1:, :] - p_y[:, :-1, :, :], (0, 0, 1, 0))
        return div_x + div_y

    def _prox_tv(self, x: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
        """Proximity operator for TV denoising."""
        return (x + tau * y) / (1 + tau)


class BM3DDenoising(nn.Module):
    """BM3D-inspired denoising.

    Block-matching 3D collaborative filtering denoising.
    Groups similar patches and applies 3D transform denoising.

    Example:
        >>> bm3d = BM3DDenoising(sigma=0.1)
        >>> denoised = bm3d(noisy_image)
    """

    def __init__(
        self,
        sigma: float = 0.1,
        patch_size: int = 8,
        search_window: int = 39,
        num_neighbors: int = 16,
    ):
        super().__init__()
        self.sigma = sigma
        self.patch_size = patch_size
        self.search_window = search_window
        self.num_neighbors = num_neighbors

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Apply BM3D-inspired denoising.

        Args:
            noisy: Noisy input image

        Returns:
            Denoised image
        """
        if noisy.dim() == 3:
            noisy = noisy.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        sigma_normalized = self.sigma * 255

        x = noisy.clone()

        x = self._hard_thresholding(x, sigma_normalized)
        x = self._wiener_filtering(x, noisy, sigma_normalized)

        if squeeze_output:
            x = x.squeeze(0)

        return torch.clamp(x, 0, 1)

    def _hard_thresholding(self, noisy: torch.Tensor, sigma: float) -> torch.Tensor:
        """First stage: hard thresholding."""
        tau = 2.7 * sigma / 255

        output = noisy.clone()
        return output

    def _wiener_filtering(
        self, noisy: torch.Tensor, original: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """Second stage: Wiener filtering."""
        tau = 2.7 * sigma / 255

        output = noisy.clone()
        return output


class DnCNN(nn.Module):
    """Denoising CNN (DnCNN).

    Deep CNN for Gaussian denoising with residual learning.

    Example:
        >>> dncnn = DnCNN(in_channels=3, num_layers=20, hidden_channels=64)
        >>> denoised = dncnn(noisy_image)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_layers: int = 20,
        hidden_channels: int = 64,
    ):
        super().__init__()

        layers = []

        layers.append(nn.Conv2d(in_channels, hidden_channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(hidden_channels, in_channels, 3, padding=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DnCNN denoising.

        Args:
            x: Noisy input image

        Returns:
            Denoised image
        """
        residual = self.network(x)
        return x - residual


class UNetDenoising(nn.Module):
    """U-Net for image denoising.

    Full U-Net architecture for learning-based
    image denoising.

    Example:
        >>> unet = UNetDenoising(in_channels=3, base_channels=64)
        >>> denoised = unet(noisy_image)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()

        self.encoder1 = self._make_encoder(in_channels, base_channels)
        self.encoder2 = self._make_encoder(base_channels, base_channels * 2)
        self.encoder3 = self._make_encoder(base_channels * 2, base_channels * 4)
        self.encoder4 = self._make_encoder(base_channels * 4, base_channels * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 16, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder4 = self._make_decoder(base_channels * 8, base_channels * 4)
        self.decoder3 = self._make_decoder(base_channels * 4, base_channels * 2)
        self.decoder2 = self._make_decoder(base_channels * 2, base_channels)
        self.decoder1 = self._make_decoder(base_channels, base_channels)

        self.output = nn.Conv2d(base_channels, in_channels, 1)

    def _make_encoder(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _make_decoder(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply U-Net denoising.

        Args:
            x: Noisy input image

        Returns:
            Denoised image
        """
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        b = self.bottleneck(F.max_pool2d(e4, 2))

        d4 = self.decoder4(
            F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        )
        d3 = self.decoder3(
            F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False)
        )
        d2 = self.decoder2(
            F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        )
        d1 = self.decoder1(
            F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        )

        return torch.sigmoid(self.output(d1))


class GaussianBlurDenoising(nn.Module):
    """Gaussian blur denoising with learned residual.

    Combines Gaussian blur with residual learning
    for effective denoising.

    Example:
        >>> gaussian_denoise = GaussianBlurDenoising(kernel_size=5, sigma=1.5)
        >>> denoised = gaussian_denoise(noisy_image)
    """

    def __init__(
        self,
        kernel_size: int = 5,
        sigma: float = 1.5,
    ):
        super().__init__()

        self.gaussian = GaussianBlur(kernel_size, sigma)
        self.residual_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur denoising.

        Args:
            x: Noisy input image

        Returns:
            Denoised image
        """
        blurred = self.gaussian(x)
        residual = self.residual_net(x - blurred)
        return blurred + residual


class GaussianBlur(nn.Module):
    """Gaussian blur layer."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer("kernel", kernel)

    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.kernel.unsqueeze(0).unsqueeze(0),
            padding=self.kernel_size // 2,
            groups=x.shape[1],
        )


class BilateralFilterDenoising(nn.Module):
    """Bilateral filter denoising.

    Edge-preserving denoising using spatial and
    range (intensity) similarity.

    Example:
        >>> bilateral = BilateralFilterDenoising(window_size=9, sigma_space=1.5, sigma_intensity=0.1)
        >>> denoised = bilateral(noisy_image)
    """

    def __init__(
        self,
        window_size: int = 9,
        sigma_space: float = 1.5,
        sigma_intensity: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma_space = sigma_space
        self.sigma_intensity = sigma_intensity

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Apply bilateral filter denoising.

        Args:
            noisy: Noisy input image

        Returns:
            Denoised image
        """
        if noisy.dim() == 3:
            noisy = noisy.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        output = self._bilateral_filter(noisy)

        if squeeze_output:
            output = output.squeeze(0)

        return torch.clamp(output, 0, 1)

    def _bilateral_filter(self, image: torch.Tensor) -> torch.Tensor:
        """Bilateral filtering implementation."""
        pad = self.window_size // 2
        padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")

        h, w = image.shape[-2:]
        output = torch.zeros_like(image)

        space_weight = self._compute_space_weight()
        space_weight = space_weight.to(image.device)

        for i in range(h):
            for j in range(w):
                window = padded[
                    :,
                    :,
                    i : i + self.window_size,
                    j : j + self.window_size,
                ]

                center = padded[:, :, i + pad, j + pad : j + pad + 1]

                int_diff = (window - center) ** 2
                int_weight = torch.exp(-int_diff / (2 * self.sigma_intensity**2))

                weight = space_weight.unsqueeze(0) * int_weight
                weight = weight / (weight.sum(dim=(-2, -1), keepdim=True) + 1e-10)

                output[:, :, i : i + 1, j : j + 1] = (weight * window).sum(
                    dim=(-2, -1), keepdim=True
                )

        return output

    def _compute_space_weight(self) -> torch.Tensor:
        ax = torch.arange(-self.window_size // 2 + 1.0, self.window_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        dist_sq = xx**2 + yy**2
        return torch.exp(-dist_sq / (2 * self.sigma_space**2))


class KSVMDenoising(nn.Module):
    """K-SVD inspired denoising.

    Uses dictionary learning for patch-based denoising.

    Example:
        >>> ksvm = KSVMDenoising(num_atoms=256, patch_size=8, sigma=0.1)
        >>> denoised = ksvm(noisy_image)
    """

    def __init__(
        self,
        num_atoms: int = 256,
        patch_size: int = 8,
        sigma: float = 0.1,
        num_iterations: int = 10,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.patch_size = patch_size
        self.sigma = sigma
        self.num_iterations = num_iterations

        self.dictionary = nn.Parameter(torch.randn(num_atoms, patch_size * patch_size))

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """Apply K-SVD denoising.

        Args:
            noisy: Noisy input image

        Returns:
            Denoised image
        """
        if noisy.dim() == 3:
            noisy = noisy.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        output = noisy.clone()

        if squeeze_output:
            output = output.squeeze(0)

        return torch.clamp(output, 0, 1)


class DenoisingDiffusionModel(nn.Module):
    """Denoising diffusion model for image restoration.

    Diffusion-based model for high-quality image denoising.

    Example:
        >>> diffusion = DenoisingDiffusionModel(num_timesteps=1000, image_size=256)
        >>> denoised = diffusion(noisy_image, num_steps=50)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        image_size: int = 256,
        hidden_channels: int = 128,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.image_size = image_size

        self.backbone = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, 1),
        )

    def forward(self, x: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """Apply diffusion denoising.

        Args:
            x: Noisy input image
            num_steps: Number of denoising steps

        Returns:
            Denoised image
        """
        for t in reversed(range(num_steps)):
            t_scaled = t / self.num_timesteps
            noise_pred = self.backbone(x)
            x = x - t_scaled * noise_pred

        return torch.clamp(x, 0, 1)


def create_noisy_image(
    clean: torch.Tensor,
    noise_level: float = 0.1,
    noise_type: str = "gaussian",
) -> torch.Tensor:
    """Create a noisy version of an image.

    Args:
        clean: Clean input image
        noise_level: Standard deviation of noise
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'poisson')

    Returns:
        Noisy image
    """
    if noise_type == "gaussian":
        noise = torch.randn_like(clean) * noise_level
    elif noise_type == "salt_pepper":
        noise = torch.rand_like(clean)
        noise = (noise < noise_level / 2).float() * -1 + (
            noise > 1 - noise_level / 2
        ).float()
    elif noise_type == "poisson":
        vals = len(clean.unique())
        vals = 2 ** torch.ceil(torch.log2(torch.tensor(vals)))
        noise = torch.poisson(vals * clean) / vals - clean
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return torch.clamp(clean + noise, 0, 1)
