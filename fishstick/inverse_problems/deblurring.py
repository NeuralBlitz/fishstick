"""
Image Deblurring Module

Implements image deblurring algorithms including:
- Richardson-Lucy deconvolution
- Wiener filter
- Blind deconvolution
- Deep learning-based deblurring
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RichardsonLucyDeconvolution(nn.Module):
    """Richardson-Lucy iterative deconvolution.

    Maximum likelihood algorithm for image deconvolution
    assuming Poisson noise statistics.

    Solves: argmax_x P(y|x) for observed y and blur kernel

    Example:
        >>> kernel = BlurKernel(kernel_size=11, kernel_type='gaussian', sigma=3.0)
        >>> rl = RichardsonLucyDeconvolution(kernel, num_iterations=30)
        >>> deblurred = rl(blurred_image)
    """

    def __init__(
        self,
        kernel: torch.Tensor,
        num_iterations: int = 30,
        clip_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.kernel = kernel
        self.num_iterations = num_iterations
        self.clip_range = clip_range

    def forward(self, blurred: torch.Tensor) -> torch.Tensor:
        """Apply Richardson-Lucy deconvolution.

        Args:
            blurred: Blurred input image (batch, channels, height, width)

        Returns:
            Deblurred image
        """
        if blurred.dim() == 3:
            blurred = blurred.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        kernel = self.kernel.to(blurred.device)
        if kernel.dim() == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0)

        x = blurred.clone()

        for _ in range(self.num_iterations):
            blurred_est = self._convolve(x, kernel)
            ratio = blurred / (blurred_est + 1e-10)
            correction = self._convolve(ratio, self._flip_kernel(kernel))
            x = x * correction

            if self.clip_range:
                x = torch.clamp(x, self.clip_range[0], self.clip_range[1])

        if squeeze_output:
            x = x.squeeze(0)

        return x

    def _convolve(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Convolve image with kernel."""
        return F.conv2d(
            image,
            kernel,
            padding=kernel.shape[-1] // 2,
            groups=image.shape[1],
        )

    def _flip_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        """Flip kernel for correlation."""
        return torch.flip(kernel, dims=[-1, -2])


class WienerFilter(nn.Module):
    """Wiener filter for image deconvolution.

    Optimal linear filter for deconvolution with
    additive Gaussian noise.

    Solves: G(f) = H*(f) / (|H(f)|^2 + Sn/Sf)

    Example:
        >>> wiener = WienerFilter(kernel, noise_variance=0.01, signal_variance=1.0)
        >>> deblurred = wiener(blurred_image)
    """

    def __init__(
        self,
        kernel: torch.Tensor,
        noise_variance: float = 0.01,
        signal_variance: float = 1.0,
    ):
        super().__init__()
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.signal_variance = signal_variance

    def forward(self, blurred: torch.Tensor) -> torch.Tensor:
        """Apply Wiener filter deconvolution.

        Args:
            blurred: Blurred input image

        Returns:
            Deblurred image
        """
        if blurred.dim() == 3:
            blurred = blurred.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        device = blurred.device

        fft_blurred = self._fft2(blurred)
        fft_kernel = self._fft2(self.kernel.to(device).unsqueeze(0).unsqueeze(0))

        kernel_magnitude_sq = torch.abs(fft_kernel) ** 2
        wiener_filter = torch.conj(fft_kernel) / (
            kernel_magnitude_sq + self.noise_variance / (self.signal_variance + 1e-10)
        )

        fft_deblurred = fft_blurred * wiener_filter
        deblurred = self._ifft2(fft_deblurred)

        if squeeze_output:
            deblurred = deblurred.squeeze(0)

        return torch.clamp(deblurred, 0, 1)

    def _fft2(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 2D FFT."""
        return torch.fft.fft2(x, dim=(-2, -1))

    def _ifft2(self, x: torch.Tensor) -> torch.Tensor:
        """Compute inverse 2D FFT."""
        return torch.fft.ifft2(x, dim=(-2, -1)).real


class BlindDeblurring(nn.Module):
    """Blind deconvolution for unknown blur kernels.

    Iteratively estimates both the blur kernel
    and the sharp image.

    Example:
        >>> blind = BlindDeblurring(kernel_size=11, num_iterations=20)
        >>> deblurred, kernel_est = blind(blurred_image)
    """

    def __init__(
        self,
        kernel_size: int = 11,
        num_iterations: int = 20,
        reg_lambda: float = 0.01,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_iterations = num_iterations
        self.reg_lambda = reg_lambda

    def forward(
        self,
        blurred: torch.Tensor,
        initial_kernel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply blind deconvolution.

        Args:
            blurred: Blurred input image
            initial_kernel: Initial kernel estimate (optional)

        Returns:
            Tuple of (deblurred image, estimated kernel)
        """
        if blurred.dim() == 3:
            blurred = blurred.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        device = blurred.device

        if initial_kernel is None:
            kernel = self._create_initial_kernel(device)
        else:
            kernel = initial_kernel.to(device)

        x = blurred.clone()

        for _ in range(self.num_iterations):
            x = self._estimate_image(blurred, x, kernel)
            kernel = self._estimate_kernel(blurred, x, kernel)

        kernel = kernel / kernel.sum()

        if squeeze_output:
            x = x.squeeze(0)

        return x, kernel

    def _create_initial_kernel(self, device: torch.Tensor) -> torch.Tensor:
        """Create initial kernel estimate."""
        kernel = torch.zeros(self.kernel_size, self.kernel_size, device=device)
        kernel[self.kernel_size // 2, self.kernel_size // 2] = 1.0
        return kernel

    def _estimate_image(
        self,
        blurred: torch.Tensor,
        x: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate sharp image given kernel."""
        rl = RichardsonLucyDeconvolution(kernel, num_iterations=5)
        return rl(blurred)

    def _estimate_kernel(
        self,
        blurred: torch.Tensor,
        x: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate blur kernel given image."""
        device = blurred.device
        blurred_fft = self._fft2(blurred)
        x_fft = self._fft2(x)

        kernel_fft = torch.conj(x_fft) * blurred_fft / (torch.abs(x_fft) ** 2 + 1e-10)
        kernel_new = self._ifft2(kernel_fft)

        kernel_new = torch.real(kernel_new)

        pad_h = (self.kernel_size - kernel_new.shape[-2]) // 2
        pad_w = (self.kernel_size - kernel_new.shape[-1]) // 2
        kernel_new = F.pad(kernel_new, (pad_w, pad_w, pad_h, pad_h))

        kernel_new = torch.clamp(kernel_new, min=0)
        kernel_new = kernel_new / (kernel_new.sum() + 1e-10)

        return kernel_new

    def _fft2(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 2D FFT."""
        return torch.fft.fft2(x, dim=(-2, -1))

    def _ifft2(self, x: torch.Tensor) -> torch.Tensor:
        """Compute inverse 2D FFT."""
        return torch.fft.ifft2(x, dim=(-2, -1))


class DeepDeblurringNetwork(nn.Module):
    """Deep learning-based image deblurring network.

    U-Net architecture with residual learning for
    efficient image deblurring.

    Example:
        >>> deblur_net = DeepDeblurringNetwork(in_channels=3, hidden_channels=64)
        >>> deblurred = deblur_net(blurred_image)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_blocks: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        ch = hidden_channels

        self.input_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        for _ in range(num_blocks):
            self.residual_blocks.append(ResidualBlock(ch))

        self.output_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply deep deblurring.

        Args:
            x: Blurred input image

        Returns:
            Deblurred image
        """
        residual = x

        x = self.input_conv(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.output_conv(x)

        return x + residual


class ResidualBlock(nn.Module):
    """Residual block for deblurring network."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


class DeblurringWithMotionKernel(nn.Module):
    """Motion blur deblurring with learnable kernel.

    Specialized deblurring for motion blur with
    learnable directional kernel estimation.

    Example:
        >>> motion_deblur = DeblurringWithMotionKernel(kernel_length=21)
        >>> deblurred = motion_deblur(blurred_image)
    """

    def __init__(
        self,
        kernel_length: int = 21,
        num_iterations: int = 30,
    ):
        super().__init__()
        self.kernel_length = kernel_length
        self.num_iterations = num_iterations

        self.kernel_angle = nn.Parameter(torch.tensor(0.0))
        self.kernel_length_param = nn.Parameter(torch.tensor(float(kernel_length // 2)))

    def forward(self, blurred: torch.Tensor) -> torch.Tensor:
        """Apply motion deblurring.

        Args:
            blurred: Motion blurred image

        Returns:
            Deblurred image
        """
        kernel = self._create_motion_kernel(blurred.device)
        rl = RichardsonLucyDeconvolution(kernel, num_iterations=self.num_iterations)
        return rl(blurred)

    def _create_motion_kernel(self, device: torch.Tensor) -> torch.Tensor:
        """Create motion blur kernel based on learned parameters."""
        length = int(torch.clamp(self.kernel_length_param, 1, self.kernel_length))
        angle = self.kernel_angle.item()

        kernel = torch.zeros(self.kernel_length, self.kernel_length, device=device)
        cx, cy = self.kernel_length // 2, self.kernel_length // 2
        angle_rad = angle * np.pi / 180

        for i in range(-length, length + 1):
            x = int(cx + i * np.cos(angle_rad))
            y = int(cy + i * np.sin(angle_rad))
            if 0 <= x < self.kernel_length and 0 <= y < self.kernel_length:
                kernel[y, x] = 1.0

        kernel = kernel / (kernel.sum() + 1e-10)
        return kernel


class MultiScaleDeblurring(nn.Module):
    """Multi-scale image deblurring.

    Coarse-to-fine deblurring at multiple resolutions
    for handling large blur kernels.

    Example:
        >>> multi_scale = MultiScaleDeblurring(num_scales=3, num_iterations=20)
        >>> deblurred = multi_scale(blurred_image)
    """

    def __init__(
        self,
        num_scales: int = 3,
        num_iterations: int = 20,
        kernel_size: int = 11,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_iterations = num_iterations

        self.deblurring_net = DeepDeblurringNetwork()

    def forward(self, blurred: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale deblurring.

        Args:
            blurred: Blurred input image

        Returns:
            Deblurred image at original resolution
        """
        if blurred.dim() == 3:
            blurred = blurred.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        h, w = blurred.shape[-2:]

        current = blurred

        for scale in range(self.num_scales):
            scale_factor = 2 ** (self.num_scales - scale - 1)
            if scale > 0:
                current = F.interpolate(
                    current,
                    scale_factor=0.5,
                    mode="bilinear",
                    align_corners=False,
                )

            current = self.deblurring_net(current)

            if scale < self.num_scales - 1:
                current = F.interpolate(
                    current,
                    size=(h // scale_factor, w // scale_factor),
                    mode="bilinear",
                    align_corners=False,
                )

        if squeeze_output:
            current = current.squeeze(0)

        return torch.clamp(current, 0, 1)


class KernelEstimationNetwork(nn.Module):
    """Network for estimating blur kernels.

    Learns to predict blur kernels from blurred images
    for use in deconvolution.

    Example:
        >>> kernel_net = KernelEstimationNetwork(image_size=(256, 256), kernel_size=21)
        >>> estimated_kernel = kernel_net(blurred_image)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        kernel_size: int = 21,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.image_size = image_size
        self.kernel_size = kernel_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels * 2, hidden_channels, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, kernel_size * kernel_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate blur kernel.

        Args:
            x: Blurred image

        Returns:
            Estimated blur kernel (kernel_size, kernel_size)
        """
        features = self.encoder(x)
        kernel_flat = self.decoder(features)
        kernel = kernel_flat.view(-1, self.kernel_size, self.kernel_size)
        kernel = F.softmax(
            kernel_flat.view(-1, self.kernel_size * self.kernel_size), dim=-1
        )
        kernel = kernel.view(-1, 1, self.kernel_size, self.kernel_size)
        return kernel.squeeze(0).squeeze(0)


def create_blur_kernel(
    kernel_type: str = "gaussian",
    kernel_size: int = 11,
    sigma: float = 3.0,
    angle: float = 0.0,
) -> torch.Tensor:
    """Create a blur kernel.

    Args:
        kernel_type: Type of kernel ('gaussian', 'motion', 'disk')
        kernel_size: Size of the kernel
        sigma: Standard deviation for gaussian/disk
        angle: Angle for motion blur

    Returns:
        Blur kernel tensor
    """
    if kernel_type == "gaussian":
        ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    elif kernel_type == "motion":
        kernel = torch.zeros(kernel_size, kernel_size)
        cx, cy = kernel_size // 2, kernel_size // 2
        angle_rad = angle * np.pi / 180

        length = kernel_size // 2
        for i in range(-length, length + 1):
            x = int(cx + i * np.cos(angle_rad))
            y = int(cy + i * np.sin(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
    elif kernel_type == "disk":
        ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        r = torch.sqrt(xx**2 + yy**2)
        kernel = (r <= sigma).float()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return kernel / kernel.sum()
