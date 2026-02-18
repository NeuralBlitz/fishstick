"""
Depth-specific Loss Functions

SSIM, smoothness, reconstruction losses for depth estimation.
"""

from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, 1)

    def _gaussian(self, window_size: int, sigma: float) -> Tensor:
        gauss = torch.Tensor(
            [
                torch.exp(
                    torch.tensor(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                )
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def _create_window(self, window_size: int, channel: int) -> Tensor:
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        """
        Compute SSIM loss.

        Args:
            img1: First image [B, C, H, W]
            img2: Second image [B, C, H, W]

        Returns:
            SSIM loss
        """
        if img1.shape[1] != self.channel:
            self.channel = img1.shape[1]
            self.window = self._create_window(self.window_size, self.channel)

        window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(
                img1 * img1, window, padding=self.window_size // 2, groups=self.channel
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                img2 * img2, window, padding=self.window_size // 2, groups=self.channel
            )
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                img1 * img2, window, padding=self.window_size // 2, groups=self.channel
            )
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return 1.0 - ssim_map.mean()
        return 1.0 - ssim_map.mean(dim=(1, 2, 3))


class SmoothnessLoss(nn.Module):
    """
    Edge-aware smoothness loss for depth/disparity.
    """

    def __init__(self):
        super().__init__()

    def forward(self, disp: Tensor, img: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            disp: Disparity map [B, 1, H, W]
            img: Reference image [B, 3, H, W]

        Returns:
            Smoothness loss
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        if img is not None:
            grad_img_x = torch.mean(
                torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True
            )
            grad_img_y = torch.mean(
                torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True
            )

            grad_disp_x = grad_disp_x / (grad_img_x + 1e-7)
            grad_disp_y = grad_disp_y / (grad_img_y + 1e-7)

        return grad_disp_x.mean() + grad_disp_y.mean()


class DisparitySmoothnessLoss(nn.Module):
    """
    Disparity smoothness loss with edge-awareness.
    """

    def __init__(self):
        super().__init__()
        self.smoothness = SmoothnessLoss()

    def forward(self, disparity: Tensor, image: Tensor) -> Tensor:
        """
        Args:
            disparity: Predicted disparity [B, 1, H, W]
            image: Input image [B, 3, H, W]

        Returns:
            Smoothness loss
        """
        return self.smoothness(disparity, image)


class ReconstructionLoss(nn.Module):
    """
    Photo-metric reconstruction loss for depth.
    """

    def __init__(self):
        super().__init__()
        self.ssim = SSIMLoss()

    def forward(
        self,
        target: Tensor,
        predicted: Tensor,
        use_ssim: bool = True,
    ) -> Tensor:
        """
        Args:
            target: Target image [B, 3, H, W]
            predicted: Predicted/warped image [B, 3, H, W]
            use_ssim: Whether to include SSIM loss

        Returns:
            Reconstruction loss
        """
        photo_loss = torch.abs(target - predicted).mean()

        if use_ssim:
            ssim_loss = self.ssim(target, predicted)
            return photo_loss + 0.85 * ssim_loss

        return photo_loss


class DepthLoss(nn.Module):
    """
    Combined depth/depth estimation loss.
    """

    def __init__(
        self,
        smoothness_weight: float = 0.001,
        ssim_weight: float = 0.85,
    ):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.ssim_weight = ssim_weight
        self.ssim = SSIMLoss()
        self.smoothness = SmoothnessLoss()

    def forward(
        self,
        disparities: list,
        target_image: Tensor,
    ) -> Tensor:
        """
        Args:
            disparities: List of disparity predictions at different scales
            target_image: Target RGB image [B, 3, H, W]

        Returns:
            Total loss
        """
        total_loss = 0

        for disp in disparities:
            recon_loss = torch.abs(disp - target_image).mean()
            smooth_loss = self.smoothness(disp, target_image)
            ssim_loss = self.ssim(target_image, disp)

            total_loss += (
                recon_loss
                + self.ssim_weight * ssim_loss
                + self.smoothness_weight * smooth_loss
            )

        return total_loss
