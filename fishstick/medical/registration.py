"""
Medical Image Registration
"""

from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ImageRegistration:
    """Image registration for medical images."""

    def __init__(self):
        pass

    def rigid_registration(
        self,
        fixed: Tensor,
        moving: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Perform rigid registration."""
        # Placeholder for rigid registration
        # In practice, would use optimization to find best rotation/translation
        return moving, torch.eye(4)

    def deformable_registration(
        self,
        fixed: Tensor,
        moving: Tensor,
    ) -> Tensor:
        """Perform deformable registration using a learned model."""
        # This would typically use a U-Net to predict displacement field
        return moving


class SpatialTransformer(nn.Module):
    """Spatial transformer network for image registration."""

    def __init__(self, size: Tuple[int, ...]):
        super().__init__()
        self.size = size

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = grid.unsqueeze(0).float()

        self.register_buffer("grid", grid)

    def forward(self, src: Tensor, flow: Tensor) -> Tensor:
        """
        Warp image using flow field.

        Args:
            src: Source image [N, C, D, H, W] or [N, C, H, W]
            flow: Displacement field [N, 3, D, H, W] or [N, 2, H, W]
        """
        # Normalize flow to [-1, 1]
        new_locs = self.grid + flow

        for i in range(len(self.size)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (self.size[i] - 1) - 0.5)

        # Move channel dim to last for grid_sample
        if len(self.size) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]  # Swap x and y for grid_sample
        else:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]  # Swap for 3D

        # Sample
        warped = F.grid_sample(
            src,
            new_locs,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return warped


class VoxelMorph(nn.Module):
    """VoxelMorph network for unsupervised medical image registration."""

    def __init__(
        self,
        in_channels: int = 2,
        enc_features: list = [16, 32, 32, 32],
        dec_features: list = [32, 32, 32, 32, 32, 16, 16],
    ):
        super().__init__()

        # U-Net encoder
        self.encoder = nn.ModuleList()
        for i, features in enumerate(enc_features):
            in_ch = in_channels if i == 0 else enc_features[i - 1]
            self.encoder.append(self._conv_block(in_ch, features))

        # U-Net decoder
        self.decoder = nn.ModuleList()
        for i, features in enumerate(dec_features):
            if i < len(enc_features):
                in_ch = (
                    enc_features[-(i + 1)] + dec_features[i - 1]
                    if i > 0
                    else enc_features[-1]
                )
            else:
                in_ch = dec_features[i - 1]
            self.decoder.append(self._conv_block(in_ch, features))

        # Flow prediction
        self.flow = nn.Conv3d(dec_features[-1], 3, kernel_size=3, padding=1)

        # Initialize flow layer with small weights
        self.flow.weight = nn.Parameter(
            torch.randn(3, dec_features[-1], 3, 3, 3) * 1e-5
        )
        self.flow.bias = nn.Parameter(torch.zeros(3))

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, moving: Tensor, fixed: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            moving: Moving image [N, 1, D, H, W]
            fixed: Fixed image [N, 1, D, H, W]

        Returns:
            warped: Registered moving image
            flow: Displacement field
        """
        x = torch.cat([moving, fixed], dim=1)

        # Encoder
        skip_connections = []
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = F.avg_pool3d(x, 2)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i, decode in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            if i < len(skip_connections):
                x = torch.cat([x, skip_connections[i]], dim=1)
            x = decode(x)

        flow = self.flow(x)

        # Warp moving image
        transformer = SpatialTransformer(moving.shape[2:])
        warped = transformer(moving, flow)

        return warped, flow


class RegistrationLoss(nn.Module):
    """Loss function for image registration."""

    def __init__(
        self,
        similarity_weight: float = 1.0,
        smoothness_weight: float = 0.01,
    ):
        super().__init__()
        self.similarity_weight = similarity_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        fixed: Tensor,
        warped: Tensor,
        flow: Tensor,
    ) -> Tensor:
        """Compute registration loss."""
        # Similarity loss (MSE)
        similarity_loss = F.mse_loss(warped, fixed)

        # Smoothness loss (gradient penalty)
        smoothness_loss = self._gradient_loss(flow)

        return (
            self.similarity_weight * similarity_loss
            + self.smoothness_weight * smoothness_loss
        )

    def _gradient_loss(self, flow: Tensor) -> Tensor:
        """Compute gradient penalty for smoothness."""
        # Compute spatial gradients
        if flow.ndim == 5:  # 3D
            dx = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
            dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
            dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
            return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean()) / 3
        else:  # 2D
            dx = flow[:, :, 1:, :] - flow[:, :, :-1, :]
            dy = flow[:, :, :, 1:] - flow[:, :, :, :-1]
            return (dx.abs().mean() + dy.abs().mean()) / 2
