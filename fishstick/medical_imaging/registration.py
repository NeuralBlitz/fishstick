"""
Medical Image Registration Module

VoxelMorph, deformable registration, and similarity metrics.
"""

from typing import Optional, List, Tuple, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VoxelMorph(nn.Module):
    """VoxelMorph: Learning deformable image registration.

    Based on the paper: "VoxelMorph: A Learning Framework for Deformable Image Registration"
    """

    def __init__(
        self,
        in_channels: int = 2,
        nb_features: List[int] = None,
        nb_conv_per_level: int = 3,
        int_steps: int = 7,
    ):
        super().__init__()

        if nb_features is None:
            nb_features = [32, 64, 96, 128, 256]

        self.int_steps = int_steps

        self.encoder = self._build_encoder(in_channels, nb_features, nb_conv_per_level)
        self.decoder = self._build_decoder(nb_features)

        self.flow = nn.Conv3d(nb_features[-1], 3, kernel_size=3, padding=1)

        self.integrate = self._integrate_velocities

    def _build_encoder(
        self,
        in_channels: int,
        nb_features: List[int],
        nb_conv_per_level: int,
    ) -> nn.Module:
        layers = []
        current_channels = in_channels

        for n_features in nb_features:
            for _ in range(nb_conv_per_level):
                layers.append(
                    nn.Sequential(
                        nn.Conv3d(
                            current_channels,
                            n_features,
                            kernel_size=3,
                            stride=2 if _ == 0 else 1,
                            padding=1,
                        ),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
                )
                current_channels = n_features

        return nn.Sequential(*layers)

    def _build_decoder(self, nb_features: List[int]) -> nn.Module:
        layers = []

        for i in range(len(nb_features) - 1, 0, -1):
            n_features = nb_features[i]
            n_to_features = nb_features[i - 1]

            layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        n_features, n_to_features, kernel_size=2, stride=2
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(n_to_features, n_to_features, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.cat([source, target], dim=1)

        x = self.encoder(x)
        x = self.decoder(x)

        flow = self.flow(x)

        if self.training:
            warped = self._warp(source, flow)
            return warped, flow

        warped = self._warp(source, flow)
        return warped, flow

    def _warp(self, x: Tensor, flow: Tensor) -> Tensor:
        b, c, d, h, w = x.shape

        grid = self._get_grid(b, d, h, w, x.device, x.dtype)

        flow = F.interpolate(flow, size=(d, h, w), mode="trilinear", align_corners=True)

        new_grid = grid + flow
        new_grid = new_grid.permute(0, 2, 3, 4, 1)
        new_grid = new_grid * 2 - 1

        warped = F.grid_sample(
            x, new_grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        return warped

    def _get_grid(
        self,
        batch: int,
        depth: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        grid_x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        grid_y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        grid_z = torch.linspace(-1, 1, depth, device=device, dtype=dtype)

        grid = torch.stack(
            [
                grid_x.unsqueeze(0).unsqueeze(0).expand(batch, depth, height, -1),
                grid_y.unsqueeze(0).unsqueeze(2).expand(batch, depth, -1, width),
                grid_z.unsqueeze(0).unsqueeze(3).expand(batch, -1, height, width),
            ],
            dim=4,
        )

        return grid

    def _integrate_velocities(self, flow: Tensor) -> Tensor:
        return flow


class VoxelMorphLoss(nn.Module):
    """Combined loss for VoxelMorph training."""

    def __init__(
        self,
        ncc_weight: float = 1.0,
        grad_weight: float = 0.5,
        mse_weight: float = 0.0,
    ):
        super().__init__()
        self.ncc_weight = ncc_weight
        self.grad_weight = grad_weight
        self.mse_weight = mse_weight

        self.ncc = NCCLoss()
        self.mse = MSELoss()
        self.grad = GradientLoss()

    def forward(
        self,
        warped: Tensor,
        target: Tensor,
        flow: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        loss = {}

        if self.ncc_weight > 0:
            ncc_loss_val = self.ncc(warped, target)
            loss["ncc"] = ncc_loss_val
            loss["total"] = (
                loss["total"] + self.ncc_weight * ncc_loss_val
                if "total" in loss
                else self.ncc_weight * ncc_loss_val
            )

        if self.mse_weight > 0:
            mse_loss_val = self.mse(warped, target)
            loss["mse"] = mse_loss_val
            loss["total"] = loss.get("total", 0) + self.mse_weight * mse_loss_val

        if self.grad_weight > 0 and flow is not None:
            grad_loss_val = self.grad(flow)
            loss["grad"] = grad_loss_val
            loss["total"] = loss.get("total", 0) + self.grad_weight * grad_loss_val

        if "total" not in loss:
            loss["total"] = torch.tensor(0.0, device=warped.device)

        return loss


class NCCLoss(nn.Module):
    """Normalized Cross Correlation loss."""

    def __init__(self, window_size: int = 9):
        super().__init__()
        self.window_size = window_size

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        source_mean = source.mean(dim=(2, 3, 4), keepdim=True)
        target_mean = target.mean(dim=(2, 3, 4), keepdim=True)

        source_centered = source - source_mean
        target_centered = target - target_mean

        source_std = torch.sqrt(
            (source_centered**2).mean(dim=(2, 3, 4), keepdim=True) + 1e-8
        )
        target_std = torch.sqrt(
            (target_centered**2).mean(dim=(2, 3, 4), keepdim=True) + 1e-8
        )

        ncc = (source_centered * target_centered).mean(dim=(2, 3, 4)) / (
            source_std.squeeze() * target_std.squeeze() + 1e-8
        )

        return 1 - ncc.mean()


class MSELoss(nn.Module):
    """Mean Squared Error loss."""

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(source, target)


class GradientLoss(nn.Module):
    """Gradient regularization loss."""

    def __init__(self, penalty: str = "l1"):
        super().__init__()
        self.penalty = penalty

    def forward(self, flow: Tensor) -> Tensor:
        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

        if self.penalty == "l1":
            return dy.mean() + dx.mean() + dz.mean()
        elif self.penalty == "l2":
            return (dy**2).mean() + (dx**2).mean() + (dz**2).mean()
        return dy.mean() + dx.mean() + dz.mean()


class AffineRegistration(nn.Module):
    """Affine image registration."""

    def __init__(self, num_iterations: int = 100):
        super().__init__()
        self.num_iterations = num_iterations

    def forward(self, source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        b, c, d, h, w = source.shape

        theta = torch.eye(b, 12, device=source.device)
        theta = theta + torch.randn(b, 12, device=source.device) * 0.01
        theta = theta.view(b, 3, 4)

        grid = F.affine_grid(theta, source.shape, align_corners=False)
        registered = F.grid_sample(source, grid, align_corners=False)

        return registered, theta


class DeformableRegistration(nn.Module):
    """Deformable image registration with UNet-based flow prediction."""

    def __init__(
        self,
        in_channels: int = 2,
        base_features: int = 32,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features, base_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(base_features, base_features * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features * 2, base_features * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_features * 2, base_features, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features * 2, base_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_features, base_features, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features * 2, base_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.flow = nn.Conv3d(base_features, 3, kernel_size=3, padding=1)

    def forward(self, source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.cat([source, target], dim=1)

        enc = self.encoder(x)
        dec = self.decoder(enc)

        flow = self.flow(dec)

        warped = self._warp(source, flow)

        return warped, flow

    def _warp(self, x: Tensor, flow: Tensor) -> Tensor:
        b, c, d, h, w = x.shape

        grid = torch.meshgrid(
            torch.linspace(-1, 1, w, device=x.device),
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, d, device=x.device),
            indexing="xy",
        )
        grid = torch.stack([grid[0], grid[1], grid[2]], dim=-1)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1, -1)

        new_grid = grid + flow.permute(0, 2, 3, 4, 1)
        new_grid = new_grid * 2 - 1

        warped = F.grid_sample(
            x, new_grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        return warped


class DemonsRegistration(nn.Module):
    """Demons-based deformable registration."""

    def __init__(
        self,
        num_iterations: int = 50,
        sigma_smooth: float = 1.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.sigma_smooth = sigma_smooth

    def forward(self, source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        flow = torch.zeros_like(source).unsqueeze(1).repeat(1, 3, 1, 1, 1)

        warped = source

        for _ in range(self.num_iterations):
            diff = target - warped
            grad = self._compute_gradient(warped)

            flow_update = diff / (grad.abs().sum(dim=1, keepdim=True) + 1e-8)
            flow = flow + flow_update * 0.5

            warped = self._warp(source, flow)

            flow = self._smooth(flow)

        final_warped = self._warp(source, flow)

        return final_warped, flow

    def _compute_gradient(self, x: Tensor) -> Tensor:
        return torch.gradient(x, dim=2)[0]

    def _warp(self, x: Tensor, flow: Tensor) -> Tensor:
        b, c, d, h, w = x.shape
        grid = torch.meshgrid(
            torch.linspace(-1, 1, w, device=x.device),
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, d, device=x.device),
            indexing="xy",
        )
        grid = torch.stack([grid[0], grid[1], grid[2]], dim=-1)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1, -1)

        new_grid = grid + flow.permute(0, 2, 3, 4, 1)
        new_grid = new_grid * 2 - 1

        return F.grid_sample(
            x, new_grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def _smooth(self, x: Tensor) -> Tensor:
        return F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)


class SymmetricNormalization(nn.Module):
    """Symmetric Normalization (SyN) registration."""

    def __init__(self, num_iterations: int = 20):
        super().__init__()
        self.num_iterations = num_iterations

    def forward(self, source: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        voxelmorph = VoxelMorph()

        warped_s_to_t, flow_s_to_t = voxelmorph(source, target)
        warped_t_to_s, flow_t_to_s = voxelmorph(target, source)

        symmetric_flow = (flow_s_to_t - flow_t_to_s) / 2

        warped = self._warp(source, symmetric_flow)

        return warped, symmetric_flow

    def _warp(self, x: Tensor, flow: Tensor) -> Tensor:
        b, c, d, h, w = x.shape

        grid = torch.meshgrid(
            torch.linspace(-1, 1, w, device=x.device),
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, d, device=x.device),
            indexing="xy",
        )
        grid = torch.stack([grid[0], grid[1], grid[2]], dim=-1)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1, -1)

        new_grid = grid + flow.permute(0, 2, 3, 4, 1)
        new_grid = new_grid * 2 - 1

        return F.grid_sample(
            x, new_grid, mode="bilinear", padding_mode="border", align_corners=True
        )


class RegistrationResult:
    """Container for registration results."""

    def __init__(
        self,
        warped: Tensor,
        flow: Tensor,
        transform: Optional[Tensor] = None,
    ):
        self.warped = warped
        self.flow = flow
        self.transform = transform


def ncc_loss(source: Tensor, target: Tensor) -> Tensor:
    """Normalized Cross Correlation loss function."""
    loss = NCCLoss()
    return loss(source, target)


def mse_loss(source: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error loss function."""
    return F.mse_loss(source, target)


def dice_score_registration(pred: Tensor, target: Tensor) -> Tensor:
    """Dice score for registration evaluation."""
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()

    return 2.0 * intersection / (union + 1e-8)


def compute_similarity_metric(
    source: Tensor,
    target: Tensor,
    metric: str = "ncc",
) -> float:
    """Compute similarity metric between two images."""
    if metric == "ncc":
        source_mean = source.mean()
        target_mean = target.mean()
        source_centered = source - source_mean
        target_centered = target - target_mean

        numerator = (source_centered * target_centered).sum()
        denominator = torch.sqrt(
            (source_centered**2).sum() * (target_centered**2).sum()
        )

        return (numerator / (denominator + 1e-8)).item()

    elif metric == "mse":
        return F.mse_loss(source, target).item()

    elif metric == "nmi":
        return 1.0 - compute_nmi(source, target)

    return 0.0


def compute_nmi(source: Tensor, target: Tensor, num_bins: int = 32) -> float:
    """Compute Normalized Mutual Information."""
    s_hist = torch.histc(source.flatten(), bins=num_bins)
    t_hist = torch.histc(target.flatten(), bins=num_bins)

    s_hist = s_hist / s_hist.sum()
    t_hist = t_hist / t_hist.sum()

    s_entropy = -(s_hist * torch.log(s_hist + 1e-8)).sum()
    t_entropy = -(t_hist * torch.log(t_hist + 1e-8)).sum()

    return 2.0 - (s_entropy + t_entropy) / num_bins


def apply_transform(
    image: Tensor,
    transform: Tensor,
    mode: str = "affine",
) -> Tensor:
    """Apply transformation to an image."""
    if mode == "affine":
        grid = F.affine_grid(transform, image.shape, align_corners=False)
        return F.grid_sample(image, grid, align_corners=False)
    elif mode == "dense":
        b, c, d, h, w = image.shape
        grid = torch.meshgrid(
            torch.linspace(-1, 1, w, device=image.device),
            torch.linspace(-1, 1, h, device=image.device),
            torch.linspace(-1, 1, d, device=image.device),
            indexing="xy",
        )
        grid = torch.stack([grid[0], grid[1], grid[2]], dim=-1)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1, -1)

        new_grid = grid + transform.permute(0, 2, 3, 4, 1)
        new_grid = new_grid * 2 - 1

        return F.grid_sample(
            image, new_grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    return image


def compose_transforms(transforms: List[Tensor], mode: str = "affine") -> Tensor:
    """Compose multiple transformations."""
    if mode == "affine":
        composed = transforms[0]
        for t in transforms[1:]:
            composed = torch.matmul(composed, t)
        return composed
    return transforms[-1]


def compute_jacobian_determinant(flow: Tensor) -> Tensor:
    """Compute Jacobian determinant of a deformation field."""
    b, _, d, h, w = flow.shape

    grad = torch.gradient(flow, dim=(2, 3, 4))

    dx_dx = grad[0][:, 0]
    dy_dy = grad[1][:, 1]
    dz_dz = grad[2][:, 2]

    jacobian = dx_dx + dy_dy + dz_dz

    return jacobian


__all__ = [
    "VoxelMorph",
    "VoxelMorphLoss",
    "NCCLoss",
    "MSELoss",
    "GradientLoss",
    "AffineRegistration",
    "DeformableRegistration",
    "DemonsRegistration",
    "SymmetricNormalization",
    "RegistrationResult",
    "ncc_loss",
    "mse_loss",
    "dice_score_registration",
    "compute_similarity_metric",
    "compute_nmi",
    "apply_transform",
    "compose_transforms",
    "compute_jacobian_determinant",
]
