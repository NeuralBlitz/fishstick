"""
Gaussian Splatting Module for Fishstick

3D Gaussian Splatting for real-time radiance field rendering.
Based on "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import math


# =============================================================================
# 1. 3D Gaussians
# =============================================================================


@dataclass
class Gaussian3D:
    """Single 3D Gaussian primitive.

    Attributes:
        position: 3D center position (x, y, z)
        quaternion: Rotation as quaternion (qw, qx, qy, qz)
        scale: Scaling factors (sx, sy, sz)
        opacity: Opacity value in [0, 1]
        sh_features: Spherical harmonics coefficients for view-dependent color
        feature_dim: Dimension of SH features per Gaussian
    """

    position: torch.Tensor  # [3]
    quaternion: torch.Tensor  # [4]
    scale: torch.Tensor  # [3]
    opacity: torch.Tensor  # [1]
    sh_features: torch.Tensor  # [num_sh_coeffs, 3]

    def __post_init__(self):
        assert self.position.shape == (3,), (
            f"Position must be [3], got {self.position.shape}"
        )
        assert self.quaternion.shape == (4,), (
            f"Quaternion must be [4], got {self.quaternion.shape}"
        )
        assert self.scale.shape == (3,), f"Scale must be [3], got {self.scale.shape}"
        assert self.opacity.shape == (1,), (
            f"Opacity must be [1], got {self.opacity.shape}"
        )

    def get_covariance(self) -> torch.Tensor:
        """Compute 3x3 covariance matrix from rotation and scale."""
        rotation = quaternion_to_rotation_matrix(self.quaternion)
        scale_matrix = torch.diag(self.scale)
        covariance = rotation @ scale_matrix @ scale_matrix.T @ rotation.T
        return covariance

    def to(self, device):
        """Move Gaussian to device."""
        return Gaussian3D(
            position=self.position.to(device),
            quaternion=self.quaternion.to(device),
            scale=self.scale.to(device),
            opacity=self.opacity.to(device),
            sh_features=self.sh_features.to(device),
        )


class GaussianModel(nn.Module):
    """Collection of 3D Gaussians representing a scene.

    Attributes:
        max_sh_degree: Maximum spherical harmonics degree
        num_gaussians: Current number of Gaussians
    """

    def __init__(self, num_gaussians: int = 10000, max_sh_degree: int = 3):
        super().__init__()
        self.max_sh_degree = max_sh_degree
        self.num_sh_coeffs = (max_sh_degree + 1) ** 2

        # Learnable parameters
        self._xyz = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)
        self._features_dc = nn.Parameter(torch.randn(num_gaussians, 1, 3))
        self._features_rest = nn.Parameter(
            torch.randn(num_gaussians, self.num_sh_coeffs - 1, 3)
        )
        self._opacity = nn.Parameter(torch.randn(num_gaussians, 1))
        self._scaling = nn.Parameter(torch.randn(num_gaussians, 3))
        self._rotation = nn.Parameter(torch.randn(num_gaussians, 4))

        self.active_sh_degree = 0
        self.num_gaussians = num_gaussians

    @property
    def get_xyz(self) -> torch.Tensor:
        """Get 3D positions."""
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        """Get all SH features."""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat([features_dc, features_rest], dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        """Get opacity with activation."""
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        """Get scaling with activation."""
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        """Get normalized rotation quaternions."""
        return F.normalize(self._rotation, dim=-1)

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """Compute covariance matrices for all Gaussians."""
        return build_covariance_3d(
            self.get_scaling * scaling_modifier, self.get_rotation
        )

    def oneupSHdegree(self):
        """Increase active SH degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def densify_and_split(
        self, grads: torch.Tensor, grad_threshold: float, scene_extent: float
    ):
        """Split large Gaussians with high view-space gradients."""
        n_init_points = self.get_xyz.shape[0]
        device = self.get_xyz.device

        padded_grad = torch.zeros(n_init_points, device=device)
        padded_grad[: grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > scene_extent * 0.01,
        )

        # Split into two Gaussians
        stds = self.get_scaling[selected_pts_mask].repeat(2, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)

        new_xyz = self.get_xyz[selected_pts_mask].repeat(2, 1) + samples
        new_scaling = self.get_scaling[selected_pts_mask].repeat(2, 1) / (1.6)
        new_rotation = self.get_rotation[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(2, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            [
                selected_pts_mask,
                torch.zeros(
                    new_xyz.shape[0] - n_init_points, device=device, dtype=torch.bool
                ),
            ]
        )
        self.prune_points(prune_filter)

    def densify_and_clone(
        self, grads: torch.Tensor, grad_threshold: float, scene_extent: float
    ):
        """Clone Gaussians with high view-space gradients."""
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= scene_extent * 0.01,
        )

        new_xyz = self.get_xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
    ):
        """Append new Gaussians to model."""
        self._xyz = nn.Parameter(torch.cat([self._xyz, new_xyz], dim=0))
        self._features_dc = nn.Parameter(
            torch.cat([self._features_dc, new_features_dc], dim=0)
        )
        self._features_rest = nn.Parameter(
            torch.cat([self._features_rest, new_features_rest], dim=0)
        )
        self._opacity = nn.Parameter(torch.cat([self._opacity, new_opacity], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, new_scaling], dim=0))
        self._rotation = nn.Parameter(torch.cat([self._rotation, new_rotation], dim=0))
        self.num_gaussians = self._xyz.shape[0]

    def prune_points(self, mask: torch.Tensor):
        """Remove Gaussians by mask."""
        valid_points_mask = ~mask
        self._xyz = nn.Parameter(self._xyz[valid_points_mask])
        self._features_dc = nn.Parameter(self._features_dc[valid_points_mask])
        self._features_rest = nn.Parameter(self._features_rest[valid_points_mask])
        self._opacity = nn.Parameter(self._opacity[valid_points_mask])
        self._scaling = nn.Parameter(self._scaling[valid_points_mask])
        self._rotation = nn.Parameter(self._rotation[valid_points_mask])
        self.num_gaussians = self._xyz.shape[0]

    def prune(self, min_opacity: float, max_screen_size: Optional[float] = None):
        """Prune Gaussians based on opacity and size."""
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size is not None:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * self.get_extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


class GaussianScene:
    """Scene representation with cameras and Gaussians.

    Manages the overall scene state including multiple Gaussian models
    and camera parameters.
    """

    def __init__(self, gaussians: GaussianModel, cameras: List["Camera"]):
        self.gaussians = gaussians
        self.cameras = cameras
        self.train_cameras = cameras
        self.test_cameras = []

    def get_train_cameras(self) -> List["Camera"]:
        """Get training cameras."""
        return self.train_cameras

    def get_test_cameras(self) -> List["Camera"]:
        """Get test cameras."""
        return self.test_cameras


# =============================================================================
# 2. Rasterization
# =============================================================================


class GaussianRasterizer:
    """Base rasterizer for splatting Gaussians to image.

    Implements the forward rasterization pass that projects 3D Gaussians
    to 2D and blends them to produce the final image.
    """

    def __init__(self, tile_size: int = 16, debug: bool = False):
        self.tile_size = tile_size
        self.debug = debug

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        sh: torch.Tensor,
        colors_precomp: Optional[torch.Tensor],
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3Ds_precomp: Optional[torch.Tensor],
        raster_settings: "RasterSettings",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward rasterization pass.

        Args:
            means3D: 3D Gaussian centers [N, 3]
            means2D: Projected 2D centers [N, 2]
            sh: Spherical harmonics features [N, K, 3]
            colors_precomp: Precomputed colors [N, 3] or None
            opacities: Opacity values [N, 1]
            scales: Scaling factors [N, 3]
            rotations: Rotation quaternions [N, 4]
            cov3Ds_precomp: Precomputed 3D covariances [N, 6] or None
            raster_settings: Rasterization settings

        Returns:
            rendered_image: [H, W, 3]
            radii: [N] radius of each Gaussian in screen space
            visibility_filter: [N] boolean visibility mask
        """
        # Project to 2D
        if colors_precomp is None:
            colors = compute_color_from_sh(
                sh, means3D, raster_settings.viewmatrix, raster_settings.sh_degree
            )
        else:
            colors = colors_precomp

        # Build 2D covariance
        if cov3Ds_precomp is None:
            cov3D = build_covariance_3d(scales, rotations)
        else:
            cov3D = cov3Ds_precomp

        cov2D = project_cov3D_to_cov2D(
            cov3D,
            means3D,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.focal_x,
            raster_settings.focal_y,
            raster_settings.image_height,
            raster_settings.image_width,
        )

        # Render
        rendered_image, radii, visibility_filter = self.render_gaussians_2D(
            means2D, colors, opacities, cov2D, raster_settings
        )

        return rendered_image, radii, visibility_filter

    def render_gaussians_2D(
        self,
        means2D: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        cov2D: torch.Tensor,
        raster_settings: "RasterSettings",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render 2D projected Gaussians.

        Args:
            means2D: 2D centers [N, 2]
            colors: RGB colors [N, 3]
            opacities: Opacity values [N, 1]
            cov2D: 2D covariances [N, 3] (packed as [xx, xy, yy])
            raster_settings: Rasterization settings

        Returns:
            image: [H, W, 3]
            radii: [N]
            visibility: [N]
        """
        N = means2D.shape[0]
        H, W = raster_settings.image_height, raster_settings.image_width
        device = means2D.device

        # Initialize output
        image = torch.zeros(H, W, 3, device=device)
        radii = torch.zeros(N, device=device)
        visibility = torch.zeros(N, dtype=torch.bool, device=device)

        # Compute 2D bounding boxes
        for i in range(N):
            det = cov2D[i, 0] * cov2D[i, 2] - cov2D[i, 1] ** 2
            if det <= 0:
                continue

            inv_det = 1.0 / det
            conic = torch.stack(
                [cov2D[i, 2] * inv_det, -cov2D[i, 1] * inv_det, cov2D[i, 0] * inv_det]
            )

            # Radius based on 3 sigma
            det_sqrt = torch.sqrt(det)
            radius = 3.0 * torch.sqrt(det_sqrt)
            radii[i] = radius

            # Skip if too small
            if radius < 0.3:
                continue

            visibility[i] = True

            # Bounding box in pixel space
            cx, cy = means2D[i]
            x_min = max(0, int(cx - radius))
            x_max = min(W, int(cx + radius) + 1)
            y_min = max(0, int(cy - radius))
            y_max = min(H, int(cy + radius) + 1)

            # Splat Gaussian
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    dx = x - cx
                    dy = y - cy

                    power = -0.5 * (
                        conic[0] * dx * dx + 2 * conic[1] * dx * dy + conic[2] * dy * dy
                    )

                    if power > 0.0:
                        alpha = opacities[i, 0] * torch.exp(power)

                        if alpha < 1.0 / 255.0:
                            continue

                        # Alpha blending
                        image[y, x] = image[y, x] + alpha * (colors[i] - image[y, x])

        return image, radii, visibility


class DifferentiableRasterizer(GaussianRasterizer):
    """Differentiable rasterizer with backward pass support.

    Extends base rasterizer to support gradient computation for
    end-to-end optimization.
    """

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        sh: torch.Tensor,
        colors_precomp: Optional[torch.Tensor],
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3Ds_precomp: Optional[torch.Tensor],
        raster_settings: "RasterSettings",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with gradient tracking."""
        ctx = {
            "means3D": means3D,
            "means2D": means2D,
            "sh": sh,
            "opacities": opacities,
            "scales": scales,
            "rotations": rotations,
            "raster_settings": raster_settings,
        }

        # Compute colors
        if colors_precomp is None:
            colors = compute_color_from_sh(
                sh, means3D, raster_settings.viewmatrix, raster_settings.sh_degree
            )
        else:
            colors = colors_precomp

        ctx["colors"] = colors

        # Build 2D covariance
        if cov3Ds_precomp is None:
            cov3D = build_covariance_3d(scales, rotations)
        else:
            cov3D = cov3Ds_precomp

        ctx["cov3D"] = cov3D

        cov2D = project_cov3D_to_cov2D(
            cov3D,
            means3D,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.focal_x,
            raster_settings.focal_y,
            raster_settings.image_height,
            raster_settings.image_width,
        )

        ctx["cov2D"] = cov2D

        # Render with gradient tracking
        rendered_image, radii, visibility = self.render_gaussians_2D(
            means2D, colors, opacities, cov2D, raster_settings
        )

        ctx["radii"] = radii
        ctx["visibility"] = visibility

        return rendered_image, radii, visibility

    @staticmethod
    def backward(ctx, grad_out_img, grad_radii, grad_visibility):
        """Backward pass for gradient computation.

        Computes gradients with respect to all Gaussian parameters.
        """
        means3D = ctx["means3D"]
        means2D = ctx["means2D"]
        colors = ctx["colors"]
        opacities = ctx["opacities"]
        cov2D = ctx["cov2D"]
        radii = ctx["radii"]
        visibility = ctx["visibility"]
        raster_settings = ctx["raster_settings"]

        grad_means3D = torch.zeros_like(means3D)
        grad_means2D = torch.zeros_like(means2D)
        grad_colors = torch.zeros_like(colors)
        grad_opacities = torch.zeros_like(opacities)
        grad_cov2D = torch.zeros_like(cov2D)

        H, W = raster_settings.image_height, raster_settings.image_width
        device = means2D.device

        # Backpropagate through splatting
        for i in range(means2D.shape[0]):
            if not visibility[i]:
                continue

            cx, cy = means2D[i]
            radius = radii[i]

            x_min = max(0, int(cx - radius))
            x_max = min(W, int(cx + radius) + 1)
            y_min = max(0, int(cy - radius))
            y_max = min(H, int(cy + radius) + 1)

            det = cov2D[i, 0] * cov2D[i, 2] - cov2D[i, 1] ** 2
            if det <= 0:
                continue

            inv_det = 1.0 / det
            conic = torch.stack(
                [cov2D[i, 2] * inv_det, -cov2D[i, 1] * inv_det, cov2D[i, 0] * inv_det]
            )

            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    dx = x - cx
                    dy = y - cy

                    power = -0.5 * (
                        conic[0] * dx * dx + 2 * conic[1] * dx * dy + conic[2] * dy * dy
                    )

                    if power <= 0.0:
                        continue

                    alpha = opacities[i, 0] * torch.exp(power)
                    if alpha < 1.0 / 255.0:
                        continue

                    # Accumulate gradients
                    grad_alpha = torch.sum(
                        grad_out_img[y, x] * (colors[i] - grad_out_img[y, x])
                    )
                    grad_colors[i] += alpha * grad_out_img[y, x]
                    grad_opacities[i] += torch.exp(power) * grad_alpha

                    # Position gradients
                    grad_power = alpha * grad_alpha
                    grad_dx = grad_power * (-conic[0] * dx - conic[1] * dy)
                    grad_dy = grad_power * (-conic[1] * dx - conic[2] * dy)
                    grad_means2D[i, 0] -= grad_dx
                    grad_means2D[i, 1] -= grad_dy

        return (
            grad_means3D,
            grad_means2D,
            None,
            grad_colors,
            grad_opacities,
            None,
            None,
            grad_cov2D,
            None,
        )


class TileRasterizer(GaussianRasterizer):
    """Tile-based rasterizer for efficient parallel rendering.

    Divides the image into tiles and processes them in parallel
    for better performance on large images.
    """

    def __init__(self, tile_size: int = 16, num_tiles_per_thread: int = 1):
        super().__init__(tile_size=tile_size)
        self.num_tiles_per_thread = num_tiles_per_thread

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        sh: torch.Tensor,
        colors_precomp: Optional[torch.Tensor],
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3Ds_precomp: Optional[torch.Tensor],
        raster_settings: "RasterSettings",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tile-based forward pass."""
        H, W = raster_settings.image_height, raster_settings.image_width
        device = means2D.device

        # Compute colors and 2D covariances
        if colors_precomp is None:
            colors = compute_color_from_sh(
                sh, means3D, raster_settings.viewmatrix, raster_settings.sh_degree
            )
        else:
            colors = colors_precomp

        if cov3Ds_precomp is None:
            cov3D = build_covariance_3d(scales, rotations)
        else:
            cov3D = cov3Ds_precomp

        cov2D = project_cov3D_to_cov2D(
            cov3D,
            means3D,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.focal_x,
            raster_settings.focal_y,
            H,
            W,
        )

        # Compute tile bounds for each Gaussian
        N = means2D.shape[0]
        num_tiles_x = (W + self.tile_size - 1) // self.tile_size
        num_tiles_y = (H + self.tile_size - 1) // self.tile_size

        # Build tile-Gaussian mapping
        gaussian_tile_indices = []
        tile_gaussian_starts = [0]

        for i in range(N):
            det = cov2D[i, 0] * cov2D[i, 2] - cov2D[i, 1] ** 2
            if det <= 0:
                continue

            det_sqrt = torch.sqrt(det)
            radius = 3.0 * torch.sqrt(det_sqrt)

            if radius < 0.3:
                continue

            cx, cy = means2D[i]
            tile_x_min = max(0, int((cx - radius) / self.tile_size))
            tile_x_max = min(num_tiles_x - 1, int((cx + radius) / self.tile_size))
            tile_y_min = max(0, int((cy - radius) / self.tile_size))
            tile_y_max = min(num_tiles_y - 1, int((cy + radius) / self.tile_size))

            for ty in range(tile_y_min, tile_y_max + 1):
                for tx in range(tile_x_min, tile_x_max + 1):
                    tile_idx = ty * num_tiles_x + tx
                    gaussian_tile_indices.append((tile_idx, i))

        # Sort by tile index
        gaussian_tile_indices.sort(key=lambda x: x[0])

        # Render tiles
        image = torch.zeros(H, W, 3, device=device)
        radii = torch.zeros(N, device=device)
        visibility = torch.zeros(N, dtype=torch.bool, device=device)

        for tile_idx in range(num_tiles_x * num_tiles_y):
            tx = tile_idx % num_tiles_x
            ty = tile_idx // num_tiles_x

            tile_x_start = tx * self.tile_size
            tile_y_start = ty * self.tile_size
            tile_x_end = min(tile_x_start + self.tile_size, W)
            tile_y_end = min(tile_y_start + self.tile_size, H)

            # Get Gaussians for this tile
            tile_gaussians = [
                idx for t_idx, idx in gaussian_tile_indices if t_idx == tile_idx
            ]

            if not tile_gaussians:
                continue

            # Render tile
            for i in tile_gaussians:
                cx, cy = means2D[i]

                det = cov2D[i, 0] * cov2D[i, 2] - cov2D[i, 1] ** 2
                inv_det = 1.0 / det
                conic = torch.stack(
                    [
                        cov2D[i, 2] * inv_det,
                        -cov2D[i, 1] * inv_det,
                        cov2D[i, 0] * inv_det,
                    ]
                )

                det_sqrt = torch.sqrt(det)
                radius = 3.0 * torch.sqrt(det_sqrt)
                radii[i] = radius
                visibility[i] = True

                # Bounding box within tile
                g_x_min = max(tile_x_start, int(cx - radius))
                g_x_max = min(tile_x_end, int(cx + radius) + 1)
                g_y_min = max(tile_y_start, int(cy - radius))
                g_y_max = min(tile_y_end, int(cy + radius) + 1)

                for y in range(g_y_min, g_y_max):
                    for x in range(g_x_min, g_x_max):
                        dx = x - cx
                        dy = y - cy

                        power = -0.5 * (
                            conic[0] * dx * dx
                            + 2 * conic[1] * dx * dy
                            + conic[2] * dy * dy
                        )

                        if power > 0.0:
                            alpha = opacities[i, 0] * torch.exp(power)
                            if alpha >= 1.0 / 255.0:
                                image[y, x] = image[y, x] + alpha * (
                                    colors[i] - image[y, x]
                                )

        return image, radii, visibility


# =============================================================================
# 3. Primitives
# =============================================================================


def build_covariance_3d(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Build 3D covariance matrices from scales and rotations.

    Args:
        scales: Scaling factors [N, 3]
        rotations: Rotation quaternions [N, 4]

    Returns:
        covariances: Packed 3D covariances [N, 6] (upper triangular)
    """
    N = scales.shape[0]
    device = scales.device

    # Build scale matrices
    scale_matrices = torch.zeros(N, 3, 3, device=device)
    for i in range(N):
        scale_matrices[i] = torch.diag(scales[i])

    # Build rotation matrices
    rotation_matrices = quaternion_to_rotation_matrix(rotations)

    # Compute covariance: R * S * S^T * R^T
    covariances = torch.zeros(N, 6, device=device)
    for i in range(N):
        S = scale_matrices[i]
        R = rotation_matrices[i]
        cov = R @ S @ S.T @ R.T
        # Pack upper triangular
        covariances[i, 0] = cov[0, 0]
        covariances[i, 1] = cov[0, 1]
        covariances[i, 2] = cov[0, 2]
        covariances[i, 3] = cov[1, 1]
        covariances[i, 4] = cov[1, 2]
        covariances[i, 5] = cov[2, 2]

    return covariances


def build_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Build 3x3 rotation matrix from quaternion.

    Args:
        quaternion: Rotation quaternion [qw, qx, qy, qz] or [N, 4]

    Returns:
        rotation_matrix: [3, 3] or [N, 3, 3]
    """
    return quaternion_to_rotation_matrix(quaternion)


def build_scaling_matrix(scales: torch.Tensor) -> torch.Tensor:
    """Build diagonal scaling matrix.

    Args:
        scales: Scaling factors [sx, sy, sz] or [N, 3]

    Returns:
        scaling_matrix: [3, 3] or [N, 3, 3]
    """
    if scales.dim() == 1:
        return torch.diag(scales)
    else:
        N = scales.shape[0]
        matrices = torch.zeros(N, 3, 3, device=scales.device)
        for i in range(N):
            matrices[i] = torch.diag(scales[i])
        return matrices


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternion(s) to rotation matrix.

    Args:
        quaternions: [4] or [N, 4] quaternions (qw, qx, qy, qz)

    Returns:
        rotation_matrices: [3, 3] or [N, 3, 3]
    """
    if quaternions.dim() == 1:
        quaternions = quaternions.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    N = quaternions.shape[0]
    qw, qx, qy, qz = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )

    # Normalize
    norm = torch.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Build rotation matrix
    R = torch.zeros(N, 3, 3, device=quaternions.device)

    R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[:, 0, 1] = 2 * (qx * qy - qw * qz)
    R[:, 0, 2] = 2 * (qx * qz + qw * qy)

    R[:, 1, 0] = 2 * (qx * qy + qw * qz)
    R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[:, 1, 2] = 2 * (qy * qz - qw * qx)

    R[:, 2, 0] = 2 * (qx * qz - qw * qy)
    R[:, 2, 1] = 2 * (qy * qz + qw * qx)
    R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)

    if squeeze:
        return R.squeeze(0)
    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion.

    Args:
        R: [3, 3] or [N, 3, 3] rotation matrix

    Returns:
        quaternions: [4] or [N, 4] (qw, qx, qy, qz)
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    N = R.shape[0]
    quaternions = torch.zeros(N, 4, device=R.device)

    for i in range(N):
        trace = R[i, 0, 0] + R[i, 1, 1] + R[i, 2, 2]

        if trace > 0:
            S = torch.sqrt(trace + 1.0) * 2
            quaternions[i, 0] = 0.25 * S
            quaternions[i, 1] = (R[i, 2, 1] - R[i, 1, 2]) / S
            quaternions[i, 2] = (R[i, 0, 2] - R[i, 2, 0]) / S
            quaternions[i, 3] = (R[i, 1, 0] - R[i, 0, 1]) / S
        elif R[i, 0, 0] > R[i, 1, 1] and R[i, 0, 0] > R[i, 2, 2]:
            S = torch.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2
            quaternions[i, 0] = (R[i, 2, 1] - R[i, 1, 2]) / S
            quaternions[i, 1] = 0.25 * S
            quaternions[i, 2] = (R[i, 0, 1] + R[i, 1, 0]) / S
            quaternions[i, 3] = (R[i, 0, 2] + R[i, 2, 0]) / S
        elif R[i, 1, 1] > R[i, 2, 2]:
            S = torch.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2]) * 2
            quaternions[i, 0] = (R[i, 0, 2] - R[i, 2, 0]) / S
            quaternions[i, 1] = (R[i, 0, 1] + R[i, 1, 0]) / S
            quaternions[i, 2] = 0.25 * S
            quaternions[i, 3] = (R[i, 1, 2] + R[i, 2, 1]) / S
        else:
            S = torch.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1]) * 2
            quaternions[i, 0] = (R[i, 1, 0] - R[i, 0, 1]) / S
            quaternions[i, 1] = (R[i, 0, 2] + R[i, 2, 0]) / S
            quaternions[i, 2] = (R[i, 1, 2] + R[i, 2, 1]) / S
            quaternions[i, 3] = 0.25 * S

    if squeeze:
        return quaternions.squeeze(0)
    return quaternions


# =============================================================================
# 4. Loss Functions
# =============================================================================


class L1Loss(nn.Module):
    """L1 loss for image reconstruction."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss.

        Args:
            pred: Predicted image [B, H, W, 3] or [H, W, 3]
            target: Target image [B, H, W, 3] or [H, W, 3]

        Returns:
            loss: Scalar loss value
        """
        return torch.abs(pred - target).mean()


class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) loss.

    Measures perceptual similarity between images.
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM."""
        sigma = 1.5
        gauss = torch.Tensor(
            [
                math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        gauss = gauss / gauss.sum()

        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between two images."""
        device = img1.device
        window = self.window.to(device)

        C1 = 0.01**2
        C2 = 0.03**2

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

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM to minimize).

        Args:
            pred: Predicted image [B, H, W, 3] or [H, W, 3]
            target: Target image [B, H, W, 3] or [H, W, 3]

        Returns:
            loss: 1 - SSIM
        """
        # Convert to [B, C, H, W] format
        if pred.dim() == 3:
            pred = pred.permute(2, 0, 1).unsqueeze(0)
            target = target.permute(2, 0, 1).unsqueeze(0)
        else:
            pred = pred.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

        return 1 - self._ssim(pred, target)


class CombinedLoss(nn.Module):
    """Combined L1 + SSIM loss.

    Common configuration: lambda_l1 = 0.8, lambda_ssim = 0.2
    """

    def __init__(self, lambda_l1: float = 0.8, lambda_ssim: float = 0.2):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)

        total = self.lambda_l1 * l1 + self.lambda_ssim * ssim

        loss_dict = {
            "total": total.item(),
            "l1": l1.item(),
            "ssim": ssim.item(),
        }

        return total, loss_dict


# =============================================================================
# 5. Optimization
# =============================================================================


class GaussianOptimizer:
    """Optimizer for Gaussian parameters.

    Uses different learning rates for different parameter groups.
    """

    def __init__(
        self,
        gaussian_model: GaussianModel,
        lr_position: float = 0.00016,
        lr_features: float = 0.0025,
        lr_opacity: float = 0.05,
        lr_scaling: float = 0.005,
        lr_rotation: float = 0.001,
    ):
        self.gaussian_model = gaussian_model

        self.optimizer = torch.optim.Adam(
            [
                {"params": [gaussian_model._xyz], "lr": lr_position, "name": "xyz"},
                {
                    "params": [gaussian_model._features_dc],
                    "lr": lr_features,
                    "name": "f_dc",
                },
                {
                    "params": [gaussian_model._features_rest],
                    "lr": lr_features / 20.0,
                    "name": "f_rest",
                },
                {
                    "params": [gaussian_model._opacity],
                    "lr": lr_opacity,
                    "name": "opacity",
                },
                {
                    "params": [gaussian_model._scaling],
                    "lr": lr_scaling,
                    "name": "scaling",
                },
                {
                    "params": [gaussian_model._rotation],
                    "lr": lr_rotation,
                    "name": "rotation",
                },
            ],
            lr=0.0,
            eps=1e-15,
        )

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=lr_position,
            lr_final=lr_position / 30.0,
            lr_delay_mult=0.01,
            max_steps=30_000,
        )

    def update_learning_rate(self, iteration: int):
        """Update learning rate with exponential decay."""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def step(self):
        """Optimization step."""
        self.optimizer.step()

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Get optimizer state."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)


class AdaptiveDensityControl:
    """Adaptive density control for Gaussian densification.

    Manages splitting and cloning of Gaussians based on view-space
    gradients and opacity values.
    """

    def __init__(
        self,
        densify_from_iter: int = 500,
        densify_until_iter: int = 15_000,
        densify_grad_threshold: float = 0.0002,
        opacity_cull_threshold: float = 0.005,
        densification_interval: int = 100,
        opacity_reset_interval: int = 3000,
    ):
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densify_grad_threshold = densify_grad_threshold
        self.opacity_cull_threshold = opacity_cull_threshold
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval

        self.xyz_gradient_accum = torch.zeros(0, 3)
        self.denom = torch.zeros(0)

    def reset_opacity(self, gaussian_model: GaussianModel):
        """Reset opacity values to avoid transparent Gaussians."""
        opacities_new = inverse_sigmoid(
            torch.min(
                gaussian_model.get_opacity,
                torch.ones_like(gaussian_model.get_opacity) * 0.01,
            )
        )
        gaussian_model._opacity = nn.Parameter(opacities_new)

    def densify_and_prune(
        self,
        gaussian_model: GaussianModel,
        scene_extent: float,
        iteration: int,
    ):
        """Perform densification and pruning.

        Args:
            gaussian_model: Gaussian model to modify
            scene_extent: Spatial extent of the scene
            iteration: Current training iteration
        """
        if iteration < self.densify_from_iter or iteration > self.densify_until_iter:
            return

        if iteration % self.densification_interval == 0:
            # Compute average gradients
            grads = self.xyz_gradient_accum / self.denom.unsqueeze(1)
            grads[grads.isnan()] = 0.0

            # Clone and split
            gaussian_model.densify_and_clone(
                grads, self.densify_grad_threshold, scene_extent
            )
            gaussian_model.densify_and_split(
                grads, self.densify_grad_threshold, scene_extent
            )

            # Reset accumulators
            self.xyz_gradient_accum = torch.zeros(
                gaussian_model.num_gaussians, 3, device=grads.device
            )
            self.denom = torch.zeros(gaussian_model.num_gaussians, device=grads.device)

        if (
            iteration > self.densify_from_iter
            and iteration % self.opacity_reset_interval == 0
        ):
            self.reset_opacity(gaussian_model)

        # Prune low-opacity Gaussians
        prune_mask = (
            gaussian_model.get_opacity < self.opacity_cull_threshold
        ).squeeze()
        gaussian_model.prune_points(prune_mask)

    def add_densification_stats(
        self, viewspace_point_tensor: torch.Tensor, update_filter: torch.Tensor
    ):
        """Accumulate gradients for densification decisions."""
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1


class SplitAndClone:
    """Explicit split and clone operations.

    Handles the geometric operations for splitting large Gaussians
    into smaller ones and cloning high-gradient Gaussians.
    """

    @staticmethod
    def split_gaussian(
        gaussian: Gaussian3D,
        num_splits: int = 2,
    ) -> List[Gaussian3D]:
        """Split a Gaussian into multiple smaller Gaussians.

        Args:
            gaussian: Gaussian to split
            num_splits: Number of resulting Gaussians

        Returns:
            List of split Gaussians
        """
        new_gaussians = []

        for _ in range(num_splits):
            # Sample new position with std = scale
            noise = torch.randn(3, device=gaussian.position.device) * gaussian.scale
            new_position = gaussian.position + noise

            # Reduce scale
            new_scale = gaussian.scale / num_splits

            new_gaussians.append(
                Gaussian3D(
                    position=new_position,
                    quaternion=gaussian.quaternion.clone(),
                    scale=new_scale,
                    opacity=gaussian.opacity.clone(),
                    sh_features=gaussian.sh_features.clone(),
                )
            )

        return new_gaussians

    @staticmethod
    def clone_gaussian(gaussian: Gaussian3D) -> Gaussian3D:
        """Clone a Gaussian.

        Args:
            gaussian: Gaussian to clone

        Returns:
            Cloned Gaussian
        """
        return Gaussian3D(
            position=gaussian.position.clone(),
            quaternion=gaussian.quaternion.clone(),
            scale=gaussian.scale.clone(),
            opacity=gaussian.opacity.clone(),
            sh_features=gaussian.sh_features.clone(),
        )


class Pruning:
    """Pruning strategies for removing redundant Gaussians.

    Removes Gaussians that contribute little to the rendered image
    or are outside the view frustum.
    """

    def __init__(
        self,
        opacity_threshold: float = 0.005,
        size_threshold: float = 20.0,
    ):
        self.opacity_threshold = opacity_threshold
        self.size_threshold = size_threshold

    def should_prune(
        self, gaussian: Gaussian3D, view_frustum: Optional[torch.Tensor] = None
    ) -> bool:
        """Determine if a Gaussian should be pruned.

        Args:
            gaussian: Gaussian to evaluate
            view_frustum: Optional view frustum bounds

        Returns:
            True if Gaussian should be pruned
        """
        # Prune low opacity
        if gaussian.opacity.item() < self.opacity_threshold:
            return True

        # Prune if outside view frustum
        if view_frustum is not None:
            if not self._in_frustum(gaussian.position, view_frustum):
                return True

        return False

    def _in_frustum(self, position: torch.Tensor, frustum: torch.Tensor) -> bool:
        """Check if position is inside view frustum."""
        # Simple AABB check
        return (
            frustum[0, 0] <= position[0] <= frustum[1, 0]
            and frustum[0, 1] <= position[1] <= frustum[1, 1]
            and frustum[0, 2] <= position[2] <= frustum[1, 2]
        )

    def prune_by_opacity(self, gaussian_model: GaussianModel) -> torch.Tensor:
        """Get mask of Gaussians to prune based on opacity.

        Returns:
            Boolean mask where True indicates pruning
        """
        return gaussian_model.get_opacity.squeeze() < self.opacity_threshold


# =============================================================================
# 6. Training
# =============================================================================


class GaussianTrainer:
    """Training loop for Gaussian Splatting.

    Orchestrates the entire training process including rendering,
    loss computation, and optimization.
    """

    def __init__(
        self,
        gaussian_model: GaussianModel,
        scene: GaussianScene,
        optimizer: GaussianOptimizer,
        loss_fn: CombinedLoss,
        rasterizer: GaussianRasterizer,
        adaptive_control: Optional[AdaptiveDensityControl] = None,
        save_iterations: List[int] = [7000, 30000],
        test_iterations: List[int] = [7000, 30000],
        checkpoint_iterations: List[int] = [7000, 30000],
    ):
        self.gaussian_model = gaussian_model
        self.scene = scene
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.rasterizer = rasterizer
        self.adaptive_control = adaptive_control

        self.save_iterations = save_iterations
        self.test_iterations = test_iterations
        self.checkpoint_iterations = checkpoint_iterations

        self.iteration = 0
        self.best_loss = float("inf")

    def train(self, num_iterations: int = 30000):
        """Main training loop.

        Args:
            num_iterations: Total number of training iterations
        """
        self.gaussian_model.train()

        for iteration in range(self.iteration, num_iterations):
            self.iteration = iteration

            # Sample random camera
            viewpoint_cam = np.random.choice(self.scene.get_train_cameras())

            # Render
            render_pkg = render_gaussians(
                self.gaussian_model,
                viewpoint_cam,
                self.rasterizer,
                self.gaussian_model.active_sh_degree,
            )

            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image

            # Compute loss
            loss, loss_dict = self.loss_fn(image, gt_image)

            # Backprop
            loss.backward()

            # Update learning rate
            self.optimizer.update_learning_rate(iteration)

            # Optimization step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Adaptive density control
            if self.adaptive_control is not None:
                self.adaptive_control.densify_and_prune(
                    self.gaussian_model,
                    self.scene.cameras[0].extent if self.scene.cameras else 1.0,
                    iteration,
                )

            # Logging
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss_dict['total']:.4f}")

            # Save checkpoint
            if iteration in self.checkpoint_iterations:
                self.save_checkpoint(iteration)

            # Test
            if iteration in self.test_iterations:
                self.test()

        print(f"Training completed. Best loss: {self.best_loss:.4f}")

    def test(self):
        """Test on validation set."""
        self.gaussian_model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for viewpoint_cam in self.scene.get_test_cameras():
                render_pkg = render_gaussians(
                    self.gaussian_model,
                    viewpoint_cam,
                    self.rasterizer,
                    self.gaussian_model.active_sh_degree,
                )
                image = render_pkg["render"]
                gt_image = viewpoint_cam.original_image

                loss, _ = self.loss_fn(image, gt_image)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.scene.get_test_cameras())
        print(f"Test Loss: {avg_loss:.4f}")

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss

        self.gaussian_model.train()

    def save_checkpoint(self, iteration: int):
        """Save training checkpoint."""
        checkpoint = {
            "iteration": iteration,
            "gaussian_model_state": self.gaussian_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "active_sh_degree": self.gaussian_model.active_sh_degree,
        }
        torch.save(checkpoint, f"checkpoint_{iteration}.pth")
        print(f"Checkpoint saved at iteration {iteration}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.iteration = checkpoint["iteration"]
        self.gaussian_model.load_state_dict(checkpoint["gaussian_model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.gaussian_model.active_sh_degree = checkpoint["active_sh_degree"]
        print(f"Checkpoint loaded from iteration {self.iteration}")


class GaussianDataLoader:
    """Data loader for Gaussian Splatting training data.

    Loads camera parameters and reference images for training.
    """

    def __init__(
        self,
        source_path: str,
        resolution: int = -1,
        shuffle: bool = True,
        resolution_scales: List[float] = [1.0],
    ):
        self.source_path = source_path
        self.resolution = resolution
        self.shuffle = shuffle
        self.resolution_scales = resolution_scales

        self.cameras = []

    def load_cameras(self) -> List["Camera"]:
        """Load camera parameters from source."""
        # This would load from COLMAP format or similar
        # For now, return empty list as placeholder
        return self.cameras

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, idx: int) -> "Camera":
        return self.cameras[idx]


@dataclass
class Camera:
    """Camera model for rendering.

    Attributes:
        camera_id: Unique camera identifier
        R: Rotation matrix [3, 3]
        T: Translation vector [3]
        FoVx: Horizontal field of view in radians
        FoVy: Vertical field of view in radians
        image: Camera image tensor [H, W, 3]
        image_name: Name of the image file
        image_width: Image width in pixels
        image_height: Image height in pixels
        device: Device to store tensors
    """

    camera_id: int
    R: torch.Tensor
    T: torch.Tensor
    FoVx: float
    FoVy: float
    image: torch.Tensor
    image_name: str
    image_width: int
    image_height: int
    device: str = "cuda"

    def __post_init__(self):
        self.original_image = self.image

    @property
    def world_view_transform(self) -> torch.Tensor:
        """Get world-to-camera transform."""
        Rt = torch.zeros(4, 4, device=self.device)
        Rt[:3, :3] = self.R
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0
        return Rt

    @property
    def full_proj_transform(self) -> torch.Tensor:
        """Get full projection transform."""
        # Build projection matrix
        znear = 0.01
        zfar = 100.0

        tanHalfFovX = math.tan(self.FoVx / 2)
        tanHalfFovY = math.tan(self.FoVy / 2)

        P = torch.zeros(4, 4, device=self.device)
        P[0, 0] = 1 / tanHalfFovX
        P[1, 1] = 1 / tanHalfFovY
        P[2, 2] = -(zfar + znear) / (zfar - znear)
        P[2, 3] = -2 * zfar * znear / (zfar - znear)
        P[3, 2] = -1

        return P @ self.world_view_transform

    @property
    def projection_matrix(self) -> torch.Tensor:
        """Get projection matrix."""
        return self.full_proj_transform

    @property
    def camera_center(self) -> torch.Tensor:
        """Get camera center in world coordinates."""
        return -self.R.T @ self.T


# =============================================================================
# 7. Rendering
# =============================================================================


def render_gaussians(
    gaussian_model: GaussianModel,
    camera: Camera,
    rasterizer: GaussianRasterizer,
    sh_degree: int = 0,
    bg_color: Optional[torch.Tensor] = None,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Render Gaussians from camera viewpoint.

    Args:
        gaussian_model: Gaussian model to render
        camera: Camera viewpoint
        rasterizer: Rasterizer to use
        sh_degree: Active SH degree
        bg_color: Background color [3] or None
        scaling_modifier: Modifier for Gaussian scales
        override_color: Override all Gaussian colors or None

    Returns:
        Dictionary containing:
            - render: Rendered image [H, W, 3]
            - viewspace_points: Projected 2D positions
            - visibility_filter: Boolean visibility mask
            - radii: Screen space radii
    """
    # Set up rasterization configuration
    raster_settings = RasterSettings(
        image_height=camera.image_height,
        image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=bg_color if bg_color is not None else torch.zeros(3, device=camera.device),
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=rasterizer.debug,
    )

    # Get Gaussian properties
    means3D = gaussian_model.get_xyz
    means2D = project_points(
        means3D, raster_settings.projmatrix, camera.image_width, camera.image_height
    )

    opacity = gaussian_model.get_opacity
    scales = gaussian_model.get_scaling
    rotations = gaussian_model.get_rotation

    # Compute colors
    if override_color is None:
        shs_view = gaussian_model.get_features.transpose(1, 2).view(
            -1, 3, (gaussian_model.max_sh_degree + 1) ** 2
        )
        dir_pp = means3D - camera.camera_center.repeat(means3D.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(gaussian_model.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = override_color

    # Rasterize
    rendered_image, radii, visibility_filter = rasterizer.forward(
        means3D=means3D,
        means2D=means2D,
        sh=gaussian_model.get_features,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=None,
        raster_settings=raster_settings,
    )

    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": visibility_filter,
        "radii": radii,
    }


def compute_color_from_sh(
    sh_features: torch.Tensor,
    positions: torch.Tensor,
    view_matrix: torch.Tensor,
    sh_degree: int,
) -> torch.Tensor:
    """Compute RGB color from spherical harmonics.

    Args:
        sh_features: SH coefficients [N, K, 3]
        positions: 3D positions [N, 3]
        view_matrix: Camera view matrix [4, 4]
        sh_degree: Maximum SH degree to use

    Returns:
        colors: RGB colors [N, 3]
    """
    # Extract camera position from view matrix
    R = view_matrix[:3, :3]
    t = view_matrix[:3, 3]
    cam_pos = -R.T @ t

    # Compute view directions
    dirs = positions - cam_pos.unsqueeze(0)
    dirs = dirs / (torch.norm(dirs, dim=1, keepdim=True) + 1e-7)

    # Evaluate SH
    num_sh = (sh_degree + 1) ** 2
    sh_features_subset = sh_features[:, :num_sh, :]

    colors = eval_sh(sh_degree, sh_features_subset, dirs)
    colors = torch.clamp(colors + 0.5, 0.0, 1.0)

    return colors


def get_view_matrix(
    camera_position: torch.Tensor,
    look_at: torch.Tensor,
    up_vector: torch.Tensor,
) -> torch.Tensor:
    """Build view transformation matrix.

    Args:
        camera_position: Camera position [3]
        look_at: Point to look at [3]
        up_vector: Up direction [3]

    Returns:
        view_matrix: [4, 4] view transformation
    """
    # Forward direction
    forward = look_at - camera_position
    forward = forward / torch.norm(forward)

    # Right direction
    right = torch.cross(forward, up_vector)
    right = right / torch.norm(right)

    # True up direction
    up = torch.cross(right, forward)

    # Build view matrix
    view = torch.zeros(4, 4, device=camera_position.device)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -forward
    view[:3, 3] = -torch.stack(
        [
            torch.dot(right, camera_position),
            torch.dot(up, camera_position),
            torch.dot(forward, camera_position),
        ]
    )
    view[3, 3] = 1.0

    return view


# =============================================================================
# Utility Functions
# =============================================================================


@dataclass
class RasterSettings:
    """Configuration for rasterization."""

    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool

    @property
    def focal_x(self) -> float:
        return self.image_width / (2.0 * self.tanfovx)

    @property
    def focal_y(self) -> float:
        return self.image_height / (2.0 * self.tanfovy)


def project_cov3D_to_cov2D(
    cov3Ds: torch.Tensor,
    means3D: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    focal_x: float,
    focal_y: float,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Project 3D covariances to 2D screen space.

    Args:
        cov3Ds: 3D covariances [N, 6] (packed upper triangular)
        means3D: 3D positions [N, 3]
        viewmatrix: View matrix [4, 4]
        projmatrix: Projection matrix [4, 4]
        focal_x: Focal length in x
        focal_y: Focal length in y
        image_height: Image height
        image_width: Image width

    Returns:
        cov2Ds: 2D covariances [N, 3] (packed as [xx, xy, yy])
    """
    N = cov3Ds.shape[0]
    device = cov3Ds.device

    # Transform to camera space
    means_cam = transform_points(means3D, viewmatrix)

    cov2Ds = torch.zeros(N, 3, device=device)

    for i in range(N):
        # Skip if behind camera
        if means_cam[i, 2] <= 0.1:
            continue

        # Unpack 3D covariance
        cov3D = torch.zeros(3, 3, device=device)
        cov3D[0, 0] = cov3Ds[i, 0]
        cov3D[0, 1] = cov3Ds[i, 1]
        cov3D[0, 2] = cov3Ds[i, 2]
        cov3D[1, 1] = cov3Ds[i, 3]
        cov3D[1, 2] = cov3Ds[i, 4]
        cov3D[2, 2] = cov3Ds[i, 5]
        cov3D[1, 0] = cov3D[0, 1]
        cov3D[2, 0] = cov3D[0, 2]
        cov3D[2, 1] = cov3D[1, 2]

        # Jacobian of projection
        t = means_cam[i, 2]
        J = torch.tensor(
            [
                [focal_x / t, 0, -(focal_x * means_cam[i, 0]) / (t * t)],
                [0, focal_y / t, -(focal_y * means_cam[i, 1]) / (t * t)],
                [0, 0, 0],
            ],
            device=device,
        )

        # Transform covariance
        W = viewmatrix[:3, :3]
        T = J @ W
        cov2D = T @ cov3D @ T.T

        # Add low-pass filter
        cov2D[0, 0] += 0.3
        cov2D[1, 1] += 0.3

        # Pack
        cov2Ds[i, 0] = cov2D[0, 0]
        cov2Ds[i, 1] = cov2D[0, 1]
        cov2Ds[i, 2] = cov2D[1, 1]

    return cov2Ds


def project_points(
    points3D: torch.Tensor,
    projmatrix: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Project 3D points to 2D screen space.

    Args:
        points3D: 3D points [N, 3]
        projmatrix: Projection matrix [4, 4]
        image_width: Image width
        image_height: Image height

    Returns:
        points2D: 2D points [N, 2]
    """
    N = points3D.shape[0]
    device = points3D.device

    # Homogeneous coordinates
    points_h = torch.cat([points3D, torch.ones(N, 1, device=device)], dim=1)

    # Project
    points_clip = points_h @ projmatrix.T

    # Perspective divide
    points_ndc = points_clip[:, :3] / (points_clip[:, 3:4] + 1e-7)

    # To screen space
    points2D = torch.zeros(N, 2, device=device)
    points2D[:, 0] = ((points_ndc[:, 0] + 1.0) * image_width - 1.0) * 0.5
    points2D[:, 1] = ((points_ndc[:, 1] + 1.0) * image_height - 1.0) * 0.5

    return points2D


def transform_points(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Transform points by 4x4 matrix.

    Args:
        points: [N, 3] points
        matrix: [4, 4] transformation

    Returns:
        transformed: [N, 3] transformed points
    """
    N = points.shape[0]
    device = points.device

    points_h = torch.cat([points, torch.ones(N, 1, device=device)], dim=1)
    transformed_h = points_h @ matrix.T
    return transformed_h[:, :3]


def eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate spherical harmonics.

    Args:
        degree: SH degree (0, 1, 2, or 3)
        sh: SH coefficients [N, K, 3]
        dirs: Directions [N, 3]

    Returns:
        colors: [N, 3] colors
    """
    result = sh[:, 0, :]  # DC component

    if degree > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]

        # Degree 1
        result = result - sh[:, 1, :] * y + sh[:, 2, :] * z - sh[:, 3, :] * x

        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z

            # Degree 2
            result = (
                result
                + sh[:, 4, :] * (xy)
                + sh[:, 5, :] * (yz)
                + sh[:, 6, :] * (2.0 * zz - xx - yy)
                + sh[:, 7, :] * (xz)
                + sh[:, 8, :] * (xx - yy)
            )

        if degree > 2:
            # Degree 3
            result = (
                result
                + sh[:, 9, :] * (y * (3 * xx - yy))
                + sh[:, 10, :] * (xy * z)
                + sh[:, 11, :] * (y * (4 * zz - xx - yy))
                + sh[:, 12, :] * (z * (2 * zz - 3 * xx - 3 * yy))
                + sh[:, 13, :] * (x * (4 * zz - xx - yy))
                + sh[:, 14, :] * (z * (xx - yy))
                + sh[:, 15, :] * (x * (xx - 3 * yy))
            )

    return result


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
):
    """Get exponential learning rate schedule function.

    Args:
        lr_init: Initial learning rate
        lr_final: Final learning rate
        lr_delay_steps: Steps to delay before starting decay
        lr_delay_mult: Multiplier during delay
        max_steps: Total number of steps

    Returns:
        Function that computes LR for a given step
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0

        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Inverse sigmoid function."""
    return torch.log(x / (1 - x + 1e-10) + 1e-10)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # 3D Gaussians
    "Gaussian3D",
    "GaussianModel",
    "GaussianScene",
    # Rasterization
    "GaussianRasterizer",
    "DifferentiableRasterizer",
    "TileRasterizer",
    "RasterSettings",
    # Primitives
    "build_covariance_3d",
    "build_rotation_matrix",
    "build_scaling_matrix",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    # Loss Functions
    "L1Loss",
    "SSIMLoss",
    "CombinedLoss",
    # Optimization
    "GaussianOptimizer",
    "AdaptiveDensityControl",
    "SplitAndClone",
    "Pruning",
    # Training
    "GaussianTrainer",
    "GaussianDataLoader",
    "Camera",
    # Rendering
    "render_gaussians",
    "compute_color_from_sh",
    "get_view_matrix",
    "project_cov3D_to_cov2D",
    "project_points",
    "eval_sh",
]
