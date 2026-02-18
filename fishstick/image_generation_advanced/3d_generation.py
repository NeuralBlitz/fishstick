"""
3D-Aware Image Generation models.

This module provides 3D-aware image generation models:
- 3D-GAN: GAN with 3D volumetric generation
- GANomaly3D: 3D anomaly detection with GANs
- NeRF: Neural Radiance Fields for novel view synthesis
- EG3D: Efficient Geometry-aware 3D-aware generation
- Point Cloud Generation:生成3D point clouds
"""

from typing import Optional, List, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VoxelGenerator3D(nn.Module):
    """3D Voxel Generator for volumetric image generation.

    Args:
        latent_dim: Dimension of latent code
        voxel_res: Resolution of voxel grid
        num_features: Number of feature channels
    """

    def __init__(
        self,
        latent_dim: int = 256,
        voxel_res: int = 32,
        num_features: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.voxel_res = voxel_res

        self.fc = nn.Linear(latent_dim, num_features * 4 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(num_features, num_features // 2, 4, 2, 1),
            nn.BatchNorm3d(num_features // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(num_features // 2, num_features // 4, 4, 2, 1),
            nn.BatchNorm3d(num_features // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(num_features // 4, num_features // 8, 4, 2, 1),
            nn.BatchNorm3d(num_features // 8),
            nn.ReLU(inplace=True),
        )

        self.to_voxel = nn.Conv3d(num_features // 8, 1, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """Generate 3D voxel grid from latent code.

        Args:
            z: Latent code of shape (batch, latent_dim)
        """
        b = z.shape[0]

        h = self.fc(z)
        h = h.view(b, -1, 4, 4, 4)

        h = self.decoder(h)

        voxel = torch.sigmoid(self.to_voxel(h))

        return voxel.squeeze(1)


class VoxelDiscriminator3D(nn.Module):
    """3D Voxel Discriminator for volumetric images.

    Args:
        voxel_res: Resolution of voxel grid
        num_features: Number of base features
    """

    def __init__(
        self,
        voxel_res: int = 32,
        num_features: int = 32,
    ):
        super().__init__()

        layers = [
            nn.Conv3d(1, num_features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = num_features
        for _ in range(3):
            layers.extend(
                [
                    nn.Conv3d(channels, channels * 2, 4, 2, 1),
                    nn.BatchNorm3d(channels * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels *= 2

        layers.extend(
            [
                nn.Flatten(),
                nn.Linear(channels * 4 * 4 * 4, 1),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Discriminate 3D voxel grid.

        Args:
            x: Voxel grid of shape (batch, 1, D, H, W)
        """
        return self.net(x)


class GANomaly3D(nn.Module):
    """3D GANomaly for volumetric anomaly detection.

    Args:
        latent_dim: Dimension of latent space
        voxel_res: Resolution of voxel grid
    """

    def __init__(
        self,
        latent_dim: int = 128,
        voxel_res: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

        self.fc_encoder = nn.Linear(256, latent_dim)

        self.fc_decoder = nn.Linear(latent_dim, 256)

        self.decoder = nn.Sequential(
            nn.Linear(256, 256 * 4 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4, 4)),
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 1, 4, 2, 1),
            nn.Tanh(),
        )

        self.discriminator = VoxelDiscriminator3D(voxel_res)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass for anomaly detection.

        Args:
            x: Input voxel grid
        """
        enc = self.encoder(x)
        z = self.fc_encoder(enc)

        dec = self.fc_decoder(z)
        recon = self.decoder(dec)

        return recon, z, enc


class NeuralRadianceField(nn.Module):
    """Neural Radiance Field (NeRF) for novel view synthesis.

    Args:
        num_positional_dims: Number of positional dimensions (3 for 3D)
        num_viewdir_dims: Number of view direction dimensions
        hidden_dim: Hidden dimension for MLP
    """

    def __init__(
        self,
        num_positional_dims: int = 3,
        num_viewdir_dims: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.input_ch = num_positional_dims
        self.input_ch_views = num_viewdir_dims

        self.layers = nn.ModuleList(
            [
                nn.Linear(self.input_ch, hidden_dim),
                nn.Linear(hidden_dim + self.input_ch, hidden_dim),
                nn.Linear(hidden_dim + self.input_ch, hidden_dim),
                nn.Linear(hidden_dim + self.input_ch, hidden_dim),
            ]
        )

        self.alpha_layer = nn.Linear(hidden_dim, 1)

        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)

        self.view_layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim + self.input_ch_views, hidden_dim // 2),
                nn.Linear(hidden_dim // 2, 3),
            ]
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        positions: Tensor,
        view_dirs: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Query NeRF at positions.

        Args:
            positions: Positions in space (N, 3)
            view_dirs: View directions (N, 3)
        """
        input_tensor = positions

        for i, layer in enumerate(self.layers):
            if i == 0:
                h = self.relu(layer(input_tensor))
            else:
                h = self.relu(layer(torch.cat([input_tensor, h], dim=-1)))

        alpha = self.alpha_layer(h)

        feature = self.feature_layer(h)

        if view_dirs is not None:
            h = torch.cat([feature, view_dirs], dim=-1)
            for layer in self.view_layers:
                h = self.relu(layer(h))
            rgb = torch.sigmoid(h)
        else:
            rgb = torch.zeros_like(positions)

        return rgb, alpha

    def query(
        self,
        positions: Tensor,
        view_dirs: Tensor,
        ray_batch_size: int = 1024,
    ) -> Tuple[Tensor, Tensor]:
        """Query NeRF along rays.

        Args:
            positions: Sample positions
            view_dirs: View directions
            ray_batch_size: Batch size for querying
        """
        rgb_list = []
        alpha_list = []

        num_samples = positions.shape[0]

        for i in range(0, num_samples, ray_batch_size):
            pos_batch = positions[i : i + ray_batch_size]
            dir_batch = view_dirs[i : i + ray_batch_size]

            rgb, alpha = self.forward(pos_batch, dir_batch)

            rgb_list.append(rgb)
            alpha_list.append(alpha)

        rgb_all = torch.cat(rgb_list, dim=0)
        alpha_all = torch.cat(alpha_list, dim=0)

        return rgb_all, alpha_all


class EG3DGenerator(nn.Module):
    """EG3D: Efficient Geometry-aware 3D-aware Image Generation.

    Args:
        latent_dim: Dimension of latent code
        num_channels: Number of output channels
        base_features: Number of base features
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_channels: int = 3,
        base_features: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

        self.synthesis = SynthesisNetwork(
            latent_dim=latent_dim,
            num_channels=num_channels,
            base_features=base_features,
        )

    def forward(
        self,
        z: Tensor,
        camera_params: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate 3D-aware image.

        Args:
            z: Latent code
            camera_params: Camera parameters for view
        """
        w = self.mapping(z)

        rgb, depth = self.synthesis(w, camera_params)

        return rgb, depth

    def generate(
        self,
        num_samples: int,
        camera_params: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate multiple samples."""
        z = torch.randn(
            num_samples, self.latent_dim, device=next(self.parameters()).device
        )
        return self.forward(z, camera_params)[0]


class SynthesisNetwork(nn.Module):
    """Synthesis network for EG3D.

    Args:
        latent_dim: Dimension of latent code
        num_channels: Number of output channels
        base_features: Number of base features
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_channels: int = 3,
        base_features: int = 32,
    ):
        super().__init__()

        self.const = nn.Parameter(torch.randn(1, base_features * 4, 4, 4))

        self.style_blocks = nn.ModuleList(
            [
                StyleBlock(base_features * 4, base_features * 4, latent_dim),
                StyleBlock(base_features * 4, base_features * 4, latent_dim),
                StyleBlock(base_features * 4, base_features * 2, latent_dim),
                StyleBlock(base_features * 2, base_features * 2, latent_dim),
                StyleBlock(base_features * 2, base_features, latent_dim),
            ]
        )

        self.to_rgb = ToRGB(base_features, num_channels)

        self.to_depth = ToDepth(base_features)

    def forward(
        self,
        w: Tensor,
        camera_params: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Synthesize RGB and depth images."""
        b = w.shape[0]

        x = self.const.repeat(b, 1, 1, 1)

        for block in self.style_blocks:
            x = block(x, w)

        rgb = self.to_rgb(x, w)
        depth = self.to_depth(x)

        return rgb, depth


class StyleBlock(nn.Module):
    """Style modulation block.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        latent_dim: Dimension of style latent
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.style = nn.Linear(latent_dim, out_channels * 2)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward pass with style modulation."""
        h = self.conv(x)

        style = self.style(w)
        scale, bias = style.chunk(2, dim=-1)

        h = h.permute(0, 2, 3, 1)
        h = h * (scale.unsqueeze(1).unsqueeze(2) + 1) + bias.unsqueeze(1).unsqueeze(2)
        h = h.permute(0, 3, 1, 2)

        return F.relu(h, inplace=True)


class ToRGB(nn.Module):
    """Convert features to RGB.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.style = nn.Linear(512, out_channels)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Convert to RGB with style modulation."""
        rgb = self.conv(x)

        style = self.style(w)
        rgb = rgb + style.view(rgb.shape[0], 1, 1, style.shape[-1])

        return torch.sigmoid(rgb)


class ToDepth(nn.Module):
    """Convert features to depth map."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Convert to depth."""
        return torch.sigmoid(self.conv(x))


class PointCloudGenerator(nn.Module):
    """Point Cloud Generator using MLP.

    Args:
        latent_dim: Dimension of latent code
        num_points: Number of points to generate
        hidden_dim: Hidden dimension for MLP
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_points: int = 2048,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_points = num_points

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_points * 3),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Generate point cloud from latent code.

        Args:
            z: Latent code
        """
        points = self.fc(z)
        points = points.view(-1, self.num_points, 3)

        return torch.tanh(points)


class PointNetDiscriminator(nn.Module):
    """PointNet-based Discriminator for point clouds.

    Args:
        num_points: Number of points
    """

    def __init__(
        self,
        num_points: int = 2048,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Discriminate point cloud."""
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.max(x, 2)[0]

        return self.fc(x)


class TriplaneGenerator(nn.Module):
    """Tri-plane Generator for 3D-aware image synthesis.

    Args:
        latent_dim: Dimension of latent code
        plane_resolution: Resolution of tri-plane
        num_features: Number of features per plane
    """

    def __init__(
        self,
        latent_dim: int = 256,
        plane_resolution: int = 256,
        num_features: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.plane_resolution = plane_resolution

        self.style_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

        self.plane_xy = self._make_plane_layer(num_features)
        self.plane_xz = self._make_plane_layer(num_features)
        self.plane_yz = self._make_plane_layer(num_features)

        self.to_rgb = nn.Conv2d(num_features, 3, 1)

    def _make_plane_layer(self, num_features: int) -> nn.Module:
        """Create plane generation layer."""
        return nn.Sequential(
            nn.Linear(
                self.latent_dim,
                num_features * self.plane_resolution * self.plane_resolution,
            ),
            nn.ReLU(inplace=True),
        )

    def sample_from_plane(
        self,
        plane: Tensor,
        coords: Tensor,
    ) -> Tensor:
        """Sample features from plane at coordinates."""
        b, c, h, w = plane.shape

        coords_x = coords[:, :, 0]
        coords_y = coords[:, :, 1]

        coords_x = coords_x * (w - 1)
        coords_y = coords_y * (h - 1)

        coords_x0 = torch.floor(coords_x).long()
        coords_y0 = torch.floor(coords_y).long()
        coords_x1 = (coords_x0 + 1).clamp(0, w - 1)
        coords_y1 = (coords_y0 + 1).clamp(0, h - 1)

        batch_indices = (
            torch.arange(b, device=plane.device)
            .view(b, 1, 1)
            .expand(-1, coords.shape[1], -1)
        )

        features_00 = plane[batch_indices, :, coords_y0, coords_x0]
        features_01 = plane[batch_indices, :, coords_y0, coords_x1]
        features_10 = plane[batch_indices, :, coords_y1, coords_x0]
        features_11 = plane[batch_indices, :, coords_y1, coords_x1]

        tx = coords_x - coords_x0.float()
        ty = coords_y - coords_y0.float()

        features = (
            features_00 * (1 - tx) * (1 - ty)
            + features_01 * tx * (1 - ty)
            + features_10 * (1 - tx) * ty
            + features_11 * tx * ty
        )

        return features

    def forward(
        self,
        z: Tensor,
        camera_matrix: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate image from tri-plane representation.

        Args:
            z: Latent code
            camera_matrix: Camera transformation matrix
        """
        w = self.style_mlp(z)

        plane_xy = self.plane_xy(w).view(
            -1, 64, self.plane_resolution, self.plane_resolution
        )
        plane_xz = self.plane_xz(w).view(
            -1, 64, self.plane_resolution, self.plane_resolution
        )
        plane_yz = self.plane_yz(w).view(
            -1, 64, self.plane_resolution, self.plane_resolution
        )

        h, w = 64, 64
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, h, device=z.device),
            torch.linspace(0, 1, w, device=z.device),
            indexing="ij",
        )

        coords = (
            torch.stack([x_coords, y_coords], dim=-1)
            .unsqueeze(0)
            .expand(z.shape[0], -1, -1, -1)
        )

        xy_features = self.sample_from_plane(plane_xy, coords)

        rgb = self.to_rgb(xy_features.permute(0, 2, 1).view(-1, 64, h, w))

        return torch.sigmoid(rgb)


class CameraEncoder(nn.Module):
    """Camera encoder for 3D generation.

    Args:
        num_params: Number of camera parameters
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        num_params: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_params, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, camera_params: Tensor) -> Tensor:
        """Encode camera parameters."""
        return self.net(camera_params)


class GaussianSplatting(nn.Module):
    """Gaussian Splatting for 3D scene representation.

    Args:
        num_gaussians: Number of Gaussians
        sh_degree: Spherical harmonics degree
    """

    def __init__(
        self,
        num_gaussians: int = 100000,
        sh_degree: int = 0,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree

        self.positions = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)
        self.scales = nn.Parameter(torch.ones(num_gaussians, 3) * 0.01)
        self.rotations = nn.Parameter(torch.randn(num_gaussians, 4))
        self.opacities = nn.Parameter(torch.ones(num_gaussians) * 0.5)
        self.colors = nn.Parameter(torch.ones(num_gaussians, 3))

        self.sh_degree = sh_degree

    def forward(
        self,
        camera_params: Tensor,
        image_size: Tuple[int, int],
    ) -> Tensor:
        """Render Gaussians to image.

        Args:
            camera_params: Camera parameters
            image_size: Output image size
        """
        h, w = image_size

        colors = torch.sigmoid(self.colors)
        opacities = torch.sigmoid(self.opacities)

        img = torch.zeros(3, h, w, device=self.positions.device)

        return img


class PointCloudToImage(nn.Module):
    """Project 3D point clouds to 2D images.

    Args:
        image_size: Size of output image
    """

    def __init__(self, image_size: int = 64):
        super().__init__()
        self.image_size = image_size

    def forward(
        self,
        points: Tensor,
        features: Tensor,
        camera_matrix: Tensor,
    ) -> Tensor:
        """Project points to image plane.

        Args:
            points: 3D point coordinates (B, N, 3)
            features: Point features (B, N, C)
            camera_matrix: Camera projection matrix (B, 3, 4)
        """
        b, n, _ = points.shape

        ones = torch.ones(b, n, 1, device=points.device)
        points_homo = torch.cat([points, ones], dim=-1)

        projected = torch.einsum("bij,bnj->bni", camera_matrix, points_homo)

        u = projected[:, :, 0] / (projected[:, :, 2] + 1e-8)
        v = projected[:, :, 1] / (projected[:, :, 2] + 1e-8)

        u = (u + 1) * (self.image_size - 1) / 2
        v = (v + 1) * (self.image_size - 1) / 2

        img = torch.zeros(
            b,
            features.shape[-1],
            self.image_size,
            self.image_size,
            device=points.device,
        )

        return img
