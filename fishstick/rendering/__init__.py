"""
Neural Rendering Module

Implements:
- NeRF (Neural Radiance Fields) with positional encoding, coarse-fine sampling
- Instant NGP (Instant Neural Graphics Primitives) with hash grid encoding
- GLO (Generative Latent Optimization) for embedding optimization
- DVR (Differentiable Volumetric Rendering) for 3D reconstruction

Components:
- Ray marching
- Volume rendering equation
- Camera poses
- Sample network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
from dataclasses import dataclass


# =============================================================================
# Mathematical Utilities
# =============================================================================


def positional_encoding(x: torch.Tensor, L: int = 10) -> torch.Tensor:
    """
    Apply positional encoding to input coordinates.

    Args:
        x: Input tensor of shape (..., 1) or (..., D)
        L: Number of frequency bands

    Returns:
        Encoded tensor of shape (..., 2*L*D) or (..., 2*L)
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    orig_shape = x.shape
    x = x.reshape(-1, 1)

    freqs = 2 ** torch.arange(L, device=x.device, dtype=torch.float32) * math.pi
    angles = x * freqs

    encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return encoded.reshape(*orig_shape[:-1], -1)


def safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe normalization for vectors."""
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


# =============================================================================
# Camera Poses
# =============================================================================


@dataclass
class CameraPose:
    """Camera pose representation."""

    position: torch.Tensor
    rotation: torch.Tensor  # 3x3 rotation matrix
    focal_length: float = 1.0
    principal_point: Tuple[float, float] = (0.0, 0.0)

    @property
    def camera_matrix(self) -> torch.Tensor:
        """Get camera intrinsic matrix."""
        fx = self.focal_length
        fy = self.focal_length
        cx, cy = self.principal_point
        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            device=self.position.device,
            dtype=self.position.dtype,
        )
        return K

    def get_view_matrix(self) -> torch.Tensor:
        """Get 4x4 view matrix."""
        R = self.rotation
        t = -R @ self.position
        V = torch.eye(4, device=R.device, dtype=R.dtype)
        V[:3, :3] = R
        V[:3, 3] = t
        return V

    @staticmethod
    def look_at(
        eye: torch.Tensor,
        target: torch.Tensor,
        up: torch.Tensor,
        focal_length: float = 1.0,
    ) -> "CameraPose":
        """
        Create camera pose from look-at parameters.

        Args:
            eye: Camera position
            target: Look-at target
            up: Up vector
            focal_length: Focal length

        Returns:
            CameraPose object
        """
        forward = safe_normalize(target - eye)
        right = safe_normalize(torch.cross(forward, up))
        up_new = torch.cross(right, forward)

        rotation = torch.stack([right, up_new, -forward], dim=0)

        return CameraPose(position=eye, rotation=rotation, focal_length=focal_length)

    @staticmethod
    def orbit_poses(
        n_poses: int,
        radius: float = 4.0,
        elevation: float = 30.0,
        focal_length: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> List["CameraPose"]:
        """
        Generate orbit camera poses.

        Args:
            n_poses: Number of poses to generate
            radius: Orbit radius
            elevation: Elevation angle in degrees
            focal_length: Focal length
            device: Device to create tensors on

        Returns:
            List of CameraPose objects
        """
        poses = []
        elevation_rad = math.radians(elevation)
        up = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)

        for i in range(n_poses):
            theta = 2 * math.pi * i / n_poses
            x = radius * math.cos(elevation_rad) * math.sin(theta)
            y = radius * math.sin(elevation_rad)
            z = radius * math.cos(elevation_rad) * math.cos(theta)

            eye = torch.tensor([x, y, z], device=device, dtype=torch.float32)
            target = torch.zeros(3, device=device, dtype=torch.float32)

            pose = CameraPose.look_at(eye, target, up, focal_length)
            poses.append(pose)

        return poses


# =============================================================================
# Ray Marching
# =============================================================================


@dataclass
class Ray:
    """Ray representation for ray marching."""

    origin: torch.Tensor  # (..., 3)
    direction: torch.Tensor  # (..., 3)

    def at(self, t: torch.Tensor) -> torch.Tensor:
        """Get point along ray at parameter t."""
        origin = self.origin
        direction = self.direction

        if origin.dim() == 1:
            origin = origin.unsqueeze(0)
        if direction.dim() == 1:
            direction = direction.unsqueeze(0)

        t_expanded = t
        if t_expanded.dim() == 1:
            t_expanded = t_expanded.unsqueeze(0)

        while t_expanded.dim() < origin.dim():
            t_expanded = t_expanded.unsqueeze(-1)

        t_expanded = t_expanded.expand(*origin.shape[:-1], -1)

        return origin.unsqueeze(-2) + t_expanded.unsqueeze(-1) * direction.unsqueeze(-2)


class RayMarcher:
    """Ray marching implementation for volume rendering."""

    def __init__(
        self,
        near: float = 0.0,
        far: float = 4.0,
        n_steps: int = 128,
        delta: float = 0.001,
    ):
        self.near = near
        self.far = far
        self.n_steps = n_steps
        self.delta = delta

    def generate_samples(
        self, rays: Ray, perturb: bool = True, stratified: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sample points along rays.

        Args:
            rays: Ray object containing origins and directions
            perturb: Whether to add perturbation to samples
            stratified: Whether to use stratified sampling

        Returns:
            Tuple of (sample points, sample deltas)
        """
        batch_shape = rays.origin.shape[:-1]

        t_vals = torch.linspace(
            self.near,
            self.far,
            self.n_steps,
            device=rays.origin.device,
            dtype=rays.origin.dtype,
        )

        if stratified:
            t_vals = (
                t_vals
                + torch.rand(
                    *batch_shape,
                    self.n_steps,
                    device=rays.origin.device,
                    dtype=rays.origin.dtype,
                )
                * (self.far - self.near)
                / self.n_steps
            )

        if perturb and stratified:
            mid = (t_vals[..., :-1] + t_vals[..., 1:]) / 2
            upper = torch.cat([mid, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mid], dim=-1)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand

        deltas = t_vals[..., 1:] - t_vals[..., :-1]
        deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], 1e10)], dim=-1)

        return t_vals, deltas

    def march(
        self, rays: Ray, density_fn, sample_network, perturb: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        March rays through volume and compute colors/densities.

        Args:
            rays: Ray objects
            density_fn: Function that computes density at points
            sample_network: Network to query for color/density
            perturb: Whether to perturb samples

        Returns:
            Tuple of (rgb, weights, depth, acc)
        """
        t_vals, deltas = self.generate_samples(rays, perturb)
        points = rays.at(t_vals)

        sample_points = points.reshape(-1, 3)
        ray_dirs = rays.direction.unsqueeze(-2).expand(points.shape).reshape(-1, 3)

        rgb, sigma = sample_network(sample_points, ray_dirs.reshape(-1, 3))

        sigma = sigma.reshape(*points.shape[:-1], 1)
        rgb = rgb.reshape(*points.shape[:-1], 3)

        sigma_delta = sigma * deltas.unsqueeze(-1)
        alpha = 1.0 - torch.exp(-sigma_delta)

        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [
                        torch.ones((*alpha.shape[:-1], 1), device=alpha.device),
                        1.0 - alpha + 1e-10,
                    ],
                    dim=-1,
                ),
                dim=-1,
            )[..., :-1]
        )

        rgb_weights = weights * rgb
        rgb_final = rgb_weights.sum(dim=-2)
        depth = (weights.squeeze(-1) * t_vals).sum(dim=-1)
        acc = weights.sum(dim=-1).clamp(0, 1)

        return rgb_final, weights, depth, acc


# =============================================================================
# Volume Rendering Equation
# =============================================================================


class VolumeRenderer:
    """Volume rendering equation implementation."""

    @staticmethod
    def compute_transmittance(
        sigma: torch.Tensor, deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute transmittance T = exp(-integral(sigma dt)).

        Args:
            sigma: Density values
            deltas: Distance between samples

        Returns:
            Transmittance values
        """
        sigma_delta = sigma * deltas.unsqueeze(-1)
        cumsum = torch.cumsum(sigma_delta, dim=-2)
        transmittance = torch.exp(-cumsum)

        ones = torch.ones(
            *transmittance.shape[:-2],
            1,
            1,
            device=transmittance.device,
            dtype=transmittance.dtype,
        )

        return torch.cat([ones, transmittance[..., :-1, :]], dim=-2)

    @staticmethod
    def render(
        rgb: torch.Tensor,
        sigma: torch.Tensor,
        deltas: torch.Tensor,
        background: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply volume rendering equation.

        C = sum(T * (1 - exp(-sigma * delta)) * rgb)
        T = exp(-sum(sigma * delta))

        Args:
            rgb: RGB colors at samples
            sigma: Density at samples
            deltas: Distance between samples
            background: Optional background color

        Returns:
            Tuple of (rendered color, weights, transmittance)
        """
        transmittance = VolumeRenderer.compute_transmittance(sigma, deltas)

        alpha = 1.0 - torch.exp(-sigma * deltas.unsqueeze(-1))
        weights = transmittance * alpha

        rgb_weights = weights * rgb
        color = rgb_weights.sum(dim=-2)

        transmittance_final = transmittance[..., -1:]

        if background is not None:
            color = color + transmittance_final * background

        acc = weights.sum(dim=-1).clamp(0, 1)

        return color, weights, transmittance


# =============================================================================
# Sample Network (NeRF MLP)
# =============================================================================


class SampleNetwork(nn.Module):
    """MLP network for NeRF sample queries."""

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 8,
        positional_encoding_dims: int = 10,
        use_viewdirs: bool = True,
        skips: List[int] = [4],
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.positional_encoding_dims = positional_encoding_dims
        self.use_viewdirs = use_viewdirs
        self.skips = skips

        self.pe_dim = 2 * positional_encoding_dims * input_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.pe_dim, hidden_dim))

        for i in range(1, n_layers):
            if i in skips:
                self.layers.append(nn.Linear(hidden_dim + self.pe_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        if use_viewdirs:
            self.view_layers = nn.ModuleList(
                [
                    nn.Linear(hidden_dim + 3, hidden_dim // 2),
                    nn.Linear(hidden_dim // 2, 3),
                ]
            )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.sigma_activation = nn.ReLU()
        self.rgb_activation = nn.Sigmoid()

    def forward(
        self, points: torch.Tensor, viewdirs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through sample network.

        Args:
            points: Sample points (N, 3)
            viewdirs: View directions (N, 3)

        Returns:
            Tuple of (RGB, density)
        """
        points_encoded = positional_encoding(points, self.positional_encoding_dims)

        x = points_encoded
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                x = torch.cat([x, points_encoded], dim=-1)
            x = layer(x)
            x = F.relu(x)

        sigma = self.sigma_activation(self.output_layer(x)[..., :1])

        if self.use_viewdirs and viewdirs is not None:
            x = torch.cat([x, viewdirs], dim=-1)

            for layer in self.view_layers:
                x = F.relu(layer(x))

            rgb = self.rgb_activation(x)
        else:
            rgb = self.rgb_activation(self.output_layer(x)[..., 1:])

        return rgb, sigma


# =============================================================================
# NeRF (Neural Radiance Fields)
# =============================================================================


class NeRF(nn.Module):
    """
    Full NeRF implementation with coarse-fine sampling.

    Architecture:
    - Positional encoding for positions and view directions
    - 8-layer MLP for coarse sampling
    - 8-layer MLP for fine sampling
    - Hierarchical volume sampling
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        positional_encoding_dims: int = 10,
        use_viewdirs: bool = True,
        n_coarse_samples: int = 64,
        n_fine_samples: int = 128,
        near: float = 0.0,
        far: float = 4.0,
    ):
        super().__init__()

        self.n_coarse_samples = n_coarse_samples
        self.n_fine_samples = n_fine_samples
        self.near = near
        self.far = far

        self.coarse_network = SampleNetwork(
            input_dim=3,
            output_dim=4,
            hidden_dim=hidden_dim,
            n_layers=8,
            positional_encoding_dims=positional_encoding_dims,
            use_viewdirs=use_viewdirs,
            skips=[4],
        )

        self.fine_network = SampleNetwork(
            input_dim=3,
            output_dim=4,
            hidden_dim=hidden_dim,
            n_layers=8,
            positional_encoding_dims=positional_encoding_dims,
            use_viewdirs=use_viewdirs,
            skips=[4],
        )

        self.coarse_marcher = RayMarcher(near, far, n_coarse_samples)
        self.fine_marcher = RayMarcher(near, far, n_fine_samples)

        self.volume_renderer = VolumeRenderer()

    def forward(
        self, rays: Ray, return_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through NeRF.

        Args:
            rays: Ray objects
            return_weights: Whether to return sample weights

        Returns:
            Tuple of (rendered RGB, depth)
        """
        rgb_coarse, weights_coarse, depth_coarse, acc_coarse = (
            self.coarse_marcher.march(rays, None, self.coarse_network, perturb=True)
        )

        t_vals_fine = self._resample(rays, weights_coarse, self.n_fine_samples)

        points_fine = rays.at(t_vals_fine)
        deltas_fine = t_vals_fine[..., 1:] - t_vals_fine[..., :-1]
        deltas_fine = torch.cat(
            [deltas_fine, torch.full_like(deltas_fine[..., :1], 1e10)], dim=-1
        )

        sample_points = points_fine.reshape(-1, 3)
        ray_dirs = rays.direction.unsqueeze(-2).expand(points_fine.shape).reshape(-1, 3)

        rgb_fine, sigma_fine = self.fine_network(sample_points, ray_dirs.reshape(-1, 3))

        rgb_fine = rgb_fine.reshape(*points_fine.shape[:-1], 3)
        sigma_fine = sigma_fine.reshape(*points_fine.shape[:-1], 1)

        rgb_final, weights_fine, depth_fine = self.volume_renderer.render(
            rgb_fine, sigma_fine, deltas_fine
        )

        if return_weights:
            return rgb_final, depth_fine, weights_fine

        return rgb_final, depth_fine

    def _resample(
        self, rays: Ray, weights: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        """Resample based on coarse weights."""
        batch_size = weights.shape[0]

        t_vals = torch.linspace(
            self.near, self.far, self.n_coarse_samples + 1, device=rays.origin.device
        )
        t_vals = t_vals.unsqueeze(0).expand(batch_size, -1)

        mid = (t_vals[..., :-1] + t_vals[..., 1:]) / 2
        upper = torch.cat([mid, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mid], dim=-1)

        cdf = torch.cumsum(weights.squeeze(-1), dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        u = torch.linspace(0, 1, n_samples, device=rays.origin.device)
        u = u.unsqueeze(0).expand(batch_size, -1)

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(inds, 0, cdf.shape[-1] - 1)

        cdf_gather_below = torch.gather(cdf, -1, below)
        cdf_gather_above = torch.gather(cdf, -1, above)

        t_vals_gather_below = torch.gather(lower, -1, below)
        t_vals_gather_above = torch.gather(upper, -1, above)

        denom = cdf_gather_above - cdf_gather_below
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

        t = (u - cdf_gather_below) / denom
        t_vals_new = t_vals_gather_below + t * (
            t_vals_gather_above - t_vals_gather_below
        )

        return t_vals_new


# =============================================================================
# Instant NGP (Instant Neural Graphics Primitives)
# =============================================================================


class HashGridEncoding(nn.Module):
    """Hash grid encoding for efficient neural rendering."""

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        hash_init_scale: float = 0.001,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        self.register_buffer("offsets", self._compute_offsets())

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(2 ** min(i + 1, log2_hashmap_size), n_features_per_level)
                for i in range(n_levels)
            ]
        )

        for emb in self.embeddings:
            emb.weight.data.uniform_(-hash_init_scale, hash_init_scale)

    def _compute_offsets(self) -> torch.Tensor:
        """Compute offsets for each hash level."""
        offsets = []
        for i in range(self.n_levels):
            resolution = self.base_resolution * (2**i)
            resolution = min(resolution, self.finest_resolution)
            offsets.append(min(2**self.log2_hashmap_size, (resolution + 1) ** 3))
        offsets = torch.cumsum(
            torch.tensor([0] + offsets[:-1], dtype=torch.long), dim=0
        )
        return offsets

    def _hash_coords(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        """Hash coordinates using random primes."""
        primes = [1, 2654435761, 29675113, 123456789]

        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]

        h = x * primes[0]
        h = h * primes[1] + y
        h = h * primes[2] + z
        h = h * primes[3]

        h = h % self.embeddings[level].num_embeddings

        return h

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hash grid.

        Args:
            coords: Coordinates in [-bound, bound] range

        Returns:
            Encoded features
        """
        batch_size = coords.shape[0]
        levels_features = []

        for level in range(self.n_levels):
            resolution = self.base_resolution * (2**level)
            resolution = min(resolution, self.finest_resolution)

            coords_scaled = (coords + 1.0) * resolution

            coords_floor = torch.floor(coords_scaled).long()
            coords_frac = coords_scaled - coords_floor.float()

            x, y, z = coords_floor[..., 0], coords_floor[..., 1], coords_floor[..., 2]

            c000 = torch.stack([x, y, z], dim=-1)
            c100 = torch.stack([x + 1, y, z], dim=-1)
            c010 = torch.stack([x, y + 1, z], dim=-1)
            c110 = torch.stack([x + 1, y + 1, z], dim=-1)
            c001 = torch.stack([x, y, z + 1], dim=-1)
            c101 = torch.stack([x + 1, y, z + 1], dim=-1)
            c011 = torch.stack([x, y + 1, z + 1], dim=-1)
            c111 = torch.stack([x + 1, y + 1, z + 1], dim=-1)

            indices = torch.stack(
                [c000, c100, c010, c110, c001, c101, c011, c111], dim=0
            )

            hash_indices = self._hash_coords(indices, level)

            features = self.embeddings[level](hash_indices)

            tx = coords_frac[..., 0:1]
            ty = coords_frac[..., 1:2]
            tz = coords_frac[..., 2:3]

            def lerp(a, b, t):
                return a + t * (b - a)

            def trilinear_interp(
                c000, c100, c010, c110, c001, c101, c011, c111, x, y, z
            ):
                return (
                    c000 * (1 - x) * (1 - y) * (1 - z)
                    + c100 * x * (1 - y) * (1 - z)
                    + c010 * (1 - x) * y * (1 - z)
                    + c110 * x * y * (1 - z)
                    + c001 * (1 - x) * (1 - y) * z
                    + c101 * x * (1 - y) * z
                    + c011 * (1 - x) * y * z
                    + c111 * x * y * z
                )

            features_interp = trilinear_interp(
                features[0],
                features[1],
                features[2],
                features[3],
                features[4],
                features[5],
                features[6],
                features[7],
                tx,
                ty,
                tz,
            )

            levels_features.append(features_interp)

        return torch.cat(levels_features, dim=-1)


class InstantNGP(nn.Module):
    """
    Instant NGP - Hash Grid based NeRF.

    Uses:
    - Hash grid encoding for positions
    - Smaller MLP for efficiency
    - Multi-resolution hash tables
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_coarse_samples: int = 64,
        n_fine_samples: int = 128,
        near: float = 0.0,
        far: float = 4.0,
    ):
        super().__init__()

        self.n_coarse_samples = n_coarse_samples
        self.n_fine_samples = n_fine_samples
        self.near = near
        self.far = far

        self.hash_encoding = HashGridEncoding(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
        )

        encoding_dim = n_levels * n_features_per_level

        self.mlp = nn.Sequential(
            nn.Linear(encoding_dim + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

        self.sigma_activation = nn.ReLU()
        self.rgb_activation = nn.Sigmoid()

        self.coarse_marcher = RayMarcher(near, far, n_coarse_samples)
        self.fine_marcher = RayMarcher(near, far, n_fine_samples)
        self.volume_renderer = VolumeRenderer()

    def query(
        self, points: torch.Tensor, viewdirs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the network at given points.

        Args:
            points: Sample points
            viewdirs: View directions

        Returns:
            Tuple of (RGB, density)
        """
        coords = points.clamp(-1.0, 1.0)
        encoded = self.hash_encoding(coords)

        viewdirs_norm = safe_normalize(viewdirs)

        x = torch.cat([encoded, viewdirs_norm], dim=-1)

        output = self.mlp(x)

        sigma = self.sigma_activation(output[..., :1])
        rgb = self.rgb_activation(output[..., 1:])

        return rgb, sigma

    def forward(
        self, rays: Ray, return_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Instant NGP.

        Args:
            rays: Ray objects
            return_weights: Whether to return weights

        Returns:
            Tuple of (rendered RGB, depth)
        """
        t_vals, deltas = self.coarse_marcher.generate_samples(rays, perturb=True)
        points = rays.at(t_vals)

        sample_points = points.reshape(-1, 3)
        ray_dirs = rays.direction.unsqueeze(-2).expand(points.shape).reshape(-1, 3)

        rgb_coarse, sigma_coarse = self.query(sample_points, ray_dirs.reshape(-1, 3))

        rgb_coarse = rgb_coarse.reshape(*points.shape[:-1], 3)
        sigma_coarse = sigma_coarse.reshape(*points.shape[:-1], 1)

        rgb_c, weights_c, _ = self.volume_renderer.render(
            rgb_coarse, sigma_coarse, deltas
        )

        t_vals_fine = self._resample(rays, weights_c, self.n_fine_samples)
        points_fine = rays.at(t_vals_fine)

        deltas_fine = t_vals_fine[..., 1:] - t_vals_fine[..., :-1]
        deltas_fine = torch.cat(
            [deltas_fine, torch.full_like(deltas_fine[..., :1], 1e10)], dim=-1
        )

        sample_points_fine = points_fine.reshape(-1, 3)
        ray_dirs_fine = rays.direction.unsqueeze(-2).expand_as(points_fine)

        rgb_fine, sigma_fine = self.query(
            sample_points_fine, ray_dirs_fine.reshape(-1, 3)
        )

        rgb_fine = rgb_fine.reshape(*points_fine.shape[:-1], 3)
        sigma_fine = sigma_fine.reshape(*points_fine.shape[:-1], 1)

        rgb_final, weights_fine, depth_fine = self.volume_renderer.render(
            rgb_fine, sigma_fine, deltas_fine
        )

        if return_weights:
            return rgb_final, depth_fine, weights_fine

        return rgb_final, depth_fine

    def _resample(
        self, rays: Ray, weights: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        """Resample based on weights."""
        batch_size = weights.shape[0]

        t_vals = torch.linspace(
            self.near, self.far, self.n_coarse_samples + 1, device=rays.origin.device
        )
        t_vals = t_vals.unsqueeze(0).expand(batch_size, -1)

        mid = (t_vals[..., :-1] + t_vals[..., 1:]) / 2
        upper = torch.cat([mid, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mid], dim=-1)

        cdf = torch.cumsum(weights.squeeze(-1), dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        u = torch.linspace(0, 1, n_samples, device=rays.origin.device)
        u = u.unsqueeze(0).expand(batch_size, -1)

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(inds, 0, cdf.shape[-1] - 1)

        cdf_gather_below = torch.gather(cdf, -1, below)
        cdf_gather_above = torch.gather(cdf, -1, above)

        t_vals_gather_below = torch.gather(lower, -1, below)
        t_vals_gather_above = torch.gather(upper, -1, above)

        denom = cdf_gather_above - cdf_gather_below
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

        t = (u - cdf_gather_below) / denom
        t_vals_new = t_vals_gather_below + t * (
            t_vals_gather_above - t_vals_gather_below
        )

        return t_vals_new


# =============================================================================
# GLO (Generative Latent Optimization)
# =============================================================================


class GLOEmbeddings(nn.Module):
    """
    GLO (Generative Latent Optimization) embeddings.

    Learnable latent codes that can be optimized to match target outputs.
    """

    def __init__(
        self, n_codes: int, code_dim: int, init: str = "normal", init_std: float = 0.01
    ):
        super().__init__()

        self.n_codes = n_codes
        self.code_dim = code_dim

        if init == "normal":
            self.codes = nn.Parameter(torch.randn(n_codes, code_dim) * init_std)
        elif init == "uniform":
            self.codes = nn.Parameter(torch.rand(n_codes, code_dim) * 2 - 1)
        else:
            self.codes = nn.Parameter(torch.zeros(n_codes, code_dim))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for given indices.

        Args:
            indices: Indices into codebook

        Returns:
            Embeddings (N, code_dim)
        """
        return self.codes[indices]

    def get_all_codes(self) -> torch.Tensor:
        """Get all learnable codes."""
        return self.codes


class GLOOptimizer:
    """
    GLO optimizer for embedding optimization.

    Optimizes latent codes to minimize reconstruction loss.
    """

    def __init__(
        self,
        embedding_module: GLOEmbeddings,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ):
        self.embedding_module = embedding_module
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.exp_avg = torch.zeros_like(embedding_module.codes.data)
        self.exp_avg_sq = torch.zeros_like(embedding_module.codes.data)
        self.step = 0

    def step(self, loss: torch.Tensor) -> None:
        """Perform optimization step."""
        loss.backward()

        with torch.no_grad():
            self.step += 1

            exp_avg, exp_avg_sq = self.exp_avg, self.exp_avg_sq

            self.exp_avg.mul_(self.beta1).add_(
                self.embedding_module.codes.grad, alpha=1 - self.beta1
            )
            self.exp_avg_sq.mul_(self.beta2).addcmul_(
                self.embedding_module.codes.grad,
                self.embedding_module.codes.grad,
                value=1 - self.beta2,
            )

            bias_correction1 = 1 - self.beta1**self.step
            bias_correction2 = 1 - self.beta2**self.step

            step_size = self.lr / bias_correction1

            denom = (self.exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)

            self.embedding_module.codes.addcdiv_(self.exp_avg, denom, value=-step_size)

            self.embedding_module.codes.grad.zero_()

    def zero_grad(self) -> None:
        """Zero gradients."""
        if self.embedding_module.codes.grad is not None:
            self.embedding_module.codes.grad.zero_()


# =============================================================================
# DVR (Differentiable Volumetric Rendering)
# =============================================================================


class DVRNetwork(nn.Module):
    """
    DVR (Differentiable Volumetric Rendering) network.

    For 3D reconstruction from images using volumetric rendering.
    """

    def __init__(
        self,
        sdf_network: nn.Module,
        color_network: nn.Module,
        n_samples: int = 128,
        n_importance: int = 64,
        near: float = 0.0,
        far: float = 4.0,
    ):
        super().__init__()

        self.sdf_network = sdf_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.near = near
        self.far = far

    def query_sdf(self, points: torch.Tensor) -> torch.Tensor:
        """Query SDF network."""
        return self.sdf_network(points)

    def query_color(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        viewdirs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Query color network."""
        return self.color_network(points, normals, viewdirs)

    def forward(
        self, rays: Ray, return_sdf: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DVR.

        Args:
            rays: Ray objects
            return_sdf: Whether to return SDF values

        Returns:
            Tuple of (rendered RGB, depth)
        """
        t_vals = torch.linspace(
            self.near,
            self.far,
            self.n_samples,
            device=rays.origin.device,
            dtype=rays.origin.dtype,
        )

        t_vals = (
            t_vals + torch.rand_like(t_vals) * (self.far - self.near) / self.n_samples
        )

        points = rays.at(t_vals)

        sdf_values = self.query_sdf(points)

        with torch.no_grad():
            sorted_indices = torch.argsort(sdf_values, dim=-1)
            t_vals_sorted = torch.gather(t_vals, -1, sorted_indices)
            sdf_sorted = torch.gather(sdf_values, -1, sorted_indices)

        dists = t_vals_sorted[..., 1:] - t_vals_sorted[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        sdf_mid = 0.5 * (sdf_sorted[..., :-1] + sdf_sorted[..., 1:])

        sdf_interp = sdf_sorted
        weights = self._sdf_to_weights(sdf_interp, dists)

        rgb_samples = self.query_color(
            points.reshape(-1, 3),
            None,
            rays.direction.unsqueeze(-2).expand(len(weights), -1, -1).reshape(-1, 3),
        )
        rgb = rgb_samples.reshape(*points.shape[:-1], 3)

        rgb_final = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
        depth = (weights * t_vals_sorted).sum(dim=-1)

        if return_sdf:
            return rgb_final, depth, sdf_sorted

        return rgb_final, depth

    def _sdf_to_weights(self, sdf: torch.Tensor, dists: torch.Tensor) -> torch.Tensor:
        """Convert SDF to weights for volume rendering."""
        eps = 1e-5

        sdf_next = sdf[..., 1:]
        sdf_curr = sdf[..., :-1]

        dists_next = dists[..., 1:]

        sdf_diff = sdf_next - sdf_curr
        sdf_diff = sdf_diff / (dists_next + eps)

        sdf_diff = torch.clamp(sdf_diff, min=-10.0, max=10.0)

        weights = sdf_diff.sigmoid()
        weights = weights / (weights + eps)

        return weights


class SDFNetwork(nn.Module):
    """SDF (Signed Distance Function) network for DVR."""

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 4,
        positional_encoding_dims: int = 4,
    ):
        super().__init__()

        self.pe_dim = 2 * positional_encoding_dims * 3

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.pe_dim, hidden_dim))

        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Forward pass to get SDF value."""
        points_encoded = positional_encoding(points, 4)

        x = points_encoded
        for layer in self.layers:
            x = F.relu(layer(x))

        sdf = self.output_layer(x)

        return sdf


class ColorNetwork(nn.Module):
    """Color network for DVR."""

    def __init__(self, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(6, hidden_dim))

        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        viewdirs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass to get color."""
        x = points

        if normals is not None and viewdirs is not None:
            x = torch.cat([x, normals, viewdirs], dim=-1)

        for layer in self.layers:
            x = F.relu(layer(x))

        rgb = torch.sigmoid(self.output_layer(x))

        return rgb


def create_dvr(
    hidden_dim: int = 256, n_samples: int = 128, n_importance: int = 64
) -> DVRNetwork:
    """Create DVR network."""
    sdf_network = SDFNetwork(hidden_dim=hidden_dim)
    color_network = ColorNetwork(hidden_dim=hidden_dim)

    return DVRNetwork(
        sdf_network=sdf_network,
        color_network=color_network,
        n_samples=n_samples,
        n_importance=n_importance,
    )


# =============================================================================
# Factory Functions
# =============================================================================


def create_nerf(
    hidden_dim: int = 256,
    n_coarse_samples: int = 64,
    n_fine_samples: int = 128,
    near: float = 0.0,
    far: float = 4.0,
) -> NeRF:
    """Create NeRF model."""
    return NeRF(
        hidden_dim=hidden_dim,
        n_coarse_samples=n_coarse_samples,
        n_fine_samples=n_fine_samples,
        near=near,
        far=far,
    )


def create_instant_ngp(
    n_levels: int = 16,
    hidden_dim: int = 64,
    n_coarse_samples: int = 64,
    n_fine_samples: int = 128,
    near: float = 0.0,
    far: float = 4.0,
) -> InstantNGP:
    """Create Instant NGP model."""
    return InstantNGP(
        n_levels=n_levels,
        hidden_dim=hidden_dim,
        n_coarse_samples=n_coarse_samples,
        n_fine_samples=n_fine_samples,
        near=near,
        far=far,
    )


def create_glo_embeddings(
    n_codes: int, code_dim: int, init: str = "normal"
) -> GLOEmbeddings:
    """Create GLO embeddings."""
    return GLOEmbeddings(n_codes=n_codes, code_dim=code_dim, init=init)
