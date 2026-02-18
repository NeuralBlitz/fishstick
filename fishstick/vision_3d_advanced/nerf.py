import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass


@dataclass
class CameraPose:
    position: torch.Tensor
    rotation: torch.Tensor

    @staticmethod
    def from_angle_distance(
        target: torch.Tensor,
        distance: float,
        azimuth: float,
        elevation: float,
    ) -> "CameraPose":
        x = distance * math.cos(elevation) * math.cos(azimuth)
        y = distance * math.cos(elevation) * math.sin(azimuth)
        z = distance * math.sin(elevation)
        position = target + torch.tensor([x, y, z])

        forward = -(position - target) / torch.norm(position - target)
        up = torch.tensor([0.0, 0.0, 1.0])
        right = torch.cross(forward, up)
        right = right / (torch.norm(right) + 1e-8)
        up = torch.cross(right, forward)

        rotation = torch.stack([right, up, forward], dim=0)
        return CameraPose(position=position, rotation=rotation)


def get_camera_rays(
    height: int,
    width: int,
    focal: float,
    camera_pose: CameraPose,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )

    x = (x - width / 2) / focal
    y = -(y - height / 2) / focal
    z = -torch.ones_like(x)

    directions = torch.stack([x, y, z], dim=-1)

    rotation = camera_pose.rotation.to(device)
    rays_d = torch.einsum("ij,...j->...i", rotation, directions)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    rays_o = camera_pose.position.to(device).reshape(1, 3).expand(height * width, -1)

    return rays_o, rays_d


def get_camera_rays_batch(
    batch_size: int,
    height: int,
    width: int,
    focal: float,
    camera_poses: List[CameraPose],
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    rays_o_list = []
    rays_d_list = []

    for camera_pose in camera_poses:
        rays_o, rays_d = get_camera_rays(height, width, focal, camera_pose, device)
        rays_o_list.append(rays_o)
        rays_d_list.append(rays_d)

    rays_o = torch.stack(rays_o_list, dim=0)
    rays_d = torch.stack(rays_d_list, dim=0)

    return rays_o, rays_d


class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.freq_bands = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = []
        if self.include_input:
            encoded.append(x)

        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * math.pi * x))
            encoded.append(torch.cos(freq * math.pi * x))

        return torch.cat(encoded, dim=-1)


class NeRF(nn.Module):
    def __init__(
        self,
        num_frequencies_pos: int = 10,
        num_frequencies_dir: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connections: List[int] = [4],
    ):
        super().__init__()
        self.num_frequencies_pos = num_frequencies_pos
        self.num_frequencies_dir = num_frequencies_dir
        self.skip_connections = skip_connections

        self.pos_encoder = PositionalEncoding(num_frequencies_pos, include_input=True)
        self.dir_encoder = PositionalEncoding(num_frequencies_dir, include_input=True)

        pos_dim = 3 * (num_frequencies_pos * 2 + 1)
        dir_dim = 3 * (num_frequencies_dir * 2 + 1)

        self.layers = nn.ModuleList()
        in_dim = pos_dim

        for i in range(num_layers):
            if i in skip_connections:
                in_dim = pos_dim + hidden_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.layers.append(nn.Linear(hidden_dim, hidden_dim + 1))

    def forward(
        self,
        positions: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_pos = self.pos_encoder(positions)

        x = encoded_pos
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            if i in self.skip_connections:
                x = torch.cat([x, encoded_pos], dim=-1)

        x = self.layers[-1](x)
        density = x[:, 0].unsqueeze(-1)
        feature = x[:, 1:]

        if directions is not None:
            encoded_dir = self.dir_encoder(directions)
            x = torch.cat([feature, encoded_dir], dim=-1)
            x = F.relu(
                nn.Linear(encoded_dir.shape[-1] + feature.shape[-1], 128).to(x.device)(
                    x
                )
            )
            rgb = torch.sigmoid(nn.Linear(128, 3).to(x.device)(x))
        else:
            rgb = torch.sigmoid(feature[:, :3])

        return rgb, density


class ConditionalNeRF(nn.Module):
    def __init__(
        self,
        num_frequencies_pos: int = 10,
        num_frequencies_dir: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_embeddings: int = 256,
    ):
        super().__init__()

        self.pos_encoder = PositionalEncoding(num_frequencies_pos, include_input=True)
        self.dir_encoder = PositionalEncoding(num_frequencies_dir, include_input=True)

        self.embedding = nn.Embedding(num_embeddings, hidden_dim)

        pos_dim = 3 * (num_frequencies_pos * 2 + 1)
        dir_dim = 3 * (num_frequencies_dir * 2 + 1)

        self.layers = nn.ModuleList()
        in_dim = pos_dim + hidden_dim

        for _ in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.density_head = nn.Linear(hidden_dim, 1)
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_pos = self.pos_encoder(positions)
        encoded_dir = self.dir_encoder(directions)

        embedded = self.embedding(condition)
        embedded = embedded.expand_as(encoded_pos)

        x = torch.cat([encoded_pos, embedded], dim=-1)

        for layer in self.layers:
            x = F.relu(layer(x))

        density = self.density_head(x)
        rgb = self.rgb_head(torch.cat([x, encoded_dir], dim=-1))

        return rgb, density


def volumetric_rendering(
    rgb: torch.Tensor,
    density: torch.Tensor,
    distances: torch.Tensor,
    background_color: torch.Tensor = torch.tensor([1.0, 1.0, 1.0]),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-10

    delta = distances[..., 1:] - distances[..., :-1]
    delta = torch.cat([delta, torch.ones_like(delta[..., :1]) * 1e10], dim=-1)

    alpha = 1.0 - torch.exp(-density[..., 0] * delta)

    weights = (
        alpha
        * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + eps], dim=-1),
            dim=-1,
        )[..., :-1]
    )

    rgb_weights = weights.unsqueeze(-1) * rgb
    accumulated_rgb = rgb_weights.sum(dim=-3) + background_color.to(rgb.device) * (
        1.0 - weights.sum(dim=-3)
    )

    accumulated_alpha = weights.sum(dim=-3)

    depth = (weights * distances).sum(dim=-3)

    return accumulated_rgb, accumulated_alpha, depth


def ray_marching(
    model: nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int = 64,
    encode_directions: bool = True,
    randomized: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = rays_o.device
    batch_size, num_rays = rays_o.shape[:2]

    if randomized:
        t_vals = torch.linspace(near, far, num_samples, device=device)
        t_vals = t_vals.unsqueeze(0).expand(batch_size, num_rays, -1)
        noise = torch.rand_like(t_vals) * (far - near) / num_samples
        t_vals = t_vals + noise
    else:
        t_vals = torch.linspace(near, far, num_samples, device=device)
        t_vals = t_vals.unsqueeze(0).expand(batch_size, num_rays, -1)

    points = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * t_vals.unsqueeze(-1)

    points_flat = points.view(-1, 3)
    directions_flat = rays_d.unsqueeze(2).expand(-1, -1, num_samples, -1).reshape(-1, 3)

    if encode_directions:
        rgb, density = model(points_flat, directions_flat)
    else:
        rgb, density = model(points_flat, None)

    rgb = rgb.view(batch_size, num_rays, num_samples, 3)
    density = density.view(batch_size, num_rays, num_samples, 1)

    rgb, accumulated_alpha, depth = volumetric_rendering(rgb, density, t_vals)

    return rgb, accumulated_alpha, depth


class HierarchicalSampling(nn.Module):
    def __init__(self, num_coarse: int = 64, num_fine: int = 128):
        super().__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine

    def forward(
        self,
        model: nn.Module,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        rgb_coarse: torch.Tensor,
        density_coarse: torch.Tensor,
        t_vals_coarse: torch.Tensor,
        near: float,
        far: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_rays = rays_o.shape[:2]

        weights = density_coarse[..., 0] * (
            t_vals_coarse[..., 1:] - t_vals_coarse[..., :-1]
        )
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        pdf = weights + 1e-5
        pdf = pdf / pdf.sum(dim=-1, keepdim=True)

        t_vals_fine = self.sample_pdf(t_vals_coarse[:, :, :-1], pdf, self.num_fine)

        t_vals_fine, _ = torch.sort(
            torch.cat([t_vals_coarse, t_vals_fine], dim=-1), dim=-1
        )

        t_vals_fine = t_vals_fine.detach()

        points_fine = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * t_vals_fine.unsqueeze(
            -1
        )
        points_flat = points_fine.view(-1, 3)
        directions_flat = (
            rays_d.unsqueeze(2)
            .expand(-1, -1, self.num_coarse + self.num_fine, -1)
            .reshape(-1, 3)
        )

        rgb_fine, density_fine = model(points_flat, directions_flat)
        rgb_fine = rgb_fine.view(batch_size, num_rays, -1, 3)
        density_fine = density_fine.view(batch_size, num_rays, -1, 1)

        rgb_final, accumulated_alpha, depth = volumetric_rendering(
            rgb_fine, density_fine, t_vals_fine
        )

        return rgb_final, accumulated_alpha, depth

    def sample_pdf(
        self, bins: torch.Tensor, weights: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        batch_size, num_rays, num_bins = weights.shape

        pdf = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        u = torch.rand(batch_size, num_rays, num_samples, device=weights.device)
        u = u.contiguous()

        indices = torch.searchsorted(cdf, u, right=True)

        below = torch.clamp(indices - 1, min=0)
        above = torch.clamp(indices, max=num_bins)

        cdf_below = torch.gather(cdf, -1, below)
        cdf_above = torch.gather(cdf, -1, above)

        denom = cdf_above - cdf_below
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

        t = (u - cdf_below) / denom
        samples = bins[..., :-1] + t * (bins[..., 1:] - bins[..., :-1])

        return samples


class NeRFSystem(nn.Module):
    def __init__(
        self,
        num_frequencies_pos: int = 10,
        num_frequencies_dir: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_coarse: int = 64,
        num_fine: int = 128,
    ):
        super().__init__()
        self.coarse_model = NeRF(
            num_frequencies_pos, num_frequencies_dir, hidden_dim, num_layers
        )
        self.fine_model = NeRF(
            num_frequencies_pos, num_frequencies_dir, hidden_dim, num_layers
        )
        self.hierarchical_sampler = HierarchicalSampling(num_coarse, num_fine)

    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float = 0.0,
        far: float = 1.0,
        train_fine: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_coarse, alpha_coarse, depth_coarse = ray_marching(
            self.coarse_model, rays_o, rays_d, near, far
        )

        if not train_fine:
            return rgb_coarse, alpha_coarse, depth_coarse, None, None

        with torch.no_grad():
            t_vals = torch.linspace(near, far, 64, device=rays_o.device)
            t_vals = t_vals.unsqueeze(0).expand(rays_o.shape[0], rays_o.shape[1], -1)

        rgb_fine, alpha_fine, depth_fine = self.hierarchical_sampler(
            self.fine_model,
            rays_o,
            rays_d,
            rgb_coarse,
            alpha_coarse.unsqueeze(-1),
            t_vals,
            near,
            far,
        )

        return rgb_coarse, alpha_coarse, depth_coarse, rgb_fine, depth_fine

    def render_image(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float = 0.0,
        far: float = 1.0,
        chunk_size: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_rays = rays_o.shape[1]
        rgb_list = []
        depth_list = []
        alpha_list = []

        for i in range(0, num_rays, chunk_size):
            chunk_rays_o = rays_o[:, i : i + chunk_size]
            chunk_rays_d = rays_d[:, i : i + chunk_size]

            rgb, alpha, depth = self.forward(chunk_rays_o, chunk_rays_d, near, far)

            rgb_list.append(rgb)
            depth_list.append(depth)
            alpha_list.append(alpha)

        rgb = torch.cat(rgb_list, dim=1)
        depth = torch.cat(depth_list, dim=1)
        alpha = torch.cat(alpha_list, dim=1)

        return rgb, alpha, depth


def create_camera_path(
    num_frames: int,
    target: torch.Tensor,
    radius: float,
    height: float = 0.0,
) -> List[CameraPose]:
    camera_poses = []
    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        pose = CameraPose.from_angle_distance(target, radius, angle, height)
        camera_poses.append(pose)
    return camera_poses
