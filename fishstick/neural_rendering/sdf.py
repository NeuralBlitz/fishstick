import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class SDFNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_frequencies: int = 10,
        output_dim: int = 1,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies

        self.encoding = PositionalEncodingSDF(num_frequencies)
        encoding_dim = self.encoding(input_dim)

        layers = []
        in_dim = encoding_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_linear(x)


class PositionalEncodingSDF(nn.Module):
    def __init__(self, num_frequencies: int):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = [x]
        for freq in self.freq_bands:
            encoding.append(torch.sin(freq * x))
            encoding.append(torch.cos(freq * x))
        return torch.cat(encoding, dim=-1)


class NeuralSDF(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_frequencies: int = 10,
    ):
        super().__init__()

        self.sdf_network = SDFNetwork(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_frequencies=num_frequencies,
            output_dim=1,
        )

        self.feature_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sdf = self.sdf_network(x)

        x_features = x[:, :3]
        for i, layer in enumerate(self.sdf_network.layers):
            if i < len(self.sdf_network.layers):
                x_features = layer(x_features)
                if i == len(self.sdf_network.layers) - 1:
                    features = self.feature_network(x_features)

        return sdf.squeeze(-1), features

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.sdf_network(x).squeeze(-1)

    def get_normal(self, x: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
        with torch.enable_grad():
            x.requires_grad_(True)
            sdf_values = self.sdf(x)
            normal = torch.autograd.grad(
                sdf_values,
                x,
                grad_outputs=torch.ones_like(sdf_values),
                create_graph=True,
                retain_graph=True,
            )[0]
        return F.normalize(normal, dim=-1)


def sdf_to_mesh(
    model: NeuralSDF,
    resolution: int = 128,
    bounds: float = 1.0,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-bounds, bounds, resolution)
    y = torch.linspace(-bounds, bounds, resolution)
    z = torch.linspace(-bounds, bounds, resolution)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    sdf_values = model.sdf(points)
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)

    vertices, faces = marching_cubes(sdf_grid, threshold)

    return vertices, faces


def marching_cubes(
    volume: torch.Tensor,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    volume_np = volume.detach().cpu().numpy()

    try:
        from skimage import measure

        vertices, faces, normals, values = measure.marching_cubes(
            volume_np, level=threshold, spacing=(1.0, 1.0, 1.0)
        )
        vertices = torch.from_numpy(vertices).float()
        faces = torch.from_numpy(faces).astype(torch.int64)
    except ImportError:
        vertices, faces = marching_cubes_numpy(volume_np, threshold)
        vertices = torch.from_numpy(vertices).float()
        faces = torch.from_numpy(faces).astype(torch.int64)

    return vertices, faces


def marching_cubes_numpy(
    volume: np.ndarray,
    threshold: float,
    level: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy import ndimage

    if volume.max() < threshold and volume.min() > threshold:
        return np.array([]), np.array([])

    cube_vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )

    edge_table = np.array(
        [
            0x0,
            0x109,
            0x203,
            0x30A,
            0x406,
            0x50F,
            0x605,
            0x70C,
            0x80C,
            0x905,
            0xA0F,
            0xB06,
            0xC0A,
            0xD03,
            0xE09,
            0xF00,
        ]
    )

    vertices = []
    faces = []

    return np.array(vertices), np.array(faces)


def compute_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    vertex_normals = torch.zeros_like(vertices)

    face_vertices = vertices[faces]
    v0 = face_vertices[:, 0]
    v1 = face_vertices[:, 1]
    v2 = face_vertices[:, 2]

    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = F.normalize(face_normals, dim=1)

    vertex_normals.index_add_(0, faces[:, 0], face_normals)
    vertex_normals.index_add_(0, faces[:, 1], face_normals)
    vertex_normals.index_add_(0, faces[:, 2], face_normals)

    vertex_normals = F.normalize(vertex_normals, dim=1)

    return vertex_normals


class SDFRenderer(nn.Module):
    def __init__(
        self,
        sdf_network: NeuralSDF,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.sdf_network = sdf_network

        self.color_network = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def render_ray(
        self,
        ray_origin: torch.Tensor,
        ray_direction: torch.Tensor,
        num_steps: int = 128,
        max_distance: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = ray_origin.device
        batch_size = ray_origin.shape[0]

        t_vals = torch.linspace(0, max_distance, num_steps, device=device)
        t_vals = t_vals.unsqueeze(0).expand(batch_size, -1)

        points = ray_origin.unsqueeze(1) + t_vals.unsqueeze(
            2
        ) * ray_direction.unsqueeze(1)

        sdf_values = self.sdf_network.sdf(points)

        hit_mask = sdf_values < 0
        first_hit = hit_mask.float().argmax(dim=1)

        colors = torch.zeros(batch_size, 3, device=device)
        depths = torch.ones(batch_size, device=device) * max_distance

        for i in range(batch_size):
            hit_idx = first_hit[i].item()
            if hit_mask[i, hit_idx]:
                hit_point = points[i, hit_idx]
                normal = self.sdf_network.get_normal(hit_point.unsqueeze(0))
                sdf_feat, _ = self.sdf_network(hit_point.unsqueeze(0))
                color_input = torch.cat([sdf_feat, ray_direction[i]], dim=-1)
                colors[i] = self.color_network(color_input).squeeze(0)
                depths[i] = t_vals[i, hit_idx]

        return colors, depths
