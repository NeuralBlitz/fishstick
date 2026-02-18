import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = []
        if self.include_input:
            encoding.append(x)
        for freq in self.freq_bands:
            encoding.append(torch.sin(freq * x))
            encoding.append(torch.cos(freq * x))
        return torch.cat(encoding, dim=-1)


class NeRF(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_frequencies: int = 10,
        use_viewdirs: bool = True,
    ):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.positional_encoding = PositionalEncoding(num_frequencies)
        encoded_dim = self.positional_encoding(input_dim).shape[-1]

        if use_viewdirs:
            viewdirs_dim = self.positional_encoding(3).shape[-1]

        layers = []
        for i in range(num_layers):
            if i == 0:
                in_dim = encoded_dim
            elif i == 4 and use_viewdirs:
                in_dim = hidden_dim + viewdirs_dim
            else:
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.feature_layers = nn.ModuleList(layers)

        self.rgb_layer = nn.Linear(hidden_dim, 3)
        self.sigma_layer = nn.Linear(hidden_dim, 1)

        if use_viewdirs:
            self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
            self.viewdirs_layer = nn.Linear(hidden_dim + viewdirs_dim, hidden_dim // 2)

    def forward(
        self, x: torch.Tensor, viewdirs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.positional_encoding(x)

        x = encoded
        for i, layer in enumerate(self.feature_layers):
            if i == 4 and self.use_viewdirs and viewdirs is not None:
                encoded_viewdirs = self.positional_encoding(viewdirs)
                x = torch.cat([self.feature_linear(x), encoded_viewdirs], dim=-1)
            x = layer(x)

        sigma = self.sigma_layer(x)
        sigma = F.relu(sigma)

        if self.use_viewdirs and viewdirs is not None:
            encoded_viewdirs = self.positional_encoding(viewdirs)
            x = self.feature_linear(x)
            x = torch.cat([x, encoded_viewdirs], dim=-1)
            x = self.viewdirs_layer(x)
            x = F.relu(x)
            rgb = self.rgb_layer(x)
            rgb = torch.sigmoid(rgb)
        else:
            rgb = self.rgb_layer(x)
            rgb = torch.sigmoid(rgb)

        return rgb, sigma.squeeze(-1)


class MipNeRF(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_frequencies: int = 4,
        use_viewdirs: bool = True,
    ):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.num_frequencies = num_frequencies

        self.input_dim = input_dim
        self.encode_principal_dim = input_dim * 2 * num_frequencies + input_dim

        layers = []
        in_dim = self.encode_principal_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.rgb_head = nn.Linear(hidden_dim, 3)

        if use_viewdirs:
            self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
            viewdirs_dim = 3 * 2 * num_frequencies + 3
            self.viewdirs_layers = nn.Sequential(
                nn.Linear(hidden_dim + viewdirs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            self.viewdirs_rgb = nn.Linear(hidden_dim // 2, 3)

    def integrate_gaussian(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return self._integrate_gaussian(mean, var, self.input_dim, self.num_frequencies)

    def _integrate_gaussian(
        self, mean: torch.Tensor, var: torch.Tensor, dim: int, num_freqs: int
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Mip-NeRF integration requires careful implementation"
        )

    def forward(
        self, x: torch.Tensor, viewdirs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x)

        sigma = F.relu(self.sigma_head(x))

        if self.use_viewdirs and viewdirs is not None:
            features = self.feature_linear(x)
            x = torch.cat([features, viewdirs], dim=-1)
            x = self.viewdirs_layers(x)
            rgb = torch.sigmoid(self.viewdirs_rgb(x))
        else:
            rgb = torch.sigmoid(self.rgb_head(x))

        return rgb, sigma.squeeze(-1)


class RefNeRF(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_frequencies: int = 4,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.input_dim = input_dim

        self.sh_degree = 0
        self.sh_dim = (self.sh_degree + 1) ** 2

        layers = []
        in_dim = input_dim * 2 * num_frequencies + input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

        self.sigma_head = nn.Linear(hidden_dim, 1)

        self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
        self.normal_layers = nn.Sequential(
            nn.Linear(hidden_dim + 3 * 2 * num_frequencies + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.reflection_rgb = nn.Linear(hidden_dim // 2, self.sh_dim * 3)

        self.diffuse_rgb = nn.Linear(hidden_dim, 3)

    def spherical_harmonics(
        self, directions: torch.Tensor, degree: int
    ) -> torch.Tensor:
        return torch.cat([directions.new_ones(directions.shape[:-1] + (1,))], dim=-1)

    def forward(
        self, x: torch.Tensor, viewdirs: torch.Tensor, normals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x)

        sigma = F.relu(self.sigma_head(x))

        diffuse_features = self.feature_linear(x)
        diffuse_rgb = torch.sigmoid(self.diffuse_rgb(diffuse_features))

        normal_encoded = self._encode_direction(normals)
        reflection_input = torch.cat([diffuse_features, normal_encoded], dim=-1)
        reflection_features = self.normal_layers(reflection_input)

        reflection_rgb = self.reflection_rgb(reflection_features)
        reflection_rgb = self.spherical_harmonics(viewdirs, self.sh_degree)
        reflection_rgb = torch.sigmoid(reflection_rgb)

        rgb = diffuse_rgb + reflection_rgb

        return rgb, sigma.squeeze(-1)

    def _encode_direction(self, x: torch.Tensor) -> torch.Tensor:
        return x


def generate_camera_rays(
    camera_to_world: torch.Tensor,
    intrinsic: torch.Tensor,
    width: int,
    height: int,
    near: float = 0.1,
    far: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = camera_to_world.shape[0]
    device = camera_to_world.device

    u = torch.linspace(0, width - 1, width, device=device)
    v = torch.linspace(0, height - 1, height, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")

    u = u.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size, -1)
    v = v.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size, -1)

    cx = intrinsic[:, 0, 2:3]
    cy = intrinsic[:, 1, 2:3]
    fx = intrinsic[:, 0, 0:1]
    fy = intrinsic[:, 1, 1:2]

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = torch.ones_like(x)

    directions = torch.stack([x, y, z], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = torch.einsum(
        "bij,bkj->bik", directions[:, :, :3], camera_to_world[:, :3, :3]
    )
    rays_o = camera_to_world[:, :3, 3].unsqueeze(1).expand(-1, rays_d.shape[1], -1)

    return rays_o, rays_d


def ray_marching(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    model: nn.Module,
    near: float = 0.1,
    far: float = 100.0,
    num_samples: int = 128,
    use_viewdirs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_rays, _ = ray_origins.shape
    device = ray_origins.device

    t_vals = torch.linspace(near, far, num_samples, device=device)
    t_vals = t_vals.unsqueeze(0).unsqueeze(0).expand(batch_size, num_rays, -1)

    noise = (torch.rand_like(t_vals) - 0.5) * (far - near) / num_samples
    t_vals = t_vals + noise

    points = ray_origins.unsqueeze(2) + t_vals.unsqueeze(3) * ray_directions.unsqueeze(
        2
    )
    points = points.reshape(batch_size, num_rays * num_samples, 3)

    viewdirs = ray_directions.unsqueeze(2).expand(-1, -1, num_samples, -1)
    viewdirs = viewdirs.reshape(batch_size, num_rays * num_samples, 3)

    rgb, sigma = model(points, viewdirs if use_viewdirs else None)

    rgb = rgb.reshape(batch_size, num_rays, num_samples, 3)
    sigma = sigma.reshape(batch_size, num_rays, num_samples)

    return rgb, sigma, t_vals


def volumetric_rendering(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    t_vals: torch.Tensor,
    white_background: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = t_vals[:, :, 1:] - t_vals[:, :, :-1]
    delta = torch.cat([delta, torch.ones_like(delta[:, :, :1]) * 1e10], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * delta)

    transmittance = torch.cumprod(
        torch.cat(
            [torch.ones_like(alpha[:, :, :1]), 1.0 - alpha[:, :, :-1] + 1e-10], dim=-1
        ),
        dim=-1,
    )

    weights = alpha * transmittance

    rgb_rendered = (weights.unsqueeze(-1) * rgb).sum(dim=2)
    depth = (weights * t_vals).sum(dim=2)

    acc = weights.sum(dim=-1)

    if white_background:
        rgb_rendered = rgb_rendered + (1.0 - acc.unsqueeze(-1))

    return rgb_rendered, depth
