"""
NeRF (Neural Radiance Fields) Core Module

Implementation of Neural Radiance Fields for novel view synthesis.
"""

from typing import Tuple, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from fishstick.vision_3d.positional_encoding import PositionalEncoder


class NeRF(nn.Module):
    """
    Neural Radiance Field (NeRF) network.

    MLP that takes 3D position and view direction to predict:
    - RGB color (density-independent)
    - Volume density (sigma)
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 4,
        num_layers: int = 8,
        view_dir_dim: int = 3,
        use_view_dirs: bool = True,
        skip_connections: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_view_dirs = use_view_dirs
        self.skip_connections = skip_connections

        self.layers = nn.ModuleList()
        self.layer_viewdirs = nn.ModuleList()

        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim

            if i == skip_connections:
                in_dim = input_dim + hidden_dim

            self.layers.append(nn.Linear(in_dim, out_dim))

            in_dim = out_dim

        if use_view_dirs:
            self.layer_viewdirs.append(
                nn.Linear(hidden_dim + view_dir_dim, hidden_dim // 2)
            )

        self.layer_rgb = nn.Linear(hidden_dim // 2 if use_view_dirs else hidden_dim, 3)
        self.layer_sigma = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: Tensor,
        view_dirs: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Input positions [N, input_dim]
            view_dirs: View directions [N, view_dir_dim]

        Returns:
            rgb: RGB color [N, 3]
            sigma: Volume density [N, 1]
        """
        original_x = x
        x = original_x

        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)

            if i == self.skip_connections:
                x = torch.cat([original_x, x], dim=-1)

        if self.use_view_dirs and view_dirs is not None:
            x_view = torch.cat([x, view_dirs], dim=-1)
            x_view = self.layer_viewdirs[0](x_view)
            x_view = F.relu(x_view)
            rgb = self.layer_rgb(x_view)
        else:
            rgb = self.layer_rgb(x)

        sigma = self.layer_sigma(x)
        sigma = F.relu(sigma)

        return rgb, sigma


class NerfModel(nn.Module):
    """
    Complete NeRF model with optional positional encoding.
    """

    def __init__(
        self,
        pos_encoder: Optional[nn.Module] = None,
        dir_encoder: Optional[nn.Module] = None,
        hidden_dim: int = 256,
        num_layers: int = 8,
    ):
        super().__init__()

        self.pos_encoder = pos_encoder
        self.dir_encoder = dir_encoder

        pos_dim = pos_encoder.output_dim if pos_encoder else 3
        dir_dim = dir_encoder.output_dim if dir_encoder else 3

        self.nerf = NeRF(
            input_dim=pos_dim,
            hidden_dim=hidden_dim,
            view_dir_dim=dir_dim,
            num_layers=num_layers,
        )

    def forward(
        self,
        positions: Tensor,
        view_dirs: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            positions: 3D positions [N, 3]
            view_dirs: View directions [N, 3]

        Returns:
            rgb: RGB color [N, 3]
            sigma: Density [N, 1]
        """
        if self.pos_encoder is not None:
            positions = self.pos_encoder(positions)

        view_dirs_encoded = None
        if view_dirs is not None and self.dir_encoder is not None:
            view_dirs_encoded = self.dir_encoder(view_dirs)

        return self.nerf(positions, view_dirs_encoded)


class NeRFRenderer(nn.Module):
    """
    NeRF volumetric rendering.
    """

    def __init__(
        self,
        num_samples: int = 64,
        near: float = 2.0,
        far: float = 6.0,
        use_hierarchical: bool = True,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.near = near
        self.far = far
        self.use_hierarchical = use_hierarchical

    def render_rays(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        model: nn.Module,
    ) -> Tuple[Tensor, Tensor]:
        """
        Render rays through the scene.

        Args:
            rays_o: Ray origins [B, 3]
            rays_d: Ray directions [B, 3]
            model: NeRF model

        Returns:
            rgb: RGB colors [B, 3]
            depth: Depth values [B]
        """
        B = rays_o.shape[0]

        t_vals = torch.linspace(
            self.near, self.far, self.num_samples, device=rays_o.device
        )
        t_vals = t_vals.unsqueeze(0).expand(B, -1)

        points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(-1)

        view_dirs = rays_d.unsqueeze(1).expand(-1, self.num_samples, -1)
        view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-8)

        points_flat = points.reshape(-1, 3)
        view_dirs_flat = view_dirs.reshape(-1, 3)

        rgb, sigma = model(points_flat, view_dirs_flat)

        rgb = rgb.reshape(B, self.num_samples, 3)
        sigma = sigma.reshape(B, self.num_samples)

        sigma = 1 - torch.exp(-sigma)

        alpha = sigma
        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [torch.ones(B, 1, device=rays_o.device), 1 - alpha + 1e-10], dim=1
                ),
                dim=1,
            )[:, :-1]
        )

        rgb_map = (weights.unsqueeze(-1) * rgb).sum(dim=1)
        depth_map = (weights * t_vals).sum(dim=1)

        return rgb_map, depth_map


class VolumetricRenderer(nn.Module):
    """
    General volumetric rendering for NeRF.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        rgb: Tensor,
        sigma: Tensor,
        t_vals: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Volumetric rendering equation.

        Args:
            rgb: RGB values at samples [B, N, 3]
            sigma: Density at samples [B, N]
            t_vals: Distance values [B, N]

        Returns:
            rgb_map: Integrated RGB [B, 3]
            depth_map: Integrated depth [B]
        """
        B, N = sigma.shape

        delta = t_vals[:, 1:] - t_vals[:, :-1]
        delta = torch.cat([delta, torch.ones(B, 1, device=sigma.device) * 1e10], dim=1)

        alpha = 1 - torch.exp(-sigma * delta)

        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [torch.ones(B, 1, device=sigma.device), 1 - alpha + 1e-10], dim=1
                ),
                dim=1,
            )[:, :-1]
        )

        rgb_map = (weights.unsqueeze(-1) * rgb).sum(dim=1)
        depth_map = (weights * t_vals).sum(dim=1)

        return rgb_map, depth_map


def hierarchical_sampling(
    rays_o: Tensor,
    rays_d: Tensor,
    weights: Tensor,
    num_samples: int,
    perturb: bool = True,
) -> Tensor:
    """
    Hierarchical sampling for NeRF (proposal sampling).

    Args:
        rays_o: Ray origins [B, 3]
        rays_d: Ray directions [B, 3]
        weights: Weights from coarse network [B, N]
        num_samples: Number of fine samples
        perturb: Whether to add noise

    Returns:
        t_vals: Sampled distance values [B, num_samples]
    """
    B, N = weights.shape

    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros(B, 1, device=weights.device), cdf], dim=1)

    u = torch.rand(B, num_samples, device=weights.device)
    if perturb:
        u = u + torch.rand_like(u) / num_samples

    t_vals = torch.zeros(B, num_samples, device=weights.device)

    for b in range(B):
        t_vals[b] = torch.searchsorted(cdf[b], u[b])

    t_vals = t_vals.clamp(0, N - 1)

    t_vals = torch.sort(t_vals, dim=-1)[0]

    return t_vals
