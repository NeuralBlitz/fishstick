from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm, Dropout, Conv2d, ConvTranspose2d
import math


class CNNDownscaler(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 128,
        scale_factor: int = 4,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.scale_factor = scale_factor
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels, 
                hidden_channels, 
                kernel_size=scale_factor, 
                stride=scale_factor
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, 1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.upsampler(x)
        x = self.decoder(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.activation(x)
        return x


class UNetDownscaler(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth
        
        self.input_conv = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i in range(depth):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            self.encoders.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ))
            self.pools.append(nn.MaxPool2d(2))
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * (2 ** depth), base_channels * (2 ** depth), 3, padding=1),
            nn.BatchNorm2d(base_channels * (2 ** depth)),
            nn.GELU(),
            ResidualBlock(base_channels * (2 ** depth)),
        )
        
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i in range(depth):
            in_ch = base_channels * (2 ** (depth - i))
            out_ch = base_channels * (2 ** (depth - i - 1))
            self.upsamples.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            self.decoders.append(nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ))
        
        self.output_conv = nn.Conv2d(base_channels, output_channels, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.input_conv(x)
        
        skips = []
        
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)
        
        x = self.bottleneck(x)
        
        for decoder, upsample, skip in zip(self.decoders, self.upsamples, reversed(skips)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        x = self.output_conv(x)
        
        return x


class DiffusionDownscaler(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 128,
        num_timesteps: int = 100,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.num_timesteps = num_timesteps
        
        self.time_embedding = nn.Sequential(
            Linear(1, hidden_channels),
            nn.GELU(),
            Linear(hidden_channels, hidden_channels),
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        
        self.mid_layers = nn.ModuleList([
            ResidualBlock(hidden_channels)
            for _ in range(4)
        ])
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, 1),
        )
    
    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        if t is None:
            t = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        
        t_emb = self.time_embedding(t)
        
        x = self.encoder(x)
        
        x = x + t_emb.view(-1, self.hidden_channels, 1, 1)
        
        for layer in self.mid_layers:
            x = layer(x)
        
        x = self.decoder(x)
        
        return x


class StatisticalDownscaling(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        scale_factor: int = 4,
        method: str = "cnn",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale_factor = scale_factor
        self.method = method
        
        if method == "cnn":
            self.model = CNNDownscaler(input_channels, output_channels, scale_factor=scale_factor)
        elif method == "unet":
            self.model = UNetDownscaler(input_channels, output_channels)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.register_buffer('mean', torch.zeros(input_channels))
        self.register_buffer('std', torch.ones(input_channels))
    
    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean.unsqueeze(-1).unsqueeze(-1)) / (
            self.std.unsqueeze(-1).unsqueeze(-1) + 1e-8
        )
    
    def denormalize(self, x: Tensor) -> Tensor:
        return x * self.std.unsqueeze(-1).unsqueeze(-1) + self.mean.unsqueeze(-1).unsqueeze(-1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.normalize(x)
        x = self.model(x)
        x = self.denormalize(x)
        return x


class DynamicalDownscaling(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 6,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Conv2d(input_channels, hidden_dim, 1)
        
        self.physics_layers = nn.ModuleList([
            Physics-informedLayer(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Conv2d(hidden_dim, output_channels, 1)
        
        self.gradient_op = self._get_gradient_operator()
    
    def _get_gradient_operator(self) -> Tensor:
        dx = torch.tensor([[[[-1, 0, 1]]]]) / 2.0
        dy = torch.tensor([[[[-1], [0], [1]]]]) / 2.0
        return dx, dy
    
    def compute_derivatives(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        dx, dy = self.gradient_op
        dx = dx.to(x.device)
        dy = dy.to(x.device)
        
        dx_conv = nn.functional.conv2d(x, dx, padding=1)
        dy_conv = nn.functional.conv2d(x, dy, padding=1)
        
        return dx_conv, dy_conv
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        
        for layer in self.physics_layers:
            x = layer(x)
        
        x = self.output_proj(x)
        
        return x


class Physics-informedLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(channels)
        
        self.physics_conv = nn.Conv2d(channels, channels, 1)
        
        self.activation = nn.GELU()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        
        x = x + self.physics_conv(x)
        
        x = self.activation(x)
        
        return x


class WeatherInterpolator(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        
        self.position_encoder = PositionEncoder(hidden_dim)
        
        self.interpolator = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        
        self.output_head = nn.Conv2d(hidden_dim, output_channels, 1)
    
    def forward(
        self,
        x_low: Tensor,
        x_high: Tensor,
        alpha: float = 0.5
    ) -> Tensor:
        feat_low = self.feature_extractor(x_low)
        feat_high = self.feature_extractor(x_high)
        
        pos_low = self.position_encoder(x_low)
        pos_high = self.position_encoder(x_high)
        
        combined = torch.cat([feat_low, feat_high, pos_low - pos_high], dim=1)
        
        x = self.interpolator(combined)
        
        x = self.output_head(x)
        
        return x


class PositionEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.conv = nn.Conv2d(2, hidden_dim, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        
        y_coords = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        x_coords = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        coords = torch.stack([y_grid, x_grid], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        
        pos_emb = self.conv(coords)
        
        return pos_emb


class SpatialWarper(nn.Module):
    def __init__(
        self,
        channels: int,
        num_control_points: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.num_control_points = num_control_points
        
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2 * num_control_points, 1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        
        offsets = self.offset_predictor(x)
        
        grid = self._generate_grid(B, H, W, x.device, x.dtype)
        
        offsets = offsets.permute(0, 2, 3, 1)
        
        warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped
    
    def _generate_grid(
        self,
        B: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tensor:
        y_coords = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        return grid


class TemporalInterpolator(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_dim: int = 64,
        num_frames: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        
        self.temporal_conv = nn.Conv2d(channels * num_frames, hidden_dim, 3, padding=1)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )
        
        self.output_conv = nn.Conv2d(hidden_dim, channels, 1)
    
    def forward(self, frames: List[Tensor]) -> Tensor:
        if len(frames) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {len(frames)}")
        
        x = torch.cat(frames, dim=1)
        
        B, C, H, W = x.shape
        
        x = self.temporal_conv(x)
        
        x_flat = x.flatten(2).transpose(1, 2)
        
        x_trans = self.transformer(x_flat)
        
        x = x_trans.transpose(1, 2).view(B, -1, H, W)
        
        x = self.output_conv(x)
        
        return x


__all__ = [
    "StatisticalDownscaling",
    "DynamicalDownscaling",
    "WeatherInterpolator",
    "CNNDownscaler",
    "UNetDownscaler",
    "DiffusionDownscaler",
]
