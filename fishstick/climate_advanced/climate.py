from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm, Dropout, MultiheadAttention, Conv2d
import math


class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.qkv = Linear(hidden_dim, hidden_dim * 3)
        self.proj = Linear(hidden_dim, hidden_dim)

        self.relative_bias = Parameter(torch.zeros(num_heads, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        spatial_size = H * W
        row_ids = torch.arange(spatial_size, device=x.device) // W
        col_ids = torch.arange(spatial_size, device=x.device) % W
        row_offset = row_ids[:, None] - row_ids[None, :]
        col_offset = col_ids[:, None] - col_ids[None, :]

        attn = attn + self.relative_bias[:, :, :, 0]

        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)

        return x


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, num_timesteps: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_timesteps = num_timesteps
        self.head_dim = hidden_dim // num_heads

        self.qkv = Linear(hidden_dim, hidden_dim * 3)
        self.proj = Linear(hidden_dim, hidden_dim)

        self.temporal_embedding = nn.Parameter(
            torch.randn(1, num_timesteps, hidden_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, H, W, C = x.shape

        x = x + self.temporal_embedding[:, :T, :]

        x_flat = x.flatten(1, 3)

        qkv = (
            self.qkv(x_flat)
            .reshape(B, T * H * W, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, T, H, W, C)
        x = self.proj(x.flatten(2, 4)).reshape(B, T, H, W, C)

        return x


class ClimateEmbedding(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embedding_dim: int = 256,
        temperature_dim: int = 12,
        spatial_resolution: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.temperature_dim = temperature_dim

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, embedding_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.GELU(),
            nn.Conv2d(embedding_dim // 2, embedding_dim, 3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
        )

        self.temporal_encoder = nn.Sequential(
            Linear(1, temperature_dim),
            nn.GELU(),
            Linear(temperature_dim, temperature_dim),
        )

        self.climate_embedding = nn.Parameter(torch.randn(1, embedding_dim))

    def forward(self, x: Tensor, time_info: Optional[Tensor] = None) -> Tensor:
        spatial_emb = self.spatial_encoder(x)

        if time_info is not None:
            time_emb = self.temporal_encoder(time_info.unsqueeze(-1))
            time_emb = (
                time_emb.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, spatial_emb.size(-2), spatial_emb.size(-1))
            )
            combined = spatial_emb + time_emb
        else:
            combined = spatial_emb

        combined = combined + self.climate_embedding

        return combined


class ClimateEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = ClimateEmbedding(input_channels, hidden_dim)

        self.spatial_attn = nn.ModuleList(
            [SpatialAttention(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        self.temporal_attn = TemporalAttention(hidden_dim, num_heads)

        self.ffn = nn.Sequential(
            Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            Linear(hidden_dim * 4, hidden_dim),
        )

        self.norms = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, x: Tensor, time_info: Optional[Tensor] = None) -> Tensor:
        x = self.embedding(x, time_info)

        B, H, W, C = x.shape

        for i, (attn, norm) in enumerate(zip(self.spatial_attn, self.norms)):
            x = norm(x + attn(x))

            x_flat = x.flatten(1, 2)
            x_flat = x_flat + self.ffn(x_flat)
            x = x_flat.view(B, H, W, C)

        return x


class ClimateRegressor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            layers.append(Linear(in_dim, out_dim))

            if i < num_layers - 1:
                layers.append(nn.GELU())

        self.predictor = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape

        x = x.permute(0, 3, 1, 2)

        x = self.pool(x).flatten(1)

        return self.predictor(x)


class ExtremeEventDetector(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 128,
        num_event_types: int = 5,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_event_types = num_event_types
        self.threshold = threshold

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(hidden_dim, num_event_types, 1)

        self.severity_regressor = nn.Conv2d(hidden_dim, 1, 1)

        self.spatial_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.temporal_encoder = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self, x: Tensor, return_heatmaps: bool = False
    ) -> Tuple[Tensor, Tensor]:
        features = self.feature_extractor(x)

        event_logits = self.classifier(features)

        severity = torch.sigmoid(self.severity_regressor(features))

        spatial_features = self.spatial_pool(features).flatten(1)

        temporal_input = spatial_features.unsqueeze(1)
        temporal_out, _ = self.temporal_encoder(temporal_input)

        confidence = torch.sigmoid(temporal_out.squeeze(1))

        if return_heatmaps:
            return event_logits, severity, confidence, features
        else:
            return event_logits, severity, confidence

    def detect_events(self, x: Tensor) -> List[dict]:
        event_logits, severity, confidence = self.forward(x)

        batch_size = x.size(0)

        events = []

        event_probs = torch.sigmoid(event_logits)

        for b in range(batch_size):
            batch_events = []

            for event_type in range(self.num_event_types):
                event_map = event_probs[b, event_type]

                above_threshold = event_map > self.threshold

                if above_threshold.any():
                    max_severity = severity[b][above_threshold].max().item()
                    mean_confidence = confidence[b, event_type].item()

                    batch_events.append(
                        {
                            "event_type": event_type,
                            "max_severity": max_severity,
                            "confidence": mean_confidence,
                            "locations": above_threshold.sum().item(),
                        }
                    )

            events.append(batch_events)

        return events


class ClimateTimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = Linear(input_dim, hidden_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        x = self.transformer(x)
        return x


class ClimateAnomalyDetector(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        self.mu_head = nn.Conv2d(hidden_dim, latent_dim, 1)
        self.logvar_head = nn.Conv2d(hidden_dim, latent_dim, 1)

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, input_channels, 1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.encoder(x)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar

    def compute_anomaly_score(self, x: Tensor) -> Tensor:
        reconstruction, _, _ = self.forward(x)

        anomaly_score = F.mse_loss(reconstruction, x, reduction="none").mean(dim=1)

        return anomaly_score


__all__ = [
    "ClimateEmbedding",
    "ExtremeEventDetector",
    "ClimateEncoder",
    "ClimateRegressor",
    "SpatialAttention",
    "TemporalAttention",
]
