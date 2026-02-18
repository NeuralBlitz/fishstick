"""
Fusion strategies for multimodal learning

Early fusion, late fusion, cross-attention, and Perceiver Resampler
for combining features from multiple modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class FusionConfig:
    input_dims: List[int]
    hidden_dim: int = 512
    output_dim: int = 512
    num_modalities: int = 2
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    perceiver_num_latents: int = 32
    perceiver_num_heads: int = 8


class EarlyFusion(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        self.projections = nn.ModuleList(
            [nn.Linear(dim, hidden_dim) for dim in input_dims]
        )

        self.fusion_block = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, modality_features: List[Tensor]) -> Tensor:
        projected = []
        for feat, proj in zip(modality_features, self.projections):
            proj_feat = proj(feat)
            projected.append(proj_feat)

        concatenated = torch.cat(projected, dim=-1)

        fused = self.fusion_block(concatenated)

        output = self.output_projection(fused)

        return output


class LateFusion(nn.Module):
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int,
        output_dim: int,
        fusion_type: str = "average",
    ):
        super().__init__()
        self.input_dims = input_dims
        self.fusion_type = fusion_type

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
                for dim in input_dims
            ]
        )

        if fusion_type == "learned":
            self.fusion_weights = nn.Parameter(torch.ones(len(input_dims)))

        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, modality_features: List[Tensor]) -> Tensor:
        projected = []
        for feat, proj in zip(modality_features, self.projections):
            proj_feat = proj(feat)
            projected.append(proj_feat)

        projected_stack = torch.stack(projected, dim=0)

        if self.fusion_type == "average":
            fused = projected_stack.mean(dim=0)
        elif self.fusion_type == "max":
            fused = projected_stack.max(dim=0)[0]
        elif self.fusion_type == "learned":
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = (projected_stack * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            fused = projected_stack.mean(dim=0)

        output = self.output_projection(fused)

        return output


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_modalities = len(input_dims)

        self.modality_projs = nn.ModuleList(
            [nn.Linear(dim, hidden_dim) for dim in input_dims]
        )

        self.query_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_modalities)]
        )

        self.key_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_modalities)]
        )

        self.value_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_modalities)]
        )

        self.cross_attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)]
        )

        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.Linear(hidden_dim * self.num_modalities, output_dim)

    def forward(
        self, modality_features: List[Tensor], return_all_modalities: bool = False
    ) -> Tensor:
        projected = [
            proj(feat) for proj, feat in zip(self.modality_projs, modality_features)
        ]

        modality_0 = projected[0]

        for i, (attn_layer, ffn) in enumerate(
            zip(self.cross_attention_layers, self.ffns)
        ):
            cross_features = []

            for j in range(self.num_modalities):
                query = self.query_projs[j](projected[j])
                key = self.key_projs[j](projected[0] if j == 0 else projected[j - 1])
                value = self.value_projs[j](
                    projected[0] if j == 0 else projected[j - 1]
                )

                attn_output, _ = attn_layer(
                    query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
                )
                attn_output = attn_output.squeeze(1)

                normalized = self.layer_norms[i * 2](projected[j] + attn_output)

                ffn_output = ffn(normalized)
                cross_features.append(
                    self.layer_norms[i * 2 + 1](normalized + ffn_output)
                )

            projected = cross_features

        concatenated = torch.cat(projected, dim=-1)

        output = self.output_projection(concatenated)

        return output


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_latents = num_latents

        self.latents = nn.Parameter(torch.randn(1, num_latents, output_dim) * 0.02)

        self.input_projection = nn.Linear(input_dim, output_dim)

        self.cross_attention = nn.MultiheadAttention(
            output_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=output_dim,
                    nhead=num_heads,
                    dim_feedforward=output_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, input_features: Tensor) -> Tensor:
        batch_size = input_features.shape[0]

        latents = self.latents.expand(batch_size, -1, -1)

        projected_input = self.input_projection(input_features)

        if projected_input.dim() == 2:
            projected_input = projected_input.unsqueeze(1)

        attended, _ = self.cross_attention(latents, projected_input, projected_input)

        x = latents + attended

        for layer in self.transformer_layers:
            x = layer(x)

        output = self.layer_norm(x)

        return output


class MultimodalPerceiverResampler(nn.Module):
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config

        self.input_projections = nn.ModuleList(
            [nn.Linear(dim, config.hidden_dim) for dim in config.input_dims]
        )

        self.resamplers = nn.ModuleList(
            [
                PerceiverResampler(
                    input_dim=config.hidden_dim,
                    output_dim=config.output_dim,
                    num_latents=config.perceiver_num_latents,
                    num_heads=config.perceiver_num_heads,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                )
                for _ in range(config.num_modalities)
            ]
        )

        self.output_projection = nn.Linear(
            config.output_dim * config.num_modalities, config.output_dim
        )

    def forward(self, modality_features: List[Tensor]) -> Tensor:
        projected = [
            proj(feat) for proj, feat in zip(self.input_projections, modality_features)
        ]

        resampled = [
            resampler(feat) for resampler, feat in zip(self.resamplers, projected)
        ]

        concatenated = torch.cat(resampled, dim=1)

        output = self.output_projection(concatenated)

        return output


class GatedFusion(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.num_modalities = len(input_dims)

        self.projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh()) for dim in input_dims]
        )

        self.gate_networks = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_modalities)]
        )

        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, modality_features: List[Tensor]) -> Tensor:
        projected = [
            proj(feat) for proj, feat in zip(self.projections, modality_features)
        ]

        gates = [
            torch.sigmoid(gate(feat))
            for gate, feat in zip(self.gate_networks, projected)
        ]

        gated_features = [proj * gate for proj, gate in zip(projected, gates)]

        fused = sum(gated_features) / len(gated_features)

        output = self.output_projection(fused)

        return output


class FiLMFusion(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.num_modalities = len(input_dims)

        self.projections = nn.ModuleList(
            [nn.Linear(dim, hidden_dim) for dim in input_dims]
        )

        self.film_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                )
                for _ in range(self.num_modalities - 1)
            ]
        )

        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, modality_features: List[Tensor]) -> Tensor:
        projected = [
            proj(feat) for proj, feat in zip(self.projections, modality_features)
        ]

        base = projected[0]

        for i, film_gen in enumerate(self.film_generators):
            conditioning = projected[i + 1]

            film_params = film_gen(conditioning)
            gamma, beta = film_params.chunk(2, dim=-1)

            base = base * (1 + gamma) + beta

        output = self.output_projection(base)

        return output


def create_fusion_module(
    fusion_type: str,
    input_dims: List[int],
    hidden_dim: int = 512,
    output_dim: int = 512,
    **kwargs,
) -> nn.Module:
    fusion_type = fusion_type.lower()

    if fusion_type == "early":
        return EarlyFusion(input_dims, hidden_dim, output_dim)
    elif fusion_type == "late":
        fusion_type = kwargs.get("late_fusion_type", "average")
        return LateFusion(input_dims, hidden_dim, output_dim, fusion_type)
    elif fusion_type == "cross_attention":
        num_heads = kwargs.get("num_heads", 8)
        num_layers = kwargs.get("num_layers", 4)
        dropout = kwargs.get("dropout", 0.1)
        return CrossAttentionFusion(
            input_dims, hidden_dim, output_dim, num_heads, num_layers, dropout
        )
    elif fusion_type == "perceiver":
        config = FusionConfig(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_modalities=len(input_dims),
            perceiver_num_latents=kwargs.get("num_latents", 32),
            perceiver_num_heads=kwargs.get("num_heads", 8),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.1),
        )
        return MultimodalPerceiverResampler(config)
    elif fusion_type == "gated":
        return GatedFusion(input_dims, hidden_dim, output_dim)
    elif fusion_type == "film":
        return FiLMFusion(input_dims, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


DEFAULT_FUSION_CONFIG = FusionConfig(
    input_dims=[512, 512], hidden_dim=512, output_dim=512, num_modalities=2
)


__all__ = [
    "FusionConfig",
    "EarlyFusion",
    "LateFusion",
    "CrossAttentionFusion",
    "PerceiverResampler",
    "MultimodalPerceiverResampler",
    "GatedFusion",
    "FiLMFusion",
    "create_fusion_module",
    "DEFAULT_FUSION_CONFIG",
]
