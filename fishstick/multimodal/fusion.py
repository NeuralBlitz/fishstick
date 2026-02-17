"""
Multi-Modal Fusion Methods
"""

from typing import List, Dict
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class EarlyFusion(nn.Module):
    """Early fusion: concatenate raw inputs."""

    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.projection = nn.Linear(sum(input_dims), output_dim)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        concatenated = torch.cat(inputs, dim=-1)
        return self.projection(concatenated)


class LateFusion(nn.Module):
    """Late fusion: combine predictions."""

    def __init__(self, num_modalities: int, output_dim: int):
        super().__init__()
        self.fusion = nn.Linear(num_modalities, output_dim)

    def forward(self, predictions: List[Tensor]) -> Tensor:
        stacked = torch.stack(predictions, dim=-1)
        return self.fusion(stacked)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for multi-modal learning."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        attn_output, _ = self.cross_attn(query, key, value)
        return attn_output


class ModalityAlignment(nn.Module):
    """Align different modalities to a common space."""

    def __init__(self, modality_dims: Dict[str, int], common_dim: int):
        super().__init__()
        self.projections = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.projections[modality] = nn.Linear(dim, common_dim)

    def forward(self, modalities: Dict[str, Tensor]) -> Dict[str, Tensor]:
        aligned = {}
        for modality, features in modalities.items():
            aligned[modality] = self.projections[modality](features)
        return aligned


class TransformerFusion(nn.Module):
    """Transformer-based multi-modal fusion."""

    def __init__(self, embed_dim: int, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, modalities: List[Tensor]) -> Tensor:
        concatenated = torch.cat([m.unsqueeze(1) for m in modalities], dim=1)
        return self.transformer(concatenated).mean(dim=1)


class GatedFusion(nn.Module):
    """Gated fusion with learnable gates."""

    def __init__(self, num_modalities: int, hidden_dim: int):
        super().__init__()
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )
                for _ in range(num_modalities)
            ]
        )
        self.fusion = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, modalities: List[Tensor]) -> Tensor:
        gates = [gate(m) for gate, m in zip(self.gates, modalities)]
        weighted = [m * g for m, g in zip(modalities, gates)]
        return self.fusion(sum(weighted))
