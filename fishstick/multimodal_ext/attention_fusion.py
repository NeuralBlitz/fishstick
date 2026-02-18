"""
Attention-Based Fusion Mechanisms for fishstick

This module provides attention-based fusion for multi-modal learning:
- Multi-head cross-attention fusion
- Self-attention fusion
- Co-attention fusion
- Low-rank bilinear attention
"""

from typing import Optional, List, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Cross-modal attention for multi-modal fusion."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.query_projection = nn.Linear(query_dim, output_dim)
        self.key_projection = nn.Linear(key_dim, output_dim)
        self.value_projection = nn.Linear(value_dim, output_dim)

        self.output_projection = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        B = query.size(0)

        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)

        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if q.size(2) == 1:
            k = k.expand(-1, 1, -1, -1)
            v = v.expand(-1, 1, -1, -1)

        attention = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            attention = attention.masked_fill(
                mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        attended = attention @ v
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(B, -1, self.num_heads * self.head_dim)
        )

        return self.output_projection(attended)


class SelfAttentionFusion(nn.Module):
    """Self-attention based fusion for multi-modal data."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(
        self,
        modalities: List[Tensor],
    ) -> Tensor:
        combined = torch.stack(modalities, dim=1)
        fused = self.transformer(combined)
        return fused.mean(dim=1)


class CoAttentionFusion(nn.Module):
    """Co-attention fusion for two modalities."""

    def __init__(
        self,
        dim1: int,
        dim2: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modality1_projection = nn.Linear(dim1, hidden_dim)
        self.modality2_projection = nn.Linear(dim2, hidden_dim)

        self.attention1 = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention2 = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        modality1: Tensor,
        modality2: Tensor,
    ) -> Tensor:
        mod1_emb = self.modality1_projection(modality1).unsqueeze(1)
        mod2_emb = self.modality2_projection(modality2).unsqueeze(1)

        attended1, _ = self.attention1(mod1_emb, mod2_emb, mod2_emb)
        attended2, _ = self.attention2(mod2_emb, mod1_emb, mod1_emb)

        fused = self.fusion(
            torch.cat([attended1.squeeze(1), attended2.squeeze(1)], dim=-1)
        )
        return fused


class LowRankBilinearAttention(nn.Module):
    """Low-rank bilinear attention for efficient multi-modal fusion."""

    def __init__(
        self,
        dim1: int,
        dim2: int,
        output_dim: int,
        rank: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank

        self.U = nn.Parameter(torch.randn(dim1, rank))
        self.V = nn.Parameter(torch.randn(dim2, rank))
        self.W = nn.Parameter(torch.randn(rank, output_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        modality1: Tensor,
        modality2: Tensor,
    ) -> Tensor:
        bilinear = (modality1 @ self.U) * (modality2 @ self.V)
        output = bilinear @ self.W
        return self.dropout(output)


class MultiHeadBilinearFusion(nn.Module):
    """Multi-head bilinear fusion for multi-modal learning."""

    def __init__(
        self,
        dim1: int,
        dim2: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.head_fusion = nn.ModuleList(
            [
                LowRankBilinearAttention(
                    dim1, dim2, output_dim // num_heads, dropout=dropout
                )
                for _ in range(num_heads)
            ]
        )

        self.output_projection = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        modality1: Tensor,
        modality2: Tensor,
    ) -> Tensor:
        heads = [head_fusion(modality1, modality2) for head_fusion in self.head_fusion]
        combined = torch.cat(heads, dim=-1)
        return self.output_projection(combined)


class TFN(nn.Module):
    """Tensor Fusion Network for multi-modal learning."""

    def __init__(
        self,
        modality_dims: List[int],
        output_dim: int,
    ):
        super().__init__()
        self.modality_dims = modality_dims

        total_dim = 1
        for dim in modality_dims:
            total_dim *= dim

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        modalities: List[Tensor],
    ) -> Tensor:
        tensors = [m.unsqueeze(-1) for m in modalities]

        while len(tensors) > 1:
            new_tensors = []
            for i in range(0, len(tensors) - 1, 2):
                t1, t2 = tensors[i], tensors[i + 1]
                product = t1 * t2
                new_tensors.append(product.squeeze(-1))
            if len(tensors) % 2 == 1:
                new_tensors.append(tensors[-1])
            tensors = new_tensors

        combined = tensors[0].squeeze(-1)
        return self.fusion(combined)


class MemoryAttentionFusion(nn.Module):
    """Memory-augmented attention fusion."""

    def __init__(
        self,
        embed_dim: int,
        memory_size: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim))

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        modality: Tensor,
    ) -> Tensor:
        query = self.query_projection(modality).unsqueeze(1)
        key = self.key_projection(self.memory)
        value = self.value_projection(self.memory)

        attended, _ = self.attention(query, key, value)
        return attended.squeeze(1)


class StackedAttentionFusion(nn.Module):
    """Stacked attention layers for multi-modal fusion."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        modalities: List[Tensor],
    ) -> Tensor:
        combined = torch.stack(modalities, dim=1)

        for attn_layer, ln in zip(self.attention_layers, self.layer_norms):
            attended, _ = attn_layer(combined, combined, combined)
            combined = ln(combined + attended)

        return combined.mean(dim=1)


def create_attention_fusion(
    fusion_type: str,
    **kwargs,
) -> nn.Module:
    """Factory function to create attention fusion modules."""
    if fusion_type == "cross":
        return CrossModalAttention(**kwargs)
    elif fusion_type == "self":
        return SelfAttentionFusion(**kwargs)
    elif fusion_type == "co":
        return CoAttentionFusion(**kwargs)
    elif fusion_type == "low_rank_bilinear":
        return LowRankBilinearAttention(**kwargs)
    elif fusion_type == "multi_head_bilinear":
        return MultiHeadBilinearFusion(**kwargs)
    elif fusion_type == "tfn":
        return TFN(**kwargs)
    elif fusion_type == "memory":
        return MemoryAttentionFusion(**kwargs)
    elif fusion_type == "stacked":
        return StackedAttentionFusion(**kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
