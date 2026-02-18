"""
Audio-Visual Fusion Networks for fishstick

This module provides advanced fusion mechanisms for audio-visual data:
- Late fusion
- Early fusion
- Cross-modal attention fusion
- Memory-augmented fusion
"""

from typing import Optional, List, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AVFuser(nn.Module):
    """Base class for audio-visual fusion."""

    def forward(
        self,
        audio_features: Tensor,
        video_features: Tensor,
    ) -> Tensor:
        raise NotImplementedError


class EarlyFusionAV(AVFuser):
    """Early fusion: concatenate raw features before processing."""

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + video_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        audio_features: Tensor,
        video_features: Tensor,
    ) -> Tensor:
        combined = torch.cat([audio_features, video_features], dim=-1)
        return self.fusion(combined)


class LateFusionAV(AVFuser):
    """Late fusion: combine predictions from separate encoders."""

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        num_classes: int,
        fusion_type: str = "average",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion_type = fusion_type

        self.audio_head = nn.Sequential(
            nn.Linear(audio_dim, audio_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(audio_dim // 2, num_classes),
        )

        self.video_head = nn.Sequential(
            nn.Linear(video_dim, video_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(video_dim // 2, num_classes),
        )

    def forward(
        self,
        audio_features: Tensor,
        video_features: Tensor,
    ) -> Tensor:
        audio_logits = self.audio_head(audio_features)
        video_logits = self.video_head(video_features)

        if self.fusion_type == "average":
            return (audio_logits + video_logits) / 2
        elif self.fusion_type == "weighted":
            weights = F.softmax(
                torch.stack([audio_logits.mean(), video_logits.mean()]), dim=0
            )
            return weights[0] * audio_logits + weights[1] * video_logits
        else:
            return torch.cat([audio_logits, video_logits], dim=-1)


class CrossModalFusion(AVFuser):
    """Cross-modal attention-based fusion."""

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.video_projection = nn.Linear(video_dim, hidden_dim)

        self.audio_to_video_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.video_to_audio_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        audio_features: Tensor,
        video_features: Tensor,
    ) -> Tensor:
        audio_emb = self.audio_projection(audio_features).unsqueeze(1)
        video_emb = self.video_projection(video_features).unsqueeze(1)

        audio_attended, _ = self.audio_to_video_attn(audio_emb, video_emb, video_emb)
        video_attended, _ = self.video_to_audio_attn(video_emb, audio_emb, audio_emb)

        fused = torch.cat(
            [audio_attended.squeeze(1), video_attended.squeeze(1)], dim=-1
        )

        return self.fusion(fused)


class TensorFusion(nn.Module):
    """Tensor fusion for multi-modal learning."""

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        output_dim: int,
        rank: int = 4,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.rank = rank

        self.W_aa = nn.Parameter(torch.randn(audio_dim, rank, audio_dim))
        self.W_bb = nn.Parameter(torch.randn(video_dim, rank, video_dim))
        self.W_ab = nn.Parameter(torch.randn(audio_dim, rank, video_dim))
        self.W_ba = nn.Parameter(torch.randn(video_dim, rank, audio_dim))

        self.bias = nn.Parameter(torch.zeros(1))

        self.projection = nn.Linear(rank * 4, output_dim)

    def forward(
        self,
        audio_features: Tensor,
        video_features: Tensor,
    ) -> Tensor:
        audio_expanded = audio_features.unsqueeze(-1)
        video_expanded = video_features.unsqueeze(-1)

        term_aa = torch.einsum(
            "bi,ijl,bkl->bk", audio_features, self.W_aa, audio_features
        )
        term_bb = torch.einsum(
            "bi,ijl,bkl->bk", video_features, self.W_bb, video_features
        )
        term_ab = torch.einsum(
            "bi,ijl,bkl->bk", audio_features, self.W_ab, video_features
        )
        term_ba = torch.einsum(
            "bi,ijl,bkl->bk", video_features, self.W_ba, audio_features
        )

        combined = torch.stack([term_aa, term_bb, term_ab, term_ba], dim=-1)
        fused = combined + self.bias

        return self.projection(fused)


class GatedFusionAV(AVFuser):
    """Gated fusion with learnable gates."""

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.audio_gate = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.video_gate = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.audio_transform = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.video_transform = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        audio_features: Tensor,
        video_features: Tensor,
    ) -> Tensor:
        audio_gate = self.audio_gate(audio_features)
        video_gate = self.video_gate(video_features)

        audio_transformed = self.audio_transform(audio_features)
        video_transformed = self.video_transform(video_features)

        audio_gated = audio_gate * audio_transformed
        video_gated = video_gate * video_transformed

        combined = audio_gated + video_gated
        return self.output(combined)


class MemoryAugmentedFusion(AVFuser):
    """Memory-augmented fusion for audio-visual data."""

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int,
        memory_size: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.memory_size = memory_size

        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.video_projection = nn.Linear(video_dim, hidden_dim)

        self.memory = nn.Parameter(torch.randn(memory_size, hidden_dim))

        self.query_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.key_network = nn.Linear(hidden_dim, hidden_dim)
        self.value_network = nn.Linear(hidden_dim, hidden_dim)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        audio_features: Tensor,
        video_features: Tensor,
    ) -> Tensor:
        audio_emb = self.audio_projection(audio_features)
        video_emb = self.video_projection(video_features)

        combined = torch.cat([audio_emb, video_emb], dim=-1)
        query = self.query_network(combined).unsqueeze(1)

        keys = self.key_network(self.memory)
        values = self.value_network(self.memory)

        attention = torch.softmax(query @ keys.t() / (keys.size(-1) ** 0.5), dim=-1)
        memory_output = attention @ values

        output = torch.cat([combined, memory_output.squeeze(1)], dim=-1)
        return self.output(output)


class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for audio-visual fusion."""

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.video_projection = nn.Linear(video_dim, hidden_dim)

        self.scale_transform = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.shift_transform = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        audio_features: Tensor,
        video_features: Tensor,
    ) -> Tensor:
        video_emb = self.video_projection(video_features)

        scale = self.scale_transform(audio_features)
        shift = self.shift_transform(audio_features)

        modulated = video_emb * (1 + scale) + shift
        return modulated


def create_av_fusion(
    fusion_type: str,
    audio_dim: int,
    video_dim: int,
    output_dim: int,
    **kwargs,
) -> AVFuser:
    """Factory function to create audio-visual fusion modules."""
    if fusion_type == "early":
        return EarlyFusionAV(audio_dim, video_dim, output_dim, **kwargs)
    elif fusion_type == "late":
        return LateFusionAV(audio_dim, video_dim, output_dim, **kwargs)
    elif fusion_type == "cross_attention":
        return CrossModalFusion(audio_dim, video_dim, output_dim, **kwargs)
    elif fusion_type == "tensor":
        return TensorFusion(audio_dim, video_dim, output_dim, **kwargs)
    elif fusion_type == "gated":
        return GatedFusionAV(audio_dim, video_dim, output_dim, **kwargs)
    elif fusion_type == "memory":
        return MemoryAugmentedFusion(audio_dim, video_dim, output_dim, **kwargs)
    elif fusion_type == "film":
        return FiLMFusion(audio_dim, video_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
