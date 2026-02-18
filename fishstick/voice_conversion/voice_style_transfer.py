"""
Voice Style Transfer Module

This module provides components for transferring voice style characteristics:
- StyleEncoder: Encodes voice style embeddings
- StyleTransferNetwork: Main style transfer network
- ExpressiveVoiceConverter: Converts prosody and style
- StyleAdaptationLayer: Adaptive instance normalization
"""

from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class StyleTransferConfig:
    """Configuration for voice style transfer."""

    style_dim: int = 256
    hidden_dim: int = 256
    num_style_layers: int = 4
    num_transfer_layers: int = 6
    num_heads: int = 4
    dropout: float = 0.1
    n_mels: int = 80
    speaker_embedding_dim: int = 256
    use_prosody: bool = True
    prosody_dim: int = 5
    use_gst: bool = True
    num_gst_tokens: int = 10


class StyleAdaptationLayer(nn.Module):
    """Adaptive Instance Normalization (AdaIN) for style transfer.

    Applies style-dependent normalization to feature representations.
    """

    def __init__(
        self,
        features: int,
        style_dim: int,
    ):
        super().__init__()
        self.features = features
        self.style_dim = style_dim

        self.norm = nn.InstanceNorm1d(features, affine=False)
        self.style_scale_proj = nn.Linear(style_dim, features)
        self.style_bias_proj = nn.Linear(style_dim, features)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        normalized = self.norm(x)
        scale = self.style_scale_proj(style).unsqueeze(2)
        bias = self.style_bias_proj(style).unsqueeze(2)
        return normalized * (1 + scale) + bias


class MultiHeadAttention(nn.Module):
    """Multi-head attention for style encoding."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(value_dim, query_dim)

        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        b, t, c = query.size()

        q = self.q_proj(query).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = (
            self.v_proj(value)
            .view(b, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.out_proj(out)

        return out, attn


class StyleEncoder(nn.Module):
    """Encodes voice style from acoustic features.

    Extracts style embeddings from mel-spectrograms using self-attention
    and reference encoder mechanisms.

    Args:
        config: StyleTransferConfig with model parameters
    """

    def __init__(self, config: StyleTransferConfig):
        super().__init__()
        self.config = config

        self.input_conv = nn.Sequential(
            nn.Conv1d(config.n_mels, config.hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        self.style_blocks = nn.ModuleList()
        for _ in range(config.num_style_layers):
            self.style_blocks.append(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.hidden_dim * 4,
                    dropout=config.dropout,
                    batch_first=True,
                )
            )

        self.style_projection = nn.Linear(config.hidden_dim, config.style_dim)

        if config.use_gst:
            self.gst_tokens = nn.Parameter(
                torch.randn(config.num_gst_tokens, config.style_dim)
            )
            self.gst_attention = MultiHeadAttention(
                config.style_dim,
                config.style_dim,
                config.style_dim,
                num_heads=config.num_heads,
            )

    def forward(self, mel: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = mel.transpose(1, 2)
        x = self.input_conv(x)
        x = x.transpose(1, 2)

        for block in self.style_blocks:
            x = block(x, src_key_padding_mask=mask)

        style = x.mean(dim=1)
        style = self.style_projection(style)

        if self.config.use_gst:
            tokens = self.gst_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
            gst_out, _ = self.gst_attention(tokens, x, x)
            gst = gst_out.mean(dim=1)
            style = style + gst

        return style


class ReferenceEncoder(nn.Module):
    """Reference encoder for extracting style embeddings from audio."""

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 256,
        style_dim: int = 256,
    ):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv1d(n_mels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.gru = nn.GRU(256, style_dim, batch_first=True, bidirectional=True)

        self.projection = nn.Linear(style_dim * 2, style_dim)

    def forward(self, mel: Tensor) -> Tensor:
        x = mel.transpose(1, 2)
        x = self.convs(x)

        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        x = self.projection(x)

        return x


class StyleTransferNetwork(nn.Module):
    """Main style transfer network for voice conversion.

    Combines spectral conversion with style adaptation.

    Args:
        config: StyleTransferConfig with model parameters
    """

    def __init__(self, config: StyleTransferConfig):
        super().__init__()
        self.config = config

        self.style_encoder = StyleEncoder(config)
        self.reference_encoder = ReferenceEncoder(
            config.n_mels, config.hidden_dim, config.style_dim
        )

        self.speaker_embedding_proj = nn.Linear(
            config.speaker_embedding_dim, config.style_dim
        )

        self.input_projection = nn.Linear(config.n_mels, config.hidden_dim)

        self.transfer_layers = nn.ModuleList()
        for _ in range(config.num_transfer_layers):
            self.transfer_layers.append(
                StyleAdaptationLayer(config.hidden_dim, config.style_dim)
            )

        self.output_projection = nn.Linear(config.hidden_dim, config.n_mels)

    def forward(
        self,
        source_mel: Tensor,
        ref_mel: Optional[Tensor] = None,
        speaker_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        style = self.style_encoder(source_mel)

        if ref_mel is not None:
            ref_style = self.reference_encoder(ref_mel)
            style = style + ref_style

        if speaker_embedding is not None:
            speaker_emb = self.speaker_embedding_proj(speaker_embedding)
            style = style + speaker_emb

        x = self.input_projection(source_mel)

        for layer in self.transfer_layers:
            x = layer(x, style)

        output = self.output_projection(x)
        return output


class ExpressiveVoiceConverter(nn.Module):
    """Converts voice with full prosody and style control.

    End-to-end expressive voice conversion with prosody preservation
    and style transfer capabilities.

    Args:
        config: StyleTransferConfig with model parameters
    """

    def __init__(self, config: StyleTransferConfig):
        super().__init__()
        self.config = config

        self.style_transfer = StyleTransferNetwork(config)

        if config.use_prosody:
            self.prosody_encoder = nn.Sequential(
                nn.Conv1d(config.n_mels, config.hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(config.hidden_dim, config.prosody_dim, 1),
            )

            self.prosody_projection = nn.Linear(config.prosody_dim, config.n_mels)

    def forward(
        self,
        source_mel: Tensor,
        ref_mel: Optional[Tensor] = None,
        speaker_embedding: Optional[Tensor] = None,
        preserve_prosody: bool = True,
    ) -> Tensor:
        if preserve_prosody and self.config.use_prosody:
            prosody = self.prosody_encoder(source_mel.transpose(1, 2))
            prosody = prosody.transpose(1, 2)

        converted = self.style_transfer(source_mel, ref_mel, speaker_embedding)

        if preserve_prosody and self.config.use_prosody:
            prosody_residual = self.prosody_projection(prosody)
            converted = converted + prosody_residual

        return converted


class ProsodyEncoder(nn.Module):
    """Encodes prosodic features for style transfer."""

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 128,
        output_dim: int = 32,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            n_mels,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.projection = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, mel: Tensor) -> Tensor:
        x, _ = self.lstm(mel)
        x = self.projection(x)
        return x


class DurationPredictor(nn.Module):
    """Predicts duration for prosody-controlled conversion."""

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x.transpose(1, 2)).squeeze(1)


def create_style_transfer_network(
    config: Optional[StyleTransferConfig] = None,
    network_type: str = "full",
) -> nn.Module:
    """Factory function to create style transfer networks.

    Args:
        config: StyleTransferConfig with parameters
        network_type: Type of network ('encoder', 'full', 'expressive')

    Returns:
        Initialized style transfer network
    """
    if config is None:
        config = StyleTransferConfig()

    if network_type == "encoder":
        return StyleEncoder(config)
    elif network_type == "full":
        return StyleTransferNetwork(config)
    elif network_type == "expressive":
        return ExpressiveVoiceConverter(config)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
