"""
Positional Encoding Utilities

Various positional encoding schemes for transformer models including
sinusoidal, learned, relative, rotary, and ALiBi encodings.
"""

from typing import Optional, Tuple
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    Creates positional encodings using sine and cosine functions
    at different frequencies:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings.

    Each position gets a learnable embedding vector.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )

        pos_encoding = self.position_embeddings(positions)

        x = x + pos_encoding
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding from "Self-Attention with Relative Position Representations".

    Encodes relative positions between tokens rather than absolute positions.
    """

    def __init__(
        self,
        d_model: int,
        max_relative_position: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.relative_attention_bias = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, num_heads)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        seq_len: int,
        device: str = "cpu",
    ) -> Tensor:
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(-1) - positions.unsqueeze(-1).t()

        relative_positions = torch.clamp(
            relative_positions, -self.max_relative_position, self.max_relative_position
        )

        relative_positions = relative_positions + self.max_relative_position

        return self.relative_attention_bias[relative_positions]

    def get_attention_bias(
        self,
        query_len: int,
        key_len: int,
        device: str = "cpu",
    ) -> Tensor:
        q_positions = torch.arange(query_len, device=device)
        k_positions = torch.arange(key_len, device=device)

        relative_positions = q_positions.unsqueeze(-1) - k_positions.unsqueeze(-2)

        relative_positions = torch.clamp(
            relative_positions, -self.max_relative_position, self.max_relative_position
        )

        relative_positions = relative_positions + self.max_relative_position

        return self.relative_attention_bias[relative_positions]


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Positional Representations".

    Applies rotary transformations to query and key matrices
    to encode relative positions without explicit position embeddings.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 5000,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len

        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(
            self.inv_freq
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: Tensor, seq_len: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        if seq_len is None:
            seq_len = x.size(1)

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return (self.cos_cached[:seq_len], self.sin_cached[:seq_len])

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., : x.size(-1) // 2]
        x2 = x[..., x.size(-1) // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
    ) -> Tuple[Tensor, Tensor]:
        q_embed = (q * cos) + (RotaryPositionalEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (RotaryPositionalEmbedding.rotate_half(k) * sin)
        return q_embed, k_embed


class ALiBiPositionalEncoding(nn.Module):
    """Attention with Linear Biases (ALiBi) from "Train Short, Test Long".

    Applies a linear bias to attention scores based on distance between positions.
    Does not use any positional embeddings.
    """

    def __init__(
        self,
        num_heads: int,
        slope: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.slope = slope

        self._compute_slopes(num_heads)

    def _compute_slopes(self, num_heads: int):
        def get_alibi_slopes_power_of_2(n: int) -> list:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = get_alibi_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_alibi_slopes_power_of_2(closest_power_of_2)

            extra_slopes = get_alibi_slopes_power_of_2(2 * closest_power_of_2)
            slopes = slopes + extra_slopes[0::2][: num_heads - closest_power_of_2]

        self.register_buffer("slopes", torch.tensor(slopes))

    def get_attention_bias(
        self,
        seq_len: int,
        device: str = "cpu",
    ) -> Tensor:
        positions = torch.arange(seq_len, device=device)
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)

        distances = distances.abs().float()

        alibi = -distances.unsqueeze(0) * self.slopes.view(-1, 1, 1).to(device)

        return alibi


class CoherentPositionalEncoding(nn.Module):
    """Coherent positional encoding for generative models.

    Combines sinusoidal encoding with learned components.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        learnable_scale: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.learnable_scale = nn.Parameter(torch.tensor(learnable_scale))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

        self.learnable_bias = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        pos_encoding = (
            self.pe[:, : x.size(1)] * self.learnable_scale
            + self.learnable_bias[:, : x.size(1)]
        )
        x = x + pos_encoding.unsqueeze(0)
        return self.dropout(x)


class T5RelativePositionalBias(nn.Module):
    """T5-style relative positional bias.

    From "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer".
    """

    def __init__(
        self,
        num_heads: int,
        max_distance: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance

        self.relative_attention_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * max_distance + 1)
        )

    def forward(
        self,
        query_len: int,
        key_len: int,
        device: str = "cpu",
    ) -> Tensor:
        q_positions = torch.arange(query_len, device=device)
        k_positions = torch.arange(key_len, device=device)

        relative_positions = q_positions.unsqueeze(-1) - k_positions.unsqueeze(-2)

        relative_positions = torch.clamp(
            relative_positions, -self.max_distance, self.max_distance
        )

        relative_positions = relative_positions + self.max_distance

        return self.relative_attention_bias[:, relative_positions]


class StreamingPositionalEncoding(nn.Module):
    """Positional encoding designed for streaming/incremental inference.

    Supports variable-length sequences without recomputation.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        self.angle_freqs = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)),
            requires_grad=False,
        )

    def forward(
        self,
        x: Tensor,
        start_position: int = 0,
    ) -> Tensor:
        seq_len = x.size(1)

        positions = torch.arange(seq_len, device=x.device).float() + start_position

        positions = positions.unsqueeze(-1)
        angles = positions * self.angle_freqs.unsqueeze(0)

        pos_encoding = torch.cat([angles.sin(), angles.cos()], dim=-1)

        return self.dropout(x + pos_encoding)


class MultiScalePositionalEncoding(nn.Module):
    """Multi-scale positional encoding for hierarchical representations.

    Combines encodings at different granularities (e.g., word, sentence, document).
    """

    def __init__(
        self,
        d_model: int,
        num_scales: int = 3,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales

        self.encodings = nn.ModuleList(
            [
                SinusoidalPositionalEncoding(
                    d_model // num_scales, max_len, dropout=0.0
                )
                for _ in range(num_scales)
            ]
        )

        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        scale_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if scale_ids is None:
            scale_ids = torch.zeros(x.size(1), dtype=torch.long, device=x.device)

        chunk_size = self.d_model // self.num_scales

        encoded_chunks = []
        for i, encoding in enumerate(self.encodings):
            mask = scale_ids == i
            if mask.any():
                chunk = x[:, mask, i * chunk_size : (i + 1) * chunk_size]
                chunk = encoding(chunk)
                encoded_chunks.append(chunk)

        if encoded_chunks:
            encoded = torch.zeros_like(x)
            current_idx = 0
            for i in range(len(encoded_chunks)):
                mask = scale_ids == i
                chunk_size_i = encoded_chunks[i].size(1)
                encoded[:, mask] = encoded_chunks[i]
        else:
            encoded = x

        encoded = self.projection(encoded)
        return self.dropout(encoded)
