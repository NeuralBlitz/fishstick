"""
Rotary Position Embedding (RoPE)

Implementation of Rotary Position Embedding (RoPE) for transformer models.
RoPE encodes position information using rotation matrices, providing
better extrapolation capabilities than sinusoidal positional encodings.
"""

import torch
from torch import nn, Tensor
import math
from typing import Optional


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    RoPE uses rotation matrices to encode positions, which allows
    the model to better handle sequences longer than those seen during training.

    Attributes:
        dim: Dimension of the embedding
        max_seq_len: Maximum sequence length to precompute
        base: Base for the frequency computation
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """Precompute cos and sin values for efficiency."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tensor:
        """Compute rotary positional embeddings.

        Args:
            seq_len: Length of the sequence
            device: Device to create tensor on

        Returns:
            Rotary positional embeddings [seq_len, dim]
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len].to(device)
            if device is not None
            else self.cos_cached[:seq_len],
            self.sin_cached[:seq_len].to(device)
            if device is not None
            else self.sin_cached[:seq_len],
        )


class RotaryEmbedding(nn.Module):
    """Complete rotary embedding layer for transformer models.

    Applies rotary position embeddings to query and key tensors
    in attention mechanism.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.rope = RotaryPositionalEmbedding(dim, max_seq_len, base)
        self.dim = dim

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        seq_len: int,
    ) -> Tuple[Tensor, Tensor]:
        """Apply rotary embeddings to query and key.

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            seq_len: Sequence length

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        cos, sin = self.rope(seq_len, q.device)

        q_embed, k_embed = self._rotate_half(q, k, cos, sin)

        return q_embed, k_embed

    def _rotate_half(
        self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Rotate half of the dimensions."""
        q_len = q.shape[-1]
        q_half = q[..., : q_len // 2]
        q_rest = q[..., q_len // 2 :]

        k_half = k[..., : q_len // 2]
        k_rest = k[..., q_len // 2 :]

        q_rotated = torch.cat(
            [
                q_half * cos - q_rest * sin,
                q_half * sin + q_rest * cos,
            ],
            dim=-1,
        )

        k_rotated = torch.cat(
            [
                k_half * cos - k_rest * sin,
                k_half * sin + k_rest * cos,
            ],
            dim=-1,
        )

        return q_rotated, k_rotated


class LinearRoPE(RotaryPositionalEmbedding):
    """Linear RoPE for efficient computation.

    A more memory-efficient implementation that computes
    rotary embeddings on-the-fly.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__(dim, max_seq_len, base)

    def forward(self, x: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        """Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor [..., seq_len, dim]
            position_ids: Optional position indices

        Returns:
            Tensor with rotary embeddings applied
        """
        seq_len = x.shape[-2]

        if position_ids is None:
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            t = position_ids.type_as(self.inv_freq)
            freqs = torch.einsum("ij,jk->ik", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        return self._apply_rotary(x, cos, sin)

    def _apply_rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotation to input tensor."""
        x_len = x.shape[-1]
        x_half = x[..., : x_len // 2]
        x_rest = x[..., x_len // 2 :]

        return torch.cat(
            [
                x_half * cos - x_rest * sin,
                x_half * sin + x_rest * cos,
            ],
            dim=-1,
        )


class YaRNScalingRotaryEmbedding(RotaryPositionalEmbedding):
    """YaRN (Yet another RoPE extensioN) with dynamic scaling.

    Extends RoPE to handle longer sequences through
    dynamic temperature scaling.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        original_seq_len: int = 2048,
        extrapolation_factor: float = 0.4,
        attn_factor: float = 1.0,
    ):
        super().__init__(dim, max_seq_len, base)
        self.original_seq_len = original_seq_len
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor

        self._compute_dynamic_temperature()

    def _compute_dynamic_temperature(self) -> None:
        """Compute the dynamic temperature for scaling."""
        dim = self.dim
        original_seq_len = self.original_seq_len

        alpha = (
            (original_seq_len / (self.max_seq_len - original_seq_len))
            if self.max_seq_len > original_seq_len
            else float("inf")
        )

        self.register_buffer(
            "dynamic_temperature",
            torch.tensor(1.0 + 1.0 / math.sqrt(dim) * alpha),
            persistent=False,
        )

    def forward(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> Tuple[Tensor, Tensor]:
        """Compute scaled rotary embeddings."""
        cos, sin = super().forward(seq_len, device)

        cos = cos * self.dynamic_temperature
        sin = sin * self.dynamic_temperature

        return cos, sin


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]
        position_ids: Optional position indices

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    q_len = q.shape[-1]

    q_real = q[..., : q_len // 2]
    q_imag = q[..., q_len // 2 :]

    k_real = k[..., : q_len // 2]
    k_imag = k[..., q_len // 2 :]

    q_rotated = torch.cat(
        [
            q_real * cos - q_imag * sin,
            q_real * sin + q_imag * cos,
        ],
        dim=-1,
    )

    k_rotated = torch.cat(
        [
            k_real * cos - k_imag * sin,
            k_real * sin + k_imag * cos,
        ],
        dim=-1,
    )

    return q_rotated, k_rotated


class MultiHeadRotaryAttention(nn.Module):
    """Multi-head attention with rotary position embeddings.

    Combines multi-head attention with rotary positional encoding
    for enhanced position awareness.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim % 2 == 0, "Head dimension must be even for RoPE"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with rotary embeddings.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rope(seq_len, x.device)

        q_rotated = self._rotate_tensor(q, cos, sin)
        k_rotated = self._rotate_tensor(k, cos, sin)

        attn_weights = torch.matmul(q_rotated, k_rotated.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)

    def _rotate_tensor(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Rotate tensor using rotary embeddings."""
        head_dim = x.shape[-1]

        x_half = x[..., : head_dim // 2]
        x_rest = x[..., head_dim // 2 :]

        return torch.cat(
            [
                x_half * cos.unsqueeze(0) - x_rest * sin.unsqueeze(0),
                x_half * sin.unsqueeze(0) + x_rest * cos.unsqueeze(0),
            ],
            dim=-1,
        )
