"""
Relative Positional Encoding

Implementation of relative positional encoding for transformer models.
Relative position encodes the distance between tokens rather than
their absolute positions.
"""

import torch
from torch import nn, Tensor
import math
from typing import Optional, Tuple, List


class RelativePositionBias(nn.Module):
    """Relative position bias for attention.

    Adds a learned bias based on the relative distance between
    query and key positions.

    Attributes:
        num_heads: Number of attention heads
        max_distance: Maximum relative distance to model
    """

    def __init__(
        self,
        num_heads: int,
        max_distance: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance

        self.num_buckets = max_distance

        self.relative_attention_bias = nn.Embedding(
            self.num_buckets * 2,
            num_heads,
        )

    def _compute_buckets(self, seq_len_q: int, seq_len_k: int) -> Tensor:
        """Compute relative position buckets.

        Args:
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length

        Returns:
            Relative position buckets [seq_len_q, seq_len_k]
        """
        position_ids_q = torch.arange(
            seq_len_q, device=self.relative_attention_bias.weight.device
        )
        position_ids_k = torch.arange(
            seq_len_k, device=self.relative_attention_bias.weight.device
        )

        relative_position = position_ids_q.unsqueeze(1) - position_ids_k.unsqueeze(0)

        relative_position = torch.clamp(
            relative_position,
            -self.max_distance,
            self.max_distance,
        )

        relative_position = relative_position + self.max_distance

        return relative_position

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tensor:
        """Compute relative position bias.

        Args:
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            Relative position bias [num_heads, seq_len, seq_len]
        """
        relative_position = self._compute_buckets(seq_len, seq_len)

        relative_bias = self.relative_attention_bias(relative_position)

        return relative_bias.permute(2, 0, 1)


class SinusoidalRelativePositionBias(nn.Module):
    """Sinusoidal relative position bias.

    Uses sinusoidal functions to compute relative position biases
    without learned parameters.
    """

    def __init__(
        self,
        num_heads: int,
        max_distance: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tensor:
        """Compute sinusoidal relative position bias.

        Args:
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            Relative position bias [num_heads, seq_len, seq_len]
        """
        position_ids = torch.arange(seq_len, device=device)

        relative_position = position_ids.unsqueeze(1) - position_ids.unsqueeze(0)

        relative_position = torch.clamp(
            relative_position,
            -self.max_distance,
            self.max_distance,
        )

        dim = self.num_heads // 2

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )

        angles = relative_position.float()[:, :, None] * inv_freq[None, None, :]

        angles = torch.cat([angles, angles], dim=-1)

        relative_bias = torch.sin(angles)

        relative_bias = relative_bias.permute(2, 0, 1)

        return relative_bias


class RelativePositionMultiHeadAttention(nn.Module):
    """Multi-head attention with relative position bias.

    Complete attention module with relative position encoding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_distance: int = 128,
        use_learned: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if use_learned:
            self.rel_pos_bias = RelativePositionBias(num_heads, max_distance)
        else:
            self.rel_pos_bias = SinusoidalRelativePositionBias(num_heads, max_distance)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with relative position bias.

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

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        rel_bias = self.rel_pos_bias(seq_len, x.device)

        attn_scores = attn_scores + rel_bias.unsqueeze(0)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)


class T5RelativePositionBias(nn.Module):
    """T5-style relative position bias.

    Implementation of the relative position bias used in T5,
    which uses a learned embedding for relative distances.
    """

    def __init__(
        self,
        num_heads: int,
        max_distance: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance

        self.relative_attention_num_buckets = max_distance
        self.relative_attention_num_buckets = 32

        self.relative_attention_bias = nn.Embedding(
            self.relative_attention_num_buckets,
            num_heads,
        )

    def _compute_buckets(self, seq_len_q: int, seq_len_k: int) -> Tensor:
        """Compute T5-style relative position buckets."""
        position_ids_q = torch.arange(
            seq_len_q, device=self.relative_attention_bias.weight.device
        )
        position_ids_k = torch.arange(
            seq_len_k, device=self.relative_attention_bias.weight.device
        )

        relative_position = position_ids_q.unsqueeze(1) - position_ids_k.unsqueeze(0)

        num_buckets = self.relative_attention_num_buckets

        max_exact = num_buckets // 2

        is_small = torch.abs(relative_position) < max_exact

        val_if_large = max_exact + (
            torch.log(torch.abs(relative_position).float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        val_if_large = torch.min(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1),
        )

        relative_buckets = torch.where(
            is_small,
            torch.abs(relative_position),
            val_if_large,
        )

        relative_buckets = relative_buckets - torch.min(relative_buckets)

        return relative_buckets

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tensor:
        """Compute T5 relative position bias."""
        relative_position = self._compute_buckets(seq_len, seq_len)

        relative_bias = self.relative_attention_bias(relative_position)

        return relative_bias.permute(2, 0, 1)


class ShawRelativePosition(nn.Module):
    """Shaw's relative position encoding.

    Original relative position encoding from "Self-Attention
    with Relative Position Representations".
    """

    def __init__(
        self,
        head_dim: int,
        max_distance: int = 32,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_distance = max_distance

        self.relative_attention_bias = nn.Embedding(
            2 * max_distance + 1,
            head_dim,
        )

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tensor:
        """Compute Shaw's relative position embeddings.

        Args:
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            Relative position embeddings [2*max_distance+1, head_dim]
        """
        position_ids = torch.arange(
            -self.max_distance, self.max_distance + 1, device=device
        )

        return self.relative_attention_bias(position_ids + self.max_distance)


class RelativePositionKeyValue(nn.Module):
    """Relative position encoding applied to both keys and queries.

    Extends relative position encoding to be applied in the
    key-value computation as well.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_distance: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_distance = max_distance

        self.rel_pos_bias = RelativePositionBias(num_heads, max_distance)

        self.rel_pos_embed_k = nn.Embedding(
            2 * max_distance + 1,
            self.head_dim,
        )
        self.rel_pos_embed_v = nn.Embedding(
            2 * max_distance + 1,
            self.head_dim,
        )

    def _get_relative_positions(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Compute relative position indices."""
        position_ids = torch.arange(seq_len, device=device)
        relative_position = position_ids.unsqueeze(1) - position_ids.unsqueeze(0)
        relative_position = torch.clamp(
            relative_position,
            -self.max_distance,
            self.max_distance,
        )
        return relative_position + self.max_distance

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply relative position to key and value.

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]

        Returns:
            Tuple of (output_k, output_v, attention_bias)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        relative_positions = self._get_relative_positions(seq_len, q.device)

        rel_pos_bias = self.rel_pos_bias(seq_len, q.device)

        rel_k = self.rel_pos_embed_k(relative_positions)
        rel_v = self.rel_pos_embed_v(relative_positions)

        rel_k = rel_k.permute(2, 0, 1).unsqueeze(0)
        rel_v = rel_v.permute(2, 0, 1).unsqueeze(0)

        k = k + rel_k
        v = v + rel_v

        return k, v, rel_pos_bias


def relative_position_bucket(
    relative_position: Tensor,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> Tensor:
    """Compute relative position buckets.

    Args:
        relative_position: Relative position tensor
        num_buckets: Number of buckets
        max_distance: Maximum distance to clamp to

    Returns:
        Relative position buckets
    """
    ret = torch.zeros_like(relative_position)

    n = -relative_position

    ret = torch.where(n > 0, -n, ret)

    max_exact = num_buckets // 2

    val_if_large = max_exact + (
        torch.log(ret.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)

    val_if_large = torch.min(
        val_if_large,
        torch.full_like(val_if_large, num_buckets - 1),
    )

    ret = torch.where(ret < max_exact, ret // 2, val_if_large)

    return torch.where(
        relative_position < 0,
        num_buckets - 1 - ret,
        ret,
    )
