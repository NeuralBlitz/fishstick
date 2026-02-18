"""
Attention with Linear Biases (ALiBi)

Implementation of ALiBi positional encoding for transformer models.
ALiBi removes positional embeddings entirely and uses linear biases
in the attention mechanism for position information.
"""

import torch
from torch import nn, Tensor
import math
from typing import Optional, Tuple, List


class ALiBiAttention(nn.Module):
    """Attention with Linear Biases (ALiBi).

    ALiBi encodes position information through linear attention biases
    rather than positional embeddings, providing better extrapolation
    to longer sequences.

    Attributes:
        num_heads: Number of attention heads
        bias: Whether to use ALiBi biases
        slope: Slope for linear bias computation
    """

    def __init__(
        self,
        num_heads: int,
        bias: bool = True,
        slope: Optional[float] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.bias = bias

        if slope is not None:
            self.slope = slope
        else:
            self.slope = self._get_alibi_slopes(num_heads)

    def _get_alibi_slopes(num_heads: int) -> List[float]:
        """Compute ALiBi slopes for each head.

        Uses the geometric sequence approach from the original paper.
        """

        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))

        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes_progressive(2 * closest_power_of_2, num_heads)[
                closest_power_of_2:
            ]
        )

    def _get_alibi_slopes_progressive(n: int, num_heads: int) -> List[float]:
        """Compute ALiBi slopes progressively."""

        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)

        closest_power_of_2 = 2 ** math.floor(math.log2(n))

        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes_progressive(2 * closest_power_of_2, n)[
                closest_power_of_2:
            ]
        )

    def _build_alibi_bias(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> Tensor:
        """Build ALiBi attention bias matrix.

        Args:
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length
            device: Device to create tensor on

        Returns:
            ALiBi bias matrix [num_heads, seq_len_q, seq_len_k]
        """
        slopes = torch.tensor(self.slope, device=device)

        positions = torch.arange(seq_len_k, device=device)

        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        relative_positions = relative_positions.abs().float()

        alibi = -slopes.view(-1, 1, 1) * relative_positions.unsqueeze(0)

        if seq_len_q != seq_len_k:
            alibi = alibi[:, :seq_len_q, :]

        return alibi

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with ALiBi biases.

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            attention_mask: Optional additive attention mask

        Returns:
            Attention output [batch, heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_k = k.shape[2]

        if self.bias:
            alibi_bias = self._build_alibi_bias(seq_len_q, seq_len_k, q.device)
        else:
            alibi_bias = None

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        if alibi_bias is not None:
            attn_scores = attn_scores + alibi_bias

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_probs, v)

        return attn_output


class ALiBiPositionalEmbedding(nn.Module):
    """ALiBi as a standalone positional embedding module.

    Can be used with any attention mechanism.
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.register_buffer(
            "alibi_bias",
            self._build_alibi_bias(max_seq_len),
            persistent=False,
        )

    def _build_alibi_bias(self, seq_len: int) -> Tensor:
        """Precompute ALiBi bias matrix."""
        slopes = torch.tensor(self._get_alibi_slopes())

        positions = torch.arange(seq_len)

        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions.abs().float()

        alibi = -slopes.view(-1, 1, 1) * relative_positions.unsqueeze(0)

        return alibi

    def _get_alibi_slopes(self) -> List[float]:
        """Compute ALiBi slopes."""

        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n)]

        if math.log2(self.num_heads).is_integer():
            return get_slopes_power_of_2(self.num_heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(self.num_heads))

        return (
            get_slopes_power_of_2(closest_power_of_2)
            + self._get_alibi_slopes_progressive(2 * closest_power_of_2)[
                closest_power_of_2:
            ]
        )

    def _get_alibi_slopes_progressive(self, n: int) -> List[float]:
        """Compute slopes progressively."""

        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)

        closest_power_of_2 = 2 ** math.floor(math.log2(n))

        return (
            get_slopes_power_of_2(closest_power_of_2)
            + self._get_alibi_slopes_progressive(2 * closest_power_of_2)[
                closest_power_of_2:
            ]
        )

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tensor:
        """Get ALiBi bias for given sequence length.

        Args:
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            ALiBi bias matrix [num_heads, seq_len, seq_len]
        """
        if seq_len > self.max_seq_len:
            return self._build_alibi_bias(seq_len).to(device)

        return self.alibi_bias[:seq_len, :seq_len].to(device)


class ALiBiMultiHeadAttention(nn.Module):
    """Multi-head attention with ALiBi positional encoding.

    Complete attention module combining ALiBi with standard
    multi-head attention mechanism.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.alibi = ALiBiPositionalEmbedding(num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass with ALiBi.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking

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

        alibi_bias = self.alibi(seq_len, x.device)

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn_scores = attn_scores + alibi_bias

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


class AliBiFlashAttention(nn.Module):
    """ALiBi with Flash Attention for memory efficiency.

    Combines ALiBi positional encoding with Flash Attention
    for faster training and reduced memory usage.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.alibi = ALiBiPositionalEmbedding(num_heads)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass with ALiBi and flash attention.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking

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

        alibi_bias = self.alibi(seq_len, x.device)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores + alibi_bias

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

        if self.dropout > 0:
            attn_probs = torch.dropout(attn_probs, p=self.dropout, train=self.training)

        attn_output = torch.matmul(attn_probs, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_output)


def build_alibi_bias(
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    device: torch.device,
    slope: Optional[List[float]] = None,
) -> Tensor:
    """Build ALiBi bias matrix for given parameters.

    Args:
        num_heads: Number of attention heads
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length
        device: Device to create tensor on
        slope: Optional custom slopes for each head

    Returns:
        ALiBi bias matrix [num_heads, seq_len_q, seq_len_k]
    """
    if slope is None:

        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[
                closest_power_of_2:
            ]
            slopes.extend(extra_slopes[: num_heads - closest_power_of_2])
    else:
        slopes = slope

    slopes_tensor = torch.tensor(slopes, device=device)

    positions = torch.arange(seq_len_k, device=device)

    relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
    relative_positions = relative_positions.abs().float()

    alibi = -slopes_tensor.view(-1, 1, 1) * relative_positions.unsqueeze(0)

    if seq_len_q != seq_len_k:
        alibi = alibi[:, :seq_len_q, :]

    return alibi
